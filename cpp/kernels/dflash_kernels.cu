#include "kernels/kernels.cuh"
#include "runtime/cuda_utils.cuh"
#include "model/model_config.hpp"
#include <cmath>
#include <cfloat>

namespace amperegen {

namespace {

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// ============================================================================
// DFlash2 Dynamic 2-Tap Grouped Depthwise Conv along block dim
// Exact match to dflash.py::_grouped_conv:
//   delta_all: [B * block_size, 2 (side), 2 (taps), 320 (groups)] flattened as [N, 1280]
//   base_kernel: [2 (taps), 5120] BF16 for this side
//   out[pos, g*16 + k] = (base[0, c] + delta[pos, side, 0, g]) * x[pos, c]
//                      + (pos_in_blk >= 1 ? (base[1, c] + delta[pos, side, 1, g]) * x[pos-1, c] : 0)
// Grid: (B * block_size), Block: 256 threads
// ============================================================================
__global__ void dflash_grouped_conv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ delta_all,
    const __nv_bfloat16* __restrict__ base_kernel, // [2, 5120]
    __nv_bfloat16* __restrict__ out,
    int B,
    int block_size,
    int side
) {
    int tok = blockIdx.x;
    int total = B * block_size;
    if (tok >= total) return;

    int pos_in_blk = tok % block_size;
    const __nv_bfloat16* x_cur = x + (size_t)tok * DFlash2Config::HIDDEN_SIZE;
    const __nv_bfloat16* x_prev = (pos_in_blk >= 1) ? (x + (size_t)(tok - 1) * DFlash2Config::HIDDEN_SIZE) : nullptr;
    const __nv_bfloat16* d_row = delta_all + (size_t)tok * DFlash2Config::CONV_PROJ_OUT + side * (2 * DFlash2Config::CONV_NUM_GROUPS);
    const __nv_bfloat16* d_tap0 = d_row;
    const __nv_bfloat16* d_tap1 = d_row + DFlash2Config::CONV_NUM_GROUPS;
    const __nv_bfloat16* b_tap0 = base_kernel;
    const __nv_bfloat16* b_tap1 = base_kernel + DFlash2Config::HIDDEN_SIZE;
    __nv_bfloat16* o_row = out + (size_t)tok * DFlash2Config::HIDDEN_SIZE;

    for (int c = threadIdx.x; c < DFlash2Config::HIDDEN_SIZE; c += blockDim.x) {
        int g = c >> 4; // group_size = 16
        float c0 = __bfloat162float(b_tap0[c]) + __bfloat162float(d_tap0[g]);
        float val = c0 * __bfloat162float(x_cur[c]);
        if (x_prev != nullptr) {
            float c1 = __bfloat162float(b_tap1[c]) + __bfloat162float(d_tap1[g]);
            val += c1 * __bfloat162float(x_prev[c]);
        }
        o_row[c] = __float2bfloat16(val);
    }
}

// ============================================================================
// Draft QK Plain-RMSNorm + Full 128-dim NeoX RoPE in-place on qkv [N, 6144]
// Grid: (N, 40) -> 32 Q heads + 8 K heads, Block: 64 threads (pairs (d, d+64) in 1 thread!)
// ============================================================================
__global__ void dflash_qk_norm_rope_kernel(
    __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const float* __restrict__ rope_cos, // [max_pos, 64]
    const float* __restrict__ rope_sin, // [max_pos, 64]
    const int32_t* __restrict__ positions,
    int N,
    float eps
) {
    int t = blockIdx.x;
    int h = blockIdx.y; // 0..31 Q, 32..39 K
    int tid = threadIdx.x; // 0..63 -> handles pair (tid, tid + 64)!
    if (t >= N) return;

    __nv_bfloat16* row = qkv + (size_t)t * DFlash2Config::QKV_DIM;
    __nv_bfloat16* ptr = (h < 32) ? (row + h * 128) : (row + DFlash2Config::Q_DIM + (h - 32) * 128);
    const __nv_bfloat16* nw = (h < 32) ? q_norm_w : k_norm_w;

    float v0 = __bfloat162float(ptr[tid]);
    float v1 = __bfloat162float(ptr[tid + 64]);

    // Block reduce over 64 threads (2 warps)
    __shared__ float sw[2];
    float ss = warp_sum(v0 * v0 + v1 * v1);
    if ((tid & 31) == 0) sw[tid >> 5] = ss;
    __syncthreads();
    float total_ss = sw[0] + sw[1];
    float inv_rms = rsqrtf(total_ss * (1.0f / 128.0f) + eps);

    // Plain RMSNorm (NO +1 for DFlash2!)
    float n0 = __bfloat162float(__float2bfloat16(v0 * inv_rms * __bfloat162float(nw[tid])));
    float n1 = __bfloat162float(__float2bfloat16(v1 * inv_rms * __bfloat162float(nw[tid + 64])));

    int pos = positions[t];
    float c = rope_cos[(size_t)pos * 64 + tid];
    float s = rope_sin[(size_t)pos * 64 + tid];
    ptr[tid]      = __float2bfloat16(n0 * c - n1 * s);
    ptr[tid + 64] = __float2bfloat16(n1 * c + n0 * s);
}

// ============================================================================
// Draft Context KV Materialization: K-norm + RoPE + ring-buffer append into
// draft_k_cache / draft_v_cache [num_slots, 2048, 8, 128] BF16!
// Ring-buffer index for logical position p is simply (p & 2047)!
// Grid: (total_tokens, 8), Block: 64 threads
// ============================================================================
__global__ void dflash_append_ctx_kv_kernel(
    const __nv_bfloat16* __restrict__ kv, // [total_tokens, 2048]
    const __nv_bfloat16* __restrict__ k_norm_w,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ slot_ids,
    const int32_t* __restrict__ commit_lens, // nullable; when non-null, skips col >= commit_lens[b]!
    __nv_bfloat16* __restrict__ draft_k_cache,
    __nv_bfloat16* __restrict__ draft_v_cache,
    int total_tokens,
    int block_size,
    float eps
) {
    int t = blockIdx.x;
    int h = blockIdx.y; // 0..7
    int tid = threadIdx.x; // 0..63
    if (t >= total_tokens) return;

    if (commit_lens != nullptr && block_size > 0) {
        int b = t / block_size;
        int col = t - b * block_size;
        if (col >= commit_lens[b]) return;
    }

    int slot = slot_ids[t];
    if (slot < 0) return;
    int pos = positions[t];
    int ring_pos = pos & (DFlash2Config::SLIDING_WINDOW - 1);

    const __nv_bfloat16* k_src = kv + (size_t)t * 2048 + h * 128;
    const __nv_bfloat16* v_src = kv + (size_t)t * 2048 + 1024 + h * 128;

    float v0 = __bfloat162float(k_src[tid]);
    float v1 = __bfloat162float(k_src[tid + 64]);

    __shared__ float sw[2];
    float ss = warp_sum(v0 * v0 + v1 * v1);
    if ((tid & 31) == 0) sw[tid >> 5] = ss;
    __syncthreads();
    float inv_rms = rsqrtf((sw[0] + sw[1]) * (1.0f / 128.0f) + eps);

    float n0 = __bfloat162float(__float2bfloat16(v0 * inv_rms * __bfloat162float(k_norm_w[tid])));
    float n1 = __bfloat162float(__float2bfloat16(v1 * inv_rms * __bfloat162float(k_norm_w[tid + 64])));

    float c = rope_cos[(size_t)pos * 64 + tid];
    float s = rope_sin[(size_t)pos * 64 + tid];

    size_t dst_off = ((((size_t)slot * DFlash2Config::SLIDING_WINDOW + ring_pos) * 8 + h) * 128);
    draft_k_cache[dst_off + tid]      = __float2bfloat16(n0 * c - n1 * s);
    draft_k_cache[dst_off + tid + 64] = __float2bfloat16(n1 * c + n0 * s);
    draft_v_cache[dst_off + tid]      = v_src[tid];
    draft_v_cache[dst_off + tid + 64] = v_src[tid + 64];
}

// ============================================================================
// Draft Bidirectional Sliding-Window Attention (32 Q / 8 KV heads, hd=128) — TENSOR CORE SPLIT-K!
// Packs ALL (4 Q-heads x block_size <= 32) queries per KV head (2 M16-tiles!)
// Processes BKV=32 context KV tokens per tile via mma.sync.aligned.m16n8k16 + Split-K=16!
// Cuts 5-layer draft attention at ctx=2048 from ~1.80ms -> ~0.12ms!
// Grid: (DRAFT_ATTN_SPLITS=16, 8 [kv_head], B), Block: 128 threads (4 warps: 2 M16-tiles x 2 D-halves!)
// ============================================================================
static constexpr int DRAFT_ATTN_SPLITS = 16;

__device__ __forceinline__ uint32_t dflash_pack_bf16x2(float a, float b) {
    __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

__global__ void dflash_bidirectional_attn_splitk_kernel(
    const __nv_bfloat16* __restrict__ qkv,           // [B * block_size, 6144]
    const __nv_bfloat16* __restrict__ draft_k_cache, // [num_slots, 2048, 8, 128]
    const __nv_bfloat16* __restrict__ draft_v_cache, // [num_slots, 2048, 8, 128]
    const int32_t* __restrict__ draft_slots,         // [B]
    const int32_t* __restrict__ prefix_lens,         // [B]
    float* __restrict__ split_out,                   // [B * block_size, 32, DRAFT_ATTN_SPLITS, 128]
    float* __restrict__ split_lse,                   // [B * block_size, 32, DRAFT_ATTN_SPLITS, 2]
    int B,
    int block_size
) {
    int split_id = blockIdx.x;
    int kv_head = blockIdx.y; // 0..7
    int b = blockIdx.z;
    if (b >= B) return;

    int tid = threadIdx.x; // 0..127
    int wid = tid >> 5;    // 0..3 -> mt = wid>>1 (0..1), ds = wid&1 (0..1 -> 64 dims each!)
    int lane = tid & 31;
    int g = lane >> 2;     // 0..7
    int t = lane & 3;      // 0..3
    int mt = wid >> 1;
    int ds = wid & 1;
    int d_base = ds * 64;

    __shared__ __nv_bfloat16 sQ[32][128];   // 8 KB (up to 4 Q-heads * 8 tokens = 32 rows)
    __shared__ __nv_bfloat16 sK[32][128];   // 8 KB
    __shared__ uint32_t sV_T[128][16];      // 8 KB (transposed pairs (V[2k,d], V[2k+1,d]) + XOR swizzle!)

    int total_q_rows = block_size * 4;
    int blk_base = b * block_size;
    constexpr float SCALE = 0.08838834764831845f; // 128^-0.5

    // Load sQ[32][128] scaled by 128^-0.5: row r = qi * 4 + gh
    for (int idx = tid; idx < 32 * 16; idx += 128) {
        int r = idx >> 4;
        int c8 = (idx & 15) << 3;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (r < total_q_rows) {
            int qi = r >> 2;
            int gh = r & 3;
            int q_head = kv_head * 4 + gh;
            val = *reinterpret_cast<const uint4*>(
                qkv + (size_t)(blk_base + qi) * DFlash2Config::QKV_DIM + q_head * 128 + c8);
            __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&val);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 f = __bfloat1622float2(p[k]);
                p[k] = __floats2bfloat162_rn(f.x * SCALE, f.y * SCALE);
            }
        }
        *reinterpret_cast<uint4*>(&sQ[r][c8]) = val;
    }
    __syncthreads();

    int slot = draft_slots[b];
    int prefix_len = prefix_lens[b];
    int ctx_start = max(0, prefix_len - DFlash2Config::SLIDING_WINDOW);
    int ctx_len = prefix_len - ctx_start;
    int total_kv = ctx_len + block_size; // [0..ctx_len-1] from ring buffer, [ctx_len..total_kv-1] from current qkv!

    int total_tiles32 = (total_kv + 31) / 32;
    int tiles_per_split = (total_tiles32 + DRAFT_ATTN_SPLITS - 1) / DRAFT_ATTN_SPLITS;
    int tile_start = split_id * tiles_per_split;
    int tile_end = min(total_tiles32, tile_start + tiles_per_split);

    const __nv_bfloat16* k_slot_base = draft_k_cache + (size_t)slot * DFlash2Config::SLIDING_WINDOW * 8 * 128;
    const __nv_bfloat16* v_slot_base = draft_v_cache + (size_t)slot * DFlash2Config::SLIDING_WINDOW * 8 * 128;

    int r_lo = mt * 16 + g;
    int r_hi = r_lo + 8;

    float m_lo = -FLT_MAX, l_lo = 0.0f;
    float m_hi = -FLT_MAX, l_hi = 0.0f;
    float o_acc[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) o_acc[nt][0] = o_acc[nt][1] = o_acc[nt][2] = o_acc[nt][3] = 0.0f;

    auto kv_ptrs = [&](int kv_idx, const __nv_bfloat16*& kp, const __nv_bfloat16*& vp) {
        if (kv_idx < ctx_len) {
            int ring_pos = (ctx_start + kv_idx) & (DFlash2Config::SLIDING_WINDOW - 1);
            size_t off = ((size_t)ring_pos * 8 + kv_head) * 128;
            kp = k_slot_base + off;
            vp = v_slot_base + off;
        } else {
            int j = kv_idx - ctx_len;
            const __nv_bfloat16* row = qkv + (size_t)(blk_base + j) * DFlash2Config::QKV_DIM;
            kp = row + DFlash2Config::Q_DIM + kv_head * 128;
            vp = row + DFlash2Config::Q_DIM + DFlash2Config::KV_DIM + kv_head * 128;
        }
    };

    for (int tile = tile_start; tile < tile_end; ++tile) {
        int kv_base = tile * 32;
        int valid_kv = min(32, total_kv - kv_base);

        // Load sK[32][128]
        for (int idx = tid; idx < 32 * 16; idx += 128) {
            int r = idx >> 4;
            int c8 = (idx & 15) << 3;
            uint4 kv = make_uint4(0, 0, 0, 0);
            if (r < valid_kv) {
                const __nv_bfloat16 *kp, *vp;
                kv_ptrs(kv_base + r, kp, vp);
                kv = __ldg(reinterpret_cast<const uint4*>(kp + c8));
            }
            *reinterpret_cast<uint4*>(&sK[r][c8]) = kv;
        }
        // Load & transpose sV_T[128][16] uint32_t (16 pairs * 32 dim-quads = 512 quads)
        for (int idx = tid; idx < 16 * 32; idx += 128) {
            int k = idx >> 5;          // 0..15 -> tokens (2k, 2k+1)
            int c4 = (idx & 31) << 2;  // 0, 4, ..., 124
            int r0 = k * 2, r1 = r0 + 1;
            uint2 v0 = make_uint2(0, 0), v1 = make_uint2(0, 0);
            if (r0 < valid_kv) {
                const __nv_bfloat16 *kp, *vp;
                kv_ptrs(kv_base + r0, kp, vp);
                v0 = __ldg(reinterpret_cast<const uint2*>(vp + c4));
            }
            if (r1 < valid_kv) {
                const __nv_bfloat16 *kp, *vp;
                kv_ptrs(kv_base + r1, kp, vp);
                v1 = __ldg(reinterpret_cast<const uint2*>(vp + c4));
            }
            int d0 = c4 + 0, d1 = c4 + 1, d2 = c4 + 2, d3 = c4 + 3;
            sV_T[d0][k ^ ((d0 & 6) << 1)] = __byte_perm(v0.x, v1.x, 0x5410);
            sV_T[d1][k ^ ((d1 & 6) << 1)] = __byte_perm(v0.x, v1.x, 0x7632);
            sV_T[d2][k ^ ((d2 & 6) << 1)] = __byte_perm(v0.y, v1.y, 0x5410);
            sV_T[d3][k ^ ((d3 & 6) << 1)] = __byte_perm(v0.y, v1.y, 0x7632);
        }
        __syncthreads();

        // 1) S[16, 32] = sQ[mt*16 ..] @ sK^T (K = 128 -> 8 k16 MMAs!)
        float s0[4] = {0, 0, 0, 0}, s1[4] = {0, 0, 0, 0};
        float s2[4] = {0, 0, 0, 0}, s3[4] = {0, 0, 0, 0};
        #pragma unroll
        for (int kk = 0; kk < 128; kk += 16) {
            uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sQ[r_lo][kk + 2 * t]);
            uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sQ[r_hi][kk + 2 * t]);
            uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sQ[r_lo][kk + 2 * t + 8]);
            uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sQ[r_hi][kk + 2 * t + 8]);
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sK[nt * 8 + g][kk + 2 * t]);
                uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sK[nt * 8 + g][kk + 2 * t + 8]);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s0[nt]), "+f"(s1[nt]), "+f"(s2[nt]), "+f"(s3[nt])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }

        // 2) Bidirectional valid-length mask + register-resident online softmax
        float rmax_lo = -FLT_MAX, rmax_hi = -FLT_MAX;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            int c0 = nt * 8 + 2 * t, c1 = c0 + 1;
            if (c0 >= valid_kv) { s0[nt] = -FLT_MAX; s2[nt] = -FLT_MAX; }
            if (c1 >= valid_kv) { s1[nt] = -FLT_MAX; s3[nt] = -FLT_MAX; }
            rmax_lo = fmaxf(rmax_lo, fmaxf(s0[nt], s1[nt]));
            rmax_hi = fmaxf(rmax_hi, fmaxf(s2[nt], s3[nt]));
        }
        rmax_lo = fmaxf(rmax_lo, __shfl_xor_sync(0xffffffff, rmax_lo, 1));
        rmax_lo = fmaxf(rmax_lo, __shfl_xor_sync(0xffffffff, rmax_lo, 2));
        rmax_hi = fmaxf(rmax_hi, __shfl_xor_sync(0xffffffff, rmax_hi, 1));
        rmax_hi = fmaxf(rmax_hi, __shfl_xor_sync(0xffffffff, rmax_hi, 2));

        float m_new_lo = fmaxf(m_lo, rmax_lo);
        float m_new_hi = fmaxf(m_hi, rmax_hi);
        float alpha_lo = (rmax_lo == -FLT_MAX) ? 1.0f : ((m_lo == -FLT_MAX) ? 0.0f : expf(m_lo - m_new_lo));
        float alpha_hi = (rmax_hi == -FLT_MAX) ? 1.0f : ((m_hi == -FLT_MAX) ? 0.0f : expf(m_hi - m_new_hi));

        float p0[4], p1[4], p2[4], p3[4];
        float rsum_lo = 0.0f, rsum_hi = 0.0f;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            p0[nt] = (s0[nt] == -FLT_MAX) ? 0.0f : expf(s0[nt] - m_new_lo);
            p1[nt] = (s1[nt] == -FLT_MAX) ? 0.0f : expf(s1[nt] - m_new_lo);
            p2[nt] = (s2[nt] == -FLT_MAX) ? 0.0f : expf(s2[nt] - m_new_hi);
            p3[nt] = (s3[nt] == -FLT_MAX) ? 0.0f : expf(s3[nt] - m_new_hi);
            rsum_lo += p0[nt] + p1[nt];
            rsum_hi += p2[nt] + p3[nt];
        }
        rsum_lo += __shfl_xor_sync(0xffffffff, rsum_lo, 1);
        rsum_lo += __shfl_xor_sync(0xffffffff, rsum_lo, 2);
        rsum_hi += __shfl_xor_sync(0xffffffff, rsum_hi, 1);
        rsum_hi += __shfl_xor_sync(0xffffffff, rsum_hi, 2);

        l_lo = l_lo * alpha_lo + rsum_lo;
        l_hi = l_hi * alpha_hi + rsum_hi;
        m_lo = m_new_lo;
        m_hi = m_new_hi;

        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            o_acc[nt][0] *= alpha_lo; o_acc[nt][1] *= alpha_lo;
            o_acc[nt][2] *= alpha_hi; o_acc[nt][3] *= alpha_hi;
        }

        // 3) O += P[16, 32] @ sV_T^T (8 N8-tiles = 64 dims per warp!)
        uint32_t pa0 = dflash_pack_bf16x2(p0[0], p1[0]);
        uint32_t pa1 = dflash_pack_bf16x2(p2[0], p3[0]);
        uint32_t pa2 = dflash_pack_bf16x2(p0[1], p1[1]);
        uint32_t pa3 = dflash_pack_bf16x2(p2[1], p3[1]);

        uint32_t pb0 = dflash_pack_bf16x2(p0[2], p1[2]);
        uint32_t pb1 = dflash_pack_bf16x2(p2[2], p3[2]);
        uint32_t pb2 = dflash_pack_bf16x2(p0[3], p1[3]);
        uint32_t pb3 = dflash_pack_bf16x2(p2[3], p3[3]);

        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            int col = d_base + nt * 8 + g;
            int sw = (col & 6) << 1;
            uint32_t b0 = sV_T[col][t ^ sw];
            uint32_t b1 = sV_T[col][(t + 4) ^ sw];
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[nt][0]), "+f"(o_acc[nt][1]), "+f"(o_acc[nt][2]), "+f"(o_acc[nt][3])
                : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3), "r"(b0), "r"(b1));
            uint32_t b2 = sV_T[col][(t + 8) ^ sw];
            uint32_t b3 = sV_T[col][(t + 12) ^ sw];
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o_acc[nt][0]), "+f"(o_acc[nt][1]), "+f"(o_acc[nt][2]), "+f"(o_acc[nt][3])
                : "r"(pb0), "r"(pb1), "r"(pb2), "r"(pb3), "r"(b2), "r"(b3));
        }
        __syncthreads();
    }

    auto store_row = [&](int r, float m_v, float l_v, bool is_hi) {
        if (r >= total_q_rows) return;
        int qi = r >> 2;
        int gh = r & 3;
        int tok_idx = blk_base + qi;
        int q_head = kv_head * 4 + gh;
        size_t base_h = (size_t)tok_idx * 32 + q_head;
        size_t soff = base_h * DRAFT_ATTN_SPLITS + split_id;
        if (ds == 0 && t == 0) {
            split_lse[soff * 2 + 0] = m_v;
            split_lse[soff * 2 + 1] = l_v;
        }
        float inv_l = (l_v > 0.0f) ? (1.0f / l_v) : 0.0f;
        float* odst = split_out + soff * 128;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            int c0 = d_base + nt * 8 + 2 * t;
            odst[c0]     = (is_hi ? o_acc[nt][2] : o_acc[nt][0]) * inv_l;
            odst[c0 + 1] = (is_hi ? o_acc[nt][3] : o_acc[nt][1]) * inv_l;
        }
    };
    store_row(r_lo, m_lo, l_lo, false);
    store_row(r_hi, m_hi, l_hi, true);
}

__global__ void dflash_bidirectional_attn_reduce_kernel(
    const float* __restrict__ split_out,
    const float* __restrict__ split_lse,
    __nv_bfloat16* __restrict__ out
) {
    int tok_idx = blockIdx.x;
    int q_head = blockIdx.y;
    int d = threadIdx.x; // 0..127

    size_t base_h = (size_t)tok_idx * 32 + q_head;
    const float* lse = split_lse + base_h * DRAFT_ATTN_SPLITS * 2;
    const float* sout = split_out + base_h * DRAFT_ATTN_SPLITS * 128;

    float m_max = -FLT_MAX;
    #pragma unroll
    for (int s = 0; s < DRAFT_ATTN_SPLITS; ++s) m_max = fmaxf(m_max, lse[s * 2 + 0]);

    float acc = 0.0f, l_sum = 0.0f;
    #pragma unroll
    for (int s = 0; s < DRAFT_ATTN_SPLITS; ++s) {
        float ms = lse[s * 2 + 0];
        float ls = lse[s * 2 + 1];
        if (ls > 0.0f && ms > -FLT_MAX * 0.5f) {
            float w = ls * expf(ms - m_max);
            l_sum += w;
            acc += w * sout[(size_t)s * 128 + d];
        }
    }
    out[base_h * 128 + d] = __float2bfloat16((l_sum > 0.0f) ? (acc / l_sum) : 0.0f);
}

// ============================================================================
// Fast Top-16 Extraction over Vocab (248,320) for DFlash2 CandidateSelector
// Grid: (M), Block: 256 threads (each thread scans ~970 logits, keeps local top-16
// or local top-4, then block merge -> exact top-16 in ~30 us!)
// Wait: maintaining a thread-local top-2 or top-4 over 256 threads = 512..1024 candidates
// guarantees we find exact top-16? Wait: what if all 16 top values land in 1 thread?
// To be 100% EXACT regardless of distribution, do 16 rounds of warp+block argmax!
// Wait: 16 rounds? Each round of scanning 248,320? NO — keep a thread-local top-16 sorted array
// with fast threshold `val > min_val` check! Since 99.99% of vocab logits are < threshold after
// the first 16 elements, the insertion loop almost NEVER triggers, so 1 single pass over HBM
// takes only ~15 us per row AND is 100% mathematically exact!
// ============================================================================
__global__ void dflash_topk16_exact_kernel(
    const __nv_bfloat16* __restrict__ logits, // [M, vocab_size]
    int32_t* __restrict__ topk_ids,           // [M, 16]
    float* __restrict__ topk_vals,            // [M, 16]
    int M,
    int vocab_size
) {
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x; // 0..255

    const __nv_bfloat16* src = logits + (size_t)row * vocab_size;

    // Thread-local top-16 (sorted descending, val[15] is threshold)
    float t_val[16];
    int32_t t_id[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) { t_val[k] = -FLT_MAX; t_id[k] = -1; }

    for (int idx = tid; idx < vocab_size; idx += 256) {
        float v = __bfloat162float(src[idx]);
        if (v > t_val[15]) {
            int pos = 15;
            while (pos > 0 && v > t_val[pos - 1]) {
                t_val[pos] = t_val[pos - 1];
                t_id[pos] = t_id[pos - 1];
                --pos;
            }
            t_val[pos] = v;
            t_id[pos] = idx;
        }
    }

    // Merge across 256 threads using 16 exact block-argmax rounds over the 256 thread-heads!
    // Each thread keeps a pointer `cur` into its sorted top-16 array (initially 0).
    // Round k = 0..15: find max (t_val[cur], thread) across all 256 threads -> emit winner -> winner increments cur!
    // 100% EXACT top-16, ZERO sorting overhead, tiny 2 KB shared memory!
    __shared__ float s_val[256];
    __shared__ int32_t s_id[256];
    __shared__ int32_t s_thr[256];

    int cur = 0;
    for (int k = 0; k < 16; ++k) {
        s_val[tid] = (cur < 16) ? t_val[cur] : -FLT_MAX;
        s_id[tid]  = (cur < 16) ? t_id[cur] : -1;
        s_thr[tid] = tid;
        __syncthreads();

        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float va = s_val[tid], vb = s_val[tid + stride];
                int ia = s_id[tid], ib = s_id[tid + stride];
                if (vb > va || (vb == va && ib < ia && ib >= 0)) {
                    s_val[tid] = vb;
                    s_id[tid] = ib;
                    s_thr[tid] = s_thr[tid + stride];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            topk_ids[(size_t)row * 16 + k] = s_id[0];
            topk_vals[(size_t)row * 16 + k] = s_val[0];
        }
        if (tid == s_thr[0]) ++cur;
        __syncthreads();
    }
}

// ============================================================================
// DFlash2 CandidateSelector Walk (exact match to dflash.py::CandidateSelector!)
// For each sequence b in 0..B-1:
//   pred_token = anchor_ids[b]
//   out_block_ids[b, 0] = pred_token
//   For slot e = 0 .. num_pred - 1 (where num_pred = block_size - 1):
//     For candidate c = 0 .. 15:
//       score[c] = unary_logits[b, e, c]
//                + dot_256(pred_cb[pred_token] * proj_hidden[b, e], succ_cb[cand_ids[b, e, c]])
//     Pick chosen c (argmax if greedy, else softmax(score / temp) with uniform[b, e])
//     pred_token = cand_ids[b, e, chosen]
//     out_block_ids[b, e + 1] = pred_token
// Grid: (B), Block: 256 threads (16 candidates x 16 threads per candidate = 256 threads!
//   Each thread sums 16 of the 256 rank dims -> 16-lane warp-half reduce!)
// ============================================================================
__global__ void dflash_selector_walk_kernel(
    const int32_t* __restrict__ candidate_ids,   // [B, num_pred, 16]
    const float* __restrict__ unary_logits,      // [B, num_pred, 16]
    const __nv_bfloat16* __restrict__ proj_h,    // [B, num_pred, 256]
    const __nv_bfloat16* __restrict__ pred_cb,   // [vocab, 256]
    const __nv_bfloat16* __restrict__ succ_cb,   // [vocab, 256]
    const int32_t* __restrict__ anchor_ids,      // [B]
    const float* __restrict__ temperatures,      // [B]
    const uint8_t* __restrict__ greedy_mask,     // [B]
    const float* __restrict__ uniforms,          // [B, num_pred]
    int32_t* __restrict__ out_block_ids,         // [B, block_size]
    float* __restrict__ out_q_rows,              // [B, num_pred, 16] (nullable)
    int B,
    int block_size
) {
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x; // 0..255
    int cand = tid >> 4;   // 0..15
    int sub = tid & 15;    // 0..15 -> 16 dims of rank 256!

    int num_pred = block_size - 1;
    __shared__ int32_t s_pred_tok;
    __shared__ float s_pred_h[256];
    __shared__ float s_scores[16];

    if (tid == 0) {
        int32_t anc = anchor_ids[b];
        s_pred_tok = anc;
        out_block_ids[(size_t)b * block_size + 0] = anc;
    }
    __syncthreads();

    bool is_greedy = (greedy_mask == nullptr) ? true : (greedy_mask[b] != 0);
    float temp = (temperatures == nullptr) ? 1.0f : fmaxf(temperatures[b], 1e-5f);

    for (int e = 0; e < num_pred; ++e) {
        int pred_tok = s_pred_tok;
        // Compute pred_cb[pred_tok, tid] * proj_h[b, e, tid]
        float ph = __bfloat162float(pred_cb[(size_t)pred_tok * 256 + tid])
                 * __bfloat162float(proj_h[((size_t)b * num_pred + e) * 256 + tid]);
        s_pred_h[tid] = ph;
        __syncthreads();

        // Candidate `cand` dot product over 256 dims using 16 threads `sub = 0..15`
        int32_t cid = candidate_ids[((size_t)b * num_pred + e) * 16 + cand];
        const __nv_bfloat16* scb = succ_cb + (size_t)cid * 256 + sub * 16;
        float dot = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            dot += s_pred_h[sub * 16 + j] * __bfloat162float(scb[j]);
        }
        // Reduce across 16 contiguous lanes in half-warp
        dot += __shfl_xor_sync(0xffffffff, dot, 8);
        dot += __shfl_xor_sync(0xffffffff, dot, 4);
        dot += __shfl_xor_sync(0xffffffff, dot, 2);
        dot += __shfl_xor_sync(0xffffffff, dot, 1);

        if (sub == 0) {
            s_scores[cand] = unary_logits[((size_t)b * num_pred + e) * 16 + cand] + dot;
        }
        __syncthreads();

        if (tid == 0) {
            int chosen = 0;
            if (is_greedy) {
                float best = s_scores[0];
                for (int c = 1; c < 16; ++c) {
                    if (s_scores[c] > best) { best = s_scores[c]; chosen = c; }
                }
                if (out_q_rows != nullptr) {
                    for (int c = 0; c < 16; ++c)
                        out_q_rows[((size_t)b * num_pred + e) * 16 + c] = (c == chosen) ? 1.0f : 0.0f;
                }
            } else {
                float mx = s_scores[0];
                for (int c = 1; c < 16; ++c) mx = fmaxf(mx, s_scores[c]);
                float probs[16];
                float sum = 0.0f;
                for (int c = 0; c < 16; ++c) {
                    probs[c] = expf((s_scores[c] - mx) / temp);
                    sum += probs[c];
                }
                float inv = 1.0f / sum;
                float u = (uniforms != nullptr) ? uniforms[(size_t)b * num_pred + e] : 0.5f;
                float cdf = 0.0f;
                chosen = 15;
                for (int c = 0; c < 16; ++c) {
                    probs[c] *= inv;
                    if (out_q_rows != nullptr)
                        out_q_rows[((size_t)b * num_pred + e) * 16 + c] = probs[c];
                    cdf += probs[c];
                    if (u < cdf && chosen == 15) chosen = c;
                }
            }
            int32_t next_tok = candidate_ids[((size_t)b * num_pred + e) * 16 + chosen];
            s_pred_tok = next_tok;
            out_block_ids[(size_t)b * block_size + (e + 1)] = next_tok;
        }
        __syncthreads();
    }
}

// ============================================================================
// Greedy Verify Accept Kernel (in-graph!)
// candidates[b, 0..block_size-1] = [anchor, d1, d2, ..., d_{block_size-1}]
// target_top1[b, 0..block_size-1] = argmax(target_logits) at each position
// Draft token d_k (at candidates[b, k+1]) is accepted iff target_top1[b, k] == candidates[b, k+1]
// for all preceding drafts!
// bonus = target_top1[b, accept_len]
// committed tokens = [d_1, ..., d_{accept_len}, bonus] (count = commit_len = accept_len + 1)
// ============================================================================
__global__ void dflash_accept_greedy_kernel(
    const int32_t* __restrict__ candidates,
    const int32_t* __restrict__ target_top1,
    int32_t* __restrict__ accept_lens,
    int32_t* __restrict__ commit_lens,
    int32_t* __restrict__ bonus_ids,
    int32_t* __restrict__ out_tokens,
    int32_t* __restrict__ seq_lens,
    int B,
    int block_size
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const int32_t* cand = candidates + (size_t)b * block_size;
    const int32_t* top1 = target_top1 + (size_t)b * block_size;
    int32_t* out = out_tokens + (size_t)b * block_size;

    int acc = 0;
    for (int k = 0; k < block_size - 1; ++k) {
        if (top1[k] == cand[k + 1]) {
            out[k] = cand[k + 1];
            ++acc;
        } else {
            break;
        }
    }
    int32_t bonus = top1[acc];
    out[acc] = bonus;
    for (int k = acc + 1; k < block_size; ++k) out[k] = -1;

    accept_lens[b] = acc;
    commit_lens[b] = acc + 1;
    bonus_ids[b] = bonus;
    if (seq_lens != nullptr) {
        seq_lens[b] += (acc + 1);
    }
}

__device__ __forceinline__ float dflash_rng_u(uint64_t seed) {
    uint64_t z = seed + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z = z ^ (z >> 31);
    return (float)(z >> 40) * (1.0f / 16777216.0f);
}

// Exact unbiased speculative rejection-sampling verify kernel (Grid: B, Block: 256)
// Processes each step 0..block_size-1 sequentially inside the CTA for sequence b!
__global__ void dflash_accept_sampling_kernel(
    const __nv_bfloat16* __restrict__ target_logits, // [B * block_size, vocab_size]
    const int32_t* __restrict__ candidates,          // [B, block_size]
    const int32_t* __restrict__ cand_ids,            // [B, block_size - 1, 16]
    const float* __restrict__ q_rows,                // [B, block_size - 1, 16]
    const float* __restrict__ temperatures,          // [B]
    const int32_t* __restrict__ top_ks,              // [B]
    const float* __restrict__ top_ps,                // [B]
    const uint64_t* __restrict__ rng_seeds,          // [B]
    int32_t* __restrict__ accept_lens,
    int32_t* __restrict__ commit_lens,
    int32_t* __restrict__ bonus_ids,
    int32_t* __restrict__ out_tokens,
    int B,
    int block_size,
    int vocab_size
) {
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;

    float temp = temperatures[b];
    int top_k = top_ks[b];
    float top_p = top_ps[b];
    uint64_t seed = rng_seeds[b];
    bool is_greedy = (temp <= 1e-5f || top_k == 1);
    int K = is_greedy ? 1 : ((top_k > 0 && top_k <= 64) ? top_k : 64);

    __shared__ float s_val[256];
    __shared__ int32_t s_id[256];
    __shared__ int32_t s_thr[256];
    __shared__ float top_vals[64];
    __shared__ int32_t top_ids[64];
    __shared__ int s_done;
    __shared__ int s_acc;
    __shared__ int32_t s_bonus;

    if (tid == 0) { s_done = 0; s_acc = 0; s_bonus = 0; }
    __syncthreads();

    const int32_t* cand_row = candidates + (size_t)b * block_size;
    int32_t* out_row = out_tokens + (size_t)b * block_size;
    int num_pred = block_size - 1;

    for (int step = 0; step < block_size; ++step) {
        if (s_done) break;
        const __nv_bfloat16* src = target_logits + (size_t)(b * block_size + step) * vocab_size;

        if (is_greedy) {
            float bv = -FLT_MAX;
            int32_t bi = 0;
            for (int i = tid; i < vocab_size; i += 256) {
                float f = __bfloat162float(src[i]);
                if (f > bv || (f == bv && i < bi)) { bv = f; bi = i; }
            }
            s_val[tid] = bv; s_id[tid] = bi;
            __syncthreads();
            for (int s = 128; s > 0; s >>= 1) {
                if (tid < s && (s_val[tid + s] > s_val[tid] || (s_val[tid + s] == s_val[tid] && s_id[tid + s] < s_id[tid]))) {
                    s_val[tid] = s_val[tid + s]; s_id[tid] = s_id[tid + s];
                }
                __syncthreads();
            }
            if (tid == 0) {
                int32_t t1 = s_id[0];
                if (step < num_pred && t1 == cand_row[step + 1]) {
                    out_row[step] = t1;
                    s_acc = step + 1;
                } else {
                    out_row[step] = t1;
                    s_bonus = t1;
                    s_done = 1;
                }
            }
            __syncthreads();
            continue;
        }

        // Exact top-K extraction
        constexpr int TCAP = 24;
        float tv[TCAP]; int32_t ti[TCAP];
        #pragma unroll
        for (int k = 0; k < TCAP; ++k) { tv[k] = -FLT_MAX; ti[k] = -1; }
        for (int i = tid; i < vocab_size; i += 256) {
            float v = __bfloat162float(src[i]);
            if (v > tv[TCAP - 1]) {
                int pos = TCAP - 1;
                while (pos > 0 && v > tv[pos - 1]) { tv[pos] = tv[pos - 1]; ti[pos] = ti[pos - 1]; --pos; }
                tv[pos] = v; ti[pos] = i;
            }
        }
        int cur = 0;
        for (int k = 0; k < K; ++k) {
            s_val[tid] = (cur < TCAP) ? tv[cur] : -FLT_MAX;
            s_id[tid] = (cur < TCAP) ? ti[cur] : -1;
            s_thr[tid] = tid;
            __syncthreads();
            for (int s = 128; s > 0; s >>= 1) {
                if (tid < s) {
                    float va = s_val[tid], vb = s_val[tid + s];
                    int ia = s_id[tid], ib = s_id[tid + s];
                    if (vb > va || (vb == va && ib < ia && ib >= 0)) {
                        s_val[tid] = vb; s_id[tid] = ib; s_thr[tid] = s_thr[tid + s];
                    }
                }
                __syncthreads();
            }
            if (tid == 0) { top_vals[k] = s_val[0]; top_ids[k] = s_id[0]; }
            if (tid == s_thr[0]) ++cur;
            __syncthreads();
        }

        if (tid == 0) {
            float inv_t = 1.0f / temp;
            float mx = top_vals[0];
            float p[64];
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) { p[k] = expf((top_vals[k] - mx) * inv_t); sum += p[k]; }
            float inv_s = 1.0f / sum;
            float cdf = 0.0f;
            int cutoff = K;
            for (int k = 0; k < K; ++k) {
                p[k] *= inv_s;
                cdf += p[k];
                if (cdf >= top_p) { cutoff = k + 1; break; }
            }
            float renorm = 0.0f;
            for (int k = 0; k < cutoff; ++k) renorm += p[k];
            float inv_r = 1.0f / fmaxf(renorm, 1e-12f);
            for (int k = 0; k < cutoff; ++k) p[k] *= inv_r;

            if (step < num_pred) {
                int32_t cand = cand_row[step + 1];
                float p_cand = 0.0f;
                for (int k = 0; k < cutoff; ++k) {
                    if (top_ids[k] == cand) { p_cand = p[k]; break; }
                }
                const int32_t* cids = cand_ids + ((size_t)b * num_pred + step) * 16;
                const float* qr = q_rows + ((size_t)b * num_pred + step) * 16;
                float q_cand = 1e-8f;
                for (int j = 0; j < 16; ++j) {
                    if (cids[j] == cand) { q_cand = fmaxf(qr[j], 1e-8f); break; }
                }
                float u_acc = dflash_rng_u(seed + (uint64_t)step * 2ULL);
                if (p_cand > 0.0f && u_acc * q_cand <= p_cand) {
                    out_row[step] = cand;
                    s_acc = step + 1;
                } else {
                    // Sample bonus from residual max(0, p - q)
                    float res[64];
                    float rsum = 0.0f;
                    for (int k = 0; k < cutoff; ++k) {
                        float qk = 0.0f;
                        for (int j = 0; j < 16; ++j) {
                            if (cids[j] == top_ids[k]) { qk = qr[j]; break; }
                        }
                        res[k] = fmaxf(0.0f, p[k] - qk);
                        rsum += res[k];
                    }
                    float u_bon = dflash_rng_u(seed + (uint64_t)step * 2ULL + 1ULL);
                    int32_t chosen = top_ids[0];
                    if (rsum > 1e-9f) {
                        float t = u_bon * rsum, acc = 0.0f;
                        chosen = top_ids[cutoff - 1];
                        for (int k = 0; k < cutoff; ++k) { acc += res[k]; if (t <= acc) { chosen = top_ids[k]; break; } }
                    } else {
                        float t = u_bon, acc = 0.0f;
                        chosen = top_ids[cutoff - 1];
                        for (int k = 0; k < cutoff; ++k) { acc += p[k]; if (t <= acc) { chosen = top_ids[k]; break; } }
                    }
                    out_row[step] = chosen;
                    s_bonus = chosen;
                    s_done = 1;
                }
            } else {
                // All drafts accepted -> sample bonus from target p at last position!
                float u_bon = dflash_rng_u(seed + (uint64_t)step * 2ULL + 1ULL);
                float acc = 0.0f;
                int32_t chosen = top_ids[cutoff - 1];
                for (int k = 0; k < cutoff; ++k) { acc += p[k]; if (u_bon <= acc) { chosen = top_ids[k]; break; } }
                out_row[step] = chosen;
                s_bonus = chosen;
                s_done = 1;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int acc = s_acc;
        for (int k = acc + 1; k < block_size; ++k) out_row[k] = -1;
        accept_lens[b] = acc;
        commit_lens[b] = acc + 1;
        bonus_ids[b] = s_bonus;
    }
}

__global__ void prepare_draft_block_kernel(
    const int32_t* __restrict__ anchor_ids,
    const int32_t* __restrict__ prefix_lens,
    int32_t* __restrict__ block_ids,
    int32_t* __restrict__ positions,
    int B,
    int block_size,
    int mask_token_id
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * block_size) return;
    int b = idx / block_size;
    int col = idx % block_size;
    block_ids[idx] = (col == 0) ? anchor_ids[b] : mask_token_id;
    positions[idx] = prefix_lens[b] + col;
}

} // namespace

void launch_dflash_grouped_conv(
    const __nv_bfloat16* x,
    const __nv_bfloat16* delta_all,
    const __nv_bfloat16* base_kernel,
    __nv_bfloat16* out,
    int B, int block_size, int side, cudaStream_t stream
) {
    if (B == 0) return;
    dflash_grouped_conv_kernel<<<B * block_size, 256, 0, stream>>>(
        x, delta_all, base_kernel, out, B, block_size, side);
}

void launch_dflash_qk_norm_rope(
    __nv_bfloat16* qkv,
    const __nv_bfloat16* q_norm_w,
    const __nv_bfloat16* k_norm_w,
    const float* rope_cos,
    const float* rope_sin,
    const int32_t* positions,
    int N, float eps, cudaStream_t stream
) {
    if (N == 0) return;
    dim3 grid(N, 40);
    dflash_qk_norm_rope_kernel<<<grid, 64, 0, stream>>>(
        qkv, q_norm_w, k_norm_w, rope_cos, rope_sin, positions, N, eps);
}

void launch_dflash_append_ctx_kv(
    const __nv_bfloat16* kv,
    const __nv_bfloat16* k_norm_w,
    const float* rope_cos,
    const float* rope_sin,
    const int32_t* positions,
    const int32_t* slot_ids,
    const int32_t* commit_lens,
    __nv_bfloat16* draft_k_cache,
    __nv_bfloat16* draft_v_cache,
    int total_tokens, int block_size, float eps, cudaStream_t stream
) {
    if (total_tokens == 0) return;
    dim3 grid(total_tokens, 8);
    dflash_append_ctx_kv_kernel<<<grid, 64, 0, stream>>>(
        kv, k_norm_w, rope_cos, rope_sin, positions, slot_ids, commit_lens,
        draft_k_cache, draft_v_cache, total_tokens, block_size, eps);
}

void launch_dflash_bidirectional_attn(
    const __nv_bfloat16* qkv,
    const __nv_bfloat16* draft_k_cache,
    const __nv_bfloat16* draft_v_cache,
    const int32_t* draft_slots,
    const int32_t* prefix_lens,
    __nv_bfloat16* out,
    float* split_out,
    float* split_lse,
    int B, int block_size, cudaStream_t stream
) {
    if (B == 0) return;
    dim3 g1(DRAFT_ATTN_SPLITS, 8, B);
    dflash_bidirectional_attn_splitk_kernel<<<g1, 128, 0, stream>>>(
        qkv, draft_k_cache, draft_v_cache, draft_slots, prefix_lens,
        split_out, split_lse, B, block_size);
    dim3 g2(B * block_size, 32);
    dflash_bidirectional_attn_reduce_kernel<<<g2, 128, 0, stream>>>(
        split_out, split_lse, out);
}

void launch_dflash_topk16(
    const __nv_bfloat16* logits,
    int32_t* topk_ids,
    float* topk_vals,
    int M, int vocab_size, cudaStream_t stream
) {
    if (M == 0) return;
    dflash_topk16_exact_kernel<<<M, 256, 0, stream>>>(logits, topk_ids, topk_vals, M, vocab_size);
}

void launch_dflash_selector_walk(
    const int32_t* candidate_ids,
    const float* unary_logits,
    const __nv_bfloat16* proj_hidden,
    const __nv_bfloat16* pred_cb,
    const __nv_bfloat16* succ_cb,
    const int32_t* anchor_ids,
    const float* temperatures,
    const uint8_t* greedy_mask,
    const float* uniforms,
    int32_t* out_block_ids,
    float* out_q_rows,
    int B, int block_size, cudaStream_t stream
) {
    if (B == 0) return;
    dflash_selector_walk_kernel<<<B, 256, 0, stream>>>(
        candidate_ids, unary_logits, proj_hidden, pred_cb, succ_cb,
        anchor_ids, temperatures, greedy_mask, uniforms, out_block_ids, out_q_rows, B, block_size);
}

void launch_dflash_accept_greedy(
    const int32_t* candidates,
    const int32_t* target_top1,
    int32_t* accept_lens,
    int32_t* commit_lens,
    int32_t* bonus_ids,
    int32_t* out_tokens,
    int32_t* seq_lens,
    int B, int block_size, cudaStream_t stream
) {
    if (B == 0) return;
    dflash_accept_greedy_kernel<<<(B + 31) / 32, 32, 0, stream>>>(
        candidates, target_top1, accept_lens, commit_lens, bonus_ids, out_tokens, seq_lens, B, block_size);
}

void launch_dflash_accept_sampling(
    const __nv_bfloat16* target_logits,
    const int32_t* candidates,
    const int32_t* cand_ids,
    const float* q_rows,
    const float* temperatures,
    const int32_t* top_ks,
    const float* top_ps,
    const uint64_t* rng_seeds,
    int32_t* accept_lens,
    int32_t* commit_lens,
    int32_t* bonus_ids,
    int32_t* out_tokens,
    int B, int block_size, int vocab_size, cudaStream_t stream
) {
    if (B == 0) return;
    dflash_accept_sampling_kernel<<<B, 256, 0, stream>>>(
        target_logits, candidates, cand_ids, q_rows,
        temperatures, top_ks, top_ps, rng_seeds,
        accept_lens, commit_lens, bonus_ids, out_tokens,
        B, block_size, vocab_size);
}

void launch_prepare_draft_block(
    const int32_t* anchor_ids,
    const int32_t* prefix_lens,
    int32_t* block_ids,
    int32_t* positions,
    int B, int block_size, int mask_token_id, cudaStream_t stream
) {
    if (B == 0) return;
    int n = B * block_size;
    prepare_draft_block_kernel<<<(n + 31) / 32, 32, 0, stream>>>(
        anchor_ids, prefix_lens, block_ids, positions, B, block_size, mask_token_id);
}

} // namespace amperegen
