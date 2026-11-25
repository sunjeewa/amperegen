#include "kernels/kernels.cuh"
#include "runtime/cuda_utils.cuh"
#include "model/model_config.hpp"
#include <cmath>
#include <cfloat>

namespace amperegen {

namespace {

// ============================================================================
// Fast exact FP8 E4M3 (scale = 1.0) -> 2x BF16x2 bit-unpack on sm_80!
// ============================================================================
__device__ __forceinline__ void fp8x4_to_bf16x2_unit(uint32_t w, uint32_t& out01, uint32_t& out23) {
    constexpr uint32_t SCALE_2P120 = 0x7B807B80u;
    uint32_t t01 = __byte_perm(w, 0, 0x4140);
    uint32_t t23 = __byte_perm(w, 0, 0x4342);
    uint32_t r01 = ((t01 & 0x00800080u) << 8) | ((t01 & 0x007F007Fu) << 4);
    uint32_t r23 = ((t23 & 0x00800080u) << 8) | ((t23 & 0x007F007Fu) << 4);
    __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(&SCALE_2P120);
    __nv_bfloat162 h01 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&r01), s2);
    __nv_bfloat162 h23 = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&r23), s2);
    out01 = *reinterpret_cast<const uint32_t*>(&h01);
    out23 = *reinterpret_cast<const uint32_t*>(&h23);
}

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
    __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// XOR-swizzled 16-byte chunk offset inside [R][256] BF16 (32 chunks of 8 BF16s per row):
// Zero shared-memory bank conflicts for uint4 writes AND ldmatrix.x4 / ldmatrix.x4.trans reads!
__device__ __forceinline__ int swz256(int row, int chunk8) {
    return (chunk8 ^ (row & 7)) << 3;
}

// Cooperatively load & fast bit-unpack 32 KV tokens (2 paged blocks of 16) from FP8 -> XOR-swizzled BF16 smem!
// Uses 16-byte coalesced uint4 loads (1,176+ GB/s on GA100) with zero software transpose!
template <int THREADS>
__device__ __forceinline__ void load_kv32_fp8_swizzled(
    const __nv_fp8_e4m3* __restrict__ k_cache,
    const __nv_fp8_e4m3* __restrict__ v_cache,
    int blk0, int blk1,
    int kv_head,
    int valid_kv,
    __nv_bfloat16 (*sK)[256],
    __nv_bfloat16 (*sV)[256]
) {
    for (int idx = threadIdx.x; idx < 32 * 16; idx += THREADS) {
        int r = idx >> 4;          // 0..31
        int c16 = (idx & 15) << 4; // 0, 16, ..., 240
        int ch0 = (idx & 15) * 2;
        int ch1 = ch0 + 1;
        uint4 ko0 = make_uint4(0, 0, 0, 0), ko1 = make_uint4(0, 0, 0, 0);
        uint4 vo0 = make_uint4(0, 0, 0, 0), vo1 = make_uint4(0, 0, 0, 0);
        if (r < valid_kv) {
            int pblk = (r < 16) ? blk0 : blk1;
            size_t off = (((size_t)pblk * 16 + (r & 15)) * 4 + kv_head) * 256 + c16;
            uint4 kf = __ldcg(reinterpret_cast<const uint4*>(k_cache + off));
            uint4 vf = __ldcg(reinterpret_cast<const uint4*>(v_cache + off));
            fp8x4_to_bf16x2_unit(kf.x, ko0.x, ko0.y); fp8x4_to_bf16x2_unit(kf.y, ko0.z, ko0.w);
            fp8x4_to_bf16x2_unit(kf.z, ko1.x, ko1.y); fp8x4_to_bf16x2_unit(kf.w, ko1.z, ko1.w);
            fp8x4_to_bf16x2_unit(vf.x, vo0.x, vo0.y); fp8x4_to_bf16x2_unit(vf.y, vo0.z, vo0.w);
            fp8x4_to_bf16x2_unit(vf.z, vo1.x, vo1.y); fp8x4_to_bf16x2_unit(vf.w, vo1.z, vo1.w);
        }
        *reinterpret_cast<uint4*>(&sK[r][swz256(r, ch0)]) = ko0;
        *reinterpret_cast<uint4*>(&sK[r][swz256(r, ch1)]) = ko1;
        *reinterpret_cast<uint4*>(&sV[r][swz256(r, ch0)]) = vo0;
        *reinterpret_cast<uint4*>(&sV[r][swz256(r, ch1)]) = vo1;
    }
}

// ============================================================================
// K4a: GQA-6x Packed Tensor Core Split-K Paged FP8-KV FlashDecoding (v2)
// Uses ldmatrix.x4 / ldmatrix.x4.trans + XOR-swizzled smem + single-cycle exp2f!
//   - <1, 4> (128 threads, 40 KB smem): q_len <= 2 (<= 12 queries)
//   - <3, 2> (192 threads, 56 KB smem): q_len = 3..8 (<= 48 queries)
// Grid: (num_splits, 4 [kv_head], B)
// ============================================================================
template <int M_TILES, int D_SPLITS>
__global__ void __launch_bounds__(M_TILES * D_SPLITS * 32, 2)
paged_attn_decode_tc_splitk_fp8_kernel(
    const __nv_bfloat16* __restrict__ q,       // [B * q_len, 24, 256]
    const __nv_fp8_e4m3* __restrict__ k_cache, // [num_blocks, 16, 4, 256]
    const __nv_fp8_e4m3* __restrict__ v_cache, // [num_blocks, 16, 4, 256]
    const int32_t* __restrict__ block_tables,  // [B, max_blocks_per_seq]
    const int32_t* __restrict__ seq_lens,      // [B]
    float* __restrict__ splitk_out,            // [B * q_len, 24, num_splits, 256]
    float* __restrict__ splitk_lse,            // [B * q_len, 24, num_splits, 2]
    int B,
    int q_len,
    int max_blocks_per_seq,
    int num_splits
) {
    constexpr int NUM_WARPS = M_TILES * D_SPLITS;
    constexpr int THREADS = NUM_WARPS * 32;
    constexpr int D_PER_WARP = 256 / D_SPLITS;
    constexpr int N8_TILES = D_PER_WARP / 8;

    int split_id = blockIdx.x;
    int kv_head = blockIdx.y; // 0..3
    int b = blockIdx.z;
    if (b >= B) return;

    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int g = lane >> 2;
    int t = lane & 3;

    int mt = wid / D_SPLITS;
    int ds = wid % D_SPLITS;
    int d_base = ds * D_PER_WARP;

    extern __shared__ __align__(16) uint8_t smem_raw[];
    __nv_bfloat16 (*sQ)[256] = reinterpret_cast<__nv_bfloat16 (*)[256]>(smem_raw);
    __nv_bfloat16 (*sK)[256] = reinterpret_cast<__nv_bfloat16 (*)[256]>(smem_raw + (size_t)(M_TILES * 16) * 256 * 2);
    __nv_bfloat16 (*sV)[256] = reinterpret_cast<__nv_bfloat16 (*)[256]>(smem_raw + (size_t)(M_TILES * 16 + 32) * 256 * 2);

    int total_q_rows = q_len * 6;
    constexpr float SCALE_LOG2E = 0.0625f * 1.4426950408889634f; // 256^-0.5 * log2(e)

    for (int idx = tid; idx < (M_TILES * 16) * 32; idx += THREADS) {
        int r = idx >> 5;
        int ch = idx & 31;
        int c8 = ch << 3;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (r < total_q_rows) {
            int qi = r / 6;
            int gh = r - qi * 6;
            int tok_idx = b * q_len + qi;
            int q_head = kv_head * 6 + gh;
            val = *reinterpret_cast<const uint4*>(q + ((size_t)tok_idx * 24 + q_head) * 256 + c8);
            __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&val);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 f = __bfloat1622float2(p[k]);
                p[k] = __floats2bfloat162_rn(f.x * SCALE_LOG2E, f.y * SCALE_LOG2E);
            }
        }
        *reinterpret_cast<uint4*>(&sQ[r][swz256(r, ch)]) = val;
    }
    __syncthreads();

    int seq_len = seq_lens[b];
    int total_tiles32 = (seq_len + 31) / 32;
    int tiles_per_split = (total_tiles32 + num_splits - 1) / num_splits;
    int tile_start = split_id * tiles_per_split;
    int tile_end = min(total_tiles32, tile_start + tiles_per_split);

    int r_lo = mt * 16 + g;
    int r_hi = r_lo + 8;
    int clim_lo = (r_lo < total_q_rows) ? (seq_len - q_len + (r_lo / 6)) : -1;
    int clim_hi = (r_hi < total_q_rows) ? (seq_len - q_len + (r_hi / 6)) : -1;

    float m_lo = -FLT_MAX, l_lo = 0.0f;
    float m_hi = -FLT_MAX, l_hi = 0.0f;
    float o_acc[N8_TILES][4];
    #pragma unroll
    for (int nt = 0; nt < N8_TILES; ++nt) {
        o_acc[nt][0] = o_acc[nt][1] = o_acc[nt][2] = o_acc[nt][3] = 0.0f;
    }

    const int32_t* bt = block_tables + (size_t)b * max_blocks_per_seq;
    int num_pages = (seq_len + 15) / 16;

    int ld_q_row = mt * 16 + (lane & 15);
    int ld_q_half = (lane >> 4) & 1;
    int ld_k_row = lane & 7;
    int ld_k_half = (lane >> 3) & 1;
    int ld_k_nt = lane >> 4;

    for (int tile = tile_start; tile < tile_end; ++tile) {
        int kv_start = tile * 32;
        int valid_kv = min(32, seq_len - kv_start);
        int pg0 = tile * 2;
        int pg1 = min(num_pages - 1, pg0 + 1);

        load_kv32_fp8_swizzled<THREADS>(k_cache, v_cache, bt[pg0], bt[pg1], kv_head, valid_kv, sK, sV);
        __syncthreads();

        // 1) S[16, 32] = sQ @ sK^T via ldmatrix.x4 + mma.m16n8k16
        float s0[4] = {0, 0, 0, 0}, s1[4] = {0, 0, 0, 0};
        float s2[4] = {0, 0, 0, 0}, s3[4] = {0, 0, 0, 0};

        #pragma unroll 4
        for (int kk = 0; kk < 256; kk += 16) {
            int ch_base = kk >> 3;
            uint32_t a0, a1, a2, a3;
            uint32_t qa_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sQ[ld_q_row][swz256(ld_q_row, ch_base + ld_q_half)]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(qa_addr));

            uint32_t b0, b1, b2, b3;
            int kr0 = ld_k_nt * 8 + ld_k_row;
            uint32_t kb0_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sK[kr0][swz256(kr0, ch_base + ld_k_half)]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(kb0_addr));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                         : "+f"(s0[0]), "+f"(s1[0]), "+f"(s2[0]), "+f"(s3[0]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                         : "+f"(s0[1]), "+f"(s1[1]), "+f"(s2[1]), "+f"(s3[1]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3));

            int kr1 = 16 + ld_k_nt * 8 + ld_k_row;
            uint32_t kb1_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sK[kr1][swz256(kr1, ch_base + ld_k_half)]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(kb1_addr));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                         : "+f"(s0[2]), "+f"(s1[2]), "+f"(s2[2]), "+f"(s3[2]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                         : "+f"(s0[3]), "+f"(s1[3]), "+f"(s2[3]), "+f"(s3[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3));
        }

        // 2) Register-resident causal mask + online softmax via single-cycle exp2f
        float rmax_lo = -FLT_MAX, rmax_hi = -FLT_MAX;
        if ((kv_start + 31 > clim_lo) || (valid_kv < 32)) {
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                int c0 = nt * 8 + 2 * t, c1 = c0 + 1;
                if (c0 >= valid_kv || kv_start + c0 > clim_lo) s0[nt] = -FLT_MAX;
                if (c1 >= valid_kv || kv_start + c1 > clim_lo) s1[nt] = -FLT_MAX;
                if (c0 >= valid_kv || kv_start + c0 > clim_hi) s2[nt] = -FLT_MAX;
                if (c1 >= valid_kv || kv_start + c1 > clim_hi) s3[nt] = -FLT_MAX;
                rmax_lo = fmaxf(rmax_lo, fmaxf(s0[nt], s1[nt]));
                rmax_hi = fmaxf(rmax_hi, fmaxf(s2[nt], s3[nt]));
            }
        } else {
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                rmax_lo = fmaxf(rmax_lo, fmaxf(s0[nt], s1[nt]));
                rmax_hi = fmaxf(rmax_hi, fmaxf(s2[nt], s3[nt]));
            }
        }
        rmax_lo = fmaxf(rmax_lo, __shfl_xor_sync(0xffffffff, rmax_lo, 1));
        rmax_lo = fmaxf(rmax_lo, __shfl_xor_sync(0xffffffff, rmax_lo, 2));
        rmax_hi = fmaxf(rmax_hi, __shfl_xor_sync(0xffffffff, rmax_hi, 1));
        rmax_hi = fmaxf(rmax_hi, __shfl_xor_sync(0xffffffff, rmax_hi, 2));

        float m_new_lo = fmaxf(m_lo, rmax_lo);
        float m_new_hi = fmaxf(m_hi, rmax_hi);
        float alpha_lo = (rmax_lo == -FLT_MAX) ? 1.0f : ((m_lo == -FLT_MAX) ? 0.0f : exp2f(m_lo - m_new_lo));
        float alpha_hi = (rmax_hi == -FLT_MAX) ? 1.0f : ((m_hi == -FLT_MAX) ? 0.0f : exp2f(m_hi - m_new_hi));

        float p0[4], p1[4], p2[4], p3[4];
        float rsum_lo = 0.0f, rsum_hi = 0.0f;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            p0[nt] = (s0[nt] == -FLT_MAX) ? 0.0f : exp2f(s0[nt] - m_new_lo);
            p1[nt] = (s1[nt] == -FLT_MAX) ? 0.0f : exp2f(s1[nt] - m_new_lo);
            p2[nt] = (s2[nt] == -FLT_MAX) ? 0.0f : exp2f(s2[nt] - m_new_hi);
            p3[nt] = (s3[nt] == -FLT_MAX) ? 0.0f : exp2f(s3[nt] - m_new_hi);
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
        for (int nt = 0; nt < N8_TILES; ++nt) {
            o_acc[nt][0] *= alpha_lo; o_acc[nt][1] *= alpha_lo;
            o_acc[nt][2] *= alpha_hi; o_acc[nt][3] *= alpha_hi;
        }

        // 3) O += P[16, 32] @ sV[32, 256] via ldmatrix.x4.trans + mma.m16n8k16
        uint32_t pa0 = pack_bf16x2(p0[0], p1[0]), pa1 = pack_bf16x2(p2[0], p3[0]);
        uint32_t pa2 = pack_bf16x2(p0[1], p1[1]), pa3 = pack_bf16x2(p2[1], p3[1]);
        uint32_t pb0 = pack_bf16x2(p0[2], p1[2]), pb1 = pack_bf16x2(p2[2], p3[2]);
        uint32_t pb2 = pack_bf16x2(p0[3], p1[3]), pb3 = pack_bf16x2(p2[3], p3[3]);

        #pragma unroll
        for (int nt = 0; nt < N8_TILES; ++nt) {
            int v_ch = (d_base >> 3) + nt;
            uint32_t b0, b1, b2, b3;
            uint32_t vb_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sV[lane][swz256(lane, v_ch)]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(vb_addr));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                         : "+f"(o_acc[nt][0]), "+f"(o_acc[nt][1]), "+f"(o_acc[nt][2]), "+f"(o_acc[nt][3])
                         : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3), "r"(b0), "r"(b1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                         : "+f"(o_acc[nt][0]), "+f"(o_acc[nt][1]), "+f"(o_acc[nt][2]), "+f"(o_acc[nt][3])
                         : "r"(pb0), "r"(pb1), "r"(pb2), "r"(pb3), "r"(b2), "r"(b3));
        }
        __syncthreads();
    }

    // Convert base-2 max back to natural-log max for splitk_reduce_kernel
    constexpr float LN2 = 0.6931471805599453f;
    auto store_row = [&](int r, float m_v, float l_v, bool is_hi) {
        if (r >= total_q_rows) return;
        int qi = r / 6;
        int gh = r - qi * 6;
        int tok_idx = b * q_len + qi;
        int q_head = kv_head * 6 + gh;
        size_t base_h = (size_t)tok_idx * 24 + q_head;
        size_t split_off = base_h * num_splits + split_id;
        if (ds == 0 && t == 0) {
            splitk_lse[split_off * 2 + 0] = (m_v == -FLT_MAX) ? -FLT_MAX : (m_v * LN2);
            splitk_lse[split_off * 2 + 1] = l_v;
        }
        float inv_l = (l_v > 0.0f) ? (1.0f / l_v) : 0.0f;
        float* o_dst = splitk_out + split_off * 256;
        #pragma unroll
        for (int nt = 0; nt < N8_TILES; ++nt) {
            int c0 = d_base + nt * 8 + 2 * t;
            o_dst[c0]     = (is_hi ? o_acc[nt][2] : o_acc[nt][0]) * inv_l;
            o_dst[c0 + 1] = (is_hi ? o_acc[nt][3] : o_acc[nt][1]) * inv_l;
        }
    };
    store_row(r_lo, m_lo, l_lo, false);
    store_row(r_hi, m_hi, l_hi, true);
}

// K4b: Reduce across splits + optional fused output gate `attn * sigmoid(gate)`!
__global__ void paged_attn_decode_reduce_kernel(
    const float* __restrict__ splitk_out,
    const float* __restrict__ splitk_lse,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ out,
    int num_splits
) {
    int tok_idx = blockIdx.x;
    int q_head = blockIdx.y;
    int d = threadIdx.x;

    size_t base_h = (size_t)tok_idx * 24 + q_head;
    const float* lse_base = splitk_lse + base_h * num_splits * 2;
    const float* out_base = splitk_out + base_h * num_splits * 256;

    float m_max = -FLT_MAX;
    for (int s = 0; s < num_splits; ++s) {
        float ms = lse_base[s * 2 + 0];
        if (ms > m_max) m_max = ms;
    }

    float acc = 0.0f;
    float l_sum = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        float ms = lse_base[s * 2 + 0];
        float ls = lse_base[s * 2 + 1];
        if (ls > 0.0f && ms > -FLT_MAX * 0.5f) {
            float w = ls * expf(ms - m_max);
            l_sum += w;
            acc += w * out_base[(size_t)s * 256 + d];
        }
    }

    float val = (l_sum > 0.0f) ? (acc / l_sum) : 0.0f;
    if (gate != nullptr) {
        float g = __bfloat162float(gate[base_h * 256 + d]);
        val *= (1.0f / (1.0f + expf(-g)));
    }
    out[base_h * 256 + d] = __float2bfloat16(val);
}

// ============================================================================
// K4c: GQA-6x Packed Tensor Core Prefill Causal FlashAttention-2 (v2)
// Grid: (ceil(max_q_len / 16), 4 [kv_head], B), Block: 384 threads (12 warps = 6 Q-heads x 2 D-halves!)
// Uses ldmatrix.x4 / ldmatrix.x4.trans + XOR-swizzled smem + exp2f + 100% smem carveout!
// ============================================================================
__global__ void __launch_bounds__(384, 1)
paged_attn_prefill_tc_fp8_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_fp8_e4m3* __restrict__ k_cache,
    const __nv_fp8_e4m3* __restrict__ v_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ kv_lens,
    __nv_bfloat16* __restrict__ out,
    int B,
    int max_blocks_per_seq
) {
    int q_tile = blockIdx.x;
    int kv_head = blockIdx.y; // 0..3
    int b = blockIdx.z;
    if (b >= B) return;

    int q_bos = cu_seqlens_q[b];
    int q_eos = cu_seqlens_q[b + 1];
    int q_len = q_eos - q_bos;
    int q_row_start = q_tile * 16;
    if (q_row_start >= q_len) return;
    int valid_q = min(16, q_len - q_row_start);

    int kv_len = kv_lens[b];
    int prefix_len = kv_len - q_len;

    int tid = threadIdx.x; // 0..383
    int wid = tid >> 5;    // 0..11
    int lane = tid & 31;
    int g = lane >> 2;
    int t = lane & 3;
    int gh = wid >> 1;     // 0..5 Q-head within GQA group
    int ds = wid & 1;      // 0..1 D-half (128 dims = 16 N8 tiles)
    int d_base = ds * 128;
    int q_head = kv_head * 6 + gh;

    extern __shared__ __align__(16) uint8_t smem_raw[];
    __nv_bfloat16 (*sQ)[16][256] = reinterpret_cast<__nv_bfloat16 (*)[16][256]>(smem_raw);
    __nv_bfloat16 (*sK)[256]     = reinterpret_cast<__nv_bfloat16 (*)[256]>(smem_raw + 49152);
    __nv_bfloat16 (*sV)[256]     = reinterpret_cast<__nv_bfloat16 (*)[256]>(smem_raw + 49152 + 16384);

    constexpr float SCALE_LOG2E = 0.0625f * 1.4426950408889634f;
    int tid64 = ds * 32 + lane;
    for (int idx = tid64; idx < 16 * 32; idx += 64) {
        int r = idx >> 5;
        int ch = idx & 31;
        int c8 = ch << 3;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (r < valid_q) {
            int tok = q_bos + q_row_start + r;
            val = *reinterpret_cast<const uint4*>(q + ((size_t)tok * 24 + q_head) * 256 + c8);
            __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&val);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 f = __bfloat1622float2(p[k]);
                p[k] = __floats2bfloat162_rn(f.x * SCALE_LOG2E, f.y * SCALE_LOG2E);
            }
        }
        *reinterpret_cast<uint4*>(&sQ[gh][r][swz256(r, ch)]) = val;
    }
    __syncthreads();

    float m_lo = -FLT_MAX, l_lo = 0.0f;
    float m_hi = -FLT_MAX, l_hi = 0.0f;
    float o_acc[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        o_acc[nt][0] = o_acc[nt][1] = o_acc[nt][2] = o_acc[nt][3] = 0.0f;
    }

    int q_pos_lo = (g < valid_q) ? (prefix_len + q_row_start + g) : -1;
    int q_pos_hi = (g + 8 < valid_q) ? (prefix_len + q_row_start + g + 8) : -1;

    int max_kv_pos = min(kv_len, prefix_len + q_row_start + valid_q);
    const int32_t* bt = block_tables + (size_t)b * max_blocks_per_seq;
    int num_pages = (kv_len + 15) / 16;

    int ld_q_row = lane & 15;
    int ld_q_half = (lane >> 4) & 1;
    int ld_k_row = lane & 7;
    int ld_k_half = (lane >> 3) & 1;
    int ld_k_nt = lane >> 4;

    for (int kv_start = 0; kv_start < max_kv_pos; kv_start += 32) {
        int valid_kv = min(32, max_kv_pos - kv_start);
        int pg0 = kv_start >> 4;
        int pg1 = min(num_pages - 1, pg0 + 1);

        load_kv32_fp8_swizzled<384>(k_cache, v_cache, bt[pg0], bt[pg1], kv_head, valid_kv, sK, sV);
        __syncthreads();

        float s0[4] = {0, 0, 0, 0}, s1[4] = {0, 0, 0, 0};
        float s2[4] = {0, 0, 0, 0}, s3[4] = {0, 0, 0, 0};

        #pragma unroll 4
        for (int kk = 0; kk < 256; kk += 16) {
            int ch_base = kk >> 3;
            uint32_t a0, a1, a2, a3;
            uint32_t qa_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sQ[gh][ld_q_row][swz256(ld_q_row, ch_base + ld_q_half)]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(qa_addr));

            uint32_t b0, b1, b2, b3;
            int kr0 = ld_k_nt * 8 + ld_k_row;
            uint32_t kb0_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sK[kr0][swz256(kr0, ch_base + ld_k_half)]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(kb0_addr));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                         : "+f"(s0[0]), "+f"(s1[0]), "+f"(s2[0]), "+f"(s3[0]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                         : "+f"(s0[1]), "+f"(s1[1]), "+f"(s2[1]), "+f"(s3[1]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3));

            int kr1 = 16 + ld_k_nt * 8 + ld_k_row;
            uint32_t kb1_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sK[kr1][swz256(kr1, ch_base + ld_k_half)]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(kb1_addr));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                         : "+f"(s0[2]), "+f"(s1[2]), "+f"(s2[2]), "+f"(s3[2]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                         : "+f"(s0[3]), "+f"(s1[3]), "+f"(s2[3]), "+f"(s3[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3));
        }

        float rmax_lo = -FLT_MAX, rmax_hi = -FLT_MAX;
        if ((kv_start + 31 > q_pos_lo) || (valid_kv < 32)) {
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                int c0 = nt * 8 + 2 * t, c1 = c0 + 1;
                if (c0 >= valid_kv || kv_start + c0 > q_pos_lo) s0[nt] = -FLT_MAX;
                if (c1 >= valid_kv || kv_start + c1 > q_pos_lo) s1[nt] = -FLT_MAX;
                if (c0 >= valid_kv || kv_start + c0 > q_pos_hi) s2[nt] = -FLT_MAX;
                if (c1 >= valid_kv || kv_start + c1 > q_pos_hi) s3[nt] = -FLT_MAX;
                rmax_lo = fmaxf(rmax_lo, fmaxf(s0[nt], s1[nt]));
                rmax_hi = fmaxf(rmax_hi, fmaxf(s2[nt], s3[nt]));
            }
        } else {
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                rmax_lo = fmaxf(rmax_lo, fmaxf(s0[nt], s1[nt]));
                rmax_hi = fmaxf(rmax_hi, fmaxf(s2[nt], s3[nt]));
            }
        }
        rmax_lo = fmaxf(rmax_lo, __shfl_xor_sync(0xffffffff, rmax_lo, 1));
        rmax_lo = fmaxf(rmax_lo, __shfl_xor_sync(0xffffffff, rmax_lo, 2));
        rmax_hi = fmaxf(rmax_hi, __shfl_xor_sync(0xffffffff, rmax_hi, 1));
        rmax_hi = fmaxf(rmax_hi, __shfl_xor_sync(0xffffffff, rmax_hi, 2));

        float m_new_lo = fmaxf(m_lo, rmax_lo);
        float m_new_hi = fmaxf(m_hi, rmax_hi);
        float alpha_lo = (rmax_lo == -FLT_MAX) ? 1.0f : ((m_lo == -FLT_MAX) ? 0.0f : exp2f(m_lo - m_new_lo));
        float alpha_hi = (rmax_hi == -FLT_MAX) ? 1.0f : ((m_hi == -FLT_MAX) ? 0.0f : exp2f(m_hi - m_new_hi));

        float p0[4], p1[4], p2[4], p3[4];
        float rsum_lo = 0.0f, rsum_hi = 0.0f;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            p0[nt] = (s0[nt] == -FLT_MAX) ? 0.0f : exp2f(s0[nt] - m_new_lo);
            p1[nt] = (s1[nt] == -FLT_MAX) ? 0.0f : exp2f(s1[nt] - m_new_lo);
            p2[nt] = (s2[nt] == -FLT_MAX) ? 0.0f : exp2f(s2[nt] - m_new_hi);
            p3[nt] = (s3[nt] == -FLT_MAX) ? 0.0f : exp2f(s3[nt] - m_new_hi);
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
        for (int nt = 0; nt < 16; ++nt) {
            o_acc[nt][0] *= alpha_lo; o_acc[nt][1] *= alpha_lo;
            o_acc[nt][2] *= alpha_hi; o_acc[nt][3] *= alpha_hi;
        }

        uint32_t pa0 = pack_bf16x2(p0[0], p1[0]), pa1 = pack_bf16x2(p2[0], p3[0]);
        uint32_t pa2 = pack_bf16x2(p0[1], p1[1]), pa3 = pack_bf16x2(p2[1], p3[1]);
        uint32_t pb0 = pack_bf16x2(p0[2], p1[2]), pb1 = pack_bf16x2(p2[2], p3[2]);
        uint32_t pb2 = pack_bf16x2(p0[3], p1[3]), pb3 = pack_bf16x2(p2[3], p3[3]);

        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            int v_ch = (d_base >> 3) + nt;
            uint32_t b0, b1, b2, b3;
            uint32_t vb_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sV[lane][swz256(lane, v_ch)]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(vb_addr));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                         : "+f"(o_acc[nt][0]), "+f"(o_acc[nt][1]), "+f"(o_acc[nt][2]), "+f"(o_acc[nt][3])
                         : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3), "r"(b0), "r"(b1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                         : "+f"(o_acc[nt][0]), "+f"(o_acc[nt][1]), "+f"(o_acc[nt][2]), "+f"(o_acc[nt][3])
                         : "r"(pb0), "r"(pb1), "r"(pb2), "r"(pb3), "r"(b2), "r"(b3));
        }
        __syncthreads();
    }

    float inv_l_lo = (l_lo > 0.0f) ? (1.0f / l_lo) : 0.0f;
    float inv_l_hi = (l_hi > 0.0f) ? (1.0f / l_hi) : 0.0f;

    auto write_elem = [&](int r, int col, float val) {
        if (r >= valid_q) return;
        int tok = q_bos + q_row_start + r;
        size_t off = ((size_t)tok * 24 + q_head) * 256 + col;
        if (gate != nullptr) {
            float gv = __bfloat162float(gate[off]);
            val *= (1.0f / (1.0f + expf(-gv)));
        }
        out[off] = __float2bfloat16(val);
    };

    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        int c0 = d_base + nt * 8 + 2 * t;
        write_elem(g,     c0,     o_acc[nt][0] * inv_l_lo);
        write_elem(g,     c0 + 1, o_acc[nt][1] * inv_l_lo);
        write_elem(g + 8, c0,     o_acc[nt][2] * inv_l_hi);
        write_elem(g + 8, c0 + 1, o_acc[nt][3] * inv_l_hi);
    }
}

} // namespace

void launch_paged_attn_decode_fp8(
    const __nv_bfloat16* q,
    const __nv_bfloat16* gate,
    const __nv_fp8_e4m3* k_cache,
    const __nv_fp8_e4m3* v_cache,
    const int32_t* block_tables,
    const int32_t* seq_lens,
    __nv_bfloat16* out,
    float* splitk_out,
    float* splitk_lse,
    int B,
    int q_len,
    int max_blocks_per_seq,
    int max_seq_len_hint,
    cudaStream_t stream
) {
    if (B == 0 || q_len == 0) return;
    static bool smem_configured = false;
    if (!smem_configured) {
        CUDA_CHECK(cudaFuncSetAttribute(
            paged_attn_decode_tc_splitk_fp8_kernel<3, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
        CUDA_CHECK(cudaFuncSetAttribute(
            paged_attn_decode_tc_splitk_fp8_kernel<3, 2>,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        CUDA_CHECK(cudaFuncSetAttribute(
            paged_attn_decode_tc_splitk_fp8_kernel<1, 4>,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        smem_configured = true;
    }

    int total_tiles32 = max(1, (max_seq_len_hint + 31) / 32);
    int target_ctas = 140; // Exactly 2 CTAs/SM on 70 SMs -> minimal wave tail!
    int base_ctas = B * 4;
    int num_splits = max(1, min(64, min(total_tiles32, (target_ctas + base_ctas - 1) / base_ctas)));

    dim3 grid(num_splits, 4, B);
    if (q_len <= 2) {
        size_t smem = (size_t)(16 + 32 + 32) * 256 * 2; // 40 KB
        paged_attn_decode_tc_splitk_fp8_kernel<1, 4><<<grid, 128, smem, stream>>>(
            q, k_cache, v_cache, block_tables, seq_lens,
            splitk_out, splitk_lse, B, q_len, max_blocks_per_seq, num_splits);
    } else {
        size_t smem = (size_t)(48 + 32 + 32) * 256 * 2; // 56 KB
        paged_attn_decode_tc_splitk_fp8_kernel<3, 2><<<grid, 192, smem, stream>>>(
            q, k_cache, v_cache, block_tables, seq_lens,
            splitk_out, splitk_lse, B, q_len, max_blocks_per_seq, num_splits);
    }

    dim3 rgrid(B * q_len, 24);
    paged_attn_decode_reduce_kernel<<<rgrid, 256, 0, stream>>>(
        splitk_out, splitk_lse, gate, out, num_splits);
}

void launch_paged_attn_prefill_fp8(
    const __nv_bfloat16* q,
    const __nv_bfloat16* gate,
    const __nv_fp8_e4m3* k_cache,
    const __nv_fp8_e4m3* v_cache,
    const int32_t* block_tables,
    const int32_t* cu_seqlens_q,
    const int32_t* kv_lens,
    __nv_bfloat16* out,
    int B,
    int max_q_len,
    int max_blocks_per_seq,
    cudaStream_t stream
) {
    if (B == 0 || max_q_len == 0) return;
    static bool prefill_smem_configured = false;
    if (!prefill_smem_configured) {
        CUDA_CHECK(cudaFuncSetAttribute(
            paged_attn_prefill_tc_fp8_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
        CUDA_CHECK(cudaFuncSetAttribute(
            paged_attn_prefill_tc_fp8_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        prefill_smem_configured = true;
    }
    size_t smem = 49152 + 16384 + 16384; // 80 KB
    dim3 grid((max_q_len + 15) / 16, 4, B);
    paged_attn_prefill_tc_fp8_kernel<<<grid, 384, smem, stream>>>(
        q, gate, k_cache, v_cache, block_tables, cu_seqlens_q, kv_lens, out, B, max_blocks_per_seq);
}

} // namespace amperegen
