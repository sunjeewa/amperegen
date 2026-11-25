#include "kernels/kernels.cuh"
#include "runtime/cuda_utils.cuh"
#include "model/model_config.hpp"
#include <cmath>

namespace amperegen {

// ============================================================================
// RoPE table init: cos[pos, i] = cos(pos * theta^(-2i / rotary_dim))
// ============================================================================
__global__ void init_rope_table_kernel(
    float* __restrict__ d_cos,
    float* __restrict__ d_sin,
    int max_pos,
    int half_dim,
    float theta
) {
    int pos = blockIdx.x;
    int i = threadIdx.x;
    if (pos >= max_pos || i >= half_dim) return;
    double exponent = -2.0 * (double)i / (double)(half_dim * 2);
    double inv_freq = pow((double)theta, exponent);
    double angle = (double)pos * inv_freq;
    d_cos[(size_t)pos * half_dim + i] = (float)cos(angle);
    d_sin[(size_t)pos * half_dim + i] = (float)sin(angle);
}

void launch_init_rope_table(
    float* d_cos, float* d_sin, int max_pos, int rotary_dim, float theta, cudaStream_t stream
) {
    int half_dim = rotary_dim / 2;
    init_rope_table_kernel<<<max_pos, half_dim, 0, stream>>>(d_cos, d_sin, max_pos, half_dim, theta);
}

// ============================================================================
// Embedding lookup (vectorized uint4 = 8 BF16s per thread)
// ============================================================================
__global__ void embed_lookup_kernel(
    const int32_t* __restrict__ token_ids,
    const uint4* __restrict__ embed_table,
    uint4* __restrict__ out,
    int num_tokens,
    int hidden_vec8
) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;
    int tid = token_ids[t];
    if (tid < 0 || tid >= Qwen35Config::VOCAB_SIZE) tid = 0;
    const uint4* src = embed_table + (size_t)tid * hidden_vec8;
    uint4* dst = out + (size_t)t * hidden_vec8;
    for (int i = threadIdx.x; i < hidden_vec8; i += blockDim.x) {
        dst[i] = src[i];
    }
}

void launch_embed_lookup(
    const int32_t* token_ids,
    const __nv_bfloat16* embed_table,
    __nv_bfloat16* out,
    int num_tokens,
    int hidden_size,
    cudaStream_t stream
) {
    if (num_tokens == 0) return;
    int hidden_vec8 = hidden_size / 8;
    embed_lookup_kernel<<<num_tokens, 256, 0, stream>>>(
        token_ids,
        reinterpret_cast<const uint4*>(embed_table),
        reinterpret_cast<uint4*>(out),
        num_tokens,
        hidden_vec8);
}

// ============================================================================
// Warp / Block reduction helper
// ============================================================================
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

template <int THREADS>
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float smem[THREADS / 32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    float total = 0.0f;
    if (wid == 0) {
        total = (lane < (THREADS / 32)) ? smem[lane] : 0.0f;
        total = warp_reduce_sum(total);
        if (lane == 0) smem[0] = total;
    }
    __syncthreads();
    return smem[0];
}

// ============================================================================
// K2: RMSNorm & Fused Add + RMSNorm (hidden_size = 5120 -> 640 uint4s per row)
// Supports both GemmaRMSNorm (1 + w) and plain RMSNorm (w)
// ============================================================================
template <bool GEMMA_PLUS_ONE, bool FUSED_ADD>
__global__ void rmsnorm_5120_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int num_tokens,
    int hidden_size,
    float eps
) {
    int row = blockIdx.x;
    if (row >= num_tokens) return;

    int vec_n = hidden_size / 8; // 640 for 5120
    const uint4* x_row = reinterpret_cast<const uint4*>(x + (size_t)row * hidden_size);
    uint4* r_row = FUSED_ADD ? reinterpret_cast<uint4*>(residual + (size_t)row * hidden_size) : nullptr;
    const uint4* w_row = reinterpret_cast<const uint4*>(weight);
    uint4* o_row = reinterpret_cast<uint4*>(out + (size_t)row * hidden_size);

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_n; i += 256) {
        uint4 xv = x_row[i];
        __nv_bfloat162* xp = reinterpret_cast<__nv_bfloat162*>(&xv);
        if constexpr (FUSED_ADD) {
            uint4 rv = r_row[i];
            __nv_bfloat162* rp = reinterpret_cast<__nv_bfloat162*>(&rv);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 xf = __bfloat1622float2(xp[k]);
                float2 rf = __bfloat1622float2(rp[k]);
                float a = xf.x + rf.x;
                float b = xf.y + rf.y;
                rp[k] = __floats2bfloat162_rn(a, b);
                // Re-read bf16-rounded residual for exact parity with torch residual add!
                float2 af = __bfloat1622float2(rp[k]);
                sum_sq += af.x * af.x + af.y * af.y;
            }
            r_row[i] = rv;
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 xf = __bfloat1622float2(xp[k]);
                sum_sq += xf.x * xf.x + xf.y * xf.y;
            }
        }
    }

    float total_sq = block_reduce_sum<256>(sum_sq);
    float inv_rms = rsqrtf(total_sq / (float)hidden_size + eps);

    const uint4* src_row = FUSED_ADD ? const_cast<const uint4*>(r_row) : x_row;
    for (int i = threadIdx.x; i < vec_n; i += 256) {
        uint4 sv = src_row[i];
        uint4 wv = w_row[i];
        uint4 ov;
        __nv_bfloat162* sp = reinterpret_cast<__nv_bfloat162*>(&sv);
        __nv_bfloat162* wp = reinterpret_cast<__nv_bfloat162*>(&wv);
        __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ov);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 sf = __bfloat1622float2(sp[k]);
            float2 wf = __bfloat1622float2(wp[k]);
            float w0 = GEMMA_PLUS_ONE ? (1.0f + wf.x) : wf.x;
            float w1 = GEMMA_PLUS_ONE ? (1.0f + wf.y) : wf.y;
            op[k] = __floats2bfloat162_rn(sf.x * inv_rms * w0, sf.y * inv_rms * w1);
        }
        o_row[i] = ov;
    }
}

void launch_rmsnorm(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    int num_tokens,
    int hidden_size,
    float eps,
    bool gemma_plus_one,
    cudaStream_t stream
) {
    if (num_tokens == 0) return;
    if (gemma_plus_one) {
        rmsnorm_5120_kernel<true, false><<<num_tokens, 256, 0, stream>>>(
            x, nullptr, weight, out, num_tokens, hidden_size, eps);
    } else {
        rmsnorm_5120_kernel<false, false><<<num_tokens, 256, 0, stream>>>(
            x, nullptr, weight, out, num_tokens, hidden_size, eps);
    }
}

void launch_fused_add_rmsnorm(
    const __nv_bfloat16* x,
    __nv_bfloat16* residual,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    int num_tokens,
    int hidden_size,
    float eps,
    bool gemma_plus_one,
    cudaStream_t stream
) {
    if (num_tokens == 0) return;
    if (gemma_plus_one) {
        rmsnorm_5120_kernel<true, true><<<num_tokens, 256, 0, stream>>>(
            x, residual, weight, out, num_tokens, hidden_size, eps);
    } else {
        rmsnorm_5120_kernel<false, true><<<num_tokens, 256, 0, stream>>>(
            x, residual, weight, out, num_tokens, hidden_size, eps);
    }
}

// ============================================================================
// K3: Fused Full-Attn QKV Prepare + QK GemmaRMSNorm + Partial NeoX RoPE (64/256)
//     + FP8 E4M3 Paged KV Cache Write (scale = 1.0, matching SGLang!)
// Grid: (T, 28) -> 24 Q+gate heads (blockIdx.y 0..23) + 4 K+V heads (blockIdx.y 24..27)
// Block: 128 threads (each handles 2 elements of head_dim=256!)
//   Specifically thread i (0..127) handles:
//     - for partial RoPE (i < 32): pair (i, i + 32) -> dims 0..63!
//     - for pass-through (i >= 32): dims (2*i) and (2*i + 1) -> dims 64..255!
//   That way NeoX pair (x[i], x[i+32]) is owned by the EXACT SAME thread i with ZERO shuffles!
// ============================================================================
__global__ void full_attn_qkv_prep_kv_write_kernel(
    const __nv_bfloat16* __restrict__ qkv,      // [T, 14336]
    const __nv_bfloat16* __restrict__ q_norm_w, // [256]
    const __nv_bfloat16* __restrict__ k_norm_w, // [256]
    const float* __restrict__ rope_cos,         // [max_pos, 32]
    const float* __restrict__ rope_sin,         // [max_pos, 32]
    const int32_t* __restrict__ positions,      // [T]
    __nv_bfloat16* __restrict__ q_out,          // [T, 24, 256]
    __nv_bfloat16* __restrict__ gate_out,       // [T, 24, 256]
    __nv_fp8_e4m3* __restrict__ k_cache_layer,  // [num_blocks, 16, 4, 256]
    __nv_fp8_e4m3* __restrict__ v_cache_layer,  // [num_blocks, 16, 4, 256]
    const int32_t* __restrict__ slot_mapping,   // [T]
    int T,
    float eps
) {
    int t = blockIdx.x;
    int h = blockIdx.y; // 0..23 = Q+gate, 24..27 = K+V
    int tid = threadIdx.x; // 0..127
    if (t >= T) return;

    int pos = positions[t];
    // Map tid -> two element indices (d0, d1) within [0..255]
    int d0 = (tid < 32) ? tid : (2 * tid);
    int d1 = (tid < 32) ? (tid + 32) : (2 * tid + 1);

    const __nv_bfloat16* row = qkv + (size_t)t * Qwen35Config::ATTN_QKV_DIM;

    if (h < 24) {
        // q_gate layout per head: [q_256, gate_256] at offset h * 512
        const __nv_bfloat16* q_src = row + h * 512;
        const __nv_bfloat16* g_src = q_src + 256;

        float v0 = __bfloat162float(q_src[d0]);
        float v1 = __bfloat162float(q_src[d1]);
        float sum_sq = block_reduce_sum<128>(v0 * v0 + v1 * v1);
        float inv_rms = rsqrtf(sum_sq * (1.0f / 256.0f) + eps);

        float n0 = v0 * inv_rms * (1.0f + __bfloat162float(q_norm_w[d0]));
        float n1 = v1 * inv_rms * (1.0f + __bfloat162float(q_norm_w[d1]));

        // Round after GemmaRMSNorm to bf16 before RoPE for exact match with torch
        n0 = __bfloat162float(__float2bfloat16(n0));
        n1 = __bfloat162float(__float2bfloat16(n1));

        if (tid < 32) {
            float c = rope_cos[(size_t)pos * 32 + tid];
            float s = rope_sin[(size_t)pos * 32 + tid];
            float r0 = n0 * c - n1 * s;
            float r1 = n1 * c + n0 * s;
            n0 = r0;
            n1 = r1;
        }

        __nv_bfloat16* q_dst = q_out + ((size_t)t * 24 + h) * 256;
        __nv_bfloat16* g_dst = gate_out + ((size_t)t * 24 + h) * 256;
        q_dst[d0] = __float2bfloat16(n0);
        q_dst[d1] = __float2bfloat16(n1);
        g_dst[d0] = g_src[d0];
        g_dst[d1] = g_src[d1];
    } else {
        int kvh = h - 24; // 0..3
        const __nv_bfloat16* k_src = row + Qwen35Config::ATTN_QGATE_DIM + kvh * 256;
        const __nv_bfloat16* v_src = row + Qwen35Config::ATTN_QGATE_DIM + Qwen35Config::ATTN_KV_DIM + kvh * 256;

        float k0 = __bfloat162float(k_src[d0]);
        float k1 = __bfloat162float(k_src[d1]);
        float sum_sq = block_reduce_sum<128>(k0 * k0 + k1 * k1);
        float inv_rms = rsqrtf(sum_sq * (1.0f / 256.0f) + eps);

        float kn0 = __bfloat162float(__float2bfloat16(k0 * inv_rms * (1.0f + __bfloat162float(k_norm_w[d0]))));
        float kn1 = __bfloat162float(__float2bfloat16(k1 * inv_rms * (1.0f + __bfloat162float(k_norm_w[d1]))));

        if (tid < 32) {
            float c = rope_cos[(size_t)pos * 32 + tid];
            float s = rope_sin[(size_t)pos * 32 + tid];
            float r0 = kn0 * c - kn1 * s;
            float r1 = kn1 * c + kn0 * s;
            kn0 = r0;
            kn1 = r1;
        }

        int slot = slot_mapping[t];
        if (slot >= 0 && k_cache_layer != nullptr) {
            // Paged KV cache layout: [num_blocks, KV_BLOCK_SIZE=16, 4, 256]
            // Since slot = block_id * 16 + offset, offset in flat [slot, kvh, d] is simply (slot * 4 + kvh) * 256 + d!
            size_t base = ((size_t)slot * 4 + kvh) * 256;
            // Clamp to FP8 E4M3 finite range [-448, 448] before converting
            kn0 = fminf(fmaxf(kn0, -448.0f), 448.0f);
            kn1 = fminf(fmaxf(kn1, -448.0f), 448.0f);
            float vv0 = fminf(fmaxf(__bfloat162float(v_src[d0]), -448.0f), 448.0f);
            float vv1 = fminf(fmaxf(__bfloat162float(v_src[d1]), -448.0f), 448.0f);
            k_cache_layer[base + d0] = __nv_fp8_e4m3(kn0);
            k_cache_layer[base + d1] = __nv_fp8_e4m3(kn1);
            v_cache_layer[base + d0] = __nv_fp8_e4m3(vv0);
            v_cache_layer[base + d1] = __nv_fp8_e4m3(vv1);
        }
    }
}

void launch_full_attn_qkv_prep_and_kv_write(
    const __nv_bfloat16* qkv,
    const __nv_bfloat16* q_norm_w,
    const __nv_bfloat16* k_norm_w,
    const float* rope_cos,
    const float* rope_sin,
    const int32_t* positions,
    __nv_bfloat16* q_out,
    __nv_bfloat16* gate_out,
    __nv_fp8_e4m3* k_cache_layer,
    __nv_fp8_e4m3* v_cache_layer,
    const int32_t* slot_mapping,
    int T,
    float eps,
    cudaStream_t stream
) {
    if (T == 0) return;
    dim3 grid(T, 28);
    full_attn_qkv_prep_kv_write_kernel<<<grid, 128, 0, stream>>>(
        qkv, q_norm_w, k_norm_w, rope_cos, rope_sin, positions,
        q_out, gate_out, k_cache_layer, v_cache_layer, slot_mapping, T, eps);
}

// ============================================================================
// Attention output gate: out = attn * sigmoid(gate)
// ============================================================================
__global__ void attn_output_gate_kernel(
    const __nv_bfloat162* __restrict__ attn,
    const __nv_bfloat162* __restrict__ gate,
    __nv_bfloat162* __restrict__ out,
    int n2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n2) return;
    float2 a = __bfloat1622float2(attn[i]);
    float2 g = __bfloat1622float2(gate[i]);
    float s0 = 1.0f / (1.0f + expf(-g.x));
    float s1 = 1.0f / (1.0f + expf(-g.y));
    out[i] = __floats2bfloat162_rn(a.x * s0, a.y * s1);
}

void launch_attn_output_gate(
    const __nv_bfloat16* attn,
    const __nv_bfloat16* gate,
    __nv_bfloat16* out,
    int num_elems,
    cudaStream_t stream
) {
    int n2 = num_elems / 2;
    attn_output_gate_kernel<<<(n2 + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(attn),
        reinterpret_cast<const __nv_bfloat162*>(gate),
        reinterpret_cast<__nv_bfloat162*>(out),
        n2);
}

// ============================================================================
// K9a: SwiGLU silu_mul (gate_up [T, 2*inter] -> out [T, inter])
// ============================================================================
__global__ void silu_mul_kernel(
    const __nv_bfloat162* __restrict__ gate_up,
    __nv_bfloat162* __restrict__ out,
    int T,
    int inter2
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)T * inter2;
    if (idx >= total) return;
    int t = (int)(idx / inter2);
    int c = (int)(idx % inter2);
    const __nv_bfloat162* row = gate_up + (size_t)t * (2 * inter2);
    float2 g = __bfloat1622float2(row[c]);
    float2 u = __bfloat1622float2(row[inter2 + c]);
    float o0 = (g.x / (1.0f + expf(-g.x))) * u.x;
    float o1 = (g.y / (1.0f + expf(-g.y))) * u.y;
    out[idx] = __floats2bfloat162_rn(o0, o1);
}

void launch_silu_mul(
    const __nv_bfloat16* gate_up,
    __nv_bfloat16* out,
    int T,
    int inter,
    cudaStream_t stream
) {
    if (T == 0) return;
    int inter2 = inter / 2;
    size_t total = (size_t)T * inter2;
    silu_mul_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(gate_up),
        reinterpret_cast<__nv_bfloat162*>(out),
        T, inter2);
}

// ============================================================================
// K9b: GDN RMSNormGated (plain RMSNorm per 128-dim head, NO +1, * silu(z))
// Reads z directly from qkvz[t, 10240 + h*128 ..] (zero extra copy!)
// Grid: (T, 48), Block: 32 threads (1 warp per 128-dim head, 4 elements/thread!)
// ============================================================================
__global__ void gdn_rmsnorm_gated_kernel(
    const __nv_bfloat16* __restrict__ o,      // [T, 48 * 128]
    const __nv_bfloat16* __restrict__ qkvz,   // [T, 16384]
    const __nv_bfloat16* __restrict__ weight, // [128]
    __nv_bfloat16* __restrict__ out,          // [T, 48 * 128]
    int T,
    float eps
) {
    int t = blockIdx.x;
    int h = blockIdx.y; // 0..47
    int lane = threadIdx.x; // 0..31
    if (t >= T) return;

    const __nv_bfloat162* o_ptr = reinterpret_cast<const __nv_bfloat162*>(o + ((size_t)t * 48 + h) * 128);
    const __nv_bfloat162* z_ptr = reinterpret_cast<const __nv_bfloat162*>(
        qkvz + (size_t)t * Qwen35Config::GDN_QKVZ_DIM + Qwen35Config::GDN_CONV_DIM + h * 128);
    const __nv_bfloat162* w_ptr = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* out_ptr = reinterpret_cast<__nv_bfloat162*>(out + ((size_t)t * 48 + h) * 128);

    float2 ov0 = __bfloat1622float2(o_ptr[lane]);
    float2 ov1 = __bfloat1622float2(o_ptr[lane + 32]);
    float ss = ov0.x * ov0.x + ov0.y * ov0.y + ov1.x * ov1.x + ov1.y * ov1.y;
    ss = warp_reduce_sum(ss);
    float inv_rms = rsqrtf(ss * (1.0f / 128.0f) + eps);

    float2 wv0 = __bfloat1622float2(w_ptr[lane]);
    float2 wv1 = __bfloat1622float2(w_ptr[lane + 32]);
    float2 zv0 = __bfloat1622float2(z_ptr[lane]);
    float2 zv1 = __bfloat1622float2(z_ptr[lane + 32]);

    auto silu = [](float x) { return x / (1.0f + expf(-x)); };

    out_ptr[lane] = __floats2bfloat162_rn(
        (ov0.x * inv_rms * wv0.x) * silu(zv0.x),
        (ov0.y * inv_rms * wv0.y) * silu(zv0.y));
    out_ptr[lane + 32] = __floats2bfloat162_rn(
        (ov1.x * inv_rms * wv1.x) * silu(zv1.x),
        (ov1.y * inv_rms * wv1.y) * silu(zv1.y));
}

void launch_gdn_rmsnorm_gated(
    const __nv_bfloat16* o,
    const __nv_bfloat16* qkvz,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    int T,
    float eps,
    cudaStream_t stream
) {
    if (T == 0) return;
    dim3 grid(T, 48);
    gdn_rmsnorm_gated_kernel<<<grid, 32, 0, stream>>>(o, qkvz, weight, out, T, eps);
}

// ============================================================================
// Aux hidden state capture for DFlash2: copies [T, 5120] -> aux[t, slot*5120..]
// ============================================================================
__global__ void copy_aux_hidden_kernel(
    const uint4* __restrict__ residual,
    uint4* __restrict__ aux_concat,
    int T,
    int slot_idx
) {
    int t = blockIdx.x;
    if (t >= T) return;
    constexpr int VEC = Qwen35Config::HIDDEN_SIZE / 8; // 640
    const uint4* src = residual + (size_t)t * VEC;
    uint4* dst = aux_concat + ((size_t)t * Qwen35Config::NUM_CAPTURE_LAYERS + slot_idx) * VEC;
    for (int i = threadIdx.x; i < VEC; i += blockDim.x) {
        dst[i] = src[i];
    }
}

void launch_copy_aux_hidden(
    const __nv_bfloat16* residual,
    __nv_bfloat16* aux_concat,
    int T,
    int slot_idx,
    cudaStream_t stream
) {
    if (T == 0) return;
    copy_aux_hidden_kernel<<<T, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(residual),
        reinterpret_cast<uint4*>(aux_concat),
        T, slot_idx);
}

__global__ void bf16_add_inplace_kernel(__nv_bfloat162* __restrict__ dst, const __nv_bfloat162* __restrict__ src, size_t n2) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n2) return;
    float2 a = __bfloat1622float2(dst[i]);
    float2 b = __bfloat1622float2(src[i]);
    dst[i] = __floats2bfloat162_rn(a.x + b.x, a.y + b.y);
}

void launch_bf16_add_inplace(
    __nv_bfloat16* dst,
    const __nv_bfloat16* src,
    size_t num_elems,
    cudaStream_t stream
) {
    size_t n2 = num_elems / 2;
    if (n2 == 0) return;
    bf16_add_inplace_kernel<<<(n2 + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat162*>(dst),
        reinterpret_cast<const __nv_bfloat162*>(src), n2);
}

__global__ void gather_hidden_rows_kernel(
    const uint4* __restrict__ src,
    const int32_t* __restrict__ indices,
    uint4* __restrict__ dst,
    int B
) {
    int b = blockIdx.x;
    if (b >= B) return;
    int row = indices[b];
    constexpr int VEC = Qwen35Config::HIDDEN_SIZE / 8;
    const uint4* s = src + (size_t)row * VEC;
    uint4* d = dst + (size_t)b * VEC;
    for (int i = threadIdx.x; i < VEC; i += blockDim.x) d[i] = s[i];
}

void launch_gather_hidden_rows(
    const __nv_bfloat16* src,
    const int32_t* indices,
    __nv_bfloat16* dst,
    int B,
    cudaStream_t stream
) {
    if (B == 0) return;
    gather_hidden_rows_kernel<<<B, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(src), indices, reinterpret_cast<uint4*>(dst), B);
}

__global__ void slice_pred_hidden_kernel(
    const uint4* __restrict__ draft_hidden,
    uint4* __restrict__ pred_hidden,
    int B,
    int block_size
) {
    int out_row = blockIdx.x;
    int num_pred = block_size - 1;
    if (out_row >= B * num_pred) return;
    int b = out_row / num_pred;
    int e = out_row % num_pred;
    int in_row = b * block_size + (e + 1);
    constexpr int VEC = DFlash2Config::HIDDEN_SIZE / 8;
    const uint4* src = draft_hidden + (size_t)in_row * VEC;
    uint4* dst = pred_hidden + (size_t)out_row * VEC;
    for (int i = threadIdx.x; i < VEC; i += blockDim.x) dst[i] = src[i];
}

void launch_slice_pred_hidden(
    const __nv_bfloat16* draft_hidden,
    __nv_bfloat16* pred_hidden,
    int B,
    int block_size,
    cudaStream_t stream
) {
    int pred_T = B * (block_size - 1);
    if (pred_T <= 0) return;
    slice_pred_hidden_kernel<<<pred_T, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(draft_hidden),
        reinterpret_cast<uint4*>(pred_hidden), B, block_size);
}

} // namespace amperegen
