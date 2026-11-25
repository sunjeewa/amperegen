#include "kernels/kernels.cuh"
#include "runtime/cuda_utils.cuh"
#include "model/model_config.hpp"
#include <cmath>

namespace amperegen {

namespace {

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplus_f(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// ============================================================================
// K8a: Causal Depthwise Conv1d (k=4, SiLU) for Prefill / Extend
// qkvz: [total_T, 16384], mixed_qkv is columns [0..10239].
// conv_w: [10240, 4] BF16 (taps 0,1,2,3 from oldest to newest).
// conv_states: [num_slots, 10240, 3] BF16 (last 3 inputs: [t-3, t-2, t-1]).
// Grid: (ceil(10240/256), B), Block: 256 threads (1 thread per channel!)
// ============================================================================
__global__ void gdn_conv1d_prefill_kernel(
    const __nv_bfloat16* __restrict__ qkvz,
    const __nv_bfloat16* __restrict__ conv_w,
    __nv_bfloat16* __restrict__ conv_states,
    const int32_t* __restrict__ state_slots,
    const int32_t* __restrict__ cu_seqlens,
    __nv_bfloat16* __restrict__ conv_out,
    int B
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (b >= B || c >= Qwen35Config::GDN_CONV_DIM) return;

    int bos = cu_seqlens[b];
    int eos = cu_seqlens[b + 1];
    int T = eos - bos;
    if (T <= 0) return;

    int slot = state_slots[b];
    __nv_bfloat16* st = conv_states + ((size_t)slot * Qwen35Config::GDN_CONV_DIM + c) * 3;

    float w0 = __bfloat162float(conv_w[c * 4 + 0]);
    float w1 = __bfloat162float(conv_w[c * 4 + 1]);
    float w2 = __bfloat162float(conv_w[c * 4 + 2]);
    float w3 = __bfloat162float(conv_w[c * 4 + 3]);

    float s0 = __bfloat162float(st[0]);
    float s1 = __bfloat162float(st[1]);
    float s2 = __bfloat162float(st[2]);

    for (int t = 0; t < T; ++t) {
        float x = __bfloat162float(qkvz[(size_t)(bos + t) * Qwen35Config::GDN_QKVZ_DIM + c]);
        float y = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x;
        conv_out[(size_t)(bos + t) * Qwen35Config::GDN_CONV_DIM + c] = __float2bfloat16(silu_f(y));
        s0 = s1;
        s1 = s2;
        s2 = x;
    }

    st[0] = __float2bfloat16(s0);
    st[1] = __float2bfloat16(s1);
    st[2] = __float2bfloat16(s2);
}

// ============================================================================
// K8b: Causal Depthwise Conv1d (k=4, SiLU) for Decode / Spec Verify (q_len=1..8)
// ============================================================================
__global__ void gdn_conv1d_decode_verify_kernel(
    const __nv_bfloat16* __restrict__ qkvz,
    const __nv_bfloat16* __restrict__ conv_w,
    __nv_bfloat16* __restrict__ conv_states,
    __nv_bfloat16* __restrict__ conv_inter, // [B, q_len, 10240, 3] (nullable if commit_inplace)
    const int32_t* __restrict__ state_slots,
    __nv_bfloat16* __restrict__ conv_out,
    int B,
    int q_len,
    bool commit_inplace
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (b >= B || c >= Qwen35Config::GDN_CONV_DIM) return;

    int slot = state_slots[b];
    __nv_bfloat16* st = conv_states + ((size_t)slot * Qwen35Config::GDN_CONV_DIM + c) * 3;

    float w0 = __bfloat162float(conv_w[c * 4 + 0]);
    float w1 = __bfloat162float(conv_w[c * 4 + 1]);
    float w2 = __bfloat162float(conv_w[c * 4 + 2]);
    float w3 = __bfloat162float(conv_w[c * 4 + 3]);

    float s0 = __bfloat162float(st[0]);
    float s1 = __bfloat162float(st[1]);
    float s2 = __bfloat162float(st[2]);

    for (int qi = 0; qi < q_len; ++qi) {
        int tok = b * q_len + qi;
        float x = __bfloat162float(qkvz[(size_t)tok * Qwen35Config::GDN_QKVZ_DIM + c]);
        float y = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x;
        conv_out[(size_t)tok * Qwen35Config::GDN_CONV_DIM + c] = __float2bfloat16(silu_f(y));
        s0 = s1;
        s1 = s2;
        s2 = x;
        if (!commit_inplace && conv_inter != nullptr) {
            __nv_bfloat16* dst = conv_inter + (((size_t)b * q_len + qi) * Qwen35Config::GDN_CONV_DIM + c) * 3;
            dst[0] = __float2bfloat16(s0);
            dst[1] = __float2bfloat16(s1);
            dst[2] = __float2bfloat16(s2);
        }
    }

    if (commit_inplace) {
        st[0] = __float2bfloat16(s0);
        st[1] = __float2bfloat16(s1);
        st[2] = __float2bfloat16(s2);
    }
}

// ============================================================================
// K5 / K7: GDN Recurrent Decode & Spec-Verify Kernel
// Exact SGLang math:
//   hk = hv / 3 (V2 verified!)
//   q_hat = L2norm(q[hk]) * 128^-0.5, k_hat = L2norm(k[hk]) (V1 verified!)
//   g = -exp(A_log[hv]) * softplus(a[hv] + dt_bias[hv]), beta = sigmoid(b[hv])
//   S = S * exp(g)
//   delta = (v - S @ k_hat) * beta
//   S = S + delta @ k_hat^T
//   o = S @ q_hat   (UPDATE BEFORE OUTPUT — V3 verified!)
//
// Thread layout for blazing speed on 70 SMs:
//   Grid: (192, B) -> 48 V-heads x 4 quarters of V (BV=32 per CTA -> 192 CTAs = 2.74 waves/SM, zero wave-2 tail!)
//   Block: 128 threads = 32 V-rows x 4 threads per V-row (each thread owns 32 K-elements, vectorized float4 smem!)
//   State S[32, 128] stays 100% in FP32 registers across all q_len steps (32 regs/thread)!
// ============================================================================
__global__ void gdn_recurrent_decode_verify_kernel(
    const __nv_bfloat16* __restrict__ conv_out, // [B * q_len, 10240]
    const __nv_bfloat16* __restrict__ ba,       // [B * q_len, 96] (b[0..47], a[48..95])
    const float* __restrict__ A_log,            // [48]
    const float* __restrict__ dt_bias,          // [48]
    __nv_bfloat16* __restrict__ ssm_states,     // [num_slots, 48, 128, 128]
    __nv_bfloat16* __restrict__ ssm_inter,      // [B, q_len, 48, 128, 128] (nullable)
    const int32_t* __restrict__ state_slots,    // [B]
    __nv_bfloat16* __restrict__ out,            // [B * q_len, 48, 128]
    int B,
    int q_len,
    bool commit_inplace
) {
    int hv = blockIdx.x >> 2;      // 0..47
    int v_qtr = blockIdx.x & 3;    // 0..3 (32 V-rows per CTA!)
    int b = blockIdx.y;
    if (b >= B) return;

    int hk = hv / 3;               // 0..15 (GQA 3:1)
    int tid = threadIdx.x;         // 0..127
    int v_local = tid >> 2;        // 0..31
    int v_row = v_qtr * 32 + v_local;
    int lane_k = tid & 3;          // 0..3 -> owns k = lane_k*32 .. lane_k*32+31 (contiguous!)

    int slot = state_slots[b];
    __nv_bfloat16* st_row = ssm_states + (((size_t)slot * 48 + hv) * 128 + v_row) * 128 + lane_k * 32;

    // Load 32 state elements (4 x uint4 = 64 bytes coalesced!) into FP32 registers
    float h[32];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint4 v = *reinterpret_cast<const uint4*>(st_row + i * 8);
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __bfloat1622float2(p[j]);
            h[i * 8 + j * 2 + 0] = f.x;
            h[i * 8 + j * 2 + 1] = f.y;
        }
    }

    __shared__ __align__(16) float s_q[128];
    __shared__ __align__(16) float s_k[128];
    __shared__ float s_exp_g;
    __shared__ float s_beta;

    float a_log_val = A_log[hv];
    float dt_bias_val = dt_bias[hv];
    constexpr float Q_SCALE = 0.08838834764831845f; // 128^-0.5

    for (int qi = 0; qi < q_len; ++qi) {
        int tok = b * q_len + qi;
        const __nv_bfloat16* row_qkv = conv_out + (size_t)tok * Qwen35Config::GDN_CONV_DIM;

        // Warp 0 (threads 0..31) loads & L2-normalizes q[hk] and k[hk] with uint2 vectorized loads
        if (tid < 32) {
            uint2 qu = *reinterpret_cast<const uint2*>(row_qkv + hk * 128 + tid * 4);
            uint2 ku = *reinterpret_cast<const uint2*>(row_qkv + 2048 + hk * 128 + tid * 4);
            const __nv_bfloat162* qp2 = reinterpret_cast<const __nv_bfloat162*>(&qu);
            const __nv_bfloat162* kp2 = reinterpret_cast<const __nv_bfloat162*>(&ku);
            float2 q0 = __bfloat1622float2(qp2[0]), q1 = __bfloat1622float2(qp2[1]);
            float2 k0 = __bfloat1622float2(kp2[0]), k1 = __bfloat1622float2(kp2[1]);
            float qss = q0.x*q0.x + q0.y*q0.y + q1.x*q1.x + q1.y*q1.y;
            float kss = k0.x*k0.x + k0.y*k0.y + k1.x*k1.x + k1.y*k1.y;
            float q_inv = rsqrtf(warp_sum(qss) + 1e-6f) * Q_SCALE;
            float k_inv = rsqrtf(warp_sum(kss) + 1e-6f);
            *reinterpret_cast<float4*>(&s_q[tid * 4]) = make_float4(q0.x*q_inv, q0.y*q_inv, q1.x*q_inv, q1.y*q_inv);
            *reinterpret_cast<float4*>(&s_k[tid * 4]) = make_float4(k0.x*k_inv, k0.y*k_inv, k1.x*k_inv, k1.y*k_inv);
            if (tid == 0) {
                float bv = __bfloat162float(ba[(size_t)tok * 96 + hv]);
                float av = __bfloat162float(ba[(size_t)tok * 96 + 48 + hv]);
                float g = -expf(a_log_val) * softplus_f(av + dt_bias_val);
                s_exp_g = expf(g);
                s_beta = 1.0f / (1.0f + expf(-bv));
            }
        }
        __syncthreads();

        float exp_g = s_exp_g;
        float beta = s_beta;
        float v_val = __bfloat162float(row_qkv[4096 + hv * 128 + v_row]);

        // 1) Decay state and compute dot_k = S @ k_hat using vectorized float4 smem loads
        float dot_k = 0.0f;
        const float4* sk4 = reinterpret_cast<const float4*>(s_k + lane_k * 32);
        const float4* sq4 = reinterpret_cast<const float4*>(s_q + lane_k * 32);
        float4 k_reg[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float4 kv = sk4[i];
            k_reg[i] = kv;
            h[i*4+0] *= exp_g; h[i*4+1] *= exp_g; h[i*4+2] *= exp_g; h[i*4+3] *= exp_g;
            dot_k += h[i*4+0]*kv.x + h[i*4+1]*kv.y + h[i*4+2]*kv.z + h[i*4+3]*kv.w;
        }
        dot_k += __shfl_xor_sync(0xffffffff, dot_k, 1);
        dot_k += __shfl_xor_sync(0xffffffff, dot_k, 2);

        // 2) Rank-1 delta update & output dot_q = S_new @ q_hat
        float delta = (v_val - dot_k) * beta;
        float dot_q = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float4 kv = k_reg[i];
            float4 qv = sq4[i];
            h[i*4+0] += delta * kv.x; h[i*4+1] += delta * kv.y;
            h[i*4+2] += delta * kv.z; h[i*4+3] += delta * kv.w;
            dot_q += h[i*4+0]*qv.x + h[i*4+1]*qv.y + h[i*4+2]*qv.z + h[i*4+3]*qv.w;
        }
        dot_q += __shfl_xor_sync(0xffffffff, dot_q, 1);
        dot_q += __shfl_xor_sync(0xffffffff, dot_q, 2);

        if (lane_k == 0) {
            out[((size_t)tok * 48 + hv) * 128 + v_row] = __float2bfloat16(dot_q);
        }

        if (!commit_inplace && ssm_inter != nullptr) {
            __nv_bfloat16* dst = ssm_inter + ((((size_t)b * q_len + qi) * 48 + hv) * 128 + v_row) * 128 + lane_k * 32;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint4 v;
                __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&v);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    p[j] = __floats2bfloat162_rn(h[i * 8 + j * 2 + 0], h[i * 8 + j * 2 + 1]);
                }
                *reinterpret_cast<uint4*>(dst + i * 8) = v;
            }
        }
        __syncthreads();
    }

    if (commit_inplace) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint4 v;
            __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&v);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                p[j] = __floats2bfloat162_rn(h[i * 8 + j * 2 + 0], h[i * 8 + j * 2 + 1]);
            }
            *reinterpret_cast<uint4*>(st_row + i * 8) = v;
        }
    }
}

// ============================================================================
// K7b: Fused Replay-Commit across ALL 48 GDN layers in ONE kernel launch!
// Replays ONLY `commit_lens[b]` (1..q_len) steps of GDN recurrence in-place on ssm_states
// AND copies conv_inter[commit_lens[b]-1] -> conv_states!
// Zero ssm_inter VRAM needed, ~0.10 ms total across all 48 layers!
// Grid: (96, 48 [gdn_layer], B), Block: 256 threads
// ============================================================================
__global__ void gdn_commit_replay_ssm_kernel(
    __nv_bfloat16* const* __restrict__ ssm_states_ptrs,
    const __nv_bfloat16* const* __restrict__ conv_out_ptrs,
    const __nv_bfloat16* const* __restrict__ ba_ptrs,
    const float* const* __restrict__ A_log_ptrs,
    const float* const* __restrict__ dt_bias_ptrs,
    const int32_t* __restrict__ state_slots,
    const int32_t* __restrict__ commit_lens,
    int B,
    int q_len
) {
    int hv = blockIdx.x >> 1;
    int v_half = blockIdx.x & 1;
    int layer_gdn = blockIdx.y; // 0..47
    int b = blockIdx.z;
    if (b >= B) return;

    int steps = commit_lens[b];
    if (steps <= 0) return;
    if (steps > q_len) steps = q_len;

    int hk = hv / 3;
    int tid = threadIdx.x;
    int v_local = tid >> 2;
    int v_row = v_half * 64 + v_local;
    int lane_k = tid & 3;
    int k_off = lane_k * 32;

    int slot = state_slots[b];
    __nv_bfloat16* st_row = ssm_states_ptrs[layer_gdn] + (((size_t)slot * 48 + hv) * 128 + v_row) * 128 + k_off;
    const __nv_bfloat16* conv_out = conv_out_ptrs[layer_gdn];
    const __nv_bfloat16* ba = ba_ptrs[layer_gdn];
    float a_log_val = A_log_ptrs[layer_gdn][hv];
    float dt_bias_val = dt_bias_ptrs[layer_gdn][hv];

    float h[32];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint4 v = *reinterpret_cast<const uint4*>(st_row + i * 8);
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __bfloat1622float2(p[j]);
            h[i * 8 + j * 2 + 0] = f.x;
            h[i * 8 + j * 2 + 1] = f.y;
        }
    }

    __shared__ float s_k[128];
    __shared__ float s_exp_g;
    __shared__ float s_beta;

    for (int qi = 0; qi < steps; ++qi) {
        int tok = b * q_len + qi;
        const __nv_bfloat16* row_qkv = conv_out + (size_t)tok * Qwen35Config::GDN_CONV_DIM;

        if (tid < 32) {
            const __nv_bfloat16* kp = row_qkv + 2048 + hk * 128;
            float kv[4];
            float kss = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                kv[j] = __bfloat162float(kp[tid * 4 + j]);
                kss += kv[j] * kv[j];
            }
            float k_inv = rsqrtf(warp_sum(kss) + 1e-6f);
            #pragma unroll
            for (int j = 0; j < 4; ++j) s_k[tid * 4 + j] = kv[j] * k_inv;
            if (tid == 0) {
                float bv = __bfloat162float(ba[(size_t)tok * 96 + hv]);
                float av = __bfloat162float(ba[(size_t)tok * 96 + 48 + hv]);
                float g = -expf(a_log_val) * softplus_f(av + dt_bias_val);
                s_exp_g = expf(g);
                s_beta = 1.0f / (1.0f + expf(-bv));
            }
        }
        __syncthreads();

        float exp_g = s_exp_g;
        float beta = s_beta;
        float v_val = __bfloat162float(row_qkv[4096 + hv * 128 + v_row]);

        float dot_k = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            h[j] *= exp_g;
            dot_k += h[j] * s_k[k_off + j];
        }
        dot_k += __shfl_xor_sync(0xffffffff, dot_k, 1);
        dot_k += __shfl_xor_sync(0xffffffff, dot_k, 2);

        float delta = (v_val - dot_k) * beta;
        #pragma unroll
        for (int j = 0; j < 32; ++j) h[j] += delta * s_k[k_off + j];
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint4 v;
        __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&v);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            p[j] = __floats2bfloat162_rn(h[i * 8 + j * 2 + 0], h[i * 8 + j * 2 + 1]);
        }
        *reinterpret_cast<uint4*>(st_row + i * 8) = v;
    }
}

__global__ void gdn_commit_conv_states_kernel(
    __nv_bfloat16* const* __restrict__ conv_states_ptrs,
    __nv_bfloat16* const* __restrict__ conv_inter_ptrs,
    const int32_t* __restrict__ state_slots,
    const int32_t* __restrict__ commit_lens,
    int B,
    int q_len
) {
    int layer_gdn = blockIdx.x;
    int b = blockIdx.y;
    if (b >= B) return;
    int slot = state_slots[b];
    int step = commit_lens[b] - 1;
    if (step < 0) step = 0;
    if (step >= q_len) step = q_len - 1;

    constexpr size_t CONV_VEC = (10240ULL * 3) / 8;
    const uint4* src_conv = reinterpret_cast<const uint4*>(
        conv_inter_ptrs[layer_gdn] + ((size_t)b * q_len + step) * (10240ULL * 3));
    uint4* dst_conv = reinterpret_cast<uint4*>(
        conv_states_ptrs[layer_gdn] + (size_t)slot * (10240ULL * 3));
    for (size_t i = threadIdx.x; i < CONV_VEC; i += blockDim.x) {
        dst_conv[i] = src_conv[i];
    }
}

// ============================================================================
// K6: GDN Chunk-64 WY Prefill / Extend — AMPERE TENSOR CORE (mma.m16n8k16 + ldmatrix)!
// Cuts 1-layer T=8192 GDN prefill from 71.5 ms -> ~3 ms (24x speedup!)
// ============================================================================
static constexpr int GDN_CHUNK = 64;

__device__ __forceinline__ int gdn_swz128(int row, int ch8) {
    return (ch8 ^ (row & 7)) << 3;
}
__device__ __forceinline__ int gdn_swz64(int row, int ch8) {
    return (ch8 ^ (row & 7)) << 3;
}

// Helper: 1 warp computes C[16, 64] = A[16, 128] @ B[64, 128]^T using ldmatrix.x4 + mma.m16n8k16!
__device__ __forceinline__ void tc_gemm_16x64x128_nt(
    const __nv_bfloat16 (*sA)[128], int a_row_base,
    const __nv_bfloat16 (*sB)[128],
    float acc[8][4], int lane
) {
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;

    int ld_a_row = a_row_base + (lane & 15);
    int ld_a_half = (lane >> 4) & 1;
    int ld_b_row = lane & 7;
    int ld_b_half = (lane >> 3) & 1;
    int ld_b_nt = lane >> 4;

    #pragma unroll 4
    for (int kk = 0; kk < 128; kk += 16) {
        int ch_base = kk >> 3;
        uint32_t a0, a1, a2, a3;
        uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sA[ld_a_row][gdn_swz128(ld_a_row, ch_base + ld_a_half)]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(a_addr));

        #pragma unroll
        for (int bgrp = 0; bgrp < 4; ++bgrp) {
            int br = bgrp * 16 + ld_b_nt * 8 + ld_b_row;
            uint32_t b0, b1, b2, b3;
            uint32_t b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sB[br][gdn_swz128(br, ch_base + ld_b_half)]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(b_addr));
            int n0 = bgrp * 2;
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                         : "+f"(acc[n0][0]), "+f"(acc[n0][1]), "+f"(acc[n0][2]), "+f"(acc[n0][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                         : "+f"(acc[n0+1][0]), "+f"(acc[n0+1][1]), "+f"(acc[n0+1][2]), "+f"(acc[n0+1][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3));
        }
    }
}

// Helper: 1 warp accumulates C[16, 128] += A[16, 64] @ B[64, 128] using ldmatrix.x4 + ldmatrix.x4.trans!
__device__ __forceinline__ void tc_gemm_16x128x64_nn_acc(
    const __nv_bfloat16 (*sA)[64], int a_row_base,
    const __nv_bfloat16 (*sB)[128],
    float acc[16][4], int lane
) {
    int ld_a_row = a_row_base + (lane & 15);
    int ld_a_half = (lane >> 4) & 1;

    #pragma unroll
    for (int kk = 0; kk < 64; kk += 32) {
        int ch0 = kk >> 3;
        uint32_t a0, a1, a2, a3, a4, a5, a6, a7;
        uint32_t a0_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sA[ld_a_row][gdn_swz64(ld_a_row, ch0 + ld_a_half)]));
        uint32_t a1_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sA[ld_a_row][gdn_swz64(ld_a_row, ch0 + 2 + ld_a_half)]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(a0_addr));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(a4), "=r"(a5), "=r"(a6), "=r"(a7) : "r"(a1_addr));

        int br = kk + lane;
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            uint32_t b0, b1, b2, b3;
            uint32_t b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sB[br][gdn_swz128(br, nt)]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(b_addr));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                         : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]), "+f"(acc[nt][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                         : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]), "+f"(acc[nt][3])
                         : "r"(a4), "r"(a5), "r"(a6), "r"(a7), "r"(b2), "r"(b3));
        }
    }
}

__global__ void compute_cu_chunks_kernel(
    const int32_t* __restrict__ cu_seqlens,
    int32_t* __restrict__ cu_chunks,
    int B
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int acc = 0;
    cu_chunks[0] = 0;
    for (int i = 0; i < B; ++i) {
        int len = cu_seqlens[i + 1] - cu_seqlens[i];
        int nc = (len + GDN_CHUNK - 1) / GDN_CHUNK;
        acc += nc;
        cu_chunks[i + 1] = acc;
    }
}

// Kernel 6a (Tensor Core): Intra-chunk prep (K@K^T, (I+A)^-1, u = T@(beta*v), w = T@(beta*exp(G)*k))!
// Zero local memory stack frame, 40.5 KB static smem, 4 warps = 128 threads!
__global__ void gdn_chunk_intra_prep_compact_kernel(
    const __nv_bfloat16* __restrict__ conv_out,
    const __nv_bfloat16* __restrict__ ba,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const int32_t* __restrict__ cu_seqlens,
    float* __restrict__ g_cum_out,
    __nv_bfloat16* __restrict__ q_norm_out,
    __nv_bfloat16* __restrict__ k_norm_out,
    __nv_bfloat16* __restrict__ w_out,
    __nv_bfloat16* __restrict__ u_out,
    int B
) {
    int c_idx = blockIdx.x;
    int hv = blockIdx.y; // 0..47
    int b = blockIdx.z;
    if (b >= B) return;

    int bos = cu_seqlens[b];
    int eos = cu_seqlens[b + 1];
    int seq_len = eos - bos;
    int t0 = c_idx * GDN_CHUNK;
    if (t0 >= seq_len) return;
    int C = min(GDN_CHUNK, seq_len - t0);
    int hk = hv / 3;
    int tid = threadIdx.x; // 0..127
    int wid = tid >> 5;
    int lane = tid & 31;
    int g = lane >> 2;
    int t = lane & 3;

    __shared__ float s_g[GDN_CHUNK];
    __shared__ float s_beta[GDN_CHUNK];
    __shared__ __nv_bfloat16 s_kv[GDN_CHUNK][128]; // 16 KB (swizzled gdn_swz128)
    __shared__ float s_A[GDN_CHUNK][GDN_CHUNK];    // 16 KB
    __shared__ __nv_bfloat16 s_T[GDN_CHUNK][64];   // 8 KB  (swizzled gdn_swz64)

    // Step 1: g, beta, cumsum(g)
    if (tid < C) {
        int tok = bos + t0 + tid;
        float bv = __bfloat162float(ba[(size_t)tok * 96 + hv]);
        float av = __bfloat162float(ba[(size_t)tok * 96 + 48 + hv]);
        s_g[tid] = -expf(A_log[hv]) * softplus_f(av + dt_bias[hv]);
        s_beta[tid] = 1.0f / (1.0f + expf(-bv));
    } else if (tid < GDN_CHUNK) {
        s_g[tid] = 0.0f;
        s_beta[tid] = 0.0f;
    }
    __syncthreads();

    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < C; ++i) {
            acc += s_g[i];
            s_g[i] = acc;
            g_cum_out[(size_t)(bos + t0 + i) * 48 + hv] = acc;
        }
    }
    __syncthreads();

    // Step 2: L2-norm k (and q if hv % 3 == 0) into swizzled s_kv + global outputs
    constexpr float Q_SCALE = 0.08838834764831845f;
    bool write_qk = ((hv % 3) == 0);
    for (int i = wid; i < GDN_CHUNK; i += 4) {
        if (i < C) {
            int tok = bos + t0 + i;
            const __nv_bfloat16* row = conv_out + (size_t)tok * Qwen35Config::GDN_CONV_DIM;
            const __nv_bfloat16* kp = row + 2048 + hk * 128;
            float kv[4];
            float kss = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                kv[j] = __bfloat162float(kp[lane * 4 + j]);
                kss += kv[j] * kv[j];
            }
            float k_inv = rsqrtf(warp_sum(kss) + 1e-6f);
            int ch = lane >> 1;      // 0..15 (8-BF16 chunk)
            int sub = (lane & 1) * 4;// 0 or 4
            int sw_col = gdn_swz128(i, ch) + sub;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                __nv_bfloat16 kn = __float2bfloat16(kv[j] * k_inv);
                s_kv[i][sw_col + j] = kn;
                if (write_qk) k_norm_out[((size_t)tok * 16 + hk) * 128 + lane * 4 + j] = kn;
            }
            if (write_qk) {
                const __nv_bfloat16* qp = row + hk * 128;
                float qv[4];
                float qss = 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    qv[j] = __bfloat162float(qp[lane * 4 + j]);
                    qss += qv[j] * qv[j];
                }
                float q_inv = rsqrtf(warp_sum(qss) + 1e-6f) * Q_SCALE;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    q_norm_out[((size_t)tok * 16 + hk) * 128 + lane * 4 + j] = __float2bfloat16(qv[j] * q_inv);
                }
            }
        } else {
            int ch = lane >> 1;
            int sub = (lane & 1) * 4;
            int sw_col = gdn_swz128(i, ch) + sub;
            #pragma unroll
            for (int j = 0; j < 4; ++j) s_kv[i][sw_col + j] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // Step 3: Tensor Core A = tril(beta * exp(G_i - G_j) * (K @ K^T))
    float kk_acc[8][4];
    tc_gemm_16x64x128_nt(s_kv, wid * 16, s_kv, kk_acc, lane);
    int r0 = wid * 16 + g, r1 = r0 + 8;
    float b0 = s_beta[r0], g0 = s_g[r0];
    float b1 = s_beta[r1], g1 = s_g[r1];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        int c0 = nt * 8 + 2 * t, c1 = c0 + 1;
        auto setA = [&](int r, int c, float br, float gr, float dot) {
            if (r < C && c < r) s_A[r][c] = br * expf(fminf(gr - s_g[c], 0.0f)) * dot;
            else s_A[r][c] = (r == c) ? 1.0f : 0.0f;
        };
        setA(r0, c0, b0, g0, kk_acc[nt][0]);
        setA(r0, c1, b0, g0, kk_acc[nt][1]);
        setA(r1, c0, b1, g1, kk_acc[nt][2]);
        setA(r1, c1, b1, g1, kk_acc[nt][3]);
    }
    __syncthreads();

    // Step 4: Invert unit lower triangular (I + A) -> T in-place
    for (int i = 1; i < C; ++i) {
        float val = 0.0f;
        if (tid < i) {
            int j = tid;
            float acc = -s_A[i][j];
            for (int m = j + 1; m < i; ++m) acc -= s_A[i][m] * s_A[m][j];
            val = acc;
        }
        __syncthreads();
        if (tid < i) s_A[i][tid] = val;
        __syncthreads();
    }

    // Pack T into XOR-swizzled BF16 s_T[64][64] AND scale s_kv in-place by beta*exp(G)
    for (int idx = tid; idx < 64 * 8; idx += 128) {
        int r = idx >> 3;
        int ch = idx & 7;
        int c8 = ch << 3;
        uint4 tv;
        __nv_bfloat162* tp = reinterpret_cast<__nv_bfloat162*>(&tv);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            tp[j] = __floats2bfloat162_rn(s_A[r][c8 + 2 * j], s_A[r][c8 + 2 * j + 1]);
        }
        *reinterpret_cast<uint4*>(&s_T[r][gdn_swz64(r, ch)]) = tv;
    }
    for (int idx = tid; idx < 64 * 16; idx += 128) {
        int r = idx >> 4;
        int ch = idx & 15;
        float sc = (r < C) ? (s_beta[r] * expf(s_g[r])) : 0.0f;
        uint4 kv = *reinterpret_cast<const uint4*>(&s_kv[r][gdn_swz128(r, ch)]);
        __nv_bfloat162* kp = reinterpret_cast<__nv_bfloat162*>(&kv);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __bfloat1622float2(kp[j]);
            kp[j] = __floats2bfloat162_rn(f.x * sc, f.y * sc);
        }
        *reinterpret_cast<uint4*>(&s_kv[r][gdn_swz128(r, ch)]) = kv;
    }
    __syncthreads();

    // Step 5a: Tensor Core w = s_T[64, 64] @ s_kv[64, 128]
    float acc128[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) acc128[nt][0] = acc128[nt][1] = acc128[nt][2] = acc128[nt][3] = 0.0f;
    tc_gemm_16x128x64_nn_acc(s_T, wid * 16, s_kv, acc128, lane);
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        int c0 = nt * 8 + 2 * t;
        if (r0 < C) {
            __nv_bfloat16* dst = w_out + ((size_t)(bos + t0 + r0) * 48 + hv) * 128 + c0;
            *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(acc128[nt][0], acc128[nt][1]);
        }
        if (r1 < C) {
            __nv_bfloat16* dst = w_out + ((size_t)(bos + t0 + r1) * 48 + hv) * 128 + c0;
            *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(acc128[nt][2], acc128[nt][3]);
        }
    }
    __syncthreads();

    // Step 5b: Load beta * v into s_kv and Tensor Core u = s_T[64, 64] @ s_kv[64, 128]
    for (int idx = tid; idx < 64 * 16; idx += 128) {
        int r = idx >> 4;
        int ch = idx & 15;
        int c8 = ch << 3;
        uint4 vv = make_uint4(0, 0, 0, 0);
        if (r < C) {
            int tok = bos + t0 + r;
            float sc = s_beta[r];
            vv = *reinterpret_cast<const uint4*>(conv_out + (size_t)tok * Qwen35Config::GDN_CONV_DIM + 4096 + hv * 128 + c8);
            __nv_bfloat162* vp = reinterpret_cast<__nv_bfloat162*>(&vv);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f = __bfloat1622float2(vp[j]);
                vp[j] = __floats2bfloat162_rn(f.x * sc, f.y * sc);
            }
        }
        *reinterpret_cast<uint4*>(&s_kv[r][gdn_swz128(r, ch)]) = vv;
    }
    __syncthreads();

    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) acc128[nt][0] = acc128[nt][1] = acc128[nt][2] = acc128[nt][3] = 0.0f;
    tc_gemm_16x128x64_nn_acc(s_T, wid * 16, s_kv, acc128, lane);
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        int c0 = nt * 8 + 2 * t;
        if (r0 < C) {
            __nv_bfloat16* dst = u_out + ((size_t)(bos + t0 + r0) * 48 + hv) * 128 + c0;
            *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(acc128[nt][0], acc128[nt][1]);
        }
        if (r1 < C) {
            __nv_bfloat16* dst = u_out + ((size_t)(bos + t0 + r1) * 48 + hv) * 128 + c0;
            *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(acc128[nt][2], acc128[nt][3]);
        }
    }
}

// Kernel 6b (Tensor Core): Inter-chunk state recurrence — ZERO stack frame, 100% Tensor Core!
// Grid: (96, B) -> 48 V-heads x 2 halves of V (64 V-rows per CTA), Block: 128 threads (4 warps x 16 V-rows!)
// State H[64, 128] stays in FP32 registers h_acc[16][4] across all chunks!
__global__ void gdn_chunk_inter_state_kernel(
    const __nv_bfloat16* __restrict__ k_norm,    // [total_T, 16, 128]
    const __nv_bfloat16* __restrict__ w_in,      // [total_T, 48, 128]
    const __nv_bfloat16* __restrict__ u_in,      // [total_T, 48, 128]
    const float* __restrict__ g_cum,             // [total_T, 48]
    __nv_bfloat16* __restrict__ ssm_states,      // [num_slots, 48, 128, 128]
    const int32_t* __restrict__ state_slots,     // [B]
    const int32_t* __restrict__ cu_seqlens,      // [B + 1]
    const int32_t* __restrict__ cu_chunks,       // [B + 1]
    __nv_bfloat16* __restrict__ h_chunks_out,    // [total_chunks, 48, 128, 128]
    __nv_bfloat16* __restrict__ v_new_out,       // [total_T, 48, 128]
    int B
) {
    int hv = blockIdx.x >> 1;
    int v_half = blockIdx.x & 1;
    int v_base = v_half * 64;
    int b = blockIdx.y;
    if (b >= B) return;

    int bos = cu_seqlens[b];
    int eos = cu_seqlens[b + 1];
    int seq_len = eos - bos;
    if (seq_len <= 0) return;

    int chunk_base = cu_chunks[b];
    int num_chunks = cu_chunks[b + 1] - chunk_base;
    int hk = hv / 3;
    int tid = threadIdx.x; // 0..127
    int wid = tid >> 5;    // 0..3 -> owns V-rows [wid*16 .. wid*16+15]
    int lane = tid & 31;
    int g = lane >> 2;
    int t = lane & 3;
    int r0 = wid * 16 + g, r1 = r0 + 8;

    int slot = state_slots[b];
    __nv_bfloat16* st_base = ssm_states + (((size_t)slot * 48 + hv) * 128 + v_base) * 128;

    __shared__ __nv_bfloat16 sH[64][128];  // 16 KB (swizzled gdn_swz128)
    __shared__ __nv_bfloat16 sW[64][128];  // 16 KB (swizzled gdn_swz128; holds W, then K_scaled)
    __shared__ __nv_bfloat16 sVN[64][64];  // 8 KB  (swizzled gdn_swz64; V_new^T [64 V, 64 C])
    __shared__ float sG[GDN_CHUNK];

    // Load initial H[64, 128] coalesced into sH, then into FP32 registers h_acc[16][4]!
    for (int idx = tid; idx < 64 * 16; idx += 128) {
        int r = idx >> 4, ch = idx & 15;
        *reinterpret_cast<uint4*>(&sH[r][gdn_swz128(r, ch)]) =
            *reinterpret_cast<const uint4*>(st_base + (size_t)r * 128 + (ch << 3));
    }
    __syncthreads();

    float h_acc[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        int col0 = gdn_swz128(r0, nt) + 2 * t;
        int col1 = gdn_swz128(r1, nt) + 2 * t;
        float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sH[r0][col0]));
        float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sH[r1][col1]));
        h_acc[nt][0] = f0.x; h_acc[nt][1] = f0.y;
        h_acc[nt][2] = f1.x; h_acc[nt][3] = f1.y;
    }

    for (int c = 0; c < num_chunks; ++c) {
        int t0 = c * GDN_CHUNK;
        int C = min(GDN_CHUNK, seq_len - t0);

        // 1) Store current H into sH (for Tensor Core H @ W^T) AND into h_chunks_out[chunk_base + c]!
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            *reinterpret_cast<__nv_bfloat162*>(&sH[r0][gdn_swz128(r0, nt) + 2 * t]) = __floats2bfloat162_rn(h_acc[nt][0], h_acc[nt][1]);
            *reinterpret_cast<__nv_bfloat162*>(&sH[r1][gdn_swz128(r1, nt) + 2 * t]) = __floats2bfloat162_rn(h_acc[nt][2], h_acc[nt][3]);
        }
        // Load W[C, 128] into sW and g_cum into sG
        if (tid < C) sG[tid] = g_cum[(size_t)(bos + t0 + tid) * 48 + hv];
        for (int idx = tid; idx < 64 * 16; idx += 128) {
            int r = idx >> 4, ch = idx & 15;
            uint4 wv = (r < C)
                ? *reinterpret_cast<const uint4*>(w_in + ((size_t)(bos + t0 + r) * 48 + hv) * 128 + (ch << 3))
                : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&sW[r][gdn_swz128(r, ch)]) = wv;
        }
        __syncthreads();

        // Coalesced write of sH -> h_chunks_out[chunk_base + c]
        __nv_bfloat16* hc_base = h_chunks_out + ((((size_t)(chunk_base + c) * 48 + hv) * 128 + v_base) * 128);
        for (int idx = tid; idx < 64 * 16; idx += 128) {
            int r = idx >> 4, ch = idx & 15;
            *reinterpret_cast<uint4*>(hc_base + (size_t)r * 128 + (ch << 3)) =
                *reinterpret_cast<const uint4*>(&sH[r][gdn_swz128(r, ch)]);
        }

        // 2) Tensor Core HW[64 V, 64 C] = sH[64, 128] @ sW[64, 128]^T!
        float hw[8][4];
        tc_gemm_16x64x128_nt(sH, wid * 16, sW, hw, lane);

        // 3) V_new[c, v] = U[c, v] - HW[v, c] -> write to v_new_out + swizzled sVN[64 V, 64 C]!
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            int c0 = nt * 8 + 2 * t, c1 = c0 + 1;
            float vn00 = 0.0f, vn01 = 0.0f, vn10 = 0.0f, vn11 = 0.0f;
            if (c0 < C) {
                vn00 = __bfloat162float(u_in[((size_t)(bos + t0 + c0) * 48 + hv) * 128 + v_base + r0]) - hw[nt][0];
                vn10 = __bfloat162float(u_in[((size_t)(bos + t0 + c0) * 48 + hv) * 128 + v_base + r1]) - hw[nt][2];
                v_new_out[((size_t)(bos + t0 + c0) * 48 + hv) * 128 + v_base + r0] = __float2bfloat16(vn00);
                v_new_out[((size_t)(bos + t0 + c0) * 48 + hv) * 128 + v_base + r1] = __float2bfloat16(vn10);
            }
            if (c1 < C) {
                vn01 = __bfloat162float(u_in[((size_t)(bos + t0 + c1) * 48 + hv) * 128 + v_base + r0]) - hw[nt][1];
                vn11 = __bfloat162float(u_in[((size_t)(bos + t0 + c1) * 48 + hv) * 128 + v_base + r1]) - hw[nt][3];
                v_new_out[((size_t)(bos + t0 + c1) * 48 + hv) * 128 + v_base + r0] = __float2bfloat16(vn01);
                v_new_out[((size_t)(bos + t0 + c1) * 48 + hv) * 128 + v_base + r1] = __float2bfloat16(vn11);
            }
            *reinterpret_cast<__nv_bfloat162*>(&sVN[r0][gdn_swz64(r0, nt) + 2 * t]) = __floats2bfloat162_rn(vn00, vn01);
            *reinterpret_cast<__nv_bfloat162*>(&sVN[r1][gdn_swz64(r1, nt) + 2 * t]) = __floats2bfloat162_rn(vn10, vn11);
        }
        __syncthreads();

        // 4) Load k_scaled[C, 128] = k_norm * exp(G_last - G[r]) into sW
        float g_last = sG[C - 1];
        for (int idx = tid; idx < 64 * 16; idx += 128) {
            int r = idx >> 4, ch = idx & 15;
            uint4 kv = make_uint4(0, 0, 0, 0);
            if (r < C) {
                float sc = expf(fminf(g_last - sG[r], 0.0f));
                kv = *reinterpret_cast<const uint4*>(k_norm + ((size_t)(bos + t0 + r) * 16 + hk) * 128 + (ch << 3));
                __nv_bfloat162* kp = reinterpret_cast<__nv_bfloat162*>(&kv);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 f = __bfloat1622float2(kp[j]);
                    kp[j] = __floats2bfloat162_rn(f.x * sc, f.y * sc);
                }
            }
            *reinterpret_cast<uint4*>(&sW[r][gdn_swz128(r, ch)]) = kv;
        }
        __syncthreads();

        // 5) Decay H and Tensor Core update: H[64, 128] += sVN[64, 64] @ sW[64, 128]!
        float decay_last = expf(g_last);
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            h_acc[nt][0] *= decay_last; h_acc[nt][1] *= decay_last;
            h_acc[nt][2] *= decay_last; h_acc[nt][3] *= decay_last;
        }
        tc_gemm_16x128x64_nn_acc(sVN, wid * 16, sW, h_acc, lane);
        __syncthreads();
    }

    // Store final H back to ssm_states (via sH for 100% coalesced uint4 global writes!)
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        *reinterpret_cast<__nv_bfloat162*>(&sH[r0][gdn_swz128(r0, nt) + 2 * t]) = __floats2bfloat162_rn(h_acc[nt][0], h_acc[nt][1]);
        *reinterpret_cast<__nv_bfloat162*>(&sH[r1][gdn_swz128(r1, nt) + 2 * t]) = __floats2bfloat162_rn(h_acc[nt][2], h_acc[nt][3]);
    }
    __syncthreads();
    for (int idx = tid; idx < 64 * 16; idx += 128) {
        int r = idx >> 4, ch = idx & 15;
        *reinterpret_cast<uint4*>(st_base + (size_t)r * 128 + (ch << 3)) =
            *reinterpret_cast<const uint4*>(&sH[r][gdn_swz128(r, ch)]);
    }
}

// Kernel 6c (Tensor Core): Chunk output o = (exp(G)*q) @ h_chunk^T + tril((q@k^T)*exp(G_i-G_j)) @ v_new!
// Coalesced single-read of h_chunk + 100% Tensor Core -> ~60x faster (51.2 ms -> ~0.8 ms)!
// Grid: (max_chunks_per_seq, 48, B), Block: 128 threads (4 warps x 16 tokens!)
__global__ void gdn_chunk_output_kernel(
    const __nv_bfloat16* __restrict__ q_norm,
    const __nv_bfloat16* __restrict__ k_norm,
    const __nv_bfloat16* __restrict__ v_new,
    const __nv_bfloat16* __restrict__ h_chunks,
    const float* __restrict__ g_cum,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ cu_chunks,
    __nv_bfloat16* __restrict__ out,
    int B
) {
    int c_idx = blockIdx.x;
    int hv = blockIdx.y;
    int b = blockIdx.z;
    if (b >= B) return;

    int bos = cu_seqlens[b];
    int eos = cu_seqlens[b + 1];
    int seq_len = eos - bos;
    int t0 = c_idx * GDN_CHUNK;
    if (t0 >= seq_len) return;
    int C = min(GDN_CHUNK, seq_len - t0);
    int chunk_id = cu_chunks[b] + c_idx;
    int hk = hv / 3;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int g = lane >> 2;
    int t = lane & 3;
    int r0 = wid * 16 + g, r1 = r0 + 8;

    __shared__ __nv_bfloat16 sQ[64][128];  // 16 KB (swizzled gdn_swz128)
    __shared__ __nv_bfloat16 sKV[64][128]; // 16 KB (swizzled gdn_swz128: K -> V_new -> H_chunk tiles!)
    __shared__ __nv_bfloat16 sS[64][64];   // 8 KB  (swizzled gdn_swz64)
    __shared__ float sG[GDN_CHUNK];

    if (tid < C) sG[tid] = g_cum[(size_t)(bos + t0 + tid) * 48 + hv];
    else if (tid < GDN_CHUNK) sG[tid] = 0.0f;

    // Load Q and K into swizzled smem
    for (int idx = tid; idx < 64 * 16; idx += 128) {
        int r = idx >> 4, ch = idx & 15;
        uint4 qv = make_uint4(0, 0, 0, 0), kv = make_uint4(0, 0, 0, 0);
        if (r < C) {
            size_t off = ((size_t)(bos + t0 + r) * 16 + hk) * 128 + (ch << 3);
            qv = *reinterpret_cast<const uint4*>(q_norm + off);
            kv = *reinterpret_cast<const uint4*>(k_norm + off);
        }
        *reinterpret_cast<uint4*>(&sQ[r][gdn_swz128(r, ch)]) = qv;
        *reinterpret_cast<uint4*>(&sKV[r][gdn_swz128(r, ch)]) = kv;
    }
    __syncthreads();

    // 1) Tensor Core S[64, 64] = tril((Q @ K^T) * exp(G_i - G_j)) -> sS
    float qk[8][4];
    tc_gemm_16x64x128_nt(sQ, wid * 16, sKV, qk, lane);
    float g0 = sG[r0], g1 = sG[r1];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        int c0 = nt * 8 + 2 * t, c1 = c0 + 1;
        float v00 = (r0 < C && c0 <= r0) ? (qk[nt][0] * expf(fminf(g0 - sG[c0], 0.0f))) : 0.0f;
        float v01 = (r0 < C && c1 <= r0) ? (qk[nt][1] * expf(fminf(g0 - sG[c1], 0.0f))) : 0.0f;
        float v10 = (r1 < C && c0 <= r1) ? (qk[nt][2] * expf(fminf(g1 - sG[c0], 0.0f))) : 0.0f;
        float v11 = (r1 < C && c1 <= r1) ? (qk[nt][3] * expf(fminf(g1 - sG[c1], 0.0f))) : 0.0f;
        *reinterpret_cast<__nv_bfloat162*>(&sS[r0][gdn_swz64(r0, nt) + 2 * t]) = __floats2bfloat162_rn(v00, v01);
        *reinterpret_cast<__nv_bfloat162*>(&sS[r1][gdn_swz64(r1, nt) + 2 * t]) = __floats2bfloat162_rn(v10, v11);
    }
    __syncthreads();

    // 2) Load V_new[64, 128] into sKV and compute intra-chunk O = sS @ V_new!
    for (int idx = tid; idx < 64 * 16; idx += 128) {
        int r = idx >> 4, ch = idx & 15;
        uint4 vv = (r < C)
            ? *reinterpret_cast<const uint4*>(v_new + ((size_t)(bos + t0 + r) * 48 + hv) * 128 + (ch << 3))
            : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(&sKV[r][gdn_swz128(r, ch)]) = vv;
    }
    __syncthreads();

    float o_acc[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) o_acc[nt][0] = o_acc[nt][1] = o_acc[nt][2] = o_acc[nt][3] = 0.0f;
    tc_gemm_16x128x64_nn_acc(sS, wid * 16, sKV, o_acc, lane);
    __syncthreads();

    // 3) Scale sQ[r] by exp(sG[r]) in-place, then accumulate inter-chunk O += sQ @ H_chunk^T
    //    in 2 tiles of 64 V-rows (reads entire H_chunk [128, 128] ONCE, 100% coalesced uint4!)
    for (int idx = tid; idx < 64 * 16; idx += 128) {
        int r = idx >> 4, ch = idx & 15;
        float sc = (r < C) ? expf(sG[r]) : 0.0f;
        uint4 qv = *reinterpret_cast<const uint4*>(&sQ[r][gdn_swz128(r, ch)]);
        __nv_bfloat162* qp = reinterpret_cast<__nv_bfloat162*>(&qv);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __bfloat1622float2(qp[j]);
            qp[j] = __floats2bfloat162_rn(f.x * sc, f.y * sc);
        }
        *reinterpret_cast<uint4*>(&sQ[r][gdn_swz128(r, ch)]) = qv;
    }

    const __nv_bfloat16* h_base = h_chunks + ((size_t)chunk_id * 48 + hv) * 128 * 128;
    for (int v_half = 0; v_half < 2; ++v_half) {
        const __nv_bfloat16* hp = h_base + (size_t)(v_half * 64) * 128;
        for (int idx = tid; idx < 64 * 16; idx += 128) {
            int r = idx >> 4, ch = idx & 15;
            *reinterpret_cast<uint4*>(&sKV[r][gdn_swz128(r, ch)]) =
                *reinterpret_cast<const uint4*>(hp + (size_t)r * 128 + (ch << 3));
        }
        __syncthreads();

        float inter[8][4];
        tc_gemm_16x64x128_nt(sQ, wid * 16, sKV, inter, lane);
        int nt_off = v_half * 8;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            o_acc[nt_off + nt][0] += inter[nt][0];
            o_acc[nt_off + nt][1] += inter[nt][1];
            o_acc[nt_off + nt][2] += inter[nt][2];
            o_acc[nt_off + nt][3] += inter[nt][3];
        }
        __syncthreads();
    }

    // Write output [C, 128]
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        int c0 = nt * 8 + 2 * t;
        if (r0 < C) {
            __nv_bfloat16* dst = out + ((size_t)(bos + t0 + r0) * 48 + hv) * 128 + c0;
            *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(o_acc[nt][0], o_acc[nt][1]);
        }
        if (r1 < C) {
            __nv_bfloat16* dst = out + ((size_t)(bos + t0 + r1) * 48 + hv) * 128 + c0;
            *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(o_acc[nt][2], o_acc[nt][3]);
        }
    }
}

} // namespace

void launch_gdn_conv1d_prefill(
    const __nv_bfloat16* qkvz,
    const __nv_bfloat16* conv_w,
    __nv_bfloat16* conv_states,
    const int32_t* state_slots,
    const int32_t* cu_seqlens,
    __nv_bfloat16* conv_out,
    int B,
    int total_T,
    cudaStream_t stream
) {
    if (B == 0 || total_T == 0) return;
    dim3 grid((Qwen35Config::GDN_CONV_DIM + 255) / 256, B);
    gdn_conv1d_prefill_kernel<<<grid, 256, 0, stream>>>(
        qkvz, conv_w, conv_states, state_slots, cu_seqlens, conv_out, B);
}

void launch_gdn_conv1d_decode_verify(
    const __nv_bfloat16* qkvz,
    const __nv_bfloat16* conv_w,
    __nv_bfloat16* conv_states,
    __nv_bfloat16* conv_inter,
    const int32_t* state_slots,
    __nv_bfloat16* conv_out,
    int B,
    int q_len,
    bool commit_inplace,
    cudaStream_t stream
) {
    if (B == 0 || q_len == 0) return;
    dim3 grid((Qwen35Config::GDN_CONV_DIM + 255) / 256, B);
    gdn_conv1d_decode_verify_kernel<<<grid, 256, 0, stream>>>(
        qkvz, conv_w, conv_states, conv_inter, state_slots, conv_out, B, q_len, commit_inplace);
}

void launch_gdn_recurrent_decode_verify(
    const __nv_bfloat16* conv_out,
    const __nv_bfloat16* ba,
    const float* A_log,
    const float* dt_bias,
    __nv_bfloat16* ssm_states,
    __nv_bfloat16* ssm_inter,
    const int32_t* state_slots,
    __nv_bfloat16* out,
    int B,
    int q_len,
    bool commit_inplace,
    cudaStream_t stream
) {
    if (B == 0 || q_len == 0) return;
    dim3 grid(192, B);
    gdn_recurrent_decode_verify_kernel<<<grid, 128, 0, stream>>>(
        conv_out, ba, A_log, dt_bias, ssm_states, ssm_inter, state_slots, out, B, q_len, commit_inplace);
}

void launch_gdn_commit_replay_states(
    __nv_bfloat16* const* ssm_states_ptrs,
    __nv_bfloat16* const* conv_states_ptrs,
    __nv_bfloat16* const* conv_inter_ptrs,
    const __nv_bfloat16* const* conv_out_ptrs,
    const __nv_bfloat16* const* ba_ptrs,
    const float* const* A_log_ptrs,
    const float* const* dt_bias_ptrs,
    const int32_t* state_slots,
    const int32_t* commit_lens,
    int B,
    int q_len,
    cudaStream_t stream
) {
    if (B == 0) return;
    dim3 g_ssm(96, Qwen35Config::NUM_GDN_LAYERS, B);
    gdn_commit_replay_ssm_kernel<<<g_ssm, 256, 0, stream>>>(
        ssm_states_ptrs, conv_out_ptrs, ba_ptrs, A_log_ptrs, dt_bias_ptrs,
        state_slots, commit_lens, B, q_len);
    dim3 g_conv(Qwen35Config::NUM_GDN_LAYERS, B);
    gdn_commit_conv_states_kernel<<<g_conv, 256, 0, stream>>>(
        conv_states_ptrs, conv_inter_ptrs, state_slots, commit_lens, B, q_len);
}

void launch_gdn_chunked_prefill(
    const __nv_bfloat16* conv_out,
    const __nv_bfloat16* ba,
    const float* A_log,
    const float* dt_bias,
    __nv_bfloat16* ssm_states,
    const int32_t* state_slots,
    const int32_t* cu_seqlens,
    __nv_bfloat16* out,
    GdnPrefillWorkspace& ws,
    int B,
    int total_T,
    cudaStream_t stream
) {
    if (B == 0 || total_T == 0) return;

    // For small prefill (e.g. single seq <= 64 tokens), recurrent kernel is ultra-fast & zero extra launch!
    // Wait: let's compute cu_chunks on host? We have cu_seqlens on device!
    // For single sequence (common prefill case) or varlen, compute max_chunks conservatively = ceil(total_T / 64) + B.
    // Let's keep a tiny device cu_chunks helper kernel so there is ZERO host sync!
    static int32_t* d_cu_chunks = nullptr;
    if (!d_cu_chunks) {
        CUDA_CHECK(cudaMalloc(&d_cu_chunks, 64 * sizeof(int32_t)));
    }
    compute_cu_chunks_kernel<<<1, 1, 0, stream>>>(cu_seqlens, d_cu_chunks, B);
    int max_chunks = (total_T + GDN_CHUNK - 1) / GDN_CHUNK;
    dim3 g6a(max_chunks, 48, B);
    gdn_chunk_intra_prep_compact_kernel<<<g6a, 128, 0, stream>>>(
        conv_out, ba, A_log, dt_bias, cu_seqlens,
        ws.g_cum, ws.q_norm, ws.k_norm, ws.w, ws.u, B);

    dim3 g6b(96, B);
    gdn_chunk_inter_state_kernel<<<g6b, 128, 0, stream>>>(
        ws.k_norm, ws.w, ws.u, ws.g_cum,
        ssm_states, state_slots, cu_seqlens, d_cu_chunks,
        ws.h_chunks, ws.v_new, B);

    dim3 g6c(max_chunks, 48, B);
    gdn_chunk_output_kernel<<<g6c, 128, 0, stream>>>(
        ws.q_norm, ws.k_norm, ws.v_new, ws.h_chunks, ws.g_cum,
        cu_seqlens, d_cu_chunks, out, B);
}

} // namespace amperegen
