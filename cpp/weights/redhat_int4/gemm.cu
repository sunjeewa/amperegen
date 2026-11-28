#include "weights/weight_backend.hpp"
#include "runtime/cuda_utils.cuh"
#include <cmath>

namespace amperegen {
namespace redhat_int4 {

namespace {

// ============================================================================
// Exact BF16x2 arithmetic helpers for Marlin-style bitwise INT4 -> BF16 unpack
// ============================================================================
__device__ __forceinline__ uint32_t bf16x2_sub_u32(uint32_t a, uint32_t b) {
    __nv_bfloat162 ha = *reinterpret_cast<const __nv_bfloat162*>(&a);
    __nv_bfloat162 hb = *reinterpret_cast<const __nv_bfloat162*>(&b);
    __nv_bfloat162 hc = __hsub2(ha, hb);
    return *reinterpret_cast<const uint32_t*>(&hc);
}

__device__ __forceinline__ uint32_t bf16x2_mul_u32(uint32_t a, uint32_t b) {
    __nv_bfloat162 ha = *reinterpret_cast<const __nv_bfloat162*>(&a);
    __nv_bfloat162 hb = *reinterpret_cast<const __nv_bfloat162*>(&b);
    __nv_bfloat162 hc = __hmul2(ha, hb);
    return *reinterpret_cast<const uint32_t*>(&hc);
}

__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Fast unscaled unpack: ONE uint32 -> 4 exact signed integer BF16x2 registers [-8..+7].
__device__ __forceinline__ void unpack_int4x8_s4_bf16x2x4(
    uint32_t w,
    uint32_t& a0,
    uint32_t& a1,
    uint32_t& a2,
    uint32_t& a3
) {
    constexpr uint32_t MAGIC_4300 = 0x43004300u; // bf16x2(128.0, 128.0)
    constexpr uint32_t SUB_136    = 0x43084308u; // bf16x2(136.0, 136.0)
    constexpr uint32_t MASK_0F    = 0x000F000Fu;

    uint32_t r0 = lop3_and_or(w,       MASK_0F, MAGIC_4300);
    uint32_t r1 = lop3_and_or(w >> 4,  MASK_0F, MAGIC_4300);
    uint32_t r2 = lop3_and_or(w >> 8,  MASK_0F, MAGIC_4300);
    uint32_t r3 = lop3_and_or(w >> 12, MASK_0F, MAGIC_4300);

    a0 = bf16x2_sub_u32(r0, SUB_136);
    a1 = bf16x2_sub_u32(r1, SUB_136);
    a2 = bf16x2_sub_u32(r2, SUB_136);
    a3 = bf16x2_sub_u32(r3, SUB_136);
}

__device__ __forceinline__ void unpack_int4x8_to_bf16x2x4(
    uint32_t w,
    uint32_t sc_lo_u32,
    uint32_t sc_hi_u32,
    uint32_t& a0,
    uint32_t& a1,
    uint32_t& a2,
    uint32_t& a3
) {
    unpack_int4x8_s4_bf16x2x4(w, a0, a1, a2, a3);
    a0 = bf16x2_mul_u32(a0, sc_lo_u32);
    a1 = bf16x2_mul_u32(a1, sc_hi_u32);
    a2 = bf16x2_mul_u32(a2, sc_lo_u32);
    a3 = bf16x2_mul_u32(a3, sc_hi_u32);
}

// Offline weight packer: compressed-tensors I32 [N, K/8] -> warp-MMA [N/16, K/64, 32] uint4
__global__ void pack_int4_mma_kernel(
    const uint32_t* __restrict__ W_i32,
    uint4* __restrict__ W_packed,
    int N, int K
) {
    int k64 = blockIdx.x;
    int n_tile = blockIdx.y;
    int lane = threadIdx.x;
    int g = lane >> 2;
    int t = lane & 3;

    int n_lo = n_tile * 16 + g;
    int n_hi = n_lo + 8;
    int k_words = K >> 3;
    const uint32_t* row_lo = W_i32 + (size_t)n_lo * k_words;
    const uint32_t* row_hi = W_i32 + (size_t)n_hi * k_words;

    auto pack8 = [&](int k16_word_base) -> uint32_t {
        uint32_t w_lo0 = row_lo[k16_word_base + 0];
        uint32_t w_lo1 = row_lo[k16_word_base + 1];
        uint32_t w_hi0 = row_hi[k16_word_base + 0];
        uint32_t w_hi1 = row_hi[k16_word_base + 1];

        int sh = t * 8;
        uint32_t a0_lo = (w_lo0 >> (sh + 0)) & 0xFu;
        uint32_t a0_hi = (w_lo0 >> (sh + 4)) & 0xFu;
        uint32_t a1_lo = (w_hi0 >> (sh + 0)) & 0xFu;
        uint32_t a1_hi = (w_hi0 >> (sh + 4)) & 0xFu;
        uint32_t a2_lo = (w_lo1 >> (sh + 0)) & 0xFu;
        uint32_t a2_hi = (w_lo1 >> (sh + 4)) & 0xFu;
        uint32_t a3_lo = (w_hi1 >> (sh + 0)) & 0xFu;
        uint32_t a3_hi = (w_hi1 >> (sh + 4)) & 0xFu;

        return (a0_lo << 0)  | (a1_lo << 4)  | (a2_lo << 8)  | (a3_lo << 12) |
               (a0_hi << 16) | (a1_hi << 20) | (a2_hi << 24) | (a3_hi << 28);
    };

    int kw0 = k64 * 8;
    uint4 out;
    out.x = pack8(kw0 + 0);
    out.y = pack8(kw0 + 2);
    out.z = pack8(kw0 + 4);
    out.w = pack8(kw0 + 6);

    W_packed[((size_t)n_tile * (K >> 6) + k64) * 32 + lane] = out;
}

// Offline scale packer: BF16 [N, K/128] -> uint32 [N/16, K/128, 8]
__global__ void pack_int4_scales_kernel(
    const __nv_bfloat16* __restrict__ S_bf16,
    uint32_t* __restrict__ S_packed,
    int N, int K
) {
    int kb = blockIdx.x;
    int n_tile = blockIdx.y;
    int g = threadIdx.x;
    int total_kb = K >> 7;
    int n_lo = n_tile * 16 + g;
    int n_hi = n_lo + 8;

    uint16_t lo = *reinterpret_cast<const uint16_t*>(&S_bf16[(size_t)n_lo * total_kb + kb]);
    uint16_t hi = *reinterpret_cast<const uint16_t*>(&S_bf16[(size_t)n_hi * total_kb + kb]);
    S_packed[((size_t)n_tile * total_kb + kb) * 8 + g] = (uint32_t)lo | ((uint32_t)hi << 16);
}

// Fast prefill unpack: warp-MMA INT4 -> row-major BF16 [N, K]
__global__ void unpack_dequant_int4_mma_to_bf16_kernel(
    const uint4* __restrict__ W_packed,
    const uint32_t* __restrict__ S_packed,
    __nv_bfloat16* __restrict__ W_bf16,
    int N, int K
) {
    int kb = blockIdx.x;
    int n_tile = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int g = lane >> 2;
    int t = lane & 3;

    int total_kb = K >> 7;
    int total_k64 = K >> 6;
    int k64 = (kb << 1) + wid;

    uint32_t sc_pair = __ldg(&S_packed[((size_t)n_tile * total_kb + kb) * 8 + g]);
    uint32_t sc_lo_u32 = __byte_perm(sc_pair, sc_pair, 0x1010);
    uint32_t sc_hi_u32 = __byte_perm(sc_pair, sc_pair, 0x3232);

    uint4 w = __ldcg(W_packed + ((size_t)n_tile * total_k64 + k64) * 32 + lane);

    __shared__ __nv_bfloat16 sTile[16][128];

    int k_local = wid * 64;
    auto write_k16 = [&](uint32_t w32, int kk) {
        uint32_t a0, a1, a2, a3;
        unpack_int4x8_to_bf16x2x4(w32, sc_lo_u32, sc_hi_u32, a0, a1, a2, a3);
        *reinterpret_cast<uint32_t*>(&sTile[g][kk + 2 * t])         = a0;
        *reinterpret_cast<uint32_t*>(&sTile[g + 8][kk + 2 * t])     = a1;
        *reinterpret_cast<uint32_t*>(&sTile[g][kk + 8 + 2 * t])     = a2;
        *reinterpret_cast<uint32_t*>(&sTile[g + 8][kk + 8 + 2 * t]) = a3;
    };
    write_k16(w.x, k_local + 0);
    write_k16(w.y, k_local + 16);
    write_k16(w.z, k_local + 32);
    write_k16(w.w, k_local + 48);
    __syncthreads();

    int n0 = n_tile * 16;
    int k0 = kb * 128;
    #pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        int idx = pass * 64 + tid;
        int r = idx >> 4;
        int c8 = (idx & 15) << 3;
        *reinterpret_cast<uint4*>(&W_bf16[(size_t)(n0 + r) * K + (k0 + c8)]) =
            *reinterpret_cast<const uint4*>(&sTile[r][c8]);
    }
}

// K1-INT4 v3: 3-stage cp.async sX + hoisted group-128 FP32 scaling (M <= 64)
template <int WARPS_N, int M_TILES>
__global__ void __launch_bounds__(WARPS_N * 32)
int4_group128_tc_gemm_v3_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint4* __restrict__ W_packed,
    const uint32_t* __restrict__ S_packed,
    __nv_bfloat16* __restrict__ Y,
    float* __restrict__ Y_split,
    int M, int N, int K, int split_k
) {
    constexpr int BN = WARPS_N * 32;
    int cta_n = blockIdx.x * BN;
    if (cta_n >= N) return;

    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n_tile0 = (cta_n >> 4) + warp_id * 2;
    int n_tile1 = n_tile0 + 1;
    int n0 = n_tile0 << 4;
    int n1 = n_tile1 << 4;

    int total_kb = K >> 7;
    int kb_per = (total_kb + split_k - 1) / split_k;
    int kb_start = blockIdx.y * kb_per;
    int kb_end = min(total_kb, kb_start + kb_per);

    float c0[2][M_TILES], c1[2][M_TILES], c2[2][M_TILES], c3[2][M_TILES];
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt)
            c0[nt][mt] = c1[nt][mt] = c2[nt][mt] = c3[nt][mt] = 0.0f;

    __shared__ __nv_bfloat16 sX[3][M_TILES * 8][128];

    int g = lane >> 2;
    int t = lane & 3;
    int total_k64 = K >> 6;
    const uint4* w0_base = W_packed + (size_t)n_tile0 * total_k64 * 32 + lane;
    const uint4* w1_base = W_packed + (size_t)n_tile1 * total_k64 * 32 + lane;
    const uint32_t* s0_base = S_packed + (size_t)n_tile0 * total_kb * 8 + g;
    const uint32_t* s1_base = S_packed + (size_t)n_tile1 * total_kb * 8 + g;

    if (kb_start < kb_end) {
        auto load_sX_stage = [&](int kb, int st) {
            int k_base = kb << 7;
            constexpr int TOTAL_VEC = (M_TILES * 8) * 16;
            #pragma unroll
            for (int idx = threadIdx.x; idx < TOTAL_VEC; idx += WARPS_N * 32) {
                int m = idx >> 4;
                int kv = (idx & 15) << 3;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sX[st][m][kv]));
                int src_bytes = (m < M) ? 16 : 0;
                const void* gmem = (m < M) ? (X + (size_t)m * K + k_base + kv) : X;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                             :: "r"(smem_addr), "l"(gmem), "r"(src_bytes));
            }
            asm volatile("cp.async.commit_group;\n");
        };

        load_sX_stage(kb_start, 0);
        if (kb_start + 1 < kb_end) {
            load_sX_stage(kb_start + 1, 1);
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

        int s_cur = 0;
        for (int kb = kb_start; kb < kb_end; ++kb) {
            if (kb + 2 < kb_end) {
                int s_ld = (s_cur == 0) ? 2 : (s_cur - 1);
                load_sX_stage(kb + 2, s_ld);
            }

            uint32_t sp0 = __ldg(s0_base + (size_t)kb * 8);
            uint32_t sp1 = __ldg(s1_base + (size_t)kb * 8);

            const uint4* w0_kb = w0_base + (size_t)(kb << 1) * 32;
            const uint4* w1_kb = w1_base + (size_t)(kb << 1) * 32;
            uint4 wa0 = __ldcg(w0_kb + 0);
            uint4 wa1 = __ldcg(w0_kb + 32);
            uint4 wb0 = __ldcg(w1_kb + 0);
            uint4 wb1 = __ldcg(w1_kb + 32);

            if constexpr (M_TILES <= 2) {
                float t0[2][M_TILES], t1[2][M_TILES], t2[2][M_TILES], t3[2][M_TILES];
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt)
                    #pragma unroll
                    for (int mt = 0; mt < M_TILES; ++mt)
                        t0[nt][mt] = t1[nt][mt] = t2[nt][mt] = t3[nt][mt] = 0.0f;

                auto mma_k16_unscaled = [&](uint32_t wa_u32, uint32_t wb_u32, int k_off) {
                    uint32_t b0[M_TILES], b1[M_TILES];
                    #pragma unroll
                    for (int mt = 0; mt < M_TILES; ++mt) {
                        int mr = mt * 8 + g;
                        b0[mt] = *reinterpret_cast<const uint32_t*>(&sX[s_cur][mr][k_off + 2 * t]);
                        b1[mt] = *reinterpret_cast<const uint32_t*>(&sX[s_cur][mr][k_off + 8 + 2 * t]);
                    }
                    uint32_t a0, a1, a2, a3;
                    unpack_int4x8_s4_bf16x2x4(wa_u32, a0, a1, a2, a3);
                    #pragma unroll
                    for (int mt = 0; mt < M_TILES; ++mt) {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                            : "+f"(t0[0][mt]), "+f"(t1[0][mt]), "+f"(t2[0][mt]), "+f"(t3[0][mt])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[mt]), "r"(b1[mt]));
                    }
                    unpack_int4x8_s4_bf16x2x4(wb_u32, a0, a1, a2, a3);
                    #pragma unroll
                    for (int mt = 0; mt < M_TILES; ++mt) {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                            : "+f"(t0[1][mt]), "+f"(t1[1][mt]), "+f"(t2[1][mt]), "+f"(t3[1][mt])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[mt]), "r"(b1[mt]));
                    }
                };

                mma_k16_unscaled(wa0.x, wb0.x, 0);
                mma_k16_unscaled(wa0.y, wb0.y, 16);
                mma_k16_unscaled(wa0.z, wb0.z, 32);
                mma_k16_unscaled(wa0.w, wb0.w, 48);
                mma_k16_unscaled(wa1.x, wb1.x, 64);
                mma_k16_unscaled(wa1.y, wb1.y, 80);
                mma_k16_unscaled(wa1.z, wb1.z, 96);
                mma_k16_unscaled(wa1.w, wb1.w, 112);

                float2 sf0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sp0));
                float2 sf1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sp1));
                #pragma unroll
                for (int mt = 0; mt < M_TILES; ++mt) {
                    c0[0][mt] += t0[0][mt] * sf0.x;
                    c1[0][mt] += t1[0][mt] * sf0.x;
                    c2[0][mt] += t2[0][mt] * sf0.y;
                    c3[0][mt] += t3[0][mt] * sf0.y;
                    c0[1][mt] += t0[1][mt] * sf1.x;
                    c1[1][mt] += t1[1][mt] * sf1.x;
                    c2[1][mt] += t2[1][mt] * sf1.y;
                    c3[1][mt] += t3[1][mt] * sf1.y;
                }
            } else {
                uint32_t sc0_lo = __byte_perm(sp0, sp0, 0x1010);
                uint32_t sc0_hi = __byte_perm(sp0, sp0, 0x3232);
                uint32_t sc1_lo = __byte_perm(sp1, sp1, 0x1010);
                uint32_t sc1_hi = __byte_perm(sp1, sp1, 0x3232);

                auto mma_k16 = [&](uint32_t wa_u32, uint32_t wb_u32, int k_off) {
                    uint32_t b0[M_TILES], b1[M_TILES];
                    #pragma unroll
                    for (int mt = 0; mt < M_TILES; ++mt) {
                        int mr = mt * 8 + g;
                        b0[mt] = *reinterpret_cast<const uint32_t*>(&sX[s_cur][mr][k_off + 2 * t]);
                        b1[mt] = *reinterpret_cast<const uint32_t*>(&sX[s_cur][mr][k_off + 8 + 2 * t]);
                    }
                    uint32_t a0, a1, a2, a3;
                    unpack_int4x8_to_bf16x2x4(wa_u32, sc0_lo, sc0_hi, a0, a1, a2, a3);
                    #pragma unroll
                    for (int mt = 0; mt < M_TILES; ++mt) {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                            : "+f"(c0[0][mt]), "+f"(c1[0][mt]), "+f"(c2[0][mt]), "+f"(c3[0][mt])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[mt]), "r"(b1[mt]));
                    }
                    unpack_int4x8_to_bf16x2x4(wb_u32, sc1_lo, sc1_hi, a0, a1, a2, a3);
                    #pragma unroll
                    for (int mt = 0; mt < M_TILES; ++mt) {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                            : "+f"(c0[1][mt]), "+f"(c1[1][mt]), "+f"(c2[1][mt]), "+f"(c3[1][mt])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[mt]), "r"(b1[mt]));
                    }
                };

                mma_k16(wa0.x, wb0.x, 0);
                mma_k16(wa0.y, wb0.y, 16);
                mma_k16(wa0.z, wb0.z, 32);
                mma_k16(wa0.w, wb0.w, 48);
                mma_k16(wa1.x, wb1.x, 64);
                mma_k16(wa1.y, wb1.y, 80);
                mma_k16(wa1.z, wb1.z, 96);
                mma_k16(wa1.w, wb1.w, 112);
            }

            if (kb + 1 < kb_end) {
                if (kb + 2 < kb_end) {
                    asm volatile("cp.async.wait_group 1;\n");
                } else {
                    asm volatile("cp.async.wait_group 0;\n");
                }
                __syncthreads();
                s_cur = (s_cur == 2) ? 0 : (s_cur + 1);
            }
        }
    }

    int n_bases[2] = {n0, n1};
    if (split_k == 1) {
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            int nb = n_bases[nt];
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                int m0 = mt * 8 + 2 * t;
                int m1 = m0 + 1;
                if (m0 < M) {
                    Y[(size_t)m0 * N + (nb + g)]     = __float2bfloat16(c0[nt][mt]);
                    Y[(size_t)m0 * N + (nb + g + 8)] = __float2bfloat16(c2[nt][mt]);
                }
                if (m1 < M) {
                    Y[(size_t)m1 * N + (nb + g)]     = __float2bfloat16(c1[nt][mt]);
                    Y[(size_t)m1 * N + (nb + g + 8)] = __float2bfloat16(c3[nt][mt]);
                }
            }
        }
    } else {
        float* ys = Y_split + (size_t)blockIdx.y * M * N;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            int nb = n_bases[nt];
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                int m0 = mt * 8 + 2 * t;
                int m1 = m0 + 1;
                if (m0 < M) {
                    ys[(size_t)m0 * N + (nb + g)]     = c0[nt][mt];
                    ys[(size_t)m0 * N + (nb + g + 8)] = c2[nt][mt];
                }
                if (m1 < M) {
                    ys[(size_t)m1 * N + (nb + g)]     = c1[nt][mt];
                    ys[(size_t)m1 * N + (nb + g + 8)] = c3[nt][mt];
                }
            }
        }
    }
}

__global__ void int4_splitk_reduce_to_bf16_kernel(
    const float* __restrict__ Y_split,
    __nv_bfloat16* __restrict__ Y,
    size_t MN4,
    int split_k
) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MN4) return;
    const float4* ys4 = reinterpret_cast<const float4*>(Y_split);
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #pragma unroll 8
    for (int s = 0; s < split_k; ++s) {
        float4 v = ys4[(size_t)s * MN4 + i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    uint2 out;
    __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&out);
    p[0] = __floats2bfloat162_rn(acc.x, acc.y);
    p[1] = __floats2bfloat162_rn(acc.z, acc.w);
    reinterpret_cast<uint2*>(Y)[i] = out;
}

} // namespace

void launch_pack_int4_mma(
    const uint32_t* d_w_i32,
    uint4* d_w_packed,
    int N, int K,
    cudaStream_t stream
) {
    dim3 grid(K / 64, N / 16);
    pack_int4_mma_kernel<<<grid, 32, 0, stream>>>(d_w_i32, d_w_packed, N, K);
}

void launch_pack_int4_scales(
    const __nv_bfloat16* d_scales_bf16,
    uint32_t* d_scales_packed,
    int N, int K,
    cudaStream_t stream
) {
    dim3 grid(K / 128, N / 16);
    pack_int4_scales_kernel<<<grid, 8, 0, stream>>>(d_scales_bf16, d_scales_packed, N, K);
}

void launch_int4_linear(
    cublasHandle_t cublas,
    const __nv_bfloat16* X,
    const QuantLinearWeight& W,
    __nv_bfloat16* Y,
    float* d_splitk_scratch,
    __nv_bfloat16* d_bf16_scratch,
    int M,
    cudaStream_t stream
) {
    if (M == 0) return;
    int N = W.N;
    int K = W.K;
    const uint4* w_packed = static_cast<const uint4*>(W.w_packed);
    const uint32_t* scales_packed = static_cast<const uint32_t*>(W.scales);

    if (M <= 64) {
        int wn = 4;
        int sk = 4;
        if (N >= 32768) {
            wn = 4;
            sk = (M <= 1) ? 2 : 1;
        } else if (N >= 14000) {
            wn = 4;
            sk = (M <= 16) ? 4 : 2;
        } else if (K >= 16384) {
            wn = (M <= 16) ? 2 : 4;
            sk = (M <= 16) ? 8 : 4;
        } else {
            wn = (M <= 16) ? 2 : 4;
            sk = (M <= 16) ? 4 : 2;
        }
        int max_sk = K / 128;
        if (sk > max_sk) sk = max_sk;
        if (sk < 1) sk = 1;

        int bn = wn * 32;
        int blocks_n = (N + bn - 1) / bn;

        for (int m_off = 0; m_off < M; m_off += 32) {
            int mc = min(32, M - m_off);
            const __nv_bfloat16* Xc = X + (size_t)m_off * K;
            __nv_bfloat16* Yc = Y + (size_t)m_off * N;
            dim3 grid(blocks_n, sk);
            int th = wn * 32;
            if (mc <= 8) {
                if (wn == 2) int4_group128_tc_gemm_v3_kernel<2, 1><<<grid, th, 0, stream>>>(Xc, w_packed, scales_packed, Yc, d_splitk_scratch, mc, N, K, sk);
                else         int4_group128_tc_gemm_v3_kernel<4, 1><<<grid, th, 0, stream>>>(Xc, w_packed, scales_packed, Yc, d_splitk_scratch, mc, N, K, sk);
            } else if (mc <= 16) {
                if (wn == 2) int4_group128_tc_gemm_v3_kernel<2, 2><<<grid, th, 0, stream>>>(Xc, w_packed, scales_packed, Yc, d_splitk_scratch, mc, N, K, sk);
                else         int4_group128_tc_gemm_v3_kernel<4, 2><<<grid, th, 0, stream>>>(Xc, w_packed, scales_packed, Yc, d_splitk_scratch, mc, N, K, sk);
            } else {
                if (wn == 2) int4_group128_tc_gemm_v3_kernel<2, 4><<<grid, th, 0, stream>>>(Xc, w_packed, scales_packed, Yc, d_splitk_scratch, mc, N, K, sk);
                else         int4_group128_tc_gemm_v3_kernel<4, 4><<<grid, th, 0, stream>>>(Xc, w_packed, scales_packed, Yc, d_splitk_scratch, mc, N, K, sk);
            }
            if (sk > 1) {
                size_t mn4 = ((size_t)mc * N) >> 2;
                int4_splitk_reduce_to_bf16_kernel<<<(mn4 + 255) / 256, 256, 0, stream>>>(
                    d_splitk_scratch, Yc, mn4, sk);
            }
        }
    } else {
        dim3 grid(K / 128, N / 16);
        unpack_dequant_int4_mma_to_bf16_kernel<<<grid, 64, 0, stream>>>(
            w_packed, scales_packed, d_bf16_scratch, N, K);
        launch_bf16_linear(cublas, X, d_bf16_scratch, Y, M, N, K, stream, false);
    }
}

} // namespace redhat_int4
} // namespace amperegen
