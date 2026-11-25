// K1 v3: Double-buffered cp.async sX + 2x N-tile per warp (32 N-rows/warp) FP8 Block-128 TC GEMM
// Targets 1,450-1,550 GB/s across M = 1, 6, 8, 16, 32 on CMP 170HX (70 SMs)!
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

__device__ __forceinline__ void fp8x4_to_bf16x2_raw(uint32_t w, uint32_t& out01, uint32_t& out23) {
    uint32_t t01 = __byte_perm(w, 0, 0x4140);
    uint32_t t23 = __byte_perm(w, 0, 0x4342);
    out01 = ((t01 & 0x00800080u) << 8) | ((t01 & 0x007F007Fu) << 4);
    out23 = ((t23 & 0x00800080u) << 8) | ((t23 & 0x007F007Fu) << 4);
}

__device__ __forceinline__ uint32_t bf16x2_mul_u32(uint32_t a, uint32_t b) {
    __nv_bfloat162 ha = *reinterpret_cast<const __nv_bfloat162*>(&a);
    __nv_bfloat162 hb = *reinterpret_cast<const __nv_bfloat162*>(&b);
    __nv_bfloat162 hc = __hmul2(ha, hb);
    return *reinterpret_cast<const uint32_t*>(&hc);
}

// Each CTA has WARPS_N warps, and EACH warp owns 2 consecutive 16-row N-tiles (= 32 N-rows)!
// With WARPS_N = 4 (128 threads): BN = 128 -> ALL 4 warps share ONE SINGLE 128x128 scale S[nb, kb]!
// With WARPS_N = 2 (64 threads):  BN = 64  -> ALL 2 warps share ONE SINGLE 128x128 scale S[nb, kb]!
// Double-buffered sX[2][M_TILES*8][128] loaded cooperatively (0 bank conflicts, amortized 4x-8x across N-tiles!)
template <int WARPS_N, int M_TILES>
__global__ void __launch_bounds__(WARPS_N * 32)
k1_v3_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint4* __restrict__ W_packed,      // [N/16, K/32, 32]
    const float* __restrict__ scales_fused,  // [N/128, K/128]
    __nv_bfloat16* __restrict__ Y,
    float* __restrict__ Y_split,             // [split_k, M, N]
    int M, int N, int K, int split_k
) {
    constexpr int BN = WARPS_N * 32; // 2 N-tiles per warp!
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
    if (kb_start >= kb_end) return;

    // Accumulators for 2 N-tiles x M_TILES
    float c0[2][M_TILES], c1[2][M_TILES], c2[2][M_TILES], c3[2][M_TILES];
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt)
            c0[nt][mt] = c1[nt][mt] = c2[nt][mt] = c3[nt][mt] = 0.0f;

    __shared__ __nv_bfloat16 sX[2][M_TILES * 8][128];

    int k_tiles_per_row = K >> 5;
    const uint4* w0_base = W_packed + (size_t)n_tile0 * k_tiles_per_row * 32 + lane;
    const uint4* w1_base = W_packed + (size_t)n_tile1 * k_tiles_per_row * 32 + lane;
    const float* s_row = scales_fused + (size_t)(n0 >> 7) * total_kb;

    int g = lane >> 2;
    int t = lane & 3;

    auto load_sX_stage = [&](int kb, int stage) {
        int k_base = kb << 7;
        int total_vec = (M_TILES * 8) * 16; // each uint4 = 8 bf16, 16 uint4s per 128-K row
        for (int idx = threadIdx.x; idx < total_vec; idx += WARPS_N * 32) {
            int m = idx >> 4;
            int kv = (idx & 15) << 3;
            uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sX[stage][m][kv]));
            if (m < M) {
                const void* gmem = X + (size_t)m * K + k_base + kv;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem));
            } else {
                *reinterpret_cast<uint4*>(&sX[stage][m][kv]) = make_uint4(0, 0, 0, 0);
            }
        }
        asm volatile("cp.async.commit_group;\n");
    };

    // Prime stage 0
    load_sX_stage(kb_start, 0);
    int stage = 0;

    for (int kb = kb_start; kb < kb_end; ++kb) {
        if (kb + 1 < kb_end) {
            load_sX_stage(kb + 1, stage ^ 1);
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

        __nv_bfloat162 sc2 = __bfloat162bfloat162(__float2bfloat16(__ldg(&s_row[kb])));
        uint32_t sc_u32 = *reinterpret_cast<const uint32_t*>(&sc2);

        const uint4* w0_kb = w0_base + (size_t)(kb << 2) * 32;
        const uint4* w1_kb = w1_base + (size_t)(kb << 2) * 32;

        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint4 wa = __ldcg(w0_kb + kt * 32);
            uint4 wb = __ldcg(w1_kb + kt * 32);

            // Half 0: k = kt*32 + 0..15 — load B ONCE, apply to BOTH N-tiles!
            uint32_t b0[M_TILES], b1[M_TILES];
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                int mr = mt * 8 + g;
                b0[mt] = *reinterpret_cast<const uint32_t*>(&sX[stage][mr][kt * 32 + 2 * t]);
                b1[mt] = *reinterpret_cast<const uint32_t*>(&sX[stage][mr][kt * 32 + 2 * t + 8]);
            }

            {
                uint32_t a0, a1, a2, a3;
                fp8x4_to_bf16x2_raw(wa.x, a0, a1);
                fp8x4_to_bf16x2_raw(wa.y, a2, a3);
                a0 = bf16x2_mul_u32(a0, sc_u32); a1 = bf16x2_mul_u32(a1, sc_u32);
                a2 = bf16x2_mul_u32(a2, sc_u32); a3 = bf16x2_mul_u32(a3, sc_u32);
                #pragma unroll
                for (int mt = 0; mt < M_TILES; ++mt) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+f"(c0[0][mt]), "+f"(c1[0][mt]), "+f"(c2[0][mt]), "+f"(c3[0][mt])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[mt]), "r"(b1[mt]));
                }
                fp8x4_to_bf16x2_raw(wb.x, a0, a1);
                fp8x4_to_bf16x2_raw(wb.y, a2, a3);
                a0 = bf16x2_mul_u32(a0, sc_u32); a1 = bf16x2_mul_u32(a1, sc_u32);
                a2 = bf16x2_mul_u32(a2, sc_u32); a3 = bf16x2_mul_u32(a3, sc_u32);
                #pragma unroll
                for (int mt = 0; mt < M_TILES; ++mt) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+f"(c0[1][mt]), "+f"(c1[1][mt]), "+f"(c2[1][mt]), "+f"(c3[1][mt])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[mt]), "r"(b1[mt]));
                }
            }

            // Half 1: k = kt*32 + 16..31
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                int mr = mt * 8 + g;
                b0[mt] = *reinterpret_cast<const uint32_t*>(&sX[stage][mr][kt * 32 + 16 + 2 * t]);
                b1[mt] = *reinterpret_cast<const uint32_t*>(&sX[stage][mr][kt * 32 + 16 + 2 * t + 8]);
            }
            {
                uint32_t a0, a1, a2, a3;
                fp8x4_to_bf16x2_raw(wa.z, a0, a1);
                fp8x4_to_bf16x2_raw(wa.w, a2, a3);
                a0 = bf16x2_mul_u32(a0, sc_u32); a1 = bf16x2_mul_u32(a1, sc_u32);
                a2 = bf16x2_mul_u32(a2, sc_u32); a3 = bf16x2_mul_u32(a3, sc_u32);
                #pragma unroll
                for (int mt = 0; mt < M_TILES; ++mt) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+f"(c0[0][mt]), "+f"(c1[0][mt]), "+f"(c2[0][mt]), "+f"(c3[0][mt])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[mt]), "r"(b1[mt]));
                }
                fp8x4_to_bf16x2_raw(wb.z, a0, a1);
                fp8x4_to_bf16x2_raw(wb.w, a2, a3);
                a0 = bf16x2_mul_u32(a0, sc_u32); a1 = bf16x2_mul_u32(a1, sc_u32);
                a2 = bf16x2_mul_u32(a2, sc_u32); a3 = bf16x2_mul_u32(a3, sc_u32);
                #pragma unroll
                for (int mt = 0; mt < M_TILES; ++mt) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+f"(c0[1][mt]), "+f"(c1[1][mt]), "+f"(c2[1][mt]), "+f"(c3[1][mt])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[mt]), "r"(b1[mt]));
                }
            }
        }
        stage ^= 1;
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

__global__ void splitk_red(const float* __restrict__ ys, __nv_bfloat16* __restrict__ Y, size_t MN, int sk) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MN) return;
    float a = 0.0f;
    for (int s = 0; s < sk; ++s) a += ys[(size_t)s * MN + i];
    Y[i] = __float2bfloat16(a);
}

int main() {
    struct Shape { const char* name; int N; int K; };
    std::vector<Shape> shapes = {
        {"G1_qkv_full", 14336, 5120},
        {"G2_o_proj",   5120,  6144},
        {"G3_qkvz_gdn", 16384, 5120},
        {"G6_gate_up",  34816, 5120},
        {"G7_down",     5120,  17408},
        {"G8_lm_fp8",   248320, 5120},
    };
    size_t maxW = (size_t)248320 * 5120;
    __nv_bfloat16 *dX, *dY;
    uint4* dW;
    float *dS, *dSplit;
    CHECK_CUDA(cudaMalloc(&dX, 32ULL * 17408 * 2));
    CHECK_CUDA(cudaMalloc(&dY, 32ULL * 248320 * 2));
    CHECK_CUDA(cudaMalloc(&dW, maxW));
    CHECK_CUDA(cudaMalloc(&dS, (248320/128) * (17408/128) * 4));
    CHECK_CUDA(cudaMalloc(&dSplit, 8ULL * 32 * 248320 * 4));
    CHECK_CUDA(cudaMemset(dX, 0x3c, 32ULL * 17408 * 2));
    CHECK_CUDA(cudaMemset(dW, 0x38, maxW));
    CHECK_CUDA(cudaMemset(dS, 0x3c, (248320/128) * (17408/128) * 4));

    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));

    printf("%-12s %6s %5s %3s | %8s %9s | cfg\n", "Shape", "N", "K", "M", "ms", "GB/s");
    printf("----------------------------------------------------------\n");
    for (auto& sh : shapes) {
        for (int M : {1, 6, 8, 16, 32}) {
            float best_ms = 1e9f; int b_wn = 4, b_sk = 1;
            for (int wn : {2, 4}) {
                for (int sk : {1, 2, 4, 8}) {
                    if (sh.K / 128 < sk) continue;
                    if (sh.N > 100000 && sk > 2) continue;
                    int bn = wn * 32;
                    dim3 grid((sh.N + bn - 1) / bn, sk);
                    int th = wn * 32;
                    auto run = [&]() {
                        if (M <= 8) {
                            if (wn == 2) k1_v3_kernel<2, 1><<<grid, th>>>(dX, dW, dS, dY, dSplit, M, sh.N, sh.K, sk);
                            else         k1_v3_kernel<4, 1><<<grid, th>>>(dX, dW, dS, dY, dSplit, M, sh.N, sh.K, sk);
                        } else if (M <= 16) {
                            if (wn == 2) k1_v3_kernel<2, 2><<<grid, th>>>(dX, dW, dS, dY, dSplit, M, sh.N, sh.K, sk);
                            else         k1_v3_kernel<4, 2><<<grid, th>>>(dX, dW, dS, dY, dSplit, M, sh.N, sh.K, sk);
                        } else {
                            if (wn == 2) k1_v3_kernel<2, 4><<<grid, th>>>(dX, dW, dS, dY, dSplit, M, sh.N, sh.K, sk);
                            else         k1_v3_kernel<4, 4><<<grid, th>>>(dX, dW, dS, dY, dSplit, M, sh.N, sh.K, sk);
                        }
                        if (sk > 1) {
                            size_t mn = (size_t)M * sh.N;
                            splitk_red<<<(mn + 255) / 256, 256>>>(dSplit, dY, mn, sk);
                        }
                    };
                    for (int i = 0; i < 4; ++i) run();
                    CHECK_CUDA(cudaDeviceSynchronize());
                    CHECK_CUDA(cudaEventRecord(e0));
                    int it = (sh.N > 100000) ? 15 : 40;
                    for (int i = 0; i < it; ++i) run();
                    CHECK_CUDA(cudaEventRecord(e1));
                    CHECK_CUDA(cudaEventSynchronize(e1));
                    float t; CHECK_CUDA(cudaEventElapsedTime(&t, e0, e1));
                    t /= it;
                    if (t < best_ms) { best_ms = t; b_wn = wn; b_sk = sk; }
                }
            }
            double bytes = (double)sh.N * sh.K;
            printf("%-12s %6d %5d %3d | %6.3fms %7.1fGB/s | wn=%d sk=%d\n",
                   sh.name, sh.N, sh.K, M, best_ms, bytes / (best_ms * 1e6), b_wn, b_sk);
        }
    }
    return 0;
}
