// K1 v2 tuning microbench: zero-syncthreads __ldg + 2x weight prefetch + non-atomic split-K
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

// K1 v2: ZERO shared memory, ZERO __syncthreads(), 2x register prefetch, non-atomic split-K!
// Each warp owns 16 output rows (1 n_tile).
// Grid: (N / (WARPS_N * 16), split_k), Block: WARPS_N * 32
template <int WARPS_N, int M_TILES>
__global__ void __launch_bounds__(WARPS_N * 32)
k1_v2_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint4* __restrict__ W_packed,
    const float* __restrict__ scales_fused,
    __nv_bfloat16* __restrict__ Y,
    float* __restrict__ Y_split, // [split_k, M, N] when split_k > 1
    int M, int N, int K, int split_k
) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n_tile = blockIdx.x * WARPS_N + warp_id;
    int n0 = n_tile << 4;
    if (n0 >= N) return;

    int total_kb = K >> 7;
    int kb_per = (total_kb + split_k - 1) / split_k;
    int kb_start = blockIdx.y * kb_per;
    int kb_end = min(total_kb, kb_start + kb_per);
    if (kb_start >= kb_end) return;

    float c0[M_TILES], c1[M_TILES], c2[M_TILES], c3[M_TILES];
    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) c0[mt] = c1[mt] = c2[mt] = c3[mt] = 0.0f;

    int k_tiles_per_row = K >> 5;
    const uint4* w_base = W_packed + (size_t)n_tile * k_tiles_per_row * 32 + lane;
    const float* s_row = scales_fused + (size_t)(n0 >> 7) * total_kb;

    int g = lane >> 2;
    int t = lane & 3;

    // Precompute X row pointers for this lane's g
    const __nv_bfloat16* x_ptrs[M_TILES];
    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
        int m_row = mt * 8 + g;
        x_ptrs[mt] = (m_row < M) ? (X + (size_t)m_row * K) : nullptr;
    }

    for (int kb = kb_start; kb < kb_end; ++kb) {
        __nv_bfloat162 sc2 = __bfloat162bfloat162(__float2bfloat16(__ldg(&s_row[kb])));
        uint32_t sc_u32 = *reinterpret_cast<const uint32_t*>(&sc2);

        const uint4* w_kb = w_base + (size_t)(kb << 2) * 32;
        int k_base = kb << 7;

        // Prefetch all 4 uint4 weight loads for this 128-K block!
        uint4 w0 = __ldcg(w_kb + 0 * 32);
        uint4 w1 = __ldcg(w_kb + 1 * 32);
        uint4 w2 = __ldcg(w_kb + 2 * 32);
        uint4 w3 = __ldcg(w_kb + 3 * 32);
        uint4 ws[4] = {w0, w1, w2, w3};

        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint4 w = ws[kt];
            int k0 = k_base + kt * 32;

            uint32_t a0, a1, a2, a3;
            fp8x4_to_bf16x2_raw(w.x, a0, a1);
            fp8x4_to_bf16x2_raw(w.y, a2, a3);
            a0 = bf16x2_mul_u32(a0, sc_u32);
            a1 = bf16x2_mul_u32(a1, sc_u32);
            a2 = bf16x2_mul_u32(a2, sc_u32);
            a3 = bf16x2_mul_u32(a3, sc_u32);

            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                uint32_t b0 = 0, b1 = 0;
                if (x_ptrs[mt]) {
                    b0 = __ldg(reinterpret_cast<const uint32_t*>(x_ptrs[mt] + k0 + 2 * t));
                    b1 = __ldg(reinterpret_cast<const uint32_t*>(x_ptrs[mt] + k0 + 2 * t + 8));
                }
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(c0[mt]), "+f"(c1[mt]), "+f"(c2[mt]), "+f"(c3[mt])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
                );
            }

            fp8x4_to_bf16x2_raw(w.z, a0, a1);
            fp8x4_to_bf16x2_raw(w.w, a2, a3);
            a0 = bf16x2_mul_u32(a0, sc_u32);
            a1 = bf16x2_mul_u32(a1, sc_u32);
            a2 = bf16x2_mul_u32(a2, sc_u32);
            a3 = bf16x2_mul_u32(a3, sc_u32);

            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                uint32_t b0 = 0, b1 = 0;
                if (x_ptrs[mt]) {
                    b0 = __ldg(reinterpret_cast<const uint32_t*>(x_ptrs[mt] + k0 + 16 + 2 * t));
                    b1 = __ldg(reinterpret_cast<const uint32_t*>(x_ptrs[mt] + k0 + 16 + 2 * t + 8));
                }
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(c0[mt]), "+f"(c1[mt]), "+f"(c2[mt]), "+f"(c3[mt])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
                );
            }
        }
    }

    if (split_k == 1) {
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            int m0 = mt * 8 + 2 * t;
            int m1 = m0 + 1;
            if (m0 < M) {
                Y[(size_t)m0 * N + (n0 + g)]     = __float2bfloat16(c0[mt]);
                Y[(size_t)m0 * N + (n0 + g + 8)] = __float2bfloat16(c2[mt]);
            }
            if (m1 < M) {
                Y[(size_t)m1 * N + (n0 + g)]     = __float2bfloat16(c1[mt]);
                Y[(size_t)m1 * N + (n0 + g + 8)] = __float2bfloat16(c3[mt]);
            }
        }
    } else {
        float* ys = Y_split + (size_t)blockIdx.y * M * N;
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            int m0 = mt * 8 + 2 * t;
            int m1 = m0 + 1;
            if (m0 < M) {
                ys[(size_t)m0 * N + (n0 + g)]     = c0[mt];
                ys[(size_t)m0 * N + (n0 + g + 8)] = c2[mt];
            }
            if (m1 < M) {
                ys[(size_t)m1 * N + (n0 + g)]     = c1[mt];
                ys[(size_t)m1 * N + (n0 + g + 8)] = c3[mt];
            }
        }
    }
}

__global__ void splitk_reduce_kernel(
    const float* __restrict__ Y_split,
    __nv_bfloat16* __restrict__ Y,
    size_t MN,
    int split_k
) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MN) return;
    float acc = 0.0f;
    for (int s = 0; s < split_k; ++s) acc += Y_split[(size_t)s * MN + i];
    Y[i] = __float2bfloat16(acc);
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
        for (int M : {1, 6, 8, 16}) {
            float best_ms = 1e9f; int b_wn = 4, b_sk = 1;
            for (int wn : {2, 4, 8}) {
                for (int sk : {1, 2, 4, 8}) {
                    if (sh.K / 128 < sk) continue;
                    if (sh.N > 100000 && sk > 2) continue;
                    int bn = wn * 16;
                    dim3 grid((sh.N + bn - 1) / bn, sk);
                    int th = wn * 32;
                    auto run = [&]() {
                        if (M <= 8) {
                            if (wn == 2) k1_v2_kernel<2, 1><<<grid, th>>>(dX, dW, dS, dY, dSplit, M, sh.N, sh.K, sk);
                            else if (wn == 4) k1_v2_kernel<4, 1><<<grid, th>>>(dX, dW, dS, dY, dSplit, M, sh.N, sh.K, sk);
                            else k1_v2_kernel<8, 1><<<grid, th>>>(dX, dW, dS, dY, dSplit, M, sh.N, sh.K, sk);
                        } else {
                            if (wn == 2) k1_v2_kernel<2, 2><<<grid, th>>>(dX, dW, dS, dY, dSplit, M, sh.N, sh.K, sk);
                            else if (wn == 4) k1_v2_kernel<4, 2><<<grid, th>>>(dX, dW, dS, dY, dSplit, M, sh.N, sh.K, sk);
                            else k1_v2_kernel<8, 2><<<grid, th>>>(dX, dW, dS, dY, dSplit, M, sh.N, sh.K, sk);
                        }
                        if (sk > 1) {
                            size_t mn = (size_t)M * sh.N;
                            splitk_reduce_kernel<<<(mn + 255) / 256, 256>>>(dSplit, dY, mn, sk);
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
