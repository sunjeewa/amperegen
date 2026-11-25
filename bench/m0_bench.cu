// M0 Roofline & FP8 Block-128x128 Dequant GEMM/GEMV Microbenchmark for CMP 170HX (sm_80)
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

#define CHECK_CUDA(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(x) do { \
    cublasStatus_t st = (x); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)st); \
        exit(1); \
    } \
} while(0)

// 1) Pure HBM read bandwidth kernel (uint4 vectorized read, minimal write)
__global__ void hbm_read_bw_kernel(const uint4* __restrict__ src, uint64_t n_vec4, float* __restrict__ out) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (uint64_t i = idx; i < n_vec4; i += stride) {
        uint4 v = __ldcg(src + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xdeadbeefU) out[idx & 255] = (float)acc;
}

// Fast exact FP8 E4M3 -> BF16x2 unpacking (value = true_fp8 * 2^-120, exact for normal, zero & subnormal!)
// Input: 4 packed FP8 bytes in uint32_t w (b0, b1, b2, b3).
// Output: r0 = bf16x2(b0, b1), r1 = bf16x2(b2, b3) scaled by 2^-120.
// Wait: let's verify bit layout carefully!
// FP8 byte b = s(7) | e3..e0(6..3) | m2..m0(2..0)
// BF16 half  = s(15)| 0 0 0 0 e3..e0(10..7) | m2..m0(6..4) 0 0 0 0
// Notice: ((b & 0x80) << 8) | ((b & 0x7F) << 4)
// For two bytes packed in bits [7:0] and [23:16] of a uint32_t t:
//   sign bits: (t & 0x80008000)? No: if we put b0 at bits[7:0] and b1 at bits[23:16]:
//   sign: (t & 0x00800080) << 8  = bits [15] and [31]
//   exp+man: (t & 0x007F007F) << 4 = bits [10:4] and [26:20]
//   Combine with 1 __byte_perm + 2 shifts + 1 lop3!
__device__ __forceinline__ void fp8x4_to_bf16x2_raw(uint32_t w, uint32_t& out01, uint32_t& out23) {
    // t01 has b0 in byte 0, b1 in byte 2 (zeros in bytes 1, 3)
    uint32_t t01 = __byte_perm(w, 0, 0x4140);
    uint32_t t23 = __byte_perm(w, 0, 0x4342);
    out01 = ((t01 & 0x00800080u) << 8) | ((t01 & 0x007F007Fu) << 4);
    out23 = ((t23 & 0x00800080u) << 8) | ((t23 & 0x007F007Fu) << 4);
}

// Multiply raw unpacked bf16x2 (value * 2^-120) by scale_bf16x2 (which has 2^120 baked in!)
// Note: Wait - does __hmul2 on sm_80 flush bf16 subnormals (FP8 subnormals e==0)?
// Let's test! Also provide a subnormal-safe helper to compare exact accuracy vs speed.
__device__ __forceinline__ uint32_t bf16x2_mul(uint32_t a, uint32_t b) {
    __nv_bfloat162 ha = *reinterpret_cast<const __nv_bfloat162*>(&a);
    __nv_bfloat162 hb = *reinterpret_cast<const __nv_bfloat162*>(&b);
    __nv_bfloat162 hc = __hmul2(ha, hb);
    return *reinterpret_cast<const uint32_t*>(&hc);
}

// ============================================================================
// Prototype K1-A: Tensor Core FP8 Block-128x128 Dequant GEMM for M <= 16
// Uses mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 with:
//   - A_mma [16 x 16] = W_dequant (16 output channels N x 16 K-elements)
//   - B_mma [16 x 8]  = X_act     (16 K-elements x up to 8 tokens M)
//   - C_mma [16 x 8]  = Y_out     (16 output channels N x up to 8 tokens M)
// Weight packing layout for ZERO-SHUFFLE coalesced 128-bit loads!
// Wait: let's see what each thread in a warp needs for mma.sync.aligned.m16n8k16 A-fragment!
// For m16n8k16, A is 16x16 BF16 row-major. Each thread (lane 0..31) holds 4 uint32_t (8 BF16s):
//   groupID = lane >> 2 (0..7), threadID_in_group = lane & 3 (0..3)
//   a0 = (row = groupID,     k = 2*tid + 0, 2*tid + 1)
//   a1 = (row = groupID + 8, k = 2*tid + 0, 2*tid + 1)
//   a2 = (row = groupID,     k = 2*tid + 8, 2*tid + 9)
//   a3 = (row = groupID + 8, k = 2*tid + 8, 2*tid + 9)
// Notice that for ONE 16x16 weight tile, each thread needs 8 FP8 weights (8 bytes = 2 uint32_t)!
// For TWO 16x16 tiles (32 N-rows x 16 K-cols, or 16 N-rows x 32 K-cols), each thread needs
// EXACTLY 16 FP8 weights = 16 bytes = 1 uint4!!
// If we pack weights offline at load time in warp-tile order:
//   For each (N_block_128, K_block_128) [which shares ONE scale!]:
//     For each warp_n in 0..7 (each 16 N-rows) and k_step in 0..3 (each 32 K-cols = 2 k16 steps):
//       Store 32 threads x 16 bytes contiguous!
//       Thread `lane` loads `uint4 w = ptr[lane]` (512 bytes = 4 cache lines, 100% coalesced!):
//         w.x -> (a0_k0, a1_k0) for k=0..15  (wait: 4 FP8s = 2x bf16x2 = a0 and a1!)
//         w.y -> (a2_k0, a3_k0) for k=0..15
//         w.z -> (a0_k1, a1_k1) for k=16..31
//         w.w -> (a2_k1, a3_k1) for k=16..31
//   WAIT A SECOND!! Look how insanely clean that is:
//   1. ONE `ld.global.v4` (16 bytes/thread = 512 bytes/warp) loads TWO full 16x16 weight tiles!
//   2. ZERO shared memory needed for weights!! ZERO bank conflicts!!
//   3. ALL 4 uint4 loads across a 128-K block (4 * 32 = 128 K-cols) share ONE SINGLE scale S[bn, bk]!!
//   4. Activations X[M, K] (for M <= 8, max 8 * 17408 * 2B = 272 KB, or 128-K tile = 2 KB!) fit in
//      shared memory or L1 cache trivially!
//   5. Multiple CTAs can split K (or split N across 70 SMs cleanly) — let's test Split-K + N-tiling!
// ============================================================================

// Offline packer: packs row-major FP8 W [N, K] into warp-MMA layout:
// Shape in uint4: [N/16, K/32, 32] (where dim 2 is warp lane 0..31).
// Each thread `lane` gets:
//   g = lane >> 2 (0..7), t = lane & 3 (0..3)
//   b0..b3   (w.x): W[n0+g,   k0+2t], W[n0+g,   k0+2t+1], W[n0+g+8, k0+2t],   W[n0+g+8, k0+2t+1]
//   b4..b7   (w.y): W[n0+g,   k0+2t+8], W[n0+g, k0+2t+9], W[n0+g+8, k0+2t+8], W[n0+g+8, k0+2t+9]
//   b8..b11  (w.z): same for k0+16
//   b12..b15 (w.w): same for k0+16+8
__global__ void pack_fp8_mma_kernel(const uint8_t* __restrict__ W_row, uint4* __restrict__ W_packed, int N, int K) {
    int n_tile = blockIdx.y; // 0 .. N/16 - 1
    int k_tile = blockIdx.x; // 0 .. K/32 - 1
    int lane = threadIdx.x;  // 0 .. 31
    int g = lane >> 2;
    int t = lane & 3;
    int n0 = n_tile * 16;
    int k0 = k_tile * 32;

    auto pack4 = [&](int kk) -> uint32_t {
        uint8_t v0 = W_row[(size_t)(n0 + g)     * K + (kk + 2 * t + 0)];
        uint8_t v1 = W_row[(size_t)(n0 + g)     * K + (kk + 2 * t + 1)];
        uint8_t v2 = W_row[(size_t)(n0 + g + 8) * K + (kk + 2 * t + 0)];
        uint8_t v3 = W_row[(size_t)(n0 + g + 8) * K + (kk + 2 * t + 1)];
        return (uint32_t)v0 | ((uint32_t)v1 << 8) | ((uint32_t)v2 << 16) | ((uint32_t)v3 << 24);
    };
    uint4 out;
    out.x = pack4(k0 + 0);
    out.y = pack4(k0 + 8);
    out.z = pack4(k0 + 16);
    out.w = pack4(k0 + 24);
    W_packed[((size_t)n_tile * (K / 32) + k_tile) * 32 + lane] = out;
}

// MMA Tensor Core FP8 Block-128x128 GEMM for M <= 8 (and M <= 16 via 2 N_mma tiles!)
// Grid: (N / (16 * WARPS_N), split_k) or (N / 128, split_k)
// Let each block handle BN = 128 (8 warps, each warp handles 16 N-rows!) OR BN = 64 (4 warps).
// Wait: with BN = 64 (4 warps = 128 threads), ALL 4 warps share the EXACT SAME 128x128 scale S[n/128, k/128]
// AND share X_smem[M, 128] loaded cooperatively!
// Let's template on MAX_M_TILES (1 for M<=8, 2 for M<=16) and SPLIT_K!
template <int WARPS_N, int M_TILES>
__global__ void __launch_bounds__(WARPS_N * 32)
fp8_block_tc_gemm_smallm(
    const __nv_bfloat16* __restrict__ X,     // [M, K] row-major
    const uint4* __restrict__ W_packed,      // [N/16, K/32, 32]
    const float* __restrict__ scales_fused,  // [N/128, K/128] = scale_inv * 2^120 (stored as bf16x2 or fp32)
    __nv_bfloat16* __restrict__ Y,           // [M, N] row-major (when split_k == 1)
    float* __restrict__ Y_partial,           // [M, N] fp32 atomic accum (when split_k > 1)
    int M, int N, int K, int split_k
) {
    // Each CTA covers WARPS_N * 16 output channels, and a slice of 128-K blocks!
    constexpr int BN = WARPS_N * 16;
    int cta_n = blockIdx.x * BN;
    if (cta_n >= N) return;

    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n_tile = (cta_n >> 4) + warp_id; // 16-row tile index
    int n_block_128 = (cta_n + warp_id * 16) >> 7;

    int total_k_blocks = K >> 7; // K / 128
    int kb_per_split = (total_k_blocks + split_k - 1) / split_k;
    int kb_start = blockIdx.y * kb_per_split;
    int kb_end = min(total_k_blocks, kb_start + kb_per_split);
    if (kb_start >= kb_end) return;

    // Shared memory for X tile: [M_TILES * 8, 128] BF16 = 2 KB (M<=8) or 4 KB (M<=16)
    __shared__ __nv_bfloat16 sX[M_TILES * 8][128];

    // Accumulators: per M_TILE, 4 FP32 values (m16n8k16 output C fragment)
    float c0[M_TILES], c1[M_TILES], c2[M_TILES], c3[M_TILES];
    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
        c0[mt] = c1[mt] = c2[mt] = c3[mt] = 0.0f;
    }

    int k_tiles_per_row = K >> 5; // K / 32
    const uint4* w_row_base = W_packed + (size_t)n_tile * k_tiles_per_row * 32;
    const float* s_row = scales_fused + (size_t)n_block_128 * total_k_blocks;

    int g = lane >> 2; // 0..7
    int t = lane & 3;  // 0..3

    for (int kb = kb_start; kb < kb_end; ++kb) {
        int k_base = kb << 7;
        // Cooperative load of X[0..M-1, k_base..k_base+127] into sX using uint4 (8 bf16 per thread)
        // Total uint4 loads = M * 16
        int total_vec = (M_TILES * 8) * 16;
        for (int idx = threadIdx.x; idx < total_vec; idx += WARPS_N * 32) {
            int m = idx >> 4;
            int kv = (idx & 15) << 3;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (m < M) {
                val = *reinterpret_cast<const uint4*>(X + (size_t)m * K + k_base + kv);
            }
            *reinterpret_cast<uint4*>(&sX[m][kv]) = val;
        }
        __syncthreads();

        // Scale for this 128x128 block: we apply scale * 2^120 in bf16x2 to unpacked weights!
        // WAIT: even better — apply `2^120` in bf16x2 (constant 0x7B807B80) or `scale * 2^120`!
        // Applying `scale * 2^120` directly in bf16x2 lets us accumulate across ALL kb directly into c0..c3!
        __nv_bfloat16 sc_bf16 = __float2bfloat16(s_row[kb]);
        __nv_bfloat162 sc2 = __bfloat162bfloat162(sc_bf16);
        uint32_t sc_u32 = *reinterpret_cast<const uint32_t*>(&sc2);

        const uint4* w_kb = w_row_base + (size_t)(kb << 2) * 32;

        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            // Load 16 FP8 weights for k = kt*32 .. kt*32+31 (2 k16 MMA steps!)
            uint4 w = __ldcg(w_kb + kt * 32 + lane);

            // Step 0: k = kt*32 + 0..15
            uint32_t a0, a1, a2, a3;
            fp8x4_to_bf16x2_raw(w.x, a0, a1);
            fp8x4_to_bf16x2_raw(w.y, a2, a3);
            a0 = bf16x2_mul(a0, sc_u32);
            a1 = bf16x2_mul(a1, sc_u32);
            a2 = bf16x2_mul(a2, sc_u32);
            a3 = bf16x2_mul(a3, sc_u32);

            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                int m_row = mt * 8 + g;
                uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sX[m_row][kt * 32 + 2 * t]);
                uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sX[m_row][kt * 32 + 2 * t + 8]);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(c0[mt]), "+f"(c1[mt]), "+f"(c2[mt]), "+f"(c3[mt])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
                );
            }

            // Step 1: k = kt*32 + 16..31
            fp8x4_to_bf16x2_raw(w.z, a0, a1);
            fp8x4_to_bf16x2_raw(w.w, a2, a3);
            a0 = bf16x2_mul(a0, sc_u32);
            a1 = bf16x2_mul(a1, sc_u32);
            a2 = bf16x2_mul(a2, sc_u32);
            a3 = bf16x2_mul(a3, sc_u32);

            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                int m_row = mt * 8 + g;
                uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sX[m_row][kt * 32 + 16 + 2 * t]);
                uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sX[m_row][kt * 32 + 16 + 2 * t + 8]);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(c0[mt]), "+f"(c1[mt]), "+f"(c2[mt]), "+f"(c3[mt])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
                );
            }
        }
        __syncthreads();
    }

    // Store C fragment:
    // For m16n8k16:
    //   c0 -> (row_n = n0 + g,     col_m = mt*8 + 2*t + 0)
    //   c1 -> (row_n = n0 + g,     col_m = mt*8 + 2*t + 1)
    //   c2 -> (row_n = n0 + g + 8, col_m = mt*8 + 2*t + 0)
    //   c3 -> (row_n = n0 + g + 8, col_m = mt*8 + 2*t + 1)
    int n0 = n_tile * 16;
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
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            int m0 = mt * 8 + 2 * t;
            int m1 = m0 + 1;
            if (m0 < M) {
                atomicAdd(&Y_partial[(size_t)m0 * N + (n0 + g)],     c0[mt]);
                atomicAdd(&Y_partial[(size_t)m0 * N + (n0 + g + 8)], c2[mt]);
            }
            if (m1 < M) {
                atomicAdd(&Y_partial[(size_t)m1 * N + (n0 + g)],     c1[mt]);
                atomicAdd(&Y_partial[(size_t)m1 * N + (n0 + g + 8)], c3[mt]);
            }
        }
    }
}

__global__ void fp32_to_bf16_kernel(const float* __restrict__ in, __nv_bfloat16* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}

// Reference dequant FP8 -> BF16
__global__ void dequant_fp8_ref_kernel(
    const __nv_fp8_e4m3* __restrict__ W,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ W_bf16,
    int N, int K
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)N * K;
    if (idx >= total) return;
    int n = (int)(idx / K);
    int k = (int)(idx % K);
    float v = float(W[idx]);
    float s = scales[(size_t)(n / 128) * (K / 128) + (k / 128)];
    W_bf16[idx] = __float2bfloat16(v * s);
}

int main(int argc, char** argv) {
    bool big_mode = (argc > 1 && strcmp(argv[1], "--big") == 0);
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("=== DEVICE ===\n");
    printf("Name: %s | CC: %d.%d | SMs: %d | L2: %.1f MB | HBM: %.2f GiB\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.l2CacheSize / (1024.0 * 1024.0), prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Max sharedMem/SM: %zu KB | Max sharedMem/Block: %zu KB\n",
           prop.sharedMemPerMultiprocessor / 1024, prop.sharedMemPerBlockOptin / 1024);

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));

    // 1) HBM bandwidth test
    printf("\n=== 1. HBM BANDWIDTH ===\n");
    size_t bw_bytes = big_mode ? (8ULL << 30) : (2ULL << 30);
    void *buf_a, *buf_b;
    CHECK_CUDA(cudaMalloc(&buf_a, bw_bytes));
    CHECK_CUDA(cudaMalloc(&buf_b, bw_bytes));
    CHECK_CUDA(cudaMemset(buf_a, 0x3c, bw_bytes));
    CHECK_CUDA(cudaMemset(buf_b, 0x00, bw_bytes));

    // D2D memcpy
    CHECK_CUDA(cudaMemcpy(buf_b, buf_a, bw_bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    int iters = 20;
    CHECK_CUDA(cudaEventRecord(ev0));
    for (int i = 0; i < iters; ++i)
        CHECK_CUDA(cudaMemcpyAsync(buf_b, buf_a, bw_bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
    printf("D2D memcpy (%.1f GB): %.1f GB/s (%.2f ms)\n",
           bw_bytes / 1e9, (2.0 * bw_bytes * iters) / (ms * 1e6), ms / iters);

    // Pure read BW (vectorized uint4)
    float* dummy_out;
    CHECK_CUDA(cudaMalloc(&dummy_out, 1024));
    uint64_t n_vec4 = bw_bytes / 16;
    hbm_read_bw_kernel<<<prop.multiProcessorCount * 8, 256>>>((const uint4*)buf_a, n_vec4, dummy_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ev0));
    for (int i = 0; i < iters; ++i)
        hbm_read_bw_kernel<<<prop.multiProcessorCount * 8, 256>>>((const uint4*)buf_a, n_vec4, dummy_out);
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));
    CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
    printf("Pure read uint4 (%.1f GB): %.1f GB/s (%.2f ms)\n",
           bw_bytes / 1e9, (1.0 * bw_bytes * iters) / (ms * 1e6), ms / iters);

    CHECK_CUDA(cudaFree(buf_a));
    CHECK_CUDA(cudaFree(buf_b));
    CHECK_CUDA(cudaFree(dummy_out));

    // 2) GEMM shapes benchmark: cuBLAS BF16 vs Prototype K1 FP8-dequant TC
    struct Shape { const char* name; int N; int K; };
    std::vector<Shape> shapes = {
        {"G1_qkv_full", 14336, 5120},
        {"G2_o_proj",   5120,  6144},
        {"G3_qkvz_gdn", 16384, 5120},
        {"G6_gate_up",  34816, 5120},
        {"G7_down",     5120,  17408},
    };
    std::vector<int> Ms = {1, 6, 8, 16};

    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));

    // Allocate max sizes
    int maxN = 34816, maxK = 17408, maxM = 4096;
    __nv_bfloat16 *dX, *dY_tc, *dY_ref, *dW_bf16;
    uint8_t* dW_fp8;
    uint4* dW_packed;
    float *dS_raw, *dS_fused, *dY_part;
    CHECK_CUDA(cudaMalloc(&dX, (size_t)maxM * maxK * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dY_tc, (size_t)maxM * maxN * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dY_ref, (size_t)maxM * maxN * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dY_part, (size_t)16 * maxN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dW_fp8, (size_t)maxN * maxK));
    CHECK_CUDA(cudaMalloc(&dW_packed, (size_t)maxN * maxK));
    CHECK_CUDA(cudaMalloc(&dW_bf16, (size_t)maxN * maxK * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dS_raw, (size_t)(maxN / 128) * (maxK / 128) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dS_fused, (size_t)(maxN / 128) * (maxK / 128) * sizeof(float)));

    // Initialize deterministic test values on host for first shape check
    printf("\n=== 2. K1 PROTOTYPE ACCURACY & SPEED (FP8 Block-128x128 TC vs cuBLAS BF16) ===\n");
    {
        int N = 5120, K = 5120, M = 6;
        std::vector<__nv_bfloat16> hX(M * K);
        std::vector<uint8_t> hW(N * K);
        std::vector<float> hS((N / 128) * (K / 128));
        std::vector<float> hSf((N / 128) * (K / 128));
        srand(42);
        for (auto& v : hX) v = __float2bfloat16(((rand() % 2001) - 1000) * 0.002f);
        for (size_t i = 0; i < hW.size(); ++i) {
            // Avoid NaN byte 0x7F and 0xFF in E4M3fn
            uint8_t b = (uint8_t)(rand() & 0xFF);
            if ((b & 0x7F) == 0x7F) b ^= 1;
            hW[i] = b;
        }
        float exp120 = ldexpf(1.0f, 120);
        for (size_t i = 0; i < hS.size(); ++i) {
            hS[i] = 1e-4f + (rand() % 1000) * 2e-6f;
            hSf[i] = hS[i] * exp120;
        }
        CHECK_CUDA(cudaMemcpy(dX, hX.data(), hX.size() * 2, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dW_fp8, hW.data(), hW.size(), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dS_raw, hS.data(), hS.size() * 4, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dS_fused, hSf.data(), hSf.size() * 4, cudaMemcpyHostToDevice));

        dim3 pack_grid(K / 32, N / 16);
        pack_fp8_mma_kernel<<<pack_grid, 32>>>(dW_fp8, dW_packed, N, K);
        size_t tot = (size_t)N * K;
        dequant_fp8_ref_kernel<<<(tot + 255) / 256, 256>>>((const __nv_fp8_e4m3*)dW_fp8, dS_raw, dW_bf16, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Reference cuBLAS: Y_ref[M, N] = X[M, K] * W_bf16[N, K]^T
        // In col-major: Y_ref^T[N, M] = W_bf16^T_cm[N, K]? Wait: row-major W[N, K] is col-major W^T[K, N].
        // Row-major X[M, K] is col-major X^T[K, M].
        // Row-major Y[M, N] is col-major Y^T[N, M] = (W^T)^T [N, K] * X^T [K, M]
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUBLAS(cublasGemmEx(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K, &alpha,
            dW_bf16, CUDA_R_16BF, K,
            dX, CUDA_R_16BF, K,
            &beta,
            dY_ref, CUDA_R_16BF, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        fp8_block_tc_gemm_smallm<4, 1><<<dim3(N / 64, 1), 128>>>(dX, dW_packed, dS_fused, dY_tc, dY_part, M, N, K, 1);
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<__nv_bfloat16> hy_ref(M * N), hy_tc(M * N);
        CHECK_CUDA(cudaMemcpy(hy_ref.data(), dY_ref, hy_ref.size() * 2, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hy_tc.data(), dY_tc, hy_tc.size() * 2, cudaMemcpyDeviceToHost));
        float max_abs = 0.0f, sum_abs = 0.0f, sum_ref = 0.0f;
        for (size_t i = 0; i < hy_ref.size(); ++i) {
            float a = __bfloat162float(hy_ref[i]);
            float b = __bfloat162float(hy_tc[i]);
            float d = fabsf(a - b);
            max_abs = std::max(max_abs, d);
            sum_abs += d;
            sum_ref += fabsf(a);
        }
        printf("Accuracy check (M=6, N=5120, K=5120, incl. FP8 subnormals!): max_abs=%.6f rel_L1=%.6f\n",
               max_abs, sum_abs / (sum_ref + 1e-12f));
    }

    // Benchmark all Qwen3.8-27B shapes!
    printf("\n%-12s %5s %6s %4s | %9s %9s | %9s %9s | %6s\n",
           "Shape", "N", "K", "M", "K1_ms", "K1_GB/s", "cuBLAS_ms", "cuB_GB/s", "Speedup");
    printf("------------------------------------------------------------------------------------\n");
    for (auto& sh : shapes) {
        int N = sh.N, K = sh.K;
        dim3 pack_grid(K / 32, N / 16);
        pack_fp8_mma_kernel<<<pack_grid, 32>>>(dW_fp8, dW_packed, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        for (int M : Ms) {
            // Try split_k in {1, 2, 4, 8} and pick best for K1
            float best_ms = 1e9f;
            int best_sk = 1;
            int best_wn = 4;
            int n_iters = 50;

            auto bench_k1 = [&](int wn, int sk) {
                int bn = wn * 16;
                dim3 grid((N + bn - 1) / bn, sk);
                int threads = wn * 32;
                auto launch = [&]() {
                    if (sk > 1) cudaMemsetAsync(dY_part, 0, (size_t)M * N * sizeof(float));
                    if (M <= 8) {
                        if (wn == 2) fp8_block_tc_gemm_smallm<2, 1><<<grid, threads>>>(dX, dW_packed, dS_fused, dY_tc, dY_part, M, N, K, sk);
                        else if (wn == 4) fp8_block_tc_gemm_smallm<4, 1><<<grid, threads>>>(dX, dW_packed, dS_fused, dY_tc, dY_part, M, N, K, sk);
                        else fp8_block_tc_gemm_smallm<8, 1><<<grid, threads>>>(dX, dW_packed, dS_fused, dY_tc, dY_part, M, N, K, sk);
                    } else {
                        if (wn == 2) fp8_block_tc_gemm_smallm<2, 2><<<grid, threads>>>(dX, dW_packed, dS_fused, dY_tc, dY_part, M, N, K, sk);
                        else if (wn == 4) fp8_block_tc_gemm_smallm<4, 2><<<grid, threads>>>(dX, dW_packed, dS_fused, dY_tc, dY_part, M, N, K, sk);
                        else fp8_block_tc_gemm_smallm<8, 2><<<grid, threads>>>(dX, dW_packed, dS_fused, dY_tc, dY_part, M, N, K, sk);
                    }
                    if (sk > 1) fp32_to_bf16_kernel<<<((size_t)M * N + 255) / 256, 256>>>(dY_part, dY_tc, (size_t)M * N);
                };
                for (int i = 0; i < 5; ++i) launch();
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaEventRecord(ev0));
                for (int i = 0; i < n_iters; ++i) launch();
                CHECK_CUDA(cudaEventRecord(ev1));
                CHECK_CUDA(cudaEventSynchronize(ev1));
                float t;
                CHECK_CUDA(cudaEventElapsedTime(&t, ev0, ev1));
                t /= n_iters;
                if (t < best_ms) { best_ms = t; best_sk = sk; best_wn = wn; }
            };

            for (int wn : {2, 4, 8}) {
                for (int sk : {1, 2, 4, 8}) {
                    if (K / 128 < sk) continue;
                    bench_k1(wn, sk);
                }
            }

            // cuBLAS BF16 benchmark (2x weight bytes!)
            float alpha = 1.0f, beta = 0.0f;
            for (int i = 0; i < 5; ++i) {
                CHECK_CUBLAS(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                    dW_bf16, CUDA_R_16BF, K, dX, CUDA_R_16BF, K, &beta,
                    dY_ref, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaEventRecord(ev0));
            for (int i = 0; i < n_iters; ++i) {
                CHECK_CUBLAS(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                    dW_bf16, CUDA_R_16BF, K, dX, CUDA_R_16BF, K, &beta,
                    dY_ref, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
            CHECK_CUDA(cudaEventRecord(ev1));
            CHECK_CUDA(cudaEventSynchronize(ev1));
            float cub_ms;
            CHECK_CUDA(cudaEventElapsedTime(&cub_ms, ev0, ev1));
            cub_ms /= n_iters;

            double fp8_bytes = (double)N * K;
            double bf16_bytes = 2.0 * (double)N * K;
            printf("%-12s %5d %6d %4d | %7.3fms %7.1fGB/s | %7.3fms %7.1fGB/s | %5.2fx (wn=%d sk=%d)\n",
                   sh.name, N, K, M,
                   best_ms, fp8_bytes / (best_ms * 1e6),
                   cub_ms, bf16_bytes / (cub_ms * 1e6),
                   cub_ms / best_ms, best_wn, best_sk);
        }
    }

    // 3) Prefill BF16 cuBLAS TFLOPS (M = 512, 4096)
    printf("\n=== 3. PREFILL BF16 TFLOPS (cuBLAS Tensor Core) ===\n");
    for (int M : {512, 4096}) {
        int N = 16384, K = 5120;
        float alpha = 1.0f, beta = 0.0f;
        for (int i = 0; i < 3; ++i)
            CHECK_CUBLAS(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                dW_bf16, CUDA_R_16BF, K, dX, CUDA_R_16BF, K, &beta,
                dY_ref, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(ev0));
        int it = 10;
        for (int i = 0; i < it; ++i)
            CHECK_CUBLAS(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                dW_bf16, CUDA_R_16BF, K, dX, CUDA_R_16BF, K, &beta,
                dY_ref, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK_CUDA(cudaEventRecord(ev1));
        CHECK_CUDA(cudaEventSynchronize(ev1));
        float t;
        CHECK_CUDA(cudaEventElapsedTime(&t, ev0, ev1));
        t /= it;
        double tflops = (2.0 * M * N * K) / (t * 1e9);
        printf("cuBLAS BF16 M=%4d N=%5d K=%5d: %.3f ms -> %.1f TFLOPS\n", M, N, K, t, tflops);
    }

    CHECK_CUBLAS(cublasDestroy(cublas));
    return 0;
}
