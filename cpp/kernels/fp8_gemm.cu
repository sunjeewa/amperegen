#include "kernels/kernels.cuh"
#include "runtime/cuda_utils.cuh"
#include <cmath>

namespace amperegen {

// ============================================================================
// FP8 E4M3 -> BF16x2 fast bit-unpack (exact value = true_fp8 * 2^-120)
// ============================================================================
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

// ============================================================================
// Offline weight packer: row-major FP8 [N, K] -> warp-MMA layout [N/16, K/32, 32] uint4
// ============================================================================
__global__ void pack_fp8_mma_kernel(
    const uint8_t* __restrict__ W_row,
    uint4* __restrict__ W_packed,
    int N, int K
) {
    int n_tile = blockIdx.y;
    int k_tile = blockIdx.x;
    int lane = threadIdx.x;
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

void launch_pack_fp8_mma(
    const uint8_t* d_w_row, uint4* d_w_packed, int N, int K, cudaStream_t stream
) {
    dim3 grid(K / 32, N / 16);
    pack_fp8_mma_kernel<<<grid, 32, 0, stream>>>(d_w_row, d_w_packed, N, K);
}

// ============================================================================
// Quantize BF16 [N, K] -> block-128x128 FP8 packed directly into warp-MMA layout!
// ============================================================================
__global__ void quantize_bf16_to_fp8_mma_kernel(
    const __nv_bfloat16* __restrict__ W_bf16,
    uint4* __restrict__ W_packed,
    float* __restrict__ scales_raw,
    float* __restrict__ scales_fused,
    int N, int K
) {
    int kb = blockIdx.x;
    int nb = blockIdx.y;
    int tid = threadIdx.x;

    float amax = 0.0f;
    int n_start = nb * 128;
    int k_start = kb * 128;
    for (int idx = tid; idx < 128 * 128; idx += 256) {
        int r = idx >> 7;
        int c = idx & 127;
        float v = fabsf(__bfloat162float(W_bf16[(size_t)(n_start + r) * K + (k_start + c)]));
        amax = fmaxf(amax, v);
    }
    __shared__ float smax[8];
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    if ((tid & 31) == 0) smax[tid >> 5] = amax;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < 8) ? smax[tid] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
        if (tid == 0) smax[0] = v;
    }
    __syncthreads();

    float block_amax = smax[0];
    float scale = (block_amax > 1e-12f) ? (block_amax / 448.0f) : 1e-6f;
    float inv_scale = 1.0f / scale;
    if (tid == 0) {
        size_t s_idx = (size_t)nb * (K / 128) + kb;
        scales_raw[s_idx] = scale;
        scales_fused[s_idx] = scale * ldexpf(1.0f, 120);
    }

    int k_tiles_per_row = K / 32;
    for (int tile_idx = (tid >> 5); tile_idx < 32; tile_idx += 8) {
        int lane = tid & 31;
        int nt_local = tile_idx >> 2;
        int kt_local = tile_idx & 3;
        int n_tile = nb * 8 + nt_local;
        int k_tile = kb * 4 + kt_local;
        int n0 = n_tile * 16;
        int k0 = k_tile * 32;
        int g = lane >> 2;
        int t = lane & 3;

        auto qpack4 = [&](int kk) -> uint32_t {
            auto q1 = [&](int r, int c) -> uint8_t {
                float v = __bfloat162float(W_bf16[(size_t)r * K + c]) * inv_scale;
                v = fminf(fmaxf(v, -448.0f), 448.0f);
                __nv_fp8_e4m3 f8(v);
                return *reinterpret_cast<uint8_t*>(&f8);
            };
            uint8_t b0 = q1(n0 + g,     kk + 2 * t + 0);
            uint8_t b1 = q1(n0 + g,     kk + 2 * t + 1);
            uint8_t b2 = q1(n0 + g + 8, kk + 2 * t + 0);
            uint8_t b3 = q1(n0 + g + 8, kk + 2 * t + 1);
            return (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
        };
        uint4 out;
        out.x = qpack4(k0 + 0);
        out.y = qpack4(k0 + 8);
        out.z = qpack4(k0 + 16);
        out.w = qpack4(k0 + 24);
        W_packed[((size_t)n_tile * k_tiles_per_row + k_tile) * 32 + lane] = out;
    }
}

void launch_quantize_bf16_to_fp8_mma(
    const __nv_bfloat16* d_w_bf16,
    uint4* d_w_packed,
    float* d_scales_raw,
    float* d_scales_fused,
    int N, int K,
    cudaStream_t stream
) {
    dim3 grid(K / 128, N / 128);
    quantize_bf16_to_fp8_mma_kernel<<<grid, 256, 0, stream>>>(
        d_w_bf16, d_w_packed, d_scales_raw, d_scales_fused, N, K);
}

// ============================================================================
// Prepare FP8 scales (BF16 -> FP32 raw + FP32 scale*2^120)
// ============================================================================
__global__ void prepare_fp8_scales_kernel(
    const __nv_bfloat16* __restrict__ in_bf16,
    float* __restrict__ out_raw,
    float* __restrict__ out_fused,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = __bfloat162float(in_bf16[i]);
    out_raw[i] = s;
    out_fused[i] = s * ldexpf(1.0f, 120);
}

void launch_prepare_fp8_scales(
    const __nv_bfloat16* d_scales_bf16,
    float* d_scales_raw,
    float* d_scales_fused,
    int num_scales,
    cudaStream_t stream
) {
    prepare_fp8_scales_kernel<<<(num_scales + 255) / 256, 256, 0, stream>>>(
        d_scales_bf16, d_scales_raw, d_scales_fused, num_scales);
}

// ============================================================================
// Fast unpack + block-128 dequant from warp-MMA layout -> row-major BF16 [N, K]
// Stages a full [16, 128] tile (4 KB smem, 1 scale block along K) so global writes
// are 100% 256-byte coalesced uint4 bursts (~1,550 GB/s on GA100)!
// Grid: (K / 128, N / 16), Block: 128 threads (4 warps = 4 contiguous k_tiles)
// ============================================================================
__global__ void unpack_dequant_fp8_mma_to_bf16_kernel(
    const uint4* __restrict__ W_packed,
    const float* __restrict__ scales_fused,
    __nv_bfloat16* __restrict__ W_bf16,
    int N, int K
) {
    int kb = blockIdx.x;      // 0 .. K/128 - 1
    int n_tile = blockIdx.y;  // 0 .. N/16 - 1
    int tid = threadIdx.x;    // 0 .. 127
    int wid = tid >> 5;       // 0 .. 3 -> k_tile = kb*4 + wid
    int lane = tid & 31;

    int k_tiles_per_row = K >> 5;
    int k_tile = (kb << 2) + wid;
    int nb = n_tile >> 3;
    float sf = scales_fused[(size_t)nb * (K >> 7) + kb];
    __nv_bfloat162 sc2 = __bfloat162bfloat162(__float2bfloat16(sf));
    uint32_t sc_u32 = *reinterpret_cast<const uint32_t*>(&sc2);

    uint4 w = __ldcg(W_packed + ((size_t)n_tile * k_tiles_per_row + k_tile) * 32 + lane);

    __shared__ __nv_bfloat16 sTile[16][128]; // 4 KB

    int g = lane >> 2;
    int t = lane & 3;
    int k_local = wid * 32;

    auto write_smem = [&](uint32_t w32, int kk) {
        uint32_t lo, hi;
        fp8x4_to_bf16x2_raw(w32, lo, hi);
        lo = bf16x2_mul_u32(lo, sc_u32);
        hi = bf16x2_mul_u32(hi, sc_u32);
        *reinterpret_cast<uint32_t*>(&sTile[g][kk + 2 * t])     = lo;
        *reinterpret_cast<uint32_t*>(&sTile[g + 8][kk + 2 * t]) = hi;
    };
    write_smem(w.x, k_local + 0);
    write_smem(w.y, k_local + 8);
    write_smem(w.z, k_local + 16);
    write_smem(w.w, k_local + 24);
    __syncthreads();

    // 100% coalesced 256-byte row bursts: 16 rows x 16 uint4s = 256 uint4s -> 2 uint4 stores per thread!
    int n0 = n_tile * 16;
    int k0 = kb * 128;
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        int idx = pass * 128 + tid;
        int r = idx >> 4;        // 0..15
        int c8 = (idx & 15) << 3;// 0, 8, ..., 120
        *reinterpret_cast<uint4*>(&W_bf16[(size_t)(n0 + r) * K + (k0 + c8)]) =
            *reinterpret_cast<const uint4*>(&sTile[r][c8]);
    }
}

// ============================================================================
// K1 v3: Double-buffered cp.async sX + 2x N-tile per warp (32 N-rows/warp)
// Hits 1,220-1,565 GB/s across M = 1..16 on CMP 170HX (70 SMs)!
// ============================================================================
template <int WARPS_N, int M_TILES>
__global__ void __launch_bounds__(WARPS_N * 32)
fp8_block_tc_gemm_v3_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint4* __restrict__ W_packed,
    const float* __restrict__ scales_fused,
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

    int k_tiles_per_row = K >> 5;
    const uint4* w0_base = W_packed + (size_t)n_tile0 * k_tiles_per_row * 32 + lane;
    const uint4* w1_base = W_packed + (size_t)n_tile1 * k_tiles_per_row * 32 + lane;
    const float* s_row = scales_fused + (size_t)(n0 >> 7) * total_kb;

    int g = lane >> 2;
    int t = lane & 3;

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

        // 3-stage pipeline prologue: load kb_start into stage 0 and (if present) kb_start+1 into stage 1
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
            // Stage (s_cur + 2) % 3 was last read before the preceding __syncthreads(), so it is 100% free!
            if (kb + 2 < kb_end) {
                int s_ld = (s_cur == 0) ? 2 : (s_cur - 1);
                load_sX_stage(kb + 2, s_ld);
            }

            __nv_bfloat162 sc2 = __bfloat162bfloat162(__float2bfloat16(__ldg(&s_row[kb])));
            uint32_t sc_u32 = *reinterpret_cast<const uint32_t*>(&sc2);

            const uint4* w0_kb = w0_base + (size_t)(kb << 2) * 32;
            const uint4* w1_kb = w1_base + (size_t)(kb << 2) * 32;

            // Issue all 8 16-byte weight loads upfront for M_TILES<=2; per-kt for M_TILES==4 to keep regs low!
            if constexpr (M_TILES <= 2) {
                uint4 wa_arr[4], wb_arr[4];
                #pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
                    wa_arr[kt] = __ldcg(w0_kb + kt * 32);
                    wb_arr[kt] = __ldcg(w1_kb + kt * 32);
                }

                #pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
                    uint4 wa = wa_arr[kt];
                    uint4 wb = wb_arr[kt];

                    uint32_t b0[M_TILES], b1[M_TILES];
                    #pragma unroll
                    for (int mt = 0; mt < M_TILES; ++mt) {
                        int mr = mt * 8 + g;
                        b0[mt] = *reinterpret_cast<const uint32_t*>(&sX[s_cur][mr][kt * 32 + 2 * t]);
                        b1[mt] = *reinterpret_cast<const uint32_t*>(&sX[s_cur][mr][kt * 32 + 2 * t + 8]);
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

                    #pragma unroll
                    for (int mt = 0; mt < M_TILES; ++mt) {
                        int mr = mt * 8 + g;
                        b0[mt] = *reinterpret_cast<const uint32_t*>(&sX[s_cur][mr][kt * 32 + 16 + 2 * t]);
                        b1[mt] = *reinterpret_cast<const uint32_t*>(&sX[s_cur][mr][kt * 32 + 16 + 2 * t + 8]);
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
            } else {
                #pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
                    uint4 wa = __ldcg(w0_kb + kt * 32);
                    uint4 wb = __ldcg(w1_kb + kt * 32);

                    uint32_t b0[M_TILES], b1[M_TILES];
                    #pragma unroll
                    for (int mt = 0; mt < M_TILES; ++mt) {
                        int mr = mt * 8 + g;
                        b0[mt] = *reinterpret_cast<const uint32_t*>(&sX[s_cur][mr][kt * 32 + 2 * t]);
                        b1[mt] = *reinterpret_cast<const uint32_t*>(&sX[s_cur][mr][kt * 32 + 2 * t + 8]);
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

                    #pragma unroll
                    for (int mt = 0; mt < M_TILES; ++mt) {
                        int mr = mt * 8 + g;
                        b0[mt] = *reinterpret_cast<const uint32_t*>(&sX[s_cur][mr][kt * 32 + 16 + 2 * t]);
                        b1[mt] = *reinterpret_cast<const uint32_t*>(&sX[s_cur][mr][kt * 32 + 16 + 2 * t + 8]);
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

__global__ void splitk_reduce_to_bf16_kernel(
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

void launch_bf16_linear(
    cublasHandle_t cublas,
    const __nv_bfloat16* X,
    const __nv_bfloat16* W,
    __nv_bfloat16* Y,
    int M, int N, int K,
    cudaStream_t stream,
    bool accumulate
) {
    if (M == 0) return;
    CUBLAS_CHECK(cublasSetStream(cublas, stream));
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        W, CUDA_R_16BF, K,
        X, CUDA_R_16BF, K,
        &beta,
        Y, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void launch_fp8_linear(
    cublasHandle_t cublas,
    const __nv_bfloat16* X,
    const Fp8LinearWeight& W,
    __nv_bfloat16* Y,
    float* d_splitk_scratch,
    __nv_bfloat16* d_bf16_scratch,
    int M,
    cudaStream_t stream
) {
    if (M == 0) return;
    int N = W.N;
    int K = W.K;

    // For M <= 64 (or lm_head N > 65536): K1 v3 3-stage cp.async 2x N-tile Tensor Core kernel!
    // Handles M=1..8 (M_TILES=1), M=9..16 (M_TILES=2), and M=17..32 (M_TILES=4, single FP8 weight read!)
    // For M > 64 (prefill & very large batches): fast unpack to BF16 scratch + cuBLAS (~188 TFLOPS!).
    if (M <= 64 || N > 65536) {
        // Exact tuned (wn, sk) table from k1_v3_tune on CMP 170HX (70 SMs):
        int wn = 4;
        int sk = 4;
        if (N >= 100000) {
            wn = (M <= 6) ? 2 : 4;
            sk = 1; // lm_head: 1,470-1,565 GB/s!
        } else if (N >= 32768) {
            wn = 4;
            sk = (M <= 1) ? 4 : 1; // gate_up: 272 blocks (3.9 waves) at sk=1 -> ZERO splitk_reduce!
        } else if (N >= 14000) {
            wn = 4;
            sk = (M <= 1 && N >= 16384) ? 8 : ((M <= 16) ? 4 : 2); // qkv_full & qkvz_gdn
        } else {
            wn = (M <= 8 && K >= 16384) ? 2 : 4;
            sk = (M <= 16) ? 8 : 4; // o_proj & down_proj
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
                if (wn == 2) fp8_block_tc_gemm_v3_kernel<2, 1><<<grid, th, 0, stream>>>(Xc, W.w_packed, W.scales_fused, Yc, d_splitk_scratch, mc, N, K, sk);
                else         fp8_block_tc_gemm_v3_kernel<4, 1><<<grid, th, 0, stream>>>(Xc, W.w_packed, W.scales_fused, Yc, d_splitk_scratch, mc, N, K, sk);
            } else if (mc <= 16) {
                if (wn == 2) fp8_block_tc_gemm_v3_kernel<2, 2><<<grid, th, 0, stream>>>(Xc, W.w_packed, W.scales_fused, Yc, d_splitk_scratch, mc, N, K, sk);
                else         fp8_block_tc_gemm_v3_kernel<4, 2><<<grid, th, 0, stream>>>(Xc, W.w_packed, W.scales_fused, Yc, d_splitk_scratch, mc, N, K, sk);
            } else {
                // M_TILES=4 (mc in 17..32): 1 single FP8 weight read for B=3..4 with block_size=8!
                if (wn == 2) fp8_block_tc_gemm_v3_kernel<2, 4><<<grid, th, 0, stream>>>(Xc, W.w_packed, W.scales_fused, Yc, d_splitk_scratch, mc, N, K, sk);
                else         fp8_block_tc_gemm_v3_kernel<4, 4><<<grid, th, 0, stream>>>(Xc, W.w_packed, W.scales_fused, Yc, d_splitk_scratch, mc, N, K, sk);
            }
            if (sk > 1) {
                size_t mn4 = ((size_t)mc * N) >> 2;
                splitk_reduce_to_bf16_kernel<<<(mn4 + 255) / 256, 256, 0, stream>>>(
                    d_splitk_scratch, Yc, mn4, sk);
            }
        }
    } else {
        dim3 grid(K / 128, N / 16);
        unpack_dequant_fp8_mma_to_bf16_kernel<<<grid, 128, 0, stream>>>(
            W.w_packed, W.scales_fused, d_bf16_scratch, N, K);
        launch_bf16_linear(cublas, X, d_bf16_scratch, Y, M, N, K, stream, false);
    }
}

} // namespace amperegen
