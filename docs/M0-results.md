# M0 Roofline, Hardware Characterization & SGLang Baseline (CMP 170HX 64GB)

Measured on an `NVIDIA CMP 170HX` (64 GB HBM2e, 70 SMs, 32 MB L2, `sm_80`, 210 W cap, CUDA 12.8).

## 1. Hardware Roofline (`bench/m0_bench --big`)

| Metric | Measured |
|---|---:|
| Usable HBM | 63.39 GiB |
| SMs / L2 / Max Shared Mem per SM | 70 SMs / 32.0 MB / 164 KB |
| HBM D2D Memcpy (8.6 GB) | **1,599.3 GB/s** |
| HBM Pure `uint4` Read (8.6 GB) | **1,651.0 GB/s** |
| cuBLAS BF16 Tensor Core GEMM (`M=512, N=16384, K=5120`) | **176.5 TFLOPS** |
| cuBLAS BF16 Tensor Core GEMM (`M=4096, N=16384, K=5120`) | **188.6 TFLOPS** |

## 2. K1 FP8 Block-128×128 Tensor Core GEMM (`bench/k1_tune`)

Warp-MMA prepacked FP8 (`m16n8k16`), zero-`__syncthreads` `__ldg`, 4-way prefetch, `2^120` exponent-bias fused into single 128×128 block scale (no 128× scale replication):

| Shape | N × K | M=1 | M=6 | M=8 |
|---|---|---:|---:|---:|
| `G1_qkv_full` | 14336 × 5120 | 0.060 ms (1,220 GB/s) | 0.068 ms (1,084 GB/s) | 0.072 ms (1,019 GB/s) |
| `G2_o_proj` | 5120 × 6144 | 0.026 ms (1,202 GB/s) | 0.028 ms (1,112 GB/s) | 0.032 ms (989 GB/s) |
| `G3_qkvz_gdn` | 16384 × 5120 | 0.065 ms (1,293 GB/s) | 0.071 ms (1,185 GB/s) | 0.074 ms (1,127 GB/s) |
| `G6_gate_up` | 34816 × 5120 | 0.130 ms (1,376 GB/s) | 0.134 ms (1,334 GB/s) | 0.141 ms (1,262 GB/s) |
| `G7_down` | 5120 × 17408 | 0.071 ms (1,264 GB/s) | 0.071 ms (1,256 GB/s) | 0.077 ms (1,164 GB/s) |
| `G8_lm_fp8` | 248320 × 5120 | 0.846 ms (**1,503 GB/s**) | 0.868 ms (**1,465 GB/s**) | 0.951 ms (1,338 GB/s) |

**Prefill decision**: for `M > 16`, on-the-fly unpack+dequant from warp-MMA FP8 to a 356 MB BF16 scratch buffer (~0.1 ms) + cuBLAS BF16 Tensor Core GEMM hits **188.6 TFLOPS** (>2.3× faster than SGLang's Marlin prefill at ~45–65 TFLOPS).

## 3. Live SGLang 0.5.20 Production Baseline (`sglang-qwen.service`, DFlash2 `block_size=6`)

| Benchmark | Prompt | Out | TTFT (s) | Decode (tok/s) | Notes |
|---|---:|---:|---:|---:|---|
| `short_code` | 26 | 256 | 0.310 | **105.35** | accept len ~3.67 |
| `short_math` | 34 | 256 | 0.130 | **135.87** | accept len ~4.53 |
| `short_tech` | 42 | 337 | 0.131 | **102.29** | accept len ~3.65 |
| `chat_short` | 54 | 512 | 0.126 | **75.24** | accept len ~2.6 |
| `chat_8k` | 13,713 | 512 | **8.080** | **75.68** | accept len ~2.6 |
| `chat_32k` | 54,994 | 512 | **39.012** | **74.96** | accept len ~2.85 |
| `batch4_1k` (4 concurrent) | 4× ~1K | 1,024 | 4.905 (mean) | **118.96** (agg) | wall 8.608 s |

## 4. Key `VERIFY:` Resolutions (vs SGLang 0.5.20 source)
- **Scale layout** (checkpoint ground truth): `weight_scale_inv` is `[N/128, K/128]`; full-attn fused QKV is `[14336, 5120]`.
- **V1/V2/V3/V4 (GDN)**: `use_qk_l2norm_in_kernel=True`, `hk = hv // 3`, `conv1d` uses **SiLU** activation, recurrence updates state **before** computing output `o = S_new @ (q_hat * 128^-0.5)`, chunk-64 WY prefill.
- **V5 (FP8 KV cache)**: SGLang uses static `k_scale = v_scale = 1.0`.
- **V7/V8/V11 (DFlash2)**: No `mask_embedding.pt`; `target_layer_ids = [5, 19, 33, 47, 61]` captures post-residual outputs of those layers; `--speculative-num-draft-tokens 6` sets `block_size = 6` (1 anchor + 5 draft), whereas the checkpoint's native trained `block_size = 8` (1 anchor + 7 draft).
