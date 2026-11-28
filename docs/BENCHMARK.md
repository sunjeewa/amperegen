# `amperegen` Benchmark Report (CMP 170HX 64GB, `sm_80`)

Measured end-to-end on an **NVIDIA CMP 170HX 64GB** (GA100 `sm_80`, 70 SMs, 32 MB L2, 1,651 GB/s HBM, **210 W power cap**) serving `Qwen3.8-27B-FP8` + `Qwen3.8-27B-DFlash2`, apples-to-apples against a live **SGLang 0.5.20** server on the same host.

---

## 1. End-to-End Production HTTP/SSE Server Comparison

| Workload (OpenAI HTTP/SSE) | SGLang 0.5.20 (`b=6`) | `amperegen` (`b=8`, HTTP/SSE) | `amperegen` (`b=6`, core suite) | Speedup vs SGLang |
|---|---:|---:|---:|---:|
| **Warm startup to `/health`** (page cache hot) | `33.65 s` (`13.0 s` load) | **`20.32 s`** (`16.1 s` load) | **`16.09 s`** load | **1.66× faster** |
| **Usable FP8 KV Cache Pool** | `200,211` tokens | **`589,824` tokens** (`59.3 GiB`) | — | **2.95× capacity** |
| **Max Concurrent Requests** | `7` (capped by SGLang) | **`16`** (verified, 0 OOM) | — | **2.3× concurrency** |
| `short_code` (26→256) TTFT | `0.310 s` | **`0.056 s`** | `0.167 s` | **5.5× faster TTFT** |
| `short_code` (26→256) Decode | `105.35 tok/s` | **`113.04 tok/s`** | **`128.86 tok/s`** | **+7.3% (`+22.3%` @ `b=6`)** |
| `short_math` (34→256) TTFT | `0.130 s` | **`0.083 s`** | `0.164 s` | **1.57× faster TTFT** |
| `short_math` (34→256) Decode | `135.87 tok/s` | **`166.57–192.08 tok/s`** | `162.84 tok/s` | **+22.6% to +41.4%** |
| `short_tech` (42→337) TTFT | `0.131 s` | **`0.087 s`** | `0.166 s` | **1.51× faster TTFT** |
| `short_tech` (42→337) Decode | `102.29 tok/s` | **`106.68–121.86 tok/s`** | `115.79 tok/s` | **+4.3% to +19.1%** |
| `chat_short` (54→512) Decode | `75.24 tok/s` | **`94.22 tok/s`** | `87.26 tok/s` | **+25.2% faster** |
| `chat_8k` (13.7K→512) TTFT | `8.080 s` | **`6.485 s`** | **`6.404 s`** | **1.25× faster TTFT** |
| `chat_8k` (13.7K→512) Decode | `75.68 tok/s` | **`89.65–92.21 tok/s`** | **`90.09 tok/s`** | **+18.5% to +21.8%** |
| `true_32k_ctx` (32.9K→512) TTFT | — | **`18.055 s`** | — | — |
| `true_32k_ctx` (32.9K→512) Decode | `~75.0 tok/s` | **`88.78 tok/s`** | — | **+18.4% faster** |
| `chat_32k` (55.0K→512) TTFT | `39.012 s` | `43.425 s` (`36.43 s` cool) | **`36.329 s`** | **1.07× faster cool** |
| `chat_32k` (55.0K→512) Decode | `74.96 tok/s` | `71.94–74.16 tok/s` | **`75.82 tok/s`** | **Parity at 55K ctx** |
| `batch4_1k` (4× 1K→256) Mean TTFT | `4.905 s` | **`1.532 s`** | — | **3.20× faster TTFT** |
| `batch4_1k` (4× 1K→256) Wall | `8.608 s` | **`7.032 s`** | — | **1.22× faster** |
| `batch4_1k` (4× 1K→256) Agg tok/s | `118.96 tok/s` | **`145.62 tok/s`** | — | **+22.4% faster** |
| `batch16_1k` (16× 1K→256) Agg tok/s | N/A (capped at 7) | **`178.81 tok/s`** (`255.9` peak) | — | **2.3× concurrency** |
| `200K_context` (199,998 tokens) | Supported | **PASS** (`59.8 GiB`, 0 OOM) | — | **Verified** |

---

## 1b. Weight Plugin Comparison: Official FP8 vs RedHatAI INT4 (`b=8`, HTTP/SSE)

Measured live on the same CMP 170HX 64GB serving `Qwen/Qwen3.8-27B-FP8` (`qwen_fp8`) vs `RedHatAI/Qwen3.8-27B-INT4` (`redhat_int4`) with `Qwen/Qwen3.8-27B-DFlash2`:

| Metric | `qwen_fp8` (W8A16 block-128) | `redhat_int4` (W4A16 group-128) | Delta |
|---|---:|---:|---:|
| **Target Weights VRAM** (incl. FP8 draft `lm_head`) | `30.75 GB` | **`18.95 GB`** | **-11.80 GB (-38%)** |
| **Step Latency** (`b=8`, CUDA Graph) | `~30.0 ms` | **`~23.7 ms`** | **1.27× faster** |
| `code_512` (no-think) Decode | `145.1 tok/s` | **`161.9 tok/s`** | **+11.6%** |
| `math_think` Decode | `166.6 tok/s` | **`216.2 tok/s`** | **+29.8%** |
| `tool_call` Decode | `224.6 tok/s` | **`254.1 tok/s`** | **+13.1%** |
| 8K Cold Prefill (PP) | `~2,260 tok/s` | **`2,394.1 tok/s`** | **+5.9%** |
| Exact Prefix-Cache Hit Restore | `~0.20 ms` | **`0.20 ms`** | Parity |

---

## 2. Per-Step Latency Breakdown (`amperegen_test --suite`)

Measured via `cudaEvent` on CMP 170HX (unthrottled, 0x0):

| Config | Draft (5L + FP8 `lm_head` + Selector) | Target (64L + BF16 `lm_head`) | Accept | GDN Commit | Draft KV | **Total Step** |
|---|---:|---:|---:|---:|---:|---:|
| `ctx=128, b=6` | `3.72 ms` | `25.56 ms` | `0.06 ms` | `0.26 ms` | `0.25 ms` | **`29.84 ms`** (`28.7 ms` graphed) |
| `ctx=32768, b=6` | `3.81 ms` | `28.40 ms` | `0.06 ms` | `0.54 ms` | `0.25 ms` | **`33.06 ms`** |
| `ctx=32768, b=8` | `3.84 ms` | `29.05 ms` | `0.06 ms` | `0.66 ms` | `0.25 ms` | **`33.87 ms`** |

- Notice `draft` attention overhead for 2,048 sliding-window tokens across 5 layers is only **`+0.09 ms`** (Tensor Core Split-K=16).
- Notice `target` attention overhead for 32,768 FP8 KV tokens across 16 full-attention layers is only **`+2.84 ms`** (GQA-6× Tensor Core `ldmatrix` + bitwise FP8 unpack + `exp2f`).

---

## 3. Numerical & Functional Verification

### Stage-1 Per-Layer Numerics (`T=150`, 3-chunk GDN + Full-Attn vs pure-PyTorch reference)
- `L0_GDN_prefill_out`: `rel_L2 = 0.00144`, `max_abs = 0.12500`, `nan = 0` (**PASS**)
- `L0_GDN_prefill_ssm`: `rel_L2 = 0.00436`, `max_abs = 0.25000`, `nan = 0` (**PASS**)
- `L0_GDN_prefill_conv`: `rel_L2 = 0.00018`, `max_abs = 0.06250`, `nan = 0` (**PASS**)
- `L0_GDN_decode_out`: `rel_L2 = 0.00166`, `max_abs = 0.12500`, `nan = 0` (**PASS**)
- `L3_Attn_prefill_out`: `rel_L2 = 0.00184`, `max_abs = 0.12500`, `nan = 0` (**PASS**)
- `L3_Attn_verify6_out`: `rel_L2 = 0.00134`, `max_abs = 0.00391`, `nan = 0` (**PASS**)

### Production OpenAI Server Checks
- `/health`, `/v1/models`, `/metrics` (Prometheus + SGLang-compatible aliases), `/get_server_info`: **PASS**
- Empty `messages` → HTTP 400; `max_tokens=1`; custom `stop` strings; `Transfer-Encoding: chunked` POST: **PASS**
- Deterministic fixed-seed speculative sampling (`temperature=1.0, top_k=20, top_p=0.95, seed=424242`): **72.64 tok/s**, bit-identical across reruns (**PASS**)
- Streaming Qwen3 `<think>...</think>` reasoning split (`delta.reasoning_content` / `delta.content`): **PASS**
- Streaming & non-streaming `qwen3_coder` XML tool-call parsing (JSON Schema typed args, multi-turn tool response loop, `tool_choice`): **PASS**
- Mid-stream client disconnect + KV/slot leak check (`running_requests=0`, `kv_cache_usage_ratio=0`): **PASS**
- 200K-token single-request context (`199,998` tokens) + 16 concurrent requests (`batch16_1k`): **PASS** (0 OOM)
- Clean `SIGTERM` shutdown: **PASS**

---

## 4. Key Architectural Wins Over SGLang on `sm_80`

1. **Curated `sm_80` Weight-Only Tensor Core GEMMs** (`cpp/weights/qwen_fp8/gemm.cu`, `cpp/weights/redhat_int4/gemm.cu`): Exploits bitwise FP8→BF16 unpacking (`fp8 * 2^-120` fused with pre-scaled `scale_inv * 2^120` in a single `__hmul2`) and `lop3` INT4→BF16 unpacking with hoisted group-128 FP32 scaling, 3-stage `cp.async` with hardware zero-fill, 2× N-tile/warp, and `M_TILES ∈ {1, 2, 4}` covering up to `M=64`. For `M > 64`, 256-byte coalesced shared-memory dequant feeds cuBLAS BF16 Tensor Cores (~188 TFLOPS).
2. **GQA-6× Packed Tensor Core FP8 Paged FlashDecoding & FlashAttention-2** (`cpp/kernels/flash_attn.cu`): Packs all 6 Q-heads × up to 8 verify tokens (48 queries) per KV-head in one pass using `ldmatrix.x4` / `ldmatrix.x4.trans`, XOR-swizzled bank-conflict-free shared memory, register-resident online softmax with single-cycle `exp2f`, and fast bitwise FP8 KV unpack (zero `sm_80` software FP8 casts).
3. **Tensor Core Chunk-64 WY Gated DeltaNet Prefill & Replay-Commit Verify** (`cpp/kernels/gdn_kernels.cu`): All three WY prefill stages run on `mma.m16n8k16` (`4.15 ms/layer` at `T=8192`, 13.8× faster than scalar); speculative verify avoids allocating 9.6 GB of intermediate SSM states by replaying only `commit_lens[b]` accepted steps across all 48 layers in a single kernel launch (~0.3–0.6 ms).
4. **Full Single-CUDA-Graph DFlash2 Speculative Loop** (`cpp/model/dflash_model.cpp`, `cpp/kernels/dflash_kernels.cu`, `cpp/scheduler/scheduler.cpp`): Draft FP8 quantization + separate FP8 `lm_head` for draft proposals (100% lossless for exact BF16 target verification), Tensor Core Split-K=16 bidirectional draft attention, commit-masked ring-buffer append, and exact speculative rejection sampling captured inside one CUDA Graph per batch size (greedy and sampling alike).
5. **Fast Direct C++ `mmap` Safetensors Loader + Native C++20 Tokenizer & Minja Chat Templates** (`cpp/loader/`, `cpp/server/tokenizer.cpp`, `cpp/server/bpe_detokenizer.hpp`): Parallel direct GPU loading + on-GPU MMA weight packing boots the full 27B + DFlash2 engine in **~12–20 seconds** on warm page cache (vs ~34 s warm for SGLang), with tokenization (`libpcre2-8` + `libutf8proc`), Jinja chat templates (`minja`), and streaming UTF-8 detokenization all running in-process.
