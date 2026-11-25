# `amperegen`

Dedicated C++20 + CUDA (`sm_80`) LLM inference engine built specifically for **NVIDIA Ampere (`sm_80`) 64 GB+ GPUs** (tested on **CMP 170HX 64GB**, GA100, 70 SMs, 210 W cap; also compatible with A100 80GB) serving **`Qwen3.8-27B-FP8`** (48 Gated DeltaNet + 16 Full-Attention layers) with **`Qwen3.8-27B-DFlash2`** speculative decoding.

Compared against **SGLang 0.5.20** on the same CMP 170HX (210 W cap):
- **20.8× faster startup** (`~20 s` vs `~422 s`)
- **+18–41% faster single-stream decode** (`89–192 tok/s` across short → 32K context)
- **1.25–5.5× faster TTFT** and **+22.4% higher batched throughput** (`4× 1K`)
- **2.95× larger FP8 KV cache pool** (`589,824` tokens in 18 GB vs `200,211`) and **16 concurrent requests** (vs SGLang's 7-request mamba state cap)
- Full OpenAI-compatible HTTP/SSE API with **Qwen3 `<think>` reasoning split**, **`qwen3_coder` XML streaming/non-streaming tool calling**, and **SGLang-compatible Prometheus telemetry**

See [docs/BENCHMARK.md](docs/BENCHMARK.md) for full measured numbers and [DESIGN.md](DESIGN.md) for deep architecture notes.

---

## Architecture Overview

| Component | Path | Highlights |
|---|---|---|
| **FP8 Block-128×128 GEMM (K1 v3)** | `cpp/kernels/fp8_gemm.cu` | Exact bitwise FP8→BF16 unpack (`×2^120` fused scale), 3-stage `cp.async`, `M_TILES∈{1,2,4}` (`M≤64`, up to 1,565 GB/s), coalesced 256 B unpack + cuBLAS BF16 (~188 TFLOPS) for prefill |
| **Paged FP8 FlashDecoding & FA2** | `cpp/kernels/flash_attn.cu` | GQA-6× Tensor Core `mma.m16n8k16` + `ldmatrix.x4(.trans)` + XOR-swizzled smem + `exp2f` + bitwise FP8 KV unpack (`+2.84 ms` across 16 layers at 32K ctx) |
| **Gated DeltaNet (GDN)** | `cpp/kernels/gdn_kernels.cu` | Tensor Core Chunk-64 WY prefill (`4.15 ms/layer` @ `T=8192`), `BV=32` 192-CTA decode, single-launch Replay-Commit verify across 48 layers |
| **DFlash2 Draft & Verify** | `cpp/kernels/dflash_kernels.cu`, `cpp/model/dflash_model.cpp` | FP8 draft weights + FP8 draft `lm_head` (lossless for exact BF16 target verify), Tensor Core Split-K=16 bidirectional sliding-window attention, exact speculative rejection sampling |
| **Direct `mmap` Weight Loader** | `cpp/loader/safetensors.cpp`, `cpp/loader/model_loader.cpp` | Zero-copy `mmap`, sliding-window background prefetch, single `GpuArena`, pinned staging DMA, on-GPU MMA packing & FP8 quantization |
| **Continuous-Batch Scheduler** | `cpp/scheduler/scheduler.cpp` | Chunked prefill (8K), paged FP8 KV pool (`589,824` tokens in 18 GB), single CUDA Graph per batch size (greedy & sampling), live PP/TG telemetry |
| **OpenAI HTTP/SSE Server** | `cpp/server/http_server.cpp`, `cpp/server/tool_parser.hpp`, `sidecar/app.py` | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics`, `/get_server_info`, Qwen3 `<think>` split, `qwen3_coder` streaming/non-streaming tool calls, exact HF Jinja chat template sidecar |

---

## Requirements

- **GPU**: NVIDIA Ampere (`sm_80`) with ≥ 64 GB VRAM (e.g. CMP 170HX 64GB, A100 80GB)
- **Toolchain**: CUDA Toolkit 12.x (`nvcc`, `cublas`, `cublasLt`), C++20 compiler (`g++` or `clang++`), CMake ≥ 3.24, Ninja
- **Python 3.10+** for the lightweight tokenizer/chat-template sidecar (`sidecar/requirements.txt`)
- **Model checkpoints**: `Qwen3.8-27B-FP8` (target) and optionally `Qwen3.8-27B-DFlash2` (speculative draft)

---

## Building

```bash
# 1. Build the C++20 / CUDA binary
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build amperegen amperegen_test

# 2. Set up the lightweight Python venv for the tokenizer sidecar
python3 -m venv .venv
.venv/bin/pip install -r sidecar/requirements.txt
```

<details>
<summary>Note for Fedora 44 (GCC 15 + CUDA 12.8)</summary>

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDACXX=/usr/local/cuda/bin/nvcc
export NVCC_PREPEND_FLAGS="-allow-unsupported-compiler -ccbin /usr/bin/g++-15 -D_GLIBCXX_DO_NOT_USE_BUILTIN_TRAITS -D__cpp_structured_bindings=201606L"

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build amperegen amperegen_test
```

</details>

---

## Running the OpenAI-Compatible Server

```bash
./build/amperegen \
  --model-path /path/to/models/Qwen3.8-27B-FP8 \
  --draft-path /path/to/models/Qwen3.8-27B-DFlash2 \
  --served-model-name Qwen3.8-27B-FP8 \
  --host 0.0.0.0 --port 8000 \
  --context-length 200000 \
  --max-running-requests 16 \
  --block-size 8 \
  --kv-pool-gb 18 \
  --python ./.venv/bin/python \
  --sidecar-script ./sidecar/app.py
```

### Supported Endpoints & Features
- **`POST /v1/chat/completions`** (streaming SSE & non-streaming, `Transfer-Encoding: chunked` supported):
  - Automatic Qwen3 `<think>...</think>` reasoning split into `reasoning_content` / `content`
  - Native `qwen3_coder` XML tool-call parsing (`<tool_call><function=...><parameter=...>`) with JSON Schema type conversion into OpenAI `tool_calls` and `finish_reason: "tool_calls"`
  - Multi-turn conversation normalization (`developer` → `system`, assistant `tool_calls` arguments normalization, `reasoning_effort` & `tool_choice`)
- **`POST /v1/completions`**, **`GET /v1/models`**, **`GET /health`**
- **`GET /metrics`** & **`GET /get_server_info`**: Prometheus metrics exposing both `amperegen_*` and SGLang-compatible `sglang:*` telemetry (`prefill_throughput`, `gen_throughput`, `spec_accept_length`, `spec_accept_rate`, token & request counters), plus live PP/TG logs on stderr.

### Running as a `systemd` Service

A template unit file is provided at [docs/amperegen.service](docs/amperegen.service). Update `User`, `WorkingDirectory`, and model paths before installing:

```bash
sudo cp docs/amperegen.service /etc/systemd/system/amperegen.service
sudo systemctl daemon-reload
sudo systemctl enable --now amperegen
journalctl -u amperegen -f
```

> **SELinux (Fedora/RHEL):** If installing under `/home` rather than `/usr/local/bin` or `/opt`, label the binary `bin_t` so systemd can execute it:
> ```bash
> sudo semanage fcontext -a -t bin_t '/path/to/amperegen/build/amperegen'
> sudo restorecon -v /path/to/amperegen/build/amperegen
> ```

---

## Verification & Benchmarking

```bash
# Stage-1: Generate PyTorch golden vectors and verify per-layer numerics
.venv/bin/python ref/torch_ref.py --model-dir /path/to/models/Qwen3.8-27B-FP8 --seq-len 150 --out-dir ref/stage1_150
./build/amperegen_test --stage1 --model-dir /path/to/models/Qwen3.8-27B-FP8 --ref-dir ref/stage1_150

# Stage-2/3: Core breakdown + nospec + DFlash2 (b=6 & b=8) suite
.venv/bin/python tools/prep_prompts.py --prep --model-dir /path/to/models/Qwen3.8-27B-FP8
./build/amperegen_test --suite --model-dir /path/to/models/Qwen3.8-27B-FP8 --draft-dir /path/to/models/Qwen3.8-27B-DFlash2

# End-to-end HTTP/SSE server validation & benchmark
.venv/bin/python tools/bench_amperegen.py --base-url http://127.0.0.1:8000
```

---

## Current Scope & Roadmap

- **Hardcoded model geometry**: Dimensions in `cpp/model/model_config.hpp` are compile-time specialized for `Qwen3.8-27B-FP8` + `Qwen3.8-27B-DFlash2` on `sm_80` to maximize kernel fusion and occupancy.
- **No hybrid radix prefix caching yet**: Multi-turn sessions currently re-prefill shared prefixes; hybrid KV + GDN + DFlash2 prefix state caching is planned next.

---

## License

Licensed under the [Apache License 2.0](LICENSE). See [THIRD_PARTY.md](THIRD_PARTY.md) for third-party attributions (SGLang Team, Flash Linear Attention).
