# `amperegen`

[![CI](https://github.com/sunjeewa/amperegen/actions/workflows/ci.yml/badge.svg)](https://github.com/sunjeewa/amperegen/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A dedicated C++20 + CUDA (`sm_80`) LLM inference engine for **NVIDIA Ampere 64 GB+ GPUs** (CMP 170HX 64GB, A100 80GB), specialized for **`Qwen3.8-27B-FP8`** with **`Qwen3.8-27B-DFlash2`** speculative decoding.

---

## Highlights (CMP 170HX 64GB @ 210 W vs SGLang 0.5.20)

| Metric | SGLang 0.5.20 | `amperegen` |
|---|---|---|
| **Warm startup to ready** | ~34 s | **~20 s** |
| **Single-stream decode (TG)** | ~75–135 tok/s | **~90–190 tok/s** |
| **Uncached prefill (PP)** | ~1,190 tok/s | **~2,260 tok/s** |
| **Multi-turn TTFT (cached 18K ctx)** | < 1 s | **0.07–0.2 s** |

- **Hybrid Radix Prefix Caching**: Reuses FP8 KV blocks, Gated DeltaNet recurrent states, DFlash2 sliding-window KV, and exact-hit logits across multi-turn conversations and shared system prompts.
- **OpenAI-Compatible HTTP/SSE API**: Drop-in `/v1/chat/completions` and `/v1/completions` with Qwen3 `<think>` reasoning split, streaming/non-streaming `qwen3_coder` tool calling, and SGLang-compatible Prometheus `/metrics`.

For full benchmark tables and kernel-level measurements, see [docs/BENCHMARK.md](docs/BENCHMARK.md) and [DESIGN.md](DESIGN.md).

---

## Architecture Overview

| Component | Path | Highlights |
|---|---|---|
| **FP8 GEMM** | `cpp/kernels/fp8_gemm.cu` | Custom `sm_80` block-scaled FP8 Tensor Core GEMM for decode (`cp.async` + bitwise FP8→BF16 unpack), coalesced dequant + cuBLAS BF16 for prefill |
| **Paged FP8 Attention** | `cpp/kernels/flash_attn.cu` | GQA-packed Tensor Core FlashDecoding & FlashAttention-2 with XOR-swizzled shared memory and bitwise FP8 KV unpack |
| **Gated DeltaNet (GDN)** | `cpp/kernels/gdn_kernels.cu` | Tensor Core chunked WY prefill, fused recurrent decode, and zero-snapshot Replay-Commit speculative verification |
| **DFlash2 Speculative Decoding** | `cpp/kernels/dflash_kernels.cu`, `cpp/model/dflash_model.cpp` | FP8 draft model + FP8 draft `lm_head` (verified by exact BF16 target), bidirectional sliding-window attention, exact rejection sampling |
| **Direct Weight Loader** | `cpp/loader/safetensors.cpp`, `cpp/loader/model_loader.cpp` | Zero-copy `mmap` safetensors reader with background prefetch, pinned DMA staging, and on-GPU weight packing |
| **Scheduler & Prefix Cache** | `cpp/scheduler/scheduler.cpp` | Continuous batching, adaptive chunked prefill, ref-counted hybrid KV+GDN+DFlash2 prefix cache, single CUDA Graph per batch size |
| **OpenAI HTTP/SSE Server** | `cpp/server/http_server.cpp`, `cpp/server/tool_parser.hpp`, `sidecar/app.py` | Reasoning split, `qwen3_coder` XML tool-call parser with JSON Schema typing, HF Jinja chat-template sidecar, live PP/TG telemetry |

---

## Requirements

- **GPU**: NVIDIA Ampere (`sm_80`) with ≥ 64 GB VRAM (e.g., CMP 170HX 64GB, A100 80GB)
- **Toolchain**: CUDA Toolkit 12.x, C++20 compiler, CMake ≥ 3.22, Ninja
- **Python 3.10+** for the tokenizer/chat-template sidecar (`sidecar/requirements.txt`)
- **Model checkpoints**: `Qwen3.8-27B-FP8` (target) and optionally `Qwen3.8-27B-DFlash2` (draft)

---

## Building

```bash
# 1. Build binaries (add -DAMPEREGEN_PORTABLE=ON for portable x86-64-v3 builds)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build amperegen amperegen_test amperegen_unit_tests

# 2. Run CPU unit tests
./build/amperegen_unit_tests

# 3. Set up the lightweight Python venv for the tokenizer sidecar
python3 -m venv .venv
.venv/bin/pip install -r sidecar/requirements.txt
```

Prebuilt Linux `x86_64` CUDA 12.8 (`sm_80`) tarballs are published on version tags via [GitHub Releases](https://github.com/sunjeewa/amperegen/releases).

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

## Running the Server

```bash
./build/amperegen \
  --model-path /path/to/models/Qwen3.8-27B-FP8 \
  --draft-path /path/to/models/Qwen3.8-27B-DFlash2 \
  --served-model-name Qwen3.8-27B-FP8 \
  --host 0.0.0.0 --port 8000 \
  --python ./.venv/bin/python \
  --sidecar-script ./sidecar/app.py
```

### API & Telemetry
- **`POST /v1/chat/completions`** (streaming SSE & non-streaming): automatic `<think>` reasoning split (`reasoning_content` / `content`), `qwen3_coder` XML tool calling (`tool_calls`), multi-turn normalization (`developer` role, `tool_choice`, `reasoning_effort`).
- **`POST /v1/completions`**, **`GET /v1/models`**, **`GET /health`**
- **`GET /metrics`** & **`GET /get_server_info`**: Prometheus metrics (both `amperegen_*` and SGLang-compatible `sglang:*` counters/gauges) plus live per-batch and per-request PP/TG logs on stderr.

### Running as a `systemd` Service

A template unit file is provided at [docs/amperegen.service](docs/amperegen.service). Update `User`, `WorkingDirectory`, and model paths before installing:

```bash
sudo cp docs/amperegen.service /etc/systemd/system/amperegen.service
sudo systemctl daemon-reload
sudo systemctl enable --now amperegen
journalctl -u amperegen -f
```

> **SELinux (Fedora/RHEL):** If installing under `/home` rather than `/opt` or `/usr/local/bin`, label the binary `bin_t` so systemd can execute it:
> ```bash
> sudo semanage fcontext -a -t bin_t '/path/to/amperegen/build/amperegen'
> sudo restorecon -v /path/to/amperegen/build/amperegen
> ```

---

## Verification & Benchmarking

```bash
# Stage-1: Per-layer numerical verification against pure-PyTorch reference
.venv/bin/python ref/torch_ref.py --model-dir /path/to/models/Qwen3.8-27B-FP8 --seq-len 150 --out-dir ref/stage1_150
./build/amperegen_test --stage1 --model-dir /path/to/models/Qwen3.8-27B-FP8 --ref-dir ref/stage1_150

# Stage-2/3: Core latency breakdown + speculative suite
.venv/bin/python tools/prep_prompts.py --prep --model-dir /path/to/models/Qwen3.8-27B-FP8
./build/amperegen_test --suite --model-dir /path/to/models/Qwen3.8-27B-FP8 --draft-dir /path/to/models/Qwen3.8-27B-DFlash2

# End-to-end HTTP/SSE server benchmark
.venv/bin/python tools/bench_amperegen.py --base-url http://127.0.0.1:8000
```

---

## Current Scope

- **Model specialization**: Layer geometry in `cpp/model/model_config.hpp` is compile-time specialized for `Qwen3.8-27B-FP8` + `Qwen3.8-27B-DFlash2` on `sm_80`.
- **VRAM prefix cache**: Hybrid prefix checkpoints live in GPU HBM with automatic LRU eviction and VRAM headroom clamping (`--prefix-cache-slots`, `--no-prefix-cache`).

---

## License

Licensed under the [Apache License 2.0](LICENSE). See [THIRD_PARTY.md](THIRD_PARTY.md) for third-party attributions (SGLang Team, Flash Linear Attention).
