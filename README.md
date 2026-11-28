# amperegen

[![CI](https://github.com/sunjeewa/amperegen/actions/workflows/ci.yml/badge.svg)](https://github.com/sunjeewa/amperegen/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A standalone C++20 and CUDA (`sm_80`) inference engine for running **[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** with **[DFlash2](https://huggingface.co/Qwen/Qwen3.8-27B-DFlash2)** speculative decoding on **NVIDIA Ampere 64GB+ GPUs** (CMP 170HX 64GB, A100 80GB). Tokenization, Jinja chat templates, inference, and the OpenAI HTTP/SSE server all run in a single native binary.

Compared to SGLang 0.5.20 on a CMP 170HX 64GB, `amperegen` boots faster (~20s vs ~34s warm), decodes faster, fits up to 3× more KV cache, and handles 16 concurrent requests instead of 7. See [docs/BENCHMARK.md](docs/BENCHMARK.md) for benchmarks and [DESIGN.md](DESIGN.md) for technical details.

## Supported Models

`amperegen` only supports a curated set of checkpoints via compile-time weight plugins under `cpp/weights/`. Unsupported checkpoints are rejected at startup.

| Plugin | HuggingFace Model | Format | Weights VRAM | Decode Speed (`b=8`) |
|---|---|---|---:|---:|
| `qwen_fp8` | [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | FP8 (block-128) | 30.8 GB | 145–225 tok/s |
| `redhat_int4` | [`RedHatAI/Qwen3.8-27B-INT4`](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) | INT4 (W4A16 group-128) | 19.0 GB | 162–254 tok/s |

Both use **[`Qwen/Qwen3.8-27B-DFlash2`](https://huggingface.co/Qwen/Qwen3.8-27B-DFlash2)** (mirror of [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)) as the speculative draft model (~2.2 GB VRAM).

By default (`--weight-backend auto`), `amperegen` detects which plugin matches your checkpoint directory. You can also pass `--weight-backend qwen_fp8` or `--weight-backend redhat_int4` explicitly.

## Recipes (64GB GPU)

Tested configurations for a 64GB Ampere GPU (`--context-length 200000 --max-running-requests 16 --block-size 8`):

| Recipe | Plugin | `--kv-pool-gb` | `--prefix-cache-slots` | KV Capacity | Total VRAM | Use Case |
|---|---|---:|---:|---:|---:|---|
| **FP8 Default** | `qwen_fp8` | `18` | `24` | 589k tok | ~59 GB | Reference FP8 quality |
| **INT4 Max Cache** | `redhat_int4` | `18` | `48` | 589k tok | ~52 GB | Fastest decode + heavy prefix caching |
| **INT4 2× 200k Agents** | `redhat_int4` | `13` | `48` | 426k tok | ~47 GB | Two parallel 200k-context agents with headroom |

Quick sizing guide:
- **KV cache (`--kv-pool-gb`)**: 1 GB holds ~32,768 tokens. `13 GB` comfortably fits two full 200k contexts; `18 GB` holds ~590k tokens.
- **Prefix cache (`--prefix-cache-slots`)**: caches KV, GDN, and draft states across turns (~121 MB per slot) and auto-clamps if VRAM is tight.
- **Concurrency (`--max-running-requests`)**: max simultaneous active requests (up to 16); extra requests wait in the queue.

## Project Layout

- `cpp/weights/` — Weight plugins (`qwen_fp8`, `redhat_int4`) with checkpoint validation, loaders, and CUDA GEMM kernels.
- `cpp/kernels/` — Attention, Gated DeltaNet, DFlash2, norms/RoPE, and sampling CUDA kernels.
- `cpp/model/` — Qwen3.8-27B target model and DFlash2 draft model.
- `cpp/scheduler/` — Continuous batching, chunked prefill, paged KV cache, prefix cache, and CUDA Graphs.
- `cpp/server/` — OpenAI-compatible HTTP/SSE server, native BPE tokenizer (`pcre2` + `utf8proc`), `minja` Jinja chat templates, and streaming `<think>` / tool-call parser.

## Requirements

- NVIDIA Ampere GPU (`sm_80`) with ≥ 64 GB VRAM (CMP 170HX 64GB, A100 80GB)
- CUDA Toolkit 12.x, C++20 compiler, CMake ≥ 3.22, Ninja, `pkg-config`
- `libpcre2-8` and `libutf8proc` development headers (`dnf install pcre2-devel utf8proc-devel` on Fedora, `apt install libpcre2-dev libutf8proc-dev` on Ubuntu/Debian)

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build amperegen amperegen_test amperegen_bpe_detok_tests amperegen_tokenizer_tests
```

To build with only one weight plugin (for example INT4 only):

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DAMPEREGEN_WEIGHT_BACKENDS="redhat_int4"
```

<details>
<summary>Note for Fedora 44 (GCC 15 + CUDA 12.8)</summary>

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDACXX=/usr/local/cuda/bin/nvcc
export NVCC_PREPEND_FLAGS="-allow-unsupported-compiler -ccbin /usr/bin/g++-15 -D_GLIBCXX_DO_NOT_USE_BUILTIN_TRAITS -D__cpp_structured_bindings=201606L"

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build amperegen amperegen_test amperegen_bpe_detok_tests amperegen_tokenizer_tests
```

</details>

## Run

```bash
# List compiled weight plugins
./build/amperegen --list-weight-backends

# Run the INT4 2x 200k agents recipe
./build/amperegen \
  --model-path /path/to/models/Qwen3.8-27B-INT4 \
  --draft-path /path/to/models/Qwen3.8-27B-DFlash2 \
  --served-model-name Qwen3.8-27B-INT4 \
  --host 0.0.0.0 --port 8000 \
  --context-length 200000 \
  --max-running-requests 16 \
  --block-size 8 \
  --kv-pool-gb 13 \
  --prefix-cache-slots 48
```

Template systemd service files are in [docs/amperegen.service](docs/amperegen.service) (FP8) and [docs/amperegen-int4.service](docs/amperegen-int4.service) (INT4). Update `User`, `WorkingDirectory`, and model paths before installing. On SELinux-enforcing Fedora/RHEL, label the binary `bin_t` if installed under `/home`:

```bash
sudo semanage fcontext -a -t bin_t '/path/to/amperegen/build/amperegen'
sudo restorecon -v /path/to/amperegen/build/amperegen
```

## Tests & Benchmarks

```bash
# Native tokenizer & minja chat-template unit tests (CPU-only)
./build/amperegen_tokenizer_tests
./build/amperegen_bpe_detok_tests

# GPU self-tests for all compiled weight plugins
./build/amperegen_test --test-weights

# Check checkpoint qualification
./build/amperegen_test --probe-weights --model-dir /path/to/models/Qwen3.8-27B-INT4
```

## License

Licensed under the [Apache License 2.0](LICENSE). See [THIRD_PARTY.md](THIRD_PARTY.md) for third-party attributions (`minja`, `nlohmann/json`, `pcre2`, `utf8proc`).
