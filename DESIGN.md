# amperegen — Detailed Design & Development Specification

**Status:** ready for review + development
**Scope:** single-GPU inference engine, Qwen3.8-27B-FP8 text-only, NVIDIA CMP 170HX 64GB
**Audience:** a developer/agent who has not seen this conversation. Everything needed is in this document, the model checkpoint, and the SGLang reference source listed in §13.

---

## 0. How to use this document

1. Read §1–§4 first (what we're building, on what hardware, and the exact model math).
2. §5–§8 are the design you implement. §7 is the kernel contract — each kernel names its reference implementation (file + function) in SGLang 0.5.20.
3. Items marked **`VERIFY:`** are details we identified but did not fully confirm against the reference. Resolve each during M1/M2 by reading the cited reference code and recording the answer in §12.5 before relying on it.
4. Milestones in §11 have hard exit criteria. Do not skip the validation protocol in §9 — the engine is not "done" at any milestone until its gate passes.
5. Follow §2.4 if sharing a single 64 GB GPU with an existing SGLang deployment.

---

## 1. Mission & success criteria

Build a dedicated inference engine ("amperegen") that **beats the existing SGLang deployment on this exact hardware and model**.

### 1.1 Baseline (measured 2026-11-22 from the live service)

| Metric | Value |
|---|---|
| Engine | SGLang 0.5.20, torch 2.13.0+cu130, CUDA 13.0, driver 610.57.04 |
| Single-stream decode, ~30K ctx, DFLASH spec (6 draft tokens) | **~75–85 tok/s** (accept len ~2.8/7, accept rate ~0.38, high variance 52–110) |
| Step cost | ~21 ms per target-verify step (7 candidate tokens) |
| GPU state | 210 W power cap (default 250 W), 1350 MHz SM under decode load (max 1695), 53.4/64 GB in use |
| Production flags | `--context-length 200000 --mem-fraction-static 0.75 --kv-cache-dtype fp8_e4m3 --max-running-requests 16 --chunked-prefill-size 8192 --disable-prefill-cuda-graph --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16 --speculative-algorithm DFLASH --speculative-draft-model-path .../Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 6` |

### 1.2 Roofline

Each verify step reads ~29 GB (target FP8 weights) + ~3.6 GB (DFlash2 draft) ≈ 33 GB of HBM.
At accept len 2.8 → ~11.8 GB per committed token → at ~1.4–1.5 TB/s (M0 confirms) → **~95–105 tok/s ceiling** single-stream. SGLang sits at ~75–85% of it. The addressable gap: CPU/scheduler overhead, untuned kernels, fixed draft length, conservative memory layout.

### 1.3 Success criteria

| # | Criterion |
|---|---|
| S1 | Single-stream decode (32K ctx) ≥ **95 tok/s sustained** (≥ 15% over SGLang baseline) |
| S2 | Bit-exact or tolerance-proven numerics vs reference (§9.1) |
| S3 | 200K single-request context works; 16 concurrent requests without OOM |
| S4 | TTFT @ 8K/32K prefill within 10% of SGLang |
| S5 | OpenAI-compatible endpoint drop-in for existing clients |

### 1.4 Out of scope (v1)

Vision/video inputs (text-only endpoint), multi-GPU, prefix/radix caching for mamba state, LoRA, FP4, the native MTP head (kept as a fallback drafter option, §4.11).

---

## 2. Environment & access

### 2.1 Hosts

- **Target GPU host** (Linux x86_64): NVIDIA CMP 170HX 64GB (`sm_80`), 210 W power cap.
- Local development machine: code editing and design; build and benchmark on the GPU host.

### 2.2 Key paths on the GPU host

| Path | What |
|---|---|
| `<MODELS_DIR>/Qwen3.8-27B-FP8/` | target checkpoint (29 GB, read-only) |
| `<MODELS_DIR>/Qwen3.8-27B-DFlash2/` | draft checkpoint (3.6 GB, read-only) |
| `<SGLANG_VENV>/lib/python3.13/site-packages/sglang/` | SGLang 0.5.20 reference implementation |
| `/etc/systemd/system/sglang-qwen.service` | baseline SGLang systemd unit |

### 2.3 Service management

- `systemctl status sglang-qwen` / `journalctl -u sglang-qwen -f` (log lines include per-step decode stats: accept len, tok/s, KV/mamba usage — used as live baseline oracle).
- Service runs `nvidia-smi -pm 1 -pl 210` in `ExecStartPre` — the 210 W cap is part of the benchmark contract.
- Build env (Fedora 44 + GCC 15 + CUDA 12.8): `CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15`, `NVCC_PREPEND_FLAGS="-allow-unsupported-compiler -ccbin /usr/bin/g++-15 -D_GLIBCXX_DO_NOT_USE_BUILTIN_TRAITS -D__cpp_structured_bindings=201606L"`.
- Reference python for quick tensor inspection: `<SGLANG_VENV>/bin/python` (with `safetensors`, `torch`).

### 2.4 Test-window protocol

SGLang uses ~53 GB; `amperegen` uses ~59 GB. **Both cannot run simultaneously on a 64 GB GPU.**

1. Stop the baseline service (`sudo systemctl stop sglang-qwen`) before starting `amperegen`.
2. Small microbenchmarks (M0: bandwidth, GEMM shapes, < 2 GB allocation) can run independently. Keep the full engine footprint under 60 GB for safe VRAM headroom on 64 GB cards.

---

## 3. Hardware (CMP 170HX)

| Property | Value | Source |
|---|---|---|
| Arch | Ampere, compute capability **8.0** | `torch.cuda.get_device_properties` |
| SMs | **70** | same |
| L2 | **32 MB** | same |
| HBM | 64 GB (63.4 GiB usable) | same |
| Max SM clock | 1695 MHz (runs ~1350 under 210 W decode) | nvidia-smi |
| Max mem clock | 1728 MHz | nvidia-smi |
| Power | **210 W cap** (default 250 W) | `nvidia-smi -pl` in unit |
| Driver / CUDA | 610.57.04 / 13.0 | nvidia-smi, torch |

**Implications (design-constraining):**
- **No FP8 tensor cores** (sm80). FP8 exists in this model only as *storage*: weights e4m3 with 128×128 block scales, KV cache e4m3. All compute is BF16 (tensor cores) / FP32 (SSM state math). GEMM kernels must **fuse dequant**.
- Decode is **HBM-bandwidth-bound** (≈33 GB read per verify step). Kernel work is dominated by streaming 29 GB of weights efficiently; compute headroom is large at batch 1 and tight at batch 16 (~190 TFLOPS BF16 at 1350 MHz vs ~0.86 TFLOP per 16-token step).
- 70 SMs → GEMM tile configs must be chosen for 70 SMs (multiples matter); 32 MB L2 → KV/weight tile reuse fits only small working sets; assume no L2 help for weight streaming.
- Power cap: do not raise it. v2 experiment candidate: per-phase clock policy (needs operator approval, §12.4).

M0 fills in the measured row (effective d2d bandwidth, stream-29GB bandwidth, TFLOPS per GEMM shape) in §11.M0 — those numbers fix the kernel targets in §7.

---

## 4. Model: complete architecture spec

Source of truth: `config.json` + model checkpoint (§2.2) and the SGLang implementation (§13). All shapes below are **verified from the checkpoint** unless marked otherwise.

### 4.1 Checkpoint map

| File | Size | Contents |
|---|---|---|
| `layers-0.safetensors` … `layers-63.safetensors` | ~372–384 MB each | one decoder layer per file (see §4.3/§4.4 tensor tables) |
| `outside.safetensors` | 6.0 GB | `lm_head.weight` [248320, 5120] **BF16**; `model.language_model.embed_tokens.weight` [248320, 5120] **BF16**; `model.language_model.norm.weight` [5120] BF16; entire vision tower `model.visual.*` (27 blocks, hidden 1152) BF16 — **skip vision in v1** |
| `mtp.safetensors` | 477 MB | native MTP head (1 layer) — unused by DFLASH; keep aside (fallback drafter, §4.11) |
| tokenizer.json / vocab.json / merges.txt / chat_template.jinja / generation_config.json | — | see §4.8, §6.4 |
| `crc32.txt` | — | optional integrity check at load |

**Full attention layers:** indices 3, 7, 11, …, 63 (every 4th, 0-based; `full_attention_interval: 4`, 16 layers). **Linear (GDN) layers:** the other 48.

### 4.2 Quantization scheme

- `quant_method: fp8`, `fmt: e4m3`, `activation_scheme: dynamic`, **weight block size 128×128**.
- Each quantized linear has `weight` (F8_E4M3, [N, K]) + `weight_scale_inv` (BF16, [ceil(K/128), ceil(N/128)]). Dequant: `W_bf16[n,k] = w[n,k] * scale_inv[k/128, n/128]`.
- Activation quantization is irrelevant on sm80 (no FP8 GEMM): activations stay BF16 end-to-end.
- **Not quantized** (BF16): all layernorms, `A_log`, `dt_bias`, `in_proj_a`, `in_proj_b`, GDN `norm`, `conv1d.weight`, `q_norm/k_norm`, embed_tokens, lm_head, vision.

### 4.3 Full-attention layer (16 layers) — tensor table + math

Tensors (per `layers-N.safetensors`, N ∈ {3,7,…,63}):

| Tensor | Shape | Dtype |
|---|---|---|
| `input_layernorm.weight`, `post_attention_layernorm.weight` | [5120] | BF16 |
| `self_attn.q_proj.weight` (+scale_inv [96, 40]) | [12288, 5120] | F8_E4M3 |
| `self_attn.k_proj.weight` (+[8, 40]) | [1024, 5120] | F8_E4M3 |
| `self_attn.v_proj.weight` (+[8, 40]) | [1024, 5120] | F8_E4M3 |
| `self_attn.o_proj.weight` (+[40, 48]) | [5120, 6144] | F8_E4M3 |
| `self_attn.q_norm.weight`, `k_norm.weight` | [256] | BF16 |
| `mlp.gate_proj.weight` (+[136, 40]) / `up_proj` / `down_proj` (+[40, 136]) | [17408, 5120] / [17408, 5120] / [5120, 17408] | F8_E4M3 |

Forward (b tokens, e.g. b=1 decode or b≤8192 prefill chunk; h = 5120):

1. **prepare**: `s = h_in + r`; `h = GemmaRMSNorm(s)` (weight+1 form, §4.6); carry `s` as residual.
2. **QKV** (fuse into one packed FP8 GEMM [23552, 5120] at load): `qkv = h·Wᵀ` → split `q_gate [b,12288] | k [b,1024] | v [b,1024]`.
   - `q_gate.view(b, 24, 512)` → `torch.chunk(2, -1)` → **q [b,24,256], gate [b,24,256]** (per-head blocks: [q₀₂₅₆, gate₀₂₅₆, q₁₂₅₆, gate₁₂₅₆, …]).
3. **QK-norm**: `q = GemmaRMSNorm(q, per head-dim 256)`, `k = GemmaRMSNorm(k, per 256)`.
4. **RoPE** (§4.7): NeoX, partial — first **64** of 256 dims of q and k, θ = 1e7. (Text-only: mrope collapses to 1-D positions.)
5. **KV write**: quantize k, v to FP8 e4m3 with **one per-token scale for K and one for V** (fp32), write to paged pool (§5.3).
6. **Attention**: causal, GQA 24:4, head_dim 256, scale = 256^-0.5 = 1/16, paged KV (fp8 dequant in kernel).
7. **Gate**: `attn = attn ⊙ sigmoid(gate)` (in-place; the "attn_output_gate").
8. **O**: `h_att = attn.view(b, 6144)·W_oᵀ`.
9. **MLP (SwiGLU)**: `s₂ = h_att + s`; `x = SiluMul(s₂·W_gateᵀ, s₂·W_upᵀ)` (fuse gate+up into [34816, 5120] GEMM); `h_mlp = x·W_downᵀ`.
10. **residual**: `r_out = h_mlp + s₂`.

### 4.4 Linear-attention (Gated DeltaNet, GDN) layer (48 layers) — tensor table + math

Tensors (per `layers-N.safetensors`, N ∉ full-attn set; all dims in units below):

| Tensor | Shape | Dtype |
|---|---|---|
| `input_layernorm.weight`, `post_attention_layernorm.weight` | [5120] | BF16 |
| `linear_attn.in_proj_qkv.weight` (+[80, 40]) | [10240, 5120] | F8_E4M3 |
| `linear_attn.in_proj_z.weight` (+[48, 40]) | [6144, 5120] | F8_E4M3 |
| `linear_attn.in_proj_a.weight` | [48, 5120] | BF16 |
| `linear_attn.in_proj_b.weight` | [48, 5120] | BF16 |
| `linear_attn.conv1d.weight` | [10240, 1, 4] | BF16 (depthwise, k=4) |
| `linear_attn.A_log` | [48] | BF16 in ckpt, **fp32 at runtime** |
| `linear_attn.dt_bias` | [48] | BF16 |
| `linear_attn.norm.weight` | [128] | BF16 |
| `linear_attn.out_proj.weight` (+[40, 48]) | [5120, 6144] | F8_E4M3 |
| `mlp.*` | same as §4.3 | F8_E4M3 |

GDN dims: `num_k_heads = 16`, `num_v_heads = 48` (GQA ratio 3:1), `head_k_dim = head_v_dim = 128`. So q, k: 16×128 = 2048; v, z: 48×128 = 6144; `conv_dim = 2048+2048+6144 = 10240`.

Forward (b tokens):

1. **prepare**: as §4.3.1.
2. **Projections**:
   - `qkvz = h·W_qkvzᵀ` [b, 16384] (FP8 dequant GEMM; load-time fused: `in_proj_qkv` [10240] + `in_proj_z` [6144]).
   - `ba = h·W_baᵀ` [b, 96] (**BF16 GEMM**; load-time fused: `in_proj_b` [48] + `in_proj_a` [48]). SGLang fuses both into one [16480, 5120] GEMM when rows ≤ 1024 (measured ~10% win at small b; use two GEMMs above that, or one always — tune in M0/M1).
   - Split: `q [b,16,128]`, `k [b,16,128]`, `v [b,48,128]`, `z [b,48,128]`, `b_β [b,48]`, `a [b,48]`.
3. **Short conv**: `mixed = cat(q,k,v) [b, 10240]` → causal **depthwise** conv1d, kernel 4, per-channel BF16 weight, no bias. Per-sequence ring state [10240, 3] (last 3 positions). (In production SGLang, conv may be fused into the GDN kernel — `use_fused_decode_proj_conv` — we do conv as its own small kernel or fold into the GDN decode kernel; either is fine.)
4. **Gated-DeltaNet recurrence** (the core; see §7 K8–K10). Per value head `hv ∈ 0..47`, key head `hk = f(hv)` (**VERIFY:** mapping `hv % 16` vs `hv // 3` against `gdn_triton.py`), state `S[hv] [128×128]` fp32 (production runs bf16 via `--mamba-ssm-dtype bfloat16` — we match production: **bf16 state, fp32 accum in kernels**):
   - `β = sigmoid(b_β[hv])`
   - `g = -exp(A_log[hv]) · softplus(a[hv] + dt_bias[hv])` (fp32; A_log loaded as fp32)
   - `q̂ = L2norm(q[hk])`, `k̂ = L2norm(k[hk])` (**VERIFY:** that q/k L2-norm is enabled for this model — `chunk_gated_delta_rule_fwd(..., use_qk_l2norm_in_kernel=True)` in `fla/chunk.py`; confirm the flag SGLang passes for qwen3.5 in `gdn_backend.py`)
   - Canonical gated-DeltaNet step (VERIFY exact operation order vs `fused_recurrent_gated_delta_rule`):
     ```
     S ← S ⊙ exp(g)              # per-head scalar decay over the whole [128×128]
     o = q̂ · S                    # [128]
     u = k̂ᵀ · S                   # [128]   (delta: prediction of v)
     S ← S + k̂ · (β (v − u))ᵀ     # rank-1 delta update
     ```
   - Output `o [b, 48, 128]`.
   - **Prefill** uses the chunked algorithm (chunk size 64 typical; WY-representation + inter-chunk state scan) — reference `fla/chunk.py::chunk_gated_delta_rule_fwd` + `chunk_fwd.py`, `chunk_delta_h.py`. **`VERIFY:`** exact chunk size and the inter-chunk scan variant SGLang uses (`gdn_triton.py::extend`).
5. **Gated output norm**: `x = (RMSNorm(o, per 128-dim head, w=norm.weight) ⊙ swish(z))` — `RMSNormGated(head_v_dim=128, norm_before_gate=True, activation="swish")`. Note: this norm is **plain RMSNorm (no +1)** — §4.6.
6. **Out**: `h_gdn = x.view(b, 6144)·W_outᵀ` (FP8 dequant GEMM).
7. **MLP + residual**: as §4.3.9–10.

### 4.5 Body forward (text tower)

```
h = embed_tokens(ids)                      # BF16 [248320, 5120]
r = None
for L in 0..63:
    (h, r) = decoder_layer_L(h, r)         # §4.3 or §4.4 (pre-norm residual pattern)
    if DFlash aux capture and L in {5, 19, 33, 47, 61}:   # target_layer_ids, 0-based layer idx +1 in SGLang internals
        aux_hidden.append(h)               # captured AFTER the layer's attention+MLP? VERIFY: capture point
                                            # (SGLang captures "last layer outputs" via layer_communicator
                                            #  prepare_attn_and_capture... — resolve in M1, §12.5)
h_out = GemmaRMSNorm(h + r) if r else GemmaRMSNorm(h)   # final norm, model.language_model.norm
logits = h_out · W_lmheadᵀ                  # BF16 GEMM [b, 248320]
```

Layer dispatch is by index (3,7,…,63 full; else GDN). The pattern is **identical for both layer types** — only the "attention" block differs — so the runtime is a loop over 64 layer descriptors with two code paths.

### 4.6 Norms — CRITICAL, three different flavors

| Norm | Formula | Used at |
|---|---|---|
| **GemmaRMSNorm** | `x / rms(x) · (1 + w)` | input_layernorm, post_attention_layernorm (all 64 layers), q_norm, k_norm, **final norm** |
| **plain RMSNorm** | `x / rms(x) · w` | GDN output norm (with gate, below), DFlash model norms |
| **RMSNormGated** | `(x / rms(x) · w) ⊙ swish(z)` | GDN output (per 128-dim head) |

eps = 1e-6 everywhere. Getting the `+1` wrong silently produces garbage — the first numerics test (§9.1) must catch it.

### 4.7 RoPE

NeoX style, `partial_rotary_factor = 0.25` → **rotary_dim = 64** (first 64 of the 256 head dims; adjacent pairs (0,1),(2,3),…), `rope_theta = 1e7`, no scaling. mrope (`mrope_section [11,11,10]`, interleaved) applies to multimodal position ids; **for text-only, all three coordinates are equal → ordinary 1-D RoPE**. Precompute cos/sin tables for positions 0..262143 (fp32, ~2 MB) at init.

Draft model RoPE: full head (128), NeoX, θ = 1e7, no partial factor.

### 4.8 Sampling & generation

`generation_config.json`: `temperature 1.0, top_k 20, top_p 0.95, do_sample true, eos [248046, 248044], pad/bos 248044`. Vocab 248320.
- Engine must support: greedy, temp+top-k+top-p (standard fused sampler over 248320 logits — a real cost: lm_head GEMM is 5120→248320 BF16 ≈ 1.27 GB weight read per sampled position; at decode that's part of the 29 GB).
- **VERIFY:** SGLang's exact top-k/top-p semantics with the 248320-vocab padding tail (masking beyond `org_vocab`?).
- Spec-decode accept rules (§4.10) use greedy argmax on the target logits when the request is greedy, and DSpark-adjusted sampling otherwise. v1 may support **greedy verify + standard (unadjusted) sampling verify** and flag the adjusted-sampling gap as a known deviation — measure its PPL impact in §9.2 before deciding.

### 4.9 DFlash2 draft model (checkpoint `Qwen3.8-27B-DFlash2`, 3.6 GB BF16)

Structure (`DFlash2DraftModel` in `sglang/srt/models/dflash.py`):

- 5 × `DFlashDecoderLayer` (qwen3-style, **pre-norm, plain RMSNorm**):
  - QKV: 32 Q / 8 KV heads × 128 (GQA 4:1), no bias, no output gate; **standard RMSNorm** q/k per head; full NeoX RoPE (128); `sliding_window = 2048`; **`is_causal: false` → bidirectional (ENCODER_ONLY) attention with the 2048 window mask** (`VERIFY:` exact window semantics — centered per position vs trailing; resolve in M3 against `RadixAttention` sliding-window code + `get_dflash_layer_attention_params`).
  - **`DFlashGroupedConv`** sandwiched on attention and MLP: *dynamic* depthwise grouped conv, **2 taps**, group_size 16 (320 groups), applied along the **block** sequence dim (pos = idx mod block; taps look back 1 within the block, zero-masked when pos < tap). Kernel = `base_kernel[side] + kernel_projection(x)` per token (`base_kernel` [2, 2, 5120], first tap 1.0). `prepare()` convolves the layer input; `finish()` convolves the layer output with the output-side coefficients.
  - MLP: SwiGLU, intermediate 17408.
- `fc`: Linear(5×5120 → 5120) — projects **concatenated target hidden states** from 5 target layers (`target_layer_ids [5, 19, 33, 47, 61]`, each 5120-dim) into draft hidden; then plain RMSNorm (`hidden_norm`).
- Final norm (plain RMSNorm) → returns **hidden states** (not logits). Logits come from the **shared target lm_head** restricted to the org vocab (`_project_candidate_logits`), and the selector (§4.9b).
- Draft has **no own embedding** in this checkpoint family — it borrows the target's `embed_tokens` (and lm_head); the engine keeps **one** copy of each in GPU memory.
- **Selector (DFlash2-specific)** — `CandidateSelector`:
  - `predecessor_codebook` [248320, 256], `successor_codebook` [248320, 256] (frozen, from checkpoint), `hidden_projection` Linear(5120 → 256).
  - Per predicted slot e ∈ 1..γ (γ = draft tokens = 6): draft hidden `h_e` at that slot → unary logits (target lm_head, top **16** candidates) + transition score
    `score(e, pred, c) = unary(e, c) + ⟨ A[pred] ⊙ W_h·h_e , B[c] ⟩`
    where `pred` = chosen candidate of the previous slot (slot 0's predecessor = the anchor token), `A/B` = codebook rows.
  - `sample_path` (`selector_walk_triton`): walks one path over the 6 slots — per slot, pick among its 16 scored candidates (argmax if greedy request, else Gumbel/softmax with uniforms + temperature). Output: **candidate block of 6 tokens**.
  - Selector codebooks are replicated (not sharded) — trivial on a single GPU.

### 4.10 Speculative loop (DFLASH) — the exact per-step flow

Per decode step, for a batch of bs sequences (block = 1 anchor + 6 draft = 7):

1. **Build draft block inputs** (per seq): `block_ids = [last_committed_token, MASK×6]` (`mask_token_id = 248070`; **VERIFY:** `_maybe_merge_trained_mask_embedding` — whether the mask token uses a trained embedding merged into the embed table; if so, port that merge at load time). positions = `prefix_len + [0..6]`.
2. **Draft forward**: embed block ids (target embed table) → 5 DFlash layers (bidirectional sliding-window attention over [context window ≤ 2048 materialized in the **draft KV cache**] + the block) → final norm → 7 hiddens; take `h[:, 1:, :]` (the 6 predicted slots).
3. **Candidate selection** (§4.9b): target-lm_head unary top-16 per slot + selector walk → 6 candidate tokens.
4. **Verify**: target model forward over the 7 tokens (`[anchor, cand₁..cand₆]`) with:
   - KV-cache slots pre-allocated in the target paged pool for all 7 positions;
   - aux hidden capture at target layers [5,19,33,47,61] (for the next step's draft input);
   - logits at the 6 prediction positions (+ bonus position).
5. **Accept**:
   - Greedy requests: `target_predict = argmax(logits)`; accept the longest prefix where candidates == target_predict; **bonus** = target_predict at the first mismatch (always emitted). `accept_len ∈ 0..6`, committed tokens = `accept_len + 1` (incl. bonus).
   - Sampling requests: DSpark-adjusted accept with the selector's q distribution (v1 may use unadjusted — §4.8 note).
6. **State updates** (committed = accepted tokens, count `commit_lens = accept_len + 1`):
   - Target KV (full-attn layers): keep the K/V of committed positions in the paged pool; **discard** K/V of rejected positions (slots were pre-allocated; free them).
   - Target GDN state: re-run the single-token recurrence over the `commit_lens` committed tokens per layer — reference `gdn_triton.py::target_verify`. (GDN state cannot be partially rolled back — it must be advanced exactly over committed tokens.)
   - Target conv state: same (last `commit_lens` positions).
   - Draft KV: append the **projected target hiddens** (`fc`+`hidden_norm` of the captured aux hiddens, then per-layer `kv_proj_only` — K/V only, Q discarded) for the committed positions — reference `_append_target_hidden_to_draft_kv_by_loc` / `_append_target_hidden_fused`.
   - Next step's anchor = bonus token.
7. **Emit** committed tokens (stream), stop at eos/stop/max_tokens.

**Prefill with DFLASH** (once per request): run target prefill (chunked, 8192) with `CaptureHiddenMode.FULL` → for **all prompt tokens** materialize draft KV from the captured aux hiddens (§4.10.6, `_append_target_hidden_to_draft_kv_by_loc`, positions = prompt positions) → first decode step starts with the draft block. This prefill-time KV materialization is mandatory for correctness (the draft's context window is the last ≤2048 of *all* prior tokens).

### 4.11 Fallback drafter (if the DFlash port slips)

The checkpoint ships a native **MTP head** (`mtp.safetensors`, 1 transformer layer, 477 MB; reference `sglang/srt/models/qwen3_5_mtp.py` + SGLang's standard MTP/EAGLE-style spec machinery). It is a conventional autoregressive drafter: far simpler to port (one layer, causal, no selector, no draft-KV materialization) at lower accept rate. **Decision point at M3**: if DFlash is on track, ship DFlash; if not, ship MTP to hit ≥ baseline −10% and defer DFlash to M5.

---

## 5. State & memory model

### 5.1 Per-sequence state

| Item | Size (per token unless noted) | Notes |
|---|---|---|
| Full-attn KV, 16 layers, 4 KV heads, hd 256, FP8 | 2·16·4·256 B + 16·4 B (scales) ≈ **32.1 KB** | paged blocks of 16 tokens |
| GDN SSM state, 48 layers, [48,128,128] **bf16** (fp32 accum) | **~7.58 MB / seq** | flat slot pool |
| GDN conv state, 48 layers, [10240, 3] bf16 | ~2.93 MB / seq | flat slot pool |
| Draft KV (5 layers, 8 KV heads, hd 128, bf16) | 2·5·8·128·2 B = 16 KB/token | bounded by 2048-window policy: keep last 2048 + eviction on append (sliding) |

### 5.2 GPU memory budget (64 GB)

| Pool | Size | Notes |
|---|---|---|
| Target weights (FP8 resident + BF16 scales) | ~24.2 GB | layers 0–63 |
| embed + lm_head (BF16) | ~5.09 GB | single copy, shared with draft |
| DFlash2 draft (BF16, incl. selector codebooks ~254 MB, fc 262 MB) | ~3.6 GB | |
| Target KV pool (FP8) | **26 GB** | = ~810K full-attn tokens ≈ 12 × 200K ctx, or 16 × ~50K |
| GDN state pool (bf16, 128 slots) | ~1.26 GB | 128 seq slots ≈ 8× max-running-requests (room for speculative over-alloc) |
| Draft KV pool | 1.5 GB | 128 slots × 2048 window × 16 KB |
| CUDA graphs + workspaces + activations (8K prefill chunk) | ~2 GB | |
| **Total** | **~63.6 GB** | tune KV pool to fit; target ≤ 63 GB with 1 GB slack |

### 5.3 Allocation strategy

- Weights: one-time `cudaMalloc` arena at load, packed layouts (§5.4). Never moved.
- KV: single big pool; **block table** [req, block] int32, block = 16 tokens. Free-list with LRU is unnecessary in v1 (no prefix cache) — simple free list; free on request end. Rejected verify slots are freed per step.
- GDN/draft-KV state: flat pools, one slot per running request, allocated at admission, freed at finish. No in-step migration (v1: no preemption; queue on exhaustion, §6.3).
- Scratch: per-stream workspace arena (max 8192 tokens × 5120 × 2B ≈ 80 MB per buffer class), reused across steps.
- CUDA graphs: static input/output buffers per captured bucket; dynamic data (token ids, scales, positions) copied in before replay (§6.2).

### 5.4 Weight packing (load-time transforms)

1. **FP8 GEMM layout**: store W in the layout the dequant GEMM kernel wants (see K1) — row-major [N, K] with 128×128 scale blocks co-sited; precompute a `scale` pointer per tile. All shapes are static (§7.0 table) → fixed tile configs baked in.
2. **Fused QKV** (§4.3): cat q(12288)+k(1024)+v(1024) rows → [23552, 5120] FP8 + merged scale grid [40, 184+8+8] — note scale grids are per-output-block: k/v scale rows [8,40] each merge cleanly into [40, 96, 8, 8] row blocks.
3. **Fused QKVZ** (GDN): q(2048)+k(2048)+v(6144)+z(6144) → [16384, 5120].
4. **Fused gate+up**: [34816, 5120] (gate, up row-interleaved per 128-scale-row? NO — cat is fine; SiluMul kernel reads two halves).
5. **Fused in_proj_ba**: [96, 5120] BF16 (b rows, then a rows) — tiny BF16 GEMM.
6. **Fused qkv for draft** (5 layers): [q 4096 + k 1024 + v 1024] [6144, 5120] BF16.
7. **Embed/lm_head** stay BF16 row-major.
8. Conv weights transposed to [channel, tap] contiguous for the conv kernel.

All packing verified by a load-time checksum (recompute a few probe dot-products against dequantized CPU reference — §9.1 stage 0).

---

## 6. Runtime architecture (C++)

### 6.1 Process & threads

Single C++20 binary, one process:
- **Main thread**: HTTP server (cpp-httplib) + SSE writers; talks to scheduler via lock-free queues (request in, events out).
- **Scheduler thread**: batch formation (§6.3), pool management, spec-loop orchestration; owns the CPU-side timeline.
- **GPU**: one primary stream per phase class (prefill, decode-verify, draft); event-based sync only; **no host syncs in the per-step hot path** beyond the final token-id copy (1–7 ints/req, async H2D pinned).
- Tokenizer/chat-template: **thin Python sidecar** (uvicorn, unix socket) in v1 — renders `chat_template.jinja` (jinja2 + HF `tokenizers`, byte-identical to SGLang's path) and does incremental detokenize for SSE. One IPC round-trip per request start + per-chunk detokenize (µs–ms, off the per-token GPU path). **v2**: replace with ctokenizers + jinja2cpp in-process. Rationale: the chat template is complex (tools, reasoning-effort instructions §4.8) and template fidelity is a correctness requirement, not an optimization.

### 6.2 CUDA graphs

- **Decode step (with spec loop)** captured per batch-size bucket {1,2,4,8,12,16} — the whole draft→verify→accept→state-update pipeline is graphable once buffer addresses are static. Token ids, positions, scales, block tables are inputs copied into static buffers before `cudaGraphLaunch`.
- **Prefill: eager** (variable lengths, chunked; matches SGLang's `--disable-prefill-cuda-graph`).
- Sampler: include in the graph for greedy; sampling path runs the distribution kernels in-graph too (uniforms written to static buffer per launch).
- Graph memory: static pools per bucket sized for the bucket; total ≤ 1 GB (§5.2).
- **VERIFY** at M3: whether the DFlash selector walk (data-dependent path length) graph-captures cleanly (SGLang captures it — `selector_walk_triton` is designed for captured graphs; our port must keep control flow static).

### 6.3 Scheduler

- Request states: `WAITING_TOKENIZE → WAITING_ADMISSION → PREFILL → DECODE → FINISHED` (+ `QUEUED_KV_FULL` on pool exhaustion — v1 has no preemption; admission control: admit only if KV + GDN slots fit; `max_running_requests = 16` default, configurable).
- **Batch formation** (continuous batching, every scheduler tick ≈ as soon as GPU is idle):
  - All running decode sequences form the verify batch.
  - One admitted request at a time gets its prefill, chunked at 8192 tokens; the chunk's tokens join the batch as an "extend" segment (mixed batch: decode verify tokens + prefill chunk tokens in one target forward — the two attention paths coexist via per-segment metadata; GDN `extend` kernel handles variable-length segments).
  - Draft runs only over the decode-verify portion.
- **Chunked prefill** 8192 (match production; tune ± in M4).
- **Spec loop**: per §4.10; the scheduler owns accept accounting, KV slot free/keep, GDN slot update dispatch.
- **Emit**: committed tokens → SSE stream (via sidecar detokenize), per-request stop checks (eos 248044/248046, stop strings, max_tokens, `allow-auto-truncate` behavior: context-length 200000 hard cap with truncation of oversized prompts — replicate the flag).

### 6.4 Server API (OpenAI-compatible)

- `POST /v1/chat/completions` (stream + non-stream), `POST /v1/completions`, `GET /v1/models`, `GET /health`, `GET /metrics` (prometheus text: tok/s, accept len, queue, KV/mamba pool usage — mirror SGLang's metric names where feasible).
- Sampling params: temperature, top_k, top_p, max_tokens, stop, stream, seed (deterministic path for tests §9.5).
- Error contract: 400 malformed / 422 context overflow (unless `allow-auto-truncate`) / 503 pool saturated (with `Retry-After`).
- Chat template: rendered by the sidecar from the checkpoint's `chat_template.jinja` (supports tools + `enable_thinking`/`reasoning_effort` vars — xhigh/medium/low; see template).

---

## 7. Kernel specifications

Conventions: BF16 compute unless stated; `b` = token count of the segment; all pointers are the packed layouts of §5.4. **Priority**: P0 = needed for M2 (first correct token), P1 = M3 (spec decode), P2 = M4/M5 (batching/opt).
Reference (REF) paths are under `<SGLANG_SITE_PACKAGES>/sglang/`.

### 7.0 GEMM shape inventory (all static)

| Name | N×K | Layers | b-range |
|---|---|---|---|
| G1 qkv (full) | 23552×5120 | 16 | 1–8192 |
| G2 o_proj (full) | 5120×6144 | 16 | 1–8192 |
| G3 qkvz (GDN) | 16384×5120 | 48 | 1–8192 |
| G4 ba (GDN, BF16) | 96×5120 | 48 | 1–8192 |
| G5 out_proj (GDN) | 5120×6144 | 48 | 1–8192 |
| G6 gate_up (all) | 34816×5120 | 64 | 1–8192 |
| G7 down (all) | 5120×17408 | 64 | 1–8192 |
| G8 lm_head (BF16) | 248320×5120 | 1 | 1–16 (verify only) |
| G9 draft qkv | 6144×5120 | 5 | 7–112 (7×bs) |
| G10 draft o/mlp | (4096×5120), (34816×5120), (5120×17408) | 5 each | 7–112 |
| G11 draft fc | 5120×25600 | 1 | ≤2048 |
| G12 selector hidden_proj | 256×5120 | 1 | 6×16 |

### 7.1 K1 — `fp8_block_dequant_gemm` (P0, THE kernel)

- **In**: A bf16 [b, K] (activations, row-major), B fp8 e4m3 [N, K] (packed), S bf16 [K/128, N/128].
- **Out**: C bf16 [b, N] (optionally split views for G1/G3 consumers).
- **Algo**: CTA tile (e.g. 128×128 C, tune for 70 SMs); load B tile, multiply by the per-128×128-block scale **in registers → bf16**, then `mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32` (or `f16` path — decide in M0 from the microbench: bf16 mma is native on sm80). A is bf16 already, no dequant needed. K-loop unrolled for K ∈ {5120, 6144, 17408, 25600, 5120...} (all K % 128 == 0 — true for every shape above; assert at load).
- **Decode b=1–16**: split-K or GEMV-style path? At b=1, N×K=23552×5120 is a skinny GEMV — this kernel at b≤16 is the entire decode performance (it streams 29 GB/step). Design it first as **memory-bound GEMV with fp8→bf16 in-register dequant** (each SM streams a contiguous N-slice, bf16 FMA against the token vector in registers/shared), then add the tiled tensor-core path for b ≥ 32 (prefill).
- **REF**: sgl-kernel fp8 block dequant GEMM for sm80 (search `site-packages/sgl_kernel/` / `python/sglang/srt/layers/quantization/fp8.py` w8a8 block path for sm80); vLLM's `cutlass_scaled_mm` block-fp8 is the other reference. Port the math/layout, then specialize to our 7 static shapes (no runtime shape dispatch).
- **Target**: ≥ 90% of measured stream bandwidth (§3, M0) at b=1; ≥ 70% of M0's BF16-peak TFLOPS at b=8192.
- **Tolerance**: vs torch dequant-GEMM reference, |Δ| ≤ 2e-2 relative on bf16 (dequant is exact; only mma accumulation order differs).

### 7.2 K2 — `fused_add_rmsnorm_gemma` (P0)

- **In**: x bf16 [b, 5120], residual r bf16 [b, 5120], w bf16 [5120], eps 1e-6.
- **Out**: s = x + r (written back to r buffer), h = (s/rms(s))·(1+w).
- One CTA per row-block; fp32 accumulation. Plain variant (no +1) for draft/GDN use. Target: bandwidth-bound, ≤ 2× traffic.

### 7.3 K3 — `qkv_gate_prepare` (P0)

Fuses §4.3.2–4: split q_gate → (q, gate) deinterleave, GemmaRMSNorm per 256-dim head on q and k, partial NeoX RoPE (64 dims) in-place. **REF**: `srt/layers/attention/... fused_qk_gemma_rmsnorm_rope_gate` (search `sgl_kernel` for the triton kernel) — port or reimplement; either is ~100 lines. Inputs are G1's output buffer; outputs q [b,24,256], k [b,4,256], v [b,4,256], gate [b,24,256] (gate may stay a strided view).

### 7.4 K4 — flash attention, full-attn layers (P0)

- **Decode (b=1/seq, up to 16 seqs)**: paged KV, hd 256, GQA 24:4, **fp8 K/V dequant in-kernel** (per-token scale fp32, §5.3), causal. Q from K3. Output bf16 [b, 24, 256] → **epilogue: `⊙ sigmoid(gate)`** (K6 folded in).
- **Prefill/extend (causal, variable length, chunked 8192, mixed with decode rows)**: standard FA2-style two-pass (online softmax), same dequant path.
- **hd 256 on sm80 is register-heavy** (FA2 supports ≤256 but occupancy drops). **VERIFY/M0**: bench FA2-style hd256 vs "two-pass 128+128 split" (treat head as two 128-dim halves sharing softmax — invalid for softmax? NO — softmax is over keys not dims, so splitting head_dim is safe: two independent hd128 attentions with the same softmax). Pick the faster in M0.
- **REF**: `sgl_kernel` flashinfer sm80 path (production uses flashinfer; unit file sets FLASHINFER flags) or triton FA2 in the venv. Port the scheduling (block sizes for 70 SMs) + add fp8 KV dequant.
- **Target**: decode ≤ 30% of step time; prefill ≥ 50% of M0 GEMM TFLOPS (attention is ~15–25% of prefill FLOPs).

### 7.5 K5 — GDN decode step (P0)

Single-token recurrence over 48 layers' states (§4.4.4): reads state [48,128,128] bf16 per layer (7.5 MB/layer-set… per layer: 48·128·128·2 B = 1.57 MB), writes updated state; q/k L2-norm; β, g in fp32; rank-1 delta update with fp32 accum then bf16 store. Grid: (48 layers × 48 heads) blocks, 128×128 tile in registers/shared per block — trivially parallel, latency-bound; target < 150 µs for 48 layers at bs=16.
- **REF**: `srt/layers/attention/linear/kernels/gdn_triton.py::decode` / `packed_decode`.
- **VERIFY:** head mapping hv→hk; exact decay/delta ordering; state dtype policy (bf16 store / fp32 accum — match production flag).

### 7.6 K6 — GDN extend (prefill, chunked) (P0, hardest)

Chunked gated-DeltaNet over variable-length segments (chunk 64): intra-chunk WY representation + inter-chunk state scan; produces per-token outputs and the final state. This is the **fla** algorithm.
- **REF**: `kernels/ops/attention/fla/chunk.py::chunk_gated_delta_rule_fwd`, `chunk_fwd.py`, `chunk_delta_h.py`, `l2norm.py`; SGLang's wrapper `gdn_triton.py::extend`.
- **Plan A (preferred)**: port to CUDA (the inter-chunk scan is sequential over ≤ 3125 chunks at 200K — keep it to a single block per (layer, head) with the chunk GEMMs on tensor cores).
- **Plan B (fallback, ship if Plan A slips 2+ weeks)**: compile the vendored Triton kernels once (into cubins on the target GPU host) and launch from C++ via the CUDA driver API (`cuModuleLoadData` + `cuLaunchKernel`). The runtime stays pure C++; the kernel ships as Triton output. Accept the ~5–10% Triton-vs-tuned gap; replace in M5.
- **Target**: prefill GDN chunk ≤ 1.5× the FA prefill per-token cost; must handle a 200K single sequence (3125 chunks).

### 7.7 K7 — GDN target_verify (P1)

Advance GDN states over the `commit_lens` committed tokens (variable per seq in the batch, 1–7 tokens) — the verify-mode recurrence. **REF**: `gdn_triton.py::target_verify`. Straightforward once K5 exists (loop the K5 step `commit_lens` times per seq, or one kernel with per-seq token lists).

### 7.8 K8 — depthwise causal conv k=4 (P0, small)

[10240, b] bf16, per-channel ring state [10240, 3]; output = Σₜ w[c,t]·x[c, pos−t]. One CTA per channel-group; trivial compute, bandwidth-bound. Fusable into K5/K6 in M5.

### 7.9 K9 — `silu_mul` + `rmsnorm_gated_swish` (P0, small)

- `silu_mul(xg, xu)` [b, 17408] → bf16.
- `rmsnorm_gated(o, z)`: per 128-dim head: (o/rms·w) ⊙ swish(z). **REF**: sgl-kernel `RMSNormGated` (swish = x·sigmoid(x)).

### 7.10 K10 — KV quantize-store + fp8 dequant-load (P0, small)

- Store: k/v bf16 [b, 1024] → fp8 e4m3 + per-token scale (max|·|/448, fp32) into paged pool slots. (One scale per token per K and per V — **VERIFY** SGLang's exact granularity in its fp8 KV path; match it.)
- Load: fused into K4 (dequant on the fly).

### 7.11 K11 — embed gather + lm_head + sampling (P0)

- Embed: bf16 row gather [b] → [b, 5120] (includes the DFlash mask-token row if the trained-mask merge applies — §4.10.1 VERIFY).
- lm_head: G8 (BF16 GEMV/GEMM; at verify b=7–112 → tensor-core path; the 1.27 GB weight read is bandwidth-bound — consider keeping a **fp8 copy of lm_head** to halve it: 248320×5120 fp8 = 635 MB, dequant in K1-style GEMV. **Decision M3:** worth ~+2–3 tok/s; do it if time permits, P2.)
- Sampler: greedy argmax (verify path) / temp+top20+top0.95 (final token) — radix-select top-k over 248320, fp32 logits from bf16 GEMM. **VERIFY** SGLarg's padding-tail masking semantics.

### 7.12 K12 — DFlash draft kernels (P1)

- `draft_qkv_norm_rope` (G9 + plain qk-norm + full RoPE), `draft_sliding_attn` (bidirectional, window 2048, hd 128, GQA 32:8, paged draft-KV + in-block bidirectional Q — **VERIFY** window semantics §4.9), `draft_grouped_conv` (2-tap dynamic, §4.9 — small; a per-token [320 groups] kernel; the reference is torch-compiled `_grouped_conv`, so we write it from the formula), `draft_fc` (G11, bf16), selector scoring (G12 + codebook dot — rank 256, tiny) + `selector_walk` (6 slots × 16 candidates; **REF** `kernels/ops/speculative/dflash.py::selector_walk_triton`), `accept_block` (greedy compare + bonus; sampling path per §4.8 note), `draft_kv_materialize` (per-layer `kv_proj_only` over projected hiddens — GEMM + K10-style bf16 store; **REF** `_append_target_hidden_fused`/`_by_loc`).

### 7.13 Misc (P0/P1)

`copy`, `fill`, `rand_uniform` (sampling), `block_table` updates (CPU-side, trivial), `positions` computation (CPU). No exotic sync — everything else is streams + events.

### 7.4 Kernel development rules

1. Every kernel ships with: (a) a CPU/torch reference in `ref/`, (b) a standalone bench in `bench/`, (c) tolerance + perf targets from this file.
2. No kernel is "done" until it hits its §7 target on the target GPU with nsys-confirmed bandwidth/TFLOPS.
3. Determinism: fixed reduction orders where feasible (tests §9.5 need bit-identical reruns at fixed seed — at minimum, same-binary reruns must agree bit-exact).

---

## 8. Weight loader

1. Parse each `layers-N.safetensors` header (8-byte LE size + JSON), mmap the data; resolve §5.4 packing; H2D via pinned staging (8 MB chunks, 3-stream async overlap of file-read → pack → copy). Target: **< 90 s total** for 29 GB (NFS — measure first; `crc32.txt` check optional, behind a flag).
2. Name mapping (checkpoint → engine): `model.language_model.layers.N.{...}` → layer N; `lm_head.weight`, `embed_tokens`, `norm` from `outside.safetensors`; skip `model.visual.*` and `mtp.safetensors` (log skipped bytes — expected ~0.9 GB vision + 0.48 GB MTP).
3. Scale handling: `weight_scale_inv` [K/128, N/128] per §4.2; for fused projections (G1/G3) merge the scale grids per §5.4. **VERIFY** at M1 that scales are per-(K-block, N-block) exactly as assumed (probe: dequant one tile, compare to `w·s` — §9.1 stage 0).
4. Draft: load DFlash2 (3.6 GB) including `predecessor_codebook`, `successor_codebook` (frozen; assert `requires_grad=False` semantics — just don't touch), `fc`, `base_kernel`, `kernel_projection`.
5. Post-load self-test: random 32-token probe → compare engine layer-0/layer-3 outputs to a CPU dequant reference (§9.1 stage 0); abort boot on mismatch > tolerance.

---

## 9. Validation & test protocol

### 9.1 Numerics (every milestone, automated on the target GPU)

- **Stage 0 (load)**: dequant probe dot-products (loader self-test).
- **Stage 1 (per-layer)**: `ref/torch_ref.py` — pure-torch BF16 reference of one GDN layer + one full-attn layer (dequantized weights, plain loops over the §4 math). Feed the same 128-token segment through engine kernel path vs torch: max |Δ| on layer outputs ≤ 1e-2 (bf16 accumulation differences). **This stage exists to catch the GemmaRMSNorm `+1`, RoPE partial-dim, q/k L2norm, and GQA-mapping bugs — the four silent killers.**
- **Stage 2 (full model)**: greedy decode 200 tokens on 5 fixed prompts (1K, 32K ctx). Compare (a) per-position logit top-1 agreement ≥ 99.9%, (b) mean |logit diff| over top-32 < 0.05, against the **live SGLang** (`/v1/completions`, `logprobs`) in the same test window.
- **Stage 3 (spec decode)**: same prompts with DFLASH: accept-len histogram within ±15% of SGLang's on identical prompts; committed-token stream identical to SGLang's greedy stream (must be exact for greedy).

### 9.2 Quality gate

Perplexity on a fixed 100-prompt corpus (e.g. a slice of C4 + the model's own docs) within **+2%** of SGLang; no degenerate outputs (repetition, truncated mid-thought) in a 200-completion soak.

### 9.3 Performance gate

- Single-stream: 32K ctx, 1024 output tokens, 5 runs: **mean tok/s ≥ 95** (S1), p95 step latency, accept len.
- TTFT: 8K and 32K prompts vs SGLang (±10%).
- Batched: 8 and 16 concurrent 8K→512 requests: aggregate tok/s, per-request TPOT, no OOM, queueing sane.
- nsys traces archived per milestone (GPU-busy%, step breakdown, HBM throughput vs M0 roofline).

### 9.4 Robustness

200K single request (KV 6.4 GB + GDN 10.5 MB — fits); pool exhaustion → 503 + queue (no crash, no state corruption); client disconnect mid-stream (clean slot release); max_tokens=1; empty message (400); concurrent boot (no double-load).

### 9.5 Determinism

Fixed-seed sampling: two runs of the same binary → bit-identical token streams. (If a kernel can't be made bit-stable, record which and use the greedy golden set for that layer.)

---

## 10. Build & repo layout

```
amperegen/
  CMakeLists.txt            # C++20, CUDA 13, -arch=sm_80, -O3
  cpp/
    main.cpp                # config (paths, pools, max-running-requests, port), boot
    runtime/                # CudaCtx, Streams, MempoolArena, GraphManager, Events
    loader/                 # safetensors.hpp/.cpp (parser+packer), draft_loader
    model/                  # LayerDesc table (64 entries), TargetForward, DraftForward,
                            #   SpecLoop (draft→verify→accept→update orchestration)
    scheduler/              # Scheduler, PagedKVManager, GdnStatePool, DraftKVPool,
                            #   Request, Batch (decode-verify + extend segments)
    kernels/                # one .cu/.cuh per §7 kernel + .cuh headers with exact
                            #   signatures; bench hooks (#ifdef AMPEREGEN_BENCH)
    server/                 # HttpServer (cpp-httplib), SseWriter, Metrics
    sidecar_client/         # unix-socket JSON client (tokenize/detokenize)
  sidecar/
    app.py                  # uvicorn: /tokenize, /detokenize, /render_chat
                            #   (HF tokenizers + jinja2, checkpoint's template)
  bench/                    # M0 microbenchmarks + per-kernel benches (K1..K12)
  ref/                      # torch_ref.py (per-layer), golden prompts, corpus
  tools/                    # load_weights.py (debug), logprob_diff.py (vs SGLang),
                            #   soak.py, bench_client.py (OpenAI-protocol load gen)
  docs/                     # this file, M0-results.md, decision log updates
```

- **Third-party (vendored)**: cpp-httplib (single header), nlohmann/json (single header). Nothing else — no pybind, no glog.
- **Flags**: `-arch=sm_80 -O3 --use_fast_math` **except** in gate/decay kernels (softplus/exp in K5/K6 must stay IEEE-accurate; `--use_fast_math` only on attention softmax paths, and A/B it).
- **Debug**: `-DAMPEREGEN_KERNEL_ASSERTS` (boundary checks), env `AMPEREGEN_LOG_STEPS=1` (per-step CPU timeline print).
- Build target: `ninja -C build` on the GPU host (~1–2 min full; per-kernel incremental < 10 s).

---

## 11. Milestones & acceptance

| # | Deliverable | Exit criteria (all must pass) |
|---|---|---|
| **M0** (2–3 d) | `bench/` microbench suite on target GPU: (1) d2d memcpy + 29 GB-stream bandwidth; (2) BF16 TFLOPS at 210 W for G1–G7 shapes, b ∈ {1, 7, 16, 64, 8192}; (3) fp8-dequant-GEMM prototype (G1, b=1) vs cuBLASLt-BF16-on-dequant; (4) nsys profile of **one live SGLang decode step** (GPU-busy%, CPU gap, per-kernel time) — done in a quiet window, non-intrusive attach; (5) hd256-vs-split attention probe | `docs/M0-results.md`: measured roofline table, beatable-gap breakdown, K1/G4 design locked, Plan-A-vs-B call for K6 |
| **M1** (wk 1–2) | Runtime skeleton + loader + K1–K3, K9–K11 (no attention/GDN yet): embed → per-layer GEMMs+norms with **torch-emulated attention/GDN stubs** | Stage-0 probe passes; load < 90 s; per-layer GEMM bench ≥ 90% of M0 stream BW at b=1 |
| **M2** (wk 2–4) | K4, K5, K6, K8 + full forward + greedy single-request decode (no spec) | Stage-1 per-layer ≤ 1e-2; Stage-2 full-model gate; single-stream tok/s ≈ 40–48 (bandwidth check: within 15% of M0 roofline without spec); 200K ctx runs |
| **M3** (wk 4–6) | DFlash port (K12 + SpecLoop) + CUDA graphs | Stage-3 gate; **S1: ≥ 95 tok/s sustained** (32K ctx); graphs active at bs buckets; accept len within ±15% of SGLang |
| **M4** (wk 6–8) | Continuous batching (16 req), chunked prefill mixed batches, server + sidecar, metrics, robustness suite | S3, S4, S5; §9.4 all pass; A/B shadow window vs SGLang (both serve the same recorded traffic, outputs compared) |
| **M5** (wk 8+) | Opt pass: lm_head fp8, GDN pure-CUDA swap (if B was shipped), per-phase clock policy (approved only), KV prefix caching (full-attn layers only), adaptive draft length, prefill tuning | Final S1/S4 margin report; production cutover decision packet |

**Decision points**: M0 end (K6 plan A/B, hd256 strategy), M3 start (DFlash vs MTP fallback, §4.11), M4 end (cutover).

---

## 12. Decisions & open questions

### 12.1 Decisions (locked)

1. **Weights stay FP8 in VRAM; fused dequant GEMM** (sm80 has no FP8 TC; 54 GB BF16 won't fit). §4.2, §7.1.
2. **GDN SSM state in bf16** (matches production `--mamba-ssm-dtype bfloat16`), fp32 accumulation in kernels.
3. **KV cache FP8 e4m3** (matches production), one per-token scale per K/V (§5.3, K10).
4. **Text-only v1** (vision tower skipped at load; endpoint rejects image content with 400 until v2).
5. **Pure C++ core; Python sidecar only for template+tokenize** (per-request, not per-token). §6.1.
6. **DFlash port is the spec path; MTP is the fallback** (§4.11, decision at M3 start).
7. **No prefix caching in v1** (KV block free list is simple; mamba prefix reuse is out of scope — production's `extra_buffer` strategy is a later optimization).
8. **210 W cap respected**; no power-limit changes without operator sign-off.
9. **Draft borrows target embed/lm_head** — one GPU copy of each.

### 12.2 Open questions (need operator/user)

1. **Test windows** — schedule for service-stop windows (M2 bring-up, M3 graphs, M4 A/B). Suggestion: nightly 02:00–05:00 + on-request.
2. **Cutover** — shadow A/B on a spare port during windows, then hard cutover; or side-by-side with load splitting? (SGLang binds 8000; engine would take e.g. 8001 during A/B.)
3. **Per-phase clock policy** (higher clocks for prefill bursts under the same 210 W) — allowed as an M5 experiment?
4. **Traffic profile** — what's the real request mix (ctx lengths, concurrency, tool-call share)? The A/B benchmark set should mirror it; ask for a week of `/metrics` scrape or access log.

### 12.3 Risks (top 3)

1. **K6 chunked GDN port** (2–4 weeks, hardest kernel) — mitigated by Plan B (Triton cubin via driver API).
2. **hd256 flash attention on sm80** register pressure — mitigated by M0 split-head probe.
3. **Single live GPU** — any test needs a service-stop window; the M2–M4 schedule depends on §12.2.1 being granted.

### 12.4 `VERIFY:` register (resolve in the stated milestone; record answers here)

| # | Item | Where | Milestone |
|---|---|---|---|
| V1 | q/k L2-norm flag for this model's GDN | `gdn_backend.py` → `chunk_gated_delta_rule_fwd(use_qk_l2norm_in_kernel=...)` | M1 |
| V2 | GDN head mapping hv→hk (`% 16` vs `// 3`) | `gdn_triton.py` | M1 |
| V3 | Exact gated-delta operation order (decay vs delta) | `fla/fused_recurrent...` via `gdn_triton.py::decode` | M1 |
| V4 | Chunk size + inter-chunk scan variant for prefill | `gdn_triton.py::extend` | M2 |
| V5 | KV fp8 scale granularity (per-token vs per-head) | SGLang fp8 KV path (`mem_cache`/kv backend) | M2 |
| V6 | DFlash draft window semantics (bidirectional 2048: trailing vs centered) | `RadixAttention` sliding window + `get_dflash_layer_attention_params` | M3 |
| V7 | Trained mask-embedding merge (`mask_token_id 248070`) | `_maybe_merge_trained_mask_embedding` | M3 |
| V8 | Aux-hidden capture point (after which op in the layer) | `layer_communicator.prepare_attn_and_capture_last_layer_outputs` | M1 |
| V9 | Sampling-verify: adjusted (DSpark) vs unadjusted gap on our traffic | measure in M3 | M3 |
| V10 | lm_head padding-tail masking in sampling | `LogitsProcessor` / org-vocab handling | M2 |
| V11 | Block size resolution: 6 draft + 1 anchor = 7 (flag overrides ckpt 8) | `set_block_size` call site | M3 |

---

## 13. Reference material index (SGLang 0.5.20)

Root: `R = <SGLANG_SITE_PACKAGES>/sglang`

| What | Path |
|---|---|
| Target text tower (wrappers, aux capture, DFlash hook) | `R/srt/models/qwen3_5_text.py` |
| Target body: GDN class (322–929), linear layer (929–1094), attention layer (1094–1550), body forward (1750–1882) | `R/srt/models/qwen3_5.py` |
| MTP fallback drafter | `R/srt/models/qwen3_5_mtp.py` |
| **DFlash draft model** (layers, grouped conv, selector, weight loading) | `R/srt/models/dflash.py` |
| **DFlash worker** (main loop 2184+, `_accept_block` 2056, `_propose_selector_block` 1360, draft-KV append 1702+) | `R/srt/speculative/dflash_worker_v2.py` |
| DFlash config parsing / helpers | `R/srt/speculative/dflash_utils.py`, `dflash_info.py` |
| Selector walk kernel (triton) | `R/kernels/ops/speculative/dflash.py` |
| **GDN kernels** (`decode`, `packed_decode`, `extend`, `target_verify`) | `R/srt/layers/attention/linear/kernels/gdn_triton.py` |
| GDN backend dispatch | `R/srt/layers/attention/linear/gdn_backend.py` |
| fla chunked algorithms | `R/kernels/ops/attention/fla/{chunk,chunk_fwd,chunk_delta_h,l2norm}.py` |
| Linear-attention module wrapper (state layout) | `R/srt/layers/radix_linear_attention.py` |
| FP8 block-quant method (scale layout, w8a8) | `R/srt/layers/quantization/fp8*.py` |
| GemmaRMSNorm / RMSNormGated / fused qk-norm-rope-gate | `R/srt/layers/layernorm.py`, sgl-kernel triton sources |
| KV cache fp8 + pool management | `R/srt/mem_cache/` |
| Sampling (top-k/p, DSpark accept) | `R/srt/layers/sampling/`, `R/srt/speculative/` (accept_sampling) |
| Chat template + generation config | checkpoint dir (§2.2) |
| Production unit (flags, env, power) | `/etc/systemd/system/sglang-qwen.service` |

**Licensing**: SGLang is Apache-2.0; porting/adapting its kernels is permitted with attribution — keep a `THIRD_PARTY.md` crediting SGLang Team (Apache-2.0) for ported kernel code.
