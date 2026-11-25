#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <cstdint>

namespace amperegen {

// ============================================================================
// K1: FP8 Block-128x128 Weight-Only Dequant GEMM / GEMV (sm_80 Tensor Cores)
// ============================================================================
struct Fp8LinearWeight {
    uint4* w_packed = nullptr;      // [N/16, K/32, 32] warp-MMA packed FP8 (for M <= 32 fast TC path)
    uint8_t* w_row = nullptr;       // Optional row-major FP8 (nullptr to save VRAM — we unpack directly from w_packed!)
    float* scales_raw = nullptr;    // [N/128, K/128] FP32 raw scale_inv (for prefill dequant)
    float* scales_fused = nullptr;  // [N/128, K/128] FP32 scale_inv * 2^120 (for decode TC)
    int N = 0;
    int K = 0;
};

// Pack row-major FP8 [N, K] into warp-MMA layout [N/16, K/32, 32] on GPU.
void launch_pack_fp8_mma(
    const uint8_t* d_w_row,
    uint4* d_w_packed,
    int N, int K,
    cudaStream_t stream);

// Convert BF16 weight [N, K] on GPU into FP8 block-128x128 packed + scales (used for lossless DFlash2 draft compression!)
void launch_quantize_bf16_to_fp8_mma(
    const __nv_bfloat16* d_w_bf16,
    uint4* d_w_packed,
    float* d_scales_raw,
    float* d_scales_fused,
    int N, int K,
    cudaStream_t stream);

// Convert BF16 scale_inv [N/128, K/128] into FP32 raw & fused (scale * 2^120)
void launch_prepare_fp8_scales(
    const __nv_bfloat16* d_scales_bf16,
    float* d_scales_raw,
    float* d_scales_fused,
    int num_scales,
    cudaStream_t stream);

// Execute Y[M, N] = X[M, K] * W[N, K]^T
// Automatically dispatches:
//   - M <= 32: custom fused FP8-dequant m16n8k16 Tensor Core GEMM (zero BF16 weight materialization!)
//   - M > 32:  fast unpack+dequant from w_packed -> d_bf16_scratch [N, K] + cuBLAS BF16 Tensor Core GEMM!
void launch_fp8_linear(
    cublasHandle_t cublas,
    const __nv_bfloat16* X,
    const Fp8LinearWeight& W,
    __nv_bfloat16* Y,
    float* d_splitk_scratch,        // >= 32 * maxN * sizeof(float)
    __nv_bfloat16* d_bf16_scratch,  // >= max(N*K) * sizeof(bf16) (~356 MB for gate_up) for prefill M > 32
    int M,
    cudaStream_t stream);

// Dense BF16 linear via cuBLAS: Y[M, N] = X[M, K] * W[N, K]^T (row-major)
void launch_bf16_linear(
    cublasHandle_t cublas,
    const __nv_bfloat16* X,
    const __nv_bfloat16* W,
    __nv_bfloat16* Y,
    int M, int N, int K,
    cudaStream_t stream,
    bool accumulate = false);

// ============================================================================
// K2 / K3 / K9 / K11: Norms, RoPE, Activations, Embeddings
// ============================================================================
// Precompute cos/sin table [max_pos, rotary_dim/2] (two separate float arrays)
void launch_init_rope_table(
    float* d_cos,
    float* d_sin,
    int max_pos,
    int rotary_dim,
    float theta,
    cudaStream_t stream);

// Embed lookup: out[t, :] = embed_table[token_ids[t], :]
void launch_embed_lookup(
    const int32_t* token_ids,
    const __nv_bfloat16* embed_table,
    __nv_bfloat16* out,
    int num_tokens,
    int hidden_size,
    cudaStream_t stream);

// RMSNorm (gemma_plus_one = true for (1+w), false for plain w)
void launch_rmsnorm(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    int num_tokens,
    int hidden_size,
    float eps,
    bool gemma_plus_one,
    cudaStream_t stream);

// Fused residual add + RMSNorm:
//   residual[t] = x[t] + residual[t]
//   out[t]      = rmsnorm(residual[t], weight)
void launch_fused_add_rmsnorm(
    const __nv_bfloat16* x,
    __nv_bfloat16* residual,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    int num_tokens,
    int hidden_size,
    float eps,
    bool gemma_plus_one,
    cudaStream_t stream);

// Fused QKV-gate prepare + QK GemmaRMSNorm + Partial NeoX RoPE + FP8 KV cache write (Full-Attn layers)
// Input:  qkv [T, 14336] = [q_gate(12288) | k(1024) | v(1024)]
// Output: q_out [T, 24, 256] BF16, gate_out [T, 24, 256] BF16
//         and writes K, V quantized to FP8 E4M3 (scale=1.0) into paged KV pool at slot_mapping[t]!
//         If k_cache_layer == nullptr, instead writes BF16 k_out [T,4,256], v_out [T,4,256] (for unit test).
void launch_full_attn_qkv_prep_and_kv_write(
    const __nv_bfloat16* qkv,          // [T, 14336]
    const __nv_bfloat16* q_norm_w,     // [256]
    const __nv_bfloat16* k_norm_w,     // [256]
    const float* rope_cos,             // [max_pos, 32]
    const float* rope_sin,             // [max_pos, 32]
    const int32_t* positions,          // [T]
    __nv_bfloat16* q_out,              // [T, 24, 256]
    __nv_bfloat16* gate_out,           // [T, 24, 256]
    __nv_fp8_e4m3* k_cache_layer,      // [num_blocks, 16, 4, 256] (or nullptr)
    __nv_fp8_e4m3* v_cache_layer,      // [num_blocks, 16, 4, 256] (or nullptr)
    const int32_t* slot_mapping,       // [T] physical token slot = block_id * 16 + block_offset
    int T,
    float eps,
    cudaStream_t stream);

// Fused attention output gating: out[i] = attn[i] * sigmoid(gate[i])
void launch_attn_output_gate(
    const __nv_bfloat16* attn,
    const __nv_bfloat16* gate,
    __nv_bfloat16* out,
    int num_elems,
    cudaStream_t stream);

// SwiGLU activation: out[t, i] = silu(gate_up[t, i]) * gate_up[t, i + inter]
void launch_silu_mul(
    const __nv_bfloat16* gate_up,  // [T, 2 * inter]
    __nv_bfloat16* out,            // [T, inter]
    int T,
    int inter,
    cudaStream_t stream);

// GDN Gated RMSNorm (plain RMSNorm per 128-dim head, NO +1, multiplied by silu(z))
// o: [T, 48, 128] BF16, z: [T, 48, 128] BF16 (strided or contiguous from qkvz[:, 10240:])
void launch_gdn_rmsnorm_gated(
    const __nv_bfloat16* o,        // [T, 6144]
    const __nv_bfloat16* qkvz,     // [T, 16384], z starts at col 10240
    const __nv_bfloat16* weight,   // [128]
    __nv_bfloat16* out,            // [T, 6144]
    int T,
    float eps,
    cudaStream_t stream);

// ============================================================================
// K4: Full-Attention Paged FP8-KV FlashAttention (head_dim=256, GQA 24:4)
// ============================================================================
// Decode / Verify attention (each sequence has q_len in [1..8] query tokens attending to seq_len tokens)
// Supports fused output gate (`attn * sigmoid(gate)`) right in the epilogue!
void launch_paged_attn_decode_fp8(
    const __nv_bfloat16* q,            // [B * q_len, 24, 256]
    const __nv_bfloat16* gate,         // [B * q_len, 24, 256] (optional, applied in epilogue if non-null)
    const __nv_fp8_e4m3* k_cache,      // [num_blocks, 16, 4, 256]
    const __nv_fp8_e4m3* v_cache,      // [num_blocks, 16, 4, 256]
    const int32_t* block_tables,       // [B, max_blocks_per_seq]
    const int32_t* seq_lens,           // [B] total length INCLUDING current q_len tokens
    __nv_bfloat16* out,                // [B * q_len, 24, 256]
    float* splitk_out,                 // workspace for FlashDecoding split-K: [B * q_len * 24 * max_splits * 256]
    float* splitk_lse,                 // workspace: [B * q_len * 24 * max_splits * 2] (m, l)
    int B,
    int q_len,
    int max_blocks_per_seq,
    int max_seq_len_hint,              // CPU hint for choosing num_splits (can be upper bound for CUDA graph)
    cudaStream_t stream);

// Prefill / Extend causal attention (single sequence or batched varlen with prefix in paged FP8 KV cache)
void launch_paged_attn_prefill_fp8(
    const __nv_bfloat16* q,            // [total_q, 24, 256]
    const __nv_bfloat16* gate,         // [total_q, 24, 256]
    const __nv_fp8_e4m3* k_cache,      // [num_blocks, 16, 4, 256]
    const __nv_fp8_e4m3* v_cache,      // [num_blocks, 16, 4, 256]
    const int32_t* block_tables,       // [B, max_blocks_per_seq]
    const int32_t* cu_seqlens_q,       // [B + 1]
    const int32_t* kv_lens,            // [B] total KV len (prefix + q_len)
    __nv_bfloat16* out,                // [total_q, 24, 256]
    int B,
    int max_q_len,
    int max_blocks_per_seq,
    cudaStream_t stream);

// ============================================================================
// K5 / K6 / K7 / K8: Gated DeltaNet (Conv1d + Decode + Verify + Chunked Prefill)
// ============================================================================
// Causal depthwise Conv1d k=4 + SiLU for prefill/extend (updates conv_state[slot, 10240, 3])
// Reads mixed_qkv from qkvz[:, :10240] (stride 16384), writes conv_out [total_T, 10240] BF16.
void launch_gdn_conv1d_prefill(
    const __nv_bfloat16* qkvz,         // [total_T, 16384]
    const __nv_bfloat16* conv_w,       // [10240, 4]
    __nv_bfloat16* conv_states,        // [num_slots, 10240, 3]
    const int32_t* state_slots,        // [B]
    const int32_t* cu_seqlens,         // [B + 1]
    __nv_bfloat16* conv_out,           // [total_T, 10240]
    int B,
    int total_T,
    cudaStream_t stream);

// Causal depthwise Conv1d k=4 + SiLU for decode / spec verify (q_len tokens per seq).
// If save_intermediate == true (spec verify), saves conv state after each step 0..q_len-1 into
// conv_inter [B, q_len, 10240, 3] and does NOT mutate conv_states yet (commit happens after accept!).
void launch_gdn_conv1d_decode_verify(
    const __nv_bfloat16* qkvz,         // [B * q_len, 16384]
    const __nv_bfloat16* conv_w,       // [10240, 4]
    __nv_bfloat16* conv_states,        // [num_slots, 10240, 3]
    __nv_bfloat16* conv_inter,         // [B, q_len, 10240, 3] (nullable if commit_inplace)
    const int32_t* state_slots,        // [B]
    __nv_bfloat16* conv_out,           // [B * q_len, 10240]
    int B,
    int q_len,
    bool commit_inplace,
    cudaStream_t stream);

// GDN single-token or multi-token (spec verify q_len=1..8) recurrence
// State layout: ssm_states [num_slots, 48, 128, 128] BF16 (accumulated in FP32 inside kernel!)
// If commit_inplace == false (spec verify): writes per-step intermediate states to
//   ssm_inter [B, q_len, 48, 128, 128] BF16 without overwriting ssm_states!
void launch_gdn_recurrent_decode_verify(
    const __nv_bfloat16* conv_out,     // [B * q_len, 10240]
    const __nv_bfloat16* ba,           // [B * q_len, 96] (b[0..47], a[48..95])
    const float* A_log,                // [48]
    const float* dt_bias,              // [48]
    __nv_bfloat16* ssm_states,         // [num_slots, 48, 128, 128]
    __nv_bfloat16* ssm_inter,          // [B, q_len, 48, 128, 128] (nullable if commit_inplace)
    const int32_t* state_slots,        // [B]
    __nv_bfloat16* out,                // [B * q_len, 48, 128]
    int B,
    int q_len,
    bool commit_inplace,
    cudaStream_t stream);

// Replay & commit accepted spec-verify states across ALL 48 GDN layers in ONE kernel launch!
// Avoids allocating 9.6 GB of ssm_inter VRAM AND avoids writing intermediate SSM states during verify!
// During verify, only conv_out [B*q_len, 10240], ba [B*q_len, 96], and conv_inter [B, q_len, 10240, 3]
// are saved per layer (~0.5 GB total for B=16, q_len=8). After accept, this kernel replays ONLY
// commit_lens[b] steps in-place on ssm_states[slot] and copies conv_inter[step] -> conv_states[slot]!
void launch_gdn_commit_replay_states(
    __nv_bfloat16* const* ssm_states_ptrs,   // device array [48]
    __nv_bfloat16* const* conv_states_ptrs,  // device array [48]
    __nv_bfloat16* const* conv_inter_ptrs,   // device array [48]
    const __nv_bfloat16* const* conv_out_ptrs, // device array [48]
    const __nv_bfloat16* const* ba_ptrs,     // device array [48]
    const float* const* A_log_ptrs,          // device array [48]
    const float* const* dt_bias_ptrs,        // device array [48]
    const int32_t* state_slots,              // [B]
    const int32_t* commit_lens,              // [B] values in [1..q_len]
    int B,
    int q_len,
    cudaStream_t stream);

// GDN Chunk-64 Prefill/Extend (WY representation + inter-chunk scan, exact to step-by-step!)
struct GdnPrefillWorkspace {
    float* g_cum = nullptr;          // [max_T * 48]
    __nv_bfloat16* q_norm = nullptr; // [max_T * 16 * 128]
    __nv_bfloat16* k_norm = nullptr; // [max_T * 16 * 128]
    __nv_bfloat16* w = nullptr;      // [max_T * 48 * 128]
    __nv_bfloat16* u = nullptr;      // [max_T * 48 * 128]
    __nv_bfloat16* v_new = nullptr;  // [max_T * 48 * 128]
    __nv_bfloat16* h_chunks = nullptr; // [max_chunks * 48 * 128 * 128]
    int max_T = 0;
};

void launch_gdn_chunked_prefill(
    const __nv_bfloat16* conv_out,     // [total_T, 10240]
    const __nv_bfloat16* ba,           // [total_T, 96]
    const float* A_log,                // [48]
    const float* dt_bias,              // [48]
    __nv_bfloat16* ssm_states,         // [num_slots, 48, 128, 128]
    const int32_t* state_slots,        // [B]
    const int32_t* cu_seqlens,         // [B + 1]
    __nv_bfloat16* out,                // [total_T, 48, 128]
    GdnPrefillWorkspace& ws,
    int B,
    int total_T,
    cudaStream_t stream);

// ============================================================================
// K12: DFlash2 Draft & Speculative Verification Kernels
// ============================================================================
// Dynamic 2-tap grouped conv along block positions (block_size tokens per seq)
// base_kernel: [2, 5120] BF16 for this side (0=prepare, 1=finish)
// delta: [B * block_size, 2, 320] BF16 (from kernel_projection split)
void launch_dflash_grouped_conv(
    const __nv_bfloat16* x,            // [B * block_size, 5120]
    const __nv_bfloat16* delta_all,    // [B * block_size, 1280] (side0: [0..639], side1: [640..1279])
    const __nv_bfloat16* base_kernel,  // [2, 5120] for this side
    __nv_bfloat16* out,                // [B * block_size, 5120]
    int B,
    int block_size,
    int side,
    cudaStream_t stream);

// Draft QK plain-RMSNorm + full 128-dim NeoX RoPE (in-place on qkv [N, 6144])
void launch_dflash_qk_norm_rope(
    __nv_bfloat16* qkv,                // [N, 6144] -> q[4096] | k[1024] | v[1024]
    const __nv_bfloat16* q_norm_w,     // [128]
    const __nv_bfloat16* k_norm_w,     // [128]
    const float* rope_cos,             // [max_pos, 64]
    const float* rope_sin,             // [max_pos, 64]
    const int32_t* positions,          // [N]
    int N,
    float eps,
    cudaStream_t stream);

// Draft KV-only K-norm + RoPE + ring-buffer append into draft KV pool [num_slots, 2048, 8, 128] BF16
void launch_dflash_append_ctx_kv(
    const __nv_bfloat16* kv,           // [total_tokens, 2048] (k[1024] | v[1024])
    const __nv_bfloat16* k_norm_w,     // [128]
    const float* rope_cos,             // [max_pos, 64]
    const float* rope_sin,             // [max_pos, 64]
    const int32_t* positions,          // [total_tokens]
    const int32_t* slot_ids,           // [total_tokens] draft slot for each token
    const int32_t* commit_lens,        // [B] (nullable; if non-null, skips col >= commit_lens[b]!)
    __nv_bfloat16* draft_k_cache,      // [num_slots, 2048, 8, 128]
    __nv_bfloat16* draft_v_cache,      // [num_slots, 2048, 8, 128]
    int total_tokens,
    int block_size,
    float eps,
    cudaStream_t stream);

// Draft bidirectional sliding-window attention (32 Q / 8 KV heads, hd=128)
// Attends over [ctx in ring buffer (len = min(prefix_len, 2048))] + [current block (block_size tokens)]!
void launch_dflash_bidirectional_attn(
    const __nv_bfloat16* qkv,          // [B * block_size, 6144]
    const __nv_bfloat16* draft_k_cache,// [num_slots, 2048, 8, 128]
    const __nv_bfloat16* draft_v_cache,// [num_slots, 2048, 8, 128]
    const int32_t* draft_slots,        // [B]
    const int32_t* prefix_lens,        // [B]
    __nv_bfloat16* out,                // [B * block_size, 4096]
    float* split_out,                  // [B * block_size, 32, 16, 128]
    float* split_lse,                  // [B * block_size, 32, 16, 2]
    int B,
    int block_size,
    cudaStream_t stream);

// CandidateSelector top-16 extraction + lattice scoring + path walk (fused for single-graph capture!)
void launch_dflash_topk16(
    const __nv_bfloat16* logits,       // [M, vocab_size]
    int32_t* topk_ids,                 // [M, 16]
    float* topk_vals,                  // [M, 16]
    int M,
    int vocab_size,
    cudaStream_t stream);

void launch_dflash_selector_walk(
    const int32_t* candidate_ids,      // [B, num_pred, 16]
    const float* unary_logits,         // [B, num_pred, 16]
    const __nv_bfloat16* proj_hidden,  // [B, num_pred, 256]
    const __nv_bfloat16* pred_cb,      // [vocab, 256]
    const __nv_bfloat16* succ_cb,      // [vocab, 256]
    const int32_t* anchor_ids,         // [B]
    const float* temperatures,         // [B]
    const uint8_t* greedy_mask,        // [B]
    const float* uniforms,             // [B, num_pred]
    int32_t* out_block_ids,            // [B, block_size] (writes anchor at col 0, chosen drafts at 1..num_pred)
    float* out_q_rows,                 // [B, num_pred, 16] (optional)
    int B,
    int block_size,
    cudaStream_t stream);

// Greedy verify accept kernel (in-graph!)
void launch_dflash_accept_greedy(
    const int32_t* candidates,         // [B, block_size]
    const int32_t* target_top1,        // [B, block_size]
    int32_t* accept_lens,              // [B] in [0..block_size-1]
    int32_t* commit_lens,              // [B] = accept_lens + 1
    int32_t* bonus_ids,                // [B]
    int32_t* out_tokens,               // [B, block_size]
    int32_t* seq_lens,                 // [B] updated in-place: += commit_lens
    int B,
    int block_size,
    cudaStream_t stream);

// Exact unbiased rejection-sampling verify kernel (handles mixed greedy & temp/top-k/top-p batches!)
void launch_dflash_accept_sampling(
    const __nv_bfloat16* target_logits,// [B * block_size, vocab_size]
    const int32_t* candidates,         // [B, block_size]
    const int32_t* cand_ids,           // [B, block_size - 1, 16]
    const float* q_rows,               // [B, block_size - 1, 16]
    const float* temperatures,         // [B]
    const int32_t* top_ks,             // [B]
    const float* top_ps,               // [B]
    const uint64_t* rng_seeds,         // [B]
    int32_t* accept_lens,              // [B]
    int32_t* commit_lens,              // [B]
    int32_t* bonus_ids,                // [B]
    int32_t* out_tokens,               // [B, block_size]
    int B,
    int block_size,
    int vocab_size,
    cudaStream_t stream);

// ============================================================================
// Sampling: Greedy Argmax + Temp / Top-K / Top-P over 248,320 vocab
// ============================================================================
void launch_argmax(
    const __nv_bfloat16* logits,       // [M, vocab_size]
    int32_t* out_ids,                  // [M]
    int M,
    int vocab_size,
    cudaStream_t stream);

void launch_sample_topkp(
    const __nv_bfloat16* logits,       // [B, vocab_size]
    const float* temperatures,         // [B]
    const int32_t* top_ks,             // [B]
    const float* top_ps,               // [B]
    const uint64_t* rng_seeds,         // [B]
    int32_t* out_ids,                  // [B]
    int B,
    int vocab_size,
    cudaStream_t stream);

// Aux hidden capture gather (copies residual [T, 5120] into aux_buf [T, 5 * 5120] at slot_idx)
void launch_copy_aux_hidden(
    const __nv_bfloat16* residual,     // [T, 5120]
    __nv_bfloat16* aux_concat,         // [T, 25600]
    int T,
    int slot_idx,
    cudaStream_t stream);

// Prepare draft input block [anchor, MASK, ..., MASK] and positions on GPU (for CUDA graph!)
void launch_prepare_draft_block(
    const int32_t* anchor_ids,         // [B]
    const int32_t* prefix_lens,        // [B]
    int32_t* block_ids,                // [B, block_size]
    int32_t* positions,                // [B, block_size]
    int B,
    int block_size,
    int mask_token_id,
    cudaStream_t stream);

// Tiny utility kernels
void launch_bf16_add_inplace(
    __nv_bfloat16* dst,
    const __nv_bfloat16* src,
    size_t num_elems,
    cudaStream_t stream);

void launch_gather_hidden_rows(
    const __nv_bfloat16* src,
    const int32_t* indices,
    __nv_bfloat16* dst,
    int B,
    cudaStream_t stream);

void launch_slice_pred_hidden(
    const __nv_bfloat16* draft_hidden,
    __nv_bfloat16* pred_hidden,
    int B,
    int block_size,
    cudaStream_t stream);

// Hybrid prefix cache fast single-launch D2D state & KV block copies
void launch_copy_gdn_state_slot(
    __nv_bfloat16* const* ssm_states_ptrs,   // [48]
    __nv_bfloat16* const* conv_states_ptrs,  // [48]
    int dst_slot,
    int src_slot,
    cudaStream_t stream);

void launch_copy_kv_block(
    __nv_fp8_e4m3* const* k_cache_ptrs,      // [16]
    __nv_fp8_e4m3* const* v_cache_ptrs,      // [16]
    int dst_block,
    int src_block,
    cudaStream_t stream);

} // namespace amperegen
