#pragma once
#include "kernels/kernels.cuh"
#include "loader/model_loader.hpp"
#include "model/model_config.hpp"
#include "runtime/cuda_utils.cuh"
#include <cstdint>
#include <string>
#include <vector>

namespace amperegen {

struct TargetStatePools {
    GpuArena kv_arena;
    GpuArena gdn_arena;
    GpuArena verify_arena;

    // Paged FP8 KV cache for 16 full-attention layers:
    // Each layer: [num_kv_blocks, KV_BLOCK_SIZE=16, 4, 256] FP8 E4M3
    __nv_fp8_e4m3* k_cache[Qwen35Config::NUM_FULL_ATTN_LAYERS] = {};
    __nv_fp8_e4m3* v_cache[Qwen35Config::NUM_FULL_ATTN_LAYERS] = {};
    int num_kv_blocks = 0;

    // GDN persistent state pool for 48 linear layers:
    // ssm_states[l]: [max_slots, 48, 128, 128] BF16
    // conv_states[l]: [max_slots, 10240, 3] BF16
    __nv_bfloat16* ssm_states[Qwen35Config::NUM_GDN_LAYERS] = {};
    __nv_bfloat16* conv_states[Qwen35Config::NUM_GDN_LAYERS] = {};
    int max_state_slots = 0;

    // GDN spec-verify replay buffers (0.5 GB total for B=16, q_len=8!):
    // conv_out_verify[l]: [max_verify_bs * max_block_size, 10240] BF16
    // ba_verify[l]:       [max_verify_bs * max_block_size, 96] BF16
    // conv_inter[l]:      [max_verify_bs, max_block_size, 10240, 3] BF16
    __nv_bfloat16* conv_out_verify[Qwen35Config::NUM_GDN_LAYERS] = {};
    __nv_bfloat16* ba_verify[Qwen35Config::NUM_GDN_LAYERS] = {};
    __nv_bfloat16* conv_inter[Qwen35Config::NUM_GDN_LAYERS] = {};
    int max_verify_bs = 0;
    int max_block_size = 0;

    // Device pointer tables [48] for single-launch commit_replay & [16] for KV block copy!
    __nv_bfloat16** d_ssm_states_ptrs = nullptr;
    __nv_bfloat16** d_conv_states_ptrs = nullptr;
    __nv_bfloat16** d_conv_inter_ptrs = nullptr;
    __nv_bfloat16** d_conv_out_verify_ptrs = nullptr;
    __nv_bfloat16** d_ba_verify_ptrs = nullptr;
    float** d_A_log_ptrs = nullptr;
    float** d_dt_bias_ptrs = nullptr;
    __nv_fp8_e4m3** d_k_cache_ptrs = nullptr;
    __nv_fp8_e4m3** d_v_cache_ptrs = nullptr;
};

struct TargetWorkspace {
    GpuArena arena;
    int max_tokens = 0; // max prefill chunk (e.g. 8192)

    __nv_bfloat16* residual = nullptr;    // [max_tokens, 5120]
    __nv_bfloat16* norm_out = nullptr;    // [max_tokens, 5120]
    __nv_bfloat16* sub_out = nullptr;     // [max_tokens, 5120]
    __nv_bfloat16* qkv_buf = nullptr;     // [max_tokens, 16384]
    __nv_bfloat16* ba_buf = nullptr;      // [max_tokens, 96]
    __nv_bfloat16* conv_out = nullptr;    // [max_tokens, 10240]
    __nv_bfloat16* q_buf = nullptr;       // [max_tokens, 6144]
    __nv_bfloat16* gate_buf = nullptr;    // [max_tokens, 6144]
    __nv_bfloat16* attn_out = nullptr;    // [max_tokens, 6144]
    __nv_bfloat16* gate_up_buf = nullptr; // [max_tokens, 34816]
    __nv_bfloat16* mlp_mid = nullptr;     // [max_tokens, 17408]
    __nv_bfloat16* aux_hidden = nullptr;  // [max_tokens, 25600] (captured 5 target layers for DFlash2!)
    __nv_bfloat16* logits = nullptr;      // [max_logits_rows, 248320]
    int max_logits_rows = 0;

    // K1 split-K & prefill BF16 dequant scratch
    float* d_splitk_scratch = nullptr;       // [8 * 16 * 248320] float -> ~127 MB (or 8*16*34816 = 17.8 MB since lm_head uses sk=1!)
    __nv_bfloat16* d_bf16_w_scratch = nullptr; // [34816 * 5120] BF16 = 356 MB

    // FlashDecoding split-K workspace
    float* fa_splitk_out = nullptr;          // [max_verify_tokens * 24 * 64 * 256]
    float* fa_splitk_lse = nullptr;          // [max_verify_tokens * 24 * 64 * 2]

    // GDN chunked prefill workspace
    GdnPrefillWorkspace gdn_ws;

    // RoPE tables [MAX_POSITION_EMBEDDINGS, 32]
    float* rope_cos = nullptr;
    float* rope_sin = nullptr;
};

class Qwen35Model {
public:
    CudaContext ctx;
    TargetWeights weights;
    TargetStatePools pools;
    TargetWorkspace ws;

    void init_context();
    void init_workspace_and_pools(
        int max_prefill_chunk,
        int max_state_slots,
        int max_verify_bs,
        int max_block_size,
        size_t kv_pool_bytes);

    void reset_state_slot(int slot, cudaStream_t stream);
    void copy_gdn_state_slot(int dst_slot, int src_slot, cudaStream_t stream);
    void copy_kv_block(int dst_block, int src_block, cudaStream_t stream);

    // Run a single decoder layer (for Stage-1 verification!)
    void forward_layer_prefill(
        int layer_idx,
        __nv_bfloat16* residual, // [total_T, 5120], updated in-place!
        const int32_t* positions,
        const int32_t* slot_mapping,
        const int32_t* block_tables,
        const int32_t* cu_seqlens,
        const int32_t* kv_lens,
        const int32_t* state_slots,
        int B,
        int total_T,
        int max_q_len,
        int max_blocks_per_seq,
        cudaStream_t stream);

    // Full 64-layer Prefill / Extend forward:
    // Computes logits for the LAST token of each sequence (or all tokens if logits_all == true),
    // and optionally captures aux_hidden [total_T, 25600] into ws.aux_hidden for DFlash2!
    void forward_prefill(
        const int32_t* d_token_ids,     // [total_T]
        const int32_t* d_positions,     // [total_T]
        const int32_t* d_slot_mapping,  // [total_T]
        const int32_t* d_block_tables,  // [B, max_blocks_per_seq]
        const int32_t* d_cu_seqlens,    // [B + 1]
        const int32_t* d_kv_lens,       // [B]
        const int32_t* d_state_slots,   // [B]
        int B,
        int total_T,
        int max_q_len,
        int max_blocks_per_seq,
        bool capture_aux_hidden,
        const int32_t* d_last_token_indices, // [B] indices in [0..total_T-1] for logits (if non-null)
        __nv_bfloat16* d_logits_out,         // [B, 248320] (if non-null)
        cudaStream_t stream);

    // Full 64-layer Decode / Spec-Verify forward (100% CUDA-graph compatible!):
    // q_len = 1 for normal decode (commit_gdn_inplace = true)
    // q_len = block_size (6..8) for DFlash2 verify (commit_gdn_inplace = false; saves replay buffers!)
    // Writes logits [B * q_len, 248320] into d_logits_out, and optionally aux_hidden [B * q_len, 25600]!
    void forward_decode_verify(
        const int32_t* d_token_ids,     // [B * q_len]
        const int32_t* d_positions,     // [B * q_len]
        const int32_t* d_slot_mapping,  // [B * q_len]
        const int32_t* d_block_tables,  // [B, max_blocks_per_seq]
        const int32_t* d_seq_lens,      // [B] total length INCLUDING q_len tokens
        const int32_t* d_state_slots,   // [B]
        int B,
        int q_len,
        int max_blocks_per_seq,
        int max_seq_len_hint,
        bool commit_gdn_inplace,
        bool capture_aux_hidden,
        __nv_bfloat16* d_logits_out,    // [B * q_len, 248320]
        cudaStream_t stream);

    // Commit accepted GDN states after spec verify (in-graph!)
    void commit_gdn_verify(
        const int32_t* d_state_slots,
        const int32_t* d_commit_lens,
        int B,
        int q_len,
        cudaStream_t stream);

    void destroy();
};

} // namespace amperegen
