#pragma once
#include "kernels/kernels.cuh"
#include "loader/model_loader.hpp"
#include "model/model_config.hpp"
#include "model/qwen35_model.hpp"
#include "runtime/cuda_utils.cuh"

namespace amperegen {

struct DraftKVPool {
    GpuArena arena;
    int max_slots = 0;
    // Per layer: [max_slots, SLIDING_WINDOW=2048, 8, 128] BF16 (ring-indexed by pos & 2047!)
    __nv_bfloat16* k_cache[DFlash2Config::NUM_LAYERS] = {};
    __nv_bfloat16* v_cache[DFlash2Config::NUM_LAYERS] = {};
};

struct DraftWorkspace {
    GpuArena arena;
    int max_ctx_tokens = 0;   // e.g. 2048 for prefill materialization
    int max_draft_tokens = 0; // max_bs * max_block_size (e.g. 16 * 8 = 128)

    __nv_bfloat16* ctx_hidden = nullptr;   // [max_ctx_tokens, 5120]
    __nv_bfloat16* ctx_norm = nullptr;     // [max_ctx_tokens, 5120]
    __nv_bfloat16* ctx_kv = nullptr;       // [max_ctx_tokens, 2048]

    __nv_bfloat16* residual = nullptr;     // [max_draft_tokens, 5120]
    __nv_bfloat16* norm_out = nullptr;     // [max_draft_tokens, 5120]
    __nv_bfloat16* conv_buf = nullptr;     // [max_draft_tokens, 5120]
    __nv_bfloat16* conv_delta = nullptr;   // [max_draft_tokens, 1280]
    __nv_bfloat16* sub_out = nullptr;      // [max_draft_tokens, 5120]
    __nv_bfloat16* qkv_buf = nullptr;      // [max_draft_tokens, 6144]
    __nv_bfloat16* attn_out = nullptr;     // [max_draft_tokens, 4096]
    __nv_bfloat16* gate_up_buf = nullptr;  // [max_draft_tokens, 34816]
    __nv_bfloat16* mlp_mid = nullptr;      // [max_draft_tokens, 17408]

    // CandidateSelector buffers
    __nv_bfloat16* pred_hidden = nullptr;  // [max_bs * (block_size-1), 5120]
    __nv_bfloat16* proj_hidden = nullptr;  // [max_bs * (block_size-1), 256]
    __nv_bfloat16* cand_logits = nullptr;  // [max_bs * (block_size-1), 248320]
    int32_t* topk_ids = nullptr;           // [max_bs * (block_size-1), 16]
    float* topk_vals = nullptr;            // [max_bs * (block_size-1), 16]
    float* q_rows = nullptr;               // [max_bs * (block_size-1), 16] (selector q probs for rejection sampling)

    // Draft Split-K attention scratch ([max_draft_tokens, 32, 16, 128] + lse)
    float* attn_split_out = nullptr;
    float* attn_split_lse = nullptr;

    // Draft RoPE tables [MAX_POSITION_EMBEDDINGS, 64] (full 128-dim NeoX!)
    float* rope_cos = nullptr;
    float* rope_sin = nullptr;
};

class DFlash2Model {
public:
    DraftWeights weights;
    DraftKVPool kv_pool;
    DraftWorkspace ws;

    void init(
        const std::string& draft_dir,
        bool quantize_to_fp8,
        int max_slots,
        int max_bs,
        int max_block_size,
        cudaStream_t stream);

    // Materialize target aux_hidden [num_tokens, 25600] into draft KV ring buffer!
    // Used both after prompt prefill (d_commit_lens=nullptr) and after verify commit (skips col >= commit_lens[b]!).
    void materialize_context_kv(
        Qwen35Model& target,
        const __nv_bfloat16* d_aux_hidden, // [num_tokens, 25600]
        const int32_t* d_positions,        // [num_tokens]
        const int32_t* d_draft_slots,      // [num_tokens] (-1 to skip!)
        int num_tokens,
        cudaStream_t stream,
        const int32_t* d_commit_lens = nullptr,
        int block_size = 0);

    // Run full DFlash2 draft forward + CandidateSelector top-16 + path walk -> writes candidate
    // block [B, block_size] = [anchor, cand_1..cand_{block_size-1}] into d_out_candidates!
    // 100% CUDA-graph compatible!
    void propose_candidates(
        Qwen35Model& target,
        const int32_t* d_anchor_ids,       // [B]
        const int32_t* d_prefix_lens,      // [B]
        const int32_t* d_draft_slots,      // [B]
        const float* d_temperatures,       // [B] (nullable -> greedy)
        const uint8_t* d_greedy_mask,      // [B] (nullable -> greedy)
        const float* d_uniforms,           // [B, block_size - 1] (nullable -> greedy)
        int32_t* d_draft_block_ids_scratch,// [B * block_size]
        int32_t* d_draft_positions_scratch,// [B * block_size]
        int32_t* d_out_candidates,         // [B * block_size]
        int B,
        int block_size,
        cudaStream_t stream);

    void copy_kv_slot(int dst_slot, int src_slot, cudaStream_t stream);

    void destroy();
};

} // namespace amperegen
