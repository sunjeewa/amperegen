#include "model/dflash_model.hpp"
#include <cstdio>

namespace amperegen {

void DFlash2Model::init(
    const std::string& draft_dir,
    bool quantize_to_fp8,
    int max_slots,
    int max_bs,
    int max_block_size,
    cudaStream_t stream
) {
    load_draft_weights(draft_dir, weights, quantize_to_fp8, stream);

    // 1) Draft KV ring-buffer pool: [max_slots, 2048, 8, 128] BF16 x 2 x 5 layers
    kv_pool.max_slots = max_slots;
    size_t elems_per_layer = (size_t)max_slots * DFlash2Config::SLIDING_WINDOW * DFlash2Config::NUM_KV_HEADS * DFlash2Config::HEAD_DIM;
    size_t kv_bytes = (size_t)DFlash2Config::NUM_LAYERS * 2 * elems_per_layer * 2 + 16ULL * 1024 * 1024;
    kv_pool.arena.init(kv_bytes);
    for (int L = 0; L < DFlash2Config::NUM_LAYERS; ++L) {
        kv_pool.k_cache[L] = kv_pool.arena.alloc_typed<__nv_bfloat16>(elems_per_layer);
        kv_pool.v_cache[L] = kv_pool.arena.alloc_typed<__nv_bfloat16>(elems_per_layer);
    }

    // 2) Draft workspace
    ws.max_ctx_tokens = DFlash2Config::SLIDING_WINDOW;
    ws.max_draft_tokens = max_bs * max_block_size;
    int max_pred_tokens = max_bs * (max_block_size - 1);

    size_t ws_bytes = 0;
    ws_bytes += (size_t)ws.max_ctx_tokens * 5120 * 2 * 2;  // ctx_hidden, ctx_norm
    ws_bytes += (size_t)ws.max_ctx_tokens * 2048 * 2;      // ctx_kv
    ws_bytes += (size_t)ws.max_draft_tokens * 5120 * 2 * 4;// residual, norm_out, conv_buf, sub_out
    ws_bytes += (size_t)ws.max_draft_tokens * 1280 * 2;    // conv_delta
    ws_bytes += (size_t)ws.max_draft_tokens * 6144 * 2;    // qkv_buf
    ws_bytes += (size_t)ws.max_draft_tokens * 4096 * 2;    // attn_out
    ws_bytes += (size_t)ws.max_draft_tokens * 34816 * 2;   // gate_up_buf
    ws_bytes += (size_t)ws.max_draft_tokens * 17408 * 2;   // mlp_mid
    ws_bytes += (size_t)max_pred_tokens * 5120 * 2;        // pred_hidden
    ws_bytes += (size_t)max_pred_tokens * 256 * 2;         // proj_hidden
    ws_bytes += (size_t)max_pred_tokens * Qwen35Config::VOCAB_SIZE * 2; // cand_logits
    ws_bytes += (size_t)max_pred_tokens * 16 * 12;         // topk_ids + topk_vals + q_rows
    ws_bytes += (size_t)ws.max_draft_tokens * 32 * 16 * 128 * 4; // attn_split_out
    ws_bytes += (size_t)ws.max_draft_tokens * 32 * 16 * 2 * 4;   // attn_split_lse
    ws_bytes += (size_t)Qwen35Config::MAX_POSITION_EMBEDDINGS * 64 * 4 * 2; // RoPE cos/sin
    ws_bytes += 32ULL * 1024 * 1024;

    ws.arena.init(ws_bytes);
    ws.ctx_hidden = ws.arena.alloc_typed<__nv_bfloat16>((size_t)ws.max_ctx_tokens * 5120);
    ws.ctx_norm = ws.arena.alloc_typed<__nv_bfloat16>((size_t)ws.max_ctx_tokens * 5120);
    ws.ctx_kv = ws.arena.alloc_typed<__nv_bfloat16>((size_t)ws.max_ctx_tokens * 2048);

    ws.residual = ws.arena.alloc_typed<__nv_bfloat16>((size_t)ws.max_draft_tokens * 5120);
    ws.norm_out = ws.arena.alloc_typed<__nv_bfloat16>((size_t)ws.max_draft_tokens * 5120);
    ws.conv_buf = ws.arena.alloc_typed<__nv_bfloat16>((size_t)ws.max_draft_tokens * 5120);
    ws.conv_delta = ws.arena.alloc_typed<__nv_bfloat16>((size_t)ws.max_draft_tokens * 1280);
    ws.sub_out = ws.arena.alloc_typed<__nv_bfloat16>((size_t)ws.max_draft_tokens * 5120);
    ws.qkv_buf = ws.arena.alloc_typed<__nv_bfloat16>((size_t)ws.max_draft_tokens * 6144);
    ws.attn_out = ws.arena.alloc_typed<__nv_bfloat16>((size_t)ws.max_draft_tokens * 4096);
    ws.gate_up_buf = ws.arena.alloc_typed<__nv_bfloat16>((size_t)ws.max_draft_tokens * 34816);
    ws.mlp_mid = ws.arena.alloc_typed<__nv_bfloat16>((size_t)ws.max_draft_tokens * 17408);

    ws.pred_hidden = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_pred_tokens * 5120);
    ws.proj_hidden = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_pred_tokens * 256);
    ws.cand_logits = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_pred_tokens * Qwen35Config::VOCAB_SIZE);
    ws.topk_ids = ws.arena.alloc_typed<int32_t>((size_t)max_pred_tokens * 16);
    ws.topk_vals = ws.arena.alloc_typed<float>((size_t)max_pred_tokens * 16);
    ws.q_rows = ws.arena.alloc_typed<float>((size_t)max_pred_tokens * 16);
    ws.attn_split_out = ws.arena.alloc_typed<float>((size_t)ws.max_draft_tokens * 32 * 16 * 128);
    ws.attn_split_lse = ws.arena.alloc_typed<float>((size_t)ws.max_draft_tokens * 32 * 16 * 2);

    ws.rope_cos = ws.arena.alloc_typed<float>((size_t)Qwen35Config::MAX_POSITION_EMBEDDINGS * 64);
    ws.rope_sin = ws.arena.alloc_typed<float>((size_t)Qwen35Config::MAX_POSITION_EMBEDDINGS * 64);
    launch_init_rope_table(
        ws.rope_cos, ws.rope_sin,
        Qwen35Config::MAX_POSITION_EMBEDDINGS,
        DFlash2Config::HEAD_DIM,
        DFlash2Config::ROPE_THETA,
        stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    fprintf(stderr, "[dflash] Initialized: KV pool=%d slots (%.2f GB), WS=%.2f GB\n",
            max_slots, kv_pool.arena.used() / 1e9, ws.arena.used() / 1e9);
}

void DFlash2Model::materialize_context_kv(
    Qwen35Model& target,
    const __nv_bfloat16* d_aux_hidden,
    const int32_t* d_positions,
    const int32_t* d_draft_slots,
    int num_tokens,
    cudaStream_t stream,
    const int32_t* d_commit_lens,
    int block_size
) {
    if (num_tokens <= 0) return;
    // Process in chunks of <= ws.max_ctx_tokens (2048)
    for (int off = 0; off < num_tokens; off += ws.max_ctx_tokens) {
        int chunk = std::min(ws.max_ctx_tokens, num_tokens - off);
        const __nv_bfloat16* aux_c = d_aux_hidden + (size_t)off * DFlash2Config::FC_IN_DIM;
        const int32_t* pos_c = d_positions + off;
        const int32_t* slot_c = d_draft_slots + off;

        // 1) fc: [chunk, 25600] -> [chunk, 5120], then plain RMSNorm (hidden_norm)!
        if (weights.use_fp8) {
            launch_fp8_linear(
                target.ctx.cublas, aux_c, weights.fc_fp8, ws.ctx_hidden,
                target.ws.d_splitk_scratch, target.ws.d_bf16_w_scratch, chunk, stream);
        } else {
            launch_bf16_linear(
                target.ctx.cublas, aux_c, weights.fc_bf16, ws.ctx_hidden,
                chunk, DFlash2Config::HIDDEN_SIZE, DFlash2Config::FC_IN_DIM, stream, false);
        }
        launch_rmsnorm(ws.ctx_hidden, weights.hidden_norm, ws.ctx_norm, chunk, DFlash2Config::HIDDEN_SIZE, DFlash2Config::RMS_EPS, false, stream);

        // 2) For each of the 5 draft layers: kv_proj_only -> plain k_norm -> RoPE -> ring-buffer store!
        for (int L = 0; L < DFlash2Config::NUM_LAYERS; ++L) {
            const auto& lw = weights.layers[L];
            if (weights.use_fp8) {
                launch_fp8_linear(
                    target.ctx.cublas, ws.ctx_norm, lw.kv_only_fp8, ws.ctx_kv,
                    target.ws.d_splitk_scratch, target.ws.d_bf16_w_scratch, chunk, stream);
            } else {
                launch_bf16_linear(
                    target.ctx.cublas, ws.ctx_norm, lw.kv_only_bf16, ws.ctx_kv,
                    chunk, DFlash2Config::KV_DIM * 2, DFlash2Config::HIDDEN_SIZE, stream, false);
            }
            launch_dflash_append_ctx_kv(
                ws.ctx_kv, lw.k_norm, ws.rope_cos, ws.rope_sin, pos_c, slot_c,
                (off == 0) ? d_commit_lens : nullptr,
                kv_pool.k_cache[L], kv_pool.v_cache[L], chunk, block_size, DFlash2Config::RMS_EPS, stream);
        }
    }
}

void DFlash2Model::propose_candidates(
    Qwen35Model& target,
    const int32_t* d_anchor_ids,
    const int32_t* d_prefix_lens,
    const int32_t* d_draft_slots,
    const float* d_temperatures,
    const uint8_t* d_greedy_mask,
    const float* d_uniforms,
    int32_t* d_draft_block_ids_scratch,
    int32_t* d_draft_positions_scratch,
    int32_t* d_out_candidates,
    int B,
    int block_size,
    cudaStream_t stream
) {
    int total_T = B * block_size;
    int num_pred = block_size - 1;
    int pred_T = B * num_pred;

    // 1) Build draft block inputs [anchor, MASK, ..., MASK] and positions [prefix_len + 0..block_size-1]
    launch_prepare_draft_block(
        d_anchor_ids, d_prefix_lens,
        d_draft_block_ids_scratch, d_draft_positions_scratch,
        B, block_size, DFlash2Config::MASK_TOKEN_ID, stream);

    // 2) Embed using shared target embed_tokens!
    launch_embed_lookup(
        d_draft_block_ids_scratch, target.weights.embed_tokens,
        ws.residual, total_T, DFlash2Config::HIDDEN_SIZE, stream);

    // 3) 5 DFlashDecoderLayers (plain RMSNorm, dynamic 2-tap grouped conv on attn & MLP, bidirectional sliding attn!)
    for (int L = 0; L < DFlash2Config::NUM_LAYERS; ++L) {
        const auto& lw = weights.layers[L];

        // Input plain RMSNorm (fused with previous layer's residual add for L > 0!)
        if (L == 0) {
            launch_rmsnorm(ws.residual, lw.input_layernorm, ws.norm_out, total_T, DFlash2Config::HIDDEN_SIZE, DFlash2Config::RMS_EPS, false, stream);
        } else {
            launch_fused_add_rmsnorm(ws.sub_out, ws.residual, lw.input_layernorm, ws.norm_out, total_T, DFlash2Config::HIDDEN_SIZE, DFlash2Config::RMS_EPS, false, stream);
        }

        // Attention grouped conv prepare
        launch_bf16_linear(target.ctx.cublas, ws.norm_out, lw.attn_conv_proj, ws.conv_delta, total_T, DFlash2Config::CONV_PROJ_OUT, DFlash2Config::HIDDEN_SIZE, stream, false);
        launch_dflash_grouped_conv(ws.norm_out, ws.conv_delta, lw.attn_conv_base, ws.conv_buf, B, block_size, 0, stream);

        // QKV proj + plain QK-norm + full 128-dim RoPE + bidirectional sliding attention + O proj
        if (weights.use_fp8) {
            launch_fp8_linear(target.ctx.cublas, ws.conv_buf, lw.qkv_fp8, ws.qkv_buf, target.ws.d_splitk_scratch, target.ws.d_bf16_w_scratch, total_T, stream);
        } else {
            launch_bf16_linear(target.ctx.cublas, ws.conv_buf, lw.qkv_bf16, ws.qkv_buf, total_T, DFlash2Config::QKV_DIM, DFlash2Config::HIDDEN_SIZE, stream, false);
        }
        launch_dflash_qk_norm_rope(ws.qkv_buf, lw.q_norm, lw.k_norm, ws.rope_cos, ws.rope_sin, d_draft_positions_scratch, total_T, DFlash2Config::RMS_EPS, stream);
        launch_dflash_bidirectional_attn(
            ws.qkv_buf, kv_pool.k_cache[L], kv_pool.v_cache[L], d_draft_slots, d_prefix_lens,
            ws.attn_out, ws.attn_split_out, ws.attn_split_lse, B, block_size, stream);
        if (weights.use_fp8) {
            launch_fp8_linear(target.ctx.cublas, ws.attn_out, lw.o_fp8, ws.conv_buf, target.ws.d_splitk_scratch, target.ws.d_bf16_w_scratch, total_T, stream);
        } else {
            launch_bf16_linear(target.ctx.cublas, ws.attn_out, lw.o_bf16, ws.conv_buf, total_T, DFlash2Config::HIDDEN_SIZE, DFlash2Config::Q_DIM, stream, false);
        }

        // Attention grouped conv finish (side = 1 -> base_kernel + 2*5120)
        launch_dflash_grouped_conv(ws.conv_buf, ws.conv_delta, lw.attn_conv_base + 2 * DFlash2Config::HIDDEN_SIZE, ws.sub_out, B, block_size, 1, stream);

        // Post-attention residual add + plain RMSNorm
        launch_fused_add_rmsnorm(ws.sub_out, ws.residual, lw.post_attention_layernorm, ws.norm_out, total_T, DFlash2Config::HIDDEN_SIZE, DFlash2Config::RMS_EPS, false, stream);

        // MLP grouped conv prepare
        launch_bf16_linear(target.ctx.cublas, ws.norm_out, lw.mlp_conv_proj, ws.conv_delta, total_T, DFlash2Config::CONV_PROJ_OUT, DFlash2Config::HIDDEN_SIZE, stream, false);
        launch_dflash_grouped_conv(ws.norm_out, ws.conv_delta, lw.mlp_conv_base, ws.conv_buf, B, block_size, 0, stream);

        // MLP SwiGLU
        if (weights.use_fp8) {
            launch_fp8_linear(target.ctx.cublas, ws.conv_buf, lw.gate_up_fp8, ws.gate_up_buf, target.ws.d_splitk_scratch, target.ws.d_bf16_w_scratch, total_T, stream);
        } else {
            launch_bf16_linear(target.ctx.cublas, ws.conv_buf, lw.gate_up_bf16, ws.gate_up_buf, total_T, DFlash2Config::INTERMEDIATE_SIZE * 2, DFlash2Config::HIDDEN_SIZE, stream, false);
        }
        launch_silu_mul(ws.gate_up_buf, ws.mlp_mid, total_T, DFlash2Config::INTERMEDIATE_SIZE, stream);
        if (weights.use_fp8) {
            launch_fp8_linear(target.ctx.cublas, ws.mlp_mid, lw.down_fp8, ws.conv_buf, target.ws.d_splitk_scratch, target.ws.d_bf16_w_scratch, total_T, stream);
        } else {
            launch_bf16_linear(target.ctx.cublas, ws.mlp_mid, lw.down_bf16, ws.conv_buf, total_T, DFlash2Config::HIDDEN_SIZE, DFlash2Config::INTERMEDIATE_SIZE, stream, false);
        }

        // MLP grouped conv finish (side = 1)
        launch_dflash_grouped_conv(ws.conv_buf, ws.conv_delta, lw.mlp_conv_base + 2 * DFlash2Config::HIDDEN_SIZE, ws.sub_out, B, block_size, 1, stream);
    }

    // 4) Final residual add + plain RMSNorm
    launch_fused_add_rmsnorm(ws.sub_out, ws.residual, weights.final_norm, ws.norm_out, total_T, DFlash2Config::HIDDEN_SIZE, DFlash2Config::RMS_EPS, false, stream);

    // 5) Slice predicted slots [:, 1:, :] -> pred_hidden [B * num_pred, 5120]
    launch_slice_pred_hidden(ws.norm_out, ws.pred_hidden, B, block_size, stream);

    // 6) Unary logits via shared target lm_head (use FP8 lm_head if enabled -> 2x faster, 100% lossless for target!)
    if (target.weights.has_fp8_lm_head) {
        launch_fp8_linear(
            target.ctx.cublas, ws.pred_hidden, target.weights.lm_head_fp8, ws.cand_logits,
            target.ws.d_splitk_scratch, target.ws.d_bf16_w_scratch, pred_T, stream);
    } else {
        launch_bf16_linear(
            target.ctx.cublas, ws.pred_hidden, target.weights.lm_head, ws.cand_logits,
            pred_T, Qwen35Config::VOCAB_SIZE, DFlash2Config::HIDDEN_SIZE, stream, false);
    }

    // 7) Exact Top-16 unary candidates + hidden projection (5120->256) + CandidateSelector walk!
    launch_dflash_topk16(ws.cand_logits, ws.topk_ids, ws.topk_vals, pred_T, Qwen35Config::VOCAB_SIZE, stream);
    launch_bf16_linear(
        target.ctx.cublas, ws.pred_hidden, weights.selector_hidden_proj, ws.proj_hidden,
        pred_T, DFlash2Config::SELECTOR_RANK, DFlash2Config::HIDDEN_SIZE, stream, false);

    launch_dflash_selector_walk(
        ws.topk_ids, ws.topk_vals, ws.proj_hidden,
        weights.pred_codebook, weights.succ_codebook,
        d_anchor_ids, d_temperatures, d_greedy_mask, d_uniforms,
        d_out_candidates, (d_temperatures != nullptr) ? ws.q_rows : nullptr,
        B, block_size, stream);
}

void DFlash2Model::copy_kv_slot(int dst_slot, int src_slot, cudaStream_t stream) {
    if (dst_slot == src_slot) return;
    size_t slot_elems = (size_t)DFlash2Config::SLIDING_WINDOW * DFlash2Config::NUM_KV_HEADS * DFlash2Config::HEAD_DIM;
    size_t slot_bytes = slot_elems * sizeof(__nv_bfloat16);
    for (int L = 0; L < DFlash2Config::NUM_LAYERS; ++L) {
        CUDA_CHECK(cudaMemcpyAsync(
            kv_pool.k_cache[L] + (size_t)dst_slot * slot_elems,
            kv_pool.k_cache[L] + (size_t)src_slot * slot_elems,
            slot_bytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            kv_pool.v_cache[L] + (size_t)dst_slot * slot_elems,
            kv_pool.v_cache[L] + (size_t)src_slot * slot_elems,
            slot_bytes, cudaMemcpyDeviceToDevice, stream));
    }
}

void DFlash2Model::destroy() {
    ws.arena.release();
    kv_pool.arena.release();
    weights.arena.release();
}

} // namespace amperegen
