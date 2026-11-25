#include "model/qwen35_model.hpp"
#include <cstdio>
#include <cstring>

namespace amperegen {

void Qwen35Model::init_context() {
    ctx.init();
}

void Qwen35Model::init_workspace_and_pools(
    int max_prefill_chunk,
    int max_state_slots,
    int max_verify_bs,
    int max_block_size,
    size_t kv_pool_bytes
) {
    // 1) Workspace arena
    ws.max_tokens = max_prefill_chunk;
    ws.max_logits_rows = std::max(max_verify_bs * max_block_size, max_verify_bs);
    int max_verify_tokens = max_verify_bs * max_block_size;
    int max_chunks = (max_prefill_chunk + 63) / 64 + max_verify_bs;

    size_t ws_bytes = 0;
    ws_bytes += (size_t)max_prefill_chunk * 5120 * 2 * 3;    // residual, norm_out, sub_out
    ws_bytes += (size_t)max_prefill_chunk * 16384 * 2;       // qkv_buf
    ws_bytes += (size_t)max_prefill_chunk * 96 * 2;          // ba_buf
    ws_bytes += (size_t)max_prefill_chunk * 10240 * 2;       // conv_out
    ws_bytes += (size_t)max_prefill_chunk * 6144 * 2 * 3;    // q_buf, gate_buf, attn_out
    ws_bytes += (size_t)max_prefill_chunk * 34816 * 2;       // gate_up_buf
    ws_bytes += (size_t)max_prefill_chunk * 17408 * 2;       // mlp_mid
    ws_bytes += (size_t)max_prefill_chunk * 25600 * 2;       // aux_hidden
    ws_bytes += (size_t)ws.max_logits_rows * Qwen35Config::VOCAB_SIZE * 2; // logits
    ws_bytes += (size_t)8 * 32 * 34816 * 4;                  // splitk scratch (~36 MB, supports mc<=32)
    ws_bytes += (size_t)34816 * 5120 * 2;                    // bf16 dequant scratch (356 MB)
    ws_bytes += (size_t)max_verify_tokens * 24 * 64 * 256 * 4; // fa_splitk_out
    ws_bytes += (size_t)max_verify_tokens * 24 * 64 * 2 * 4;   // fa_splitk_lse
    // GDN chunked prefill workspace
    ws_bytes += (size_t)max_prefill_chunk * 48 * 4;          // g_cum
    ws_bytes += (size_t)max_prefill_chunk * 16 * 128 * 2 * 2;// q_norm, k_norm
    ws_bytes += (size_t)max_prefill_chunk * 48 * 128 * 2 * 3;// w, u, v_new
    ws_bytes += (size_t)max_chunks * 48 * 128 * 128 * 2;     // h_chunks
    // RoPE tables
    ws_bytes += (size_t)Qwen35Config::MAX_POSITION_EMBEDDINGS * 32 * 4 * 2;
    ws_bytes += 64ULL * 1024 * 1024; // alignment margin

    ws.arena.init(ws_bytes);
    ws.residual = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_prefill_chunk * 5120);
    ws.norm_out = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_prefill_chunk * 5120);
    ws.sub_out = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_prefill_chunk * 5120);
    ws.qkv_buf = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_prefill_chunk * 16384);
    ws.ba_buf = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_prefill_chunk * 96);
    ws.conv_out = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_prefill_chunk * 10240);
    ws.q_buf = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_prefill_chunk * 6144);
    ws.gate_buf = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_prefill_chunk * 6144);
    ws.attn_out = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_prefill_chunk * 6144);
    ws.gate_up_buf = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_prefill_chunk * 34816);
    ws.mlp_mid = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_prefill_chunk * 17408);
    ws.aux_hidden = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_prefill_chunk * 25600);
    ws.logits = ws.arena.alloc_typed<__nv_bfloat16>((size_t)ws.max_logits_rows * Qwen35Config::VOCAB_SIZE);
    ws.d_splitk_scratch = ws.arena.alloc_typed<float>((size_t)8 * 32 * 34816);
    ws.d_bf16_w_scratch = ws.arena.alloc_typed<__nv_bfloat16>((size_t)34816 * 5120);
    ws.fa_splitk_out = ws.arena.alloc_typed<float>((size_t)max_verify_tokens * 24 * 64 * 256);
    ws.fa_splitk_lse = ws.arena.alloc_typed<float>((size_t)max_verify_tokens * 24 * 64 * 2);

    ws.gdn_ws.max_T = max_prefill_chunk;
    ws.gdn_ws.g_cum = ws.arena.alloc_typed<float>((size_t)max_prefill_chunk * 48);
    ws.gdn_ws.q_norm = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_prefill_chunk * 16 * 128);
    ws.gdn_ws.k_norm = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_prefill_chunk * 16 * 128);
    ws.gdn_ws.w = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_prefill_chunk * 48 * 128);
    ws.gdn_ws.u = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_prefill_chunk * 48 * 128);
    ws.gdn_ws.v_new = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_prefill_chunk * 48 * 128);
    ws.gdn_ws.h_chunks = ws.arena.alloc_typed<__nv_bfloat16>((size_t)max_chunks * 48 * 128 * 128);

    ws.rope_cos = ws.arena.alloc_typed<float>((size_t)Qwen35Config::MAX_POSITION_EMBEDDINGS * 32);
    ws.rope_sin = ws.arena.alloc_typed<float>((size_t)Qwen35Config::MAX_POSITION_EMBEDDINGS * 32);
    launch_init_rope_table(
        ws.rope_cos, ws.rope_sin,
        Qwen35Config::MAX_POSITION_EMBEDDINGS,
        Qwen35Config::ATTN_ROTARY_DIM,
        Qwen35Config::ROPE_THETA,
        ctx.stream);

    // 2) Target FP8 KV cache pool
    size_t bytes_per_block_all_layers = (size_t)Qwen35Config::NUM_FULL_ATTN_LAYERS * 2 *
        Qwen35Config::KV_BLOCK_SIZE * Qwen35Config::ATTN_KV_HEADS * Qwen35Config::ATTN_HEAD_DIM;
    pools.num_kv_blocks = (int)(kv_pool_bytes / bytes_per_block_all_layers);
    size_t per_layer_kv_elems = (size_t)pools.num_kv_blocks * Qwen35Config::KV_BLOCK_SIZE *
        Qwen35Config::ATTN_KV_HEADS * Qwen35Config::ATTN_HEAD_DIM;
    pools.kv_arena.init((size_t)pools.num_kv_blocks * bytes_per_block_all_layers + 16ULL * 1024 * 1024);
    for (int l = 0; l < Qwen35Config::NUM_FULL_ATTN_LAYERS; ++l) {
        pools.k_cache[l] = pools.kv_arena.alloc_typed<__nv_fp8_e4m3>(per_layer_kv_elems);
        pools.v_cache[l] = pools.kv_arena.alloc_typed<__nv_fp8_e4m3>(per_layer_kv_elems);
    }

    // 3) GDN persistent state pool
    pools.max_state_slots = max_state_slots;
    size_t ssm_per_layer = (size_t)max_state_slots * 48 * 128 * 128;
    size_t conv_per_layer = (size_t)max_state_slots * 10240 * 3;
    pools.gdn_arena.init((size_t)Qwen35Config::NUM_GDN_LAYERS * (ssm_per_layer + conv_per_layer) * 2 + 16ULL * 1024 * 1024);
    for (int l = 0; l < Qwen35Config::NUM_GDN_LAYERS; ++l) {
        pools.ssm_states[l] = pools.gdn_arena.alloc_typed<__nv_bfloat16>(ssm_per_layer);
        pools.conv_states[l] = pools.gdn_arena.alloc_typed<__nv_bfloat16>(conv_per_layer);
    }
    CUDA_CHECK(cudaMemsetAsync(pools.gdn_arena.base(), 0, pools.gdn_arena.used(), ctx.stream));

    // 4) GDN verify replay buffers
    pools.max_verify_bs = max_verify_bs;
    pools.max_block_size = max_block_size;
    size_t co_elems = (size_t)max_verify_bs * max_block_size * 10240;
    size_t ba_elems = (size_t)max_verify_bs * max_block_size * 96;
    size_t ci_elems = (size_t)max_verify_bs * max_block_size * 10240 * 3;
    size_t ver_bytes = (size_t)Qwen35Config::NUM_GDN_LAYERS * (co_elems + ba_elems + ci_elems) * 2 + 16ULL * 1024 * 1024;
    pools.verify_arena.init(ver_bytes);
    for (int l = 0; l < Qwen35Config::NUM_GDN_LAYERS; ++l) {
        pools.conv_out_verify[l] = pools.verify_arena.alloc_typed<__nv_bfloat16>(co_elems);
        pools.ba_verify[l] = pools.verify_arena.alloc_typed<__nv_bfloat16>(ba_elems);
        pools.conv_inter[l] = pools.verify_arena.alloc_typed<__nv_bfloat16>(ci_elems);
    }

    // Device pointer tables for single-launch commit_replay
    pools.d_ssm_states_ptrs = pools.verify_arena.alloc_typed<__nv_bfloat16*>(48);
    pools.d_conv_states_ptrs = pools.verify_arena.alloc_typed<__nv_bfloat16*>(48);
    pools.d_conv_inter_ptrs = pools.verify_arena.alloc_typed<__nv_bfloat16*>(48);
    pools.d_conv_out_verify_ptrs = pools.verify_arena.alloc_typed<__nv_bfloat16*>(48);
    pools.d_ba_verify_ptrs = pools.verify_arena.alloc_typed<__nv_bfloat16*>(48);
    pools.d_A_log_ptrs = pools.verify_arena.alloc_typed<float*>(48);
    pools.d_dt_bias_ptrs = pools.verify_arena.alloc_typed<float*>(48);

    float* h_alog[48];
    float* h_dt[48];
    for (int l = 0; l < 48; ++l) {
        h_alog[l] = weights.gdn[l].A_log;
        h_dt[l] = weights.gdn[l].dt_bias;
    }
    CUDA_CHECK(cudaMemcpyAsync(pools.d_ssm_states_ptrs, pools.ssm_states, 48 * sizeof(void*), cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(pools.d_conv_states_ptrs, pools.conv_states, 48 * sizeof(void*), cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(pools.d_conv_inter_ptrs, pools.conv_inter, 48 * sizeof(void*), cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(pools.d_conv_out_verify_ptrs, pools.conv_out_verify, 48 * sizeof(void*), cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(pools.d_ba_verify_ptrs, pools.ba_verify, 48 * sizeof(void*), cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(pools.d_A_log_ptrs, h_alog, 48 * sizeof(void*), cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(pools.d_dt_bias_ptrs, h_dt, 48 * sizeof(void*), cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    fprintf(stderr, "[target] Pools initialized: KV=%d blocks (%d tokens, %.2f GB), GDN=%d slots (%.2f GB), WS=%.2f GB, Verify=%.2f GB\n",
            pools.num_kv_blocks, pools.num_kv_blocks * Qwen35Config::KV_BLOCK_SIZE,
            pools.kv_arena.used() / 1e9, max_state_slots, pools.gdn_arena.used() / 1e9,
            ws.arena.used() / 1e9, pools.verify_arena.used() / 1e9);
}

void Qwen35Model::reset_state_slot(int slot, cudaStream_t stream) {
    for (int l = 0; l < Qwen35Config::NUM_GDN_LAYERS; ++l) {
        CUDA_CHECK(cudaMemsetAsync(
            pools.ssm_states[l] + (size_t)slot * 48 * 128 * 128,
            0, (size_t)48 * 128 * 128 * 2, stream));
        CUDA_CHECK(cudaMemsetAsync(
            pools.conv_states[l] + (size_t)slot * 10240 * 3,
            0, (size_t)10240 * 3 * 2, stream));
    }
}

void Qwen35Model::forward_layer_prefill(
    int layer_idx,
    __nv_bfloat16* residual,
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
    cudaStream_t stream
) {
    if (Qwen35Config::is_full_attn(layer_idx)) {
        int fa = Qwen35Config::full_attn_idx(layer_idx);
        const auto& lw = weights.full_attn[fa];

        // 1) Input GemmaRMSNorm
        launch_rmsnorm(residual, lw.input_layernorm, ws.norm_out, total_T, Qwen35Config::HIDDEN_SIZE, Qwen35Config::RMS_EPS, true, stream);
        // 2) QKV projection [total_T, 14336]
        launch_fp8_linear(ctx.cublas, ws.norm_out, lw.qkv_proj, ws.qkv_buf, ws.d_splitk_scratch, ws.d_bf16_w_scratch, total_T, stream);
        // 3) Split q/gate + QK GemmaRMSNorm + partial NeoX RoPE + FP8 KV cache write
        launch_full_attn_qkv_prep_and_kv_write(
            ws.qkv_buf, lw.q_norm, lw.k_norm, ws.rope_cos, ws.rope_sin, positions,
            ws.q_buf, ws.gate_buf, pools.k_cache[fa], pools.v_cache[fa], slot_mapping,
            total_T, Qwen35Config::RMS_EPS, stream);
        // 4) Prefill causal FlashAttention-2 with fused output gate!
        launch_paged_attn_prefill_fp8(
            ws.q_buf, ws.gate_buf, pools.k_cache[fa], pools.v_cache[fa],
            block_tables, cu_seqlens, kv_lens, ws.attn_out,
            B, max_q_len, max_blocks_per_seq, stream);
        // 5) O projection
        launch_fp8_linear(ctx.cublas, ws.attn_out, lw.o_proj, ws.sub_out, ws.d_splitk_scratch, ws.d_bf16_w_scratch, total_T, stream);
        // 6) Fused residual add + post_attention GemmaRMSNorm
        launch_fused_add_rmsnorm(ws.sub_out, residual, lw.post_attention_layernorm, ws.norm_out, total_T, Qwen35Config::HIDDEN_SIZE, Qwen35Config::RMS_EPS, true, stream);
        // 7) MLP SwiGLU
        launch_fp8_linear(ctx.cublas, ws.norm_out, lw.gate_up_proj, ws.gate_up_buf, ws.d_splitk_scratch, ws.d_bf16_w_scratch, total_T, stream);
        launch_silu_mul(ws.gate_up_buf, ws.mlp_mid, total_T, Qwen35Config::INTERMEDIATE_SIZE, stream);
        launch_fp8_linear(ctx.cublas, ws.mlp_mid, lw.down_proj, ws.sub_out, ws.d_splitk_scratch, ws.d_bf16_w_scratch, total_T, stream);
        // 8) Residual add
        launch_bf16_add_inplace(residual, ws.sub_out, (size_t)total_T * Qwen35Config::HIDDEN_SIZE, stream);
    } else {
        int gi = Qwen35Config::gdn_idx(layer_idx);
        const auto& lw = weights.gdn[gi];

        // 1) Input GemmaRMSNorm
        launch_rmsnorm(residual, lw.input_layernorm, ws.norm_out, total_T, Qwen35Config::HIDDEN_SIZE, Qwen35Config::RMS_EPS, true, stream);
        // 2) QKVZ (FP8) + BA (BF16) projections
        launch_fp8_linear(ctx.cublas, ws.norm_out, lw.in_proj_qkvz, ws.qkv_buf, ws.d_splitk_scratch, ws.d_bf16_w_scratch, total_T, stream);
        launch_bf16_linear(ctx.cublas, ws.norm_out, lw.in_proj_ba, ws.ba_buf, total_T, Qwen35Config::GDN_BA_DIM, Qwen35Config::HIDDEN_SIZE, stream, false);
        // 3) Causal depthwise Conv1d + SiLU
        launch_gdn_conv1d_prefill(
            ws.qkv_buf, lw.conv1d_w, pools.conv_states[gi], state_slots, cu_seqlens,
            ws.conv_out, B, total_T, stream);
        // 4) GDN Chunk-64 WY Prefill
        launch_gdn_chunked_prefill(
            ws.conv_out, ws.ba_buf, lw.A_log, lw.dt_bias,
            pools.ssm_states[gi], state_slots, cu_seqlens,
            ws.q_buf, ws.gdn_ws, B, total_T, stream);
        // 5) Plain RMSNormGated (with Swish(z) read directly from ws.qkv_buf[:, 10240:])
        launch_gdn_rmsnorm_gated(ws.q_buf, ws.qkv_buf, lw.gdn_norm, ws.attn_out, total_T, Qwen35Config::RMS_EPS, stream);
        // 6) Out projection
        launch_fp8_linear(ctx.cublas, ws.attn_out, lw.out_proj, ws.sub_out, ws.d_splitk_scratch, ws.d_bf16_w_scratch, total_T, stream);
        // 7) Fused residual add + post_attention GemmaRMSNorm
        launch_fused_add_rmsnorm(ws.sub_out, residual, lw.post_attention_layernorm, ws.norm_out, total_T, Qwen35Config::HIDDEN_SIZE, Qwen35Config::RMS_EPS, true, stream);
        // 8) MLP SwiGLU
        launch_fp8_linear(ctx.cublas, ws.norm_out, lw.gate_up_proj, ws.gate_up_buf, ws.d_splitk_scratch, ws.d_bf16_w_scratch, total_T, stream);
        launch_silu_mul(ws.gate_up_buf, ws.mlp_mid, total_T, Qwen35Config::INTERMEDIATE_SIZE, stream);
        launch_fp8_linear(ctx.cublas, ws.mlp_mid, lw.down_proj, ws.sub_out, ws.d_splitk_scratch, ws.d_bf16_w_scratch, total_T, stream);
        // 9) Residual add
        launch_bf16_add_inplace(residual, ws.sub_out, (size_t)total_T * Qwen35Config::HIDDEN_SIZE, stream);
    }
}

void Qwen35Model::forward_prefill(
    const int32_t* d_token_ids,
    const int32_t* d_positions,
    const int32_t* d_slot_mapping,
    const int32_t* d_block_tables,
    const int32_t* d_cu_seqlens,
    const int32_t* d_kv_lens,
    const int32_t* d_state_slots,
    int B,
    int total_T,
    int max_q_len,
    int max_blocks_per_seq,
    bool capture_aux_hidden,
    const int32_t* d_last_token_indices,
    __nv_bfloat16* d_logits_out,
    cudaStream_t stream
) {
    launch_embed_lookup(d_token_ids, weights.embed_tokens, ws.residual, total_T, Qwen35Config::HIDDEN_SIZE, stream);

    for (int L = 0; L < Qwen35Config::NUM_LAYERS; ++L) {
        forward_layer_prefill(
            L, ws.residual, d_positions, d_slot_mapping, d_block_tables,
            d_cu_seqlens, d_kv_lens, d_state_slots,
            B, total_T, max_q_len, max_blocks_per_seq, stream);
        if (capture_aux_hidden) {
            int cslot = Qwen35Config::capture_slot(L);
            if (cslot >= 0) {
                launch_copy_aux_hidden(ws.residual, ws.aux_hidden, total_T, cslot, stream);
            }
        }
    }

    if (d_logits_out != nullptr && d_last_token_indices != nullptr) {
        // Gather only the B last-token hidden states -> final GemmaRMSNorm -> exact BF16 lm_head!
        launch_gather_hidden_rows(ws.residual, d_last_token_indices, ws.sub_out, B, stream);
        launch_rmsnorm(ws.sub_out, weights.final_norm, ws.norm_out, B, Qwen35Config::HIDDEN_SIZE, Qwen35Config::RMS_EPS, true, stream);
        launch_bf16_linear(ctx.cublas, ws.norm_out, weights.lm_head, d_logits_out, B, Qwen35Config::VOCAB_SIZE, Qwen35Config::HIDDEN_SIZE, stream, false);
    }
}

void Qwen35Model::forward_decode_verify(
    const int32_t* d_token_ids,
    const int32_t* d_positions,
    const int32_t* d_slot_mapping,
    const int32_t* d_block_tables,
    const int32_t* d_seq_lens,
    const int32_t* d_state_slots,
    int B,
    int q_len,
    int max_blocks_per_seq,
    int max_seq_len_hint,
    bool commit_gdn_inplace,
    bool capture_aux_hidden,
    __nv_bfloat16* d_logits_out,
    cudaStream_t stream
) {
    int total_T = B * q_len;
    launch_embed_lookup(d_token_ids, weights.embed_tokens, ws.residual, total_T, Qwen35Config::HIDDEN_SIZE, stream);

    // Pre-norm loop optimization:
    // Layer 0 does standalone input_layernorm; layers 1..63 fuse previous layer's down_proj residual add
    // directly with layer L's input_layernorm (saving 63 full activation passes!)!
    // Wait: when capturing aux_hidden at layer L (5,19,33,47,61), we need the post-residual output
    // BEFORE layer L+1's input_layernorm — and `launch_fused_add_rmsnorm` updates `ws.residual` IN-PLACE
    // to `down_proj + residual` before computing `norm_out`, so capturing `ws.residual` right after
    // layer L+1's fused_add_rmsnorm captures the EXACT same post-residual tensor (just like SGLang's
    // `prepare_attn_and_capture_last_layer_outputs`!!)!
    for (int L = 0; L < Qwen35Config::NUM_LAYERS; ++L) {
        if (Qwen35Config::is_full_attn(L)) {
            int fa = Qwen35Config::full_attn_idx(L);
            const auto& lw = weights.full_attn[fa];

            if (L == 0) {
                launch_rmsnorm(ws.residual, lw.input_layernorm, ws.norm_out, total_T, Qwen35Config::HIDDEN_SIZE, Qwen35Config::RMS_EPS, true, stream);
            } else {
                launch_fused_add_rmsnorm(ws.sub_out, ws.residual, lw.input_layernorm, ws.norm_out, total_T, Qwen35Config::HIDDEN_SIZE, Qwen35Config::RMS_EPS, true, stream);
                if (capture_aux_hidden) {
                    int cslot = Qwen35Config::capture_slot(L - 1);
                    if (cslot >= 0) launch_copy_aux_hidden(ws.residual, ws.aux_hidden, total_T, cslot, stream);
                }
            }

            launch_fp8_linear(ctx.cublas, ws.norm_out, lw.qkv_proj, ws.qkv_buf, ws.d_splitk_scratch, ws.d_bf16_w_scratch, total_T, stream);
            launch_full_attn_qkv_prep_and_kv_write(
                ws.qkv_buf, lw.q_norm, lw.k_norm, ws.rope_cos, ws.rope_sin, d_positions,
                ws.q_buf, ws.gate_buf, pools.k_cache[fa], pools.v_cache[fa], d_slot_mapping,
                total_T, Qwen35Config::RMS_EPS, stream);
            launch_paged_attn_decode_fp8(
                ws.q_buf, ws.gate_buf, pools.k_cache[fa], pools.v_cache[fa],
                d_block_tables, d_seq_lens, ws.attn_out,
                ws.fa_splitk_out, ws.fa_splitk_lse,
                B, q_len, max_blocks_per_seq, max_seq_len_hint, stream);
            launch_fp8_linear(ctx.cublas, ws.attn_out, lw.o_proj, ws.sub_out, ws.d_splitk_scratch, ws.d_bf16_w_scratch, total_T, stream);
            launch_fused_add_rmsnorm(ws.sub_out, ws.residual, lw.post_attention_layernorm, ws.norm_out, total_T, Qwen35Config::HIDDEN_SIZE, Qwen35Config::RMS_EPS, true, stream);
            launch_fp8_linear(ctx.cublas, ws.norm_out, lw.gate_up_proj, ws.gate_up_buf, ws.d_splitk_scratch, ws.d_bf16_w_scratch, total_T, stream);
            launch_silu_mul(ws.gate_up_buf, ws.mlp_mid, total_T, Qwen35Config::INTERMEDIATE_SIZE, stream);
            launch_fp8_linear(ctx.cublas, ws.mlp_mid, lw.down_proj, ws.sub_out, ws.d_splitk_scratch, ws.d_bf16_w_scratch, total_T, stream);
        } else {
            int gi = Qwen35Config::gdn_idx(L);
            const auto& lw = weights.gdn[gi];

            if (L == 0) {
                launch_rmsnorm(ws.residual, lw.input_layernorm, ws.norm_out, total_T, Qwen35Config::HIDDEN_SIZE, Qwen35Config::RMS_EPS, true, stream);
            } else {
                launch_fused_add_rmsnorm(ws.sub_out, ws.residual, lw.input_layernorm, ws.norm_out, total_T, Qwen35Config::HIDDEN_SIZE, Qwen35Config::RMS_EPS, true, stream);
                if (capture_aux_hidden) {
                    int cslot = Qwen35Config::capture_slot(L - 1);
                    if (cslot >= 0) launch_copy_aux_hidden(ws.residual, ws.aux_hidden, total_T, cslot, stream);
                }
            }

            __nv_bfloat16* conv_dst = commit_gdn_inplace ? ws.conv_out : pools.conv_out_verify[gi];
            __nv_bfloat16* ba_dst = commit_gdn_inplace ? ws.ba_buf : pools.ba_verify[gi];
            __nv_bfloat16* conv_inter_dst = commit_gdn_inplace ? nullptr : pools.conv_inter[gi];

            launch_fp8_linear(ctx.cublas, ws.norm_out, lw.in_proj_qkvz, ws.qkv_buf, ws.d_splitk_scratch, ws.d_bf16_w_scratch, total_T, stream);
            launch_bf16_linear(ctx.cublas, ws.norm_out, lw.in_proj_ba, ba_dst, total_T, Qwen35Config::GDN_BA_DIM, Qwen35Config::HIDDEN_SIZE, stream, false);
            launch_gdn_conv1d_decode_verify(
                ws.qkv_buf, lw.conv1d_w, pools.conv_states[gi], conv_inter_dst,
                d_state_slots, conv_dst, B, q_len, commit_gdn_inplace, stream);
            launch_gdn_recurrent_decode_verify(
                conv_dst, ba_dst, lw.A_log, lw.dt_bias,
                pools.ssm_states[gi], nullptr, d_state_slots, ws.q_buf,
                B, q_len, commit_gdn_inplace, stream);
            launch_gdn_rmsnorm_gated(ws.q_buf, ws.qkv_buf, lw.gdn_norm, ws.attn_out, total_T, Qwen35Config::RMS_EPS, stream);
            launch_fp8_linear(ctx.cublas, ws.attn_out, lw.out_proj, ws.sub_out, ws.d_splitk_scratch, ws.d_bf16_w_scratch, total_T, stream);
            launch_fused_add_rmsnorm(ws.sub_out, ws.residual, lw.post_attention_layernorm, ws.norm_out, total_T, Qwen35Config::HIDDEN_SIZE, Qwen35Config::RMS_EPS, true, stream);
            launch_fp8_linear(ctx.cublas, ws.norm_out, lw.gate_up_proj, ws.gate_up_buf, ws.d_splitk_scratch, ws.d_bf16_w_scratch, total_T, stream);
            launch_silu_mul(ws.gate_up_buf, ws.mlp_mid, total_T, Qwen35Config::INTERMEDIATE_SIZE, stream);
            launch_fp8_linear(ctx.cublas, ws.mlp_mid, lw.down_proj, ws.sub_out, ws.d_splitk_scratch, ws.d_bf16_w_scratch, total_T, stream);
        }
    }

    // Final residual add + GemmaRMSNorm + exact BF16 lm_head!
    launch_fused_add_rmsnorm(ws.sub_out, ws.residual, weights.final_norm, ws.norm_out, total_T, Qwen35Config::HIDDEN_SIZE, Qwen35Config::RMS_EPS, true, stream);
    launch_bf16_linear(ctx.cublas, ws.norm_out, weights.lm_head, d_logits_out, total_T, Qwen35Config::VOCAB_SIZE, Qwen35Config::HIDDEN_SIZE, stream, false);
}

void Qwen35Model::commit_gdn_verify(
    const int32_t* d_state_slots,
    const int32_t* d_commit_lens,
    int B,
    int q_len,
    cudaStream_t stream
) {
    launch_gdn_commit_replay_states(
        pools.d_ssm_states_ptrs,
        pools.d_conv_states_ptrs,
        pools.d_conv_inter_ptrs,
        const_cast<const __nv_bfloat16* const*>(pools.d_conv_out_verify_ptrs),
        const_cast<const __nv_bfloat16* const*>(pools.d_ba_verify_ptrs),
        const_cast<const float* const*>(pools.d_A_log_ptrs),
        const_cast<const float* const*>(pools.d_dt_bias_ptrs),
        d_state_slots,
        d_commit_lens,
        B, q_len, stream);
}

void Qwen35Model::destroy() {
    pools.verify_arena.release();
    pools.gdn_arena.release();
    pools.kv_arena.release();
    ws.arena.release();
    weights.arena.release();
    ctx.destroy();
}

} // namespace amperegen
