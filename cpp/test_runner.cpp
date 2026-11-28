#include "kernels/kernels.cuh"
#include "loader/model_loader.hpp"
#include "model/dflash_model.hpp"
#include "model/qwen35_model.hpp"
#include "runtime/cuda_utils.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace amperegen;

namespace {

std::vector<__nv_bfloat16> read_bf16_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open " + path);
    size_t sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<__nv_bfloat16> v(sz / 2);
    f.read(reinterpret_cast<char*>(v.data()), sz);
    return v;
}

std::vector<int32_t> read_i32_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open " + path);
    size_t sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> v(sz / 4);
    f.read(reinterpret_cast<char*>(v.data()), sz);
    return v;
}

void compare_bf16(const char* name, const std::vector<__nv_bfloat16>& a, const std::vector<__nv_bfloat16>& b) {
    if (a.size() != b.size()) {
        printf("[FAIL] %s: size mismatch %zu vs %zu\n", name, a.size(), b.size());
        exit(1);
    }
    double sum_abs = 0.0, sum_ref = 0.0, sum_sq = 0.0, ref_sq = 0.0;
    float max_abs = 0.0f;
    int bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        float va = __bfloat162float(a[i]);
        float vb = __bfloat162float(b[i]);
        if (std::isnan(va) || std::isinf(va)) { ++bad; continue; }
        float d = fabsf(va - vb);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        sum_ref += fabsf(vb);
        sum_sq += (double)d * d;
        ref_sq += (double)vb * vb;
    }
    double rel_l1 = sum_abs / (sum_ref + 1e-12);
    double rel_l2 = sqrt(sum_sq / (ref_sq + 1e-12));
    printf("[%s] %-22s | max_abs=%.5f | rel_L1=%.5f | rel_L2=%.5f | nan=%d\n",
           (bad == 0 && rel_l2 < 0.02) ? "PASS" : "WARN",
           name, max_abs, rel_l1, rel_l2, bad);
}

// ============================================================================
// Stage-1: Verify Layer 0 (GDN chunked prefill + step-by-step decode) & Layer 3 (Full-Attn prefill + decode)
// against pure-PyTorch reference in ref/stage1/!
// ============================================================================
void run_stage1(const std::string& model_dir, const std::string& ref_dir, const std::string& weight_backend) {
    printf("=== STAGE-1: PER-LAYER NUMERICAL VERIFICATION (vs pure-PyTorch ref) ===\n");
    Qwen35Model model;
    model.init_context();
    load_single_target_layer(model_dir, 0, model.weights, model.ctx.stream, weight_backend);
    load_single_target_layer(model_dir, 3, model.weights, model.ctx.stream, weight_backend);
    model.init_workspace_and_pools(256, 2, 1, 8, 64ULL * 1024 * 1024);

    auto x_in = read_bf16_file(ref_dir + "/x_in.bf16");
    auto l0_ref = read_bf16_file(ref_dir + "/layer0_out.bf16");
    auto l0_state_ref = read_bf16_file(ref_dir + "/layer0_state.bf16");
    auto l0_conv_ref = read_bf16_file(ref_dir + "/layer0_conv.bf16");
    auto l3_ref = read_bf16_file(ref_dir + "/layer3_out.bf16");

    int T = (int)(x_in.size() / Qwen35Config::HIDDEN_SIZE);
    printf("Loaded ref/stage1 with T=%d tokens.\n", T);

    // Prepare device metadata for T tokens
    int32_t *d_pos, *d_slots, *d_bt, *d_cu, *d_kvlen, *d_sslots;
    CUDA_CHECK(cudaMalloc(&d_pos, T * 4));
    CUDA_CHECK(cudaMalloc(&d_slots, T * 4));
    CUDA_CHECK(cudaMalloc(&d_bt, 16 * 4));
    CUDA_CHECK(cudaMalloc(&d_cu, 2 * 4));
    CUDA_CHECK(cudaMalloc(&d_kvlen, 4));
    CUDA_CHECK(cudaMalloc(&d_sslots, 4));

    std::vector<int32_t> h_pos(T), h_slots(T), h_bt(16);
    for (int i = 0; i < T; ++i) { h_pos[i] = i; h_slots[i] = i; }
    for (int i = 0; i < 16; ++i) h_bt[i] = i;
    int32_t h_cu[2] = {0, T};
    int32_t h_kvlen = T;
    int32_t h_slot0 = 0;

    CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), T * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots, h_slots.data(), T * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bt, h_bt.data(), 16 * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cu, h_cu, 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kvlen, &h_kvlen, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sslots, &h_slot0, 4, cudaMemcpyHostToDevice));

    // --- Test 1A: Layer 0 GDN Chunked Prefill ---
    model.reset_state_slot(0, model.ctx.stream);
    CUDA_CHECK(cudaMemcpy(model.ws.residual, x_in.data(), x_in.size() * 2, cudaMemcpyHostToDevice));
    model.forward_layer_prefill(0, model.ws.residual, d_pos, d_slots, d_bt, d_cu, d_kvlen, d_sslots, 1, T, T, 16, model.ctx.stream);
    CUDA_CHECK(cudaStreamSynchronize(model.ctx.stream));

    std::vector<__nv_bfloat16> l0_got(l0_ref.size()), l0_st_got(l0_state_ref.size()), l0_cv_got(l0_conv_ref.size());
    CUDA_CHECK(cudaMemcpy(l0_got.data(), model.ws.residual, l0_got.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(l0_st_got.data(), model.pools.ssm_states[0], l0_st_got.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(l0_cv_got.data(), model.pools.conv_states[0], l0_cv_got.size() * 2, cudaMemcpyDeviceToHost));
    compare_bf16("L0_GDN_prefill_out", l0_got, l0_ref);
    compare_bf16("L0_GDN_prefill_ssm", l0_st_got, l0_state_ref);
    compare_bf16("L0_GDN_prefill_conv", l0_cv_got, l0_conv_ref);

    // --- Test 1B: Layer 0 GDN Token-by-Token Decode (verify decode recurrence matches chunked prefill!) ---
    model.reset_state_slot(0, model.ctx.stream);
    std::vector<__nv_bfloat16> l0_dec_got(l0_ref.size());
    for (int t = 0; t < T; ++t) {
        CUDA_CHECK(cudaMemcpy(model.ws.residual, x_in.data() + (size_t)t * 5120, 5120 * 2, cudaMemcpyHostToDevice));
        const auto& lw = model.weights.gdn[0];
        launch_rmsnorm(model.ws.residual, lw.input_layernorm, model.ws.norm_out, 1, 5120, 1e-6f, true, model.ctx.stream);
        launch_quant_linear(model.ctx.cublas, model.ws.norm_out, lw.in_proj_qkvz, model.ws.qkv_buf, model.ws.d_splitk_scratch, model.ws.d_bf16_w_scratch, 1, model.ctx.stream);
        launch_bf16_linear(model.ctx.cublas, model.ws.norm_out, lw.in_proj_ba, model.ws.ba_buf, 1, 96, 5120, model.ctx.stream, false);
        launch_gdn_conv1d_decode_verify(model.ws.qkv_buf, lw.conv1d_w, model.pools.conv_states[0], nullptr, d_sslots, model.ws.conv_out, 1, 1, true, model.ctx.stream);
        launch_gdn_recurrent_decode_verify(model.ws.conv_out, model.ws.ba_buf, lw.A_log, lw.dt_bias, model.pools.ssm_states[0], nullptr, d_sslots, model.ws.q_buf, 1, 1, true, model.ctx.stream);
        launch_gdn_rmsnorm_gated(model.ws.q_buf, model.ws.qkv_buf, lw.gdn_norm, model.ws.attn_out, 1, 1e-6f, model.ctx.stream);
        launch_quant_linear(model.ctx.cublas, model.ws.attn_out, lw.out_proj, model.ws.sub_out, model.ws.d_splitk_scratch, model.ws.d_bf16_w_scratch, 1, model.ctx.stream);
        launch_fused_add_rmsnorm(model.ws.sub_out, model.ws.residual, lw.post_attention_layernorm, model.ws.norm_out, 1, 5120, 1e-6f, true, model.ctx.stream);
        launch_quant_linear(model.ctx.cublas, model.ws.norm_out, lw.gate_up_proj, model.ws.gate_up_buf, model.ws.d_splitk_scratch, model.ws.d_bf16_w_scratch, 1, model.ctx.stream);
        launch_silu_mul(model.ws.gate_up_buf, model.ws.mlp_mid, 1, 17408, model.ctx.stream);
        launch_quant_linear(model.ctx.cublas, model.ws.mlp_mid, lw.down_proj, model.ws.sub_out, model.ws.d_splitk_scratch, model.ws.d_bf16_w_scratch, 1, model.ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(model.ctx.stream));
        std::vector<__nv_bfloat16> r(5120), s(5120);
        CUDA_CHECK(cudaMemcpy(r.data(), model.ws.residual, 5120 * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(s.data(), model.ws.sub_out, 5120 * 2, cudaMemcpyDeviceToHost));
        for (int i = 0; i < 5120; ++i)
            l0_dec_got[(size_t)t * 5120 + i] = __float2bfloat16(__bfloat162float(r[i]) + __bfloat162float(s[i]));
    }
    compare_bf16("L0_GDN_decode_out", l0_dec_got, l0_ref);

    // --- Test 1C: Layer 3 Full-Attn Prefill (Tensor Core FP8-KV FA2) ---
    CUDA_CHECK(cudaMemcpy(model.ws.residual, l0_ref.data(), l0_ref.size() * 2, cudaMemcpyHostToDevice));
    model.forward_layer_prefill(3, model.ws.residual, d_pos, d_slots, d_bt, d_cu, d_kvlen, d_sslots, 1, T, T, 16, model.ctx.stream);
    CUDA_CHECK(cudaStreamSynchronize(model.ctx.stream));
    std::vector<__nv_bfloat16> l3_got(l3_ref.size());
    CUDA_CHECK(cudaMemcpy(l3_got.data(), model.ws.residual, l3_got.size() * 2, cudaMemcpyDeviceToHost));
    compare_bf16("L3_Attn_prefill_out", l3_got, l3_ref);

    // --- Test 1D: Layer 3 Full-Attn Decode (FlashDecoding Split-K on last 6 tokens!) ---
    int q_len = 6;
    CUDA_CHECK(cudaMemcpy(model.ws.residual, l0_ref.data() + (size_t)(T - q_len) * 5120, (size_t)q_len * 5120 * 2, cudaMemcpyHostToDevice));
    const auto& l3w = model.weights.full_attn[0];
    launch_rmsnorm(model.ws.residual, l3w.input_layernorm, model.ws.norm_out, q_len, 5120, 1e-6f, true, model.ctx.stream);
    launch_quant_linear(model.ctx.cublas, model.ws.norm_out, l3w.qkv_proj, model.ws.qkv_buf, model.ws.d_splitk_scratch, model.ws.d_bf16_w_scratch, q_len, model.ctx.stream);
    launch_full_attn_qkv_prep_and_kv_write(
        model.ws.qkv_buf, l3w.q_norm, l3w.k_norm, model.ws.rope_cos, model.ws.rope_sin, d_pos + (T - q_len),
        model.ws.q_buf, model.ws.gate_buf, model.pools.k_cache[0], model.pools.v_cache[0], d_slots + (T - q_len),
        q_len, 1e-6f, model.ctx.stream);
    launch_paged_attn_decode_fp8(
        model.ws.q_buf, model.ws.gate_buf, model.pools.k_cache[0], model.pools.v_cache[0],
        d_bt, d_kvlen, model.ws.attn_out, model.ws.fa_splitk_out, model.ws.fa_splitk_lse,
        1, q_len, 16, T, model.ctx.stream);
    launch_quant_linear(model.ctx.cublas, model.ws.attn_out, l3w.o_proj, model.ws.sub_out, model.ws.d_splitk_scratch, model.ws.d_bf16_w_scratch, q_len, model.ctx.stream);
    launch_fused_add_rmsnorm(model.ws.sub_out, model.ws.residual, l3w.post_attention_layernorm, model.ws.norm_out, q_len, 5120, 1e-6f, true, model.ctx.stream);
    launch_quant_linear(model.ctx.cublas, model.ws.norm_out, l3w.gate_up_proj, model.ws.gate_up_buf, model.ws.d_splitk_scratch, model.ws.d_bf16_w_scratch, q_len, model.ctx.stream);
    launch_silu_mul(model.ws.gate_up_buf, model.ws.mlp_mid, q_len, 17408, model.ctx.stream);
    launch_quant_linear(model.ctx.cublas, model.ws.mlp_mid, l3w.down_proj, model.ws.sub_out, model.ws.d_splitk_scratch, model.ws.d_bf16_w_scratch, q_len, model.ctx.stream);
    CUDA_CHECK(cudaStreamSynchronize(model.ctx.stream));
    std::vector<__nv_bfloat16> r6(q_len * 5120), s6(q_len * 5120), l3_dec_got(q_len * 5120), l3_tail(q_len * 5120);
    CUDA_CHECK(cudaMemcpy(r6.data(), model.ws.residual, r6.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(s6.data(), model.ws.sub_out, s6.size() * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < r6.size(); ++i) {
        l3_dec_got[i] = __float2bfloat16(__bfloat162float(r6[i]) + __bfloat162float(s6[i]));
        l3_tail[i] = l3_ref[(size_t)(T - q_len) * 5120 + i];
    }
    compare_bf16("L3_Attn_verify6_out", l3_dec_got, l3_tail);

    cudaFree(d_pos); cudaFree(d_slots); cudaFree(d_bt); cudaFree(d_cu); cudaFree(d_kvlen); cudaFree(d_sslots);
    model.destroy();
}

// ============================================================================
// Stage-2 & Stage-3: Full 64-layer Qwen3.8-27B-FP8 greedy decode + DFlash2 + CUDA Graph benchmark!
// ============================================================================
void run_full_bench(
    const std::string& model_dir,
    const std::string& draft_dir,
    const std::string& prompt_ids_path,
    int max_new_tokens,
    bool use_dflash,
    bool draft_fp8,
    bool fp8_lm_head_for_draft,
    int block_size,
    bool use_cuda_graph,
    const std::string& out_ids_path,
    const std::string& weight_backend
) {
    auto prompt_ids = read_i32_file(prompt_ids_path);
    int prompt_len = (int)prompt_ids.size();
    printf("=== FULL MODEL BENCHMARK ===\n");
    printf("prompt_len=%d max_new=%d dflash=%d draft_fp8=%d fp8_lm_draft=%d block_size=%d graph=%d backend=%s\n",
           prompt_len, max_new_tokens, (int)use_dflash, (int)draft_fp8, (int)fp8_lm_head_for_draft,
           block_size, (int)use_cuda_graph, weight_backend.c_str());

    Qwen35Model model;
    model.init_context();
    load_target_weights(model_dir, model.weights, use_dflash && fp8_lm_head_for_draft, model.ctx.stream, weight_backend);

    // Reserve KV pool comfortably inside 64GB
    int max_ctx = std::max(prompt_len + max_new_tokens + 64, 65536);
    int max_blocks = (max_ctx + 15) / 16 + 16;
    size_t kv_bytes = 12ULL * 1024 * 1024 * 1024; // 12 GB (~370K KV tokens)
    model.init_workspace_and_pools(8192, 4, 2, 8, kv_bytes);

    DFlash2Model draft;
    if (use_dflash) {
        draft.init(draft_dir, draft_fp8, 4, 2, 8, model.ctx.stream);
    }

    // Allocate device metadata buffers (static addresses for CUDA Graph!)
    int32_t *d_tok, *d_pos, *d_slots, *d_bt, *d_cu, *d_kvlen, *d_sslots, *d_last_idx;
    int32_t *d_anchor, *d_prefix_len, *d_draft_ids, *d_draft_pos, *d_candidates, *d_top1;
    int32_t *d_accept, *d_commit, *d_bonus, *d_out_toks, *d_ctx_slots;
    CUDA_CHECK(cudaMalloc(&d_tok, 8192 * 4));
    CUDA_CHECK(cudaMalloc(&d_pos, 8192 * 4));
    CUDA_CHECK(cudaMalloc(&d_slots, 8192 * 4));
    CUDA_CHECK(cudaMalloc(&d_bt, max_blocks * 4));
    CUDA_CHECK(cudaMalloc(&d_cu, 2 * 4));
    CUDA_CHECK(cudaMalloc(&d_kvlen, 4));
    CUDA_CHECK(cudaMalloc(&d_sslots, 4));
    CUDA_CHECK(cudaMalloc(&d_last_idx, 4));
    CUDA_CHECK(cudaMalloc(&d_anchor, 4));
    CUDA_CHECK(cudaMalloc(&d_prefix_len, 4));
    CUDA_CHECK(cudaMalloc(&d_draft_ids, 16 * 4));
    CUDA_CHECK(cudaMalloc(&d_draft_pos, 16 * 4));
    CUDA_CHECK(cudaMalloc(&d_candidates, 16 * 4));
    CUDA_CHECK(cudaMalloc(&d_top1, 16 * 4));
    CUDA_CHECK(cudaMalloc(&d_accept, 4));
    CUDA_CHECK(cudaMalloc(&d_commit, 4));
    CUDA_CHECK(cudaMalloc(&d_bonus, 4));
    CUDA_CHECK(cudaMalloc(&d_out_toks, 16 * 4));
    CUDA_CHECK(cudaMalloc(&d_ctx_slots, 8192 * 4));

    std::vector<int32_t> h_bt(max_blocks);
    std::iota(h_bt.begin(), h_bt.end(), 0);
    CUDA_CHECK(cudaMemcpy(d_bt, h_bt.data(), max_blocks * 4, cudaMemcpyHostToDevice));
    int32_t slot0 = 0;
    CUDA_CHECK(cudaMemcpy(d_sslots, &slot0, 4, cudaMemcpyHostToDevice));

    // Pinned host output buffer for async D2H
    int32_t* h_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned, 32 * sizeof(int32_t)));

    // --- PREFILL (chunked at 8192) ---
    model.reset_state_slot(0, model.ctx.stream);
    CUDA_CHECK(cudaStreamSynchronize(model.ctx.stream));

    auto t_prefill_start = std::chrono::steady_clock::now();
    int first_tok = 0;
    for (int off = 0; off < prompt_len; off += 8192) {
        int chunk = std::min(8192, prompt_len - off);
        bool is_last_chunk = (off + chunk == prompt_len);
        std::vector<int32_t> h_pos(chunk), h_sl(chunk);
        for (int i = 0; i < chunk; ++i) { h_pos[i] = off + i; h_sl[i] = off + i; }
        int32_t h_cu[2] = {0, chunk};
        int32_t h_kv = off + chunk;
        int32_t last_i = chunk - 1;

        CUDA_CHECK(cudaMemcpyAsync(d_tok, prompt_ids.data() + off, chunk * 4, cudaMemcpyHostToDevice, model.ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pos, h_pos.data(), chunk * 4, cudaMemcpyHostToDevice, model.ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(d_slots, h_sl.data(), chunk * 4, cudaMemcpyHostToDevice, model.ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(d_cu, h_cu, 8, cudaMemcpyHostToDevice, model.ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(d_kvlen, &h_kv, 4, cudaMemcpyHostToDevice, model.ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(d_last_idx, &last_i, 4, cudaMemcpyHostToDevice, model.ctx.stream));

        bool need_aux = use_dflash && (off + chunk > prompt_len - DFlash2Config::SLIDING_WINDOW);
        model.forward_prefill(
            d_tok, d_pos, d_slots, d_bt, d_cu, d_kvlen, d_sslots,
            1, chunk, chunk, max_blocks, need_aux,
            is_last_chunk ? d_last_idx : nullptr,
            is_last_chunk ? model.ws.logits : nullptr,
            model.ctx.stream);

        if (need_aux) {
            int start_in_chunk = std::max(0, (prompt_len - DFlash2Config::SLIDING_WINDOW) - off);
            int n_mat = chunk - start_in_chunk;
            std::vector<int32_t> h_cs(n_mat, 0);
            CUDA_CHECK(cudaMemcpyAsync(d_ctx_slots, h_cs.data(), n_mat * 4, cudaMemcpyHostToDevice, model.ctx.stream));
            draft.materialize_context_kv(
                model,
                model.ws.aux_hidden + (size_t)start_in_chunk * DFlash2Config::FC_IN_DIM,
                d_pos + start_in_chunk,
                d_ctx_slots, n_mat, model.ctx.stream);
        }

        if (is_last_chunk) {
            launch_argmax(model.ws.logits, d_top1, 1, Qwen35Config::VOCAB_SIZE, model.ctx.stream);
            CUDA_CHECK(cudaMemcpyAsync(h_pinned, d_top1, 4, cudaMemcpyDeviceToHost, model.ctx.stream));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(model.ctx.stream));
    auto t_prefill_end = std::chrono::steady_clock::now();
    first_tok = h_pinned[0];
    double ttft_s = std::chrono::duration<double>(t_prefill_end - t_prefill_start).count();
    printf("[TTFT] prompt=%d tokens -> %.3f s (%.1f tok/s) | first_tok=%d\n",
           prompt_len, ttft_s, prompt_len / ttft_s, first_tok);

    std::vector<int32_t> generated;
    generated.push_back(first_tok);
    int cur_seq_len = prompt_len; // positions 0..prompt_len-1 are already in KV! Next token goes at pos=cur_seq_len!

    // --- DECODE LOOP ---
    auto t_dec_start = std::chrono::steady_clock::now();
    int steps = 0;

    if (!use_dflash) {
        // Non-speculative single-token decode (with optional CUDA graph!)
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;

        auto enqueue_step = [&]() {
            model.forward_decode_verify(
                d_tok, d_pos, d_slots, d_bt, d_kvlen, d_sslots,
                1, 1, max_blocks, prompt_len + max_new_tokens, true, false,
                model.ws.logits, model.ctx.stream);
            launch_argmax(model.ws.logits, d_top1, 1, Qwen35Config::VOCAB_SIZE, model.ctx.stream);
        };

        int next_tok = first_tok;
        while ((int)generated.size() < max_new_tokens) {
            if (next_tok == Qwen35Config::EOS_TOKEN_0 || next_tok == Qwen35Config::EOS_TOKEN_1) break;
            int32_t h_in[4] = {next_tok, cur_seq_len, cur_seq_len, cur_seq_len + 1};
            CUDA_CHECK(cudaMemcpyAsync(d_tok, &h_in[0], 4, cudaMemcpyHostToDevice, model.ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(d_pos, &h_in[1], 4, cudaMemcpyHostToDevice, model.ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(d_slots, &h_in[2], 4, cudaMemcpyHostToDevice, model.ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(d_kvlen, &h_in[3], 4, cudaMemcpyHostToDevice, model.ctx.stream));

            if (use_cuda_graph && steps == 1 && !graph_exec) {
                CUDA_CHECK(cudaStreamSynchronize(model.ctx.stream));
                CUDA_CHECK(cudaStreamBeginCapture(model.ctx.stream, cudaStreamCaptureModeGlobal));
                enqueue_step();
                CUDA_CHECK(cudaStreamEndCapture(model.ctx.stream, &graph));
                CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
                CUDA_CHECK(cudaGraphLaunch(graph_exec, model.ctx.stream));
            } else if (graph_exec) {
                CUDA_CHECK(cudaGraphLaunch(graph_exec, model.ctx.stream));
            } else {
                enqueue_step();
            }
            CUDA_CHECK(cudaMemcpyAsync(h_pinned, d_top1, 4, cudaMemcpyDeviceToHost, model.ctx.stream));
            CUDA_CHECK(cudaStreamSynchronize(model.ctx.stream));
            next_tok = h_pinned[0];
            generated.push_back(next_tok);
            ++cur_seq_len;
            ++steps;
        }
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
    } else {
        // DFlash2 Speculative Decode Loop (100% single CUDA graph for Draft + Verify + Accept + GDN Commit + Draft KV Append!)
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;

        // Device helper to build masked draft ctx_slots for commit_len tokens inside graph!
        auto enqueue_dflash_step = [&]() {
            // 1) Propose block_size candidates [anchor, d_1..d_{block-1}]
            draft.propose_candidates(
                model, d_anchor, d_prefix_len, d_sslots,
                nullptr, nullptr, nullptr,
                d_draft_ids, d_draft_pos, d_candidates,
                1, block_size, model.ctx.stream);

            // 2) Verify all block_size tokens through target 64 layers (capture aux_hidden for next step!)
            // Note: d_draft_pos holds [prefix_len + 0 .. prefix_len + block_size - 1]
            //       and d_slots holds [prefix_len + 0 .. prefix_len + block_size - 1]!
            model.forward_decode_verify(
                d_candidates, d_draft_pos, d_draft_pos, d_bt, d_kvlen, d_sslots,
                1, block_size, max_blocks, prompt_len + max_new_tokens,
                /*commit_gdn_inplace=*/false, /*capture_aux_hidden=*/true,
                model.ws.logits, model.ctx.stream);

            // 3) Greedy argmax on target logits + accept longest prefix + bonus
            launch_argmax(model.ws.logits, d_top1, block_size, Qwen35Config::VOCAB_SIZE, model.ctx.stream);
            launch_dflash_accept_greedy(
                d_candidates, d_top1, d_accept, d_commit, d_bonus, d_out_toks, nullptr,
                1, block_size, model.ctx.stream);

            // 4) Commit accepted GDN states (replays ONLY commit_len steps across all 48 layers!)
            model.commit_gdn_verify(d_sslots, d_commit, 1, block_size, model.ctx.stream);

            // 5) Materialize ONLY committed aux_hidden into draft KV ring buffer (skips rejected tokens so ring never wraps onto valid context!)
            draft.materialize_context_kv(
                model, model.ws.aux_hidden, d_draft_pos, d_ctx_slots, block_size, model.ctx.stream, d_commit, block_size);
        };

        // Pre-fill d_ctx_slots with 0 for all block_size positions
        std::vector<int32_t> zeros(block_size, 0);
        CUDA_CHECK(cudaMemcpy(d_ctx_slots, zeros.data(), block_size * 4, cudaMemcpyHostToDevice));

        int anchor = first_tok;
        long total_committed = 0;
        while ((int)generated.size() < max_new_tokens) {
            if (anchor == Qwen35Config::EOS_TOKEN_0 || anchor == Qwen35Config::EOS_TOKEN_1) break;
            int32_t kv_tot = cur_seq_len + block_size;
            CUDA_CHECK(cudaMemcpyAsync(d_anchor, &anchor, 4, cudaMemcpyHostToDevice, model.ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(d_prefix_len, &cur_seq_len, 4, cudaMemcpyHostToDevice, model.ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(d_kvlen, &kv_tot, 4, cudaMemcpyHostToDevice, model.ctx.stream));

            if (use_cuda_graph && steps == 1 && !graph_exec) {
                CUDA_CHECK(cudaStreamSynchronize(model.ctx.stream));
                CUDA_CHECK(cudaStreamBeginCapture(model.ctx.stream, cudaStreamCaptureModeGlobal));
                enqueue_dflash_step();
                CUDA_CHECK(cudaStreamEndCapture(model.ctx.stream, &graph));
                CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
                CUDA_CHECK(cudaGraphLaunch(graph_exec, model.ctx.stream));
            } else if (graph_exec) {
                CUDA_CHECK(cudaGraphLaunch(graph_exec, model.ctx.stream));
            } else {
                enqueue_dflash_step();
            }

            // Copy commit_len, bonus, and out_tokens[block_size] in tiny D2H
            CUDA_CHECK(cudaMemcpyAsync(&h_pinned[0], d_commit, 4, cudaMemcpyDeviceToHost, model.ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(&h_pinned[1], d_bonus, 4, cudaMemcpyDeviceToHost, model.ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(&h_pinned[2], d_out_toks, block_size * 4, cudaMemcpyDeviceToHost, model.ctx.stream));
            CUDA_CHECK(cudaStreamSynchronize(model.ctx.stream));

            int c_len = h_pinned[0];
            int bonus = h_pinned[1];
            bool hit_eos = false;
            for (int k = 0; k < c_len && (int)generated.size() < max_new_tokens; ++k) {
                int tok = h_pinned[2 + k];
                generated.push_back(tok);
                if (tok == Qwen35Config::EOS_TOKEN_0 || tok == Qwen35Config::EOS_TOKEN_1) {
                    hit_eos = true;
                    break;
                }
            }
            cur_seq_len += c_len;
            total_committed += c_len;
            anchor = bonus;
            ++steps;
            if (hit_eos) break;
        }
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
        printf("[DFLASH] steps=%d total_committed=%ld mean_accept_len=%.3f\n",
               steps, total_committed, steps > 0 ? (double)total_committed / steps : 0.0);
    }

    auto t_dec_end = std::chrono::steady_clock::now();
    double dec_s = std::chrono::duration<double>(t_dec_end - t_dec_start).count();
    int gen_after_first = (int)generated.size() - 1;
    printf("[DECODE] generated=%zu tokens in %.3f s -> %.2f tok/s (%.2f ms/step)\n",
           generated.size(), dec_s, gen_after_first / std::max(dec_s, 1e-6),
           steps > 0 ? (dec_s * 1000.0 / steps) : 0.0);

    if (!out_ids_path.empty()) {
        std::ofstream of(out_ids_path, std::ios::binary);
        of.write(reinterpret_cast<const char*>(generated.data()), generated.size() * 4);
    }

    cudaFreeHost(h_pinned);
    cudaFree(d_tok); cudaFree(d_pos); cudaFree(d_slots); cudaFree(d_bt); cudaFree(d_cu);
    cudaFree(d_kvlen); cudaFree(d_sslots); cudaFree(d_last_idx); cudaFree(d_anchor);
    cudaFree(d_prefix_len); cudaFree(d_draft_ids); cudaFree(d_draft_pos); cudaFree(d_candidates);
    cudaFree(d_top1); cudaFree(d_accept); cudaFree(d_commit); cudaFree(d_bonus);
    cudaFree(d_out_toks); cudaFree(d_ctx_slots);
    if (use_dflash) draft.destroy();
    model.destroy();
}

// ============================================================================
// Suite mode: Load model ONCE, run all golden prompts across Stage-2 (no spec) & Stage-3 (DFlash2 b=6, b=8)!
// ============================================================================
void run_suite(
    const std::string& model_dir,
    const std::string& draft_dir,
    const std::string& prompt_dir,
    bool draft_fp8,
    bool fp8_lm_draft,
    const std::string& weight_backend
) {
    printf("=== FULL SUITE BENCHMARK (single weight load, draft_fp8=%d, fp8_lm_draft=%d, backend=%s) ===\n",
           (int)draft_fp8, (int)fp8_lm_draft, weight_backend.c_str());

    Qwen35Model model;
    model.init_context();
    load_target_weights(model_dir, model.weights, fp8_lm_draft, model.ctx.stream, weight_backend);

    int max_ctx = 65536;
    int max_blocks = (max_ctx + 15) / 16 + 32;
    size_t kv_bytes = 12ULL * 1024 * 1024 * 1024;
    model.init_workspace_and_pools(8192, 4, 2, 8, kv_bytes);

    DFlash2Model draft;
    draft.init(draft_dir, draft_fp8, 4, 2, 8, model.ctx.stream);

    int32_t *d_tok, *d_pos, *d_slots, *d_bt, *d_cu, *d_kvlen, *d_sslots, *d_last_idx;
    int32_t *d_anchor, *d_prefix_len, *d_draft_ids, *d_draft_pos, *d_candidates, *d_top1;
    int32_t *d_accept, *d_commit, *d_bonus, *d_out_toks, *d_ctx_slots;
    CUDA_CHECK(cudaMalloc(&d_tok, 8192 * 4));
    CUDA_CHECK(cudaMalloc(&d_pos, 8192 * 4));
    CUDA_CHECK(cudaMalloc(&d_slots, 8192 * 4));
    CUDA_CHECK(cudaMalloc(&d_bt, max_blocks * 4));
    CUDA_CHECK(cudaMalloc(&d_cu, 2 * 4));
    CUDA_CHECK(cudaMalloc(&d_kvlen, 4));
    CUDA_CHECK(cudaMalloc(&d_sslots, 4));
    CUDA_CHECK(cudaMalloc(&d_last_idx, 4));
    CUDA_CHECK(cudaMalloc(&d_anchor, 4));
    CUDA_CHECK(cudaMalloc(&d_prefix_len, 4));
    CUDA_CHECK(cudaMalloc(&d_draft_ids, 16 * 4));
    CUDA_CHECK(cudaMalloc(&d_draft_pos, 16 * 4));
    CUDA_CHECK(cudaMalloc(&d_candidates, 16 * 4));
    CUDA_CHECK(cudaMalloc(&d_top1, 16 * 4));
    CUDA_CHECK(cudaMalloc(&d_accept, 4));
    CUDA_CHECK(cudaMalloc(&d_commit, 4));
    CUDA_CHECK(cudaMalloc(&d_bonus, 4));
    CUDA_CHECK(cudaMalloc(&d_out_toks, 16 * 4));
    CUDA_CHECK(cudaMalloc(&d_ctx_slots, 8192 * 4));

    std::vector<int32_t> h_bt(max_blocks);
    std::iota(h_bt.begin(), h_bt.end(), 0);
    CUDA_CHECK(cudaMemcpy(d_bt, h_bt.data(), max_blocks * 4, cudaMemcpyHostToDevice));
    int32_t slot0 = 0;
    CUDA_CHECK(cudaMemcpy(d_sslots, &slot0, 4, cudaMemcpyHostToDevice));

    int32_t* h_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned, 32 * sizeof(int32_t)));

    auto run_one = [&](const std::string& name, int max_new, bool use_dflash, int block_size, const std::string& tag) {
        auto prompt_ids = read_i32_file(prompt_dir + "/" + name + ".bin");
        int prompt_len = (int)prompt_ids.size();
        model.reset_state_slot(0, model.ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(model.ctx.stream));

        auto t0 = std::chrono::steady_clock::now();
        for (int off = 0; off < prompt_len; off += 8192) {
            int chunk = std::min(8192, prompt_len - off);
            bool is_last = (off + chunk == prompt_len);
            std::vector<int32_t> h_pos(chunk), h_sl(chunk);
            for (int i = 0; i < chunk; ++i) { h_pos[i] = off + i; h_sl[i] = off + i; }
            int32_t h_cu[2] = {0, chunk};
            int32_t h_kv = off + chunk;
            int32_t last_i = chunk - 1;
            CUDA_CHECK(cudaMemcpyAsync(d_tok, prompt_ids.data() + off, chunk * 4, cudaMemcpyHostToDevice, model.ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(d_pos, h_pos.data(), chunk * 4, cudaMemcpyHostToDevice, model.ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(d_slots, h_sl.data(), chunk * 4, cudaMemcpyHostToDevice, model.ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(d_cu, h_cu, 8, cudaMemcpyHostToDevice, model.ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(d_kvlen, &h_kv, 4, cudaMemcpyHostToDevice, model.ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(d_last_idx, &last_i, 4, cudaMemcpyHostToDevice, model.ctx.stream));

            bool need_aux = use_dflash && (off + chunk > prompt_len - DFlash2Config::SLIDING_WINDOW);
            model.forward_prefill(
                d_tok, d_pos, d_slots, d_bt, d_cu, d_kvlen, d_sslots,
                1, chunk, chunk, max_blocks, need_aux,
                is_last ? d_last_idx : nullptr,
                is_last ? model.ws.logits : nullptr,
                model.ctx.stream);
            if (need_aux) {
                int start_in = std::max(0, (prompt_len - DFlash2Config::SLIDING_WINDOW) - off);
                int n_mat = chunk - start_in;
                std::vector<int32_t> h_cs(n_mat, 0);
                CUDA_CHECK(cudaMemcpyAsync(d_ctx_slots, h_cs.data(), n_mat * 4, cudaMemcpyHostToDevice, model.ctx.stream));
                draft.materialize_context_kv(
                    model, model.ws.aux_hidden + (size_t)start_in * DFlash2Config::FC_IN_DIM,
                    d_pos + start_in, d_ctx_slots, n_mat, model.ctx.stream);
            }
            if (is_last) {
                launch_argmax(model.ws.logits, d_top1, 1, Qwen35Config::VOCAB_SIZE, model.ctx.stream);
                CUDA_CHECK(cudaMemcpyAsync(h_pinned, d_top1, 4, cudaMemcpyDeviceToHost, model.ctx.stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(model.ctx.stream));
        auto t1 = std::chrono::steady_clock::now();
        double ttft = std::chrono::duration<double>(t1 - t0).count();
        int first_tok = h_pinned[0];

        std::vector<int32_t> gen;
        gen.push_back(first_tok);
        int cur_seq_len = prompt_len;
        int steps = 0;
        long total_committed = 0;

        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        int seq_hint = prompt_len + max_new + 16;

        // Warmup & capture CUDA graph BEFORE timing? No, capture on step 1 (takes ~3ms), report steady-state AND total!
        auto t_dec0 = std::chrono::steady_clock::now();
        auto t_steady0 = t_dec0;
        int steady_toks = 0;
        int steady_steps = 0;

        if (!use_dflash) {
            auto enq = [&]() {
                model.forward_decode_verify(
                    d_tok, d_pos, d_slots, d_bt, d_kvlen, d_sslots,
                    1, 1, max_blocks, seq_hint, true, false, model.ws.logits, model.ctx.stream);
                launch_argmax(model.ws.logits, d_top1, 1, Qwen35Config::VOCAB_SIZE, model.ctx.stream);
            };
            int next_tok = first_tok;
            while ((int)gen.size() < max_new) {
                if (next_tok == Qwen35Config::EOS_TOKEN_0 || next_tok == Qwen35Config::EOS_TOKEN_1) break;
                int32_t h_in[4] = {next_tok, cur_seq_len, cur_seq_len, cur_seq_len + 1};
                CUDA_CHECK(cudaMemcpyAsync(d_tok, &h_in[0], 4, cudaMemcpyHostToDevice, model.ctx.stream));
                CUDA_CHECK(cudaMemcpyAsync(d_pos, &h_in[1], 4, cudaMemcpyHostToDevice, model.ctx.stream));
                CUDA_CHECK(cudaMemcpyAsync(d_slots, &h_in[2], 4, cudaMemcpyHostToDevice, model.ctx.stream));
                CUDA_CHECK(cudaMemcpyAsync(d_kvlen, &h_in[3], 4, cudaMemcpyHostToDevice, model.ctx.stream));
                if (steps == 1 && !graph_exec) {
                    CUDA_CHECK(cudaStreamSynchronize(model.ctx.stream));
                    CUDA_CHECK(cudaStreamBeginCapture(model.ctx.stream, cudaStreamCaptureModeGlobal));
                    enq();
                    CUDA_CHECK(cudaStreamEndCapture(model.ctx.stream, &graph));
                    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
                    CUDA_CHECK(cudaGraphLaunch(graph_exec, model.ctx.stream));
                } else if (graph_exec) {
                    CUDA_CHECK(cudaGraphLaunch(graph_exec, model.ctx.stream));
                } else {
                    enq();
                }
                CUDA_CHECK(cudaMemcpyAsync(h_pinned, d_top1, 4, cudaMemcpyDeviceToHost, model.ctx.stream));
                CUDA_CHECK(cudaStreamSynchronize(model.ctx.stream));
                if (steps == 2) { t_steady0 = std::chrono::steady_clock::now(); steady_toks = 0; steady_steps = 0; }
                next_tok = h_pinned[0];
                gen.push_back(next_tok);
                ++cur_seq_len; ++steps; ++total_committed;
                if (steps > 2) { ++steady_toks; ++steady_steps; }
            }
        } else {
            std::vector<int32_t> zeros(block_size, 0);
            CUDA_CHECK(cudaMemcpy(d_ctx_slots, zeros.data(), block_size * 4, cudaMemcpyHostToDevice));
            auto enq = [&]() {
                draft.propose_candidates(
                    model, d_anchor, d_prefix_len, d_sslots,
                    nullptr, nullptr, nullptr,
                    d_draft_ids, d_draft_pos, d_candidates,
                    1, block_size, model.ctx.stream);
                model.forward_decode_verify(
                    d_candidates, d_draft_pos, d_draft_pos, d_bt, d_kvlen, d_sslots,
                    1, block_size, max_blocks, seq_hint, false, true, model.ws.logits, model.ctx.stream);
                launch_argmax(model.ws.logits, d_top1, block_size, Qwen35Config::VOCAB_SIZE, model.ctx.stream);
                launch_dflash_accept_greedy(
                    d_candidates, d_top1, d_accept, d_commit, d_bonus, d_out_toks, nullptr,
                    1, block_size, model.ctx.stream);
                model.commit_gdn_verify(d_sslots, d_commit, 1, block_size, model.ctx.stream);
                draft.materialize_context_kv(
                    model, model.ws.aux_hidden, d_draft_pos, d_ctx_slots, block_size, model.ctx.stream, d_commit, block_size);
            };
            int anchor = first_tok;
            while ((int)gen.size() < max_new) {
                if (anchor == Qwen35Config::EOS_TOKEN_0 || anchor == Qwen35Config::EOS_TOKEN_1) break;
                int32_t kv_tot = cur_seq_len + block_size;
                CUDA_CHECK(cudaMemcpyAsync(d_anchor, &anchor, 4, cudaMemcpyHostToDevice, model.ctx.stream));
                CUDA_CHECK(cudaMemcpyAsync(d_prefix_len, &cur_seq_len, 4, cudaMemcpyHostToDevice, model.ctx.stream));
                CUDA_CHECK(cudaMemcpyAsync(d_kvlen, &kv_tot, 4, cudaMemcpyHostToDevice, model.ctx.stream));
                if (steps == 1 && !graph_exec) {
                    CUDA_CHECK(cudaStreamSynchronize(model.ctx.stream));
                    CUDA_CHECK(cudaStreamBeginCapture(model.ctx.stream, cudaStreamCaptureModeGlobal));
                    enq();
                    CUDA_CHECK(cudaStreamEndCapture(model.ctx.stream, &graph));
                    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
                    CUDA_CHECK(cudaGraphLaunch(graph_exec, model.ctx.stream));
                } else if (graph_exec) {
                    CUDA_CHECK(cudaGraphLaunch(graph_exec, model.ctx.stream));
                } else {
                    enq();
                }
                CUDA_CHECK(cudaMemcpyAsync(&h_pinned[0], d_commit, 4, cudaMemcpyDeviceToHost, model.ctx.stream));
                CUDA_CHECK(cudaMemcpyAsync(&h_pinned[1], d_bonus, 4, cudaMemcpyDeviceToHost, model.ctx.stream));
                CUDA_CHECK(cudaMemcpyAsync(&h_pinned[2], d_out_toks, block_size * 4, cudaMemcpyDeviceToHost, model.ctx.stream));
                CUDA_CHECK(cudaStreamSynchronize(model.ctx.stream));
                if (steps == 2) { t_steady0 = std::chrono::steady_clock::now(); steady_toks = 0; steady_steps = 0; }
                int c_len = h_pinned[0];
                int bonus = h_pinned[1];
                bool hit_eos = false;
                int added = 0;
                for (int k = 0; k < c_len && (int)gen.size() < max_new; ++k) {
                    int tok = h_pinned[2 + k];
                    gen.push_back(tok);
                    ++added;
                    if (tok == Qwen35Config::EOS_TOKEN_0 || tok == Qwen35Config::EOS_TOKEN_1) { hit_eos = true; break; }
                }
                cur_seq_len += c_len;
                total_committed += c_len;
                anchor = bonus;
                ++steps;
                if (steps > 2) { steady_toks += added; ++steady_steps; }
                if (hit_eos) break;
            }
        }
        auto t_dec1 = std::chrono::steady_clock::now();
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);

        double dec_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
        double steady_s = std::chrono::duration<double>(t_dec1 - t_steady0).count();
        double e2e_tps = ((int)gen.size() - 1) / std::max(dec_s, 1e-6);
        double steady_tps = steady_toks / std::max(steady_s, 1e-6);
        double step_ms = steady_steps > 0 ? (steady_s * 1000.0 / steady_steps) : 0.0;
        double acc = steps > 0 ? (double)total_committed / steps : 0.0;
        printf("%-12s %-10s | prompt=%5d out=%3zu | TTFT=%6.3fs | e2e=%6.2f | steady=%6.2f tok/s | step=%5.2fms | acc=%.2f\n",
               name.c_str(), tag.c_str(), prompt_len, gen.size(), ttft, e2e_tps, steady_tps, step_ms, acc);
        fflush(stdout);

        std::ofstream of(prompt_dir + "/" + name + "__" + tag + ".out.bin", std::ios::binary);
        of.write(reinterpret_cast<const char*>(gen.data()), gen.size() * 4);
    };

    // Quick component breakdown profile at ctx=32768 (10 iterations with cudaEvents!)
    auto run_breakdown = [&](int ctx_len, int block_size) {
        model.reset_state_slot(0, model.ctx.stream);
        std::vector<int32_t> zeros(block_size, 0), pos_v(block_size);
        for (int i = 0; i < block_size; ++i) pos_v[i] = ctx_len + i;
        int32_t kv_tot = ctx_len + block_size;
        int32_t anc = 100;
        CUDA_CHECK(cudaMemcpy(d_anchor, &anc, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_prefix_len, &ctx_len, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kvlen, &kv_tot, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_draft_pos, pos_v.data(), block_size * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ctx_slots, zeros.data(), block_size * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_candidates, zeros.data(), block_size * 4, cudaMemcpyHostToDevice));

        cudaEvent_t ev[8];
        for (int i = 0; i < 8; ++i) CUDA_CHECK(cudaEventCreate(&ev[i]));
        auto stream = model.ctx.stream;
        // Warmup 2
        for (int w = 0; w < 2; ++w) {
            draft.propose_candidates(model, d_anchor, d_prefix_len, d_sslots, nullptr, nullptr, nullptr, d_draft_ids, d_draft_pos, d_candidates, 1, block_size, stream);
            model.forward_decode_verify(d_candidates, d_draft_pos, d_draft_pos, d_bt, d_kvlen, d_sslots, 1, block_size, max_blocks, ctx_len + block_size, false, true, model.ws.logits, stream);
            launch_argmax(model.ws.logits, d_top1, block_size, Qwen35Config::VOCAB_SIZE, stream);
            launch_dflash_accept_greedy(d_candidates, d_top1, d_accept, d_commit, d_bonus, d_out_toks, nullptr, 1, block_size, stream);
            model.commit_gdn_verify(d_sslots, d_commit, 1, block_size, stream);
            draft.materialize_context_kv(model, model.ws.aux_hidden, d_draft_pos, d_ctx_slots, block_size, stream, d_commit, block_size);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double sum_ms[6] = {};
        int IT = 10;
        for (int it = 0; it < IT; ++it) {
            CUDA_CHECK(cudaEventRecord(ev[0], stream));
            draft.propose_candidates(model, d_anchor, d_prefix_len, d_sslots, nullptr, nullptr, nullptr, d_draft_ids, d_draft_pos, d_candidates, 1, block_size, stream);
            CUDA_CHECK(cudaEventRecord(ev[1], stream));
            model.forward_decode_verify(d_candidates, d_draft_pos, d_draft_pos, d_bt, d_kvlen, d_sslots, 1, block_size, max_blocks, ctx_len + block_size, false, true, model.ws.logits, stream);
            CUDA_CHECK(cudaEventRecord(ev[2], stream));
            launch_argmax(model.ws.logits, d_top1, block_size, Qwen35Config::VOCAB_SIZE, stream);
            launch_dflash_accept_greedy(d_candidates, d_top1, d_accept, d_commit, d_bonus, d_out_toks, nullptr, 1, block_size, stream);
            CUDA_CHECK(cudaEventRecord(ev[3], stream));
            model.commit_gdn_verify(d_sslots, d_commit, 1, block_size, stream);
            CUDA_CHECK(cudaEventRecord(ev[4], stream));
            draft.materialize_context_kv(model, model.ws.aux_hidden, d_draft_pos, d_ctx_slots, block_size, stream, d_commit, block_size);
            CUDA_CHECK(cudaEventRecord(ev[5], stream));
            CUDA_CHECK(cudaEventSynchronize(ev[5]));
            for (int k = 0; k < 5; ++k) {
                float t = 0; CUDA_CHECK(cudaEventElapsedTime(&t, ev[k], ev[k + 1]));
                sum_ms[k] += t;
            }
            float tot = 0; CUDA_CHECK(cudaEventElapsedTime(&tot, ev[0], ev[5]));
            sum_ms[5] += tot;
        }
        printf("[BREAKDOWN ctx=%5d b=%d] draft=%.2fms | target64+lm=%.2fms | accept=%.2fms | gdn_commit=%.2fms | draft_kv=%.2fms | TOTAL=%.2fms\n",
               ctx_len, block_size, sum_ms[0]/IT, sum_ms[1]/IT, sum_ms[2]/IT, sum_ms[3]/IT, sum_ms[4]/IT, sum_ms[5]/IT);
        for (int i = 0; i < 8; ++i) cudaEventDestroy(ev[i]);
    };

    run_breakdown(128, 6);
    run_breakdown(32768, 6);
    run_breakdown(32768, 8);

    printf("\n%-12s %-10s | %12s %7s | %9s | %10s | %16s | %9s | %s\n",
           "Prompt", "Mode", "Prompt", "Out", "TTFT", "E2E_tps", "Steady_tps", "Step_ms", "AccLen");
    printf("------------------------------------------------------------------------------------------------------------\n");

    // 1) Stage-2 baseline (no spec) on short_code
    run_one("short_code", 256, false, 1, "nospec");

    // 2) Stage-3 DFlash2 block_size=6 (apples-to-apples with SGLang's --speculative-num-draft-tokens 6!)
    run_one("short_code", 256, true, 6, "dflash_b6");
    run_one("short_math", 256, true, 6, "dflash_b6");
    run_one("short_tech", 337, true, 6, "dflash_b6");
    run_one("chat_short", 512, true, 6, "dflash_b6");
    run_one("chat_8k",    512, true, 6, "dflash_b6");
    run_one("chat_32k",   512, true, 6, "dflash_b6");

    // 3) Stage-3 DFlash2 block_size=8 (checkpoint native!)
    run_one("short_code", 256, true, 8, "dflash_b8");
    run_one("short_math", 256, true, 8, "dflash_b8");
    run_one("short_tech", 337, true, 8, "dflash_b8");
    run_one("chat_short", 512, true, 8, "dflash_b8");
    run_one("chat_8k",    512, true, 8, "dflash_b8");
    run_one("chat_32k",   512, true, 8, "dflash_b8");

    cudaFreeHost(h_pinned);
    cudaFree(d_tok); cudaFree(d_pos); cudaFree(d_slots); cudaFree(d_bt); cudaFree(d_cu);
    cudaFree(d_kvlen); cudaFree(d_sslots); cudaFree(d_last_idx); cudaFree(d_anchor);
    cudaFree(d_prefix_len); cudaFree(d_draft_ids); cudaFree(d_draft_pos); cudaFree(d_candidates);
    cudaFree(d_top1); cudaFree(d_accept); cudaFree(d_commit); cudaFree(d_bonus);
    cudaFree(d_out_toks); cudaFree(d_ctx_slots);
    draft.destroy();
    model.destroy();
}

int run_weight_self_tests(const std::string& requested_backend) {
    printf("=== CURATED WEIGHT PLUGIN SELF-TESTS (compiled: %s) ===\n",
           WeightRegistry::compiled_ids_string().c_str());
    CudaContext ctx;
    ctx.init();
    bool all_ok = true;
    int ran = 0;

    if (requested_backend.empty() || requested_backend == "auto") {
        for (const auto* b : WeightRegistry::all()) {
            printf("--- Testing plugin '%s': %s ---\n", b->id(), b->description());
            bool ok = b->self_test(ctx.cublas, ctx.stream);
            if (!ok) all_ok = false;
            ++ran;
        }
    } else {
        const auto* b = WeightRegistry::find(requested_backend);
        if (!b) {
            fprintf(stderr, "FATAL: weight backend '%s' is not compiled in (available: %s)\n",
                    requested_backend.c_str(), WeightRegistry::compiled_ids_string().c_str());
            ctx.destroy();
            return 1;
        }
        printf("--- Testing plugin '%s': %s ---\n", b->id(), b->description());
        all_ok = b->self_test(ctx.cublas, ctx.stream);
        ran = 1;
    }

    ctx.destroy();
    printf("=== WEIGHT SELF-TESTS: %d plugin(s) ran -> %s ===\n", ran, (all_ok && ran > 0) ? "ALL PASS" : "FAILED");
    return (all_ok && ran > 0) ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    std::string mode = "stage1";
    std::string model_dir = "./models/Qwen3.8-27B-FP8";
    std::string draft_dir = "./models/Qwen3.8-27B-DFlash2";
    std::string ref_dir = "ref/stage1";
    std::string prompt_ids = "ref/stage1/tok_ids.bin";
    std::string prompt_dir = "ref/prompts";
    std::string out_ids;
    std::string weight_backend = "auto";
    int max_new = 256;
    int block_size = 8;
    bool use_dflash = false;
    bool draft_fp8 = true;
    bool fp8_lm_draft = true;
    bool use_graph = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--stage1") mode = "stage1";
        else if (a == "--stage2") { mode = "full"; use_dflash = false; }
        else if (a == "--stage3") { mode = "full"; use_dflash = true; }
        else if (a == "--suite") mode = "suite";
        else if (a == "--test-weights") mode = "test_weights";
        else if (a == "--probe-weights") mode = "probe_weights";
        else if (a == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--draft-dir" && i + 1 < argc) draft_dir = argv[++i];
        else if (a == "--weight-backend" && i + 1 < argc) weight_backend = argv[++i];
        else if (a == "--ref-dir" && i + 1 < argc) ref_dir = argv[++i];
        else if (a == "--prompt-ids" && i + 1 < argc) prompt_ids = argv[++i];
        else if (a == "--prompt-dir" && i + 1 < argc) prompt_dir = argv[++i];
        else if (a == "--out-ids" && i + 1 < argc) out_ids = argv[++i];
        else if (a == "--max-new" && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (a == "--block-size" && i + 1 < argc) block_size = atoi(argv[++i]);
        else if (a == "--draft-bf16") draft_fp8 = false;
        else if (a == "--bf16-lm-draft") fp8_lm_draft = false;
        else if (a == "--no-graph") use_graph = false;
    }

    try {
        if (mode == "test_weights") {
            return run_weight_self_tests(weight_backend);
        } else if (mode == "probe_weights") {
            const WeightBackend* b = WeightRegistry::resolve(model_dir, weight_backend);
            printf("[probe] QUALIFIED: %s -> %s (%s)\n", model_dir.c_str(), b->id(), b->description());
            return 0;
        } else if (mode == "stage1") {
            run_stage1(model_dir, ref_dir, weight_backend);
        } else if (mode == "suite") {
            run_suite(model_dir, draft_dir, prompt_dir, draft_fp8, fp8_lm_draft, weight_backend);
        } else {
            run_full_bench(model_dir, draft_dir, prompt_ids, max_new, use_dflash, draft_fp8, fp8_lm_draft, block_size, use_graph, out_ids, weight_backend);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return 0;
}
