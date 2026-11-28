#include "loader/model_loader.hpp"
#include "loader/safetensors.hpp"
#include "weights/staging_uploader.hpp"
#include "weights/weight_backend.hpp"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace amperegen {

namespace {

void load_fused_bf16_as_fp8(
    const WeightBackend* fp8_backend,
    StagingUploader& up,
    const SafetensorsFile& sf,
    const std::vector<std::string>& names,
    int total_N, int K,
    GpuArena& arena,
    QuantLinearWeight& out,
    cudaStream_t stream
) {
    size_t bf16_bytes = (size_t)total_N * K * 2;
    up.stage_concat_to_device(sf, names, bf16_bytes, "draft BF16->FP8", stream);
    if (!fp8_backend->quantize_from_bf16_device(
            reinterpret_cast<const __nv_bfloat16*>(up.d_stage),
            total_N, K, arena, out, stream)) {
        throw std::runtime_error("qwen_fp8 plugin failed to quantize draft BF16 tensor");
    }
}

} // namespace

void load_single_target_layer(
    const std::string& model_dir,
    int layer_idx,
    TargetWeights& dst,
    cudaStream_t stream,
    const std::string& weight_backend
) {
    const WeightBackend* backend = WeightRegistry::resolve(model_dir, weight_backend);
    fprintf(stderr, "[loader] Weight plugin: %s (%s)\n", backend->id(), backend->description());
    backend->load_single_layer(model_dir, layer_idx, dst, stream);
}

void load_target_weights(
    const std::string& model_dir,
    TargetWeights& dst,
    bool build_fp8_lm_head,
    cudaStream_t stream,
    const std::string& weight_backend
) {
    const WeightBackend* backend = WeightRegistry::resolve(model_dir, weight_backend);
    fprintf(stderr, "[loader] Weight plugin: %s (%s)\n", backend->id(), backend->description());
    backend->load_target(model_dir, dst, build_fp8_lm_head, stream);
}

void load_draft_weights(
    const std::string& draft_dir,
    DraftWeights& dst,
    bool quantize_to_fp8,
    cudaStream_t stream
) {
    auto t0 = std::chrono::steady_clock::now();

    const WeightBackend* fp8_backend = nullptr;
    if (quantize_to_fp8) {
        fp8_backend = WeightRegistry::find("qwen_fp8");
        if (!fp8_backend) {
            fprintf(stderr, "[loader] Note: qwen_fp8 plugin not compiled in; loading DFlash2 draft in BF16.\n");
            quantize_to_fp8 = false;
        }
    }

    dst.use_fp8 = quantize_to_fp8;
    // FP8 draft: ~1.8 GB weights + 0.25 GB codebooks + 0.13 GB conv_proj = ~2.3 GB (vs 4.0 GB BF16)
    size_t arena_bytes = quantize_to_fp8 ? (2600ULL * 1024 * 1024) : (4200ULL * 1024 * 1024);
    dst.arena.init(arena_bytes);

    StagingUploader up;
    up.init(400ULL * 1024 * 1024); // Fits BF16 gate_up (356 MB) & fc (262 MB)!

    try {
        SafetensorsFile sf;
        sf.open(draft_dir + "/model.safetensors");

        if (quantize_to_fp8) {
            load_fused_bf16_as_fp8(
                fp8_backend, up, sf, {"fc.weight"},
                DFlash2Config::HIDDEN_SIZE, DFlash2Config::FC_IN_DIM,
                dst.arena, dst.fc_fp8, stream);
        } else {
            dst.fc_bf16 = up.load_bf16(sf, "fc.weight", dst.arena, stream);
        }

        dst.hidden_norm = up.load_bf16(sf, "hidden_norm.weight", dst.arena, stream);
        dst.final_norm = up.load_bf16(sf, "norm.weight", dst.arena, stream);
        dst.pred_codebook = up.load_bf16(sf, "candidate_selector.predecessor_codebook", dst.arena, stream);
        dst.succ_codebook = up.load_bf16(sf, "candidate_selector.successor_codebook", dst.arena, stream);
        dst.selector_hidden_proj = up.load_bf16(sf, "candidate_selector.hidden_projection.weight", dst.arena, stream);

        for (int L = 0; L < DFlash2Config::NUM_LAYERS; ++L) {
            auto& lw = dst.layers[L];
            std::string p = "layers." + std::to_string(L) + ".";
            lw.input_layernorm = up.load_bf16(sf, p + "input_layernorm.weight", dst.arena, stream);
            lw.post_attention_layernorm = up.load_bf16(sf, p + "post_attention_layernorm.weight", dst.arena, stream);
            lw.q_norm = up.load_bf16(sf, p + "self_attn.q_norm.weight", dst.arena, stream);
            lw.k_norm = up.load_bf16(sf, p + "self_attn.k_norm.weight", dst.arena, stream);
            lw.attn_conv_base = up.load_bf16(sf, p + "attention_conv.base_kernel", dst.arena, stream);
            lw.attn_conv_proj = up.load_bf16(sf, p + "attention_conv.kernel_projection.weight", dst.arena, stream);
            lw.mlp_conv_base = up.load_bf16(sf, p + "mlp_conv.base_kernel", dst.arena, stream);
            lw.mlp_conv_proj = up.load_bf16(sf, p + "mlp_conv.kernel_projection.weight", dst.arena, stream);

            if (quantize_to_fp8) {
                load_fused_bf16_as_fp8(
                    fp8_backend, up, sf,
                    {p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"},
                    DFlash2Config::QKV_DIM, DFlash2Config::HIDDEN_SIZE,
                    dst.arena, lw.qkv_fp8, stream);
                load_fused_bf16_as_fp8(
                    fp8_backend, up, sf,
                    {p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"},
                    DFlash2Config::KV_DIM * 2, DFlash2Config::HIDDEN_SIZE,
                    dst.arena, lw.kv_only_fp8, stream);
                load_fused_bf16_as_fp8(
                    fp8_backend, up, sf, {p + "self_attn.o_proj.weight"},
                    DFlash2Config::HIDDEN_SIZE, DFlash2Config::Q_DIM,
                    dst.arena, lw.o_fp8, stream);
                load_fused_bf16_as_fp8(
                    fp8_backend, up, sf,
                    {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"},
                    DFlash2Config::INTERMEDIATE_SIZE * 2, DFlash2Config::HIDDEN_SIZE,
                    dst.arena, lw.gate_up_fp8, stream);
                load_fused_bf16_as_fp8(
                    fp8_backend, up, sf, {p + "mlp.down_proj.weight"},
                    DFlash2Config::HIDDEN_SIZE, DFlash2Config::INTERMEDIATE_SIZE,
                    dst.arena, lw.down_fp8, stream);
            } else {
                lw.qkv_bf16 = up.load_fused_bf16(
                    sf, {p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"},
                    dst.arena, stream);
                lw.kv_only_bf16 = lw.qkv_bf16 + (size_t)DFlash2Config::Q_DIM * DFlash2Config::HIDDEN_SIZE;
                lw.o_bf16 = up.load_bf16(sf, p + "self_attn.o_proj.weight", dst.arena, stream);
                lw.gate_up_bf16 = up.load_fused_bf16(
                    sf, {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"},
                    dst.arena, stream);
                lw.down_bf16 = up.load_bf16(sf, p + "mlp.down_proj.weight", dst.arena, stream);
            }
        }
    } catch (...) {
        up.destroy();
        throw;
    }

    up.destroy();
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "[loader] DFlash2 draft loaded (%s) in %.2f s (%.2f GB VRAM)\n",
            quantize_to_fp8 ? "FP8" : "BF16", elapsed, dst.arena.used() / 1e9);
}

} // namespace amperegen
