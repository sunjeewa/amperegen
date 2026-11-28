#include "weights/weight_backend.hpp"
#include "weights/checkpoint_probe.hpp"
#include "weights/staging_uploader.hpp"
#include "loader/model_loader.hpp"
#include "loader/safetensors.hpp"
#include "model/model_config.hpp"
#include "runtime/cuda_utils.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace amperegen {

namespace redhat_int4 {

void launch_pack_int4_mma(
    const uint32_t* d_w_i32,
    uint4* d_w_packed,
    int N, int K,
    cudaStream_t stream);

void launch_pack_int4_scales(
    const __nv_bfloat16* d_scales_bf16,
    uint32_t* d_scales_packed,
    int N, int K,
    cudaStream_t stream);

void launch_int4_linear(
    cublasHandle_t cublas,
    const __nv_bfloat16* X,
    const QuantLinearWeight& W,
    __nv_bfloat16* Y,
    float* d_splitk_scratch,
    __nv_bfloat16* d_bf16_scratch,
    int M,
    cudaStream_t stream);

} // namespace redhat_int4

namespace {

class RedHatInt4Backend final : public WeightBackend {
public:
    const char* id() const override { return "redhat_int4"; }

    const char* description() const override {
        return "RedHatAI Qwen3.8-27B INT4 (compressed-tensors sym g128 W4A16)";
    }

    bool probe(const std::string& model_dir, std::string& reason) const override {
        Json cfg;
        if (!read_checkpoint_config_json(model_dir, cfg, reason)) return false;
        if (!verify_qwen35_27b_architecture(cfg, reason)) return false;

        if (!cfg.contains("quantization_config") || !cfg.at("quantization_config").is_object()) {
            reason = "config.json missing quantization_config";
            return false;
        }
        const Json& qc = cfg.at("quantization_config");
        std::string qm = qc.contains("quant_method") ? qc.at("quant_method").as_string_or("") : "";
        if (qm != "compressed-tensors") {
            reason = "quant_method is '" + qm + "' (expected 'compressed-tensors')";
            return false;
        }
        if (!qc.contains("config_groups") || !qc.at("config_groups").is_object() ||
            !qc.at("config_groups").contains("group_0") || !qc.at("config_groups").at("group_0").is_object()) {
            reason = "quantization_config missing config_groups.group_0";
            return false;
        }
        const Json& g0 = qc.at("config_groups").at("group_0");
        std::string fmt = g0.contains("format") ? g0.at("format").as_string_or("") : "";
        if (fmt != "pack-quantized") {
            reason = "group_0.format is '" + fmt + "' (expected 'pack-quantized')";
            return false;
        }
        if (g0.contains("input_activations") && !g0.at("input_activations").is_null()) {
            reason = "group_0.input_activations is not null (only W4A16 weight-only INT4 is qualified)";
            return false;
        }
        if (!g0.contains("weights") || !g0.at("weights").is_object()) {
            reason = "group_0 missing weights descriptor";
            return false;
        }
        const Json& wq = g0.at("weights");
        if (wq.contains("type") ? wq.at("type").as_string_or("") != "int" : true) {
            reason = "group_0.weights.type is not 'int'";
            return false;
        }
        if (!wq.contains("num_bits") || wq.at("num_bits").as_int(-1) != 4) {
            reason = "group_0.weights.num_bits is not 4";
            return false;
        }
        if (!wq.contains("symmetric") || !wq.at("symmetric").is_bool() || !wq.at("symmetric").as_bool(false)) {
            reason = "group_0.weights.symmetric is not true";
            return false;
        }
        if (!wq.contains("group_size") || wq.at("group_size").as_int(-1) != 128) {
            reason = "group_0.weights.group_size is not 128";
            return false;
        }
        if (wq.contains("strategy") ? wq.at("strategy").as_string_or("") != "group" : true) {
            reason = "group_0.weights.strategy is not 'group'";
            return false;
        }

        std::string sf_path = model_dir + "/model.safetensors";
        if (access(sf_path.c_str(), R_OK) != 0) {
            reason = "missing model.safetensors";
            return false;
        }

        try {
            SafetensorsFile sf;
            sf.open(sf_path);
            const std::string base = "model.language_model.layers.0.mlp.gate_proj";
            const std::string wp_key = base + ".weight_packed";
            const std::string ws_key = base + ".weight_scale";
            if (!sf.has(wp_key) || !sf.has(ws_key)) {
                reason = "model.safetensors missing gate_proj weight_packed/weight_scale tensors";
                return false;
            }
            if (sf.has(base + ".weight_zero_point")) {
                reason = "asymmetric zero_point tensors present (only symmetric INT4 is qualified)";
                return false;
            }
            if (sf.has(base + ".weight_g_idx")) {
                reason = "weight_g_idx permutation present (not qualified for redhat_int4)";
                return false;
            }
            const auto& wi = sf.info(wp_key);
            const auto& si = sf.info(ws_key);
            if (wi.dtype != DType::INT32) {
                reason = "gate_proj.weight_packed dtype is not INT32";
                return false;
            }
            if (si.dtype != DType::BF16) {
                reason = "gate_proj.weight_scale dtype is not BF16";
                return false;
            }
            if (wi.shape.size() != 2 ||
                wi.shape[0] != Qwen35Config::INTERMEDIATE_SIZE ||
                wi.shape[1] != Qwen35Config::HIDDEN_SIZE / 8) {
                reason = "gate_proj.weight_packed shape mismatch";
                return false;
            }
            if (si.shape.size() != 2 ||
                si.shape[0] != Qwen35Config::INTERMEDIATE_SIZE ||
                si.shape[1] != Qwen35Config::HIDDEN_SIZE / 128) {
                reason = "gate_proj.weight_scale shape mismatch";
                return false;
            }
        } catch (const std::exception& e) {
            reason = std::string("failed to inspect model.safetensors: ") + e.what();
            return false;
        }
        return true;
    }

    void load_single_layer(
        const std::string& model_dir,
        int layer_idx,
        TargetWeights& dst,
        cudaStream_t stream
    ) const override {
        dst.backend = this;
        if (dst.arena.capacity() == 0) {
            dst.arena.init(1024ULL * 1024 * 1024);
        }
        StagingUploader up;
        up.init(200ULL * 1024 * 1024);
        try {
            SafetensorsFile sf;
            sf.open(model_dir + "/model.safetensors");
            load_layer_from_file(sf, layer_idx, up, dst, stream);
        } catch (...) {
            up.destroy();
            throw;
        }
        up.destroy();
    }

    void load_target(
        const std::string& model_dir,
        TargetWeights& dst,
        bool build_fp8_lm_head,
        cudaStream_t stream
    ) const override {
        auto t0 = std::chrono::steady_clock::now();
        dst.backend = this;

        // ~12.2 GB packed INT4 + 0.38 GB scales + 0.08 GB misc + 5.09 GB embed+lm_head + optional 1.28 GB FP8 lm_head
        size_t arena_bytes = build_fp8_lm_head
            ? (19800ULL * 1024 * 1024)
            : (18500ULL * 1024 * 1024);
        dst.arena.init(arena_bytes);

        StagingUploader up;
        up.init(200ULL * 1024 * 1024);

        fprintf(stderr, "[weights:%s] Loading RedHatAI Qwen3.8-27B INT4 (W4A16 g128) from model.safetensors...\n", id());
        try {
            SafetensorsFile sf;
            sf.open(model_dir + "/model.safetensors");

            for (int L = 0; L < Qwen35Config::NUM_LAYERS; ++L) {
                load_layer_from_file(sf, L, up, dst, stream);
                if ((L + 1) % 16 == 0) {
                    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    fprintf(stderr, "[weights:%s]   loaded %d/64 layers (%.1f s, %.2f GB used)\n",
                            id(), L + 1, elapsed, dst.arena.used() / 1e9);
                }
            }

            dst.embed_tokens = up.load_bf16(sf, "model.language_model.embed_tokens.weight", dst.arena, stream);
            dst.final_norm = up.load_bf16(sf, "model.language_model.norm.weight", dst.arena, stream);
            dst.lm_head = up.load_bf16(sf, "lm_head.weight", dst.arena, stream);
            if (build_fp8_lm_head) {
                maybe_build_draft_fp8_lm_head(dst, stream);
            }
        } catch (...) {
            up.destroy();
            throw;
        }

        up.destroy();
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[weights:%s] Target loaded in %.2f s (%.2f GB VRAM)\n",
                id(), elapsed, dst.arena.used() / 1e9);
    }

    void launch_linear(
        cublasHandle_t cublas,
        const __nv_bfloat16* X,
        const QuantLinearWeight& W,
        __nv_bfloat16* Y,
        float* d_splitk_scratch,
        __nv_bfloat16* d_bf16_scratch,
        int M,
        cudaStream_t stream
    ) const override {
        redhat_int4::launch_int4_linear(cublas, X, W, Y, d_splitk_scratch, d_bf16_scratch, M, stream);
    }

    bool self_test(cublasHandle_t cublas, cudaStream_t stream) const override {
        constexpr int N = 512;
        constexpr int K = 512;
        constexpr int M_MAX = 128;
        const int test_ms[] = {1, 8, 16, 24, 64, 128};

        // Build synthetic symmetric INT4 packed I32 [N, K/8] + BF16 group-128 scales [N, K/128]
        std::vector<uint32_t> h_w_i32((size_t)N * (K / 8), 0);
        std::vector<__nv_bfloat16> h_scales((size_t)N * (K / 128));
        std::vector<float> h_w_ref((size_t)N * K);

        uint32_t rng = 0x13579BDFu;
        auto next_u32 = [&]() -> uint32_t {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            return rng;
        };

        for (int n = 0; n < N; ++n) {
            for (int g = 0; g < K / 128; ++g) {
                float sc = 0.02f + 0.18f * (float)(next_u32() & 0xFFFFu) / 65535.0f;
                __nv_bfloat16 sc_bf16 = __float2bfloat16(sc);
                h_scales[(size_t)n * (K / 128) + g] = sc_bf16;
                float sc_exact = __bfloat162float(sc_bf16);
                for (int j = 0; j < 128; j += 8) {
                    uint32_t packed = 0;
                    for (int b = 0; b < 8; ++b) {
                        uint32_t nib = next_u32() & 0xFu; // 0..15 -> signed -8..+7
                        packed |= (nib << (b * 4));
                        int k = g * 128 + j + b;
                        h_w_ref[(size_t)n * K + k] = (float)((int)nib - 8) * sc_exact;
                    }
                    h_w_i32[(size_t)n * (K / 8) + (g * 16 + j / 8)] = packed;
                }
            }
        }

        std::vector<__nv_bfloat16> h_X((size_t)M_MAX * K);
        for (int m = 0; m < M_MAX; ++m) {
            for (int k = 0; k < K; ++k) {
                float v = 0.6f * std::sin(0.019f * (float)(m * 23 + k)) -
                          0.3f * std::cos(0.013f * (float)(m * 7 - k * 5));
                h_X[(size_t)m * K + k] = __float2bfloat16(v);
            }
        }

        GpuArena arena;
        arena.init(16ULL * 1024 * 1024);
        auto* d_w_i32 = arena.alloc_typed<uint32_t>(h_w_i32.size());
        auto* d_scales_raw = arena.alloc_typed<__nv_bfloat16>(h_scales.size());
        auto* d_w_packed = arena.alloc_typed<uint4>(((size_t)N * K / 2) / 16);
        auto* d_scales_packed = arena.alloc_typed<uint32_t>((size_t)(N / 16) * (K / 128) * 8);
        auto* d_X = arena.alloc_typed<__nv_bfloat16>((size_t)M_MAX * K);
        auto* d_Y = arena.alloc_typed<__nv_bfloat16>((size_t)M_MAX * N);
        auto* d_splitk = arena.alloc_typed<float>((size_t)8 * 32 * N);
        auto* d_bf16_scr = arena.alloc_typed<__nv_bfloat16>((size_t)N * K);

        CUDA_CHECK(cudaMemcpyAsync(d_w_i32, h_w_i32.data(), h_w_i32.size() * 4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_scales_raw, h_scales.data(), h_scales.size() * 2, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_X, h_X.data(), h_X.size() * 2, cudaMemcpyHostToDevice, stream));

        redhat_int4::launch_pack_int4_mma(d_w_i32, d_w_packed, N, K, stream);
        redhat_int4::launch_pack_int4_scales(d_scales_raw, d_scales_packed, N, K, stream);

        QuantLinearWeight W_q;
        W_q.backend = this;
        W_q.w_packed = d_w_packed;
        W_q.scales = d_scales_packed;
        W_q.aux_scales = nullptr;
        W_q.N = N;
        W_q.K = K;

        bool all_ok = true;
        std::vector<__nv_bfloat16> h_Y((size_t)M_MAX * N);
        for (int M : test_ms) {
            CUDA_CHECK(cudaMemsetAsync(d_Y, 0, (size_t)M * N * 2, stream));
            launch_linear(cublas, d_X, W_q, d_Y, d_splitk, d_bf16_scr, M, stream);
            CUDA_CHECK(cudaMemcpyAsync(h_Y.data(), d_Y, (size_t)M * N * 2, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            double dot = 0.0, na = 0.0, nb = 0.0, diff_sq = 0.0;
            int bad = 0;
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    double ref = 0.0;
                    for (int k = 0; k < K; ++k) {
                        ref += (double)__bfloat162float(h_X[(size_t)m * K + k]) *
                               (double)h_w_ref[(size_t)n * K + k];
                    }
                    float got = __bfloat162float(h_Y[(size_t)m * N + n]);
                    if (std::isnan(got) || std::isinf(got)) { ++bad; continue; }
                    double d = (double)got - ref;
                    dot += (double)got * ref;
                    na += (double)got * got;
                    nb += ref * ref;
                    diff_sq += d * d;
                }
            }
            double cos_sim = dot / (std::sqrt(na * nb) + 1e-12);
            double rel_l2 = std::sqrt(diff_sq / (nb + 1e-12));
            bool ok = (bad == 0 && cos_sim > 0.9999 && rel_l2 < 0.005);
            printf("[self_test:%s] M=%3d N=%d K=%d | cos=%.6f rel_L2=%.5f nan=%d -> %s\n",
                   id(), M, N, K, cos_sim, rel_l2, bad, ok ? "PASS" : "FAIL");
            if (!ok) all_ok = false;
        }

        arena.release();
        return all_ok;
    }

private:
    void load_fused_int4(
        const SafetensorsFile& sf,
        const std::vector<std::string>& w_names,
        const std::vector<std::string>& s_names,
        int total_N, int K,
        StagingUploader& up,
        GpuArena& arena,
        QuantLinearWeight& out,
        cudaStream_t stream
    ) const {
        size_t w_bytes = ((size_t)total_N * K) / 2;
        size_t scale_elems = (size_t)total_N * (K / 128);
        size_t packed_scale_u32s = (size_t)(total_N / 16) * (K / 128) * 8;

        uint4* w_packed = arena.alloc_typed<uint4>(w_bytes / 16);
        uint32_t* scales_packed = arena.alloc_typed<uint32_t>(packed_scale_u32s);

        up.stage_concat_to_device(sf, w_names, w_bytes, "redhat_int4 weights", stream);
        redhat_int4::launch_pack_int4_mma(
            reinterpret_cast<const uint32_t*>(up.d_stage), w_packed, total_N, K, stream);
        // Wait for H2D DMA + pack to finish BEFORE reusing h_stage / d_stage for scales!
        CUDA_CHECK(cudaStreamSynchronize(stream));

        up.stage_concat_to_device(sf, s_names, scale_elems * 2, "redhat_int4 scales", stream);
        redhat_int4::launch_pack_int4_scales(
            reinterpret_cast<const __nv_bfloat16*>(up.d_stage), scales_packed, total_N, K, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        out.backend = this;
        out.w_packed = w_packed;
        out.scales = scales_packed;
        out.aux_scales = nullptr;
        out.N = total_N;
        out.K = K;
    }

    void load_layer_from_file(
        const SafetensorsFile& sf,
        int layer_idx,
        StagingUploader& up,
        TargetWeights& dst,
        cudaStream_t stream
    ) const {
        std::string p = "model.language_model.layers." + std::to_string(layer_idx) + ".";
        if (Qwen35Config::is_full_attn(layer_idx)) {
            auto& lw = dst.full_attn[Qwen35Config::full_attn_idx(layer_idx)];
            lw.input_layernorm = up.load_bf16(sf, p + "input_layernorm.weight", dst.arena, stream);
            lw.post_attention_layernorm = up.load_bf16(sf, p + "post_attention_layernorm.weight", dst.arena, stream);
            lw.q_norm = up.load_bf16(sf, p + "self_attn.q_norm.weight", dst.arena, stream);
            lw.k_norm = up.load_bf16(sf, p + "self_attn.k_norm.weight", dst.arena, stream);

            load_fused_int4(
                sf,
                {p + "self_attn.q_proj.weight_packed", p + "self_attn.k_proj.weight_packed", p + "self_attn.v_proj.weight_packed"},
                {p + "self_attn.q_proj.weight_scale", p + "self_attn.k_proj.weight_scale", p + "self_attn.v_proj.weight_scale"},
                Qwen35Config::ATTN_QKV_DIM, Qwen35Config::HIDDEN_SIZE,
                up, dst.arena, lw.qkv_proj, stream);
            load_fused_int4(
                sf, {p + "self_attn.o_proj.weight_packed"}, {p + "self_attn.o_proj.weight_scale"},
                Qwen35Config::HIDDEN_SIZE, Qwen35Config::ATTN_Q_DIM,
                up, dst.arena, lw.o_proj, stream);
            load_fused_int4(
                sf, {p + "mlp.gate_proj.weight_packed", p + "mlp.up_proj.weight_packed"},
                {p + "mlp.gate_proj.weight_scale", p + "mlp.up_proj.weight_scale"},
                Qwen35Config::INTERMEDIATE_SIZE * 2, Qwen35Config::HIDDEN_SIZE,
                up, dst.arena, lw.gate_up_proj, stream);
            load_fused_int4(
                sf, {p + "mlp.down_proj.weight_packed"}, {p + "mlp.down_proj.weight_scale"},
                Qwen35Config::HIDDEN_SIZE, Qwen35Config::INTERMEDIATE_SIZE,
                up, dst.arena, lw.down_proj, stream);
        } else {
            auto& lw = dst.gdn[Qwen35Config::gdn_idx(layer_idx)];
            lw.input_layernorm = up.load_bf16(sf, p + "input_layernorm.weight", dst.arena, stream);
            lw.post_attention_layernorm = up.load_bf16(sf, p + "post_attention_layernorm.weight", dst.arena, stream);

            load_fused_int4(
                sf,
                {p + "linear_attn.in_proj_qkv.weight_packed", p + "linear_attn.in_proj_z.weight_packed"},
                {p + "linear_attn.in_proj_qkv.weight_scale", p + "linear_attn.in_proj_z.weight_scale"},
                Qwen35Config::GDN_QKVZ_DIM, Qwen35Config::HIDDEN_SIZE,
                up, dst.arena, lw.in_proj_qkvz, stream);

            lw.in_proj_ba = up.load_fused_bf16(
                sf, {p + "linear_attn.in_proj_b.weight", p + "linear_attn.in_proj_a.weight"},
                dst.arena, stream);

            lw.conv1d_w = up.load_bf16(sf, p + "linear_attn.conv1d.weight", dst.arena, stream);
            lw.A_log = up.load_as_fp32(sf, p + "linear_attn.A_log", dst.arena, stream);
            lw.dt_bias = up.load_as_fp32(sf, p + "linear_attn.dt_bias", dst.arena, stream);
            lw.gdn_norm = up.load_bf16(sf, p + "linear_attn.norm.weight", dst.arena, stream);

            load_fused_int4(
                sf, {p + "linear_attn.out_proj.weight_packed"}, {p + "linear_attn.out_proj.weight_scale"},
                Qwen35Config::HIDDEN_SIZE, Qwen35Config::GDN_V_DIM,
                up, dst.arena, lw.out_proj, stream);
            load_fused_int4(
                sf, {p + "mlp.gate_proj.weight_packed", p + "mlp.up_proj.weight_packed"},
                {p + "mlp.gate_proj.weight_scale", p + "mlp.up_proj.weight_scale"},
                Qwen35Config::INTERMEDIATE_SIZE * 2, Qwen35Config::HIDDEN_SIZE,
                up, dst.arena, lw.gate_up_proj, stream);
            load_fused_int4(
                sf, {p + "mlp.down_proj.weight_packed"}, {p + "mlp.down_proj.weight_scale"},
                Qwen35Config::HIDDEN_SIZE, Qwen35Config::INTERMEDIATE_SIZE,
                up, dst.arena, lw.down_proj, stream);
        }
    }
};

} // namespace

const WeightBackend* get_redhat_int4_backend() {
    static const RedHatInt4Backend instance;
    return &instance;
}

} // namespace amperegen
