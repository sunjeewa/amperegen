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
#include <thread>
#include <unistd.h>
#include <vector>

namespace amperegen {

namespace qwen_fp8 {

void launch_pack_fp8_mma(
    const uint8_t* d_w_row,
    uint4* d_w_packed,
    int N, int K,
    cudaStream_t stream);

void launch_quantize_bf16_to_fp8_mma(
    const __nv_bfloat16* d_w_bf16,
    uint4* d_w_packed,
    float* d_scales_raw,
    float* d_scales_fused,
    int N, int K,
    cudaStream_t stream);

void launch_prepare_fp8_scales(
    const __nv_bfloat16* d_scales_bf16,
    float* d_scales_raw,
    float* d_scales_fused,
    int num_scales,
    cudaStream_t stream);

void launch_fp8_linear(
    cublasHandle_t cublas,
    const __nv_bfloat16* X,
    const QuantLinearWeight& W,
    __nv_bfloat16* Y,
    float* d_splitk_scratch,
    __nv_bfloat16* d_bf16_scratch,
    int M,
    cudaStream_t stream);

} // namespace qwen_fp8

namespace {

class QwenFp8Backend final : public WeightBackend {
public:
    const char* id() const override { return "qwen_fp8"; }

    const char* description() const override {
        return "Official Qwen3.8-27B FP8 (block-128x128 E4M3 W8A16)";
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
        if (qm != "fp8") {
            reason = "quant_method is '" + qm + "' (expected 'fp8')";
            return false;
        }
        std::string fmt = qc.contains("fmt") ? qc.at("fmt").as_string_or("") : "";
        if (!fmt.empty() && fmt != "e4m3") {
            reason = "fp8 fmt is '" + fmt + "' (expected 'e4m3')";
            return false;
        }
        if (!qc.contains("weight_block_size") || !qc.at("weight_block_size").is_array() ||
            qc.at("weight_block_size").as_array().size() != 2 ||
            qc.at("weight_block_size").as_array()[0].as_int(-1) != 128 ||
            qc.at("weight_block_size").as_array()[1].as_int(-1) != 128) {
            reason = "weight_block_size is not [128, 128]";
            return false;
        }

        std::string l0_path = model_dir + "/layers-0.safetensors";
        std::string out_path = model_dir + "/outside.safetensors";
        if (access(l0_path.c_str(), R_OK) != 0) {
            reason = "missing layers-0.safetensors";
            return false;
        }
        if (access(out_path.c_str(), R_OK) != 0) {
            reason = "missing outside.safetensors";
            return false;
        }

        try {
            SafetensorsFile sf;
            sf.open(l0_path);
            const std::string w_key = "model.language_model.layers.0.mlp.gate_proj.weight";
            const std::string s_key = "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv";
            if (!sf.has(w_key) || !sf.has(s_key)) {
                reason = "layers-0.safetensors missing gate_proj FP8 weight/weight_scale_inv tensors";
                return false;
            }
            const auto& wi = sf.info(w_key);
            const auto& si = sf.info(s_key);
            if (wi.dtype != DType::F8_E4M3) {
                reason = "gate_proj.weight dtype is not F8_E4M3";
                return false;
            }
            if (si.dtype != DType::BF16) {
                reason = "gate_proj.weight_scale_inv dtype is not BF16";
                return false;
            }
            if (wi.shape.size() != 2 ||
                wi.shape[0] != Qwen35Config::INTERMEDIATE_SIZE ||
                wi.shape[1] != Qwen35Config::HIDDEN_SIZE) {
                reason = "gate_proj.weight shape mismatch";
                return false;
            }
            if (si.shape.size() != 2 ||
                si.shape[0] != Qwen35Config::INTERMEDIATE_SIZE / 128 ||
                si.shape[1] != Qwen35Config::HIDDEN_SIZE / 128) {
                reason = "gate_proj.weight_scale_inv shape mismatch";
                return false;
            }
        } catch (const std::exception& e) {
            reason = std::string("failed to inspect layers-0.safetensors: ") + e.what();
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
            sf.open(model_dir + "/layers-" + std::to_string(layer_idx) + ".safetensors");
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

        // ~24.4 GB layers + 5.09 GB embed+lm_head + optional 1.28 GB FP8 lm_head
        size_t arena_bytes = build_fp8_lm_head
            ? (31500ULL * 1024 * 1024)
            : (30200ULL * 1024 * 1024);
        dst.arena.init(arena_bytes);

        StagingUploader up;
        up.init(200ULL * 1024 * 1024); // fits gate_up = 178 MB

        std::vector<std::thread> prefetchers;
        auto prefetch_file = [](std::string path) {
            SafetensorsFile sf;
            try {
                sf.open(path);
                const auto& ts = sf.tensors();
                volatile uint64_t sink = 0;
                for (const auto& [k, ti] : ts) {
                    if (k.find(".visual.") != std::string::npos) continue;
                    const uint8_t* p = sf.data_ptr(k);
                    size_t nb = ti.nbytes();
                    for (size_t o = 0; o < nb; o += 4096) sink += p[o];
                }
                (void)sink;
            } catch (...) {}
        };

        int next_prefetch = 0;
        auto kick_prefetch = [&](int up_to) {
            while (next_prefetch <= up_to && next_prefetch < Qwen35Config::NUM_LAYERS) {
                prefetchers.emplace_back(
                    prefetch_file, model_dir + "/layers-" + std::to_string(next_prefetch) + ".safetensors");
                ++next_prefetch;
            }
            while (prefetchers.size() > 6) {
                prefetchers.front().join();
                prefetchers.erase(prefetchers.begin());
            }
        };

        fprintf(stderr, "[weights:%s] Loading Qwen3.8-27B FP8 (64 layers + outside.safetensors)...\n", id());
        try {
            kick_prefetch(5);
            for (int L = 0; L < Qwen35Config::NUM_LAYERS; ++L) {
                kick_prefetch(L + 4);
                SafetensorsFile sf;
                sf.open(model_dir + "/layers-" + std::to_string(L) + ".safetensors");
                load_layer_from_file(sf, L, up, dst, stream);
                if ((L + 1) % 16 == 0) {
                    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    fprintf(stderr, "[weights:%s]   loaded %d/64 layers (%.1f s, %.2f GB used)\n",
                            id(), L + 1, elapsed, dst.arena.used() / 1e9);
                }
            }
            for (auto& th : prefetchers) th.join();
            prefetchers.clear();

            {
                SafetensorsFile sf;
                sf.open(model_dir + "/outside.safetensors");
                dst.embed_tokens = up.load_bf16(sf, "model.language_model.embed_tokens.weight", dst.arena, stream);
                dst.final_norm = up.load_bf16(sf, "model.language_model.norm.weight", dst.arena, stream);
                dst.lm_head = up.load_bf16(sf, "lm_head.weight", dst.arena, stream);
                if (build_fp8_lm_head) {
                    maybe_build_draft_fp8_lm_head(dst, stream);
                }
            }
        } catch (...) {
            for (auto& th : prefetchers) {
                if (th.joinable()) th.join();
            }
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
        qwen_fp8::launch_fp8_linear(cublas, X, W, Y, d_splitk_scratch, d_bf16_scratch, M, stream);
    }

    bool quantize_from_bf16_device(
        const __nv_bfloat16* d_w_bf16,
        int N, int K,
        GpuArena& arena,
        QuantLinearWeight& out,
        cudaStream_t stream
    ) const override {
        if ((N % 128) != 0 || (K % 128) != 0) return false;
        size_t w_bytes = (size_t)N * K;
        size_t num_scales = (size_t)(N / 128) * (K / 128);
        uint4* w_packed = arena.alloc_typed<uint4>(w_bytes / 16);
        float* scales_raw = arena.alloc_typed<float>(num_scales);
        float* scales_fused = arena.alloc_typed<float>(num_scales);

        qwen_fp8::launch_quantize_bf16_to_fp8_mma(
            d_w_bf16, w_packed, scales_raw, scales_fused, N, K, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        out.backend = this;
        out.w_packed = w_packed;
        out.scales = scales_fused;
        out.aux_scales = scales_raw;
        out.N = N;
        out.K = K;
        return true;
    }

    bool self_test(cublasHandle_t cublas, cudaStream_t stream) const override {
        constexpr int N = 512;
        constexpr int K = 512;
        constexpr int M_MAX = 128;
        const int test_ms[] = {1, 8, 16, 24, 64, 128};

        std::vector<__nv_bfloat16> h_W((size_t)N * K);
        std::vector<__nv_bfloat16> h_X((size_t)M_MAX * K);
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                float v = 0.7f * std::sin(0.013f * (float)(n * 31 + k)) +
                          0.3f * std::cos(0.029f * (float)(n * 7 - k * 3));
                h_W[(size_t)n * K + k] = __float2bfloat16(v);
            }
        }
        for (int m = 0; m < M_MAX; ++m) {
            for (int k = 0; k < K; ++k) {
                float v = 0.5f * std::cos(0.017f * (float)(m * 19 + k)) -
                          0.25f * std::sin(0.011f * (float)(m * 5 + k * 2));
                h_X[(size_t)m * K + k] = __float2bfloat16(v);
            }
        }

        GpuArena arena;
        arena.init(16ULL * 1024 * 1024);
        auto* d_W_bf16 = arena.alloc_typed<__nv_bfloat16>((size_t)N * K);
        auto* d_X = arena.alloc_typed<__nv_bfloat16>((size_t)M_MAX * K);
        auto* d_Y = arena.alloc_typed<__nv_bfloat16>((size_t)M_MAX * N);
        auto* d_splitk = arena.alloc_typed<float>((size_t)8 * 32 * N);
        auto* d_bf16_scr = arena.alloc_typed<__nv_bfloat16>((size_t)N * K);

        CUDA_CHECK(cudaMemcpyAsync(d_W_bf16, h_W.data(), h_W.size() * 2, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_X, h_X.data(), h_X.size() * 2, cudaMemcpyHostToDevice, stream));

        QuantLinearWeight W_q;
        if (!quantize_from_bf16_device(d_W_bf16, N, K, arena, W_q, stream)) {
            fprintf(stderr, "[self_test:%s] FAIL: quantize_from_bf16_device returned false\n", id());
            arena.release();
            return false;
        }

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
                               (double)__bfloat162float(h_W[(size_t)n * K + k]);
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
            bool ok = (bad == 0 && cos_sim > 0.999 && rel_l2 < 0.04);
            printf("[self_test:%s] M=%3d N=%d K=%d | cos=%.6f rel_L2=%.5f nan=%d -> %s\n",
                   id(), M, N, K, cos_sim, rel_l2, bad, ok ? "PASS" : "FAIL");
            if (!ok) all_ok = false;
        }

        arena.release();
        return all_ok;
    }

private:
    void load_fused_fp8(
        const SafetensorsFile& sf,
        const std::vector<std::string>& w_names,
        const std::vector<std::string>& s_names,
        int total_N, int K,
        StagingUploader& up,
        GpuArena& arena,
        QuantLinearWeight& out,
        cudaStream_t stream
    ) const {
        size_t w_bytes = (size_t)total_N * K;
        size_t num_scales = (size_t)(total_N / 128) * (K / 128);

        uint4* w_packed = arena.alloc_typed<uint4>(w_bytes / 16);
        float* scales_raw = arena.alloc_typed<float>(num_scales);
        float* scales_fused = arena.alloc_typed<float>(num_scales);

        up.stage_concat_to_device(sf, w_names, w_bytes, "qwen_fp8 weights", stream);
        qwen_fp8::launch_pack_fp8_mma(up.d_stage, w_packed, total_N, K, stream);
        // Wait for H2D DMA + pack to finish BEFORE reusing h_stage / d_stage for scales!
        CUDA_CHECK(cudaStreamSynchronize(stream));

        up.stage_concat_to_device(sf, s_names, num_scales * 2, "qwen_fp8 scales", stream);
        qwen_fp8::launch_prepare_fp8_scales(
            reinterpret_cast<const __nv_bfloat16*>(up.d_stage),
            scales_raw, scales_fused, (int)num_scales, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        out.backend = this;
        out.w_packed = w_packed;
        out.scales = scales_fused;
        out.aux_scales = scales_raw;
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

            load_fused_fp8(
                sf,
                {p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"},
                {p + "self_attn.q_proj.weight_scale_inv", p + "self_attn.k_proj.weight_scale_inv", p + "self_attn.v_proj.weight_scale_inv"},
                Qwen35Config::ATTN_QKV_DIM, Qwen35Config::HIDDEN_SIZE,
                up, dst.arena, lw.qkv_proj, stream);
            load_fused_fp8(
                sf, {p + "self_attn.o_proj.weight"}, {p + "self_attn.o_proj.weight_scale_inv"},
                Qwen35Config::HIDDEN_SIZE, Qwen35Config::ATTN_Q_DIM,
                up, dst.arena, lw.o_proj, stream);
            load_fused_fp8(
                sf, {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"},
                {p + "mlp.gate_proj.weight_scale_inv", p + "mlp.up_proj.weight_scale_inv"},
                Qwen35Config::INTERMEDIATE_SIZE * 2, Qwen35Config::HIDDEN_SIZE,
                up, dst.arena, lw.gate_up_proj, stream);
            load_fused_fp8(
                sf, {p + "mlp.down_proj.weight"}, {p + "mlp.down_proj.weight_scale_inv"},
                Qwen35Config::HIDDEN_SIZE, Qwen35Config::INTERMEDIATE_SIZE,
                up, dst.arena, lw.down_proj, stream);
        } else {
            auto& lw = dst.gdn[Qwen35Config::gdn_idx(layer_idx)];
            lw.input_layernorm = up.load_bf16(sf, p + "input_layernorm.weight", dst.arena, stream);
            lw.post_attention_layernorm = up.load_bf16(sf, p + "post_attention_layernorm.weight", dst.arena, stream);

            load_fused_fp8(
                sf,
                {p + "linear_attn.in_proj_qkv.weight", p + "linear_attn.in_proj_z.weight"},
                {p + "linear_attn.in_proj_qkv.weight_scale_inv", p + "linear_attn.in_proj_z.weight_scale_inv"},
                Qwen35Config::GDN_QKVZ_DIM, Qwen35Config::HIDDEN_SIZE,
                up, dst.arena, lw.in_proj_qkvz, stream);

            lw.in_proj_ba = up.load_fused_bf16(
                sf, {p + "linear_attn.in_proj_b.weight", p + "linear_attn.in_proj_a.weight"},
                dst.arena, stream);

            lw.conv1d_w = up.load_bf16(sf, p + "linear_attn.conv1d.weight", dst.arena, stream);
            lw.A_log = up.load_as_fp32(sf, p + "linear_attn.A_log", dst.arena, stream);
            lw.dt_bias = up.load_as_fp32(sf, p + "linear_attn.dt_bias", dst.arena, stream);
            lw.gdn_norm = up.load_bf16(sf, p + "linear_attn.norm.weight", dst.arena, stream);

            load_fused_fp8(
                sf, {p + "linear_attn.out_proj.weight"}, {p + "linear_attn.out_proj.weight_scale_inv"},
                Qwen35Config::HIDDEN_SIZE, Qwen35Config::GDN_V_DIM,
                up, dst.arena, lw.out_proj, stream);
            load_fused_fp8(
                sf, {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"},
                {p + "mlp.gate_proj.weight_scale_inv", p + "mlp.up_proj.weight_scale_inv"},
                Qwen35Config::INTERMEDIATE_SIZE * 2, Qwen35Config::HIDDEN_SIZE,
                up, dst.arena, lw.gate_up_proj, stream);
            load_fused_fp8(
                sf, {p + "mlp.down_proj.weight"}, {p + "mlp.down_proj.weight_scale_inv"},
                Qwen35Config::HIDDEN_SIZE, Qwen35Config::INTERMEDIATE_SIZE,
                up, dst.arena, lw.down_proj, stream);
        }
    }
};

} // namespace

const WeightBackend* get_qwen_fp8_backend() {
    static const QwenFp8Backend instance;
    return &instance;
}

} // namespace amperegen
