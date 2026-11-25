#include "loader/model_loader.hpp"
#include "loader/safetensors.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace amperegen {

namespace {

// Helper to upload raw bytes synchronously/async via pinned host staging buffer
struct StagingUploader {
    uint8_t* h_stage = nullptr;
    uint8_t* d_stage = nullptr;
    size_t cap = 0;

    void init(size_t bytes) {
        cap = bytes;
        CUDA_CHECK(cudaMallocHost(&h_stage, cap));
        CUDA_CHECK(cudaMalloc(&d_stage, cap));
    }
    void destroy() {
        if (d_stage) { cudaFree(d_stage); d_stage = nullptr; }
        if (h_stage) { cudaFreeHost(h_stage); h_stage = nullptr; }
    }

    // Copy host data directly to device
    void copy_to_device(const void* src, void* dst_dev, size_t bytes, cudaStream_t stream) {
        size_t off = 0;
        while (off < bytes) {
            size_t chunk = std::min(cap, bytes - off);
            std::memcpy(h_stage, static_cast<const uint8_t*>(src) + off, chunk);
            CUDA_CHECK(cudaMemcpyAsync(static_cast<uint8_t*>(dst_dev) + off, h_stage, chunk, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            off += chunk;
        }
    }

    // Upload fused FP8 weights (vertical concat of N_i rows) + fused BF16 scales -> pack MMA layout on GPU!
    void load_fused_fp8(
        const SafetensorsFile& sf,
        const std::vector<std::string>& w_names,
        const std::vector<std::string>& s_names,
        int total_N, int K,
        GpuArena& arena,
        Fp8LinearWeight& out,
        cudaStream_t stream
    ) {
        out.N = total_N;
        out.K = K;
        size_t w_bytes = (size_t)total_N * K;
        int sn_total = total_N / 128;
        int sk = K / 128;
        size_t num_scales = (size_t)sn_total * sk;

        out.w_packed = arena.alloc_typed<uint4>(w_bytes / 16);
        out.scales_raw = arena.alloc_typed<float>(num_scales);
        out.scales_fused = arena.alloc_typed<float>(num_scales);

        // Stage raw row-major FP8 into d_stage (which must be >= max weight size, e.g. 200 MB!)
        size_t dst_off = 0;
        for (const auto& wn : w_names) {
            const auto& ti = sf.info(wn);
            const uint8_t* ptr = sf.data_ptr(wn);
            size_t nb = ti.nbytes();
            std::memcpy(h_stage + dst_off, ptr, nb);
            dst_off += nb;
        }
        if (dst_off != w_bytes) {
            throw std::runtime_error("FP8 fused weight byte size mismatch!");
        }
        CUDA_CHECK(cudaMemcpyAsync(d_stage, h_stage, w_bytes, cudaMemcpyHostToDevice, stream));
        launch_pack_fp8_mma(d_stage, out.w_packed, total_N, K, stream);

        // Stage BF16 scales [N/128, K/128]
        size_t s_off = 0;
        for (const auto& sn : s_names) {
            const auto& ti = sf.info(sn);
            const uint8_t* ptr = sf.data_ptr(sn);
            size_t nb = ti.nbytes();
            std::memcpy(h_stage + s_off, ptr, nb);
            s_off += nb;
        }
        if (s_off != num_scales * 2) {
            throw std::runtime_error("FP8 fused scale byte size mismatch!");
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpyAsync(d_stage, h_stage, s_off, cudaMemcpyHostToDevice, stream));
        launch_prepare_fp8_scales(
            reinterpret_cast<const __nv_bfloat16*>(d_stage),
            out.scales_raw, out.scales_fused, (int)num_scales, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    __nv_bfloat16* load_bf16(
        const SafetensorsFile& sf,
        const std::string& name,
        GpuArena& arena,
        cudaStream_t stream
    ) {
        const auto& ti = sf.info(name);
        size_t nb = ti.nbytes();
        auto* dst = arena.alloc_typed<__nv_bfloat16>(nb / 2);
        copy_to_device(sf.data_ptr(name), dst, nb, stream);
        return dst;
    }

    __nv_bfloat16* load_fused_bf16(
        const SafetensorsFile& sf,
        const std::vector<std::string>& names,
        GpuArena& arena,
        cudaStream_t stream
    ) {
        size_t total_bytes = 0;
        for (const auto& n : names) total_bytes += sf.info(n).nbytes();
        auto* dst = arena.alloc_typed<__nv_bfloat16>(total_bytes / 2);
        size_t off = 0;
        for (const auto& n : names) {
            size_t nb = sf.info(n).nbytes();
            copy_to_device(sf.data_ptr(n), reinterpret_cast<uint8_t*>(dst) + off, nb, stream);
            off += nb;
        }
        return dst;
    }

    float* load_bf16_as_fp32(
        const SafetensorsFile& sf,
        const std::string& name,
        GpuArena& arena,
        cudaStream_t stream
    ) {
        const auto& ti = sf.info(name);
        size_t count = ti.numel();
        const uint16_t* src = reinterpret_cast<const uint16_t*>(sf.data_ptr(name));
        std::vector<float> tmp(count);
        for (size_t i = 0; i < count; ++i) {
            uint32_t bits = (uint32_t)src[i] << 16;
            std::memcpy(&tmp[i], &bits, sizeof(float));
        }
        auto* dst = arena.alloc_typed<float>(count);
        copy_to_device(tmp.data(), dst, count * sizeof(float), stream);
        return dst;
    }

    // Load BF16 weight(s) and quantize to FP8 block-128x128 MMA layout on GPU!
    void load_fused_bf16_as_fp8(
        const SafetensorsFile& sf,
        const std::vector<std::string>& names,
        int total_N, int K,
        GpuArena& arena,
        Fp8LinearWeight& out,
        cudaStream_t stream
    ) {
        out.N = total_N;
        out.K = K;
        size_t w_bytes = (size_t)total_N * K;
        int num_scales = (total_N / 128) * (K / 128);
        out.w_packed = arena.alloc_typed<uint4>(w_bytes / 16);
        out.scales_raw = arena.alloc_typed<float>(num_scales);
        out.scales_fused = arena.alloc_typed<float>(num_scales);

        size_t bf16_bytes = w_bytes * 2;
        if (bf16_bytes > cap) {
            throw std::runtime_error("Staging buffer too small for BF16->FP8 quant!");
        }
        size_t off = 0;
        for (const auto& n : names) {
            size_t nb = sf.info(n).nbytes();
            std::memcpy(h_stage + off, sf.data_ptr(n), nb);
            off += nb;
        }
        CUDA_CHECK(cudaMemcpyAsync(d_stage, h_stage, bf16_bytes, cudaMemcpyHostToDevice, stream));
        launch_quantize_bf16_to_fp8_mma(
            reinterpret_cast<const __nv_bfloat16*>(d_stage),
            out.w_packed, out.scales_raw, out.scales_fused, total_N, K, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

void load_layer_from_file(
    const SafetensorsFile& sf,
    int layer_idx,
    StagingUploader& up,
    TargetWeights& dst,
    cudaStream_t stream
) {
    std::string p = "model.language_model.layers." + std::to_string(layer_idx) + ".";
    if (Qwen35Config::is_full_attn(layer_idx)) {
        auto& lw = dst.full_attn[Qwen35Config::full_attn_idx(layer_idx)];
        lw.input_layernorm = up.load_bf16(sf, p + "input_layernorm.weight", dst.arena, stream);
        lw.post_attention_layernorm = up.load_bf16(sf, p + "post_attention_layernorm.weight", dst.arena, stream);
        lw.q_norm = up.load_bf16(sf, p + "self_attn.q_norm.weight", dst.arena, stream);
        lw.k_norm = up.load_bf16(sf, p + "self_attn.k_norm.weight", dst.arena, stream);

        up.load_fused_fp8(
            sf,
            {p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"},
            {p + "self_attn.q_proj.weight_scale_inv", p + "self_attn.k_proj.weight_scale_inv", p + "self_attn.v_proj.weight_scale_inv"},
            Qwen35Config::ATTN_QKV_DIM, Qwen35Config::HIDDEN_SIZE,
            dst.arena, lw.qkv_proj, stream);
        up.load_fused_fp8(
            sf, {p + "self_attn.o_proj.weight"}, {p + "self_attn.o_proj.weight_scale_inv"},
            Qwen35Config::HIDDEN_SIZE, Qwen35Config::ATTN_Q_DIM,
            dst.arena, lw.o_proj, stream);
        up.load_fused_fp8(
            sf, {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"},
            {p + "mlp.gate_proj.weight_scale_inv", p + "mlp.up_proj.weight_scale_inv"},
            Qwen35Config::INTERMEDIATE_SIZE * 2, Qwen35Config::HIDDEN_SIZE,
            dst.arena, lw.gate_up_proj, stream);
        up.load_fused_fp8(
            sf, {p + "mlp.down_proj.weight"}, {p + "mlp.down_proj.weight_scale_inv"},
            Qwen35Config::HIDDEN_SIZE, Qwen35Config::INTERMEDIATE_SIZE,
            dst.arena, lw.down_proj, stream);
    } else {
        auto& lw = dst.gdn[Qwen35Config::gdn_idx(layer_idx)];
        lw.input_layernorm = up.load_bf16(sf, p + "input_layernorm.weight", dst.arena, stream);
        lw.post_attention_layernorm = up.load_bf16(sf, p + "post_attention_layernorm.weight", dst.arena, stream);

        up.load_fused_fp8(
            sf,
            {p + "linear_attn.in_proj_qkv.weight", p + "linear_attn.in_proj_z.weight"},
            {p + "linear_attn.in_proj_qkv.weight_scale_inv", p + "linear_attn.in_proj_z.weight_scale_inv"},
            Qwen35Config::GDN_QKVZ_DIM, Qwen35Config::HIDDEN_SIZE,
            dst.arena, lw.in_proj_qkvz, stream);

        // Fuse in_proj_b [48, 5120] + in_proj_a [48, 5120] -> [96, 5120] BF16!
        lw.in_proj_ba = up.load_fused_bf16(
            sf, {p + "linear_attn.in_proj_b.weight", p + "linear_attn.in_proj_a.weight"},
            dst.arena, stream);

        lw.conv1d_w = up.load_bf16(sf, p + "linear_attn.conv1d.weight", dst.arena, stream);
        lw.A_log = up.load_bf16_as_fp32(sf, p + "linear_attn.A_log", dst.arena, stream);
        lw.dt_bias = up.load_bf16_as_fp32(sf, p + "linear_attn.dt_bias", dst.arena, stream);
        lw.gdn_norm = up.load_bf16(sf, p + "linear_attn.norm.weight", dst.arena, stream);

        up.load_fused_fp8(
            sf, {p + "linear_attn.out_proj.weight"}, {p + "linear_attn.out_proj.weight_scale_inv"},
            Qwen35Config::HIDDEN_SIZE, Qwen35Config::GDN_V_DIM,
            dst.arena, lw.out_proj, stream);
        up.load_fused_fp8(
            sf, {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"},
            {p + "mlp.gate_proj.weight_scale_inv", p + "mlp.up_proj.weight_scale_inv"},
            Qwen35Config::INTERMEDIATE_SIZE * 2, Qwen35Config::HIDDEN_SIZE,
            dst.arena, lw.gate_up_proj, stream);
        up.load_fused_fp8(
            sf, {p + "mlp.down_proj.weight"}, {p + "mlp.down_proj.weight_scale_inv"},
            Qwen35Config::HIDDEN_SIZE, Qwen35Config::INTERMEDIATE_SIZE,
            dst.arena, lw.down_proj, stream);
    }
}

} // namespace

void load_single_target_layer(
    const std::string& model_dir,
    int layer_idx,
    TargetWeights& dst,
    cudaStream_t stream
) {
    if (dst.arena.capacity() == 0) {
        dst.arena.init(1024ULL * 1024 * 1024); // 1 GB for unit test
    }
    StagingUploader up;
    up.init(200ULL * 1024 * 1024);
    SafetensorsFile sf;
    sf.open(model_dir + "/layers-" + std::to_string(layer_idx) + ".safetensors");
    load_layer_from_file(sf, layer_idx, up, dst, stream);
    up.destroy();
}

void load_target_weights(
    const std::string& model_dir,
    TargetWeights& dst,
    bool build_fp8_lm_head,
    cudaStream_t stream
) {
    auto t0 = std::chrono::steady_clock::now();
    // Target arena: ~24.4 GB layers + 5.09 GB embed+lm_head + optional 1.28 GB fp8 lm_head = ~31.0 GB
    size_t arena_bytes = build_fp8_lm_head ? (31500ULL * 1024 * 1024) : (30200ULL * 1024 * 1024);
    dst.arena.init(arena_bytes);

    StagingUploader up;
    up.init(200ULL * 1024 * 1024); // 200 MB staging (fits gate_up = 178 MB!)

    // Prefetch threads to warm NFS page cache ahead of GPU upload!
    std::vector<std::thread> prefetchers;
    auto prefetch_file = [](std::string path) {
        SafetensorsFile sf;
        try {
            sf.open(path);
            // Touch pages sequentially via madvise/read
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

    // Prefetch in sliding window of 4 threads so we saturate NFS without thrashing RAM
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

    fprintf(stderr, "[loader] Loading Qwen3.8-27B-FP8 (64 layers + outside.safetensors)...\n");
    kick_prefetch(5);

    for (int L = 0; L < Qwen35Config::NUM_LAYERS; ++L) {
        kick_prefetch(L + 4);
        SafetensorsFile sf;
        sf.open(model_dir + "/layers-" + std::to_string(L) + ".safetensors");
        load_layer_from_file(sf, L, up, dst, stream);
        if ((L + 1) % 16 == 0) {
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "[loader]   loaded %d/64 layers (%.1f s, %.2f GB used)\n",
                    L + 1, elapsed, dst.arena.used() / 1e9);
        }
    }
    for (auto& th : prefetchers) th.join();

    // Load outside.safetensors (skip model.visual.*)
    {
        SafetensorsFile sf;
        sf.open(model_dir + "/outside.safetensors");
        dst.embed_tokens = up.load_bf16(sf, "model.language_model.embed_tokens.weight", dst.arena, stream);
        dst.final_norm = up.load_bf16(sf, "model.language_model.norm.weight", dst.arena, stream);
        dst.lm_head = up.load_bf16(sf, "lm_head.weight", dst.arena, stream);

        if (build_fp8_lm_head) {
            // Quantize lm_head directly on GPU from dst.lm_head into FP8 block-128 MMA layout!
            int N = Qwen35Config::VOCAB_SIZE;
            int K = Qwen35Config::HIDDEN_SIZE;
            dst.lm_head_fp8.N = N;
            dst.lm_head_fp8.K = K;
            size_t w_bytes = (size_t)N * K;
            int num_scales = (N / 128) * (K / 128);
            dst.lm_head_fp8.w_packed = dst.arena.alloc_typed<uint4>(w_bytes / 16);
            dst.lm_head_fp8.scales_raw = dst.arena.alloc_typed<float>(num_scales);
            dst.lm_head_fp8.scales_fused = dst.arena.alloc_typed<float>(num_scales);
            launch_quantize_bf16_to_fp8_mma(
                dst.lm_head, dst.lm_head_fp8.w_packed,
                dst.lm_head_fp8.scales_raw, dst.lm_head_fp8.scales_fused, N, K, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            dst.has_fp8_lm_head = true;
        }
    }

    up.destroy();
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "[loader] Target loaded in %.2f s (%.2f GB VRAM)\n", elapsed, dst.arena.used() / 1e9);
}

void load_draft_weights(
    const std::string& draft_dir,
    DraftWeights& dst,
    bool quantize_to_fp8,
    cudaStream_t stream
) {
    auto t0 = std::chrono::steady_clock::now();
    dst.use_fp8 = quantize_to_fp8;
    // FP8 draft: ~1.8 GB weights + 0.25 GB codebooks + 0.13 GB conv_proj = ~2.3 GB (vs 4.0 GB BF16)
    size_t arena_bytes = quantize_to_fp8 ? (2600ULL * 1024 * 1024) : (4200ULL * 1024 * 1024);
    dst.arena.init(arena_bytes);

    StagingUploader up;
    up.init(400ULL * 1024 * 1024); // Fits BF16 gate_up (356 MB) & fc (262 MB)!

    SafetensorsFile sf;
    sf.open(draft_dir + "/model.safetensors");

    if (quantize_to_fp8) {
        up.load_fused_bf16_as_fp8(
            sf, {"fc.weight"},
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
            up.load_fused_bf16_as_fp8(
                sf, {p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"},
                DFlash2Config::QKV_DIM, DFlash2Config::HIDDEN_SIZE,
                dst.arena, lw.qkv_fp8, stream);
            up.load_fused_bf16_as_fp8(
                sf, {p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"},
                DFlash2Config::KV_DIM * 2, DFlash2Config::HIDDEN_SIZE,
                dst.arena, lw.kv_only_fp8, stream);
            up.load_fused_bf16_as_fp8(
                sf, {p + "self_attn.o_proj.weight"},
                DFlash2Config::HIDDEN_SIZE, DFlash2Config::Q_DIM,
                dst.arena, lw.o_fp8, stream);
            up.load_fused_bf16_as_fp8(
                sf, {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"},
                DFlash2Config::INTERMEDIATE_SIZE * 2, DFlash2Config::HIDDEN_SIZE,
                dst.arena, lw.gate_up_fp8, stream);
            up.load_fused_bf16_as_fp8(
                sf, {p + "mlp.down_proj.weight"},
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

    up.destroy();
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "[loader] DFlash2 draft loaded (%s) in %.2f s (%.2f GB VRAM)\n",
            quantize_to_fp8 ? "FP8" : "BF16", elapsed, dst.arena.used() / 1e9);
}

} // namespace amperegen
