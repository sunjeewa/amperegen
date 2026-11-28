#pragma once
#include "kernels/kernels.cuh"
#include "model/model_config.hpp"
#include "runtime/cuda_utils.cuh"
#include "weights/weight_backend.hpp"
#include <string>
#include <vector>

namespace amperegen {

struct FullAttnLayerWeights {
    __nv_bfloat16* input_layernorm = nullptr;          // [5120]
    __nv_bfloat16* post_attention_layernorm = nullptr; // [5120]
    __nv_bfloat16* q_norm = nullptr;                   // [256]
    __nv_bfloat16* k_norm = nullptr;                   // [256]
    QuantLinearWeight qkv_proj;                        // [14336, 5120] fused q_gate(12288)+k(1024)+v(1024)
    QuantLinearWeight o_proj;                          // [5120, 6144]
    QuantLinearWeight gate_up_proj;                    // [34816, 5120]
    QuantLinearWeight down_proj;                       // [5120, 17408]
};

struct GdnLayerWeights {
    __nv_bfloat16* input_layernorm = nullptr;          // [5120]
    __nv_bfloat16* post_attention_layernorm = nullptr; // [5120]
    QuantLinearWeight in_proj_qkvz;                    // [16384, 5120] fused qkv(10240)+z(6144)
    __nv_bfloat16* in_proj_ba = nullptr;               // [96, 5120] BF16 fused b(48)+a(48)
    __nv_bfloat16* conv1d_w = nullptr;                 // [10240, 4] BF16
    float* A_log = nullptr;                            // [48] FP32
    float* dt_bias = nullptr;                          // [48] FP32
    __nv_bfloat16* gdn_norm = nullptr;                 // [128] BF16
    QuantLinearWeight out_proj;                        // [5120, 6144]
    QuantLinearWeight gate_up_proj;                    // [34816, 5120]
    QuantLinearWeight down_proj;                       // [5120, 17408]
};

struct TargetWeights {
    GpuArena arena;
    const WeightBackend* backend = nullptr;
    __nv_bfloat16* embed_tokens = nullptr;  // [248320, 5120] BF16
    __nv_bfloat16* final_norm = nullptr;    // [5120] BF16 (GemmaRMSNorm +1)
    __nv_bfloat16* lm_head = nullptr;       // [248320, 5120] BF16 (exact for target verify!)
    QuantLinearWeight lm_head_fp8;          // [248320, 5120] optional FP8 copy for lossless draft top-16!
    bool has_fp8_lm_head = false;

    FullAttnLayerWeights full_attn[Qwen35Config::NUM_FULL_ATTN_LAYERS];
    GdnLayerWeights gdn[Qwen35Config::NUM_GDN_LAYERS];
};

struct DraftLayerWeights {
    __nv_bfloat16* input_layernorm = nullptr;          // [5120] (plain RMSNorm)
    __nv_bfloat16* post_attention_layernorm = nullptr; // [5120] (plain RMSNorm)
    __nv_bfloat16* q_norm = nullptr;                   // [128] (plain RMSNorm)
    __nv_bfloat16* k_norm = nullptr;                   // [128] (plain RMSNorm)

    __nv_bfloat16* attn_conv_base = nullptr;           // [2, 2, 5120]
    __nv_bfloat16* attn_conv_proj = nullptr;           // [1280, 5120] BF16
    __nv_bfloat16* mlp_conv_base = nullptr;            // [2, 2, 5120]
    __nv_bfloat16* mlp_conv_proj = nullptr;            // [1280, 5120] BF16

    // Can run either quantized FP8 (2x faster draft, 100% lossless for target!) or exact BF16
    QuantLinearWeight qkv_fp8;      // [6144, 5120]
    QuantLinearWeight kv_only_fp8;  // [2048, 5120] (for draft KV materialization!)
    QuantLinearWeight o_fp8;        // [5120, 4096]
    QuantLinearWeight gate_up_fp8;  // [34816, 5120]
    QuantLinearWeight down_fp8;     // [5120, 17408]

    __nv_bfloat16* qkv_bf16 = nullptr;
    __nv_bfloat16* kv_only_bf16 = nullptr;
    __nv_bfloat16* o_bf16 = nullptr;
    __nv_bfloat16* gate_up_bf16 = nullptr;
    __nv_bfloat16* down_bf16 = nullptr;
};

struct DraftWeights {
    GpuArena arena;
    bool use_fp8 = true;
    QuantLinearWeight fc_fp8;                    // [5120, 25600]
    __nv_bfloat16* fc_bf16 = nullptr;            // [5120, 25600] (if !use_fp8)
    __nv_bfloat16* hidden_norm = nullptr;        // [5120]
    __nv_bfloat16* final_norm = nullptr;         // [5120]
    __nv_bfloat16* pred_codebook = nullptr;      // [248320, 256]
    __nv_bfloat16* succ_codebook = nullptr;      // [248320, 256]
    __nv_bfloat16* selector_hidden_proj = nullptr; // [256, 5120]
    DraftLayerWeights layers[DFlash2Config::NUM_LAYERS];
};

// Load single target layer via qualified weight plugin (used for Stage-1 unit test of layer 0 & layer 3).
void load_single_target_layer(
    const std::string& model_dir,
    int layer_idx,
    TargetWeights& dst,
    cudaStream_t stream,
    const std::string& weight_backend = "auto");

// Load full 64-layer target checkpoint + embed + lm_head via qualified weight plugin.
void load_target_weights(
    const std::string& model_dir,
    TargetWeights& dst,
    bool build_fp8_lm_head,
    cudaStream_t stream,
    const std::string& weight_backend = "auto");

// Load DFlash2 draft model checkpoint (optional load-time FP8 compression via qwen_fp8 plugin if compiled in).
void load_draft_weights(
    const std::string& draft_dir,
    DraftWeights& dst,
    bool quantize_to_fp8,
    cudaStream_t stream);

} // namespace amperegen
