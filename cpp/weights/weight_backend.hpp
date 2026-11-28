#pragma once
// Curated static weight-plugin interface & registry for amperegen (Ampere sm_80).
//
// Each supported checkpoint format lives in its own self-contained folder under
// `cpp/weights/<id>/` with its own strict checkpoint qualification probe, loader,
// offline MMA packer, and compile-time-specialized sm_80 Tensor Core kernels.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace amperegen {

class GpuArena;
class SafetensorsFile;
struct TargetWeights;
class WeightBackend;

// Backend-agnostic linear weight handle stored in model layers.
// Inside captured CUDA graphs, `backend->launch_linear(...)` runs only once on host
// during graph capture; replayed graphs execute strictly the plugin's sm_80 kernels.
struct QuantLinearWeight {
    const WeightBackend* backend = nullptr;
    void* w_packed = nullptr;   // backend-owned warp-MMA packed device weight buffer
    void* scales = nullptr;     // backend-owned primary scale buffer (e.g. FP8 fused / INT4 packed)
    void* aux_scales = nullptr; // optional secondary scale buffer (e.g. FP8 raw scales)
    int N = 0;
    int K = 0;

    bool valid() const { return backend != nullptr && w_packed != nullptr && N > 0 && K > 0; }
};

class WeightBackend {
public:
    virtual ~WeightBackend() = default;

    // Short stable plugin ID (e.g. "qwen_fp8", "redhat_int4").
    virtual const char* id() const = 0;

    // Human-readable format summary for logs and /get_server_info.
    virtual const char* description() const = 0;

    // Strict qualification check on model_dir (config.json + safetensors header).
    // Returns false and populates `reason` if the checkpoint does not strictly match.
    virtual bool probe(const std::string& model_dir, std::string& reason) const = 0;

    // Load a single decoder layer (used by Stage-1 numerical verification).
    virtual void load_single_layer(
        const std::string& model_dir,
        int layer_idx,
        TargetWeights& dst,
        cudaStream_t stream) const = 0;

    // Load full 64-layer target weights + embed_tokens + final_norm + lm_head.
    virtual void load_target(
        const std::string& model_dir,
        TargetWeights& dst,
        bool build_fp8_lm_head,
        cudaStream_t stream) const = 0;

    // Execute Y[M, N] = X[M, K] * W[N, K]^T using this plugin's sm_80 kernels.
    virtual void launch_linear(
        cublasHandle_t cublas,
        const __nv_bfloat16* X,
        const QuantLinearWeight& W,
        __nv_bfloat16* Y,
        float* d_splitk_scratch,
        __nv_bfloat16* d_bf16_scratch,
        int M,
        cudaStream_t stream) const = 0;

    // Optional: quantize a device BF16 weight [N, K] into this plugin's format (supported by qwen_fp8
    // for lossless DFlash2 draft & draft lm_head compression). Returns false if unsupported.
    virtual bool quantize_from_bf16_device(
        const __nv_bfloat16* d_w_bf16,
        int N, int K,
        GpuArena& arena,
        QuantLinearWeight& out,
        cudaStream_t stream) const {
        (void)d_w_bf16; (void)N; (void)K; (void)arena; (void)out; (void)stream;
        return false;
    }

    // Fast synthetic GPU numerical parity self-test (no checkpoint files needed).
    virtual bool self_test(cublasHandle_t cublas, cudaStream_t stream) const = 0;
};

class WeightRegistry {
public:
    // Returns all weight plugins compiled into this binary.
    static const std::vector<const WeightBackend*>& all();

    // Find a compiled plugin by exact ID (returns nullptr if not compiled in).
    static const WeightBackend* find(const std::string& id);

    // Resolve the qualified plugin for `model_dir`.
    // If `requested_id` is empty or "auto", probes all compiled plugins and requires an exact match;
    // otherwise validates that `requested_id` exists and strictly qualifies `model_dir`.
    // Throws std::runtime_error with per-plugin rejection reasons if unqualified.
    static const WeightBackend* resolve(
        const std::string& model_dir,
        const std::string& requested_id = "auto");

    // Comma-separated list of compiled plugin IDs.
    static std::string compiled_ids_string();
};

// Dispatch linear projection to the weight's owning backend plugin.
inline void launch_quant_linear(
    cublasHandle_t cublas,
    const __nv_bfloat16* X,
    const QuantLinearWeight& W,
    __nv_bfloat16* Y,
    float* d_splitk_scratch,
    __nv_bfloat16* d_bf16_scratch,
    int M,
    cudaStream_t stream
) {
    if (M == 0) return;
    W.backend->launch_linear(cublas, X, W, Y, d_splitk_scratch, d_bf16_scratch, M, stream);
}

// Dense BF16 linear via cuBLAS: Y[M, N] = X[M, K] * W[N, K]^T (row-major).
void launch_bf16_linear(
    cublasHandle_t cublas,
    const __nv_bfloat16* X,
    const __nv_bfloat16* W,
    __nv_bfloat16* Y,
    int M, int N, int K,
    cudaStream_t stream,
    bool accumulate = false);

// Helper shared by weight plugins: build optional FP8 draft lm_head from dst.lm_head if qwen_fp8 is compiled in.
void maybe_build_draft_fp8_lm_head(TargetWeights& dst, cudaStream_t stream);

} // namespace amperegen
