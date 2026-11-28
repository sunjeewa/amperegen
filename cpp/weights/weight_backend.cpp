#include "weights/weight_backend.hpp"
#include "loader/model_loader.hpp"
#include "runtime/cuda_utils.cuh"

#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace amperegen {

#ifdef AMPEREGEN_ENABLE_WEIGHT_QWEN_FP8
const WeightBackend* get_qwen_fp8_backend();
#endif

#ifdef AMPEREGEN_ENABLE_WEIGHT_REDHAT_INT4
const WeightBackend* get_redhat_int4_backend();
#endif

const std::vector<const WeightBackend*>& WeightRegistry::all() {
    static const std::vector<const WeightBackend*> backends = []() {
        std::vector<const WeightBackend*> v;
#ifdef AMPEREGEN_ENABLE_WEIGHT_QWEN_FP8
        v.push_back(get_qwen_fp8_backend());
#endif
#ifdef AMPEREGEN_ENABLE_WEIGHT_REDHAT_INT4
        v.push_back(get_redhat_int4_backend());
#endif
        return v;
    }();
    return backends;
}

const WeightBackend* WeightRegistry::find(const std::string& id) {
    for (const auto* b : all()) {
        if (id == b->id()) return b;
    }
    return nullptr;
}

std::string WeightRegistry::compiled_ids_string() {
    std::string out;
    for (const auto* b : all()) {
        if (!out.empty()) out += ", ";
        out += b->id();
    }
    return out.empty() ? "(none)" : out;
}

const WeightBackend* WeightRegistry::resolve(
    const std::string& model_dir,
    const std::string& requested_id
) {
    const auto& backends = all();
    if (backends.empty()) {
        throw std::runtime_error("No weight backends were compiled into amperegen!");
    }

    if (!requested_id.empty() && requested_id != "auto") {
        const WeightBackend* b = find(requested_id);
        if (!b) {
            throw std::runtime_error(
                "Requested weight backend '" + requested_id + "' is not compiled in (available: " +
                compiled_ids_string() + ")");
        }
        std::string reason;
        if (!b->probe(model_dir, reason)) {
            throw std::runtime_error(
                "Checkpoint '" + model_dir + "' failed qualification for weight backend '" +
                requested_id + "': " + reason);
        }
        return b;
    }

    std::vector<const WeightBackend*> matched;
    std::ostringstream rejects;
    for (const auto* b : backends) {
        std::string reason;
        if (b->probe(model_dir, reason)) {
            matched.push_back(b);
        } else {
            rejects << "\n  - " << b->id() << ": " << reason;
        }
    }

    if (matched.size() == 1) {
        return matched[0];
    }
    if (matched.empty()) {
        throw std::runtime_error(
            "Checkpoint '" + model_dir + "' did not match any qualified amperegen weight plugin "
            "(compiled: " + compiled_ids_string() + "):" + rejects.str());
    }

    std::string ids;
    for (const auto* b : matched) {
        if (!ids.empty()) ids += ", ";
        ids += b->id();
    }
    throw std::runtime_error(
        "Checkpoint '" + model_dir + "' matched multiple weight plugins (" + ids +
        "); disambiguate with --weight-backend");
}

void launch_bf16_linear(
    cublasHandle_t cublas,
    const __nv_bfloat16* X,
    const __nv_bfloat16* W,
    __nv_bfloat16* Y,
    int M, int N, int K,
    cudaStream_t stream,
    bool accumulate
) {
    if (M == 0) return;
    CUBLAS_CHECK(cublasSetStream(cublas, stream));
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        W, CUDA_R_16BF, K,
        X, CUDA_R_16BF, K,
        &beta,
        Y, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void maybe_build_draft_fp8_lm_head(TargetWeights& dst, cudaStream_t stream) {
    const WeightBackend* fp8 = WeightRegistry::find("qwen_fp8");
    if (!fp8) {
        fprintf(stderr, "[weights] Note: qwen_fp8 plugin not compiled in; keeping BF16 lm_head for DFlash2 draft.\n");
        dst.has_fp8_lm_head = false;
        return;
    }
    if (fp8->quantize_from_bf16_device(
            dst.lm_head, Qwen35Config::VOCAB_SIZE, Qwen35Config::HIDDEN_SIZE,
            dst.arena, dst.lm_head_fp8, stream)) {
        dst.has_fp8_lm_head = true;
    }
}

} // namespace amperegen
