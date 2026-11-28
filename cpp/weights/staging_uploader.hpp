#pragma once
// Shared pinned host-to-device staging helper used by curated weight plugins & draft loader.

#include "loader/safetensors.hpp"
#include "runtime/cuda_utils.cuh"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace amperegen {

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
        cap = 0;
    }

    // Copy arbitrary host buffer to device via pinned staging chunks.
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

    // Concatenate tensors sequentially into h_stage and upload synchronously to d_stage.
    size_t stage_concat_to_device(
        const SafetensorsFile& sf,
        const std::vector<std::string>& names,
        size_t expected_bytes,
        const char* label,
        cudaStream_t stream
    ) {
        size_t off = 0;
        for (const auto& n : names) {
            const auto& ti = sf.info(n);
            size_t nb = ti.nbytes();
            if (off + nb > cap) {
                throw std::runtime_error(std::string(label) + ": staging buffer overflow");
            }
            std::memcpy(h_stage + off, sf.data_ptr(n), nb);
            off += nb;
        }
        if (expected_bytes != 0 && off != expected_bytes) {
            throw std::runtime_error(std::string(label) + ": byte size mismatch");
        }
        CUDA_CHECK(cudaMemcpyAsync(d_stage, h_stage, off, cudaMemcpyHostToDevice, stream));
        return off;
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

    // Load either BF16 or FP32 tensor into device FP32 array.
    float* load_as_fp32(
        const SafetensorsFile& sf,
        const std::string& name,
        GpuArena& arena,
        cudaStream_t stream
    ) {
        const auto& ti = sf.info(name);
        size_t count = static_cast<size_t>(ti.numel());
        auto* dst = arena.alloc_typed<float>(count);
        if (ti.dtype == DType::FP32) {
            copy_to_device(sf.data_ptr(name), dst, count * sizeof(float), stream);
            return dst;
        }
        const uint16_t* src = reinterpret_cast<const uint16_t*>(sf.data_ptr(name));
        std::vector<float> tmp(count);
        for (size_t i = 0; i < count; ++i) {
            uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
            std::memcpy(&tmp[i], &bits, sizeof(float));
        }
        copy_to_device(tmp.data(), dst, count * sizeof(float), stream);
        return dst;
    }
};

} // namespace amperegen
