#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(_err), (int)_err); \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err)); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _st = (call); \
    if (_st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[cuBLAS ERROR] %s:%d: status=%d\n", __FILE__, __LINE__, (int)_st); \
        throw std::runtime_error("cuBLAS error " + std::to_string((int)_st)); \
    } \
} while (0)

namespace amperegen {

// Simple static bump-pointer GPU memory arena (128-byte aligned allocations)
class GpuArena {
public:
    GpuArena() = default;
    ~GpuArena() { release(); }

    GpuArena(const GpuArena&) = delete;
    GpuArena& operator=(const GpuArena&) = delete;

    void init(size_t capacity_bytes) {
        release();
        CUDA_CHECK(cudaMalloc(&base_, capacity_bytes));
        capacity_ = capacity_bytes;
        offset_ = 0;
    }

    void release() {
        if (base_) {
            cudaFree(base_);
            base_ = nullptr;
        }
        capacity_ = 0;
        offset_ = 0;
    }

    void reset() { offset_ = 0; }

    void* alloc(size_t bytes, size_t align = 256) {
        size_t aligned = (offset_ + align - 1) & ~(align - 1);
        if (aligned + bytes > capacity_) {
            fprintf(stderr, "[GpuArena OOM] requested %zu bytes (used %zu / %zu)\n",
                    bytes, aligned, capacity_);
            throw std::runtime_error("GpuArena OOM");
        }
        void* ptr = static_cast<uint8_t*>(base_) + aligned;
        offset_ = aligned + bytes;
        return ptr;
    }

    template <typename T>
    T* alloc_typed(size_t count, size_t align = 256) {
        return static_cast<T*>(alloc(count * sizeof(T), align));
    }

    size_t used() const { return offset_; }
    size_t capacity() const { return capacity_; }
    void* base() const { return base_; }

private:
    void* base_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
};

struct CudaContext {
    cublasHandle_t cublas = nullptr;
    cublasLtHandle_t cublas_lt = nullptr;
    cudaStream_t stream = nullptr;
    cudaStream_t aux_stream = nullptr;
    void* cublas_workspace = nullptr;
    size_t cublas_workspace_bytes = 32 * 1024 * 1024; // 32 MB

    void init() {
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&aux_stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasCreate(&cublas));
        CUBLAS_CHECK(cublasSetStream(cublas, stream));
        CUBLAS_CHECK(cublasLtCreate(&cublas_lt));
        CUDA_CHECK(cudaMalloc(&cublas_workspace, cublas_workspace_bytes));
        CUBLAS_CHECK(cublasSetWorkspace(cublas, cublas_workspace, cublas_workspace_bytes));
    }

    void destroy() {
        if (cublas_workspace) { cudaFree(cublas_workspace); cublas_workspace = nullptr; }
        if (cublas_lt) { cublasLtDestroy(cublas_lt); cublas_lt = nullptr; }
        if (cublas) { cublasDestroy(cublas); cublas = nullptr; }
        if (aux_stream) { cudaStreamDestroy(aux_stream); aux_stream = nullptr; }
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
    }
};

} // namespace amperegen
