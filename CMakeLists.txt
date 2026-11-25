cmake_minimum_required(VERSION 3.24)
project(amperegen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Common CUDA flags for CMP 170HX (sm_80, Fedora 44 gcc-15 + CUDA 12.8 compat)
set(AMPEREGEN_CUDA_FLAGS
    -O3
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -lineinfo
    -Xcompiler=-O3,-march=native
)

set(AMPEREGEN_CXX_FLAGS
    -O3
    -march=native
    -Wall
    -Wextra
    -Wno-unused-parameter
)

add_library(amperegen_core STATIC
    cpp/loader/safetensors.cpp
    cpp/loader/model_loader.cpp
    cpp/kernels/fp8_gemm.cu
    cpp/kernels/norms_rope.cu
    cpp/kernels/flash_attn.cu
    cpp/kernels/gdn_kernels.cu
    cpp/kernels/dflash_kernels.cu
    cpp/kernels/sampling.cu
    cpp/model/qwen35_model.cpp
    cpp/model/dflash_model.cpp
)

target_include_directories(amperegen_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)

target_compile_options(amperegen_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${AMPEREGEN_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${AMPEREGEN_CUDA_FLAGS}>
)

target_link_libraries(amperegen_core PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cuda_driver
    Threads::Threads
)

# Standalone layer & model verification runner (Stage 0/1/2/3 tests + benchmark)
add_executable(amperegen_test
    cpp/test_runner.cpp
)
target_link_libraries(amperegen_test PRIVATE amperegen_core)

# Production OpenAI-compatible HTTP server
add_executable(amperegen
    cpp/main.cpp
    cpp/server/http_server.cpp
    cpp/scheduler/scheduler.cpp
)
target_link_libraries(amperegen PRIVATE amperegen_core)
