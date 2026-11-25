cmake_minimum_required(VERSION 3.22)
project(amperegen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 80)

option(AMPEREGEN_PORTABLE "Build portable x86-64-v3 binaries instead of -march=native" OFF)

if(AMPEREGEN_PORTABLE)
    set(AMPEREGEN_ARCH_FLAG -march=x86-64-v3)
else()
    set(AMPEREGEN_ARCH_FLAG -march=native)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Common CUDA flags for Ampere sm_80 (CMP 170HX / A100)
set(AMPEREGEN_CUDA_FLAGS
    -O3
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -lineinfo
    -Xcompiler=-O3,${AMPEREGEN_ARCH_FLAG}
)

set(AMPEREGEN_CXX_FLAGS
    -O3
    ${AMPEREGEN_ARCH_FLAG}
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

# CPU-only unit tests (JSON parser + qwen3_coder streaming/non-streaming tool parser)
add_executable(amperegen_unit_tests
    tests/test_tool_parser.cpp
)
target_include_directories(amperegen_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_compile_options(amperegen_unit_tests PRIVATE ${AMPEREGEN_CXX_FLAGS})
