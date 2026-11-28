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
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED libpcre2-8)
pkg_check_modules(UTF8PROC REQUIRED libutf8proc)

# Curated compile-time static weight plugins for Ampere sm_80.
# Select any non-empty subset of: qwen_fp8, redhat_int4.
set(AMPEREGEN_WEIGHT_BACKENDS "qwen_fp8;redhat_int4" CACHE STRING
    "Curated weight plugins to compile into amperegen (subset of: qwen_fp8;redhat_int4)")

set(AMPEREGEN_WEIGHT_SOURCES cpp/weights/weight_backend.cpp)
set(AMPEREGEN_WEIGHT_DEFINES "")
set(AMPEREGEN_ENABLED_BACKENDS "")

foreach(_wb IN LISTS AMPEREGEN_WEIGHT_BACKENDS)
    string(STRIP "${_wb}" _wb)
    if(_wb STREQUAL "")
        continue()
    elseif(_wb STREQUAL "qwen_fp8")
        list(APPEND AMPEREGEN_WEIGHT_SOURCES
            cpp/weights/qwen_fp8/backend.cpp
            cpp/weights/qwen_fp8/gemm.cu
        )
        list(APPEND AMPEREGEN_WEIGHT_DEFINES AMPEREGEN_ENABLE_WEIGHT_QWEN_FP8=1)
        list(APPEND AMPEREGEN_ENABLED_BACKENDS "qwen_fp8")
    elseif(_wb STREQUAL "redhat_int4")
        list(APPEND AMPEREGEN_WEIGHT_SOURCES
            cpp/weights/redhat_int4/backend.cpp
            cpp/weights/redhat_int4/gemm.cu
        )
        list(APPEND AMPEREGEN_WEIGHT_DEFINES AMPEREGEN_ENABLE_WEIGHT_REDHAT_INT4=1)
        list(APPEND AMPEREGEN_ENABLED_BACKENDS "redhat_int4")
    else()
        message(FATAL_ERROR
            "Unknown amperegen weight plugin '${_wb}'. Supported curated plugins: qwen_fp8, redhat_int4.")
    endif()
endforeach()

if(NOT AMPEREGEN_ENABLED_BACKENDS)
    message(FATAL_ERROR "AMPEREGEN_WEIGHT_BACKENDS must include at least one curated weight plugin.")
endif()
message(STATUS "amperegen curated weight plugins: ${AMPEREGEN_ENABLED_BACKENDS}")

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

# CPU-only native C++20 tokenizer & minja chat-template engine
add_library(amperegen_tokenizer STATIC
    cpp/server/tokenizer.cpp
)
target_include_directories(amperegen_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)
set(AMPEREGEN_TOK_SYS_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${PCRE2_INCLUDE_DIRS}
    ${UTF8PROC_INCLUDE_DIRS}
)
# Never pass -isystem /usr/include to GCC (breaks #include_next <stdlib.h> in libstdc++)
list(REMOVE_ITEM AMPEREGEN_TOK_SYS_INCLUDES "/usr/include")
target_include_directories(amperegen_tokenizer SYSTEM PRIVATE
    ${AMPEREGEN_TOK_SYS_INCLUDES}
)
target_compile_options(amperegen_tokenizer PRIVATE ${AMPEREGEN_CXX_FLAGS})
target_link_libraries(amperegen_tokenizer PRIVATE
    ${PCRE2_LINK_LIBRARIES}
    ${UTF8PROC_LINK_LIBRARIES}
)

add_library(amperegen_core STATIC
    cpp/loader/safetensors.cpp
    cpp/loader/model_loader.cpp
    ${AMPEREGEN_WEIGHT_SOURCES}
    cpp/kernels/norms_rope.cu
    cpp/kernels/flash_attn.cu
    cpp/kernels/gdn_kernels.cu
    cpp/kernels/dflash_kernels.cu
    cpp/kernels/sampling.cu
    cpp/model/qwen35_model.cpp
    cpp/model/dflash_model.cpp
)

target_compile_definitions(amperegen_core PUBLIC
    ${AMPEREGEN_WEIGHT_DEFINES}
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

# Standalone layer & model verification runner (Stage 0/1/2/3 tests + weight plugin self-tests + benchmark)
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
target_compile_options(amperegen PRIVATE ${AMPEREGEN_CXX_FLAGS})
target_link_libraries(amperegen PRIVATE amperegen_core amperegen_tokenizer)

# CPU-only unit tests (JSON parser + qwen3_coder streaming/non-streaming tool parser)
add_executable(amperegen_unit_tests
    tests/test_tool_parser.cpp
)
target_include_directories(amperegen_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_compile_options(amperegen_unit_tests PRIVATE ${AMPEREGEN_CXX_FLAGS})

# CPU-only native BPE detokenizer unit tests
add_executable(amperegen_bpe_detok_tests
    tests/test_bpe_detokenizer.cpp
)
target_include_directories(amperegen_bpe_detok_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_compile_options(amperegen_bpe_detok_tests PRIVATE ${AMPEREGEN_CXX_FLAGS})

# CPU-only native tokenizer + minja chat-template unit & parity tests
add_executable(amperegen_tokenizer_tests
    tests/test_native_tokenizer.cpp
)
target_compile_options(amperegen_tokenizer_tests PRIVATE ${AMPEREGEN_CXX_FLAGS})
target_link_libraries(amperegen_tokenizer_tests PRIVATE amperegen_tokenizer)
