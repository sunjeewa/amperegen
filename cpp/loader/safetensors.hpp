#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace amperegen {

enum class DType {
    BF16,
    FP16,
    FP32,
    F8_E4M3,
    INT32,
    INT64,
    U8,
    UNKNOWN
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::F8_E4M3:
        case DType::U8: return 1;
        case DType::BF16:
        case DType::FP16: return 2;
        case DType::FP32:
        case DType::INT32: return 4;
        case DType::INT64: return 8;
        default: return 0;
    }
}

struct TensorInfo {
    std::string name;
    DType dtype = DType::UNKNOWN;
    std::vector<int64_t> shape;
    uint64_t data_offset_begin = 0; // relative to data section start
    uint64_t data_offset_end = 0;

    uint64_t nbytes() const { return data_offset_end - data_offset_begin; }
    int64_t numel() const {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

// Lightweight zero-dependency mmap safetensors reader.
class SafetensorsFile {
public:
    SafetensorsFile() = default;
    ~SafetensorsFile();

    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;
    SafetensorsFile(SafetensorsFile&& o) noexcept;
    SafetensorsFile& operator=(SafetensorsFile&& o) noexcept;

    void open(const std::string& path);
    void close();

    bool has(const std::string& name) const { return tensors_.count(name) > 0; }
    const TensorInfo& info(const std::string& name) const;
    const uint8_t* data_ptr(const std::string& name) const;
    const std::unordered_map<std::string, TensorInfo>& tensors() const { return tensors_; }
    const std::string& path() const { return path_; }

private:
    std::string path_;
    int fd_ = -1;
    void* mapped_ = nullptr;
    size_t file_size_ = 0;
    uint64_t data_base_offset_ = 0;
    std::unordered_map<std::string, TensorInfo> tensors_;
};

} // namespace amperegen
