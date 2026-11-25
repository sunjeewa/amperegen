#include "loader/safetensors.hpp"

#include <cctype>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace amperegen {

namespace {

// Minimal strict JSON parser specifically for safetensors headers.
struct JsonCursor {
    const char* s;
    size_t len;
    size_t pos = 0;

    void skip_ws() {
        while (pos < len && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
            ++pos;
    }

    char peek() {
        skip_ws();
        if (pos >= len) throw std::runtime_error("Unexpected EOF in safetensors JSON");
        return s[pos];
    }

    void expect(char c) {
        if (peek() != c) {
            throw std::runtime_error(std::string("Expected '") + c + "' in safetensors JSON");
        }
        ++pos;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (pos < len) {
            char c = s[pos++];
            if (c == '"') return out;
            if (c == '\\') {
                if (pos >= len) throw std::runtime_error("Truncated escape in JSON");
                char e = s[pos++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default: out.push_back(e); break;
                }
            } else {
                out.push_back(c);
            }
        }
        throw std::runtime_error("Unterminated string in safetensors JSON");
    }

    uint64_t parse_uint() {
        skip_ws();
        if (pos >= len || !std::isdigit((unsigned char)s[pos]))
            throw std::runtime_error("Expected uint in safetensors JSON");
        uint64_t val = 0;
        while (pos < len && std::isdigit((unsigned char)s[pos])) {
            val = val * 10 + (uint64_t)(s[pos++] - '0');
        }
        return val;
    }

    void skip_value() {
        char c = peek();
        if (c == '"') {
            (void)parse_string();
        } else if (c == '{') {
            ++pos;
            if (peek() == '}') { ++pos; return; }
            while (true) {
                (void)parse_string();
                expect(':');
                skip_value();
                if (peek() == ',') { ++pos; continue; }
                expect('}');
                break;
            }
        } else if (c == '[') {
            ++pos;
            if (peek() == ']') { ++pos; return; }
            while (true) {
                skip_value();
                if (peek() == ',') { ++pos; continue; }
                expect(']');
                break;
            }
        } else {
            while (pos < len && s[pos] != ',' && s[pos] != '}' && s[pos] != ']' &&
                   !std::isspace((unsigned char)s[pos])) {
                ++pos;
            }
        }
    }
};

DType parse_dtype(const std::string& d) {
    if (d == "BF16") return DType::BF16;
    if (d == "F16") return DType::FP16;
    if (d == "F32") return DType::FP32;
    if (d == "F8_E4M3") return DType::F8_E4M3;
    if (d == "I32") return DType::INT32;
    if (d == "I64") return DType::INT64;
    if (d == "U8") return DType::U8;
    return DType::UNKNOWN;
}

} // namespace

SafetensorsFile::~SafetensorsFile() { close(); }

SafetensorsFile::SafetensorsFile(SafetensorsFile&& o) noexcept
    : path_(std::move(o.path_)),
      fd_(o.fd_),
      mapped_(o.mapped_),
      file_size_(o.file_size_),
      data_base_offset_(o.data_base_offset_),
      tensors_(std::move(o.tensors_)) {
    o.fd_ = -1;
    o.mapped_ = nullptr;
    o.file_size_ = 0;
}

SafetensorsFile& SafetensorsFile::operator=(SafetensorsFile&& o) noexcept {
    if (this != &o) {
        close();
        path_ = std::move(o.path_);
        fd_ = o.fd_;
        mapped_ = o.mapped_;
        file_size_ = o.file_size_;
        data_base_offset_ = o.data_base_offset_;
        tensors_ = std::move(o.tensors_);
        o.fd_ = -1;
        o.mapped_ = nullptr;
        o.file_size_ = 0;
    }
    return *this;
}

void SafetensorsFile::open(const std::string& path) {
    close();
    path_ = path;
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("Failed to open safetensors file: " + path);
    }
    struct stat st {};
    if (fstat(fd_, &st) != 0) {
        close();
        throw std::runtime_error("fstat failed on: " + path);
    }
    file_size_ = static_cast<size_t>(st.st_size);
    if (file_size_ < 8) {
        close();
        throw std::runtime_error("Safetensors file too small: " + path);
    }
    mapped_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_ == MAP_FAILED) {
        mapped_ = nullptr;
        close();
        throw std::runtime_error("mmap failed on: " + path);
    }
    madvise(mapped_, file_size_, MADV_SEQUENTIAL);

    uint64_t hdr_len = 0;
    std::memcpy(&hdr_len, mapped_, sizeof(uint64_t));
    if (8 + hdr_len > file_size_) {
        close();
        throw std::runtime_error("Corrupt safetensors header length in: " + path);
    }
    data_base_offset_ = 8 + hdr_len;

    JsonCursor cur{static_cast<const char*>(mapped_) + 8, static_cast<size_t>(hdr_len), 0};
    cur.expect('{');
    if (cur.peek() != '}') {
        while (true) {
            std::string key = cur.parse_string();
            cur.expect(':');
            if (key == "__metadata__") {
                cur.skip_value();
            } else {
                TensorInfo ti;
                ti.name = key;
                cur.expect('{');
                if (cur.peek() != '}') {
                    while (true) {
                        std::string field = cur.parse_string();
                        cur.expect(':');
                        if (field == "dtype") {
                            ti.dtype = parse_dtype(cur.parse_string());
                        } else if (field == "shape") {
                            cur.expect('[');
                            if (cur.peek() != ']') {
                                while (true) {
                                    ti.shape.push_back((int64_t)cur.parse_uint());
                                    if (cur.peek() == ',') { ++cur.pos; continue; }
                                    break;
                                }
                            }
                            cur.expect(']');
                        } else if (field == "data_offsets") {
                            cur.expect('[');
                            ti.data_offset_begin = cur.parse_uint();
                            cur.expect(',');
                            ti.data_offset_end = cur.parse_uint();
                            cur.expect(']');
                        } else {
                            cur.skip_value();
                        }
                        if (cur.peek() == ',') { ++cur.pos; continue; }
                        break;
                    }
                }
                cur.expect('}');
                tensors_.emplace(std::move(key), std::move(ti));
            }
            if (cur.peek() == ',') { ++cur.pos; continue; }
            break;
        }
    }
    cur.expect('}');
}

void SafetensorsFile::close() {
    if (mapped_) {
        munmap(mapped_, file_size_);
        mapped_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    file_size_ = 0;
    data_base_offset_ = 0;
    tensors_.clear();
}

const TensorInfo& SafetensorsFile::info(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found in " + path_ + ": " + name);
    }
    return it->second;
}

const uint8_t* SafetensorsFile::data_ptr(const std::string& name) const {
    const auto& ti = info(name);
    return static_cast<const uint8_t*>(mapped_) + data_base_offset_ + ti.data_offset_begin;
}

} // namespace amperegen
