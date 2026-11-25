#pragma once
// Minimal zero-dependency C++20 JSON DOM parser & serializer for amperegen HTTP server & sidecar IPC.
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace amperegen {

struct Json {
    using Object = std::map<std::string, Json>;
    using Array = std::vector<Json>;
    using Value = std::variant<std::nullptr_t, bool, int64_t, double, std::string, Array, Object>;

    Value val = nullptr;

    Json() = default;
    Json(std::nullptr_t) : val(nullptr) {}
    Json(bool b) : val(b) {}
    Json(int v) : val((int64_t)v) {}
    Json(int64_t v) : val(v) {}
    Json(double v) : val(v) {}
    Json(const char* s) : val(std::string(s)) {}
    Json(std::string s) : val(std::move(s)) {}
    Json(Array a) : val(std::move(a)) {}
    Json(Object o) : val(std::move(o)) {}

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(val); }
    bool is_bool() const { return std::holds_alternative<bool>(val); }
    bool is_int() const { return std::holds_alternative<int64_t>(val); }
    bool is_number() const { return std::holds_alternative<int64_t>(val) || std::holds_alternative<double>(val); }
    bool is_string() const { return std::holds_alternative<std::string>(val); }
    bool is_array() const { return std::holds_alternative<Array>(val); }
    bool is_object() const { return std::holds_alternative<Object>(val); }

    bool as_bool(bool def = false) const { return is_bool() ? std::get<bool>(val) : def; }
    int64_t as_int(int64_t def = 0) const {
        if (std::holds_alternative<int64_t>(val)) return std::get<int64_t>(val);
        if (std::holds_alternative<double>(val)) return (int64_t)std::get<double>(val);
        return def;
    }
    double as_double(double def = 0.0) const {
        if (std::holds_alternative<double>(val)) return std::get<double>(val);
        if (std::holds_alternative<int64_t>(val)) return (double)std::get<int64_t>(val);
        return def;
    }
    const std::string& as_string() const { return std::get<std::string>(val); }
    std::string as_string_or(const std::string& def) const { return is_string() ? std::get<std::string>(val) : def; }
    const Array& as_array() const { return std::get<Array>(val); }
    Array& as_array() { return std::get<Array>(val); }
    const Object& as_object() const { return std::get<Object>(val); }
    Object& as_object() { return std::get<Object>(val); }

    bool contains(const std::string& k) const {
        return is_object() && std::get<Object>(val).count(k) > 0;
    }
    const Json& at(const std::string& k) const { return std::get<Object>(val).at(k); }
    Json& operator[](const std::string& k) {
        if (!is_object()) val = Object{};
        return std::get<Object>(val)[k];
    }

    static std::string escape(const std::string& s) {
        std::string out;
        out.reserve(s.size() + 8);
        out.push_back('"');
        for (unsigned char c : s) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (c < 0x20) {
                        char buf[8];
                        snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    } else {
                        out.push_back((char)c);
                    }
            }
        }
        out.push_back('"');
        return out;
    }

    std::string dump() const {
        if (is_null()) return "null";
        if (is_bool()) return std::get<bool>(val) ? "true" : "false";
        if (is_int()) return std::to_string(std::get<int64_t>(val));
        if (std::holds_alternative<double>(val)) {
            double d = std::get<double>(val);
            if (!std::isfinite(d)) return "null";
            char buf[64];
            snprintf(buf, sizeof(buf), "%.10g", d);
            return buf;
        }
        if (is_string()) return escape(std::get<std::string>(val));
        if (is_array()) {
            std::string out = "[";
            const auto& a = as_array();
            for (size_t i = 0; i < a.size(); ++i) {
                if (i) out.push_back(',');
                out += a[i].dump();
            }
            out.push_back(']');
            return out;
        }
        std::string out = "{";
        bool first = true;
        for (const auto& [k, v] : as_object()) {
            if (!first) out.push_back(',');
            first = false;
            out += escape(k);
            out.push_back(':');
            out += v.dump();
        }
        out.push_back('}');
        return out;
    }

    static Json parse(const std::string& input) {
        size_t pos = 0;
        auto skip = [&]() {
            while (pos < input.size() && std::isspace((unsigned char)input[pos])) ++pos;
        };
        auto parse_str = [&]() -> std::string {
            skip();
            if (pos >= input.size() || input[pos] != '"') throw std::runtime_error("JSON: expected '\"'");
            ++pos;
            std::string out;
            while (pos < input.size()) {
                char c = input[pos++];
                if (c == '"') return out;
                if (c == '\\') {
                    if (pos >= input.size()) throw std::runtime_error("JSON: truncated escape");
                    char e = input[pos++];
                    switch (e) {
                        case '"': out.push_back('"'); break;
                        case '\\': out.push_back('\\'); break;
                        case '/': out.push_back('/'); break;
                        case 'b': out.push_back('\b'); break;
                        case 'f': out.push_back('\f'); break;
                        case 'n': out.push_back('\n'); break;
                        case 'r': out.push_back('\r'); break;
                        case 't': out.push_back('\t'); break;
                        case 'u': {
                            if (pos + 4 > input.size()) throw std::runtime_error("JSON: bad \\u");
                            unsigned code = (unsigned)std::stoul(input.substr(pos, 4), nullptr, 16);
                            pos += 4;
                            // Handle UTF-16 surrogate pair
                            if (code >= 0xD800 && code <= 0xDBFF && pos + 6 <= input.size() &&
                                input[pos] == '\\' && input[pos + 1] == 'u') {
                                unsigned low = (unsigned)std::stoul(input.substr(pos + 2, 4), nullptr, 16);
                                if (low >= 0xDC00 && low <= 0xDFFF) {
                                    pos += 6;
                                    code = 0x10000 + (((code - 0xD800) << 10) | (low - 0xDC00));
                                }
                            }
                            if (code < 0x80) {
                                out.push_back((char)code);
                            } else if (code < 0x800) {
                                out.push_back((char)(0xC0 | (code >> 6)));
                                out.push_back((char)(0x80 | (code & 0x3F)));
                            } else if (code < 0x10000) {
                                out.push_back((char)(0xE0 | (code >> 12)));
                                out.push_back((char)(0x80 | ((code >> 6) & 0x3F)));
                                out.push_back((char)(0x80 | (code & 0x3F)));
                            } else {
                                out.push_back((char)(0xF0 | (code >> 18)));
                                out.push_back((char)(0x80 | ((code >> 12) & 0x3F)));
                                out.push_back((char)(0x80 | ((code >> 6) & 0x3F)));
                                out.push_back((char)(0x80 | (code & 0x3F)));
                            }
                            break;
                        }
                        default: out.push_back(e); break;
                    }
                } else {
                    out.push_back(c);
                }
            }
            throw std::runtime_error("JSON: unterminated string");
        };

        auto parse_val = [&](auto& self) -> Json {
            skip();
            if (pos >= input.size()) throw std::runtime_error("JSON: unexpected EOF");
            char c = input[pos];
            if (c == '"') return Json(parse_str());
            if (c == '{') {
                ++pos;
                Object obj;
                skip();
                if (pos < input.size() && input[pos] == '}') { ++pos; return Json(std::move(obj)); }
                while (true) {
                    std::string k = parse_str();
                    skip();
                    if (pos >= input.size() || input[pos] != ':') throw std::runtime_error("JSON: expected ':'");
                    ++pos;
                    obj[std::move(k)] = self(self);
                    skip();
                    if (pos < input.size() && input[pos] == ',') { ++pos; continue; }
                    if (pos < input.size() && input[pos] == '}') { ++pos; break; }
                    throw std::runtime_error("JSON: expected ',' or '}'");
                }
                return Json(std::move(obj));
            }
            if (c == '[') {
                ++pos;
                Array arr;
                skip();
                if (pos < input.size() && input[pos] == ']') { ++pos; return Json(std::move(arr)); }
                while (true) {
                    arr.push_back(self(self));
                    skip();
                    if (pos < input.size() && input[pos] == ',') { ++pos; continue; }
                    if (pos < input.size() && input[pos] == ']') { ++pos; break; }
                    throw std::runtime_error("JSON: expected ',' or ']'");
                }
                return Json(std::move(arr));
            }
            if (input.compare(pos, 4, "true") == 0) { pos += 4; return Json(true); }
            if (input.compare(pos, 5, "false") == 0) { pos += 5; return Json(false); }
            if (input.compare(pos, 4, "null") == 0) { pos += 4; return Json(nullptr); }
            if (c == '-' || std::isdigit((unsigned char)c)) {
                size_t start = pos;
                bool is_float = false;
                if (input[pos] == '-') ++pos;
                while (pos < input.size() && std::isdigit((unsigned char)input[pos])) ++pos;
                if (pos < input.size() && input[pos] == '.') {
                    is_float = true; ++pos;
                    while (pos < input.size() && std::isdigit((unsigned char)input[pos])) ++pos;
                }
                if (pos < input.size() && (input[pos] == 'e' || input[pos] == 'E')) {
                    is_float = true; ++pos;
                    if (pos < input.size() && (input[pos] == '+' || input[pos] == '-')) ++pos;
                    while (pos < input.size() && std::isdigit((unsigned char)input[pos])) ++pos;
                }
                std::string num = input.substr(start, pos - start);
                if (is_float) return Json(std::stod(num));
                try {
                    return Json((int64_t)std::stoll(num));
                } catch (const std::out_of_range&) {
                    return Json(std::stod(num));
                }
            }
            throw std::runtime_error("JSON: unexpected token");
        };
        return parse_val(parse_val);
    }
};

} // namespace amperegen
