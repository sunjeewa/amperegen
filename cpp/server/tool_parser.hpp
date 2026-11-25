#pragma once
// Qwen3-Coder XML (<tool_call><function=name><parameter=k>v</parameter></function></tool_call>)
// streaming & non-streaming tool-call parser with JSON Schema parameter type conversion.
// Mirrors SGLang's `--tool-call-parser qwen3_coder` (Qwen3CoderDetector) plus JSON fallback.

#include "server/json_lite.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace amperegen {

struct ParsedToolCall {
    int index = 0;
    std::string id;
    std::string name;
    std::string arguments; // JSON-encoded object string
};

struct ToolCallDelta {
    int index = 0;
    bool is_header = false; // true -> emit id, type="function", function.name, function.arguments=""
    std::string id;
    std::string name;
    std::string arguments_delta;
};

struct StreamParseOutput {
    std::string normal_text;
    std::vector<ToolCallDelta> tool_deltas;
};

namespace detail {

inline std::string trim_sv(std::string_view sv) {
    size_t b = 0;
    while (b < sv.size() && std::isspace((unsigned char)sv[b])) ++b;
    size_t e = sv.size();
    while (e > b && std::isspace((unsigned char)sv[e - 1])) --e;
    return std::string(sv.substr(b, e - b));
}

inline std::string to_lower_str(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

inline bool is_all_ws(std::string_view sv) {
    for (char c : sv) {
        if (!std::isspace((unsigned char)c)) return false;
    }
    return true;
}

inline bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.substr(0, p.size()) == p;
}

inline std::string strip_param_newlines(std::string_view raw) {
    if (!raw.empty() && raw.front() == '\r' && raw.size() >= 2 && raw[1] == '\n') {
        raw.remove_prefix(2);
    } else if (!raw.empty() && raw.front() == '\n') {
        raw.remove_prefix(1);
    }
    if (raw.size() >= 2 && raw[raw.size() - 2] == '\r' && raw.back() == '\n') {
        raw.remove_suffix(2);
    } else if (!raw.empty() && raw.back() == '\n') {
        raw.remove_suffix(1);
    }
    return std::string(raw);
}

// Find parameter properties schema object for a given function name in OpenAI `tools` array.
inline const Json* find_param_properties(const Json* tools, const std::string& func_name) {
    if (!tools || !tools->is_array()) return nullptr;
    for (const auto& item : tools->as_array()) {
        if (!item.is_object()) continue;
        const Json* fn = nullptr;
        if (item.contains("function") && item.at("function").is_object()) {
            fn = &item.at("function");
        } else if (item.contains("name")) {
            fn = &item;
        }
        if (!fn || !fn->contains("name") || !fn->at("name").is_string()) continue;
        if (fn->at("name").as_string() != func_name) continue;
        if (!fn->contains("parameters") || !fn->at("parameters").is_object()) return nullptr;
        const Json& params = fn->at("parameters");
        if (params.contains("properties") && params.at("properties").is_object()) {
            return &params.at("properties");
        }
        return &params;
    }
    return nullptr;
}

// Infer primary JSON Schema type string from a parameter schema (handles type, type[], anyOf, oneOf).
inline std::string infer_schema_type(const Json& schema) {
    if (!schema.is_object()) return "string";
    if (schema.contains("type")) {
        const Json& t = schema.at("type");
        if (t.is_string()) return to_lower_str(t.as_string());
        if (t.is_array()) {
            for (const auto& el : t.as_array()) {
                if (el.is_string()) {
                    std::string lt = to_lower_str(el.as_string());
                    if (lt != "null") return lt;
                }
            }
        }
    }
    for (const char* key : {"anyOf", "oneOf"}) {
        if (schema.contains(key) && schema.at(key).is_array()) {
            for (const auto& sub : schema.at(key).as_array()) {
                if (!sub.is_object()) continue;
                std::string lt = infer_schema_type(sub);
                if (!lt.empty() && lt != "null") return lt;
            }
        }
    }
    return "string";
}

// Convert XML string parameter value according to its JSON Schema type (matching SGLang Qwen3CoderDetector).
inline Json convert_param_value(
    const std::string& raw_value,
    const std::string& param_name,
    const Json* param_props
) {
    if (to_lower_str(trim_sv(raw_value)) == "null") {
        return Json(nullptr);
    }
    if (!param_props || !param_props->is_object() || !param_props->contains(param_name)) {
        return Json(raw_value);
    }

    std::string ptype = infer_schema_type(param_props->at(param_name));
    if (ptype == "string" || ptype == "str" || ptype == "text" ||
        ptype == "varchar" || ptype == "char" || ptype == "enum") {
        return Json(raw_value);
    }

    std::string trimmed = trim_sv(raw_value);
    if (starts_with(ptype, "int") || starts_with(ptype, "uint") ||
        starts_with(ptype, "long") || starts_with(ptype, "short") ||
        starts_with(ptype, "unsigned")) {
        try {
            size_t idx = 0;
            int64_t v = std::stoll(trimmed, &idx, 10);
            if (idx == trimmed.size()) return Json(v);
        } catch (...) {}
        return Json(raw_value);
    }

    if (starts_with(ptype, "num") || starts_with(ptype, "float")) {
        try {
            size_t idx = 0;
            double v = std::stod(trimmed, &idx);
            if (idx == trimmed.size() && std::isfinite(v)) {
                bool has_dot_or_exp = (trimmed.find('.') != std::string::npos ||
                                       trimmed.find('e') != std::string::npos ||
                                       trimmed.find('E') != std::string::npos);
                if (!has_dot_or_exp && std::floor(v) == v &&
                    v >= -9007199254740992.0 && v <= 9007199254740992.0) {
                    return Json((int64_t)v);
                }
                return Json(v);
            }
        } catch (...) {}
        return Json(raw_value);
    }

    if (ptype == "boolean" || ptype == "bool" || ptype == "binary") {
        std::string lo = to_lower_str(trimmed);
        return Json(lo == "true");
    }

    if (ptype == "object" || ptype == "array" || ptype == "arr" ||
        starts_with(ptype, "dict") || starts_with(ptype, "list")) {
        try {
            return Json::parse(trimmed);
        } catch (...) {}
        return Json(raw_value);
    }

    // Fallback: try JSON parse for numbers/booleans/objects/arrays
    try {
        return Json::parse(trimmed);
    } catch (...) {}
    return Json(raw_value);
}

inline std::string make_call_id(const std::string& prefix, int idx) {
    return prefix + std::to_string(idx);
}

} // namespace detail

class Qwen3CoderStreamParser {
public:
    static constexpr std::string_view TC_START = "<tool_call>";
    static constexpr std::string_view TC_END   = "</tool_call>";
    static constexpr std::string_view FN_PREF  = "<function=";
    static constexpr std::string_view FN_END   = "</function>";
    static constexpr std::string_view P_PREF   = "<parameter=";
    static constexpr std::string_view P_END    = "</parameter>";

    Qwen3CoderStreamParser(const Json* tools, std::string id_prefix)
        : tools_(tools), id_prefix_(std::move(id_prefix)) {}

    bool has_tool_calls() const { return any_tool_calls_; }

    StreamParseOutput feed(std::string_view delta) {
        if (!delta.empty()) buf_.append(delta);
        return parse_available(/*is_final=*/false);
    }

    StreamParseOutput finish() {
        return parse_available(/*is_final=*/true);
    }

private:
    const Json* tools_ = nullptr;
    std::string id_prefix_;

    std::string buf_;
    size_t pos_ = 0;
    bool inside_tool_call_ = false;
    bool any_tool_calls_ = false;
    int current_tool_idx_ = -1;
    int current_param_count_ = 0;
    bool json_started_ = false;
    std::string current_func_name_;
    std::string pending_ws_;

    void emit_normal(std::string_view text, std::string& out) {
        if (text.empty()) return;
        if (any_tool_calls_) {
            if (detail::is_all_ws(text)) {
                pending_ws_.append(text);
                return;
            }
            if (!pending_ws_.empty()) {
                out.append(pending_ws_);
                pending_ws_.clear();
            }
        }
        out.append(text);
    }

    void close_current_function(std::vector<ToolCallDelta>& deltas) {
        if (current_func_name_.empty()) return;
        std::string frag = json_started_ ? "}" : "{}";
        json_started_ = true;
        deltas.push_back(ToolCallDelta{current_tool_idx_, false, "", "", std::move(frag)});
        current_func_name_.clear();
    }

    StreamParseOutput parse_available(bool is_final) {
        StreamParseOutput out;

        while (pos_ < buf_.size()) {
            std::string_view sv = std::string_view(buf_).substr(pos_);

            // 1. <tool_call>
            if (detail::starts_with(sv, TC_START)) {
                pos_ += TC_START.size();
                inside_tool_call_ = true;
                pending_ws_.clear();
                continue;
            }

            // 2. <function=NAME>
            if (detail::starts_with(sv, FN_PREF)) {
                size_t gt = sv.find('>');
                if (gt != std::string_view::npos) {
                    std::string func_name = detail::trim_sv(sv.substr(FN_PREF.size(), gt - FN_PREF.size()));
                    // Close any unclosed previous function defensively
                    if (!current_func_name_.empty()) close_current_function(out.tool_deltas);

                    inside_tool_call_ = true;
                    any_tool_calls_ = true;
                    pending_ws_.clear();
                    ++current_tool_idx_;
                    current_param_count_ = 0;
                    json_started_ = false;
                    current_func_name_ = func_name;

                    out.tool_deltas.push_back(ToolCallDelta{
                        current_tool_idx_,
                        /*is_header=*/true,
                        detail::make_call_id(id_prefix_, current_tool_idx_),
                        func_name,
                        ""
                    });
                    pos_ += gt + 1;
                    continue;
                }
                if (!is_final) break;
            }

            // 3. <parameter=NAME>VALUE(</parameter>|<parameter=|</function>)
            if (detail::starts_with(sv, P_PREF) && !current_func_name_.empty()) {
                size_t gt = sv.find('>');
                if (gt != std::string_view::npos) {
                    std::string_view rest = sv.substr(gt + 1);
                    size_t c_end_p = rest.find(P_END);
                    size_t c_next_p = rest.find(P_PREF);
                    size_t c_end_f = rest.find(FN_END);
                    size_t c_end_tc = rest.find(TC_END);

                    size_t end_pos = std::string_view::npos;
                    size_t end_tok_len = 0;
                    auto consider = [&](size_t p, size_t len) {
                        if (p != std::string_view::npos && p < end_pos) {
                            end_pos = p;
                            end_tok_len = len;
                        }
                    };
                    consider(c_end_p, P_END.size());
                    consider(c_next_p, 0);
                    consider(c_end_f, 0);
                    consider(c_end_tc, 0);

                    if (end_pos != std::string_view::npos || is_final) {
                        if (end_pos == std::string_view::npos) {
                            end_pos = rest.size();
                            end_tok_len = 0;
                        }
                        std::string param_name = detail::trim_sv(sv.substr(P_PREF.size(), gt - P_PREF.size()));
                        std::string raw_val = detail::strip_param_newlines(rest.substr(0, end_pos));

                        const Json* props = detail::find_param_properties(tools_, current_func_name_);
                        Json conv = detail::convert_param_value(raw_val, param_name, props);

                        std::string frag;
                        if (!json_started_) {
                            frag.push_back('{');
                            json_started_ = true;
                        }
                        if (current_param_count_ > 0) {
                            frag += ", ";
                        }
                        frag += Json::escape(param_name);
                        frag += ": ";
                        frag += conv.dump();
                        ++current_param_count_;

                        out.tool_deltas.push_back(ToolCallDelta{
                            current_tool_idx_, false, "", "", std::move(frag)
                        });
                        pos_ += (gt + 1) + end_pos + end_tok_len;
                        continue;
                    }
                }
                if (!is_final) break;
            }

            // 4. </function>
            if (detail::starts_with(sv, FN_END)) {
                close_current_function(out.tool_deltas);
                pos_ += FN_END.size();
                continue;
            }

            // 5. </tool_call>
            if (detail::starts_with(sv, TC_END)) {
                close_current_function(out.tool_deltas);
                pos_ += TC_END.size();
                inside_tool_call_ = false;
                continue;
            }

            // 6. Fallback: classic Qwen JSON format inside <tool_call>{"name":..., "arguments":...}</tool_call>
            if (inside_tool_call_ && current_func_name_.empty()) {
                size_t first_non_ws = 0;
                while (first_non_ws < sv.size() && std::isspace((unsigned char)sv[first_non_ws])) ++first_non_ws;
                if (first_non_ws < sv.size() && sv[first_non_ws] == '{') {
                    size_t tc_end = sv.find(TC_END, first_non_ws);
                    if (tc_end == std::string_view::npos && !is_final) {
                        break; // wait for closing </tool_call>
                    }
                    size_t json_end = (tc_end == std::string_view::npos) ? sv.size() : tc_end;
                    std::string body = detail::trim_sv(sv.substr(first_non_ws, json_end - first_non_ws));
                    try {
                        Json obj = Json::parse(body);
                        if (obj.is_object() && obj.contains("name") && obj.at("name").is_string()) {
                            std::string func_name = obj.at("name").as_string();
                            std::string args_str = "{}";
                            if (obj.contains("arguments")) {
                                const Json& a = obj.at("arguments");
                                args_str = a.is_string() ? a.as_string() : a.dump();
                            }
                            any_tool_calls_ = true;
                            pending_ws_.clear();
                            ++current_tool_idx_;
                            out.tool_deltas.push_back(ToolCallDelta{
                                current_tool_idx_,
                                true,
                                detail::make_call_id(id_prefix_, current_tool_idx_),
                                func_name,
                                ""
                            });
                            out.tool_deltas.push_back(ToolCallDelta{
                                current_tool_idx_, false, "", "", std::move(args_str)
                            });
                            pos_ += json_end + (tc_end != std::string_view::npos ? TC_END.size() : 0);
                            inside_tool_call_ = false;
                            continue;
                        }
                    } catch (...) {
                        // Fall through to normal handling if malformed JSON
                    }
                }
            }

            // 7. Plain text / whitespace / potential partial sentinel tag
            size_t next_lt = sv.find('<');
            if (next_lt == std::string_view::npos) {
                if (!inside_tool_call_) emit_normal(sv, out.normal_text);
                pos_ += sv.size();
                continue;
            }
            if (next_lt > 0) {
                if (!inside_tool_call_) emit_normal(sv.substr(0, next_lt), out.normal_text);
                pos_ += next_lt;
                continue;
            }

            // next_lt == 0: starts with '<', check if prefix of any known tag
            constexpr std::string_view TAGS[] = {TC_START, TC_END, FN_PREF, FN_END, P_PREF, P_END};
            bool potential_tag = false;
            for (auto tag : TAGS) {
                if (sv.size() < tag.size() && detail::starts_with(tag, sv)) {
                    potential_tag = true;
                    break;
                }
            }
            if (potential_tag && !is_final) {
                break; // wait for more characters
            }
            // Literal '<'
            if (!inside_tool_call_) emit_normal("<", out.normal_text);
            pos_ += 1;
        }

        if (is_final) {
            if (!current_func_name_.empty()) {
                close_current_function(out.tool_deltas);
            }
            pending_ws_.clear();
            buf_.clear();
            pos_ = 0;
        } else if (pos_ > 0) {
            buf_.erase(0, pos_);
            pos_ = 0;
        }

        return out;
    }
};

struct NonStreamParseResult {
    std::string normal_text;
    std::vector<ParsedToolCall> tool_calls;
};

// One-shot parser built on top of the exact same streaming state machine for 100% parity.
inline NonStreamParseResult parse_qwen3_coder_non_stream(
    const std::string& content,
    const Json* tools,
    const std::string& id_prefix
) {
    NonStreamParseResult res;
    Qwen3CoderStreamParser parser(tools, id_prefix);
    auto o1 = parser.feed(content);
    auto o2 = parser.finish();
    res.normal_text = o1.normal_text + o2.normal_text;

    auto apply_deltas = [&](const std::vector<ToolCallDelta>& deltas) {
        for (const auto& d : deltas) {
            if (d.is_header) {
                if ((int)res.tool_calls.size() <= d.index) res.tool_calls.resize(d.index + 1);
                res.tool_calls[d.index].index = d.index;
                res.tool_calls[d.index].id = d.id;
                res.tool_calls[d.index].name = d.name;
                res.tool_calls[d.index].arguments = d.arguments_delta;
            } else if (d.index >= 0 && d.index < (int)res.tool_calls.size()) {
                res.tool_calls[d.index].arguments += d.arguments_delta;
            }
        }
    };
    apply_deltas(o1.tool_deltas);
    apply_deltas(o2.tool_deltas);

    if (!res.tool_calls.empty()) {
        // Trim trailing whitespace before <tool_call> in non-stream content
        while (!res.normal_text.empty() && std::isspace((unsigned char)res.normal_text.back())) {
            res.normal_text.pop_back();
        }
    }
    return res;
}

} // namespace amperegen
