#include "server/tokenizer.hpp"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <utf8proc.h>

#include <minja/chat-template.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace amperegen {

namespace {

using OrderedJson = nlohmann::ordered_json;

// Build forward GPT-2 / HuggingFace ByteLevel table: raw byte (0..255) -> UTF-8 string (1 or 2 bytes).
std::array<std::string, 256> make_byte_to_utf8_table() {
    std::array<std::string, 256> tbl;
    std::array<bool, 256> direct{};
    for (int b = 33; b <= 126; ++b) direct[static_cast<size_t>(b)] = true;
    for (int b = 161; b <= 172; ++b) direct[static_cast<size_t>(b)] = true;
    for (int b = 174; b <= 255; ++b) direct[static_cast<size_t>(b)] = true;

    auto encode_cp = [](uint32_t cp) -> std::string {
        std::string s;
        if (cp < 0x80) {
            s.push_back(static_cast<char>(cp));
        } else {
            s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        return s;
    };

    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = direct[static_cast<size_t>(b)]
            ? static_cast<uint32_t>(b)
            : static_cast<uint32_t>(256 + extra++);
        tbl[static_cast<size_t>(b)] = encode_cp(cp);
    }
    return tbl;
}

bool read_file_all(const std::string& path, std::string& out) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) return false;
    std::streamsize sz = ifs.tellg();
    if (sz < 0) return false;
    ifs.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(sz));
    if (sz > 0 && !ifs.read(out.data(), sz)) return false;
    return true;
}

void set_err(std::string* err_out, const std::string& msg) {
    if (err_out) *err_out = msg;
}

bool is_pure_ascii(std::string_view sv) {
    for (unsigned char c : sv) {
        if (c >= 0x80) return false;
    }
    return true;
}

std::string trim_ws(std::string_view sv) {
    size_t b = 0;
    while (b < sv.size() && std::isspace(static_cast<unsigned char>(sv[b]))) ++b;
    size_t e = sv.size();
    while (e > b && std::isspace(static_cast<unsigned char>(sv[e - 1]))) --e;
    return std::string(sv.substr(b, e - b));
}

std::string rtrim_ws(std::string_view sv) {
    size_t e = sv.size();
    while (e > 0 && std::isspace(static_cast<unsigned char>(sv[e - 1]))) --e;
    return std::string(sv.substr(0, e));
}

bool is_word_byte(unsigned char c) {
    return std::isalnum(c) != 0 || c == '_' || c >= 0x80;
}

// Convert amperegen::Json into nlohmann::ordered_json directly (zero serialization round-trip).
OrderedJson to_ordered_json(const Json& j) {
    if (j.is_null()) return nullptr;
    if (j.is_bool()) return j.as_bool();
    if (j.is_int()) return j.as_int();
    if (j.is_number()) return j.as_double();
    if (j.is_string()) return j.as_string();
    if (j.is_array()) {
        OrderedJson arr = OrderedJson::array();
        for (const auto& item : j.as_array()) {
            arr.push_back(to_ordered_json(item));
        }
        return arr;
    }
    OrderedJson obj = OrderedJson::object();
    for (const auto& [k, v] : j.as_object()) {
        obj[k] = to_ordered_json(v);
    }
    return obj;
}

// Extract string token from tokenizer_config.json field (may be string, null, or AddedToken object).
std::string extract_config_token(const nlohmann::json& cfg, const char* key) {
    if (!cfg.contains(key) || cfg[key].is_null()) return "";
    const auto& v = cfg[key];
    if (v.is_string()) return v.get<std::string>();
    if (v.is_object() && v.contains("content") && v["content"].is_string()) {
        return v["content"].get<std::string>();
    }
    return "";
}

// Normalize OpenAI Chat Completions messages for Jinja chat templates:
//  - Converts `developer` role -> `system` and merges multiple system messages at index 0.
//  - Copies `reasoning` -> `reasoning_content` fallback on assistant messages.
//  - Parses assistant `tool_calls[].function.arguments` JSON strings into objects so
//    Jinja `tool_call.arguments|items` iterates key/value pairs cleanly.
//  - Stringifies unexpected non-string/non-array content values.
OrderedJson normalize_openai_messages(const OrderedJson& raw_messages) {
    std::vector<std::string> system_parts;
    OrderedJson other_msgs = OrderedJson::array();

    auto parse_args_str = [](const std::string& raw) -> OrderedJson {
        std::string s = trim_ws(raw);
        if (s.empty()) return OrderedJson::object();
        try {
            OrderedJson parsed = OrderedJson::parse(s);
            if (parsed.is_object()) return parsed;
            OrderedJson wrap = OrderedJson::object();
            wrap["value"] = std::move(parsed);
            return wrap;
        } catch (...) {
            OrderedJson wrap = OrderedJson::object();
            wrap["raw"] = s;
            return wrap;
        }
    };

    if (!raw_messages.is_array()) return OrderedJson::array();
    for (const auto& orig : raw_messages) {
        if (!orig.is_object()) continue;
        OrderedJson msg = orig;
        std::string role = (msg.contains("role") && msg["role"].is_string())
            ? msg["role"].get<std::string>() : "";

        if (msg.contains("content") && !msg["content"].is_null() &&
            !msg["content"].is_string() && !msg["content"].is_array()) {
            msg["content"] = msg["content"].dump();
        }

        if (role == "system" || role == "developer") {
            if (msg.contains("content")) {
                const auto& c = msg["content"];
                if (c.is_string()) {
                    std::string t = trim_ws(c.get<std::string>());
                    if (!t.empty()) system_parts.push_back(std::move(t));
                } else if (c.is_array()) {
                    std::string joined;
                    for (const auto& p : c) {
                        if (p.is_object() && p.contains("text") && p["text"].is_string()) {
                            const std::string& txt = p["text"].get<std::string>();
                            if (!txt.empty()) {
                                if (!joined.empty()) joined.push_back('\n');
                                joined += txt;
                            }
                        }
                    }
                    std::string t = trim_ws(joined);
                    if (!t.empty()) system_parts.push_back(std::move(t));
                }
            }
            continue;
        }

        if (role == "assistant") {
            bool has_rc = msg.contains("reasoning_content") &&
                          msg["reasoning_content"].is_string() &&
                          !msg["reasoning_content"].get<std::string>().empty();
            if (!has_rc && msg.contains("reasoning") && msg["reasoning"].is_string()) {
                msg["reasoning_content"] = msg["reasoning"];
            }

            if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
                OrderedJson norm_tcs = OrderedJson::array();
                for (const auto& tc : msg["tool_calls"]) {
                    if (!tc.is_object()) continue;
                    OrderedJson tc_copy = tc;
                    if (tc_copy.contains("function") && tc_copy["function"].is_object()) {
                        auto& fn = tc_copy["function"];
                        if (fn.contains("arguments")) {
                            if (fn["arguments"].is_string()) {
                                fn["arguments"] = parse_args_str(fn["arguments"].get<std::string>());
                            } else if (fn["arguments"].is_null()) {
                                fn["arguments"] = OrderedJson::object();
                            }
                        }
                    } else if (tc_copy.contains("arguments") && tc_copy["arguments"].is_string()) {
                        tc_copy["arguments"] = parse_args_str(tc_copy["arguments"].get<std::string>());
                    }
                    norm_tcs.push_back(std::move(tc_copy));
                }
                msg["tool_calls"] = std::move(norm_tcs);
            }
        }

        other_msgs.push_back(std::move(msg));
    }

    OrderedJson out = OrderedJson::array();
    if (!system_parts.empty()) {
        std::string merged;
        for (size_t i = 0; i < system_parts.size(); ++i) {
            if (i > 0) merged += "\n\n";
            merged += system_parts[i];
        }
        OrderedJson sys = OrderedJson::object();
        sys["role"] = "system";
        sys["content"] = std::move(merged);
        out.push_back(std::move(sys));
    }
    for (auto& m : other_msgs) {
        out.push_back(std::move(m));
    }
    return out;
}

// ============================================================================
// Normalizer step (utf8proc)
// ============================================================================
enum class NormKind { NFC, NFD, NFKC, NFKD, Lowercase, StripAccents };

std::string apply_utf8proc_norm(std::string_view input, utf8proc_option_t opts) {
    if (input.empty()) return {};
    if ((opts & (UTF8PROC_CASEFOLD | UTF8PROC_STRIPMARK)) == 0 && is_pure_ascii(input)) {
        return std::string(input);
    }
    utf8proc_uint8_t* dst = nullptr;
    utf8proc_ssize_t out_len = utf8proc_map(
        reinterpret_cast<const utf8proc_uint8_t*>(input.data()),
        static_cast<utf8proc_ssize_t>(input.size()),
        &dst,
        opts);
    if (out_len < 0 || dst == nullptr) {
        if (dst) std::free(dst);
        // Fallback: preserve original bytes if input contains invalid UTF-8
        return std::string(input);
    }
    std::string result(reinterpret_cast<const char*>(dst), static_cast<size_t>(out_len));
    std::free(dst);
    return result;
}

// ============================================================================
// Pre-tokenizer step (PCRE2 Split / ByteLevel)
// ============================================================================
enum class SplitBehavior {
    Isolated,
    Removed,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous
};

struct RegexWrapper {
    pcre2_code* code = nullptr;
    bool jit_ok = false;

    RegexWrapper() = default;
    ~RegexWrapper() { reset(); }
    RegexWrapper(RegexWrapper&& o) noexcept : code(o.code), jit_ok(o.jit_ok) {
        o.code = nullptr;
        o.jit_ok = false;
    }
    RegexWrapper& operator=(RegexWrapper&& o) noexcept {
        if (this != &o) {
            reset();
            code = o.code;
            jit_ok = o.jit_ok;
            o.code = nullptr;
            o.jit_ok = false;
        }
        return *this;
    }
    RegexWrapper(const RegexWrapper&) = delete;
    RegexWrapper& operator=(const RegexWrapper&) = delete;

    void reset() {
        if (code) {
            pcre2_code_free(code);
            code = nullptr;
        }
        jit_ok = false;
    }

    bool compile(const std::string& pattern, std::string& err_msg) {
        reset();
        int err_code = 0;
        PCRE2_SIZE err_off = 0;
        code = pcre2_compile(
            reinterpret_cast<PCRE2_SPTR>(pattern.data()),
            static_cast<PCRE2_SIZE>(pattern.size()),
            PCRE2_UTF | PCRE2_UCP,
            &err_code, &err_off, nullptr);
        if (!code) {
            PCRE2_UCHAR buf[256];
            pcre2_get_error_message(err_code, buf, sizeof(buf));
            err_msg = "PCRE2 compile failed at offset " + std::to_string(err_off) +
                      ": " + reinterpret_cast<const char*>(buf);
            return false;
        }
        jit_ok = (pcre2_jit_compile(code, PCRE2_JIT_COMPLETE) == 0);
        return true;
    }
};

struct PreTokStep {
    enum class Kind { SplitRegex, SplitString, ByteLevel };
    Kind kind = Kind::SplitRegex;
    RegexWrapper regex;
    std::string literal;
    SplitBehavior behavior = SplitBehavior::Isolated;
    bool invert = false;
    bool add_prefix_space = false;
};

static constexpr const char* kGpt2DefaultRegex =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

// Advance UTF-8 byte offset by one Unicode codepoint.
size_t next_utf8_offset(std::string_view s, size_t off) {
    if (off >= s.size()) return s.size();
    ++off;
    while (off < s.size() && (static_cast<unsigned char>(s[off]) & 0xC0) == 0x80) ++off;
    return off;
}

void split_piece_with_spans(
    std::string_view piece,
    const std::vector<std::pair<size_t, size_t>>& raw_matches,
    SplitBehavior behavior,
    bool invert,
    std::vector<std::string>& out
) {
    // Build tagged spans: (begin, end, is_match)
    struct Span {
        size_t b, e;
        bool matched;
    };
    std::vector<Span> spans;
    spans.reserve(raw_matches.size() * 2 + 1);
    size_t cursor = 0;
    for (const auto& [mb, me] : raw_matches) {
        if (mb > cursor) {
            spans.push_back(Span{cursor, mb, false});
        }
        if (me > mb) {
            spans.push_back(Span{mb, me, true});
        }
        cursor = std::max(cursor, me);
    }
    if (cursor < piece.size()) {
        spans.push_back(Span{cursor, piece.size(), false});
    }
    if (invert) {
        for (auto& sp : spans) sp.matched = !sp.matched;
    }

    switch (behavior) {
        case SplitBehavior::Isolated:
            for (const auto& sp : spans) {
                if (sp.e > sp.b) out.emplace_back(piece.substr(sp.b, sp.e - sp.b));
            }
            break;
        case SplitBehavior::Removed:
            for (const auto& sp : spans) {
                if (!sp.matched && sp.e > sp.b) out.emplace_back(piece.substr(sp.b, sp.e - sp.b));
            }
            break;
        case SplitBehavior::MergedWithPrevious:
            for (const auto& sp : spans) {
                if (sp.e <= sp.b) continue;
                if (sp.matched && !out.empty()) {
                    out.back().append(piece.data() + sp.b, sp.e - sp.b);
                } else {
                    out.emplace_back(piece.substr(sp.b, sp.e - sp.b));
                }
            }
            break;
        case SplitBehavior::MergedWithNext: {
            std::string pending;
            for (const auto& sp : spans) {
                if (sp.e <= sp.b) continue;
                if (sp.matched) {
                    pending.append(piece.data() + sp.b, sp.e - sp.b);
                } else {
                    if (!pending.empty()) {
                        pending.append(piece.data() + sp.b, sp.e - sp.b);
                        out.push_back(std::move(pending));
                        pending.clear();
                    } else {
                        out.emplace_back(piece.substr(sp.b, sp.e - sp.b));
                    }
                }
            }
            if (!pending.empty()) out.push_back(std::move(pending));
            break;
        }
        case SplitBehavior::Contiguous: {
            bool prev_matched = false;
            bool has_prev = false;
            for (const auto& sp : spans) {
                if (sp.e <= sp.b) continue;
                if (has_prev && sp.matched && prev_matched) {
                    out.back().append(piece.data() + sp.b, sp.e - sp.b);
                } else {
                    out.emplace_back(piece.substr(sp.b, sp.e - sp.b));
                }
                prev_matched = sp.matched;
                has_prev = true;
            }
            break;
        }
    }
}

void run_pcre2_split(
    const PreTokStep& step,
    std::string_view piece,
    std::vector<std::string>& out
) {
    if (piece.empty()) return;
    pcre2_match_data* md = pcre2_match_data_create_from_pattern(step.regex.code, nullptr);
    if (!md) throw std::runtime_error("pcre2_match_data_create_from_pattern failed");

    std::vector<std::pair<size_t, size_t>> matches;
    const auto* subj = reinterpret_cast<PCRE2_SPTR>(piece.data());
    const PCRE2_SIZE subj_len = static_cast<PCRE2_SIZE>(piece.size());
    PCRE2_SIZE offset = 0;

    while (offset <= subj_len) {
        int rc = step.regex.jit_ok
            ? pcre2_jit_match(step.regex.code, subj, subj_len, offset, 0, md, nullptr)
            : pcre2_match(step.regex.code, subj, subj_len, offset, 0, md, nullptr);
        if (rc < 0) break;

        PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
        size_t mb = static_cast<size_t>(ov[0]);
        size_t me = static_cast<size_t>(ov[1]);
        if (me > mb) {
            matches.emplace_back(mb, me);
            offset = me;
        } else {
            // Empty match: try non-empty match at the same offset (standard PCRE2 pattern),
            // then advance by one UTF-8 codepoint to guarantee progress.
            if (offset < subj_len) {
                int rc2 = pcre2_match(
                    step.regex.code, subj, subj_len, offset,
                    PCRE2_NOTEMPTY_ATSTART | PCRE2_ANCHORED, md, nullptr);
                if (rc2 >= 0) {
                    PCRE2_SIZE* ov2 = pcre2_get_ovector_pointer(md);
                    if (ov2[1] > ov2[0]) {
                        matches.emplace_back(static_cast<size_t>(ov2[0]), static_cast<size_t>(ov2[1]));
                        offset = ov2[1];
                        continue;
                    }
                }
            }
            size_t next = next_utf8_offset(piece, static_cast<size_t>(offset));
            if (next == offset) break;
            offset = next;
        }
    }

    pcre2_match_data_free(md);
    split_piece_with_spans(piece, matches, step.behavior, step.invert, out);
}

void run_literal_split(
    const PreTokStep& step,
    std::string_view piece,
    std::vector<std::string>& out
) {
    if (piece.empty()) return;
    if (step.literal.empty()) {
        out.emplace_back(piece);
        return;
    }
    std::vector<std::pair<size_t, size_t>> matches;
    size_t pos = 0;
    while (pos < piece.size()) {
        size_t f = piece.find(step.literal, pos);
        if (f == std::string_view::npos) break;
        matches.emplace_back(f, f + step.literal.size());
        pos = f + step.literal.size();
    }
    split_piece_with_spans(piece, matches, step.behavior, step.invert, out);
}

// ============================================================================
// Added token descriptor + LeftmostLongest matcher
// ============================================================================
struct AddedTokenSpec {
    int32_t id = -1;
    std::string content;
    bool special = false;
    bool single_word = false;
    bool lstrip = false;
    bool rstrip = false;
    bool normalized = false;
};

struct AddedMatcher {
    std::array<uint8_t, 256> first_byte{};
    std::array<std::vector<const AddedTokenSpec*>, 256> by_first_byte;
    bool empty = true;

    void build(const std::vector<AddedTokenSpec>& specs, bool want_normalized) {
        first_byte.fill(0);
        for (auto& v : by_first_byte) v.clear();
        empty = true;
        for (const auto& sp : specs) {
            if (sp.content.empty() || sp.id < 0) continue;
            if (sp.normalized != want_normalized) continue;
            unsigned char fb = static_cast<unsigned char>(sp.content[0]);
            first_byte[fb] = 1;
            by_first_byte[fb].push_back(&sp);
            empty = false;
        }
        for (auto& v : by_first_byte) {
            // Longest-first order implements exact LeftmostLongest matching.
            std::sort(v.begin(), v.end(), [](const AddedTokenSpec* a, const AddedTokenSpec* b) {
                return a->content.size() > b->content.size();
            });
        }
    }

    struct Segment {
        size_t begin = 0;
        size_t end = 0;
        int32_t added_id = -1; // >= 0 iff this segment is a matched added token
    };

    void split(std::string_view text, std::vector<Segment>& out) const {
        out.clear();
        if (text.empty()) return;
        if (empty) {
            out.push_back(Segment{0, text.size(), -1});
            return;
        }

        const size_t n = text.size();
        size_t seg_start = 0;
        size_t pos = 0;
        while (pos < n) {
            unsigned char fb = static_cast<unsigned char>(text[pos]);
            if (!first_byte[fb]) {
                ++pos;
                continue;
            }

            const AddedTokenSpec* hit = nullptr;
            for (const auto* cand : by_first_byte[fb]) {
                size_t clen = cand->content.size();
                if (pos + clen > n) continue;
                if (std::memcmp(text.data() + pos, cand->content.data(), clen) != 0) continue;
                if (cand->single_word) {
                    bool left_word = (pos > 0) && is_word_byte(static_cast<unsigned char>(text[pos - 1]));
                    bool right_word = (pos + clen < n) && is_word_byte(static_cast<unsigned char>(text[pos + clen]));
                    if (left_word || right_word) continue;
                }
                hit = cand;
                break;
            }

            if (!hit) {
                ++pos;
                continue;
            }

            size_t match_b = pos;
            size_t match_e = pos + hit->content.size();
            if (hit->lstrip) {
                while (match_b > seg_start && std::isspace(static_cast<unsigned char>(text[match_b - 1]))) {
                    --match_b;
                }
            }
            if (hit->rstrip) {
                while (match_e < n && std::isspace(static_cast<unsigned char>(text[match_e]))) {
                    ++match_e;
                }
            }

            if (match_b > seg_start) {
                out.push_back(Segment{seg_start, match_b, -1});
            }
            out.push_back(Segment{match_b, match_e, hit->id});
            pos = match_e;
            seg_start = match_e;
        }
        if (seg_start < n) {
            out.push_back(Segment{seg_start, n, -1});
        }
    }
};

// ============================================================================
// Integer-ID ByteLevel BPE Merge Table
// ============================================================================
struct MergeEntry {
    int32_t rank = 0;
    int32_t merged_id = -1;
};

inline uint64_t pack_pair(int32_t a, int32_t b) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(b));
}

SplitBehavior parse_split_behavior(const std::string& s) {
    if (s == "Isolated") return SplitBehavior::Isolated;
    if (s == "Removed") return SplitBehavior::Removed;
    if (s == "MergedWithPrevious") return SplitBehavior::MergedWithPrevious;
    if (s == "MergedWithNext") return SplitBehavior::MergedWithNext;
    if (s == "Contiguous") return SplitBehavior::Contiguous;
    return SplitBehavior::Isolated;
}

bool parse_normalizer_json(const nlohmann::json& nj, std::vector<NormKind>& out, std::string& err) {
    if (nj.is_null()) return true;
    if (!nj.is_object() || !nj.contains("type") || !nj["type"].is_string()) {
        err = "tokenizer.json: malformed normalizer";
        return false;
    }
    const std::string t = nj["type"].get<std::string>();
    if (t == "NFC") { out.push_back(NormKind::NFC); return true; }
    if (t == "NFD") { out.push_back(NormKind::NFD); return true; }
    if (t == "NFKC") { out.push_back(NormKind::NFKC); return true; }
    if (t == "NFKD") { out.push_back(NormKind::NFKD); return true; }
    if (t == "Lowercase") { out.push_back(NormKind::Lowercase); return true; }
    if (t == "StripAccents") { out.push_back(NormKind::StripAccents); return true; }
    if (t == "Sequence") {
        if (nj.contains("normalizers") && nj["normalizers"].is_array()) {
            for (const auto& sub : nj["normalizers"]) {
                if (!parse_normalizer_json(sub, out, err)) return false;
            }
        }
        return true;
    }
    err = "tokenizer.json: unsupported normalizer type '" + t + "'";
    return false;
}

bool parse_pretokenizer_json(const nlohmann::json& pj, std::vector<PreTokStep>& out, std::string& err) {
    if (pj.is_null()) return true;
    if (!pj.is_object() || !pj.contains("type") || !pj["type"].is_string()) {
        err = "tokenizer.json: malformed pre_tokenizer";
        return false;
    }
    const std::string t = pj["type"].get<std::string>();
    if (t == "Sequence") {
        if (pj.contains("pretokenizers") && pj["pretokenizers"].is_array()) {
            for (const auto& sub : pj["pretokenizers"]) {
                if (!parse_pretokenizer_json(sub, out, err)) return false;
            }
        }
        return true;
    }
    if (t == "Split") {
        if (!pj.contains("pattern") || !pj["pattern"].is_object()) {
            err = "tokenizer.json: Split pre_tokenizer missing pattern";
            return false;
        }
        const auto& pat = pj["pattern"];
        PreTokStep step;
        step.behavior = pj.contains("behavior") && pj["behavior"].is_string()
            ? parse_split_behavior(pj["behavior"].get<std::string>())
            : SplitBehavior::Isolated;
        step.invert = pj.contains("invert") && pj["invert"].is_boolean()
            ? pj["invert"].get<bool>() : false;
        if (pat.contains("Regex") && pat["Regex"].is_string()) {
            step.kind = PreTokStep::Kind::SplitRegex;
            if (!step.regex.compile(pat["Regex"].get<std::string>(), err)) return false;
            out.push_back(std::move(step));
            return true;
        }
        if (pat.contains("String") && pat["String"].is_string()) {
            step.kind = PreTokStep::Kind::SplitString;
            step.literal = pat["String"].get<std::string>();
            out.push_back(std::move(step));
            return true;
        }
        err = "tokenizer.json: unsupported Split pattern";
        return false;
    }
    if (t == "ByteLevel") {
        bool add_prefix = pj.contains("add_prefix_space") && pj["add_prefix_space"].is_boolean()
            ? pj["add_prefix_space"].get<bool>() : false;
        bool use_regex = pj.contains("use_regex") && pj["use_regex"].is_boolean()
            ? pj["use_regex"].get<bool>() : true;
        if (add_prefix) {
            PreTokStep bl;
            bl.kind = PreTokStep::Kind::ByteLevel;
            bl.add_prefix_space = true;
            out.push_back(std::move(bl));
        }
        if (use_regex) {
            PreTokStep re;
            re.kind = PreTokStep::Kind::SplitRegex;
            re.behavior = SplitBehavior::Isolated;
            re.invert = false;
            if (!re.regex.compile(kGpt2DefaultRegex, err)) return false;
            out.push_back(std::move(re));
        }
        return true;
    }
    if (t == "Digits") {
        bool indiv = pj.contains("individual_digits") && pj["individual_digits"].is_boolean()
            ? pj["individual_digits"].get<bool>() : false;
        PreTokStep re;
        re.kind = PreTokStep::Kind::SplitRegex;
        re.behavior = SplitBehavior::Isolated;
        re.invert = false;
        if (!re.regex.compile(indiv ? R"([0-9])" : R"([0-9]+)", err)) return false;
        out.push_back(std::move(re));
        return true;
    }
    err = "tokenizer.json: unsupported pre_tokenizer type '" + t + "'";
    return false;
}

} // namespace

struct NativeTokenizer::Impl {
    BpeDetokenizer detok;
    std::vector<NormKind> normalizers;
    std::vector<PreTokStep> pretokenizers;

    std::vector<AddedTokenSpec> added_specs;
    AddedMatcher raw_added_matcher;
    AddedMatcher norm_added_matcher;

    std::array<int32_t, 256> byte_to_id{};
    std::unordered_map<uint64_t, MergeEntry> merges;
    std::unordered_map<std::string, int32_t> piece_to_id; // populated only if ignore_merges == true
    bool ignore_merges = false;

    std::string bos_token;
    std::string eos_token;
    int32_t bos_id = -1;
    int32_t eos_id = -1;
    bool add_bos_token = false;
    bool add_eos_token = false;

    std::unique_ptr<minja::chat_template> chat_tpl;
    bool is_loaded = false;

    void clear() {
        normalizers.clear();
        pretokenizers.clear();
        added_specs.clear();
        raw_added_matcher.build(added_specs, false);
        norm_added_matcher.build(added_specs, true);
        byte_to_id.fill(-1);
        merges.clear();
        piece_to_id.clear();
        ignore_merges = false;
        bos_token.clear();
        eos_token.clear();
        bos_id = -1;
        eos_id = -1;
        add_bos_token = false;
        add_eos_token = false;
        chat_tpl.reset();
        is_loaded = false;
    }

    std::string normalize_text(std::string_view input) const {
        if (normalizers.empty() || input.empty()) return std::string(input);
        std::string cur(input);
        for (NormKind nk : normalizers) {
            switch (nk) {
                case NormKind::NFC:
                    cur = apply_utf8proc_norm(cur, static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
                    break;
                case NormKind::NFD:
                    cur = apply_utf8proc_norm(cur, static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_DECOMPOSE));
                    break;
                case NormKind::NFKC:
                    cur = apply_utf8proc_norm(cur, static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE | UTF8PROC_COMPAT));
                    break;
                case NormKind::NFKD:
                    cur = apply_utf8proc_norm(cur, static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_DECOMPOSE | UTF8PROC_COMPAT));
                    break;
                case NormKind::Lowercase:
                    cur = apply_utf8proc_norm(cur, static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_CASEFOLD));
                    break;
                case NormKind::StripAccents:
                    cur = apply_utf8proc_norm(cur, static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_DECOMPOSE | UTF8PROC_STRIPMARK));
                    break;
            }
        }
        return cur;
    }

    void bpe_encode_piece(std::string_view raw_bytes, std::vector<int32_t>& out) const {
        const size_t n = raw_bytes.size();
        if (n == 0) return;
        if (n == 1) {
            int32_t id = byte_to_id[static_cast<unsigned char>(raw_bytes[0])];
            if (id >= 0) out.push_back(id);
            return;
        }
        if (ignore_merges && !piece_to_id.empty()) {
            auto it = piece_to_id.find(std::string(raw_bytes));
            if (it != piece_to_id.end()) {
                out.push_back(it->second);
                return;
            }
        }

        // Small-piece stack fast path (covers ~98% of natural-language & code tokens with zero heap allocs).
        // Exact (min_rank, leftmost_pos) selection matches HuggingFace Rust tokenizers BinaryHeap order.
        if (n <= 32) {
            int32_t ids[32];
            size_t len = n;
            for (size_t i = 0; i < n; ++i) {
                ids[i] = byte_to_id[static_cast<unsigned char>(raw_bytes[i])];
            }
            while (len > 1) {
                int32_t best_rank = std::numeric_limits<int32_t>::max();
                int32_t best_merged = -1;
                size_t best_pos = 0;
                for (size_t i = 0; i + 1 < len; ++i) {
                    auto it = merges.find(pack_pair(ids[i], ids[i + 1]));
                    if (it != merges.end() && it->second.rank < best_rank) {
                        best_rank = it->second.rank;
                        best_merged = it->second.merged_id;
                        best_pos = i;
                    }
                }
                if (best_merged < 0) break;
                ids[best_pos] = best_merged;
                for (size_t i = best_pos + 1; i + 1 < len; ++i) {
                    ids[i] = ids[i + 1];
                }
                --len;
            }
            for (size_t i = 0; i < len; ++i) {
                if (ids[i] >= 0) out.push_back(ids[i]);
            }
            return;
        }

        // General O(L log L) doubly-linked list + min-heap for arbitrarily long pieces (e.g. long whitespace/hex).
        struct Node {
            int32_t id = -1;
            int32_t prev = -1;
            int32_t next = -1;
            bool alive = true;
        };
        struct HeapItem {
            int32_t rank;
            int32_t pos;
            int32_t right_pos;
            int32_t left_id;
            int32_t right_id;
            int32_t merged_id;
            bool operator>(const HeapItem& o) const {
                if (rank != o.rank) return rank > o.rank;
                return pos > o.pos;
            }
        };

        std::vector<Node> nodes(n);
        for (size_t i = 0; i < n; ++i) {
            nodes[i].id = byte_to_id[static_cast<unsigned char>(raw_bytes[i])];
            nodes[i].prev = static_cast<int32_t>(i) - 1;
            nodes[i].next = (i + 1 < n) ? static_cast<int32_t>(i + 1) : -1;
            nodes[i].alive = true;
        }

        std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> pq;
        auto push_if_merge = [&](int32_t lp) {
            int32_t rp = nodes[static_cast<size_t>(lp)].next;
            if (rp < 0) return;
            int32_t lid = nodes[static_cast<size_t>(lp)].id;
            int32_t rid = nodes[static_cast<size_t>(rp)].id;
            auto it = merges.find(pack_pair(lid, rid));
            if (it != merges.end()) {
                pq.push(HeapItem{it->second.rank, lp, rp, lid, rid, it->second.merged_id});
            }
        };

        for (size_t i = 0; i + 1 < n; ++i) {
            push_if_merge(static_cast<int32_t>(i));
        }

        while (!pq.empty()) {
            HeapItem top = pq.top();
            pq.pop();
            auto& ln = nodes[static_cast<size_t>(top.pos)];
            auto& rn = nodes[static_cast<size_t>(top.right_pos)];
            if (!ln.alive || !rn.alive || ln.next != top.right_pos ||
                ln.id != top.left_id || rn.id != top.right_id) {
                continue;
            }
            ln.id = top.merged_id;
            rn.alive = false;
            int32_t rr = rn.next;
            ln.next = rr;
            if (rr >= 0) nodes[static_cast<size_t>(rr)].prev = top.pos;
            if (ln.prev >= 0) push_if_merge(ln.prev);
            if (ln.next >= 0) push_if_merge(top.pos);
        }

        for (int32_t cur = 0; cur >= 0; cur = nodes[static_cast<size_t>(cur)].next) {
            if (nodes[static_cast<size_t>(cur)].alive && nodes[static_cast<size_t>(cur)].id >= 0) {
                out.push_back(nodes[static_cast<size_t>(cur)].id);
            }
        }
    }

    void encode_normalized_segment(std::string_view norm_text, std::vector<int32_t>& out) const {
        if (norm_text.empty()) return;
        std::vector<std::string> pieces;
        pieces.emplace_back(norm_text);
        std::vector<std::string> next_pieces;

        for (const auto& step : pretokenizers) {
            next_pieces.clear();
            next_pieces.reserve(pieces.size() * 2);
            switch (step.kind) {
                case PreTokStep::Kind::ByteLevel:
                    for (auto& p : pieces) {
                        if (step.add_prefix_space && !p.empty() && p[0] != ' ') {
                            next_pieces.push_back(" " + p);
                        } else {
                            next_pieces.push_back(std::move(p));
                        }
                    }
                    break;
                case PreTokStep::Kind::SplitRegex:
                    for (const auto& p : pieces) {
                        run_pcre2_split(step, p, next_pieces);
                    }
                    break;
                case PreTokStep::Kind::SplitString:
                    for (const auto& p : pieces) {
                        run_literal_split(step, p, next_pieces);
                    }
                    break;
            }
            pieces.swap(next_pieces);
        }

        for (const auto& piece : pieces) {
            bpe_encode_piece(piece, out);
        }
    }

    std::vector<int32_t> encode(std::string_view text, bool add_special_tokens) const {
        std::vector<int32_t> out;
        if (!is_loaded) return out;
        out.reserve(text.size() / 3 + 16);

        if (add_special_tokens && add_bos_token && bos_id >= 0) {
            out.push_back(bos_id);
        }

        std::vector<AddedMatcher::Segment> raw_segs;
        raw_added_matcher.split(text, raw_segs);

        std::vector<AddedMatcher::Segment> norm_segs;
        for (const auto& seg : raw_segs) {
            if (seg.added_id >= 0) {
                out.push_back(seg.added_id);
                continue;
            }
            std::string_view sub = text.substr(seg.begin, seg.end - seg.begin);
            std::string norm = normalize_text(sub);
            if (norm_added_matcher.empty) {
                encode_normalized_segment(norm, out);
            } else {
                norm_added_matcher.split(norm, norm_segs);
                for (const auto& nseg : norm_segs) {
                    if (nseg.added_id >= 0) {
                        out.push_back(nseg.added_id);
                    } else {
                        encode_normalized_segment(
                            std::string_view(norm).substr(nseg.begin, nseg.end - nseg.begin), out);
                    }
                }
            }
        }

        if (add_special_tokens && add_eos_token && eos_id >= 0) {
            out.push_back(eos_id);
        }
        return out;
    }
};

NativeTokenizer::NativeTokenizer() : impl_(std::make_unique<Impl>()) {}
NativeTokenizer::~NativeTokenizer() = default;
NativeTokenizer::NativeTokenizer(NativeTokenizer&&) noexcept = default;
NativeTokenizer& NativeTokenizer::operator=(NativeTokenizer&&) noexcept = default;

bool NativeTokenizer::load_from_model_dir(const std::string& model_dir, std::string* err_out) {
    std::string sep = (!model_dir.empty() && model_dir.back() == '/') ? "" : "/";
    std::string tok_data;
    if (!read_file_all(model_dir + sep + "tokenizer.json", tok_data)) {
        set_err(err_out, "could not read " + model_dir + sep + "tokenizer.json");
        return false;
    }
    std::string cfg_data;
    read_file_all(model_dir + sep + "tokenizer_config.json", cfg_data); // optional
    std::string tpl_data;
    read_file_all(model_dir + sep + "chat_template.jinja", tpl_data);   // optional (fallback to tokenizer_config.json)
    return load_from_strings(tok_data, cfg_data, tpl_data, err_out);
}

bool NativeTokenizer::load_from_strings(
    std::string_view tokenizer_json,
    std::string_view tokenizer_config_json,
    std::string_view chat_template_jinja,
    std::string* err_out
) {
    impl_->clear();

    // 1) Load fast lock-free BpeDetokenizer first (validates ByteLevel decoder & vocab).
    std::string detok_err;
    if (!impl_->detok.load_from_json_strings(tokenizer_json, tokenizer_config_json, &detok_err)) {
        set_err(err_out, "detokenizer load failed: " + detok_err);
        return false;
    }

    // 2) Parse tokenizer.json for encoder (normalizer, pre_tokenizer, added_tokens, vocab, merges).
    nlohmann::json tj;
    try {
        tj = nlohmann::json::parse(tokenizer_json.begin(), tokenizer_json.end());
    } catch (const std::exception& e) {
        set_err(err_out, std::string("tokenizer.json parse error: ") + e.what());
        return false;
    }
    if (!tj.is_object() || !tj.contains("model") || !tj["model"].is_object()) {
        set_err(err_out, "tokenizer.json missing model object");
        return false;
    }

    std::string err;
    if (tj.contains("normalizer") && !parse_normalizer_json(tj["normalizer"], impl_->normalizers, err)) {
        set_err(err_out, err);
        return false;
    }
    if (tj.contains("pre_tokenizer") && !parse_pretokenizer_json(tj["pre_tokenizer"], impl_->pretokenizers, err)) {
        set_err(err_out, err);
        return false;
    }

    const auto& model = tj["model"];
    if (!model.contains("vocab") || !model["vocab"].is_object()) {
        set_err(err_out, "tokenizer.json missing model.vocab");
        return false;
    }
    if (model.contains("ignore_merges") && model["ignore_merges"].is_boolean()) {
        impl_->ignore_merges = model["ignore_merges"].get<bool>();
    }

    // Build string->id vocab map for merge resolution & single-byte ByteLevel lookup.
    const auto& vocab_obj = model["vocab"];
    std::unordered_map<std::string, int32_t> vocab_map;
    vocab_map.reserve(vocab_obj.size() * 2);
    for (auto it = vocab_obj.begin(); it != vocab_obj.end(); ++it) {
        if (it.value().is_number_integer()) {
            vocab_map.emplace(it.key(), it.value().get<int32_t>());
        }
    }

    const auto byte_to_utf8 = make_byte_to_utf8_table();
    for (int b = 0; b < 256; ++b) {
        auto it = vocab_map.find(byte_to_utf8[static_cast<size_t>(b)]);
        if (it == vocab_map.end()) {
            set_err(err_out, "tokenizer.json missing ByteLevel token for byte " + std::to_string(b));
            return false;
        }
        impl_->byte_to_id[static_cast<size_t>(b)] = it->second;
    }

    if (impl_->ignore_merges) {
        // Pre-populate raw-byte piece lookup only if model sets ignore_merges=true.
        impl_->piece_to_id.reserve(vocab_map.size());
        for (const auto& [tok_str, tid] : vocab_map) {
            std::string raw = impl_->detok.decode(&tid, 1, false);
            impl_->piece_to_id.emplace(std::move(raw), tid);
        }
    }

    // Parse BPE merges (supports both legacy `"left right"` strings and HF v0.20+ `["left", "right"]` arrays).
    if (model.contains("merges") && model["merges"].is_array()) {
        const auto& merges_arr = model["merges"];
        impl_->merges.reserve(merges_arr.size() * 2);
        int32_t rank = 0;
        std::string merged_key;
        for (const auto& item : merges_arr) {
            std::string left, right;
            if (item.is_string()) {
                const std::string& s = item.get_ref<const std::string&>();
                size_t sp = s.find(' ');
                if (sp == std::string::npos) { ++rank; continue; }
                left.assign(s, 0, sp);
                right.assign(s, sp + 1, std::string::npos);
            } else if (item.is_array() && item.size() == 2 &&
                       item[0].is_string() && item[1].is_string()) {
                left = item[0].get<std::string>();
                right = item[1].get<std::string>();
            } else {
                ++rank;
                continue;
            }
            auto it_l = vocab_map.find(left);
            auto it_r = vocab_map.find(right);
            if (it_l != vocab_map.end() && it_r != vocab_map.end()) {
                merged_key.clear();
                merged_key.reserve(left.size() + right.size());
                merged_key.append(left);
                merged_key.append(right);
                auto it_m = vocab_map.find(merged_key);
                if (it_m != vocab_map.end()) {
                    impl_->merges.emplace(pack_pair(it_l->second, it_r->second),
                                          MergeEntry{rank, it_m->second});
                }
            }
            ++rank;
        }
    }

    // Parse added_tokens from tokenizer.json
    if (tj.contains("added_tokens") && tj["added_tokens"].is_array()) {
        for (const auto& a : tj["added_tokens"]) {
            if (!a.is_object() || !a.contains("id") || !a.contains("content")) continue;
            if (!a["id"].is_number_integer() || !a["content"].is_string()) continue;
            AddedTokenSpec sp;
            sp.id = a["id"].get<int32_t>();
            sp.content = a["content"].get<std::string>();
            sp.special = a.contains("special") && a["special"].is_boolean() ? a["special"].get<bool>() : false;
            sp.single_word = a.contains("single_word") && a["single_word"].is_boolean() ? a["single_word"].get<bool>() : false;
            sp.lstrip = a.contains("lstrip") && a["lstrip"].is_boolean() ? a["lstrip"].get<bool>() : false;
            sp.rstrip = a.contains("rstrip") && a["rstrip"].is_boolean() ? a["rstrip"].get<bool>() : false;
            sp.normalized = a.contains("normalized") && a["normalized"].is_boolean() ? a["normalized"].get<bool>() : false;
            impl_->added_specs.push_back(std::move(sp));
        }
    }

    // 3) Parse optional tokenizer_config.json (bos/eos tokens + fallback chat_template)
    std::string tpl_src(chat_template_jinja);
    if (!tokenizer_config_json.empty()) {
        try {
            nlohmann::json tc = nlohmann::json::parse(tokenizer_config_json.begin(), tokenizer_config_json.end());
            if (tc.is_object()) {
                impl_->bos_token = extract_config_token(tc, "bos_token");
                impl_->eos_token = extract_config_token(tc, "eos_token");
                if (tc.contains("add_bos_token") && tc["add_bos_token"].is_boolean()) {
                    impl_->add_bos_token = tc["add_bos_token"].get<bool>();
                }
                if (tc.contains("add_eos_token") && tc["add_eos_token"].is_boolean()) {
                    impl_->add_eos_token = tc["add_eos_token"].get<bool>();
                }
                if (tpl_src.empty() && tc.contains("chat_template") && tc["chat_template"].is_string()) {
                    tpl_src = tc["chat_template"].get<std::string>();
                }
            }
        } catch (const std::exception& e) {
            set_err(err_out, std::string("tokenizer_config.json parse error: ") + e.what());
            return false;
        }
    }

    // Resolve bos_id / eos_id (from added_tokens or vocab_map)
    auto find_tok_id = [&](const std::string& tok) -> int32_t {
        if (tok.empty()) return -1;
        for (const auto& sp : impl_->added_specs) {
            if (sp.content == tok) return sp.id;
        }
        auto it = vocab_map.find(tok);
        return (it != vocab_map.end()) ? it->second : -1;
    };
    impl_->bos_id = find_tok_id(impl_->bos_token);
    impl_->eos_id = find_tok_id(impl_->eos_token);

    impl_->raw_added_matcher.build(impl_->added_specs, false);
    impl_->norm_added_matcher.build(impl_->added_specs, true);

    // 4) Compile chat_template.jinja via Google minja
    if (!tpl_src.empty()) {
        try {
            impl_->chat_tpl = std::make_unique<minja::chat_template>(
                tpl_src, impl_->bos_token, impl_->eos_token);
        } catch (const std::exception& e) {
            set_err(err_out, std::string("minja chat_template parse error: ") + e.what());
            return false;
        }
    }

    impl_->is_loaded = true;
    return true;
}

bool NativeTokenizer::loaded() const { return impl_->is_loaded; }
bool NativeTokenizer::has_chat_template() const { return impl_->chat_tpl != nullptr; }
size_t NativeTokenizer::vocab_size() const { return impl_->detok.vocab_size(); }
size_t NativeTokenizer::merge_count() const { return impl_->merges.size(); }
size_t NativeTokenizer::added_token_count() const { return impl_->added_specs.size(); }

std::vector<int32_t> NativeTokenizer::encode(std::string_view text, bool add_special_tokens) const {
    return impl_->encode(text, add_special_tokens);
}

std::string NativeTokenizer::decode(const std::vector<int32_t>& ids, bool skip_special_tokens) const {
    return impl_->detok.decode(ids, skip_special_tokens);
}

std::string NativeTokenizer::decode(const int32_t* ids, size_t count, bool skip_special_tokens) const {
    return impl_->detok.decode(ids, count, skip_special_tokens);
}

const BpeDetokenizer& NativeTokenizer::detokenizer() const {
    return impl_->detok;
}

std::string NativeTokenizer::render_chat_text(
    const Json& messages,
    const Json* tools,
    const Json* kwargs,
    bool add_generation_prompt
) const {
    if (!impl_->chat_tpl) {
        throw std::runtime_error("No chat_template loaded for this checkpoint");
    }

    minja::chat_template_inputs inputs;
    inputs.messages = normalize_openai_messages(to_ordered_json(messages));
    inputs.add_generation_prompt = add_generation_prompt;
    if (tools != nullptr && !tools->is_null()) {
        inputs.tools = to_ordered_json(*tools);
    }

    OrderedJson extra = OrderedJson::object();
    if (kwargs != nullptr && kwargs->is_object()) {
        extra = to_ordered_json(*kwargs);
    }

    // Map OpenAI reasoning_effort="none"/"off"/"disabled" -> enable_thinking=false
    // (Qwen3.8 chat_template.jinja only accepts xhigh/medium/low when thinking is enabled).
    if (extra.contains("reasoning_effort") && extra["reasoning_effort"].is_string()) {
        std::string re = extra["reasoning_effort"].get<std::string>();
        std::string re_lo = re;
        for (char& c : re_lo) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (re_lo == "none" || re_lo == "off" || re_lo == "disabled") {
            extra.erase("reasoning_effort");
            if (!extra.contains("enable_thinking")) {
                extra["enable_thinking"] = false;
            }
        }
    }
    if (!extra.empty()) {
        inputs.extra_context = std::move(extra);
    }

    minja::chat_template_options opts;
    opts.apply_polyfills = false;
    return impl_->chat_tpl->apply(inputs, opts);
}

ChatRenderResult NativeTokenizer::render_chat(
    const Json& messages,
    const Json* tools,
    const Json* kwargs,
    bool add_generation_prompt
) const {
    ChatRenderResult res;
    res.rendered_text = render_chat_text(messages, tools, kwargs, add_generation_prompt);
    res.token_ids = impl_->encode(res.rendered_text, false);

    std::string trimmed = rtrim_ws(res.rendered_text);
    res.in_think = trimmed.size() >= 7 &&
                   trimmed.compare(trimmed.size() - 7, 7, "<think>") == 0;

    // Detect exact system-prompt (+ tools) prefix boundary for shared-system prefix caching.
    constexpr std::string_view kSysStart = "<|im_start|>system\n";
    constexpr std::string_view kSysEnd = "<|im_end|>\n";
    if (res.rendered_text.compare(0, kSysStart.size(), kSysStart) == 0) {
        size_t end_idx = res.rendered_text.find(kSysEnd);
        if (end_idx != std::string::npos) {
            std::string_view sys_str(res.rendered_text.data(), end_idx + kSysEnd.size());
            auto sys_ids = impl_->encode(sys_str, false);
            if (sys_ids.size() >= 256 && sys_ids.size() < res.token_ids.size() &&
                std::equal(sys_ids.begin(), sys_ids.end(), res.token_ids.begin())) {
                res.sys_prefix_len = static_cast<int>(sys_ids.size());
            }
        }
    }
    return res;
}

} // namespace amperegen
