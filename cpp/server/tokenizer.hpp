#pragma once
// Native C++20 HuggingFace ByteLevel BPE Tokenizer + Minja Jinja2 Chat Template Engine.
//
// Zero Python runtime dependencies:
//  - Normalization: libutf8proc (NFC / NFD / NFKC / NFKD / Sequence)
//  - Pre-tokenization: libpcre2-8 (UTF-8 + UCP + JIT) executing tokenizer.json regexes
//  - Encoding: integer-ID ByteLevel BPE with exact (min_rank, leftmost_pos) merge order
//  - Decoding: fast lock-free BpeDetokenizer with RFC 3629 streaming UTF-8 holdback
//  - Chat templates: Google minja executing each checkpoint's chat_template.jinja directly
//
// After load_from_model_dir(), all methods are const and thread-safe across HTTP workers.

#include "server/bpe_detokenizer.hpp"
#include "server/json_lite.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace amperegen {

struct ChatRenderResult {
    std::vector<int32_t> token_ids;
    std::string rendered_text;
    bool in_think = false;
    int sys_prefix_len = 0;
};

class NativeTokenizer {
public:
    NativeTokenizer();
    ~NativeTokenizer();
    NativeTokenizer(NativeTokenizer&&) noexcept;
    NativeTokenizer& operator=(NativeTokenizer&&) noexcept;
    NativeTokenizer(const NativeTokenizer&) = delete;
    NativeTokenizer& operator=(const NativeTokenizer&) = delete;

    // Load tokenizer.json, tokenizer_config.json, and chat_template.jinja from `model_dir`.
    bool load_from_model_dir(const std::string& model_dir, std::string* err_out = nullptr);

    // Load directly from in-memory JSON/template strings (used by unit tests).
    bool load_from_strings(
        std::string_view tokenizer_json,
        std::string_view tokenizer_config_json = {},
        std::string_view chat_template_jinja = {},
        std::string* err_out = nullptr);

    bool loaded() const;
    bool has_chat_template() const;
    size_t vocab_size() const;
    size_t merge_count() const;
    size_t added_token_count() const;

    // Encode raw text into token IDs (matches HuggingFace AutoTokenizer.encode).
    std::vector<int32_t> encode(std::string_view text, bool add_special_tokens = false) const;

    // Decode token IDs into UTF-8 text (matches HuggingFace AutoTokenizer.decode).
    std::string decode(const std::vector<int32_t>& ids, bool skip_special_tokens = true) const;
    std::string decode(const int32_t* ids, size_t count, bool skip_special_tokens = true) const;

    // Direct access to the lock-free streaming detokenizer.
    const BpeDetokenizer& detokenizer() const;

    // Render OpenAI chat messages/tools/kwargs via checkpoint chat_template.jinja (minja)
    // and return the rendered prompt text without tokenizing.
    std::string render_chat_text(
        const Json& messages,
        const Json* tools = nullptr,
        const Json* kwargs = nullptr,
        bool add_generation_prompt = true) const;

    // Render OpenAI chat messages/tools/kwargs via minja, encode to token IDs, and detect
    // `<think>` prompt state + shared system-prompt prefix boundary.
    ChatRenderResult render_chat(
        const Json& messages,
        const Json* tools = nullptr,
        const Json* kwargs = nullptr,
        bool add_generation_prompt = true) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace amperegen
