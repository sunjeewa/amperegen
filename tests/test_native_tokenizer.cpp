// Native C++20 tokenizer (PCRE2 + utf8proc + ByteLevel BPE) + minja chat-template tests
// and live HuggingFace AutoTokenizer parity harness.

#include "server/json_lite.hpp"
#include "server/tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace amperegen;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "CHECK FAILED at %s:%d: %s\n", __FILE__, __LINE__,   \
                    #cond);                                                      \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

namespace {

// Build a minimal valid ByteLevel tokenizer.json containing all 256 byte tokens plus custom merges & added tokens.
std::string make_synthetic_tokenizer_json(bool array_merges) {
    // Standard GPT-2 ByteLevel byte -> UTF-8 char mapping
    bool direct[256] = {};
    for (int b = 33; b <= 126; ++b) direct[b] = true;
    for (int b = 161; b <= 172; ++b) direct[b] = true;
    for (int b = 174; b <= 255; ++b) direct[b] = true;

    auto encode_cp = [](uint32_t cp) -> std::string {
        std::string s;
        if (cp < 0x80) s.push_back((char)cp);
        else {
            s.push_back((char)(0xC0 | (cp >> 6)));
            s.push_back((char)(0x80 | (cp & 0x3F)));
        }
        return s;
    };

    std::string b2u[256];
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = direct[b] ? (uint32_t)b : (uint32_t)(256 + extra++);
        b2u[b] = encode_cp(cp);
    }

    auto j_vocab = Json::Object{};
    for (int b = 0; b < 256; ++b) {
        j_vocab[b2u[b]] = b;
    }
    // Merged tokens:
    // 256: "  " (Ġ + Ġ)
    // 257: "    " (ĠĠ + ĠĠ)
    // 258: "he"
    // 259: "ll"
    // 260: "hell"
    // 261: "hello"
    // 262: "é" (UTF-8 bytes 0xC3 0xA9)
    std::string sp = b2u[(unsigned char)' '];
    std::string sp2 = sp + sp;
    std::string sp4 = sp2 + sp2;
    std::string e_acute = b2u[0xC3] + b2u[0xA9];
    j_vocab[sp2] = 256;
    j_vocab[sp4] = 257;
    j_vocab["he"] = 258;
    j_vocab["ll"] = 259;
    j_vocab["hell"] = 260;
    j_vocab["hello"] = 261;
    j_vocab[e_acute] = 262;

    Json::Array merges;
    auto add_merge = [&](const std::string& a, const std::string& b) {
        if (array_merges) {
            Json::Array pair;
            pair.emplace_back(a);
            pair.emplace_back(b);
            merges.emplace_back(std::move(pair));
        } else {
            merges.emplace_back(a + " " + b);
        }
    };
    add_merge(sp, sp);
    add_merge(sp2, sp2);
    add_merge("h", "e");
    add_merge("l", "l");
    add_merge("he", "ll");
    add_merge("hell", "o");
    add_merge(b2u[0xC3], b2u[0xA9]);

    Json::Array added;
    auto add_tok = [&](int id, const std::string& content, bool special) {
        Json o;
        o["id"] = id;
        o["content"] = content;
        o["single_word"] = false;
        o["lstrip"] = false;
        o["rstrip"] = false;
        o["normalized"] = false;
        o["special"] = special;
        added.push_back(std::move(o));
    };
    add_tok(300, "<|im_start|>", true);
    add_tok(301, "<|im_end|>", true);
    add_tok(302, "<|im_start|>extended", true); // tests LeftmostLongest over <|im_start|>
    add_tok(303, "<think>", false);
    add_tok(304, "</think>", false);

    Json root;
    root["normalizer"]["type"] = "NFC";
    Json split_pt;
    split_pt["type"] = "Split";
    split_pt["pattern"]["Regex"] =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    split_pt["behavior"] = "Isolated";
    split_pt["invert"] = false;
    Json bl_pt;
    bl_pt["type"] = "ByteLevel";
    bl_pt["add_prefix_space"] = false;
    bl_pt["trim_offsets"] = true;
    bl_pt["use_regex"] = false;
    root["pre_tokenizer"]["type"] = "Sequence";
    root["pre_tokenizer"]["pretokenizers"] = Json::Array{std::move(split_pt), std::move(bl_pt)};
    root["decoder"]["type"] = "ByteLevel";
    root["added_tokens"] = std::move(added);
    root["model"]["type"] = "BPE";
    root["model"]["vocab"] = std::move(j_vocab);
    root["model"]["merges"] = std::move(merges);
    return root.dump();
}

const char* kMiniQwenTemplate = R"JINJA(
{%- if enable_thinking is undefined or enable_thinking is true -%}
  {%- set r_inst = 'THINK_ON' -%}
{%- else -%}
  {%- set r_inst = '' -%}
{%- endif -%}
{%- if tools and tools is iterable and tools is not mapping -%}
{{- '<|im_start|>system\n' -}}
{%- if r_inst -%}{{- r_inst + '\n' -}}{%- endif -%}
{{- '<tools>' -}}
{%- for t in tools -%}{{- '\n' -}}{{- t | tojson -}}{%- endfor -%}
{{- '\n</tools>' -}}
{%- if messages[0].role == 'system' -%}{{- '\n' + (messages[0].content | trim) -}}{%- endif -%}
{{- '<|im_end|>\n' -}}
{%- elif messages[0].role == 'system' -%}
{{- '<|im_start|>system\n' + (messages[0].content | trim) + '<|im_end|>\n' -}}
{%- endif -%}
{%- for m in messages -%}
  {%- if m.role == 'user' -%}
    {{- '<|im_start|>user\n' + (m.content | trim) + '<|im_end|>\n' -}}
  {%- elif m.role == 'assistant' -%}
    {{- '<|im_start|>assistant\n' -}}
    {%- if m.reasoning_content is string and m.reasoning_content -%}
      {{- '<think>\n' + (m.reasoning_content | trim) + '\n</think>\n' -}}
    {%- endif -%}
    {%- if m.content is string -%}{{- m.content | trim -}}{%- endif -%}
    {%- if m.tool_calls -%}
      {%- for tc in m.tool_calls -%}
        {%- set fn = tc.function if tc.function is defined else tc -%}
        {{- '<tool_call><fn=' + fn.name + '>' -}}
        {%- for k, v in fn.arguments | items -%}
          {{- k + '=' + (v | string if v is string else v | tojson | safe) + ';' -}}
        {%- endfor -%}
        {{- '</tool_call>' -}}
      {%- endfor -%}
    {%- endif -%}
    {{- '<|im_end|>\n' -}}
  {%- elif m.role == 'tool' -%}
    {{- '<|im_start|>user\n<tool_response>' + (m.content | trim) + '</tool_response><|im_end|>\n' -}}
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
  {{- '<|im_start|>assistant\n' -}}
  {%- if enable_thinking is defined and enable_thinking is false -%}
    {{- '<think>\n\n</think>\n\n' -}}
  {%- else -%}
    {{- '<think>\n' -}}
  {%- endif -%}
{%- endif -%}
)JINJA";

void run_unit_tests() {
    for (int fmt = 0; fmt < 2; ++fmt) {
        bool array_merges = (fmt == 1);
        std::string tj = make_synthetic_tokenizer_json(array_merges);
        NativeTokenizer tok;
        std::string err;
        CHECK(tok.load_from_strings(tj, R"({"eos_token":"<|im_end|>","clean_up_tokenization_spaces":false})",
                                    kMiniQwenTemplate, &err));
        CHECK(tok.loaded());
        CHECK(tok.has_chat_template());
        CHECK(tok.merge_count() == 7);
        CHECK(tok.added_token_count() == 5);

        // 1) Basic BPE merge order
        auto ids_hello = tok.encode("hello");
        CHECK(ids_hello.size() == 1 && ids_hello[0] == 261);
        CHECK(tok.decode(ids_hello) == "hello");

        // 2) Exact leftmost-first merge tie-breaking on repeated spaces (small stack path)
        auto ids_sp4 = tok.encode("    ");
        CHECK(ids_sp4.size() == 1 && ids_sp4[0] == 257);
        CHECK(tok.decode(ids_sp4) == "    ");

        // 3) Long piece (>32 bytes -> min-heap O(L log L) path): 40 spaces -> 10 * id(257)
        std::string sp40(40, ' ');
        auto ids_sp40 = tok.encode(sp40);
        CHECK(ids_sp40.size() == 10);
        for (int32_t id : ids_sp40) CHECK(id == 257);
        CHECK(tok.decode(ids_sp40) == sp40);

        // 4) NFC normalization via utf8proc: NFD "e\xCC\x81" == NFC "\xC3\xA9" == token 262
        auto ids_nfc = tok.encode("\xC3\xA9");
        auto ids_nfd = tok.encode("e\xCC\x81");
        CHECK(ids_nfc.size() == 1 && ids_nfc[0] == 262);
        CHECK(ids_nfd == ids_nfc);

        // 5) Added token LeftmostLongest matching
        auto ids_added = tok.encode("<|im_start|>extended<|im_start|>hello<|im_end|>");
        CHECK(ids_added.size() == 4);
        CHECK(ids_added[0] == 302);
        CHECK(ids_added[1] == 300);
        CHECK(ids_added[2] == 261);
        CHECK(ids_added[3] == 301);
        CHECK(tok.decode(ids_added, true) == "hello");
        CHECK(tok.decode(ids_added, false) == "<|im_start|>extended<|im_start|>hello<|im_end|>");

        // 6) Unicode/emoji round-trip
        std::string uni = "Hello 世界 🚀 café 're 12345\n  indented";
        auto ids_uni = tok.encode(uni);
        CHECK(tok.decode(ids_uni) == uni);

        // 7) Minja chat template: developer->system merge, tools, tool_calls string args parsing, thinking modes
        Json msgs = Json::parse(R"JSON([
            {"role": "developer", "content": "Rule A"},
            {"role": "system", "content": "Rule B"},
            {"role": "user", "content": "Check weather"},
            {"role": "assistant", "reasoning": "Need tool", "content": null,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "get_weather", "arguments": "{\"city\":\"Melbourne\",\"days\":3}"}}]},
            {"role": "tool", "content": "22C sunny"}
        ])JSON");
        Json tools = Json::parse(R"JSON([{"type":"function","function":{"name":"get_weather"}}])JSON");

        // Default thinking enabled
        ChatRenderResult r_think = tok.render_chat(msgs, &tools, nullptr, true);
        CHECK(r_think.in_think);
        CHECK(r_think.rendered_text.find("Rule A\n\nRule B") != std::string::npos);
        CHECK(r_think.rendered_text.find("THINK_ON") != std::string::npos);
        CHECK(r_think.rendered_text.find("<think>\nNeed tool\n</think>") != std::string::npos);
        CHECK(r_think.rendered_text.find("city=Melbourne;") != std::string::npos);
        CHECK(r_think.rendered_text.find("days=3;") != std::string::npos);
        CHECK(r_think.rendered_text.find("<tool_response>22C sunny</tool_response>") != std::string::npos);
        CHECK(!r_think.token_ids.empty());

        // reasoning_effort="none" -> enable_thinking=false
        Json kw_none = Json::parse(R"({"reasoning_effort": "none"})");
        ChatRenderResult r_nothink = tok.render_chat(msgs, &tools, &kw_none, true);
        CHECK(!r_nothink.in_think);
        CHECK(r_nothink.rendered_text.find("THINK_ON") == std::string::npos);
        CHECK(r_nothink.rendered_text.find("<think>\n\n</think>\n\n") != std::string::npos);

        printf("[PASS] synthetic suite (%s merges)\n", array_merges ? "array" : "string");
    }
    printf("ALL NATIVE TOKENIZER UNIT TESTS PASSED!\n");
}

int run_parity_server(const std::string& model_dir) {
    NativeTokenizer tok;
    std::string err;
    if (!tok.load_from_model_dir(model_dir, &err)) {
        fprintf(stderr, "FATAL: failed to load %s: %s\n", model_dir.c_str(), err.c_str());
        return 1;
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        Json resp;
        try {
            Json req = Json::parse(line);
            std::string op = req.at("op").as_string();
            if (op == "encode") {
                bool add_sp = req.contains("add_special_tokens") && req.at("add_special_tokens").as_bool(false);
                auto ids = tok.encode(req.at("text").as_string(), add_sp);
                Json::Array arr;
                arr.reserve(ids.size());
                for (int32_t id : ids) arr.emplace_back((int64_t)id);
                resp["ids"] = std::move(arr);
            } else if (op == "decode") {
                bool skip_sp = !req.contains("skip_special_tokens") || req.at("skip_special_tokens").as_bool(true);
                std::vector<int32_t> ids;
                for (const auto& x : req.at("ids").as_array()) ids.push_back((int32_t)x.as_int());
                resp["text"] = tok.decode(ids, skip_sp);
            } else if (op == "render_chat") {
                const Json* tools = (req.contains("tools") && !req.at("tools").is_null()) ? &req.at("tools") : nullptr;
                const Json* kw = (req.contains("kwargs") && req.at("kwargs").is_object()) ? &req.at("kwargs") : nullptr;
                bool add_gen = !req.contains("add_generation_prompt") || req.at("add_generation_prompt").as_bool(true);
                ChatRenderResult cr = tok.render_chat(req.at("messages"), tools, kw, add_gen);
                resp["rendered"] = cr.rendered_text;
                resp["in_think"] = cr.in_think;
                resp["sys_prefix_len"] = cr.sys_prefix_len;
                Json::Array arr;
                arr.reserve(cr.token_ids.size());
                for (int32_t id : cr.token_ids) arr.emplace_back((int64_t)id);
                resp["ids"] = std::move(arr);
            } else {
                resp["error"] = "unknown op";
            }
        } catch (const std::exception& e) {
            resp["error"] = std::string(e.what());
        }
        std::cout << resp.dump() << "\n" << std::flush;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--parity") {
        return run_parity_server(argv[2]);
    }
    run_unit_tests();
    return 0;
}
