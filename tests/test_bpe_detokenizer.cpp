#include "server/bpe_detokenizer.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace amperegen;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            fprintf(stderr, "[FAIL] %s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);        \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

namespace {

// Helper: verify one-shot decode() == incremental StreamDecoder across all batch sizes 1..N.
void check_stream_parity(const BpeDetokenizer& detok,
                         const std::vector<int32_t>& ids,
                         bool skip_special,
                         const std::string& expected,
                         const char* label) {
    std::string oneshot = detok.decode(ids, skip_special);
    if (oneshot != expected) {
        fprintf(stderr, "[FAIL] %s oneshot mismatch:\n  got:      %s (len=%zu)\n  expected: %s (len=%zu)\n",
                label, oneshot.c_str(), oneshot.size(), expected.c_str(), expected.size());
        std::abort();
    }

    for (size_t step : {size_t(1), size_t(2), size_t(3), size_t(5)}) {
        BpeDetokenizer::StreamDecoder sd(detok, skip_special);
        std::string streamed;
        for (size_t i = 0; i < ids.size(); i += step) {
            size_t cnt = std::min(step, ids.size() - i);
            sd.push_batch(ids.data() + i, cnt, streamed);
        }
        sd.finish(streamed);
        if (streamed != expected) {
            fprintf(stderr, "[FAIL] %s stream(step=%zu) mismatch:\n  got:      %s\n  expected: %s\n",
                    label, step, streamed.c_str(), expected.c_str());
            std::abort();
        }
    }
    printf("[PASS] %s\n", label);
}

// Build a synthetic ByteLevel tokenizer.json covering ASCII, space (Ġ), newline (Ċ),
// split 4-byte emoji (🚀 = F0 9F 9A 80 -> ðŁ / ļ / Ģ), split 3-byte CJK (中 = E4 B8 AD),
// invalid UTF-8 raw bytes, JSON \u escapes, NUL byte (Ā), added special & non-special tokens,
// and an unrelated multi-element merges array to exercise skip_value().
std::string make_synthetic_tokenizer_json() {
    // ByteLevel unicode encodings:
    //  0x00 -> U+0100 (\u0100)
    //  0x0A ('\n') -> U+010A (\u010a)
    //  0x20 (' ')  -> U+0120 (\u0120 = Ġ)
    //  0x41 ('A')  -> 'A'
    //  0x80 -> U+0122 (\u0122 = Ģ)
    //  0x90 -> U+0132 (\u0132)
    //  0x9A -> U+013C (\u013c = ļ)
    //  0x9F -> U+0141 (\u0141 = Ł)
    //  0xA0 -> U+0142 (\u0142)
    //  0xB8 -> U+00B8 (¸)
    //  0xAD -> U+0143 (\u0143 = Ń)
    //  0xC2 -> U+00C2 (Â)
    //  0xE0 -> U+00E0 (à)
    //  0xE4 -> U+00E4 (ä)
    //  0xED -> U+00ED (í)
    //  0xF0 -> U+00F0 (ð)
    //  0xF4 -> U+00F4 (ô)
    return R"JSON({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 100, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 101, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 102, "content": "<think>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
    {"id": 103, "content": "</think>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
    {"id": 104, "content": "<tool_call>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}
  ],
  "normalizer": {"type": "NFC"},
  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "ByteLevel", "add_prefix_space": false}]},
  "post_processor": {"type": "ByteLevel", "add_prefix_space": false},
  "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "byte_fallback": false,
    "vocab": {
      "Hello": 0,
      ",": 1,
      "\u0120world": 2,
      "!": 3,
      "\u010a": 4,
      "\u00f0\u0141": 5,
      "\u013c": 6,
      "\u0122": 7,
      "\u00e4\u00b8": 8,
      "\u0143": 9,
      "A": 10,
      "\u00e0": 11,
      "\u00ed": 12,
      "\u0142": 13,
      "\u00f4": 14,
      "\u0132": 15,
      "\u00c2": 16,
      "\u0100": 17,
      "\"quoted\"\\slash": 18,
      "should_be_overridden_by_added": 102
    },
    "merges": [
      ["H", "e"],
      ["He", "llo"],
      ["\u0120", "world"]
    ]
  }
})JSON";
}

std::string make_synthetic_tokenizer_config() {
    return R"JSON({
  "clean_up_tokenization_spaces": false,
  "added_tokens_decoder": {
    "105": {"content": "<|audio_start|>", "special": true, "lstrip": false, "rstrip": false},
    "106": {"content": "<custom_tag>", "special": false, "lstrip": false, "rstrip": false}
  }
})JSON";
}

} // namespace

int main(int argc, char** argv) {
    // 1. Core ByteLevel, added_tokens override, tokenizer_config merge, split UTF-8, invalid UTF-8
    {
        BpeDetokenizer detok;
        std::string err;
        bool ok = detok.load_from_json_strings(make_synthetic_tokenizer_json(), make_synthetic_tokenizer_config(), &err);
        if (!ok) {
            fprintf(stderr, "Load failed: %s\n", err.c_str());
            return 1;
        }
        CHECK(detok.loaded());
        CHECK(detok.added_token_count() == 7);

        // Plain ASCII + space + newline + JSON escapes
        check_stream_parity(detok, {0, 1, 2, 3, 4, 18}, true, "Hello, world!\n\"quoted\"\\slash", "ascii_and_escapes");

        // Split 4-byte emoji 🚀 (5 = F0 9F, 6 = 9A, 7 = 80) + split 3-byte CJK 中 (8 = E4 B8, 9 = AD)
        check_stream_parity(detok, {0, 2, 5, 6, 7, 8, 9}, true, "Hello world\xF0\x9F\x9A\x80\xE4\xB8\xAD", "split_emoji_and_cjk");

        // Verify mid-stream incremental buffering does NOT emit premature U+FFFD!
        {
            BpeDetokenizer::StreamDecoder sd(detok, true);
            std::string out;
            sd.push(5, out); // 0xF0 0x9F
            CHECK(out.empty());
            CHECK(sd.has_pending_bytes());
            sd.push(6, out); // 0x9A
            CHECK(out.empty());
            CHECK(sd.has_pending_bytes());
            sd.push(7, out); // 0x80 -> completes 🚀
            CHECK(out == "\xF0\x9F\x9A\x80");
            CHECK(!sd.has_pending_bytes());
            sd.finish(out);
            CHECK(out == "\xF0\x9F\x9A\x80");
            printf("[PASS] streaming_multibyte_holdback\n");
        }

        // Special tokens skipped vs preserved, and non-special added tokens (<think>, <custom_tag>) always preserved
        check_stream_parity(detok, {100, 102, 0, 103, 104, 105, 106, 101}, true,
                            "<think>Hello</think><tool_call><custom_tag>", "special_skip_true");
        check_stream_parity(detok, {100, 102, 0, 103, 105, 101}, false,
                            "<|im_start|><think>Hello</think><|audio_start|><|im_end|>", "special_skip_false");

        // Split UTF-8 with special=True token skipped in the middle vs special=False added token in the middle
        // Matches HuggingFace tokenizers 0.22.2:
        //  [5(F0 9F), 100(special=True), 6(9A)] with skip=True -> single U+FFFD
        //  [5(F0 9F), 100(special=True), 6(9A)] with skip=False -> U+FFFD + "<|im_start|>" + U+FFFD
        //  [5(F0 9F), 102(<think>, special=False), 6(9A)] with skip=True -> U+FFFD + "<think>" + U+FFFD
        check_stream_parity(detok, {5, 100, 6}, true, "\xEF\xBF\xBD", "split_with_skipped_special");
        check_stream_parity(detok, {5, 100, 6}, false, "\xEF\xBF\xBD<|im_start|>\xEF\xBF\xBD", "split_with_kept_special");
        check_stream_parity(detok, {5, 102, 6}, true, "\xEF\xBF\xBD<think>\xEF\xBF\xBD", "split_with_nonspecial_added");

        // Exact WHATWG / Rust String::from_utf8_lossy maximal-subpart ill-formed UTF-8 rules:
        check_stream_parity(detok, {5}, true, "\xEF\xBF\xBD", "illformed_F0_9F");
        check_stream_parity(detok, {5, 10}, true, "\xEF\xBF\xBD" "A", "illformed_F0_9F_41");
        check_stream_parity(detok, {11, 7, 7}, true, "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD", "illformed_overlong_E0_80_80");
        check_stream_parity(detok, {12, 13, 7}, true, "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD", "illformed_surrogate_ED_A0_80");
        check_stream_parity(detok, {14, 15, 7, 7}, true, "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD", "illformed_gt10FFFF_F4_90_80_80");
        check_stream_parity(detok, {16, 10}, true, "\xEF\xBF\xBD" "A", "illformed_C2_41");

        // Embedded NUL byte (0x00) and out-of-range / negative IDs
        std::string expected_nul("A");
        expected_nul.push_back('\0');
        expected_nul.push_back('A');
        check_stream_parity(detok, {10, 17, 10, -1, 999999}, true, expected_nul, "embedded_nul_and_oob_ids");
    }

    // 2. Non-ByteLevel decoder and clean_up_tokenization_spaces=true must cleanly refuse (so server falls back)
    {
        BpeDetokenizer detok;
        std::string err;
        std::string metaspace_json = R"JSON({
            "decoder": {"type": "Metaspace", "replacement": "\u2581"},
            "model": {"type": "BPE", "vocab": {"a": 0}}
        })JSON";
        bool ok_meta = detok.load_from_json_strings(metaspace_json, {}, &err);
        CHECK(!ok_meta);
        CHECK(!detok.loaded());
        printf("[PASS] refuses_non_bytelevel_decoder (%s)\n", err.c_str());

        err.clear();
        std::string cleanup_cfg = R"JSON({"clean_up_tokenization_spaces": true})JSON";
        bool ok_cleanup = detok.load_from_json_strings(make_synthetic_tokenizer_json(), cleanup_cfg, &err);
        CHECK(!ok_cleanup);
        CHECK(!detok.loaded());
        printf("[PASS] refuses_clean_up_spaces_true (%s)\n", err.c_str());
    }

    // 3. Optional real checkpoint mode: `./test_bpe_detokenizer /path/to/tokenizer.json [/path/to/tokenizer_config.json]`
    if (argc >= 2) {
        BpeDetokenizer detok;
        std::string err;
        auto t0 = std::chrono::steady_clock::now();
        std::string cfg_path = (argc >= 3) ? std::string(argv[2]) : std::string();
        bool ok = detok.load_from_files(argv[1], cfg_path, &err);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!ok) {
            fprintf(stderr, "[FAIL] Real tokenizer load failed: %s\n", err.c_str());
            return 1;
        }
        printf("[PASS] Loaded real tokenizer.json: vocab=%zu, added=%zu, mem=%.2f MB, load=%.1f ms\n",
               detok.vocab_size(), detok.added_token_count(),
               static_cast<double>(detok.memory_bytes()) / (1024.0 * 1024.0), ms);

        // Benchmark throughput on 100K incremental token pushes
        std::vector<int32_t> sample_ids = {9419, 11, 1814, 0, 10838, 248, 222, 9008, 242, 98};
        auto b0 = std::chrono::steady_clock::now();
        std::string out;
        out.reserve(512 * 1024);
        BpeDetokenizer::StreamDecoder sd(detok, true);
        constexpr int kLoops = 10000;
        for (int i = 0; i < kLoops; ++i) {
            sd.push_batch(sample_ids.data(), sample_ids.size(), out);
        }
        sd.finish(out);
        double detok_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - b0).count();
        size_t total_toks = sample_ids.size() * kLoops;
        printf("[PERF] Incremental detokenize: %zu tokens in %.2f ms (%.1f ns/tok, %.2f M tok/s, %zu bytes)\n",
               total_toks, detok_us / 1000.0, (detok_us * 1000.0) / static_cast<double>(total_toks),
               static_cast<double>(total_toks) / detok_us, out.size());
    }

    printf("ALL BPE DETOKENIZER TESTS PASSED!\n");
    return 0;
}
