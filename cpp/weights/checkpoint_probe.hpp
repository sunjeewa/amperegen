#pragma once
// Shared strict checkpoint config.json qualification helpers for curated amperegen weight plugins.

#include "model/model_config.hpp"
#include "server/json_lite.hpp"

#include <fstream>
#include <sstream>
#include <string>

namespace amperegen {

inline bool read_checkpoint_config_json(
    const std::string& model_dir,
    Json& out_cfg,
    std::string& reason
) {
    std::string cfg_path = model_dir + "/config.json";
    std::ifstream f(cfg_path);
    if (!f) {
        reason = "missing config.json";
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    try {
        out_cfg = Json::parse(ss.str());
    } catch (const std::exception& e) {
        reason = std::string("failed to parse config.json: ") + e.what();
        return false;
    }
    if (!out_cfg.is_object()) {
        reason = "config.json root is not a JSON object";
        return false;
    }
    return true;
}

// Verify checkpoint geometry strictly matches Qwen3.8-27B (Qwen3.5 architecture: 64L, H=5120, I=17408, V=248320).
inline bool verify_qwen35_27b_architecture(const Json& cfg, std::string& reason) {
    std::string model_type = cfg.contains("model_type") ? cfg.at("model_type").as_string_or("") : "";
    if (model_type != "qwen3_5" && model_type != "qwen3_5_text") {
        reason = "unsupported model_type '" + model_type + "' (expected 'qwen3_5')";
        return false;
    }
    const Json& tc = (cfg.contains("text_config") && cfg.at("text_config").is_object())
        ? cfg.at("text_config")
        : cfg;

    auto expect_int = [&](const char* key, int expected) -> bool {
        if (!tc.contains(key) || !tc.at(key).is_number()) {
            reason = std::string("config.json missing text_config.") + key;
            return false;
        }
        int64_t got = tc.at(key).as_int(-1);
        if (got != expected) {
            reason = std::string("text_config.") + key + "=" + std::to_string(got) +
                     " (expected " + std::to_string(expected) + ")";
            return false;
        }
        return true;
    };

    if (!expect_int("hidden_size", Qwen35Config::HIDDEN_SIZE)) return false;
    if (!expect_int("num_hidden_layers", Qwen35Config::NUM_LAYERS)) return false;
    if (!expect_int("intermediate_size", Qwen35Config::INTERMEDIATE_SIZE)) return false;
    if (!expect_int("vocab_size", Qwen35Config::VOCAB_SIZE)) return false;
    if (!expect_int("num_attention_heads", Qwen35Config::ATTN_Q_HEADS)) return false;
    if (!expect_int("num_key_value_heads", Qwen35Config::ATTN_KV_HEADS)) return false;
    if (!expect_int("head_dim", Qwen35Config::ATTN_HEAD_DIM)) return false;
    if (!expect_int("linear_num_key_heads", Qwen35Config::GDN_K_HEADS)) return false;
    if (!expect_int("linear_num_value_heads", Qwen35Config::GDN_V_HEADS)) return false;
    return true;
}

} // namespace amperegen
