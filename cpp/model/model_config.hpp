#pragma once
#include <cstdint>

namespace amperegen {

// Static compile-time architecture constants for Qwen3.8-27B-FP8 + DFlash2
struct Qwen35Config {
    static constexpr int VOCAB_SIZE = 248320;
    static constexpr int HIDDEN_SIZE = 5120;
    static constexpr int INTERMEDIATE_SIZE = 17408;
    static constexpr int NUM_LAYERS = 64;
    static constexpr int FULL_ATTN_INTERVAL = 4;
    static constexpr int NUM_FULL_ATTN_LAYERS = 16; // indices 3, 7, ..., 63
    static constexpr int NUM_GDN_LAYERS = 48;       // all other indices

    // Full attention (GQA 24:4, head_dim 256, output gate, partial NeoX RoPE 64/256)
    static constexpr int ATTN_Q_HEADS = 24;
    static constexpr int ATTN_KV_HEADS = 4;
    static constexpr int ATTN_GQA_RATIO = ATTN_Q_HEADS / ATTN_KV_HEADS; // 6
    static constexpr int ATTN_HEAD_DIM = 256;
    static constexpr int ATTN_Q_DIM = ATTN_Q_HEADS * ATTN_HEAD_DIM;     // 6144
    static constexpr int ATTN_QGATE_DIM = ATTN_Q_DIM * 2;               // 12288
    static constexpr int ATTN_KV_DIM = ATTN_KV_HEADS * ATTN_HEAD_DIM;   // 1024
    static constexpr int ATTN_QKV_DIM = ATTN_QGATE_DIM + 2 * ATTN_KV_DIM; // 14336
    static constexpr int ATTN_ROTARY_DIM = 64;
    static constexpr float ROPE_THETA = 10000000.0f;
    static constexpr float RMS_EPS = 1e-6f;

    // Linear attention (Gated DeltaNet: 16 K heads, 48 V heads, head_dim 128, conv k=4)
    static constexpr int GDN_K_HEADS = 16;
    static constexpr int GDN_V_HEADS = 48;
    static constexpr int GDN_GQA_RATIO = GDN_V_HEADS / GDN_K_HEADS;     // 3
    static constexpr int GDN_HEAD_DIM = 128;
    static constexpr int GDN_Q_DIM = GDN_K_HEADS * GDN_HEAD_DIM;        // 2048
    static constexpr int GDN_K_DIM = GDN_K_HEADS * GDN_HEAD_DIM;        // 2048
    static constexpr int GDN_V_DIM = GDN_V_HEADS * GDN_HEAD_DIM;        // 6144
    static constexpr int GDN_Z_DIM = GDN_V_HEADS * GDN_HEAD_DIM;        // 6144
    static constexpr int GDN_CONV_DIM = GDN_Q_DIM + GDN_K_DIM + GDN_V_DIM; // 10240
    static constexpr int GDN_QKVZ_DIM = GDN_CONV_DIM + GDN_Z_DIM;       // 16384
    static constexpr int GDN_BA_DIM = GDN_V_HEADS * 2;                  // 96
    static constexpr int GDN_CONV_K = 4;

    // KV cache paging
    static constexpr int KV_BLOCK_SIZE = 16;
    static constexpr int MAX_POSITION_EMBEDDINGS = 262144;
    static constexpr int FP8_BLOCK = 128;

    // Special token IDs (from generation_config.json)
    static constexpr int EOS_TOKEN_0 = 248044;
    static constexpr int EOS_TOKEN_1 = 248046;
    static constexpr int BOS_TOKEN = 248044;

    // DFlash2 aux-hidden capture target layer indices (post-residual output of these layers)
    static constexpr int NUM_CAPTURE_LAYERS = 5;
    static constexpr int CAPTURE_LAYERS[NUM_CAPTURE_LAYERS] = {5, 19, 33, 47, 61};

    static constexpr bool is_full_attn(int layer_idx) {
        return (layer_idx % FULL_ATTN_INTERVAL) == (FULL_ATTN_INTERVAL - 1);
    }
    static constexpr int full_attn_idx(int layer_idx) {
        return layer_idx / FULL_ATTN_INTERVAL;
    }
    static constexpr int gdn_idx(int layer_idx) {
        return layer_idx - (layer_idx / FULL_ATTN_INTERVAL);
    }
    static constexpr int capture_slot(int layer_idx) {
        for (int i = 0; i < NUM_CAPTURE_LAYERS; ++i) {
            if (CAPTURE_LAYERS[i] == layer_idx) return i;
        }
        return -1;
    }
};

struct DFlash2Config {
    static constexpr int NUM_LAYERS = 5;
    static constexpr int HIDDEN_SIZE = 5120;
    static constexpr int INTERMEDIATE_SIZE = 17408;
    static constexpr int NUM_Q_HEADS = 32;
    static constexpr int NUM_KV_HEADS = 8;
    static constexpr int GQA_RATIO = NUM_Q_HEADS / NUM_KV_HEADS; // 4
    static constexpr int HEAD_DIM = 128;
    static constexpr int Q_DIM = NUM_Q_HEADS * HEAD_DIM;         // 4096
    static constexpr int KV_DIM = NUM_KV_HEADS * HEAD_DIM;       // 1024
    static constexpr int QKV_DIM = Q_DIM + 2 * KV_DIM;           // 6144
    static constexpr int SLIDING_WINDOW = 2048;
    static constexpr int DEFAULT_BLOCK_SIZE = 8; // Checkpoint native block_size=8 (1 anchor + 7 draft)
    static constexpr int MAX_BLOCK_SIZE = 8;
    static constexpr int MASK_TOKEN_ID = 248070;
    static constexpr int SELECTOR_RANK = 256;
    static constexpr int SELECTOR_TOP_K = 16;
    static constexpr int CONV_GROUP_SIZE = 16;
    static constexpr int CONV_NUM_GROUPS = HIDDEN_SIZE / CONV_GROUP_SIZE; // 320
    static constexpr int CONV_TAPS = 2;
    static constexpr int CONV_PROJ_OUT = 2 * CONV_TAPS * CONV_NUM_GROUPS; // 1280
    static constexpr int FC_IN_DIM = Qwen35Config::NUM_CAPTURE_LAYERS * HIDDEN_SIZE; // 25600
    static constexpr float ROPE_THETA = 10000000.0f;
    static constexpr float RMS_EPS = 1e-6f;
};

} // namespace amperegen
