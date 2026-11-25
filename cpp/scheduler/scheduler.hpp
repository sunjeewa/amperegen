#pragma once
#include "model/dflash_model.hpp"
#include "model/qwen35_model.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace amperegen {

struct SamplingParams {
    float temperature = 1.0f;
    int top_k = 20;
    float top_p = 0.95f;
    int max_tokens = 2048;
    uint64_t seed = 0;
    bool has_seed = false;
    std::vector<std::string> stop_strings;

    bool is_greedy() const { return temperature <= 1e-5f || top_k == 1; }
};

struct TokenEvent {
    int32_t token_id = -1;
    bool finished = false;
    std::string finish_reason; // "stop", "length", "cancelled", "error"
};

struct Request {
    uint64_t id = 0;
    std::vector<int32_t> prompt_ids;
    SamplingParams sampling;

    // Runtime state (owned by scheduler thread once admitted)
    int state_slot = -1;
    std::vector<int32_t> kv_blocks;
    int prefill_offset = 0;
    int seq_len = 0;          // total tokens currently committed in KV cache
    int32_t anchor_token = 0; // last emitted token (next step's anchor)
    int generated_tokens = 0;
    uint64_t rng_state = 123456789ULL;
    bool prefill_done = false;
    bool active = false;

    // Per-request telemetry timers & counters (PP, TG, TTFT, spec accept)
    std::chrono::steady_clock::time_point t_submit{};
    std::chrono::steady_clock::time_point t_first_token{};
    bool has_first_token = false;
    double prefill_sec = 0.0;
    uint64_t spec_steps = 0;
    uint64_t spec_committed = 0;
    uint64_t spec_correct_drafts = 0;
    uint64_t spec_proposed_drafts = 0;

    // Thread-safe streaming output queue to HTTP worker thread
    std::mutex out_mu;
    std::condition_variable out_cv;
    std::deque<TokenEvent> out_queue;
    std::atomic<bool> cancelled{false};

    void push_token(int32_t tok) {
        std::lock_guard<std::mutex> lk(out_mu);
        out_queue.push_back(TokenEvent{tok, false, ""});
        out_cv.notify_one();
    }
    void push_finish(const std::string& reason) {
        std::lock_guard<std::mutex> lk(out_mu);
        out_queue.push_back(TokenEvent{-1, true, reason});
        out_cv.notify_one();
    }
    bool pop_event(TokenEvent& ev) {
        std::unique_lock<std::mutex> lk(out_mu);
        out_cv.wait(lk, [&] { return !out_queue.empty(); });
        ev = out_queue.front();
        out_queue.pop_front();
        return true;
    }
    void pop_events_batch(std::vector<TokenEvent>& out) {
        out.clear();
        std::unique_lock<std::mutex> lk(out_mu);
        out_cv.wait(lk, [&] { return !out_queue.empty(); });
        while (!out_queue.empty()) {
            out.push_back(std::move(out_queue.front()));
            out_queue.pop_front();
        }
    }
};

struct EngineConfig {
    std::string model_dir = "./models/Qwen3.8-27B-FP8";
    std::string draft_dir = "./models/Qwen3.8-27B-DFlash2";
    std::string served_model_name = "Qwen3.8-27B-FP8";
    int context_length = 200000;
    bool allow_auto_truncate = true;
    int max_running_requests = 16;
    int chunked_prefill_size = 8192;
    bool enable_dflash = true;
    int block_size = 8;        // 8 = checkpoint native (1 anchor + 7 draft); 6 = SGLang prod
    bool draft_fp8 = true;     // 100% lossless for target!
    bool fp8_lm_draft = true;  // 100% lossless for target!
    bool enable_cuda_graph = true;
    size_t kv_pool_bytes = 18ULL * 1024 * 1024 * 1024; // ~589K FP8 KV tokens (~3x SGLang!) with 4.5 GB VRAM headroom!
    int decode_log_interval = 40; // SGLang-style periodic decode telemetry log interval (steps)
};

struct EngineMetrics {
    std::atomic<int> running_requests{0};
    std::atomic<int> waiting_requests{0};
    std::atomic<int> kv_blocks_used{0};
    std::atomic<int> kv_blocks_total{0};
    std::atomic<uint64_t> total_prompt_tokens{0};
    std::atomic<uint64_t> total_cached_tokens{0};
    std::atomic<uint64_t> total_gen_tokens{0};
    std::atomic<uint64_t> total_requests_completed{0};
    std::atomic<uint64_t> total_spec_steps{0};
    std::atomic<uint64_t> total_spec_committed{0};
    std::atomic<uint64_t> total_spec_correct_drafts{0};
    std::atomic<uint64_t> total_spec_proposed_drafts{0};
    std::atomic<double> recent_prefill_tps{0.0};
    std::atomic<double> recent_decode_tps{0.0};
    std::atomic<double> recent_accept_len{0.0};
    std::atomic<double> recent_accept_rate{0.0};
    std::atomic<double> last_ttft_sec{0.0};
    std::atomic<double> total_ttft_sec{0.0};
    std::atomic<uint64_t> total_ttft_count{0};
    std::atomic<double> total_e2e_sec{0.0};
};

class Scheduler {
public:
    EngineConfig cfg;
    EngineMetrics metrics;
    Qwen35Model target;
    DFlash2Model draft;

    void init(const EngineConfig& config);
    void start();
    void stop();

    // Submit a new request (returns false if queue saturated)
    std::shared_ptr<Request> submit(std::vector<int32_t> prompt_ids, const SamplingParams& sp);

private:
    std::thread worker_;
    std::atomic<bool> stop_flag_{false};
    std::atomic<uint64_t> next_req_id_{1};

    std::mutex q_mu_;
    std::condition_variable q_cv_;
    std::deque<std::shared_ptr<Request>> waiting_;

    std::vector<std::shared_ptr<Request>> running_;
    std::vector<int32_t> free_kv_blocks_;
    std::vector<int32_t> free_state_slots_;

    // Static device metadata buffers for prefill & CUDA-graphed batched decode/verify
    static constexpr int MAX_BS = 16;
    int max_blocks_per_seq_ = 0;

    int32_t* d_tok_ = nullptr;
    int32_t* d_pos_ = nullptr;
    int32_t* d_slots_ = nullptr;
    int32_t* d_bt_ = nullptr;
    int32_t* d_cu_ = nullptr;
    int32_t* d_kvlen_ = nullptr;
    int32_t* d_sslots_ = nullptr;
    int32_t* d_last_idx_ = nullptr;
    int32_t* d_anchor_ = nullptr;
    int32_t* d_prefix_len_ = nullptr;
    int32_t* d_draft_ids_ = nullptr;
    int32_t* d_draft_pos_ = nullptr;
    int32_t* d_candidates_ = nullptr;
    int32_t* d_top1_ = nullptr;
    int32_t* d_accept_ = nullptr;
    int32_t* d_commit_ = nullptr;
    int32_t* d_bonus_ = nullptr;
    int32_t* d_out_toks_ = nullptr;
    int32_t* d_ctx_slots_ = nullptr;
    float* d_temps_ = nullptr;
    int32_t* d_topks_ = nullptr;
    float* d_topps_ = nullptr;
    uint64_t* d_seeds_ = nullptr;
    uint8_t* d_greedy_mask_ = nullptr;
    float* d_uniforms_ = nullptr;

    // Pinned host staging buffers for zero-overhead step I/O
    int32_t* h_bt_ = nullptr;
    int32_t* h_slots_ = nullptr;
    int32_t* h_meta_ = nullptr;
    int32_t* h_out_ = nullptr;

    // Per-batch-size CUDA graphs (key = B for greedy spec or non-spec decode)
    struct GraphEntry {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
    };
    std::unordered_map<int, GraphEntry> greedy_spec_graphs_;
    std::unordered_map<int, GraphEntry> sampling_spec_graphs_;
    std::unordered_map<int, GraphEntry> nospec_graphs_;

    // Windowed decode telemetry accumulators (reset every decode_log_interval steps)
    int decode_log_steps_ = 0;
    int64_t win_decode_tokens_ = 0;
    int64_t win_spec_steps_ = 0;
    int64_t win_spec_committed_ = 0;
    int64_t win_spec_correct_ = 0;
    int64_t win_spec_proposed_ = 0;
    double win_decode_sec_ = 0.0;

    void loop();
    void admit_waiting();
    void cleanup_cancelled();
    void finish_request(const std::shared_ptr<Request>& req, const std::string& reason);
    void release_request(const std::shared_ptr<Request>& req);
    bool ensure_kv_blocks(const std::shared_ptr<Request>& req, int needed_tokens);
    void trim_unused_kv_blocks(const std::shared_ptr<Request>& req, int actual_tokens);

    void step_prefill_one(const std::shared_ptr<Request>& req);
    void step_decode_batch(const std::vector<std::shared_ptr<Request>>& batch);
};

} // namespace amperegen
