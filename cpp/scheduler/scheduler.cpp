#include "scheduler/scheduler.hpp"
#include "kernels/kernels.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>

namespace amperegen {

namespace {

inline uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

inline float next_uniform(uint64_t& state) {
    return (float)(splitmix64(state) >> 40) * (1.0f / 16777216.0f);
}

inline void atomic_add_double(std::atomic<double>& target, double delta) {
    double cur = target.load(std::memory_order_relaxed);
    while (!target.compare_exchange_weak(cur, cur + delta, std::memory_order_relaxed)) {}
}

inline std::string local_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf {};
    localtime_r(&tt, &tm_buf);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_buf);
    return buf;
}

} // namespace

void Scheduler::init(const EngineConfig& config) {
    cfg = config;
    target.init_context();

    load_target_weights(cfg.model_dir, target.weights, cfg.enable_dflash && cfg.fp8_lm_draft, target.ctx.stream);

    int max_slots = cfg.max_running_requests + 2;
    max_blocks_per_seq_ = (cfg.context_length + Qwen35Config::KV_BLOCK_SIZE - 1) / Qwen35Config::KV_BLOCK_SIZE + 8;

    target.init_workspace_and_pools(
        cfg.chunked_prefill_size,
        max_slots,
        cfg.max_running_requests,
        cfg.block_size,
        cfg.kv_pool_bytes);

    if (cfg.enable_dflash) {
        draft.init(cfg.draft_dir, cfg.draft_fp8, max_slots, cfg.max_running_requests, cfg.block_size, target.ctx.stream);
    }

    // Initialize free lists
    free_kv_blocks_.resize(target.pools.num_kv_blocks);
    for (int i = 0; i < target.pools.num_kv_blocks; ++i) {
        free_kv_blocks_[i] = target.pools.num_kv_blocks - 1 - i;
    }
    free_state_slots_.resize(max_slots);
    for (int i = 0; i < max_slots; ++i) {
        free_state_slots_[i] = max_slots - 1 - i;
    }
    metrics.kv_blocks_total.store(target.pools.num_kv_blocks);
    metrics.kv_blocks_used.store(0);

    // Allocate static device & pinned host metadata buffers
    int max_tok = std::max(cfg.chunked_prefill_size, MAX_BS * cfg.block_size);
    CUDA_CHECK(cudaMalloc(&d_tok_, (size_t)max_tok * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_pos_, (size_t)max_tok * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_slots_, (size_t)max_tok * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_bt_, (size_t)MAX_BS * max_blocks_per_seq_ * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_cu_, (size_t)(MAX_BS + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_kvlen_, (size_t)MAX_BS * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_sslots_, (size_t)MAX_BS * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_last_idx_, (size_t)MAX_BS * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_anchor_, (size_t)MAX_BS * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_prefix_len_, (size_t)MAX_BS * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_draft_ids_, (size_t)MAX_BS * cfg.block_size * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_draft_pos_, (size_t)MAX_BS * cfg.block_size * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_candidates_, (size_t)MAX_BS * cfg.block_size * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_top1_, (size_t)MAX_BS * cfg.block_size * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_accept_, (size_t)MAX_BS * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_commit_, (size_t)MAX_BS * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_bonus_, (size_t)MAX_BS * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_out_toks_, (size_t)MAX_BS * cfg.block_size * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_ctx_slots_, (size_t)max_tok * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_temps_, (size_t)MAX_BS * cfg.block_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topks_, (size_t)MAX_BS * cfg.block_size * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_topps_, (size_t)MAX_BS * cfg.block_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_seeds_, (size_t)MAX_BS * cfg.block_size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_greedy_mask_, (size_t)MAX_BS * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_uniforms_, (size_t)MAX_BS * cfg.block_size * sizeof(float)));

    CUDA_CHECK(cudaMallocHost(&h_bt_, (size_t)MAX_BS * max_blocks_per_seq_ * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocHost(&h_slots_, (size_t)max_tok * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocHost(&h_meta_, (size_t)max_tok * 4 * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocHost(&h_out_, (size_t)MAX_BS * (cfg.block_size + 4) * sizeof(int32_t)));

    fprintf(stderr, "[scheduler] Ready: max_running=%d ctx=%d dflash=%d block_size=%d kv_blocks=%d (%d tokens)\n",
            cfg.max_running_requests, cfg.context_length, (int)cfg.enable_dflash, cfg.block_size,
            target.pools.num_kv_blocks, target.pools.num_kv_blocks * Qwen35Config::KV_BLOCK_SIZE);
}

void Scheduler::start() {
    stop_flag_.store(false);
    worker_ = std::thread([this] { loop(); });
}

void Scheduler::stop() {
    stop_flag_.store(true);
    q_cv_.notify_all();
    if (worker_.joinable()) worker_.join();

    for (auto& [k, g] : greedy_spec_graphs_) {
        if (g.exec) cudaGraphExecDestroy(g.exec);
        if (g.graph) cudaGraphDestroy(g.graph);
    }
    greedy_spec_graphs_.clear();
    for (auto& [k, g] : sampling_spec_graphs_) {
        if (g.exec) cudaGraphExecDestroy(g.exec);
        if (g.graph) cudaGraphDestroy(g.graph);
    }
    sampling_spec_graphs_.clear();
    for (auto& [k, g] : nospec_graphs_) {
        if (g.exec) cudaGraphExecDestroy(g.exec);
        if (g.graph) cudaGraphDestroy(g.graph);
    }
    nospec_graphs_.clear();

    if (h_out_) { cudaFreeHost(h_out_); h_out_ = nullptr; }
    if (h_meta_) { cudaFreeHost(h_meta_); h_meta_ = nullptr; }
    if (h_slots_) { cudaFreeHost(h_slots_); h_slots_ = nullptr; }
    if (h_bt_) { cudaFreeHost(h_bt_); h_bt_ = nullptr; }

    auto dfree = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
    dfree(d_tok_); dfree(d_pos_); dfree(d_slots_); dfree(d_bt_); dfree(d_cu_);
    dfree(d_kvlen_); dfree(d_sslots_); dfree(d_last_idx_); dfree(d_anchor_);
    dfree(d_prefix_len_); dfree(d_draft_ids_); dfree(d_draft_pos_); dfree(d_candidates_);
    dfree(d_top1_); dfree(d_accept_); dfree(d_commit_); dfree(d_bonus_);
    dfree(d_out_toks_); dfree(d_ctx_slots_); dfree(d_temps_); dfree(d_topks_);
    dfree(d_topps_); dfree(d_seeds_); dfree(d_greedy_mask_); dfree(d_uniforms_);

    if (cfg.enable_dflash) draft.destroy();
    target.destroy();
}

std::shared_ptr<Request> Scheduler::submit(std::vector<int32_t> prompt_ids, const SamplingParams& sp) {
    auto req = std::make_shared<Request>();
    req->id = next_req_id_.fetch_add(1);
    req->prompt_ids = std::move(prompt_ids);
    req->sampling = sp;
    req->t_submit = std::chrono::steady_clock::now();
    req->rng_state = sp.has_seed ? sp.seed : (req->id * 0x9e3779b97f4a7c15ULL ^ (uint64_t)req->t_submit.time_since_epoch().count());
    {
        std::lock_guard<std::mutex> lk(q_mu_);
        waiting_.push_back(req);
        metrics.waiting_requests.store((int)waiting_.size());
    }
    q_cv_.notify_one();
    return req;
}

bool Scheduler::ensure_kv_blocks(const std::shared_ptr<Request>& req, int needed_tokens) {
    int needed_blocks = (needed_tokens + Qwen35Config::KV_BLOCK_SIZE - 1) / Qwen35Config::KV_BLOCK_SIZE;
    int have = (int)req->kv_blocks.size();
    if (needed_blocks <= have) return true;
    int delta = needed_blocks - have;
    if ((int)free_kv_blocks_.size() < delta) return false;
    for (int i = 0; i < delta; ++i) {
        req->kv_blocks.push_back(free_kv_blocks_.back());
        free_kv_blocks_.pop_back();
    }
    metrics.kv_blocks_used.store(target.pools.num_kv_blocks - (int)free_kv_blocks_.size());
    return true;
}

void Scheduler::trim_unused_kv_blocks(const std::shared_ptr<Request>& req, int actual_tokens) {
    int keep = (actual_tokens + Qwen35Config::KV_BLOCK_SIZE - 1) / Qwen35Config::KV_BLOCK_SIZE;
    while ((int)req->kv_blocks.size() > keep) {
        free_kv_blocks_.push_back(req->kv_blocks.back());
        req->kv_blocks.pop_back();
    }
    metrics.kv_blocks_used.store(target.pools.num_kv_blocks - (int)free_kv_blocks_.size());
}

void Scheduler::finish_request(const std::shared_ptr<Request>& req, const std::string& reason) {
    auto now = std::chrono::steady_clock::now();
    int prompt_toks = (int)req->prompt_ids.size();
    int gen_toks = req->generated_tokens;
    double pp_sec = req->prefill_sec;
    double pp_tps = (pp_sec > 1e-6) ? ((double)prompt_toks / pp_sec) : 0.0;
    double ttft = req->has_first_token
        ? std::chrono::duration<double>(req->t_first_token - req->t_submit).count()
        : std::chrono::duration<double>(now - req->t_submit).count();
    double tg_sec = req->has_first_token
        ? std::chrono::duration<double>(now - req->t_first_token).count()
        : 0.0;
    // First token is produced at the end of prefill; remaining (gen_toks - 1) tokens are pure decode (TG)
    double tg_tps = (gen_toks > 1 && tg_sec > 1e-6)
        ? ((double)(gen_toks - 1) / tg_sec)
        : 0.0;
    double e2e_sec = std::chrono::duration<double>(now - req->t_submit).count();

    double acc_len = req->spec_steps > 0
        ? ((double)req->spec_committed / (double)req->spec_steps)
        : 1.0;
    double acc_rate = req->spec_proposed_drafts > 0
        ? ((double)req->spec_correct_drafts / (double)req->spec_proposed_drafts)
        : 0.0;

    int kv_used = metrics.kv_blocks_used.load();
    int kv_tot = std::max(1, metrics.kv_blocks_total.load());
    double kv_pct = 100.0 * (double)kv_used / (double)kv_tot;

    metrics.total_requests_completed.fetch_add(1);
    atomic_add_double(metrics.total_e2e_sec, e2e_sec);
    if (tg_tps > 0.0) {
        metrics.recent_decode_tps.store(tg_tps);
    }
    if (req->spec_steps > 0) {
        metrics.recent_accept_len.store(acc_len);
        metrics.recent_accept_rate.store(acc_rate);
    }

    std::string ts = local_timestamp();
    fprintf(stderr,
            "[%s] Done req=#%llu finish=%s | PP: %d tok in %.2fs (%.1f tok/s) | TG: %d tok in %.2fs (%.1f tok/s) | accept: %.2f tok/step (%.1f%%) | TTFT: %.2fs | KV: %.1f%%\n",
            ts.c_str(),
            (unsigned long long)req->id,
            reason.c_str(),
            prompt_toks, pp_sec, pp_tps,
            gen_toks, tg_sec, tg_tps,
            acc_len, acc_rate * 100.0,
            ttft, kv_pct);
    fflush(stderr);

    req->push_finish(reason);
    release_request(req);
}

void Scheduler::release_request(const std::shared_ptr<Request>& req) {
    for (int32_t blk : req->kv_blocks) free_kv_blocks_.push_back(blk);
    req->kv_blocks.clear();
    if (req->state_slot >= 0) {
        free_state_slots_.push_back(req->state_slot);
        req->state_slot = -1;
    }
    req->active = false;
    metrics.kv_blocks_used.store(target.pools.num_kv_blocks - (int)free_kv_blocks_.size());
}

void Scheduler::cleanup_cancelled() {
    auto it = running_.begin();
    while (it != running_.end()) {
        if ((*it)->cancelled.load()) {
            finish_request(*it, "cancelled");
            it = running_.erase(it);
        } else {
            ++it;
        }
    }
    metrics.running_requests.store((int)running_.size());
}

void Scheduler::admit_waiting() {
    std::lock_guard<std::mutex> lk(q_mu_);
    while (!waiting_.empty() && (int)running_.size() < cfg.max_running_requests && !free_state_slots_.empty()) {
        auto req = waiting_.front();
        if (req->cancelled.load()) {
            waiting_.pop_front();
            req->push_finish("cancelled");
            continue;
        }
        int prompt_blocks = ((int)req->prompt_ids.size() + cfg.block_size + Qwen35Config::KV_BLOCK_SIZE - 1) / Qwen35Config::KV_BLOCK_SIZE;
        if ((int)free_kv_blocks_.size() < prompt_blocks) {
            break; // Wait for running requests to free KV blocks
        }
        waiting_.pop_front();
        req->state_slot = free_state_slots_.back();
        free_state_slots_.pop_back();
        req->active = true;
        req->prefill_offset = 0;
        req->seq_len = 0;
        req->generated_tokens = 0;
        req->prefill_done = false;
        target.reset_state_slot(req->state_slot, target.ctx.stream);
        running_.push_back(req);
    }
    metrics.waiting_requests.store((int)waiting_.size());
    metrics.running_requests.store((int)running_.size());
}

void Scheduler::step_prefill_one(const std::shared_ptr<Request>& req) {
    auto t0 = std::chrono::steady_clock::now();
    int prompt_len = (int)req->prompt_ids.size();
    int off = req->prefill_offset;
    int chunk = std::min(cfg.chunked_prefill_size, prompt_len - off);
    bool is_last = (off + chunk == prompt_len);

    if (!ensure_kv_blocks(req, off + chunk)) {
        finish_request(req, "error");
        return;
    }

    for (int i = 0; i < chunk; ++i) {
        int pos = off + i;
        h_meta_[i] = pos;
        int blk_idx = pos / Qwen35Config::KV_BLOCK_SIZE;
        int blk_off = pos % Qwen35Config::KV_BLOCK_SIZE;
        h_slots_[i] = req->kv_blocks[blk_idx] * Qwen35Config::KV_BLOCK_SIZE + blk_off;
    }
    int n_blks = (int)req->kv_blocks.size();
    std::memcpy(h_bt_, req->kv_blocks.data(), (size_t)n_blks * sizeof(int32_t));

    int32_t h_cu[2] = {0, chunk};
    int32_t h_kv = off + chunk;
    int32_t last_i = chunk - 1;
    int32_t sslot = req->state_slot;

    cudaStream_t st = target.ctx.stream;
    CUDA_CHECK(cudaMemcpyAsync(d_tok_, req->prompt_ids.data() + off, chunk * 4, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(d_pos_, h_meta_, chunk * 4, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(d_slots_, h_slots_, chunk * 4, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(d_bt_, h_bt_, n_blks * 4, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(d_cu_, h_cu, 8, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(d_kvlen_, &h_kv, 4, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(d_sslots_, &sslot, 4, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(d_last_idx_, &last_i, 4, cudaMemcpyHostToDevice, st));

    bool need_aux = cfg.enable_dflash && (off + chunk > prompt_len - DFlash2Config::SLIDING_WINDOW);
    target.forward_prefill(
        d_tok_, d_pos_, d_slots_, d_bt_, d_cu_, d_kvlen_, d_sslots_,
        1, chunk, chunk, max_blocks_per_seq_, need_aux,
        is_last ? d_last_idx_ : nullptr,
        is_last ? target.ws.logits : nullptr,
        st);

    if (need_aux) {
        int start_in = std::max(0, (prompt_len - DFlash2Config::SLIDING_WINDOW) - off);
        int n_mat = chunk - start_in;
        std::vector<int32_t> cs(n_mat, sslot);
        CUDA_CHECK(cudaMemcpyAsync(d_ctx_slots_, cs.data(), n_mat * 4, cudaMemcpyHostToDevice, st));
        draft.materialize_context_kv(
            target,
            target.ws.aux_hidden + (size_t)start_in * DFlash2Config::FC_IN_DIM,
            d_pos_ + start_in, d_ctx_slots_, n_mat, st);
    }

    req->prefill_offset = off + chunk;
    req->seq_len = off + chunk;
    metrics.total_prompt_tokens.fetch_add(chunk);

    if (is_last) {
        if (req->sampling.is_greedy()) {
            launch_argmax(target.ws.logits, d_top1_, 1, Qwen35Config::VOCAB_SIZE, st);
        } else {
            float t = req->sampling.temperature;
            int32_t tk = req->sampling.top_k;
            float tp = req->sampling.top_p;
            uint64_t sd = splitmix64(req->rng_state);
            CUDA_CHECK(cudaMemcpyAsync(d_temps_, &t, 4, cudaMemcpyHostToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(d_topks_, &tk, 4, cudaMemcpyHostToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(d_topps_, &tp, 4, cudaMemcpyHostToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(d_seeds_, &sd, 8, cudaMemcpyHostToDevice, st));
            launch_sample_topkp(target.ws.logits, d_temps_, d_topks_, d_topps_, d_seeds_, d_top1_, 1, Qwen35Config::VOCAB_SIZE, st);
        }
        CUDA_CHECK(cudaMemcpyAsync(h_out_, d_top1_, 4, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        req->prefill_sec += dt;
        req->t_first_token = t1;
        req->has_first_token = true;

        double ttft = std::chrono::duration<double>(t1 - req->t_submit).count();
        metrics.last_ttft_sec.store(ttft);
        atomic_add_double(metrics.total_ttft_sec, ttft);
        metrics.total_ttft_count.fetch_add(1);

        double chunk_pp_tps = (dt > 1e-6) ? ((double)chunk / dt) : 0.0;
        double full_pp_tps = (req->prefill_sec > 1e-6) ? ((double)prompt_len / req->prefill_sec) : chunk_pp_tps;
        metrics.recent_prefill_tps.store(full_pp_tps);

        double kv_usage = (double)metrics.kv_blocks_used.load() / std::max(1, metrics.kv_blocks_total.load());
        std::string ts = local_timestamp();
        fprintf(stderr,
                "[%s] Prefill batch, req=#%llu, #new-token: %d (%d/%d), #cached-token: 0, token usage: %.2f, #running-req: %d, #queue-req: %d, PP (input throughput): %.2f tok/s (%.2fs)\n",
                ts.c_str(), (unsigned long long)req->id, chunk, off + chunk, prompt_len,
                kv_usage, metrics.running_requests.load(), metrics.waiting_requests.load(),
                chunk_pp_tps, dt);
        fflush(stderr);

        int32_t tok = h_out_[0];
        req->prefill_done = true;
        req->anchor_token = tok;
        req->generated_tokens = 1;
        metrics.total_gen_tokens.fetch_add(1);
        req->push_token(tok);

        if (tok == Qwen35Config::EOS_TOKEN_0 || tok == Qwen35Config::EOS_TOKEN_1) {
            finish_request(req, "stop");
        } else if (req->generated_tokens >= req->sampling.max_tokens || req->seq_len + 1 >= cfg.context_length) {
            finish_request(req, "length");
        }
    } else {
        CUDA_CHECK(cudaStreamSynchronize(st));
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        req->prefill_sec += dt;
        double chunk_pp_tps = (dt > 1e-6) ? ((double)chunk / dt) : 0.0;
        metrics.recent_prefill_tps.store(chunk_pp_tps);

        double kv_usage = (double)metrics.kv_blocks_used.load() / std::max(1, metrics.kv_blocks_total.load());
        std::string ts = local_timestamp();
        fprintf(stderr,
                "[%s] Prefill batch, req=#%llu, #new-token: %d (%d/%d), #cached-token: 0, token usage: %.2f, #running-req: %d, #queue-req: %d, PP (input throughput): %.2f tok/s (%.2fs)\n",
                ts.c_str(), (unsigned long long)req->id, chunk, off + chunk, prompt_len,
                kv_usage, metrics.running_requests.load(), metrics.waiting_requests.load(),
                chunk_pp_tps, dt);
        fflush(stderr);
    }
}

void Scheduler::step_decode_batch(const std::vector<std::shared_ptr<Request>>& batch) {
    int B = (int)batch.size();
    if (B == 0) return;
    cudaStream_t st = target.ctx.stream;
    auto t0 = std::chrono::steady_clock::now();

    bool all_greedy = true;
    for (const auto& r : batch) {
        if (!r->sampling.is_greedy()) { all_greedy = false; break; }
    }

    // Use DFlash2 speculative decoding for ALL batches (greedy, sampling, and mixed!) when enabled!
    bool run_spec = cfg.enable_dflash;
    int q_len = run_spec ? cfg.block_size : 1;

    // Ensure KV blocks for q_len tokens for every request in batch
    for (int b = 0; b < B; ++b) {
        auto& req = batch[b];
        if (!ensure_kv_blocks(req, req->seq_len + q_len)) {
            finish_request(req, "error");
            return;
        }
    }

    // Pack host metadata
    int max_seq_hint = 0;
    std::vector<int32_t> h_anchor(B), h_prefix(B), h_kv(B), h_ss(B), h_tok(B);
    std::vector<int32_t> h_ctx_sl(B * q_len);
    for (int b = 0; b < B; ++b) {
        const auto& req = batch[b];
        h_anchor[b] = req->anchor_token;
        h_tok[b] = req->anchor_token;
        h_prefix[b] = req->seq_len;
        h_kv[b] = req->seq_len + q_len;
        h_ss[b] = req->state_slot;
        max_seq_hint = std::max(max_seq_hint, h_kv[b]);

        int nb = (int)req->kv_blocks.size();
        std::memcpy(h_bt_ + (size_t)b * max_blocks_per_seq_, req->kv_blocks.data(), (size_t)nb * sizeof(int32_t));

        for (int k = 0; k < q_len; ++k) {
            int pos = req->seq_len + k;
            h_meta_[b * q_len + k] = pos;
            h_slots_[b * q_len + k] = req->kv_blocks[pos / Qwen35Config::KV_BLOCK_SIZE] * Qwen35Config::KV_BLOCK_SIZE
                                    + (pos % Qwen35Config::KV_BLOCK_SIZE);
            h_ctx_sl[b * q_len + k] = req->state_slot;
        }
    }

    // Copy metadata H2D
    CUDA_CHECK(cudaMemcpyAsync(d_bt_, h_bt_, (size_t)B * max_blocks_per_seq_ * 4, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(d_slots_, h_slots_, (size_t)B * q_len * 4, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(d_kvlen_, h_kv.data(), B * 4, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(d_sslots_, h_ss.data(), B * 4, cudaMemcpyHostToDevice, st));

    if (run_spec) {
        CUDA_CHECK(cudaMemcpyAsync(d_anchor_, h_anchor.data(), B * 4, cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(d_prefix_len_, h_prefix.data(), B * 4, cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(d_ctx_slots_, h_ctx_sl.data(), (size_t)B * q_len * 4, cudaMemcpyHostToDevice, st));

        if (!all_greedy) {
            std::vector<float> ts(B), tps(B), unifs((size_t)B * (q_len - 1));
            std::vector<int32_t> tks(B);
            std::vector<uint64_t> sds(B);
            std::vector<uint8_t> gm(B);
            for (int b = 0; b < B; ++b) {
                ts[b] = batch[b]->sampling.temperature;
                tks[b] = batch[b]->sampling.top_k;
                tps[b] = batch[b]->sampling.top_p;
                gm[b] = batch[b]->sampling.is_greedy() ? 1 : 0;
                sds[b] = splitmix64(batch[b]->rng_state);
                for (int e = 0; e < q_len - 1; ++e) {
                    unifs[(size_t)b * (q_len - 1) + e] = next_uniform(batch[b]->rng_state);
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(d_temps_, ts.data(), B * 4, cudaMemcpyHostToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(d_topks_, tks.data(), B * 4, cudaMemcpyHostToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(d_topps_, tps.data(), B * 4, cudaMemcpyHostToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(d_seeds_, sds.data(), B * 8, cudaMemcpyHostToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(d_greedy_mask_, gm.data(), B, cudaMemcpyHostToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(d_uniforms_, unifs.data(), unifs.size() * 4, cudaMemcpyHostToDevice, st));
        }

        auto enqueue_spec = [&]() {
            draft.propose_candidates(
                target, d_anchor_, d_prefix_len_, d_sslots_,
                all_greedy ? nullptr : d_temps_,
                all_greedy ? nullptr : d_greedy_mask_,
                all_greedy ? nullptr : d_uniforms_,
                d_draft_ids_, d_draft_pos_, d_candidates_,
                B, q_len, st);
            target.forward_decode_verify(
                d_candidates_, d_draft_pos_, d_slots_, d_bt_, d_kvlen_, d_sslots_,
                B, q_len, max_blocks_per_seq_, 65536,
                /*commit_gdn_inplace=*/false, /*capture_aux_hidden=*/true,
                target.ws.logits, st);
            if (all_greedy) {
                launch_argmax(target.ws.logits, d_top1_, B * q_len, Qwen35Config::VOCAB_SIZE, st);
                launch_dflash_accept_greedy(
                    d_candidates_, d_top1_, d_accept_, d_commit_, d_bonus_, d_out_toks_, nullptr,
                    B, q_len, st);
            } else {
                launch_dflash_accept_sampling(
                    target.ws.logits, d_candidates_, draft.ws.topk_ids, draft.ws.q_rows,
                    d_temps_, d_topks_, d_topps_, d_seeds_,
                    d_accept_, d_commit_, d_bonus_, d_out_toks_,
                    B, q_len, Qwen35Config::VOCAB_SIZE, st);
            }
            target.commit_gdn_verify(d_sslots_, d_commit_, B, q_len, st);
            draft.materialize_context_kv(
                target, target.ws.aux_hidden, d_draft_pos_, d_ctx_slots_, B * q_len, st, d_commit_, q_len);
        };

        auto& gmap = all_greedy ? greedy_spec_graphs_ : sampling_spec_graphs_;
        if (cfg.enable_cuda_graph) {
            auto it = gmap.find(B);
            if (it == gmap.end()) {
                enqueue_spec();
                CUDA_CHECK(cudaStreamSynchronize(st));
                gmap[B] = GraphEntry{nullptr, nullptr};
            } else if (it->second.exec == nullptr) {
                CUDA_CHECK(cudaStreamSynchronize(st));
                CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
                enqueue_spec();
                CUDA_CHECK(cudaStreamEndCapture(st, &it->second.graph));
                CUDA_CHECK(cudaGraphInstantiate(&it->second.exec, it->second.graph, nullptr, nullptr, 0));
                CUDA_CHECK(cudaGraphLaunch(it->second.exec, st));
            } else {
                CUDA_CHECK(cudaGraphLaunch(it->second.exec, st));
            }
        } else {
            enqueue_spec();
        }

        // Copy commit_lens[B], bonus_ids[B], out_tokens[B * q_len] D2H
        CUDA_CHECK(cudaMemcpyAsync(h_out_, d_commit_, B * 4, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(h_out_ + B, d_bonus_, B * 4, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(h_out_ + 2 * B, d_out_toks_, (size_t)B * q_len * 4, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));

        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();

        int total_emitted = 0;
        int total_pushed = 0;
        int total_correct = 0;
        int total_proposed = B * std::max(0, q_len - 1);
        int active_ctx_tokens = 0;
        std::vector<std::pair<std::shared_ptr<Request>, std::string>> to_finish;
        for (int b = 0; b < B; ++b) {
            auto& req = batch[b];
            int c_len = h_out_[b];
            int32_t bonus = h_out_[B + b];
            const int32_t* toks = h_out_ + 2 * B + b * q_len;

            req->seq_len += c_len;
            active_ctx_tokens += req->seq_len;
            trim_unused_kv_blocks(req, req->seq_len);
            req->anchor_token = bonus;
            total_emitted += c_len;
            int correct_drafts = std::max(0, c_len - 1);
            total_correct += correct_drafts;

            req->spec_steps += 1;
            req->spec_committed += (uint64_t)c_len;
            req->spec_correct_drafts += (uint64_t)correct_drafts;
            req->spec_proposed_drafts += (uint64_t)std::max(0, q_len - 1);

            bool finished = false;
            std::string reason;
            for (int k = 0; k < c_len; ++k) {
                int32_t tok = toks[k];
                req->generated_tokens += 1;
                total_pushed += 1;
                req->push_token(tok);
                if (tok == Qwen35Config::EOS_TOKEN_0 || tok == Qwen35Config::EOS_TOKEN_1) {
                    finished = true; reason = "stop"; break;
                }
                if (req->generated_tokens >= req->sampling.max_tokens || req->seq_len + 1 >= cfg.context_length) {
                    finished = true; reason = "length"; break;
                }
            }
            if (finished) {
                to_finish.emplace_back(req, std::move(reason));
            }
        }
        metrics.total_gen_tokens.fetch_add(total_pushed);
        metrics.total_spec_steps.fetch_add(B);
        metrics.total_spec_committed.fetch_add(total_emitted);
        metrics.total_spec_correct_drafts.fetch_add(total_correct);
        metrics.total_spec_proposed_drafts.fetch_add(total_proposed);

        decode_log_steps_ += 1;
        win_decode_tokens_ += total_emitted;
        win_spec_steps_ += B;
        win_spec_committed_ += total_emitted;
        win_spec_correct_ += total_correct;
        win_spec_proposed_ += total_proposed;
        win_decode_sec_ += dt;

        if (win_decode_sec_ > 1e-6) {
            metrics.recent_decode_tps.store((double)win_decode_tokens_ / win_decode_sec_);
        }
        if (win_spec_steps_ > 0) {
            metrics.recent_accept_len.store((double)win_spec_committed_ / (double)win_spec_steps_);
        }
        if (win_spec_proposed_ > 0) {
            metrics.recent_accept_rate.store((double)win_spec_correct_ / (double)win_spec_proposed_);
        }

        if (cfg.decode_log_interval > 0 && decode_log_steps_ >= cfg.decode_log_interval) {
            double win_tps = (win_decode_sec_ > 1e-6) ? ((double)win_decode_tokens_ / win_decode_sec_) : 0.0;
            double win_acc_len = (win_spec_steps_ > 0) ? ((double)win_spec_committed_ / (double)win_spec_steps_) : 1.0;
            double win_acc_rate = (win_spec_proposed_ > 0) ? ((double)win_spec_correct_ / (double)win_spec_proposed_) : 0.0;
            double kv_usage = (double)metrics.kv_blocks_used.load() / std::max(1, metrics.kv_blocks_total.load());
            std::string ts = local_timestamp();
            fprintf(stderr,
                    "[%s] Decode batch, #running-req: %d, #token: %d, token usage: %.2f, accept len: %.2f, accept rate: %.2f, cuda graph: %s, TG (gen throughput): %.2f tok/s, #queue-req: %d\n",
                    ts.c_str(), B, active_ctx_tokens, kv_usage,
                    win_acc_len, win_acc_rate,
                    cfg.enable_cuda_graph ? "True" : "False",
                    win_tps, metrics.waiting_requests.load());
            fflush(stderr);
            decode_log_steps_ = 0;
            win_decode_tokens_ = 0;
            win_spec_steps_ = 0;
            win_spec_committed_ = 0;
            win_spec_correct_ = 0;
            win_spec_proposed_ = 0;
            win_decode_sec_ = 0.0;
        }

        for (auto& [req, reason] : to_finish) {
            finish_request(req, reason);
        }
        if ((int)to_finish.size() == B) {
            // Reset partial decode window once batch drains so next request starts clean
            decode_log_steps_ = 0;
            win_decode_tokens_ = 0;
            win_spec_steps_ = 0;
            win_spec_committed_ = 0;
            win_spec_correct_ = 0;
            win_spec_proposed_ = 0;
            win_decode_sec_ = 0.0;
        }
    } else {
        // Standard 1-token-per-request decode (handles temperature > 0, top_k, top_p, or --no-spec)
        CUDA_CHECK(cudaMemcpyAsync(d_tok_, h_tok.data(), B * 4, cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(d_pos_, h_meta_, B * 4, cudaMemcpyHostToDevice, st));

        auto enqueue_dec = [&]() {
            target.forward_decode_verify(
                d_tok_, d_pos_, d_slots_, d_bt_, d_kvlen_, d_sslots_,
                B, 1, max_blocks_per_seq_, 65536,
                /*commit_gdn_inplace=*/true, /*capture_aux_hidden=*/cfg.enable_dflash,
                target.ws.logits, st);
            if (cfg.enable_dflash) {
                draft.materialize_context_kv(target, target.ws.aux_hidden, d_pos_, d_sslots_, B, st);
            }
        };

        if (cfg.enable_cuda_graph) {
            auto it = nospec_graphs_.find(B);
            if (it == nospec_graphs_.end()) {
                enqueue_dec();
                CUDA_CHECK(cudaStreamSynchronize(st));
                nospec_graphs_[B] = GraphEntry{nullptr, nullptr};
            } else if (it->second.exec == nullptr) {
                CUDA_CHECK(cudaStreamSynchronize(st));
                CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
                enqueue_dec();
                CUDA_CHECK(cudaStreamEndCapture(st, &it->second.graph));
                CUDA_CHECK(cudaGraphInstantiate(&it->second.exec, it->second.graph, nullptr, nullptr, 0));
                CUDA_CHECK(cudaGraphLaunch(it->second.exec, st));
            } else {
                CUDA_CHECK(cudaGraphLaunch(it->second.exec, st));
            }
        } else {
            enqueue_dec();
        }

        if (all_greedy) {
            launch_argmax(target.ws.logits, d_top1_, B, Qwen35Config::VOCAB_SIZE, st);
        } else {
            std::vector<float> ts(B), tps(B);
            std::vector<int32_t> tks(B);
            std::vector<uint64_t> sds(B);
            for (int b = 0; b < B; ++b) {
                ts[b] = batch[b]->sampling.temperature;
                tks[b] = batch[b]->sampling.top_k;
                tps[b] = batch[b]->sampling.top_p;
                sds[b] = splitmix64(batch[b]->rng_state);
            }
            CUDA_CHECK(cudaMemcpyAsync(d_temps_, ts.data(), B * 4, cudaMemcpyHostToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(d_topks_, tks.data(), B * 4, cudaMemcpyHostToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(d_topps_, tps.data(), B * 4, cudaMemcpyHostToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(d_seeds_, sds.data(), B * 8, cudaMemcpyHostToDevice, st));
            launch_sample_topkp(target.ws.logits, d_temps_, d_topks_, d_topps_, d_seeds_, d_top1_, B, Qwen35Config::VOCAB_SIZE, st);
        }

        CUDA_CHECK(cudaMemcpyAsync(h_out_, d_top1_, B * 4, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));

        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();

        int active_ctx_tokens = 0;
        std::vector<std::pair<std::shared_ptr<Request>, std::string>> to_finish;
        for (int b = 0; b < B; ++b) {
            auto& req = batch[b];
            int32_t tok = h_out_[b];
            req->seq_len += 1;
            active_ctx_tokens += req->seq_len;
            req->anchor_token = tok;
            req->generated_tokens += 1;
            req->push_token(tok);
            if (tok == Qwen35Config::EOS_TOKEN_0 || tok == Qwen35Config::EOS_TOKEN_1) {
                to_finish.emplace_back(req, "stop");
            } else if (req->generated_tokens >= req->sampling.max_tokens || req->seq_len + 1 >= cfg.context_length) {
                to_finish.emplace_back(req, "length");
            }
        }
        metrics.total_gen_tokens.fetch_add(B);
        decode_log_steps_ += 1;
        win_decode_tokens_ += B;
        win_decode_sec_ += dt;
        if (win_decode_sec_ > 1e-6) {
            metrics.recent_decode_tps.store((double)win_decode_tokens_ / win_decode_sec_);
        }
        if (cfg.decode_log_interval > 0 && decode_log_steps_ >= cfg.decode_log_interval) {
            double win_tps = (win_decode_sec_ > 1e-6) ? ((double)win_decode_tokens_ / win_decode_sec_) : 0.0;
            double kv_usage = (double)metrics.kv_blocks_used.load() / std::max(1, metrics.kv_blocks_total.load());
            std::string ts = local_timestamp();
            fprintf(stderr,
                    "[%s] Decode batch, #running-req: %d, #token: %d, token usage: %.2f, cuda graph: %s, TG (gen throughput): %.2f tok/s, #queue-req: %d\n",
                    ts.c_str(), B, active_ctx_tokens, kv_usage,
                    cfg.enable_cuda_graph ? "True" : "False",
                    win_tps, metrics.waiting_requests.load());
            fflush(stderr);
            decode_log_steps_ = 0;
            win_decode_tokens_ = 0;
            win_decode_sec_ = 0.0;
        }
        for (auto& [req, reason] : to_finish) {
            finish_request(req, reason);
        }
        if ((int)to_finish.size() == B) {
            decode_log_steps_ = 0;
            win_decode_tokens_ = 0;
            win_decode_sec_ = 0.0;
        }
    }
}

void Scheduler::loop() {
    while (!stop_flag_.load()) {
        cleanup_cancelled();
        admit_waiting();

        // Remove any finished requests from running_
        running_.erase(
            std::remove_if(running_.begin(), running_.end(), [](const auto& r) { return !r->active; }),
            running_.end());
        metrics.running_requests.store((int)running_.size());

        if (running_.empty()) {
            std::unique_lock<std::mutex> lk(q_mu_);
            q_cv_.wait_for(lk, std::chrono::milliseconds(5), [&] {
                return stop_flag_.load() || !waiting_.empty();
            });
            continue;
        }

        // Prioritize chunked prefill for any newly admitted request (1 chunk per tick)
        std::shared_ptr<Request> prefill_req;
        std::vector<std::shared_ptr<Request>> decode_batch;
        for (const auto& r : running_) {
            if (!r->prefill_done) {
                if (!prefill_req) prefill_req = r;
            } else {
                decode_batch.push_back(r);
            }
        }

        if (prefill_req) {
            step_prefill_one(prefill_req);
        }
        if (!decode_batch.empty()) {
            step_decode_batch(decode_batch);
        }
    }
}

} // namespace amperegen
