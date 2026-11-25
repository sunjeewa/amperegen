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

    int active_slots = cfg.max_running_requests + 2;
    if (cfg.prefix_cache_slots < 0) cfg.prefix_cache_slots = 0;

    // Auto-cap prefix_cache_slots against live free VRAM so we always keep >= 2.0 GiB headroom
    {
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        size_t bytes_per_slot = (size_t)Qwen35Config::NUM_GDN_LAYERS *
            ((size_t)48 * 128 * 128 + (size_t)10240 * 3) * 2;
        if (cfg.enable_dflash) {
            bytes_per_slot += (size_t)DFlash2Config::NUM_LAYERS * 2 *
                DFlash2Config::SLIDING_WINDOW * DFlash2Config::NUM_KV_HEADS * DFlash2Config::HEAD_DIM * 2;
        }
        bytes_per_slot += (size_t)Qwen35Config::VOCAB_SIZE * 2; // cached exact-hit logits
        // Conservative estimate of target WS + verify + KV pool + draft weights/WS + active slots + 2.0 GiB reserve
        size_t fixed_est = (size_t)(4.3e9) + cfg.kv_pool_bytes +
            (cfg.enable_dflash ? (size_t)(2.8e9) : 0) +
            (size_t)active_slots * bytes_per_slot +
            (2048ULL * 1024 * 1024);
        int max_safe_cache = (free_b > fixed_est) ? (int)((free_b - fixed_est) / bytes_per_slot) : 0;
        if (cfg.prefix_cache_slots > max_safe_cache) {
            fprintf(stderr, "[scheduler] Clamping prefix_cache_slots %d -> %d to preserve >=2 GiB VRAM headroom (free=%.2f GB)\n",
                    cfg.prefix_cache_slots, max_safe_cache, free_b / 1e9);
            cfg.prefix_cache_slots = max_safe_cache;
        }
    }

    int max_slots = active_slots + cfg.prefix_cache_slots;
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

    // Initialize refcounted KV blocks & state slots free lists
    free_kv_blocks_.resize(target.pools.num_kv_blocks);
    kv_block_refcnt_.assign(target.pools.num_kv_blocks, 0);
    for (int i = 0; i < target.pools.num_kv_blocks; ++i) {
        free_kv_blocks_[i] = target.pools.num_kv_blocks - 1 - i;
    }
    free_state_slots_.resize(max_slots);
    for (int i = 0; i < max_slots; ++i) {
        free_state_slots_[i] = max_slots - 1 - i;
    }
    if (cfg.prefix_cache_slots > 0) {
        CUDA_CHECK(cudaMalloc(&d_cached_logits_,
            (size_t)cfg.prefix_cache_slots * Qwen35Config::VOCAB_SIZE * sizeof(__nv_bfloat16)));
        free_logits_slots_.resize(cfg.prefix_cache_slots);
        for (int i = 0; i < cfg.prefix_cache_slots; ++i) {
            free_logits_slots_[i] = cfg.prefix_cache_slots - 1 - i;
        }
    }
    metrics.kv_blocks_total.store(target.pools.num_kv_blocks);
    metrics.kv_blocks_used.store(0);
    metrics.prefix_cache_checkpoints.store(0);
    metrics.prefix_cache_capacity.store(cfg.prefix_cache_slots);

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

    fprintf(stderr, "[scheduler] Ready: max_running=%d ctx=%d dflash=%d block_size=%d kv_blocks=%d (%d tokens) prefix_cache=%d slots\n",
            cfg.max_running_requests, cfg.context_length, (int)cfg.enable_dflash, cfg.block_size,
            target.pools.num_kv_blocks, target.pools.num_kv_blocks * Qwen35Config::KV_BLOCK_SIZE,
            cfg.prefix_cache_slots);
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
    dfree(d_cached_logits_);
    checkpoints_.clear();

    if (cfg.enable_dflash) draft.destroy();
    target.destroy();
}

std::shared_ptr<Request> Scheduler::submit(std::vector<int32_t> prompt_ids, const SamplingParams& sp, int sys_prefix_len) {
    auto req = std::make_shared<Request>();
    req->id = next_req_id_.fetch_add(1);
    req->prompt_ids = std::move(prompt_ids);
    req->sys_prefix_len = sys_prefix_len;
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

int32_t Scheduler::alloc_kv_block(uint64_t exclude_ckpt_id) {
    while (free_kv_blocks_.empty()) {
        if (!evict_one_checkpoint(exclude_ckpt_id)) return -1;
    }
    int32_t blk = free_kv_blocks_.back();
    free_kv_blocks_.pop_back();
    kv_block_refcnt_[blk] = 1;
    metrics.kv_blocks_used.store(target.pools.num_kv_blocks - (int)free_kv_blocks_.size());
    return blk;
}

void Scheduler::incref_kv_block(int32_t blk) {
    ++kv_block_refcnt_[blk];
}

void Scheduler::decref_kv_block(int32_t blk) {
    if (--kv_block_refcnt_[blk] == 0) {
        free_kv_blocks_.push_back(blk);
        metrics.kv_blocks_used.store(target.pools.num_kv_blocks - (int)free_kv_blocks_.size());
    }
}

bool Scheduler::ensure_kv_blocks(const std::shared_ptr<Request>& req, int needed_tokens, uint64_t exclude_ckpt_id) {
    int needed_blocks = (needed_tokens + Qwen35Config::KV_BLOCK_SIZE - 1) / Qwen35Config::KV_BLOCK_SIZE;
    int have = (int)req->kv_blocks.size();
    if (needed_blocks <= have) return true;
    int delta = needed_blocks - have;
    while ((int)free_kv_blocks_.size() < delta) {
        if (!evict_one_checkpoint(exclude_ckpt_id)) return false;
    }
    for (int i = 0; i < delta; ++i) {
        int32_t blk = alloc_kv_block(exclude_ckpt_id);
        if (blk < 0) return false;
        req->kv_blocks.push_back(blk);
    }
    return true;
}

void Scheduler::trim_unused_kv_blocks(const std::shared_ptr<Request>& req, int actual_tokens) {
    int keep = (actual_tokens + Qwen35Config::KV_BLOCK_SIZE - 1) / Qwen35Config::KV_BLOCK_SIZE;
    while ((int)req->kv_blocks.size() > keep) {
        decref_kv_block(req->kv_blocks.back());
        req->kv_blocks.pop_back();
    }
}

bool Scheduler::evict_one_checkpoint(uint64_t exclude_id) {
    int victim = -1;
    uint64_t best_tick = UINT64_MAX;
    for (int i = 0; i < (int)checkpoints_.size(); ++i) {
        if (exclude_id != 0 && checkpoints_[i].id == exclude_id) continue;
        if (checkpoints_[i].last_used_tick < best_tick) {
            best_tick = checkpoints_[i].last_used_tick;
            victim = i;
        }
    }
    if (victim < 0) return false;

    auto& cp = checkpoints_[victim];
    for (int32_t blk : cp.full_kv_blocks) decref_kv_block(blk);
    if (cp.tail_kv_block >= 0) decref_kv_block(cp.tail_kv_block);
    if (cp.state_slot >= 0) free_state_slots_.push_back(cp.state_slot);
    if (cp.logits_slot >= 0) free_logits_slots_.push_back(cp.logits_slot);

    checkpoints_.erase(checkpoints_.begin() + victim);
    metrics.prefix_cache_checkpoints.store((int)checkpoints_.size());
    return true;
}

PrefixCheckpoint* Scheduler::match_prefix(const std::vector<int32_t>& prompt_ids) {
    if (checkpoints_.empty() || prompt_ids.empty()) return nullptr;
    int P = (int)prompt_ids.size();
    PrefixCheckpoint* best = nullptr;
    int best_len = 0;
    for (auto& cp : checkpoints_) {
        int L = (int)cp.tokens.size();
        if (L <= best_len || L > P) continue;
        if (L == P && cp.logits_slot < 0) continue; // exact hit requires cached logits
        if (cp.tokens[0] != prompt_ids[0] || cp.tokens[L - 1] != prompt_ids[L - 1]) continue;
        if (std::memcmp(cp.tokens.data(), prompt_ids.data(), (size_t)L * sizeof(int32_t)) == 0) {
            best = &cp;
            best_len = L;
        }
    }
    if (best) best->last_used_tick = ++cache_clock_;
    return best;
}

bool Scheduler::has_exact_checkpoint(const int32_t* toks, int L) {
    if (L <= 0) return false;
    for (auto& cp : checkpoints_) {
        if ((int)cp.tokens.size() == L &&
            cp.tokens[0] == toks[0] && cp.tokens[L - 1] == toks[L - 1] &&
            std::memcmp(cp.tokens.data(), toks, (size_t)L * sizeof(int32_t)) == 0) {
            cp.last_used_tick = ++cache_clock_;
            return true;
        }
    }
    return false;
}

void Scheduler::save_prefix_checkpoint(
    const std::shared_ptr<Request>& req,
    int L,
    bool save_logits,
    bool transfer_slot_on_finish
) {
    if (cfg.prefix_cache_slots <= 0 || L < 64) return;
    if ((int)req->committed_tokens.size() < L) return;

    cudaStream_t st = target.ctx.stream;
    const int32_t* tok_data = req->committed_tokens.data();

    // Deduplicate against existing checkpoint of identical token sequence
    for (auto& cp : checkpoints_) {
        if ((int)cp.tokens.size() == L &&
            cp.tokens[0] == tok_data[0] && cp.tokens[L - 1] == tok_data[L - 1] &&
            std::memcmp(cp.tokens.data(), tok_data, (size_t)L * sizeof(int32_t)) == 0) {
            cp.last_used_tick = ++cache_clock_;
            if (save_logits && cp.logits_slot < 0 && !free_logits_slots_.empty()) {
                cp.logits_slot = free_logits_slots_.back();
                free_logits_slots_.pop_back();
                CUDA_CHECK(cudaMemcpyAsync(
                    d_cached_logits_ + (size_t)cp.logits_slot * Qwen35Config::VOCAB_SIZE,
                    target.ws.logits,
                    (size_t)Qwen35Config::VOCAB_SIZE * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, st));
            }
            return;
        }
    }

    // Ensure slot capacity (keep >= 2 active slots free when not transferring req's own slot)
    while (((int)checkpoints_.size() >= cfg.prefix_cache_slots ||
            (!transfer_slot_on_finish && (int)free_state_slots_.size() <= 2)) &&
           !checkpoints_.empty()) {
        if (!evict_one_checkpoint()) break;
    }
    if ((int)checkpoints_.size() >= cfg.prefix_cache_slots) return;
    if (!transfer_slot_on_finish && (int)free_state_slots_.size() <= 2) return;

    int n_full = L / Qwen35Config::KV_BLOCK_SIZE;
    bool has_tail = (L % Qwen35Config::KV_BLOCK_SIZE) != 0;
    if ((int)req->kv_blocks.size() < n_full + (has_tail ? 1 : 0)) return;

    int32_t tail_blk = -1;
    if (has_tail) {
        if (transfer_slot_on_finish) {
            // Request is done writing; safe to share its tail block directly (refcount will become 1 after release_request)
            tail_blk = req->kv_blocks[n_full];
            incref_kv_block(tail_blk);
        } else {
            tail_blk = alloc_kv_block();
            if (tail_blk < 0) return;
            target.copy_kv_block(tail_blk, req->kv_blocks[n_full], st);
        }
    }

    int cp_slot = -1;
    if (transfer_slot_on_finish) {
        cp_slot = req->state_slot;
        req->state_slot = -1; // ownership transferred zero-copy!
    } else {
        cp_slot = free_state_slots_.back();
        free_state_slots_.pop_back();
        target.copy_gdn_state_slot(cp_slot, req->state_slot, st);
        if (cfg.enable_dflash) {
            draft.copy_kv_slot(cp_slot, req->state_slot, st);
        }
    }

    PrefixCheckpoint cp;
    cp.id = next_ckpt_id_++;
    cp.last_used_tick = ++cache_clock_;
    cp.tokens.assign(tok_data, tok_data + L);
    cp.full_kv_blocks.assign(req->kv_blocks.begin(), req->kv_blocks.begin() + n_full);
    for (int32_t blk : cp.full_kv_blocks) incref_kv_block(blk);
    cp.tail_kv_block = tail_blk;
    cp.state_slot = cp_slot;

    if (save_logits && !free_logits_slots_.empty()) {
        cp.logits_slot = free_logits_slots_.back();
        free_logits_slots_.pop_back();
        CUDA_CHECK(cudaMemcpyAsync(
            d_cached_logits_ + (size_t)cp.logits_slot * Qwen35Config::VOCAB_SIZE,
            target.ws.logits,
            (size_t)Qwen35Config::VOCAB_SIZE * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice, st));
    }

    checkpoints_.push_back(std::move(cp));
    metrics.prefix_cache_checkpoints.store((int)checkpoints_.size());
}

void Scheduler::complete_exact_cache_hit(const std::shared_ptr<Request>& req, int logits_slot) {
    auto t0 = std::chrono::steady_clock::now();
    cudaStream_t st = target.ctx.stream;
    const __nv_bfloat16* logits_ptr = d_cached_logits_ + (size_t)logits_slot * Qwen35Config::VOCAB_SIZE;

    if (req->sampling.is_greedy()) {
        launch_argmax(logits_ptr, d_top1_, 1, Qwen35Config::VOCAB_SIZE, st);
    } else {
        float t = req->sampling.temperature;
        int32_t tk = req->sampling.top_k;
        float tp = req->sampling.top_p;
        uint64_t sd = splitmix64(req->rng_state);
        CUDA_CHECK(cudaMemcpyAsync(d_temps_, &t, 4, cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(d_topks_, &tk, 4, cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(d_topps_, &tp, 4, cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(d_seeds_, &sd, 8, cudaMemcpyHostToDevice, st));
        launch_sample_topkp(logits_ptr, d_temps_, d_topks_, d_topps_, d_seeds_, d_top1_, 1, Qwen35Config::VOCAB_SIZE, st);
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

    double kv_usage = (double)metrics.kv_blocks_used.load() / std::max(1, metrics.kv_blocks_total.load());
    std::string ts = local_timestamp();
    fprintf(stderr,
            "[%s] Prefill batch, req=#%llu, #new-token: 0 (%d/%d), #cached-token: %d (100.0%% exact hit), token usage: %.2f, #running-req: %d, #queue-req: %d, restore: %.2fms\n",
            ts.c_str(), (unsigned long long)req->id, (int)req->prompt_ids.size(), (int)req->prompt_ids.size(),
            req->cached_tokens, kv_usage, metrics.running_requests.load(), metrics.waiting_requests.load(),
            dt * 1000.0);
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
}

void Scheduler::finish_request(const std::shared_ptr<Request>& req, const std::string& reason) {
    // Save end-of-decode hybrid checkpoint (zero-copy state transfer!) if committed tokens are clean
    if ((reason == "stop" || reason == "length") &&
        req->prefill_done && req->clean_commit_for_cache &&
        req->state_slot >= 0 &&
        req->seq_len > (int)req->prompt_ids.size() &&
        (int)req->committed_tokens.size() == req->seq_len) {
        save_prefix_checkpoint(req, req->seq_len, /*save_logits=*/false, /*transfer_slot_on_finish=*/true);
    }

    auto now = std::chrono::steady_clock::now();
    int prompt_toks = (int)req->prompt_ids.size();
    int new_pp_toks = std::max(0, prompt_toks - req->cached_tokens);
    double cache_pct = prompt_toks > 0 ? (100.0 * (double)req->cached_tokens / (double)prompt_toks) : 0.0;
    int gen_toks = req->generated_tokens;
    double pp_sec = req->prefill_sec;
    double pp_tps = (pp_sec > 1e-6) ? ((double)new_pp_toks / pp_sec) : 0.0;
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
            "[%s] Done req=#%llu finish=%s | PP: %d tok (%d cached, %.1f%%) in %.2fs (%.1f tok/s) | TG: %d tok in %.2fs (%.1f tok/s) | accept: %.2f tok/step (%.1f%%) | TTFT: %.2fs | KV: %.1f%%\n",
            ts.c_str(),
            (unsigned long long)req->id,
            reason.c_str(),
            prompt_toks, req->cached_tokens, cache_pct, pp_sec, pp_tps,
            gen_toks, tg_sec, tg_tps,
            acc_len, acc_rate * 100.0,
            ttft, kv_pct);
    fflush(stderr);

    req->push_finish(reason);
    release_request(req);
}

void Scheduler::release_request(const std::shared_ptr<Request>& req) {
    for (int32_t blk : req->kv_blocks) decref_kv_block(blk);
    req->kv_blocks.clear();
    if (req->state_slot >= 0) {
        free_state_slots_.push_back(req->state_slot);
        req->state_slot = -1;
    }
    req->active = false;
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
    while ((int)running_.size() < cfg.max_running_requests) {
        std::shared_ptr<Request> req;
        {
            std::lock_guard<std::mutex> lk(q_mu_);
            while (!waiting_.empty() && waiting_.front()->cancelled.load()) {
                auto c = waiting_.front();
                waiting_.pop_front();
                c->push_finish("cancelled");
            }
            if (waiting_.empty()) {
                metrics.waiting_requests.store(0);
                break;
            }
            req = waiting_.front();
        }

        // Lookup longest matching hybrid prefix checkpoint BEFORE reserving resources
        PrefixCheckpoint* cp = match_prefix(req->prompt_ids);
        uint64_t cp_id = cp ? cp->id : 0;
        int matched_len = cp ? (int)cp->tokens.size() : 0;
        int shared_full = matched_len / Qwen35Config::KV_BLOCK_SIZE;
        int total_needed_blocks = ((int)req->prompt_ids.size() + cfg.block_size + Qwen35Config::KV_BLOCK_SIZE - 1) / Qwen35Config::KV_BLOCK_SIZE;
        int new_blocks_needed = std::max(0, total_needed_blocks - shared_full);

        // Evict LRU checkpoints (excluding our matched cp!) if needed for state slot or KV blocks
        while (free_state_slots_.empty()) {
            if (!evict_one_checkpoint(cp_id)) break;
        }
        while ((int)free_kv_blocks_.size() < new_blocks_needed) {
            if (!evict_one_checkpoint(cp_id)) break;
        }
        if (free_state_slots_.empty() || (int)free_kv_blocks_.size() < new_blocks_needed) {
            break; // Wait for running requests to free resources
        }

        // Re-fetch cp pointer in case evict_one_checkpoint shifted vector indices
        cp = nullptr;
        if (cp_id != 0) {
            for (auto& c : checkpoints_) {
                if (c.id == cp_id) { cp = &c; break; }
            }
        }

        // Pop req from waiting_
        {
            std::lock_guard<std::mutex> lk(q_mu_);
            if (waiting_.empty() || waiting_.front() != req) continue;
            waiting_.pop_front();
            metrics.waiting_requests.store((int)waiting_.size());
        }

        req->state_slot = free_state_slots_.back();
        free_state_slots_.pop_back();
        req->active = true;
        req->generated_tokens = 0;
        req->prefill_done = false;
        req->clean_commit_for_cache = true;
        req->kv_blocks.clear();

        cudaStream_t st = target.ctx.stream;
        if (cp != nullptr && matched_len > 0) {
            // Snapshot scalar fields BEFORE any alloc_kv_block() that could mutate checkpoints_
            const int cp_state_slot = cp->state_slot;
            const int32_t cp_tail_blk = cp->tail_kv_block;
            const int cp_logits_slot = cp->logits_slot;
            // Share immutable full KV blocks zero-copy
            req->kv_blocks = cp->full_kv_blocks;
            for (int32_t blk : req->kv_blocks) incref_kv_block(blk);
            // Copy partial tail block if matched_len is not block-aligned
            if (cp_tail_blk >= 0) {
                int32_t new_tail = alloc_kv_block(cp_id);
                target.copy_kv_block(new_tail, cp_tail_blk, st);
                req->kv_blocks.push_back(new_tail);
            }
            // Restore 48-layer GDN SSM+conv + 5-layer DFlash2 ring buffer (~0.15 ms GPU-to-GPU!)
            target.copy_gdn_state_slot(req->state_slot, cp_state_slot, st);
            if (cfg.enable_dflash) {
                draft.copy_kv_slot(req->state_slot, cp_state_slot, st);
            }
            req->cached_tokens = matched_len;
            req->prefill_offset = matched_len;
            req->seq_len = matched_len;
            req->committed_tokens.assign(req->prompt_ids.begin(), req->prompt_ids.begin() + matched_len);
            metrics.total_prompt_tokens.fetch_add(matched_len);
            metrics.total_cached_tokens.fetch_add(matched_len);
            metrics.total_prefix_cache_hits.fetch_add(1);

            running_.push_back(req);
            metrics.running_requests.store((int)running_.size());

            // 100% exact prompt match: sample token 0 straight from cached logits!
            if (matched_len == (int)req->prompt_ids.size() && cp_logits_slot >= 0) {
                complete_exact_cache_hit(req, cp_logits_slot);
            }
        } else {
            req->cached_tokens = 0;
            req->prefill_offset = 0;
            req->seq_len = 0;
            req->committed_tokens.clear();
            target.reset_state_slot(req->state_slot, st);
            running_.push_back(req);
            metrics.running_requests.store((int)running_.size());
        }
    }
}

void Scheduler::step_prefill_one(const std::shared_ptr<Request>& req, bool has_active_decode) {
    auto t0 = std::chrono::steady_clock::now();
    int prompt_len = (int)req->prompt_ids.size();
    int off = req->prefill_offset;

    // Adaptive chunk sizing when other requests are actively streaming decode tokens
    int max_chunk = cfg.chunked_prefill_size;
    if (has_active_decode) {
        if (off >= 65536) max_chunk = std::min(max_chunk, 1024);
        else if (off >= 16384) max_chunk = std::min(max_chunk, 2048);
        else max_chunk = std::min(max_chunk, 4096);
    }

    // Check if we should snapshot the shared system-prompt (+ tools) boundary along the way
    bool want_sys_ckpt = (cfg.prefix_cache_slots > 0 &&
                          req->sys_prefix_len >= 256 &&
                          req->sys_prefix_len < prompt_len &&
                          off < req->sys_prefix_len &&
                          !has_exact_checkpoint(req->prompt_ids.data(), req->sys_prefix_len));

    int chunk = std::min(max_chunk, prompt_len - off);
    if (want_sys_ckpt && off + chunk > req->sys_prefix_len) {
        chunk = req->sys_prefix_len - off;
    }
    bool saving_sys_ckpt = want_sys_ckpt && (off + chunk == req->sys_prefix_len);
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

    // Materialize aux_hidden for DFlash2 whenever within 2048 tokens of prompt_len OR sys_prefix_len checkpoint!
    int aux_lo = chunk;
    if (off + chunk > prompt_len - DFlash2Config::SLIDING_WINDOW) {
        aux_lo = std::min(aux_lo, std::max(0, (prompt_len - DFlash2Config::SLIDING_WINDOW) - off));
    }
    if (want_sys_ckpt && off + chunk > req->sys_prefix_len - DFlash2Config::SLIDING_WINDOW) {
        aux_lo = std::min(aux_lo, std::max(0, (req->sys_prefix_len - DFlash2Config::SLIDING_WINDOW) - off));
    }
    bool need_aux = cfg.enable_dflash && (aux_lo < chunk);
    target.forward_prefill(
        d_tok_, d_pos_, d_slots_, d_bt_, d_cu_, d_kvlen_, d_sslots_,
        1, chunk, chunk, max_blocks_per_seq_, need_aux,
        is_last ? d_last_idx_ : nullptr,
        is_last ? target.ws.logits : nullptr,
        st);

    if (need_aux) {
        int n_mat = chunk - aux_lo;
        std::vector<int32_t> cs(n_mat, sslot);
        CUDA_CHECK(cudaMemcpyAsync(d_ctx_slots_, cs.data(), n_mat * 4, cudaMemcpyHostToDevice, st));
        draft.materialize_context_kv(
            target,
            target.ws.aux_hidden + (size_t)aux_lo * DFlash2Config::FC_IN_DIM,
            d_pos_ + aux_lo, d_ctx_slots_, n_mat, st);
    }

    req->committed_tokens.insert(
        req->committed_tokens.end(),
        req->prompt_ids.begin() + off,
        req->prompt_ids.begin() + off + chunk);
    req->prefill_offset = off + chunk;
    req->seq_len = off + chunk;
    metrics.total_prompt_tokens.fetch_add(chunk);

    if (saving_sys_ckpt) {
        save_prefix_checkpoint(req, req->sys_prefix_len, /*save_logits=*/false, /*transfer_slot_on_finish=*/false);
    }

    if (is_last) {
        // Save end-of-prefill checkpoint WITH exact last-token logits before decode starts (~0.15 ms!)
        save_prefix_checkpoint(req, prompt_len, /*save_logits=*/true, /*transfer_slot_on_finish=*/false);

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

        int new_total = std::max(1, prompt_len - req->cached_tokens);
        double chunk_pp_tps = (dt > 1e-6) ? ((double)chunk / dt) : 0.0;
        double full_pp_tps = (req->prefill_sec > 1e-6) ? ((double)new_total / req->prefill_sec) : chunk_pp_tps;
        metrics.recent_prefill_tps.store(full_pp_tps);

        double kv_usage = (double)metrics.kv_blocks_used.load() / std::max(1, metrics.kv_blocks_total.load());
        std::string ts = local_timestamp();
        fprintf(stderr,
                "[%s] Prefill batch, req=#%llu, #new-token: %d (%d/%d), #cached-token: %d, token usage: %.2f, #running-req: %d, #queue-req: %d, PP (input throughput): %.2f tok/s (%.2fs)\n",
                ts.c_str(), (unsigned long long)req->id, chunk, off + chunk, prompt_len,
                req->cached_tokens, kv_usage, metrics.running_requests.load(), metrics.waiting_requests.load(),
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
                "[%s] Prefill batch, req=#%llu, #new-token: %d (%d/%d), #cached-token: %d, token usage: %.2f, #running-req: %d, #queue-req: %d, PP (input throughput): %.2f tok/s (%.2fs)\n",
                ts.c_str(), (unsigned long long)req->id, chunk, off + chunk, prompt_len,
                req->cached_tokens, kv_usage, metrics.running_requests.load(), metrics.waiting_requests.load(),
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
            int32_t old_anchor = req->anchor_token;

            // Exact tokens committed into KV/GDN/DFlash2 on GPU are [old_anchor, toks[0..c_len-2]]
            req->committed_tokens.push_back(old_anchor);
            for (int k = 0; k < c_len - 1; ++k) {
                req->committed_tokens.push_back(toks[k]);
            }

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
            int stop_k = -1;
            std::string reason;
            for (int k = 0; k < c_len; ++k) {
                int32_t tok = toks[k];
                req->generated_tokens += 1;
                total_pushed += 1;
                req->push_token(tok);
                if (tok == Qwen35Config::EOS_TOKEN_0 || tok == Qwen35Config::EOS_TOKEN_1) {
                    finished = true; stop_k = k; reason = "stop"; break;
                }
                if (req->generated_tokens >= req->sampling.max_tokens || req->seq_len + 1 >= cfg.context_length) {
                    finished = true; stop_k = k; reason = "length"; break;
                }
            }
            if (finished) {
                if (stop_k < c_len - 1) req->clean_commit_for_cache = false;
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
            req->committed_tokens.push_back(req->anchor_token);
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
            bool has_dec = !decode_batch.empty();
            step_prefill_one(prefill_req, has_dec);
            if (has_dec) {
                // Give active streaming decode requests multiple steps between concurrent prefill chunks
                for (int d = 0; d < 8 && !decode_batch.empty(); ++d) {
                    step_decode_batch(decode_batch);
                    decode_batch.erase(
                        std::remove_if(decode_batch.begin(), decode_batch.end(),
                                       [](const auto& r) { return !r->active || r->cancelled.load(); }),
                        decode_batch.end());
                }
            }
        } else if (!decode_batch.empty()) {
            step_decode_batch(decode_batch);
        }
    }
}

} // namespace amperegen
