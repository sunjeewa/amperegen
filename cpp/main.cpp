#include "scheduler/scheduler.hpp"
#include "server/http_server.hpp"
#include "server/tokenizer.hpp"
#include "weights/weight_backend.hpp"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
amperegen::HttpServer* g_server = nullptr;
void on_signal(int) {
    if (g_server) g_server->stop();
}
} // namespace

int main(int argc, char** argv) {
    amperegen::EngineConfig cfg;
    std::string host = "0.0.0.0";
    int port = 8000;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--model-path" || a == "--model") && i + 1 < argc) cfg.model_dir = argv[++i];
        else if (a == "--draft-path" && i + 1 < argc) cfg.draft_dir = argv[++i];
        else if (a == "--served-model-name" && i + 1 < argc) cfg.served_model_name = argv[++i];
        else if (a == "--weight-backend" && i + 1 < argc) cfg.weight_backend = argv[++i];
        else if (a == "--list-weight-backends") {
            printf("Compiled curated weight plugins:\n");
            for (const auto* b : amperegen::WeightRegistry::all()) {
                printf("  %-14s %s\n", b->id(), b->description());
            }
            return 0;
        }
        else if (a == "--host" && i + 1 < argc) host = argv[++i];
        else if (a == "--port" && i + 1 < argc) port = atoi(argv[++i]);
        else if (a == "--context-length" && i + 1 < argc) cfg.context_length = atoi(argv[++i]);
        else if (a == "--max-running-requests" && i + 1 < argc) cfg.max_running_requests = atoi(argv[++i]);
        else if (a == "--chunked-prefill-size" && i + 1 < argc) cfg.chunked_prefill_size = atoi(argv[++i]);
        else if (a == "--block-size" && i + 1 < argc) cfg.block_size = atoi(argv[++i]);
        else if (a == "--kv-pool-gb" && i + 1 < argc) cfg.kv_pool_bytes = (size_t)atof(argv[++i]) * 1024ULL * 1024 * 1024;
        else if (a == "--prefix-cache-slots" && i + 1 < argc) cfg.prefix_cache_slots = atoi(argv[++i]);
        else if (a == "--no-prefix-cache") cfg.prefix_cache_slots = 0;
        else if (a == "--decode-log-interval" && i + 1 < argc) cfg.decode_log_interval = atoi(argv[++i]);
        else if (a == "--no-spec") cfg.enable_dflash = false;
        else if (a == "--draft-bf16") cfg.draft_fp8 = false;
        else if (a == "--bf16-lm-draft") cfg.fp8_lm_draft = false;
        else if (a == "--no-cuda-graph") cfg.enable_cuda_graph = false;
        else if (a == "--no-auto-truncate") cfg.allow_auto_truncate = false;
        else if (a == "-h" || a == "--help") {
            printf("amperegen — Dedicated CUDA C++20 Inference Engine for Qwen3.8-27B + DFlash2 on Ampere sm_80\n"
                   "Options:\n"
                   "  --model-path PATH           Target checkpoint (default: %s)\n"
                   "  --draft-path PATH           DFlash2 draft checkpoint (default: %s)\n"
                   "  --served-model-name NAME    Model name in API (default: %s)\n"
                   "  --weight-backend ID         Curated weight plugin: auto | %s (default: auto)\n"
                   "  --list-weight-backends      List compiled curated weight plugins and exit\n"
                   "  --host HOST --port PORT     Bind address (default: 0.0.0.0:8000)\n"
                   "  --context-length N          Max sequence length (default: 200000)\n"
                   "  --max-running-requests N    Max concurrent requests (default: 16)\n"
                   "  --block-size N              DFlash2 spec block size 6..8 (default: 8)\n"
                   "  --kv-pool-gb GB             FP8 KV cache pool size in GB (default: 18)\n"
                   "  --prefix-cache-slots N      Hybrid prefix cache slots in VRAM (default: 24, 0=off)\n"
                   "  --no-spec                   Disable DFlash2 speculative decoding\n"
                   "  --draft-bf16                Keep draft weights in BF16 instead of FP8\n"
                   "  --bf16-lm-draft             Use BF16 lm_head for draft instead of FP8\n"
                   "  --no-cuda-graph             Disable CUDA graphs\n",
                   cfg.model_dir.c_str(), cfg.draft_dir.c_str(), cfg.served_model_name.c_str(),
                   amperegen::WeightRegistry::compiled_ids_string().c_str());
            return 0;
        }
    }

    try {
        amperegen::NativeTokenizer tok;
        {
            auto t0 = std::chrono::steady_clock::now();
            std::string err;
            if (!tok.load_from_model_dir(cfg.model_dir, &err)) {
                throw std::runtime_error("Failed to load native tokenizer from " + cfg.model_dir + ": " + err);
            }
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr,
                    "[tokenizer] Loaded native C++20 tokenizer (PCRE2 + utf8proc + minja): "
                    "%zu vocab, %zu merges, %zu added, chat_template=%s in %.1f ms\n",
                    tok.vocab_size(), tok.merge_count(), tok.added_token_count(),
                    tok.has_chat_template() ? "yes" : "no", ms);
        }

        amperegen::Scheduler sched;
        sched.init(cfg);
        sched.start();

        amperegen::HttpServer server(sched, tok, host, port);
        g_server = &server;
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);

        server.serve();

        fprintf(stderr, "[main] Shutting down...\n");
        sched.stop();
    } catch (const std::exception& e) {
        fprintf(stderr, "[FATAL] %s\n", e.what());
        return 1;
    }
    return 0;
}
