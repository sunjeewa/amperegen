#include "scheduler/scheduler.hpp"
#include "server/http_server.hpp"
#include "server/sidecar_client.hpp"

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
    std::string python_bin = "python3";
    std::string sidecar_script = "sidecar/app.py";
    std::string sidecar_socket = "/tmp/amperegen_sidecar.sock";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--model-path" || a == "--model") && i + 1 < argc) cfg.model_dir = argv[++i];
        else if (a == "--draft-path" && i + 1 < argc) cfg.draft_dir = argv[++i];
        else if (a == "--served-model-name" && i + 1 < argc) cfg.served_model_name = argv[++i];
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
        else if (a == "--python" && i + 1 < argc) python_bin = argv[++i];
        else if (a == "--sidecar-script" && i + 1 < argc) sidecar_script = argv[++i];
        else if (a == "--sidecar-socket" && i + 1 < argc) sidecar_socket = argv[++i];
        else if (a == "-h" || a == "--help") {
            printf("amperegen — Dedicated CUDA C++20 Inference Engine for Qwen3.8-27B-FP8 + DFlash2 on CMP 170HX\n"
                   "Options:\n"
                   "  --model-path PATH           Target FP8 checkpoint (default: %s)\n"
                   "  --draft-path PATH           DFlash2 draft checkpoint (default: %s)\n"
                   "  --served-model-name NAME    Model name in API (default: %s)\n"
                   "  --host HOST --port PORT     Bind address (default: 0.0.0.0:8000)\n"
                   "  --context-length N          Max sequence length (default: 200000)\n"
                   "  --max-running-requests N    Max concurrent requests (default: 16)\n"
                   "  --block-size N              DFlash2 spec block size 6..8 (default: 8)\n"
                   "  --kv-pool-gb GB             FP8 KV cache pool size in GB (default: 18)\n"
                   "  --prefix-cache-slots N      Hybrid KV+GDN+DFlash2 cached prefix slots (default: 24, 0=off)\n"
                   "  --no-prefix-cache           Disable hybrid radix prefix caching\n"
                   "  --decode-log-interval N     Periodic decode telemetry interval in steps (default: 40)\n"
                   "  --python PATH               Python binary for tokenizer sidecar (default: python3)\n"
                   "  --sidecar-script PATH       Path to sidecar/app.py (default: sidecar/app.py)\n"
                   "  --no-spec                   Disable DFlash2 speculative decoding\n"
                   "  --draft-bf16                Keep draft weights in BF16 instead of FP8\n"
                   "  --bf16-lm-draft             Use BF16 lm_head for draft instead of FP8\n"
                   "  --no-cuda-graph             Disable CUDA graphs\n",
                   cfg.model_dir.c_str(), cfg.draft_dir.c_str(), cfg.served_model_name.c_str());
            return 0;
        }
    }

    try {
        amperegen::SidecarClient sidecar;
        sidecar.start(python_bin, sidecar_script, cfg.model_dir, sidecar_socket);

        amperegen::Scheduler sched;
        sched.init(cfg);
        sched.start();

        amperegen::HttpServer server(sched, sidecar, host, port);
        g_server = &server;
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);

        server.serve();

        fprintf(stderr, "[main] Shutting down...\n");
        sched.stop();
        sidecar.stop();
    } catch (const std::exception& e) {
        fprintf(stderr, "[FATAL] %s\n", e.what());
        return 1;
    }
    return 0;
}
