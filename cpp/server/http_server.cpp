#include "server/http_server.hpp"
#include "server/json_lite.hpp"
#include "server/tool_parser.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <optional>
#include <sstream>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace amperegen {

namespace {

bool send_all(int fd, const void* data, size_t len) {
    const char* p = static_cast<const char*>(data);
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, p + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += (size_t)n;
    }
    return true;
}

void send_response(int fd, int status, const char* status_text, const char* content_type, const std::string& body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << " " << status_text << "\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n"
        << "Access-Control-Allow-Origin: *\r\n\r\n";
    std::string hdr = oss.str();
    send_all(fd, hdr.data(), hdr.size());
    send_all(fd, body.data(), body.size());
}

void send_json_error(int fd, int status, const char* status_text, const std::string& msg) {
    Json err;
    err["error"]["message"] = msg;
    err["error"]["type"] = "invalid_request_error";
    send_response(fd, status, status_text, "application/json", err.dump());
}

bool send_sse_chunk(int fd, const std::string& payload) {
    std::string msg = "data: " + payload + "\n\n";
    return send_all(fd, msg.data(), msg.size());
}

// Parse Qwen3 <think>...</think> split for non-stream responses
std::pair<std::string, std::string> split_qwen3_reasoning(const std::string& text, bool start_in_think) {
    size_t start = text.find("<think>");
    size_t end = text.find("</think>");
    if (end != std::string::npos && (start_in_think || (start != std::string::npos && start < end))) {
        size_t r_start = (start != std::string::npos && start < end) ? (start + 7) : 0;
        std::string reasoning = text.substr(r_start, end - r_start);
        std::string content = text.substr(end + 8);
        while (!content.empty() && (content.front() == '\n' || content.front() == '\r')) content.erase(content.begin());
        return {reasoning, content};
    }
    if (start_in_think) {
        size_t r_start = (start != std::string::npos) ? (start + 7) : 0;
        return {text.substr(r_start), ""};
    }
    if (start != std::string::npos) {
        return {text.substr(start + 7), ""};
    }
    return {"", text};
}

// Decode HTTP/1.1 Transfer-Encoding: chunked request body from socket + initial buffered bytes.
bool read_chunked_body(int fd, std::string& buf, std::string& out) {
    char tmp[8192];
    size_t pos = 0;
    auto ensure_bytes = [&](size_t need) -> bool {
        while (buf.size() - pos < need) {
            ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
            if (n <= 0) return false;
            buf.append(tmp, (size_t)n);
        }
        return true;
    };
    while (true) {
        size_t crlf = std::string::npos;
        while ((crlf = buf.find("\r\n", pos)) == std::string::npos) {
            ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
            if (n <= 0) return false;
            buf.append(tmp, (size_t)n);
        }
        std::string size_line = buf.substr(pos, crlf - pos);
        size_t semi = size_line.find(';');
        if (semi != std::string::npos) size_line.resize(semi);
        size_t chunk_sz = 0;
        try {
            chunk_sz = (size_t)std::stoull(size_line, nullptr, 16);
        } catch (...) {
            return false;
        }
        pos = crlf + 2;
        if (chunk_sz == 0) {
            // Consume trailing CRLF (or trailers ending in CRLF)
            return true;
        }
        if (!ensure_bytes(chunk_sz + 2)) return false;
        out.append(buf, pos, chunk_sz);
        pos += chunk_sz + 2; // skip trailing \r\n after chunk data
    }
}

Json build_tool_delta_json(const ToolCallDelta& td) {
    Json tc;
    tc["index"] = td.index;
    if (td.is_header) {
        tc["id"] = td.id;
        tc["type"] = "function";
        Json fn;
        fn["name"] = td.name;
        fn["arguments"] = td.arguments_delta;
        tc["function"] = std::move(fn);
    } else {
        Json fn;
        fn["arguments"] = td.arguments_delta;
        tc["function"] = std::move(fn);
    }
    return tc;
}

} // namespace

HttpServer::HttpServer(Scheduler& sched, const NativeTokenizer& tok, const std::string& host, int port)
    : sched_(sched), tok_(tok), host_(host), port_(port) {}

HttpServer::~HttpServer() { stop(); }

void HttpServer::stop() {
    stop_flag_.store(true);
    if (listen_fd_ >= 0) {
        shutdown(listen_fd_, SHUT_RDWR);
        close(listen_fd_);
        listen_fd_ = -1;
    }
}

void HttpServer::serve() {
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) throw std::runtime_error("socket() failed");
    int opt = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port_);
    if (host_ == "0.0.0.0" || host_.empty()) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        inet_pton(AF_INET, host_.c_str(), &addr.sin_addr);
    }

    if (bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(listen_fd_);
        listen_fd_ = -1;
        throw std::runtime_error("bind() failed on port " + std::to_string(port_));
    }
    if (listen(listen_fd_, 128) != 0) {
        close(listen_fd_);
        listen_fd_ = -1;
        throw std::runtime_error("listen() failed");
    }

    fprintf(stderr, "[server] amperegen OpenAI API listening on http://%s:%d\n", host_.c_str(), port_);

    while (!stop_flag_.load()) {
        struct sockaddr_in caddr {};
        socklen_t clen = sizeof(caddr);
        int cfd = accept(listen_fd_, reinterpret_cast<struct sockaddr*>(&caddr), &clen);
        if (cfd < 0) {
            if (stop_flag_.load()) break;
            continue;
        }
        int one = 1;
        setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        std::thread([this, cfd]() {
            handle_client(cfd);
            close(cfd);
        }).detach();
    }
}

void HttpServer::handle_client(int fd) {
    // Read HTTP request headers + body
    std::string raw;
    char buf[8192];
    size_t hdr_end = std::string::npos;
    size_t content_len = 0;

    while (hdr_end == std::string::npos) {
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) return;
        raw.append(buf, (size_t)n);
        hdr_end = raw.find("\r\n\r\n");
        if (raw.size() > 64 * 1024 && hdr_end == std::string::npos) {
            send_json_error(fd, 400, "Bad Request", "Headers too large");
            return;
        }
    }

    std::string headers = raw.substr(0, hdr_end);
    std::string body = raw.substr(hdr_end + 4);

    // Parse request line
    size_t eol = headers.find("\r\n");
    std::string req_line = (eol == std::string::npos) ? headers : headers.substr(0, eol);
    std::istringstream iss(req_line);
    std::string method, path;
    iss >> method >> path;

    // Strip query string
    size_t qpos = path.find('?');
    if (qpos != std::string::npos) path = path.substr(0, qpos);

    // Find Content-Length / Transfer-Encoding / Expect (case-insensitive)
    std::string lower_hdr = headers;
    for (char& c : lower_hdr) c = (char)std::tolower((unsigned char)c);
    if (lower_hdr.find("expect: 100-continue") != std::string::npos) {
        const char* cont = "HTTP/1.1 100 Continue\r\n\r\n";
        send_all(fd, cont, strlen(cont));
    }
    bool is_chunked = (lower_hdr.find("transfer-encoding:") != std::string::npos &&
                       lower_hdr.find("chunked") != std::string::npos);
    if (is_chunked) {
        std::string decoded;
        if (!read_chunked_body(fd, body, decoded)) {
            send_json_error(fd, 400, "Bad Request", "Malformed chunked request body");
            return;
        }
        body = std::move(decoded);
    } else {
        size_t cl_pos = lower_hdr.find("content-length:");
        if (cl_pos != std::string::npos) {
            size_t val_start = cl_pos + 15;
            size_t val_end = lower_hdr.find("\r\n", val_start);
            content_len = (size_t)std::stoull(lower_hdr.substr(val_start, val_end - val_start));
        }
        while (body.size() < content_len) {
            ssize_t n = recv(fd, buf, sizeof(buf), 0);
            if (n <= 0) return;
            body.append(buf, (size_t)n);
        }
        if (body.size() > content_len) body.resize(content_len);
    }

    if (method == "OPTIONS") {
        std::string resp =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
            "Connection: close\r\n\r\n";
        send_all(fd, resp.data(), resp.size());
        return;
    }

    if (method == "GET" && (path == "/health" || path == "/health_generate")) {
        handle_health(fd);
    } else if (method == "GET" && path == "/v1/models") {
        handle_models(fd);
    } else if (method == "GET" && path == "/metrics") {
        handle_metrics(fd);
    } else if (method == "GET" && (path == "/get_server_info" || path == "/server_info")) {
        handle_server_info(fd);
    } else if (method == "POST" && path == "/v1/completions") {
        handle_completions(fd, body, false);
    } else if (method == "POST" && path == "/v1/chat/completions") {
        handle_completions(fd, body, true);
    } else {
        send_json_error(fd, 404, "Not Found", "Unknown endpoint: " + path);
    }
}

void HttpServer::handle_health(int fd) {
    send_response(fd, 200, "OK", "application/json", "{\"status\":\"ok\"}");
}

void HttpServer::handle_models(int fd) {
    Json m;
    m["id"] = sched_.cfg.served_model_name;
    m["object"] = "model";
    m["owned_by"] = "amperegen";
    m["max_model_len"] = sched_.cfg.context_length;
    Json resp;
    resp["object"] = "list";
    resp["data"] = Json::Array{std::move(m)};
    send_response(fd, 200, "OK", "application/json", resp.dump());
}

void HttpServer::handle_metrics(int fd) {
    std::ostringstream oss;
    int running = sched_.metrics.running_requests.load();
    int waiting = sched_.metrics.waiting_requests.load();
    int used = sched_.metrics.kv_blocks_used.load();
    int total = std::max(1, sched_.metrics.kv_blocks_total.load());
    double kv_ratio = (double)used / (double)total;

    uint64_t prompt_tot = sched_.metrics.total_prompt_tokens.load();
    uint64_t cached_tot = sched_.metrics.total_cached_tokens.load();
    uint64_t gen_tot = sched_.metrics.total_gen_tokens.load();
    uint64_t reqs_tot = sched_.metrics.total_requests_completed.load();

    uint64_t steps = sched_.metrics.total_spec_steps.load();
    uint64_t comm = sched_.metrics.total_spec_committed.load();
    uint64_t corr = sched_.metrics.total_spec_correct_drafts.load();
    uint64_t prop = sched_.metrics.total_spec_proposed_drafts.load();

    double recent_acc_len = sched_.metrics.recent_accept_len.load();
    double recent_acc_rate = sched_.metrics.recent_accept_rate.load();
    double mean_acc_len = recent_acc_len > 0.0 ? recent_acc_len : (steps > 0 ? (double)comm / (double)steps : 0.0);
    double mean_acc_rate = recent_acc_rate > 0.0 ? recent_acc_rate : (prop > 0 ? (double)corr / (double)prop : 0.0);

    double pp_tps = sched_.metrics.recent_prefill_tps.load();
    double tg_tps = sched_.metrics.recent_decode_tps.load();
    double ttft_sum = sched_.metrics.total_ttft_sec.load();
    uint64_t ttft_cnt = sched_.metrics.total_ttft_count.load();
    double e2e_sum = sched_.metrics.total_e2e_sec.load();
    double last_ttft = sched_.metrics.last_ttft_sec.load();

    int pc_ckpts = sched_.metrics.prefix_cache_checkpoints.load();
    uint64_t pc_hits = sched_.metrics.total_prefix_cache_hits.load();
    double cache_hit_ratio = prompt_tot > 0 ? (double)cached_tot / (double)prompt_tot : 0.0;

    std::string lbl = "{model_name=" + Json::escape(sched_.cfg.served_model_name) + "}";

    // Native amperegen Prometheus metrics
    oss << "# TYPE amperegen_running_requests gauge\n"
        << "amperegen_running_requests" << lbl << " " << running << "\n"
        << "# TYPE amperegen_waiting_requests gauge\n"
        << "amperegen_waiting_requests" << lbl << " " << waiting << "\n"
        << "# TYPE amperegen_kv_cache_usage_ratio gauge\n"
        << "amperegen_kv_cache_usage_ratio" << lbl << " " << kv_ratio << "\n"
        << "# TYPE amperegen_prefix_cache_checkpoints gauge\n"
        << "amperegen_prefix_cache_checkpoints" << lbl << " " << pc_ckpts << "\n"
        << "# TYPE amperegen_prefix_cache_hit_ratio gauge\n"
        << "amperegen_prefix_cache_hit_ratio" << lbl << " " << cache_hit_ratio << "\n"
        << "# TYPE amperegen_prefix_cache_hits_total counter\n"
        << "amperegen_prefix_cache_hits_total" << lbl << " " << pc_hits << "\n"
        << "# TYPE amperegen_prefill_throughput_toks_per_sec gauge\n"
        << "amperegen_prefill_throughput_toks_per_sec" << lbl << " " << pp_tps << "\n"
        << "# TYPE amperegen_gen_throughput_toks_per_sec gauge\n"
        << "amperegen_gen_throughput_toks_per_sec" << lbl << " " << tg_tps << "\n"
        << "# TYPE amperegen_spec_accept_length gauge\n"
        << "amperegen_spec_accept_length" << lbl << " " << mean_acc_len << "\n"
        << "# TYPE amperegen_spec_accept_rate gauge\n"
        << "amperegen_spec_accept_rate" << lbl << " " << mean_acc_rate << "\n"
        << "# TYPE amperegen_ttft_seconds_last gauge\n"
        << "amperegen_ttft_seconds_last" << lbl << " " << last_ttft << "\n"
        << "# TYPE amperegen_prompt_tokens_total counter\n"
        << "amperegen_prompt_tokens_total" << lbl << " " << prompt_tot << "\n"
        << "# TYPE amperegen_cached_tokens_total counter\n"
        << "amperegen_cached_tokens_total" << lbl << " " << cached_tot << "\n"
        << "# TYPE amperegen_generation_tokens_total counter\n"
        << "amperegen_generation_tokens_total" << lbl << " " << gen_tot << "\n"
        << "# TYPE amperegen_requests_completed_total counter\n"
        << "amperegen_requests_completed_total" << lbl << " " << reqs_tot << "\n"
        // SGLang-compatible Prometheus aliases (for sglang-stats / vllm-stats / llm-stats / Grafana)
        << "# TYPE sglang:num_running_reqs gauge\n"
        << "sglang:num_running_reqs" << lbl << " " << running << "\n"
        << "# TYPE sglang:num_queue_reqs gauge\n"
        << "sglang:num_queue_reqs" << lbl << " " << waiting << "\n"
        << "# TYPE sglang:token_usage gauge\n"
        << "sglang:token_usage" << lbl << " " << kv_ratio << "\n"
        << "# TYPE sglang:prefill_throughput gauge\n"
        << "sglang:prefill_throughput" << lbl << " " << pp_tps << "\n"
        << "# TYPE sglang:gen_throughput gauge\n"
        << "sglang:gen_throughput" << lbl << " " << tg_tps << "\n"
        << "# TYPE sglang:spec_accept_length gauge\n"
        << "sglang:spec_accept_length" << lbl << " " << mean_acc_len << "\n"
        << "# TYPE sglang:spec_accept_rate gauge\n"
        << "sglang:spec_accept_rate" << lbl << " " << mean_acc_rate << "\n"
        << "# TYPE sglang:prompt_tokens_total counter\n"
        << "sglang:prompt_tokens_total" << lbl << " " << prompt_tot << "\n"
        << "# TYPE sglang:cached_tokens_total counter\n"
        << "sglang:cached_tokens_total" << lbl << " " << cached_tot << "\n"
        << "# TYPE sglang:generation_tokens_total counter\n"
        << "sglang:generation_tokens_total" << lbl << " " << gen_tot << "\n"
        << "# TYPE sglang:num_requests_total counter\n"
        << "sglang:num_requests_total" << lbl << " " << reqs_tot << "\n"
        << "# TYPE sglang:time_to_first_token_seconds_sum counter\n"
        << "sglang:time_to_first_token_seconds_sum" << lbl << " " << ttft_sum << "\n"
        << "# TYPE sglang:time_to_first_token_seconds_count counter\n"
        << "sglang:time_to_first_token_seconds_count" << lbl << " " << ttft_cnt << "\n"
        << "# TYPE sglang:e2e_request_latency_seconds_sum counter\n"
        << "sglang:e2e_request_latency_seconds_sum" << lbl << " " << e2e_sum << "\n"
        << "# TYPE sglang:e2e_request_latency_seconds_count counter\n"
        << "sglang:e2e_request_latency_seconds_count" << lbl << " " << reqs_tot << "\n";
    send_response(fd, 200, "OK", "text/plain; version=0.0.4", oss.str());
}

void HttpServer::handle_server_info(int fd) {
    uint64_t steps = sched_.metrics.total_spec_steps.load();
    uint64_t comm = sched_.metrics.total_spec_committed.load();
    uint64_t corr = sched_.metrics.total_spec_correct_drafts.load();
    uint64_t prop = sched_.metrics.total_spec_proposed_drafts.load();

    Json mem;
    mem["token_capacity"] = sched_.target.pools.num_kv_blocks * Qwen35Config::KV_BLOCK_SIZE;
    mem["weight"] = (double)(sched_.target.weights.arena.used() +
                             (sched_.cfg.enable_dflash ? sched_.draft.weights.arena.used() : 0)) / 1e9;
    mem["kvcache"] = (double)sched_.target.pools.kv_arena.used() / 1e9;

    Json st;
    st["memory_usage"] = std::move(mem);
    st["last_prefill_throughput"] = sched_.metrics.recent_prefill_tps.load();
    st["last_gen_throughput"] = sched_.metrics.recent_decode_tps.load();
    st["avg_spec_accept_length"] = steps > 0 ? (double)comm / (double)steps : 0.0;
    st["avg_spec_accept_rate"] = prop > 0 ? (double)corr / (double)prop : 0.0;
    st["num_running_reqs"] = sched_.metrics.running_requests.load();
    st["num_queue_reqs"] = sched_.metrics.waiting_requests.load();
    st["total_prompt_tokens"] = (int64_t)sched_.metrics.total_prompt_tokens.load();
    st["total_gen_tokens"] = (int64_t)sched_.metrics.total_gen_tokens.load();
    st["total_requests"] = (int64_t)sched_.metrics.total_requests_completed.load();

    Json resp;
    resp["engine"] = "amperegen";
    resp["version"] = "1.0.0";
    resp["model_path"] = sched_.cfg.model_dir;
    resp["served_model_name"] = sched_.cfg.served_model_name;
    resp["context_length"] = sched_.cfg.context_length;
    resp["max_running_requests"] = sched_.cfg.max_running_requests;
    resp["speculative_algorithm"] = sched_.cfg.enable_dflash ? "DFLASH" : "NONE";
    resp["speculative_num_draft_tokens"] = sched_.cfg.enable_dflash ? sched_.cfg.block_size : 0;
    const auto* wb = sched_.target.weights.backend;
    resp["weight_backend"] = wb ? wb->id() : "unknown";
    resp["weight_quant"] = wb ? wb->description() : "unknown";
    resp["tokenizer"] = "native_cpp_bpe";
    resp["chat_template"] = tok_.has_chat_template() ? "minja" : "none";
    resp["detokenizer"] = "native_cpp_bpe";
    resp["internal_states"] = Json::Array{std::move(st)};
    send_response(fd, 200, "OK", "application/json", resp.dump());
}

void HttpServer::handle_completions(int fd, const std::string& body, bool is_chat) {
    Json req_json;
    try {
        req_json = Json::parse(body);
        if (!req_json.is_object()) throw std::runtime_error("Body must be JSON object");
    } catch (const std::exception& e) {
        send_json_error(fd, 400, "Bad Request", std::string("Invalid JSON: ") + e.what());
        return;
    }

    // Tokenize natively in C++ (minja chat template + utf8proc + PCRE2 + ByteLevel BPE)
    std::vector<int32_t> prompt_ids;
    bool start_in_think = false;
    int sys_prefix_len = 0;
    bool enable_tool_parsing = false;
    Json effective_tools;
    const Json* effective_tools_ptr = nullptr;
    try {
        if (is_chat) {
            if (!req_json.contains("messages") || !req_json.at("messages").is_array() ||
                req_json.at("messages").as_array().empty()) {
                send_json_error(fd, 400, "Bad Request", "messages must be a non-empty array");
                return;
            }
            bool tool_choice_none = false;
            std::string forced_tool_name;
            if (req_json.contains("tool_choice")) {
                const Json& tc = req_json.at("tool_choice");
                if (tc.is_string() && tc.as_string() == "none") {
                    tool_choice_none = true;
                } else if (tc.is_object() && tc.contains("function") && tc.at("function").is_object() &&
                           tc.at("function").contains("name") && tc.at("function").at("name").is_string()) {
                    forced_tool_name = tc.at("function").at("name").as_string();
                }
            }
            if (!tool_choice_none && req_json.contains("tools") &&
                req_json.at("tools").is_array() && !req_json.at("tools").as_array().empty()) {
                if (!forced_tool_name.empty()) {
                    Json::Array filtered;
                    for (const auto& t : req_json.at("tools").as_array()) {
                        if (t.is_object() && t.contains("function") && t.at("function").is_object() &&
                            t.at("function").contains("name") && t.at("function").at("name").is_string() &&
                            t.at("function").at("name").as_string() == forced_tool_name) {
                            filtered.push_back(t);
                        }
                    }
                    effective_tools = filtered.empty() ? req_json.at("tools") : Json(std::move(filtered));
                } else {
                    effective_tools = req_json.at("tools");
                }
                effective_tools_ptr = &effective_tools;
                enable_tool_parsing = true;
            }

            Json merged_ctk = Json::Object{};
            if (req_json.contains("chat_template_kwargs") && req_json.at("chat_template_kwargs").is_object()) {
                merged_ctk = req_json.at("chat_template_kwargs");
            }
            if (!merged_ctk.contains("reasoning_effort") && req_json.contains("reasoning_effort") &&
                req_json.at("reasoning_effort").is_string()) {
                merged_ctk["reasoning_effort"] = req_json.at("reasoning_effort").as_string();
            }
            if (!merged_ctk.contains("enable_thinking") && req_json.contains("enable_thinking") &&
                req_json.at("enable_thinking").is_bool()) {
                merged_ctk["enable_thinking"] = req_json.at("enable_thinking").as_bool();
            }
            const Json* ctk_ptr = merged_ctk.as_object().empty() ? nullptr : &merged_ctk;
            ChatRenderResult cr = tok_.render_chat(req_json.at("messages"), effective_tools_ptr, ctk_ptr, true);
            prompt_ids = std::move(cr.token_ids);
            start_in_think = cr.in_think;
            sys_prefix_len = cr.sys_prefix_len;
        } else {
            if (!req_json.contains("prompt")) {
                send_json_error(fd, 400, "Bad Request", "Missing prompt field");
                return;
            }
            if (req_json.at("prompt").is_string()) {
                prompt_ids = tok_.encode(req_json.at("prompt").as_string(), false);
            } else if (req_json.at("prompt").is_array()) {
                for (const auto& x : req_json.at("prompt").as_array()) prompt_ids.push_back((int32_t)x.as_int());
            } else {
                send_json_error(fd, 400, "Bad Request", "prompt must be string or int array");
                return;
            }
        }
    } catch (const std::exception& e) {
        send_json_error(fd, 400, "Bad Request", std::string("Tokenization failed: ") + e.what());
        return;
    }

    if (prompt_ids.empty()) {
        send_json_error(fd, 400, "Bad Request", "Empty prompt after tokenization");
        return;
    }

    // Context length check / auto-truncate
    int max_ctx = sched_.cfg.context_length;
    if ((int)prompt_ids.size() >= max_ctx) {
        if (sched_.cfg.allow_auto_truncate) {
            prompt_ids.erase(prompt_ids.begin(), prompt_ids.begin() + ((int)prompt_ids.size() - (max_ctx - 2)));
            sys_prefix_len = 0;
        } else {
            send_json_error(fd, 422, "Unprocessable Entity", "Prompt exceeds context_length");
            return;
        }
    }

    SamplingParams sp;
    if (req_json.contains("temperature") && req_json.at("temperature").is_number())
        sp.temperature = (float)req_json.at("temperature").as_double();
    if (req_json.contains("top_k") && req_json.at("top_k").is_number())
        sp.top_k = (int)req_json.at("top_k").as_int();
    if (req_json.contains("top_p") && req_json.at("top_p").is_number())
        sp.top_p = (float)req_json.at("top_p").as_double();
    if (req_json.contains("max_tokens") && req_json.at("max_tokens").is_number())
        sp.max_tokens = (int)req_json.at("max_tokens").as_int();
    else if (req_json.contains("max_completion_tokens") && req_json.at("max_completion_tokens").is_number())
        sp.max_tokens = (int)req_json.at("max_completion_tokens").as_int();
    else
        sp.max_tokens = std::max(1, std::min(32768, max_ctx - (int)prompt_ids.size()));
    if (sp.max_tokens < 1) sp.max_tokens = 1;
    if (req_json.contains("seed") && req_json.at("seed").is_number()) {
        sp.seed = (uint64_t)req_json.at("seed").as_int();
        sp.has_seed = true;
    }
    if (req_json.contains("stop")) {
        if (req_json.at("stop").is_string()) sp.stop_strings.push_back(req_json.at("stop").as_string());
        else if (req_json.at("stop").is_array()) {
            for (const auto& s : req_json.at("stop").as_array())
                if (s.is_string() && !s.as_string().empty()) sp.stop_strings.push_back(s.as_string());
        }
    }

    bool stream = req_json.contains("stream") && req_json.at("stream").as_bool(false);
    bool include_usage = false;
    if (req_json.contains("stream_options") && req_json.at("stream_options").is_object()) {
        const auto& so = req_json.at("stream_options");
        if (so.contains("include_usage")) include_usage = so.at("include_usage").as_bool(false);
    }

    int prompt_tokens = (int)prompt_ids.size();
    auto req = sched_.submit(std::move(prompt_ids), sp, sys_prefix_len);
    std::string req_id = (is_chat ? "chatcmpl-" : "cmpl-") + std::to_string(req->id);
    int64_t created = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    char tc_prefix_buf[64];
    snprintf(tc_prefix_buf, sizeof(tc_prefix_buf), "call_%08llx_%llu_",
             (unsigned long long)(created & 0xFFFFFFFFULL), (unsigned long long)req->id);
    std::string tool_id_prefix(tc_prefix_buf);

    if (stream) {
        std::string hdr =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n"
            "Access-Control-Allow-Origin: *\r\n\r\n";
        if (!send_all(fd, hdr.data(), hdr.size())) {
            req->cancelled.store(true);
            return;
        }

        std::vector<int32_t> all_ids;
        std::string full_text;
        std::string prev_text;
        std::string prev_reasoning;
        std::string prev_content;
        std::string finish_reason = "stop";
        bool sent_role = false;
        Qwen3CoderStreamParser tool_parser(effective_tools_ptr, tool_id_prefix);

        auto send_chat_delta = [&](const std::string& dr, const std::string& dc, const ToolCallDelta* tc_delta) -> bool {
            Json chunk;
            chunk["id"] = req_id;
            chunk["object"] = "chat.completion.chunk";
            chunk["created"] = created;
            chunk["model"] = sched_.cfg.served_model_name;
            Json ch;
            ch["index"] = 0;
            Json d;
            if (!sent_role) { d["role"] = "assistant"; sent_role = true; }
            if (!dr.empty()) d["reasoning_content"] = dr;
            if (!dc.empty()) d["content"] = dc;
            if (tc_delta != nullptr) {
                d["tool_calls"] = Json::Array{build_tool_delta_json(*tc_delta)};
            }
            ch["delta"] = std::move(d);
            ch["finish_reason"] = nullptr;
            chunk["choices"] = Json::Array{std::move(ch)};
            return send_sse_chunk(fd, chunk.dump());
        };

        BpeDetokenizer::StreamDecoder stream_dec(tok_.detokenizer(), true);

        std::vector<TokenEvent> batch;
        bool done = false;
        while (!done) {
            size_t prev_full_len = full_text.size();
            req->pop_events_batch(batch);
            for (const auto& ev : batch) {
                if (ev.finished) {
                    finish_reason = ev.finish_reason;
                    done = true;
                    break;
                }
                all_ids.push_back(ev.token_id);
                stream_dec.push(ev.token_id, full_text);
            }
            if (done) {
                stream_dec.finish(full_text);
            }
            if (all_ids.empty()) {
                if (done) break;
                continue;
            }

            // Native incremental decoder holds incomplete multi-byte UTF-8 sequences in pending_
            // and skips special tokens in O(1); skip emitting when no new valid UTF-8 arrived yet.
            if (!done && full_text.size() == prev_full_len) {
                continue;
            }

            // Check stop strings
            bool hit_stop = false;
            for (const auto& st : sp.stop_strings) {
                size_t p = full_text.find(st);
                if (p != std::string::npos) {
                    full_text = full_text.substr(0, p);
                    hit_stop = true;
                    finish_reason = "stop";
                    req->cancelled.store(true);
                    break;
                }
            }

            if (is_chat) {
                // Streaming Qwen3 <think>...</think> reasoning split!
                // Handles both prompt-opened `<think>` (Qwen3 default when enable_thinking=true) and inline `<think>`.
                std::string cur_r, cur_c;
                size_t th_start = full_text.find("<think>");
                size_t th_end = full_text.find("</think>");
                bool has_closed_think = (th_end != std::string::npos) &&
                                        (start_in_think || (th_start != std::string::npos && th_start < th_end));
                if (has_closed_think) {
                    size_t r0 = (th_start != std::string::npos && th_start < th_end) ? (th_start + 7) : 0;
                    cur_r = full_text.substr(r0, th_end - r0);
                    cur_c = full_text.substr(th_end + 8);
                    while (!cur_c.empty() && (cur_c.front() == '\n' || cur_c.front() == '\r')) cur_c.erase(cur_c.begin());
                } else if (start_in_think || th_start != std::string::npos) {
                    size_t r0 = (th_start != std::string::npos) ? (th_start + 7) : 0;
                    cur_r = full_text.substr(r0);
                    // Hold back trailing partial "</think>" tag prefix if stream still running
                    if (!done && !hit_stop) {
                        size_t lt = cur_r.rfind('<');
                        if (lt != std::string::npos && cur_r.size() - lt < 8) cur_r.resize(lt);
                    }
                } else {
                    cur_c = full_text;
                    if (!done && !hit_stop && cur_c.size() < 7 && std::string("<think>").compare(0, cur_c.size(), cur_c) == 0) {
                        cur_c.clear();
                    }
                }

                std::string dr = (cur_r.size() > prev_reasoning.size() && cur_r.compare(0, prev_reasoning.size(), prev_reasoning) == 0)
                    ? cur_r.substr(prev_reasoning.size()) : "";
                std::string dc = (cur_c.size() > prev_content.size() && cur_c.compare(0, prev_content.size(), prev_content) == 0)
                    ? cur_c.substr(prev_content.size()) : "";
                if (!dr.empty()) prev_reasoning = cur_r;
                if (!dc.empty()) prev_content = cur_c;

                if (enable_tool_parsing) {
                    StreamParseOutput parsed;
                    if (!dc.empty()) {
                        parsed = tool_parser.feed(dc);
                    }
                    if (done || hit_stop) {
                        StreamParseOutput tail = tool_parser.finish();
                        parsed.normal_text += tail.normal_text;
                        for (auto& td : tail.tool_deltas) parsed.tool_deltas.push_back(std::move(td));
                    }

                    if (!dr.empty() || !parsed.normal_text.empty() || (!sent_role && parsed.tool_deltas.empty())) {
                        if (!send_chat_delta(dr, parsed.normal_text, nullptr)) {
                            req->cancelled.store(true);
                            return;
                        }
                    }
                    for (const auto& td : parsed.tool_deltas) {
                        if (!send_chat_delta("", "", &td)) {
                            req->cancelled.store(true);
                            return;
                        }
                    }
                } else {
                    if (!dr.empty() || !dc.empty() || !sent_role) {
                        if (!send_chat_delta(dr, dc, nullptr)) {
                            req->cancelled.store(true);
                            return;
                        }
                    }
                }
            } else {
                std::string delta;
                if (full_text.size() > prev_text.size() && full_text.compare(0, prev_text.size(), prev_text) == 0) {
                    delta = full_text.substr(prev_text.size());
                    prev_text = full_text;
                }
                if (!delta.empty()) {
                    Json chunk;
                    chunk["id"] = req_id;
                    chunk["object"] = "text_completion";
                    chunk["created"] = created;
                    chunk["model"] = sched_.cfg.served_model_name;
                    Json ch;
                    ch["index"] = 0;
                    ch["text"] = delta;
                    ch["finish_reason"] = nullptr;
                    chunk["choices"] = Json::Array{std::move(ch)};
                    if (!send_sse_chunk(fd, chunk.dump())) {
                        req->cancelled.store(true);
                        return;
                    }
                }
            }
            if (hit_stop) break;
        }

        if (is_chat && enable_tool_parsing && tool_parser.has_tool_calls() && finish_reason == "stop") {
            finish_reason = "tool_calls";
        }

        // Emit final chunk with finish_reason (and usage if requested)
        Json fin;
        fin["id"] = req_id;
        fin["object"] = is_chat ? "chat.completion.chunk" : "text_completion";
        fin["created"] = created;
        fin["model"] = sched_.cfg.served_model_name;
        Json ch;
        ch["index"] = 0;
        if (is_chat) ch["delta"] = Json::Object{};
        else ch["text"] = "";
        ch["finish_reason"] = finish_reason;
        fin["choices"] = Json::Array{std::move(ch)};
        if (include_usage) {
            Json u;
            u["prompt_tokens"] = prompt_tokens;
            u["completion_tokens"] = (int)all_ids.size();
            u["total_tokens"] = prompt_tokens + (int)all_ids.size();
            fin["usage"] = std::move(u);
        }
        send_sse_chunk(fd, fin.dump());
        send_all(fd, "data: [DONE]\n\n", 14);
    } else {
        std::vector<int32_t> all_ids;
        std::string finish_reason = "stop";
        TokenEvent ev;
        while (req->pop_event(ev)) {
            if (ev.finished) { finish_reason = ev.finish_reason; break; }
            all_ids.push_back(ev.token_id);
        }
        std::string full_text = tok_.decode(all_ids, true);
        for (const auto& st : sp.stop_strings) {
            size_t p = full_text.find(st);
            if (p != std::string::npos) {
                full_text = full_text.substr(0, p);
                finish_reason = "stop";
                break;
            }
        }

        Json resp;
        resp["id"] = req_id;
        resp["object"] = is_chat ? "chat.completion" : "text_completion";
        resp["created"] = created;
        resp["model"] = sched_.cfg.served_model_name;
        Json ch;
        ch["index"] = 0;
        if (is_chat) {
            auto [reasoning, content] = split_qwen3_reasoning(full_text, start_in_think);
            Json msg;
            msg["role"] = "assistant";
            if (!reasoning.empty()) msg["reasoning_content"] = reasoning;

            if (enable_tool_parsing) {
                auto parsed = parse_qwen3_coder_non_stream(content, effective_tools_ptr, tool_id_prefix);
                if (!parsed.tool_calls.empty()) {
                    if (parsed.normal_text.empty()) msg["content"] = nullptr;
                    else msg["content"] = parsed.normal_text;

                    Json::Array tc_arr;
                    tc_arr.reserve(parsed.tool_calls.size());
                    for (const auto& tc : parsed.tool_calls) {
                        Json item;
                        item["id"] = tc.id;
                        item["type"] = "function";
                        Json fn;
                        fn["name"] = tc.name;
                        fn["arguments"] = tc.arguments;
                        item["function"] = std::move(fn);
                        tc_arr.push_back(std::move(item));
                    }
                    msg["tool_calls"] = std::move(tc_arr);
                    if (finish_reason == "stop") finish_reason = "tool_calls";
                } else {
                    msg["content"] = parsed.normal_text;
                }
            } else {
                msg["content"] = content;
            }
            ch["message"] = std::move(msg);
        } else {
            ch["text"] = full_text;
        }
        ch["finish_reason"] = finish_reason;
        resp["choices"] = Json::Array{std::move(ch)};
        Json u;
        u["prompt_tokens"] = prompt_tokens;
        u["completion_tokens"] = (int)all_ids.size();
        u["total_tokens"] = prompt_tokens + (int)all_ids.size();
        resp["usage"] = std::move(u);

        send_response(fd, 200, "OK", "application/json", resp.dump());
    }
}

} // namespace amperegen
