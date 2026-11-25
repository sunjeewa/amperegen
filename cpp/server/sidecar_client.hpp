#pragma once
#include "server/json_lite.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace amperegen {

class SidecarClient {
public:
    SidecarClient() = default;
    ~SidecarClient() { stop(); }

    void start(const std::string& python_bin, const std::string& script_path,
               const std::string& model_dir, const std::string& sock_path) {
        sock_path_ = sock_path;
        pid_ = fork();
        if (pid_ < 0) throw std::runtime_error("fork() failed for tokenizer sidecar");
        if (pid_ == 0) {
            execlp(python_bin.c_str(), python_bin.c_str(), script_path.c_str(),
                   "--model-dir", model_dir.c_str(),
                   "--socket", sock_path.c_str(),
                   (char*)nullptr);
            _exit(127);
        }

        // Wait up to 30 seconds for sidecar Unix socket to respond to ping
        for (int i = 0; i < 300; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            try {
                Json req;
                req["op"] = "ping";
                Json resp = rpc(req);
                if (resp.contains("ok") && resp.at("ok").as_bool()) {
                    fprintf(stderr, "[sidecar] Connected to tokenizer sidecar (pid=%d, sock=%s)\n",
                            (int)pid_, sock_path_.c_str());
                    return;
                }
            } catch (...) {}
        }
        stop();
        throw std::runtime_error("Tokenizer sidecar failed to become ready on " + sock_path_);
    }

    void connect_existing(const std::string& sock_path) {
        sock_path_ = sock_path;
        pid_ = -1;
    }

    void stop() {
        std::lock_guard<std::mutex> lk(mu_);
        if (fd_ >= 0) { close(fd_); fd_ = -1; }
        if (pid_ > 0) {
            kill(pid_, SIGTERM);
            int st = 0;
            waitpid(pid_, &st, 0);
            pid_ = -1;
        }
    }

    std::vector<int32_t> tokenize(const std::string& text, bool add_special = false) {
        Json req;
        req["op"] = "tokenize";
        req["text"] = text;
        req["add_special_tokens"] = add_special;
        Json resp = rpc(req);
        return extract_ids(resp);
    }

    std::vector<int32_t> render_chat(
        const Json& messages,
        const Json* tools,
        const Json* kwargs,
        bool* out_in_think = nullptr,
        int* out_sys_prefix_len = nullptr
    ) {
        Json req;
        req["op"] = "render_chat";
        req["messages"] = messages;
        req["add_generation_prompt"] = true;
        if (tools != nullptr && !tools->is_null()) req["tools"] = *tools;
        if (kwargs != nullptr && kwargs->is_object()) req["chat_template_kwargs"] = *kwargs;
        Json resp = rpc(req);
        if (out_in_think != nullptr) {
            *out_in_think = resp.contains("in_think") && resp.at("in_think").as_bool(false);
        }
        if (out_sys_prefix_len != nullptr) {
            *out_sys_prefix_len = resp.contains("sys_prefix_len") ? (int)resp.at("sys_prefix_len").as_int(0) : 0;
        }
        return extract_ids(resp);
    }

    std::string detokenize(const std::vector<int32_t>& ids, bool skip_special = true) {
        if (ids.empty()) return "";
        Json req;
        req["op"] = "detokenize";
        Json::Array arr;
        arr.reserve(ids.size());
        for (int32_t id : ids) arr.emplace_back((int64_t)id);
        req["token_ids"] = std::move(arr);
        req["skip_special_tokens"] = skip_special;
        Json resp = rpc(req);
        if (resp.contains("error")) throw std::runtime_error("sidecar detokenize: " + resp.at("error").as_string());
        return resp.at("text").as_string();
    }

private:
    std::string sock_path_;
    pid_t pid_ = -1;
    int fd_ = -1;
    std::mutex mu_;

    static std::vector<int32_t> extract_ids(const Json& resp) {
        if (resp.contains("error")) throw std::runtime_error("sidecar error: " + resp.at("error").as_string());
        const auto& arr = resp.at("token_ids").as_array();
        std::vector<int32_t> ids;
        ids.reserve(arr.size());
        for (const auto& x : arr) ids.push_back((int32_t)x.as_int());
        return ids;
    }

    void ensure_connected_locked() {
        if (fd_ >= 0) return;
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) throw std::runtime_error("socket(AF_UNIX) failed");
        struct sockaddr_un addr {};
        addr.sun_family = AF_UNIX;
        if (sock_path_.size() >= sizeof(addr.sun_path)) {
            close(fd);
            throw std::runtime_error("Unix socket path too long");
        }
        std::strncpy(addr.sun_path, sock_path_.c_str(), sizeof(addr.sun_path) - 1);
        if (connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
            close(fd);
            throw std::runtime_error("connect() failed on " + sock_path_);
        }
        fd_ = fd;
    }

    Json rpc(const Json& req) {
        std::lock_guard<std::mutex> lk(mu_);
        ensure_connected_locked();
        std::string payload = req.dump() + "\n";
        size_t sent = 0;
        while (sent < payload.size()) {
            ssize_t n = send(fd_, payload.data() + sent, payload.size() - sent, MSG_NOSIGNAL);
            if (n <= 0) {
                close(fd_); fd_ = -1;
                throw std::runtime_error("sidecar send failed");
            }
            sent += (size_t)n;
        }

        std::string line;
        char buf[4096];
        while (true) {
            ssize_t n = recv(fd_, buf, sizeof(buf), 0);
            if (n <= 0) {
                close(fd_); fd_ = -1;
                throw std::runtime_error("sidecar recv failed");
            }
            for (ssize_t i = 0; i < n; ++i) {
                if (buf[i] == '\n') {
                    return Json::parse(line);
                }
                line.push_back(buf[i]);
            }
        }
    }
};

} // namespace amperegen
