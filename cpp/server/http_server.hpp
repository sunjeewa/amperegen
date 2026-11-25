#pragma once
#include "scheduler/scheduler.hpp"
#include "server/sidecar_client.hpp"
#include <atomic>
#include <string>

namespace amperegen {

class HttpServer {
public:
    HttpServer(Scheduler& sched, SidecarClient& sidecar, const std::string& host, int port);
    ~HttpServer();

    void serve(); // blocking accept loop until stop()
    void stop();

private:
    Scheduler& sched_;
    SidecarClient& sidecar_;
    std::string host_;
    int port_;
    int listen_fd_ = -1;
    std::atomic<bool> stop_flag_{false};

    void handle_client(int client_fd);
    void handle_health(int fd);
    void handle_models(int fd);
    void handle_metrics(int fd);
    void handle_server_info(int fd);
    void handle_completions(int fd, const std::string& body, bool is_chat);
};

} // namespace amperegen
