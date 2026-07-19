#include "network/TcpNetworkTransport.h"

#include <chrono>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using socket_t = SOCKET;
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
constexpr socket_t INVALID_SOCKET = -1;
constexpr int SOCKET_ERROR = -1;
#define closesocket ::close
#endif

#include "core/Logger.h"

namespace anpr {

namespace {

constexpr const char* kComponent = "Network";

bool initSockets() {
#ifdef _WIN32
    static const bool ok = [] {
        WSADATA data;
        return WSAStartup(MAKEWORD(2, 2), &data) == 0;
    }();
    return ok;
#else
    return true;
#endif
}

} // namespace

TcpNetworkTransport::TcpNetworkTransport(NetworkConfig config) : config_(std::move(config)) {}

TcpNetworkTransport::~TcpNetworkTransport() {
    stop();
}

void TcpNetworkTransport::setMessageHandler(MessageHandler handler) {
    messageHandler_ = std::move(handler);
}

void TcpNetworkTransport::setStateHandler(StateHandler handler) {
    stateHandler_ = std::move(handler);
}

std::string TcpNetworkTransport::lastError() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lastError_;
}

ConnectionState TcpNetworkTransport::state() const {
    return state_.load();
}

void TcpNetworkTransport::setState(ConnectionState state) {
    const auto previous = state_.exchange(state);
    if (previous != state && stateHandler_) {
        stateHandler_(state);
    }
}

bool TcpNetworkTransport::start() {
    if (!initSockets()) {
        std::lock_guard<std::mutex> lock(mutex_);
        lastError_ = "socket subsystem initialization failed";
        return false;
    }
    if (running_.exchange(true)) {
        return true; // Already started.
    }
    ioThread_ = std::thread([this] { ioLoop(); });
    LOG_INFO(kComponent, "transport started, target ", config_.serverHost, ":",
             config_.serverPort);
    return true;
}

void TcpNetworkTransport::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    queueCv_.notify_all();
    if (ioThread_.joinable()) {
        ioThread_.join();
    }
    setState(ConnectionState::Disconnected);
    LOG_INFO(kComponent, "transport stopped");
}

bool TcpNetworkTransport::send(const std::string& payload) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_.load()) {
            return false;
        }
        if (sendQueue_.size() >= config_.sendQueueCapacity) {
            sendQueue_.pop_front(); // Drop-oldest: keep the freshest reports.
        }
        sendQueue_.push_back(payload);
    }
    queueCv_.notify_one();
    return true;
}

long long TcpNetworkTransport::connectToServer() {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    addrinfo* addresses = nullptr;
    const std::string port = std::to_string(config_.serverPort);
    if (::getaddrinfo(config_.serverHost.c_str(), port.c_str(), &hints, &addresses) != 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        lastError_ = "cannot resolve " + config_.serverHost;
        return -1;
    }

    socket_t sock = INVALID_SOCKET;
    for (addrinfo* a = addresses; a; a = a->ai_next) {
        sock = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (sock == INVALID_SOCKET) continue;
        if (::connect(sock, a->ai_addr, static_cast<int>(a->ai_addrlen)) == 0) {
            break;
        }
        closesocket(sock);
        sock = INVALID_SOCKET;
    }
    ::freeaddrinfo(addresses);

    if (sock == INVALID_SOCKET) {
        std::lock_guard<std::mutex> lock(mutex_);
        lastError_ = "connect failed to " + config_.serverHost + ":" +
                     std::to_string(config_.serverPort);
        return -1;
    }

    const int nodelay = 1; // Plate reports are small and latency-sensitive.
    ::setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&nodelay),
                 sizeof(nodelay));
    return static_cast<long long>(sock);
}

void TcpNetworkTransport::serveConnection(long long socketFd) {
    const auto sock = static_cast<socket_t>(socketFd);
    std::string inbound;
    char recvBuffer[4096];

    while (running_.load()) {
        // 1) Flush queued outgoing messages.
        std::string toSend;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            queueCv_.wait_for(lock, std::chrono::milliseconds(200),
                              [this] { return !sendQueue_.empty() || !running_.load(); });
            while (!sendQueue_.empty()) {
                toSend += sendQueue_.front();
                toSend.push_back('\n'); // Newline-delimited framing.
                sendQueue_.pop_front();
            }
        }
        for (std::size_t sent = 0; sent < toSend.size();) {
            const int n = ::send(sock, toSend.data() + sent,
                                 static_cast<int>(toSend.size() - sent), 0);
            if (n <= 0) {
                std::lock_guard<std::mutex> lock(mutex_);
                lastError_ = "send failed; connection lost";
                return;
            }
            sent += static_cast<std::size_t>(n);
        }

        // 2) Read any inbound data (non-blocking-ish via a short timeout).
        timeval tv{0, 50 * 1000};
        fd_set readSet;
        FD_ZERO(&readSet);
        FD_SET(sock, &readSet);
        const int ready = ::select(static_cast<int>(sock) + 1, &readSet, nullptr, nullptr, &tv);
        if (ready < 0) {
            std::lock_guard<std::mutex> lock(mutex_);
            lastError_ = "select failed; connection lost";
            return;
        }
        if (ready == 0) continue;

        const int received = ::recv(sock, recvBuffer, sizeof(recvBuffer), 0);
        if (received <= 0) {
            std::lock_guard<std::mutex> lock(mutex_);
            lastError_ = "server closed the connection";
            return;
        }
        inbound.append(recvBuffer, static_cast<std::size_t>(received));

        // Dispatch each complete newline-delimited message.
        std::size_t newline;
        while ((newline = inbound.find('\n')) != std::string::npos) {
            std::string message = inbound.substr(0, newline);
            inbound.erase(0, newline + 1);
            if (!message.empty() && message.back() == '\r') message.pop_back();
            if (!message.empty() && messageHandler_) {
                messageHandler_(message);
            }
        }
    }
}

void TcpNetworkTransport::ioLoop() {
    const auto reconnectDelay =
        std::chrono::duration<double>(config_.reconnectIntervalSeconds);

    while (running_.load()) {
        setState(ConnectionState::Connecting);
        const long long sock = connectToServer();
        if (sock < 0) {
            setState(ConnectionState::Disconnected);
            LOG_WARN(kComponent, "connect failed (", lastError(), "); retrying in ",
                     config_.reconnectIntervalSeconds, " s");
            // Interruptible sleep so stop() is responsive.
            std::unique_lock<std::mutex> lock(mutex_);
            queueCv_.wait_for(lock, std::chrono::duration_cast<std::chrono::milliseconds>(
                                        reconnectDelay),
                              [this] { return !running_.load(); });
            continue;
        }

        setState(ConnectionState::Connected);
        LOG_INFO(kComponent, "connected to ", config_.serverHost, ":", config_.serverPort);
        serveConnection(sock);
        closesocket(static_cast<socket_t>(sock));

        if (running_.load()) {
            setState(ConnectionState::Disconnected);
            LOG_WARN(kComponent, "disconnected (", lastError(), "); reconnecting in ",
                     config_.reconnectIntervalSeconds, " s");
            std::unique_lock<std::mutex> lock(mutex_);
            queueCv_.wait_for(lock, std::chrono::duration_cast<std::chrono::milliseconds>(
                                        reconnectDelay),
                              [this] { return !running_.load(); });
        }
    }
}

} // namespace anpr
