/**
 * @file TcpNetworkTransport.h
 * @brief INetworkTransport implementation over a persistent TCP connection.
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

#include "core/Config.h"
#include "core/interfaces/INetworkTransport.h"

namespace anpr {

/**
 * @brief TCP client transport to the site server.
 *
 * Design (see docs/ARCHITECTURE.md for the rationale of TCP over UDP):
 *  - Persistent client connection to NetworkConfig::serverHost:serverPort.
 *  - Newline-delimited JSON messages in both directions.
 *  - One internal I/O thread owns the socket: connects, reconnects with
 *    reconnectIntervalSeconds backoff, drains the bounded send queue, and
 *    reads inbound commands, dispatching each to the message handler.
 *  - Bounded send queue (drop-oldest): while disconnected, plate reports
 *    buffer up to sendQueueCapacity and flush on reconnect; the oldest is
 *    dropped past the bound so memory never grows without limit.
 *  - BSD/POSIX socket API with a thin Winsock compatibility shim so the
 *    same code builds on the Windows dev machine and the Linux target.
 */
class TcpNetworkTransport final : public INetworkTransport {
public:
    explicit TcpNetworkTransport(NetworkConfig config);
    ~TcpNetworkTransport() override;

    bool start() override;
    void stop() override;
    ConnectionState state() const override;
    bool send(const std::string& payload) override;
    void setMessageHandler(MessageHandler handler) override;
    void setStateHandler(StateHandler handler) override;
    std::string lastError() const override;

private:
    void ioLoop();
    void setState(ConnectionState state);
    /// Blocking connect to the configured endpoint; returns a socket fd or -1.
    long long connectToServer();
    /// Drain the queue and read inbound data until the socket dies or stop.
    void serveConnection(long long socketFd);

    NetworkConfig config_;
    MessageHandler messageHandler_;
    StateHandler stateHandler_;

    std::thread ioThread_;
    std::atomic<bool> running_{false};
    std::atomic<ConnectionState> state_{ConnectionState::Disconnected};

    mutable std::mutex mutex_;
    std::condition_variable queueCv_;   ///< Wakes the I/O thread on send/stop.
    std::deque<std::string> sendQueue_;
    std::string lastError_;
};

} // namespace anpr
