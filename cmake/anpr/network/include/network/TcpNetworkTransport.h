/**
 * @file TcpNetworkTransport.h
 * @brief INetworkTransport implementation over a persistent TCP connection.
 */
#pragma once

#include <string>

#include "core/Config.h"
#include "core/interfaces/INetworkTransport.h"

namespace anpr {

/**
 * @brief TCP client transport to the site server. SKELETON in step 1;
 *        implemented in step 4 together with the mock test server.
 *
 * Design (see docs/ARCHITECTURE.md for the rationale of TCP over UDP):
 *  - Persistent client connection to NetworkConfig::serverHost:serverPort.
 *  - Newline-delimited JSON messages in both directions.
 *  - Internal thread owns the socket: connects, reconnects with
 *    reconnectIntervalSeconds backoff, flushes the bounded send queue,
 *    reads inbound commands and dispatches them to the message handler.
 *  - BSD/POSIX socket API with a thin Winsock compatibility shim so the
 *    same code builds on the Windows dev machine and the Linux target.
 */
class TcpNetworkTransport final : public INetworkTransport {
public:
    explicit TcpNetworkTransport(NetworkConfig config);

    bool start() override;   // TODO: spawn I/O thread, begin connect loop.
    void stop() override;    // TODO: signal thread, close socket, join.
    ConnectionState state() const override;
    bool send(const std::string& payload) override; // TODO: enqueue.
    void setMessageHandler(MessageHandler handler) override;
    void setStateHandler(StateHandler handler) override;
    std::string lastError() const override;

private:
    NetworkConfig config_;
    MessageHandler messageHandler_;
    StateHandler stateHandler_;
    std::string lastError_;
};

} // namespace anpr
