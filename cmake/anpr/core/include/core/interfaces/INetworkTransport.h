/**
 * @file INetworkTransport.h
 * @brief Abstract interface for bidirectional communication with the site server.
 */
#pragma once

#include <functional>
#include <string>

namespace anpr {

/// Connection state reported by INetworkTransport.
enum class ConnectionState {
    Disconnected, ///< No connection; reconnect may be in progress.
    Connecting,   ///< Connection attempt in progress.
    Connected     ///< Live connection to the site server.
};

/**
 * @brief Abstract bidirectional, message-oriented transport.
 *
 * Protocol decision: the primary implementation is TCP (not UDP) because
 * plate reports must be lossless and ordered, the site server must be able
 * to push requests/config changes back to the device, and connection state
 * (connected/disconnected) must be observable — all properties TCP provides
 * natively. This interface hides the transport so a future switch to MQTT
 * or another protocol only requires a new implementation.
 *
 * Messages are opaque payload strings (JSON documents, framed by the
 * implementation, e.g. newline-delimited). Serialization/schema versioning
 * lives above this interface.
 *
 * Threading: send() may be called from any thread. Handlers are invoked on
 * the transport's internal thread and must not block.
 */
class INetworkTransport {
public:
    using MessageHandler = std::function<void(const std::string& payload)>;
    using StateHandler = std::function<void(ConnectionState state)>;

    virtual ~INetworkTransport() = default;

    /**
     * @brief Start the transport (connects and keeps reconnecting until stop()).
     * @return true if the transport was started; actual connection progress is
     *         reported via the state handler.
     */
    virtual bool start() = 0;

    /// Disconnect and stop all internal activity. Safe to call multiple times.
    virtual void stop() = 0;

    /// @return Current connection state.
    virtual ConnectionState state() const = 0;

    /**
     * @brief Enqueue an outgoing message.
     *
     * Implementations queue messages while disconnected (bounded) and flush
     * them after reconnect.
     * @return false if the message had to be dropped (queue full / stopped).
     */
    virtual bool send(const std::string& payload) = 0;

    /// Register handler for messages arriving from the site server.
    virtual void setMessageHandler(MessageHandler handler) = 0;

    /// Register handler for connection state transitions.
    virtual void setStateHandler(StateHandler handler) = 0;

    /// Description of the last error, or empty string if none occurred.
    virtual std::string lastError() const = 0;
};

} // namespace anpr
