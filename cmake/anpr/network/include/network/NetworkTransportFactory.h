/**
 * @file NetworkTransportFactory.h
 * @brief Creates the configured INetworkTransport implementation.
 */
#pragma once

#include <memory>

#include "core/Config.h"
#include "core/interfaces/INetworkTransport.h"

namespace anpr {

/**
 * @brief Instantiate the transport for the network config.
 *
 * Today this always builds a TcpNetworkTransport; a future MQTT/TLS
 * transport plugs in here behind the same interface without touching the
 * pipeline. Returns nullptr when networking is disabled.
 */
std::unique_ptr<INetworkTransport> createNetworkTransport(const NetworkConfig& config);

} // namespace anpr
