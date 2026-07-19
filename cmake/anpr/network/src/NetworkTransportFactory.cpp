#include "network/NetworkTransportFactory.h"

#include "network/TcpNetworkTransport.h"

namespace anpr {

std::unique_ptr<INetworkTransport> createNetworkTransport(const NetworkConfig& config) {
    if (!config.enabled) {
        return nullptr;
    }
    return std::make_unique<TcpNetworkTransport>(config);
}

} // namespace anpr
