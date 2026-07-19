#include "capture/HikvisionDiscovery.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <random>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
using socket_t = SOCKET;
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
constexpr int INVALID_SOCKET = -1;
#define closesocket ::close
#endif

#include "core/Logger.h"

namespace anpr {

namespace {

constexpr const char* kComponent = "Discovery";
constexpr const char* kSadpMulticastGroup = "239.255.255.250";
constexpr std::uint16_t kSadpPort = 37020;

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

std::string randomUuid() {
    static const char* hex = "0123456789abcdef";
    std::mt19937 rng(std::random_device{}());
    std::string uuid = "xxxxxxxx-xxxx-4xxx-8xxx-xxxxxxxxxxxx";
    for (auto& c : uuid) {
        if (c == 'x') c = hex[rng() % 16];
    }
    return uuid;
}

/// Extract <Tag>value</Tag> from a flat XML blob (SADP replies are flat).
std::string xmlValue(const std::string& xml, const std::string& tag) {
    const auto open = xml.find('<' + tag + '>');
    if (open == std::string::npos) return {};
    const auto start = open + tag.size() + 2;
    const auto end = xml.find("</" + tag + '>', start);
    if (end == std::string::npos) return {};
    return xml.substr(start, end - start);
}

void setNonBlocking(socket_t sock) {
#ifdef _WIN32
    u_long mode = 1;
    ::ioctlsocket(sock, FIONBIO, &mode);
#else
    ::fcntl(sock, F_SETFL, ::fcntl(sock, F_GETFL, 0) | O_NONBLOCK);
#endif
}

/// IPv4 /24 network bases ("192.168.1.") of all usable local interfaces.
std::vector<std::string> localSubnetBases() {
    std::vector<std::string> bases;
    const auto add = [&bases](std::uint32_t hostOrderIp) {
        const int a = (hostOrderIp >> 24) & 0xff;
        const int b = (hostOrderIp >> 16) & 0xff;
        const int c = (hostOrderIp >> 8) & 0xff;
        if (a == 127 || (a == 169 && b == 254)) return;          // loopback / link-local
        if (a == 100 && b >= 64 && b <= 127) return;             // CGNAT (VPN tunnels)
        std::string base = std::to_string(a) + '.' + std::to_string(b) + '.' +
                           std::to_string(c) + '.';
        if (std::find(bases.begin(), bases.end(), base) == bases.end()) {
            bases.push_back(std::move(base));
        }
    };
#ifdef _WIN32
    ULONG size = 0;
    ::GetAdaptersInfo(nullptr, &size);
    std::vector<char> buffer(size);
    auto* info = reinterpret_cast<IP_ADAPTER_INFO*>(buffer.data());
    if (size && ::GetAdaptersInfo(info, &size) == NO_ERROR) {
        for (auto* adapter = info; adapter; adapter = adapter->Next) {
            for (auto* addr = &adapter->IpAddressList; addr; addr = addr->Next) {
                in_addr ip{};
                if (::inet_pton(AF_INET, addr->IpAddress.String, &ip) == 1 && ip.s_addr) {
                    add(ntohl(ip.s_addr));
                }
            }
        }
    }
#else
    ifaddrs* interfaces = nullptr;
    if (::getifaddrs(&interfaces) == 0) {
        for (auto* ifa = interfaces; ifa; ifa = ifa->ifa_next) {
            if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET) {
                add(ntohl(reinterpret_cast<sockaddr_in*>(ifa->ifa_addr)->sin_addr.s_addr));
            }
        }
        ::freeifaddrs(interfaces);
    }
#endif
    return bases;
}

/// Which of @p hosts accept TCP connections on @p port (parallel connect scan).
std::vector<std::string> scanTcpPort(const std::vector<std::string>& hosts, std::uint16_t port,
                                     int timeoutMs) {
    std::vector<std::string> open;
    constexpr std::size_t kBatch = 128;
    for (std::size_t offset = 0; offset < hosts.size(); offset += kBatch) {
        std::vector<std::pair<socket_t, std::string>> pending;
        for (std::size_t i = offset; i < std::min(offset + kBatch, hosts.size()); ++i) {
            const socket_t sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (sock == INVALID_SOCKET) continue;
            setNonBlocking(sock);
            sockaddr_in dst{};
            dst.sin_family = AF_INET;
            dst.sin_port = htons(port);
            ::inet_pton(AF_INET, hosts[i].c_str(), &dst.sin_addr);
            ::connect(sock, reinterpret_cast<sockaddr*>(&dst), sizeof(dst)); // In progress.
            pending.emplace_back(sock, hosts[i]);
        }

        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        while (!pending.empty() && std::chrono::steady_clock::now() < deadline) {
            fd_set writeSet;
            FD_ZERO(&writeSet);
            socket_t maxFd = 0;
            for (const auto& [sock, host] : pending) {
                FD_SET(sock, &writeSet);
                maxFd = std::max(maxFd, sock);
            }
            timeval tv{0, 50 * 1000};
            if (::select(static_cast<int>(maxFd) + 1, nullptr, &writeSet, nullptr, &tv) <= 0) {
                continue;
            }
            for (auto it = pending.begin(); it != pending.end();) {
                if (!FD_ISSET(it->first, &writeSet)) {
                    ++it;
                    continue;
                }
                int soError = 0;
#ifdef _WIN32
                int len = sizeof(soError);
#else
                socklen_t len = sizeof(soError);
#endif
                ::getsockopt(it->first, SOL_SOCKET, SO_ERROR,
                             reinterpret_cast<char*>(&soError), &len);
                if (soError == 0) open.push_back(it->second);
                closesocket(it->first);
                it = pending.erase(it);
            }
        }
        for (const auto& [sock, host] : pending) closesocket(sock);
    }
    return open;
}

/// Fetch the ISAPI HTTP signature of a host; returns the device description
/// (from the Digest realm) or empty when the host does not look like a
/// Hikvision camera.
std::string isapiSignature(const std::string& host, std::uint16_t port, int timeoutMs) {
    const socket_t sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) return {};
    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_port = htons(port);
    ::inet_pton(AF_INET, host.c_str(), &dst.sin_addr);

    timeval tv{timeoutMs / 1000, (timeoutMs % 1000) * 1000};
    ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv), sizeof(tv));
    ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&tv), sizeof(tv));
    if (::connect(sock, reinterpret_cast<sockaddr*>(&dst), sizeof(dst)) != 0) {
        closesocket(sock);
        return {};
    }
    const std::string request = "GET /ISAPI/System/deviceInfo HTTP/1.1\r\nHost: " + host +
                                "\r\nConnection: close\r\n\r\n";
    ::send(sock, request.c_str(), static_cast<int>(request.size()), 0);
    std::string response;
    char buffer[2048];
    int received;
    while (response.size() < 8192 && (received = ::recv(sock, buffer, sizeof(buffer), 0)) > 0) {
        response.append(buffer, static_cast<std::size_t>(received));
    }
    closesocket(sock);

    // Hikvision web servers answer ISAPI paths with "Server: webserver" and a
    // Digest realm that names the device family.
    if (response.find("Server: webserver") == std::string::npos &&
        response.find("WWW-Authenticate: Digest") == std::string::npos) {
        return {};
    }
    const auto realmPos = response.find("realm=\"");
    if (realmPos != std::string::npos) {
        const auto start = realmPos + 7;
        const auto end = response.find('"', start);
        if (end != std::string::npos) return response.substr(start, end - start);
    }
    return "Hikvision device";
}

/// Firewall-friendly fallback: scan local /24s for hosts with the HCNetSDK
/// port AND RTSP open, then confirm via the ISAPI HTTP signature.
std::vector<DiscoveredCamera> tcpFallbackScan() {
    std::vector<DiscoveredCamera> result;
    for (const auto& base : localSubnetBases()) {
        LOG_INFO(kComponent, "TCP fallback scan on ", base, "0/24");
        std::vector<std::string> hosts;
        hosts.reserve(254);
        for (int i = 1; i <= 254; ++i) hosts.push_back(base + std::to_string(i));

        const auto sdkOpen = scanTcpPort(hosts, 8000, 700);
        if (sdkOpen.empty()) continue;
        const auto rtspOpen = scanTcpPort(sdkOpen, 554, 700);
        for (const auto& host : rtspOpen) {
            const std::string signature = isapiSignature(host, 80, 1500);
            if (signature.empty()) continue;
            DiscoveredCamera cam;
            cam.ipv4 = host;
            cam.description = signature;
            cam.discoveredVia = "tcp-scan";
            result.push_back(std::move(cam));
        }
    }
    return result;
}

} // namespace

std::vector<DiscoveredCamera> discoverHikvisionCameras(double timeoutSeconds,
                                                       std::string& error) {
    std::vector<DiscoveredCamera> result;
    error.clear();
    if (!initSockets()) {
        error = "socket subsystem initialization failed";
        return result;
    }

    const socket_t sock = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == INVALID_SOCKET) {
        error = "cannot create UDP socket";
        return result;
    }

    // Allow both multicast and broadcast probes; some networks filter one.
    const int enable = 1;
    ::setsockopt(sock, SOL_SOCKET, SO_BROADCAST, reinterpret_cast<const char*>(&enable),
                 sizeof(enable));
    ::setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&enable),
                 sizeof(enable));
    const int ttl = 2;
    ::setsockopt(sock, IPPROTO_IP, IP_MULTICAST_TTL, reinterpret_cast<const char*>(&ttl),
                 sizeof(ttl));

    // Devices send their ProbeMatch back to the multicast GROUP, not to the
    // prober's address — so we must sit on port 37020 and join the group to
    // hear them (this is what the SADP tool itself does).
    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_addr.s_addr = INADDR_ANY;
    local.sin_port = htons(kSadpPort);
    if (::bind(sock, reinterpret_cast<sockaddr*>(&local), sizeof(local)) != 0) {
        error = "cannot bind UDP port 37020 (is another SADP tool running?)";
        closesocket(sock);
        return result;
    }
    ip_mreq membership{};
    ::inet_pton(AF_INET, kSadpMulticastGroup, &membership.imr_multiaddr);
    membership.imr_interface.s_addr = INADDR_ANY;
    ::setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                 reinterpret_cast<const char*>(&membership), sizeof(membership));

    const std::string probe =
        "<?xml version=\"1.0\" encoding=\"utf-8\"?><Probe><Uuid>" + randomUuid() +
        "</Uuid><Types>inquiry</Types></Probe>";

    const auto sendTo = [&](const char* address) {
        sockaddr_in dst{};
        dst.sin_family = AF_INET;
        dst.sin_port = htons(kSadpPort);
        ::inet_pton(AF_INET, address, &dst.sin_addr);
        ::sendto(sock, probe.c_str(), static_cast<int>(probe.size()), 0,
                 reinterpret_cast<sockaddr*>(&dst), sizeof(dst));
    };
    sendTo(kSadpMulticastGroup);
    sendTo("255.255.255.255");

    // Collect replies until the timeout; devices may answer more than once.
    std::map<std::string, DiscoveredCamera> bySerial;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(timeoutSeconds));
    char buffer[8192];
    while (std::chrono::steady_clock::now() < deadline) {
        fd_set readSet;
        FD_ZERO(&readSet);
        FD_SET(sock, &readSet);
        timeval tv{0, 200 * 1000}; // 200 ms poll slices.
        const int ready = ::select(static_cast<int>(sock) + 1, &readSet, nullptr, nullptr, &tv);
        if (ready <= 0) continue;

        sockaddr_in from{};
#ifdef _WIN32
        int fromLen = sizeof(from);
#else
        socklen_t fromLen = sizeof(from);
#endif
        const int received = ::recvfrom(sock, buffer, sizeof(buffer) - 1, 0,
                                        reinterpret_cast<sockaddr*>(&from), &fromLen);
        if (received <= 0) continue;
        buffer[received] = '\0';
        const std::string xml(buffer, static_cast<std::size_t>(received));
        if (xml.find("<ProbeMatch>") == std::string::npos) continue;

        DiscoveredCamera cam;
        cam.ipv4 = xmlValue(xml, "IPv4Address");
        cam.description = xmlValue(xml, "DeviceDescription");
        cam.serialNumber = xmlValue(xml, "DeviceSN");
        cam.mac = xmlValue(xml, "MAC");
        cam.firmwareVersion = xmlValue(xml, "SoftwareVersion");
        cam.discoveredVia = "sadp";
        if (const auto port = xmlValue(xml, "HttpPort"); !port.empty()) {
            cam.httpPort = std::atoi(port.c_str());
        }
        if (const auto port = xmlValue(xml, "CommandPort"); !port.empty()) {
            cam.sdkPort = std::atoi(port.c_str());
        }
        if (cam.ipv4.empty()) {
            char fromIp[INET_ADDRSTRLEN] = {};
            ::inet_ntop(AF_INET, &from.sin_addr, fromIp, sizeof(fromIp));
            cam.ipv4 = fromIp;
        }
        const std::string key = cam.serialNumber.empty() ? cam.ipv4 : cam.serialNumber;
        bySerial[key] = std::move(cam);
    }
    closesocket(sock);

    result.reserve(bySerial.size());
    for (auto& [key, cam] : bySerial) {
        LOG_DEBUG(kComponent, "SADP found ", cam.description, " at ", cam.ipv4, " (", key, ")");
        result.push_back(std::move(cam));
    }

    // SADP replies are frequently swallowed by the Windows firewall / VPNs.
    // When nothing answered, fall back to the firewall-friendly TCP scan.
    if (result.empty()) {
        LOG_INFO(kComponent, "SADP returned nothing; trying TCP fallback scan");
        result = tcpFallbackScan();
    }
    return result;
}

} // namespace anpr
