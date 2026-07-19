/**
 * @file HikvisionDiscovery.h
 * @brief LAN auto-discovery of Hikvision cameras via the SADP protocol.
 *
 * HCNetSDK itself offers no discovery API (SADP is a separate Hikvision
 * tool/protocol; the SDK only exposes on/off flags such as
 * byEnablePrivateMulticastDiscovery). Devices ship with SADP enabled and
 * answer an XML "inquiry" probe sent to UDP multicast 239.255.255.250:37020
 * with an XML description (model, serial, IP, ports, firmware). Speaking
 * that protocol directly keeps discovery dependency-free (no Windows-only
 * SDK DLLs) and portable to the Linux/embedded target.
 */
#pragma once

#include <string>
#include <vector>

namespace anpr {

/// One camera found on the LAN.
struct DiscoveredCamera {
    std::string ipv4;          ///< e.g. "192.168.1.174"
    std::string description;   ///< Model, e.g. "DS-2CD1023G0-IUF"
    std::string serialNumber;
    std::string mac;
    std::string firmwareVersion;
    int httpPort = 80;         ///< ISAPI/web port.
    int sdkPort = 8000;        ///< HCNetSDK command port.
    std::string discoveredVia; ///< "sadp" or "tcp-scan".

    /// Ready-to-edit RTSP URL templates for this device.
    std::string mainStreamUrl(const std::string& user = "admin",
                              const std::string& password = "<password>") const {
        return "rtsp://" + user + ":" + password + "@" + ipv4 + ":554/Streaming/Channels/101";
    }
};

/**
 * @brief Probe the local network for Hikvision devices.
 *
 * Two strategies, in order:
 *  1. SADP inquiry probe (multicast + broadcast, replies collected for
 *     @p timeoutSeconds). Native and instant, but inbound UDP replies are
 *     often eaten by the Windows firewall or VPN clients.
 *  2. TCP fallback when SADP stays silent: every local /24 subnet is scanned
 *     for hosts with BOTH the HCNetSDK port (8000) and RTSP (554) open; hits
 *     are identified via their ISAPI HTTP signature (Digest realm carries
 *     the model, e.g. realm="IP Camera(AX773)"). Outbound TCP passes
 *     firewalls, so this works where SADP cannot.
 *
 * @param timeoutSeconds SADP listen window; 2-3 s is plenty on a LAN.
 * @param error Filled with a description when probing could not run at all.
 */
std::vector<DiscoveredCamera> discoverHikvisionCameras(double timeoutSeconds,
                                                       std::string& error);

} // namespace anpr
