/**
 * @file mock_server.cpp
 * @brief Mock site-server for testing the ANPR network layer.
 *
 * Listens on a TCP port, accepts one device connection, prints every
 * newline-delimited JSON message the device sends (plate reports,
 * heartbeats), and lets the operator type commands back to the device:
 *
 *   Interactive stdin commands:
 *     alpr <cameraId> on|off   -> sends a set_alpr command to the device
 *     ping                     -> sends a ping command
 *     quit                     -> stop the server
 *
 * Standalone (only depends on the OS socket API + nlohmann/json) so it can
 * run anywhere the device can reach. Usage: mock_server [port] (default 9000).
 */
#include <atomic>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using socket_t = SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
constexpr socket_t INVALID_SOCKET = -1;
#define closesocket ::close
#endif

#include <nlohmann/json.hpp>

namespace {

std::atomic<socket_t> g_client{INVALID_SOCKET};
std::atomic<bool> g_running{true};

void sendLine(socket_t sock, const std::string& payload) {
    if (sock == INVALID_SOCKET) {
        std::cout << "(no device connected)\n";
        return;
    }
    const std::string framed = payload + "\n";
    ::send(sock, framed.data(), static_cast<int>(framed.size()), 0);
}

/// Read commands from stdin and forward them to the connected device.
void stdinLoop() {
    std::string line;
    std::cout << "Commands: alpr <cameraId> on|off | ping | quit\n";
    while (g_running.load() && std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string verb;
        iss >> verb;
        if (verb == "quit") {
            g_running.store(false);
            break;
        }
        if (verb == "ping") {
            sendLine(g_client.load(),
                     nlohmann::json{{"v", 1}, {"type", "command"},
                                    {"payload", {{"name", "ping"}}}}
                         .dump());
        } else if (verb == "alpr") {
            std::string camera, state;
            iss >> camera >> state;
            if (camera.empty() || (state != "on" && state != "off")) {
                std::cout << "usage: alpr <cameraId> on|off\n";
                continue;
            }
            sendLine(g_client.load(),
                     nlohmann::json{{"v", 1}, {"type", "command"},
                                    {"payload", {{"name", "set_alpr"},
                                                 {"camera", camera},
                                                 {"enabled", state == "on"}}}}
                         .dump());
            std::cout << "-> set_alpr " << camera << " " << state << "\n";
        } else if (!verb.empty()) {
            std::cout << "unknown command: " << verb << "\n";
        }
    }
}

void printInbound(const std::string& message) {
    try {
        const auto j = nlohmann::json::parse(message);
        const std::string type = j.value("type", "?");
        if (type == "plate_detection") {
            const auto& p = j.at("payload");
            std::cout << "[PLATE] " << p.value("plate", "?") << " cam="
                      << p.value("camera", "?") << " ocr=" << p.value("ocr_confidence", 0.0)
                      << " @ " << p.value("timestamp", "?") << "\n";
        } else if (type == "heartbeat") {
            const auto& p = j.at("payload");
            std::cout << "[HEARTBEAT] cameras_online=" << p.value("cameras_online", 0)
                      << " reports=" << p.value("reports_total", 0) << "\n";
        } else {
            std::cout << "[" << type << "] " << message << "\n";
        }
    } catch (const std::exception&) {
        std::cout << "[raw] " << message << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    const unsigned short port =
        (argc > 1) ? static_cast<unsigned short>(std::atoi(argv[1])) : 9000;

    // Flush stdout on every insertion so output is visible immediately even
    // when redirected to a file/pipe (full buffering otherwise hides it).
    std::cout.setf(std::ios::unitbuf);

#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    const socket_t listener = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listener == INVALID_SOCKET) {
        std::cerr << "cannot create socket\n";
        return 1;
    }
    const int reuse = 1;
    ::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&reuse),
                 sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (::bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        ::listen(listener, 1) != 0) {
        std::cerr << "cannot bind/listen on port " << port << "\n";
        closesocket(listener);
        return 1;
    }

    std::cout << "Mock ANPR site-server listening on port " << port << "\n";
    std::thread stdinThread(stdinLoop);

    while (g_running.load()) {
        std::cout << "Waiting for a device to connect...\n";
        sockaddr_in from{};
#ifdef _WIN32
        int fromLen = sizeof(from);
#else
        socklen_t fromLen = sizeof(from);
#endif
        const socket_t client = ::accept(listener, reinterpret_cast<sockaddr*>(&from), &fromLen);
        if (client == INVALID_SOCKET) {
            if (!g_running.load()) break;
            continue;
        }
        char ip[INET_ADDRSTRLEN] = {};
        ::inet_ntop(AF_INET, &from.sin_addr, ip, sizeof(ip));
        std::cout << "Device connected from " << ip << "\n";
        g_client.store(client);

        std::string inbound;
        char buffer[4096];
        int received;
        while (g_running.load() &&
               (received = ::recv(client, buffer, sizeof(buffer), 0)) > 0) {
            inbound.append(buffer, static_cast<std::size_t>(received));
            std::size_t newline;
            while ((newline = inbound.find('\n')) != std::string::npos) {
                std::string message = inbound.substr(0, newline);
                inbound.erase(0, newline + 1);
                if (!message.empty() && message.back() == '\r') message.pop_back();
                if (!message.empty()) printInbound(message);
            }
        }
        std::cout << "Device disconnected\n";
        g_client.store(INVALID_SOCKET);
        closesocket(client);
    }

    closesocket(listener);
    g_running.store(false);
    if (stdinThread.joinable()) stdinThread.detach(); // getline may block on stdin.
    return 0;
}
