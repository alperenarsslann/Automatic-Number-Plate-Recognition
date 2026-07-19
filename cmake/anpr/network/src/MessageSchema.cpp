#include "network/MessageSchema.h"

#include <chrono>
#include <ctime>

#include <nlohmann/json.hpp>

namespace anpr {

namespace {

using nlohmann::json;

std::string isoTimestamp(Timestamp tp) {
    using namespace std::chrono;
    const auto ms = duration_cast<milliseconds>(tp.time_since_epoch()) % 1000;
    const std::time_t t = system_clock::to_time_t(tp);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    char out[40];
    std::snprintf(out, sizeof(out), "%s.%03dZ", buf, static_cast<int>(ms.count()));
    return out;
}

} // namespace

std::string buildPlateMessage(const PlateDetection& d) {
    const json message = {
        {"v", kMessageSchemaVersion},
        {"type", "plate_detection"},
        {"payload",
         {{"plate", d.text},
          {"camera", d.sourceId},
          {"detection_confidence", d.detectionConfidence},
          {"ocr_confidence", d.ocrConfidence},
          {"bbox", {d.boundingBox.x, d.boundingBox.y, d.boundingBox.width, d.boundingBox.height}},
          {"timestamp", isoTimestamp(d.timestamp)},
          {"frame", d.frameSequence}}}};
    return message.dump();
}

std::string buildHeartbeatMessage(std::size_t camerasOnline, std::uint64_t totalReports) {
    const json message = {
        {"v", kMessageSchemaVersion},
        {"type", "heartbeat"},
        {"payload",
         {{"cameras_online", camerasOnline},
          {"reports_total", totalReports},
          {"timestamp", isoTimestamp(std::chrono::system_clock::now())}}}};
    return message.dump();
}

ServerCommand parseServerCommand(const std::string& message) {
    ServerCommand command;
    json j;
    try {
        j = json::parse(message);
    } catch (const json::parse_error&) {
        return command; // invalid stays false
    }
    if (!j.contains("type") || j.value("type", "") != "command" || !j.contains("payload")) {
        return command;
    }
    const auto& payload = j.at("payload");
    command.valid = true;
    command.name = payload.value("name", "");
    command.cameraId = payload.value("camera", "");
    command.boolArg = payload.value("enabled", false);
    command.rawPayload = payload.dump();
    return command;
}

} // namespace anpr
