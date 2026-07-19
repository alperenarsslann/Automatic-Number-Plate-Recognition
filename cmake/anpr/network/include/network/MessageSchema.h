/**
 * @file MessageSchema.h
 * @brief Versioned JSON wire schema between the device and the site server.
 */
#pragma once

#include <string>

#include "core/Types.h"

namespace anpr {

/// Wire schema version (bump on breaking changes; receivers check "v").
constexpr int kMessageSchemaVersion = 1;

/**
 * @brief Serialize a plate detection into an outbound wire message.
 *
 * Shape (newline framing added by the transport):
 *   { "v":1, "type":"plate_detection",
 *     "payload": { "plate": "...", "camera": "...",
 *                  "detection_confidence": 0.0, "ocr_confidence": 0.0,
 *                  "bbox": [x,y,w,h], "timestamp": "ISO-8601",
 *                  "frame": 0 } }
 */
std::string buildPlateMessage(const PlateDetection& detection);

/// Device heartbeat: lets the server track liveness between detections.
std::string buildHeartbeatMessage(std::size_t camerasOnline, std::uint64_t totalReports);

/**
 * @brief A command parsed from an inbound server message.
 *
 * Inbound shape:
 *   { "v":1, "type":"command",
 *     "payload": { "name": "set_alpr", "camera": "cam1", "enabled": false } }
 */
struct ServerCommand {
    bool valid = false;
    std::string name;        ///< e.g. "set_alpr", "ping".
    std::string cameraId;    ///< Target camera when applicable.
    bool boolArg = false;    ///< Generic boolean argument (e.g. enabled).
    std::string rawPayload;  ///< Original payload for command-specific parsing.
};

/// Parse an inbound message; ServerCommand::valid is false on malformed input.
ServerCommand parseServerCommand(const std::string& message);

} // namespace anpr
