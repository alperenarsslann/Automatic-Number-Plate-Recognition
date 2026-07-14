/**
 * @file HikvisionRtspFrameSource.h
 * @brief IFrameSource implementation for the Hikvision IP camera (skeleton).
 *
 * Target hardware: Hikvision DS-2CD1023G0E-IF (Ethernet/PoE, ONVIF,
 * H.265/H.264, ISAPI snapshot). Real connection code will be written once
 * the camera is available; upper layers already program against
 * IFrameSource, so plugging this in requires no changes elsewhere.
 */
#pragma once

#include <string>

#include "core/Config.h"
#include "core/interfaces/IFrameSource.h"

namespace anpr {

/**
 * @brief RTSP frame source for the Hikvision camera. SKELETON ONLY.
 *
 * Planned implementation notes:
 *  - Open the RTSP main/sub stream with cv::VideoCapture(FFmpeg backend),
 *    e.g. "rtsp://user:pass@host:554/Streaming/Channels/101".
 *  - Force RTP-over-TCP when config.transport == "tcp" (set
 *    OPENCV_FFMPEG_CAPTURE_OPTIONS "rtsp_transport;tcp") to avoid packet
 *    loss artifacts on congested links.
 *  - Detect stalls (no frame within a timeout) and reconnect with
 *    config.reconnectIntervalSeconds backoff.
 *  - H.265/H.264 decoding is handled by the FFmpeg backend; consider a
 *    hardware decoder (VAAPI / embedded SoC) when moving to the target board.
 */
class HikvisionRtspFrameSource final : public IFrameSource {
public:
    explicit HikvisionRtspFrameSource(CameraConfig config);

    bool open() override;                 // TODO: connect to RTSP stream.
    bool getNextFrame(Frame& frame) override; // TODO: read + reconnect logic.
    void close() override;                // TODO: release stream.
    bool isOpen() const override;
    std::string id() const override;
    std::string lastError() const override;

private:
    CameraConfig config_;
    std::string lastError_;
};

} // namespace anpr
