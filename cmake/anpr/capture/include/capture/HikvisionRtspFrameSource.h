/**
 * @file HikvisionRtspFrameSource.h
 * @brief IFrameSource implementation for Hikvision (and any RTSP) cameras.
 *
 * Design decision: video is pulled over standard RTSP through OpenCV's
 * FFmpeg backend instead of the proprietary HCNetSDK real-play path
 * (NET_DVR_RealPlay + PlayCtrl). RTSP keeps the code platform-independent
 * (the Windows-only SDK DLLs would block the Linux/embedded target), works
 * with every ONVIF/RTSP camera brand, and Hikvision exposes both streams as
 *   rtsp://user:pass@host:554/Streaming/Channels/101   (main)
 *   rtsp://user:pass@host:554/Streaming/Channels/102   (substream)
 * The HCNetSDK remains available for advanced device features (ISAPI
 * config, alarms); LAN discovery is handled separately via the SADP
 * protocol (see HikvisionDiscovery.h).
 */
#pragma once

#include <memory>
#include <string>

#include "core/Config.h"
#include "core/interfaces/IFrameSource.h"

namespace cv {
class VideoCapture;
}

namespace anpr {

/**
 * @brief RTSP frame source (FFmpeg backend).
 *
 * - RTP-over-TCP is forced when config.transport == "tcp" (default) to avoid
 *   UDP packet-loss artifacts; a socket timeout guards against dead streams
 *   blocking reads forever.
 * - When config.substreamUrl is set, THAT stream is opened for analysis (the
 *   standard NVR trick: low-res substream for detection, main stream stays
 *   available for recording/viewing).
 * - open()/getNextFrame() report failures instead of retrying internally;
 *   the camera's capture thread owns the reconnect policy (it also knows
 *   about shutdown), using config.reconnectIntervalSeconds.
 * - Credentials never reach the logs: URLs are sanitized before printing.
 */
class HikvisionRtspFrameSource final : public IFrameSource {
public:
    explicit HikvisionRtspFrameSource(CameraConfig config);
    ~HikvisionRtspFrameSource() override;

    bool open() override;
    bool getNextFrame(Frame& frame) override;
    void close() override;
    bool isOpen() const override;
    std::string id() const override;
    std::string lastError() const override;

private:
    /// The URL actually opened (substream when configured, else main).
    const std::string& activeUrl() const;

    CameraConfig config_;
    std::unique_ptr<cv::VideoCapture> capture_;
    std::uint64_t nextSequence_ = 0;
    std::string lastError_;
};

/// Strip "user:password@" from an RTSP/HTTP URL so it can be logged safely.
std::string sanitizeUrl(const std::string& url);

} // namespace anpr
