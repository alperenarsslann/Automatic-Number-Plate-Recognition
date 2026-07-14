#include "capture/HikvisionRtspFrameSource.h"

namespace anpr {

// SKELETON: real RTSP connection code will be written once the Hikvision
// DS-2CD1023G0E-IF camera is available. See the header for the design notes.

HikvisionRtspFrameSource::HikvisionRtspFrameSource(CameraConfig config)
    : config_(std::move(config)) {}

bool HikvisionRtspFrameSource::open() {
    // TODO: cv::VideoCapture(config_.rtspUrl, cv::CAP_FFMPEG) with
    //       rtsp_transport option from config_.transport.
    lastError_ = "HikvisionRtspFrameSource is not implemented yet "
                 "(camera hardware not available); use --source=simulation";
    return false;
}

bool HikvisionRtspFrameSource::getNextFrame(Frame& /*frame*/) {
    // TODO: read with stall detection + reconnect (config_.reconnectIntervalSeconds).
    lastError_ = "HikvisionRtspFrameSource is not implemented yet";
    return false;
}

void HikvisionRtspFrameSource::close() {
    // TODO: release the stream.
}

bool HikvisionRtspFrameSource::isOpen() const {
    return false;
}

std::string HikvisionRtspFrameSource::id() const {
    return "camera:hikvision";
}

std::string HikvisionRtspFrameSource::lastError() const {
    return lastError_;
}

} // namespace anpr
