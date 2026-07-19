#include "capture/HikvisionRtspFrameSource.h"

#include <chrono>
#include <cstdlib>
#include <regex>

#include <opencv2/videoio.hpp>

#include "core/Logger.h"

namespace anpr {

namespace {

constexpr const char* kComponent = "Capture";

/// FFmpeg demuxer options are passed through this environment variable
/// (read by OpenCV when a capture is opened). "timeout" is the RTSP socket
/// I/O timeout in microseconds — without it a dead camera blocks read()
/// forever instead of failing so the reconnect logic can kick in.
void applyFfmpegOptions(const std::string& transport) {
    std::string options = "timeout;5000000";
    if (transport == "tcp") {
        options = "rtsp_transport;tcp|" + options;
    } else if (transport == "udp") {
        options = "rtsp_transport;udp|" + options;
    }
#ifdef _WIN32
    _putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", options.c_str());
#else
    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", options.c_str(), 1);
#endif
}

} // namespace

std::string sanitizeUrl(const std::string& url) {
    static const std::regex credentials(R"((\w+://)[^/@]*@)");
    return std::regex_replace(url, credentials, "$1***@");
}

HikvisionRtspFrameSource::HikvisionRtspFrameSource(CameraConfig config)
    : config_(std::move(config)) {}

HikvisionRtspFrameSource::~HikvisionRtspFrameSource() {
    close();
}

const std::string& HikvisionRtspFrameSource::activeUrl() const {
    return config_.substreamUrl.empty() ? config_.rtspUrl : config_.substreamUrl;
}

bool HikvisionRtspFrameSource::open() {
    close();

    if (activeUrl().empty()) {
        lastError_ = "camera.rtsp_url is empty";
        return false;
    }

    applyFfmpegOptions(config_.transport);
    capture_ = std::make_unique<cv::VideoCapture>(activeUrl(), cv::CAP_FFMPEG);
    if (!capture_->isOpened()) {
        lastError_ = "cannot open RTSP stream: " + sanitizeUrl(activeUrl()) +
                     " (check address, credentials and that the camera is reachable)";
        capture_.reset();
        return false;
    }

    lastError_.clear();
    LOG_INFO(kComponent, "RTSP stream opened: ", sanitizeUrl(activeUrl()),
             " (", capture_->get(cv::CAP_PROP_FRAME_WIDTH), "x",
             capture_->get(cv::CAP_PROP_FRAME_HEIGHT),
             ", reported ", capture_->get(cv::CAP_PROP_FPS), " fps, transport ",
             config_.transport, ")");
    return true;
}

bool HikvisionRtspFrameSource::getNextFrame(Frame& frame) {
    if (!capture_) {
        lastError_ = "source is not open";
        return false;
    }

    cv::Mat image;
    if (!capture_->read(image) || image.empty()) {
        // Timeout or stream error — the owning capture thread reconnects.
        lastError_ = "RTSP read failed/stalled: " + sanitizeUrl(activeUrl());
        return false;
    }

    // Clone: VideoCapture may reuse its internal buffer on the next read(),
    // and frames outlive this call once they enter the pipeline queue.
    frame.image = image.clone();
    frame.sequence = nextSequence_++;
    frame.timestamp = std::chrono::system_clock::now();
    frame.sourceId = id();
    return true;
}

void HikvisionRtspFrameSource::close() {
    if (capture_) {
        capture_->release();
        capture_.reset();
    }
}

bool HikvisionRtspFrameSource::isOpen() const {
    return capture_ != nullptr;
}

std::string HikvisionRtspFrameSource::id() const {
    // "camera:<host>" — host without credentials.
    static const std::regex hostPattern(R"(\w+://(?:[^/@]*@)?([^/:]+))");
    std::smatch match;
    if (std::regex_search(config_.rtspUrl, match, hostPattern)) {
        return "camera:" + match[1].str();
    }
    return "camera:rtsp";
}

std::string HikvisionRtspFrameSource::lastError() const {
    return lastError_;
}

} // namespace anpr
