#include "capture/SimulatedFileFrameSource.h"

#include <filesystem>
#include <thread>

#include <opencv2/videoio.hpp>

#include "core/Logger.h"

namespace anpr {

namespace {
constexpr const char* kComponent = "Capture";
constexpr double kFallbackFps = 25.0; ///< Used when the container reports no FPS.
} // namespace

SimulatedFileFrameSource::SimulatedFileFrameSource(SimulationConfig config)
    : config_(std::move(config)) {}

SimulatedFileFrameSource::~SimulatedFileFrameSource() {
    close();
}

bool SimulatedFileFrameSource::open() {
    close();

    if (config_.videoPath.empty()) {
        lastError_ = "simulation.video_path is empty (set it in the config or pass --video=<path>)";
        return false;
    }

    if (!std::filesystem::exists(config_.videoPath)) {
        lastError_ = "video file not found: " + config_.videoPath + " (resolved to " +
                     std::filesystem::absolute(config_.videoPath).string() +
                     " — relative paths resolve against the current working directory)";
        return false;
    }

    capture_ = std::make_unique<cv::VideoCapture>(config_.videoPath);
    if (!capture_->isOpened()) {
        lastError_ = "cannot open video file (unsupported codec/container?): " + config_.videoPath;
        capture_.reset();
        return false;
    }

    nativeFps_ = capture_->get(cv::CAP_PROP_FPS);
    if (!(nativeFps_ > 0.0)) {
        LOG_WARN(kComponent, "video reports no FPS, assuming ", kFallbackFps);
        nativeFps_ = kFallbackFps;
    }

    if (config_.startTimeSeconds > 0.0) {
        capture_->set(cv::CAP_PROP_POS_MSEC, config_.startTimeSeconds * 1000.0);
    }

    nextSequence_ = 0;
    nextFrameDue_ = std::chrono::steady_clock::now();
    lastError_.clear();

    const double totalFrames = capture_->get(cv::CAP_PROP_FRAME_COUNT);
    LOG_INFO(kComponent, "opened ", config_.videoPath,
             " (", capture_->get(cv::CAP_PROP_FRAME_WIDTH), "x",
             capture_->get(cv::CAP_PROP_FRAME_HEIGHT),
             ", ", nativeFps_, " fps, ", static_cast<long long>(totalFrames), " frames)",
             config_.realtime ? "" : " [max-speed mode]",
             config_.loop ? " [loop]" : "");
    return true;
}

bool SimulatedFileFrameSource::getNextFrame(Frame& frame) {
    if (!capture_) {
        lastError_ = "source is not open";
        return false;
    }

    cv::Mat image;
    if (!capture_->read(image) || image.empty()) {
        if (!config_.loop) {
            // Normal end-of-stream: not an error, lastError_ stays empty.
            LOG_INFO(kComponent, "end of video reached (", nextSequence_, " frames delivered)");
            close();
            return false;
        }
        // Reopen instead of seeking: some backends (e.g. MSMF on Windows)
        // reject seeks once end-of-stream has been reached.
        LOG_DEBUG(kComponent, "end of video, rewinding (loop enabled)");
        capture_->release();
        if (!capture_->open(config_.videoPath)) {
            lastError_ = "failed to reopen for looping: " + config_.videoPath;
            close();
            return false;
        }
        if (config_.startTimeSeconds > 0.0) {
            capture_->set(cv::CAP_PROP_POS_MSEC, config_.startTimeSeconds * 1000.0);
        }
        if (!capture_->read(image) || image.empty()) {
            lastError_ = "failed to read after rewind: " + config_.videoPath;
            close();
            return false;
        }
    }

    // Skip frames without decoding delay for the delivered pace.
    for (int i = 0; i < config_.frameSkip; ++i) {
        capture_->grab();
    }

    if (config_.realtime) {
        // Pace delivery to the effective FPS, as a live camera would.
        const double pacingFps = (config_.targetFps > 0.0) ? config_.targetFps : nativeFps_;
        const auto interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>((1.0 + config_.frameSkip) / pacingFps));
        std::this_thread::sleep_until(nextFrameDue_);
        const auto now = std::chrono::steady_clock::now();
        nextFrameDue_ += interval;
        if (nextFrameDue_ < now) {
            nextFrameDue_ = now; // Fell behind (slow decode); don't try to catch up.
        }
    }

    // Clone: VideoCapture may reuse its internal buffer on the next read(),
    // and frames outlive this call once they enter the pipeline queue.
    frame.image = image.clone();
    frame.sequence = nextSequence_++;
    frame.timestamp = std::chrono::system_clock::now();
    frame.sourceId = id();
    return true;
}

void SimulatedFileFrameSource::close() {
    if (capture_) {
        capture_->release();
        capture_.reset();
    }
}

bool SimulatedFileFrameSource::isOpen() const {
    return capture_ != nullptr;
}

std::string SimulatedFileFrameSource::id() const {
    return "simulation:" + std::filesystem::path(config_.videoPath).filename().string();
}

std::string SimulatedFileFrameSource::lastError() const {
    return lastError_;
}

} // namespace anpr
