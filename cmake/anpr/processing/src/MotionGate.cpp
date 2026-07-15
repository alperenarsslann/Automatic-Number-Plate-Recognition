#include "processing/MotionGate.h"

#include <opencv2/imgproc.hpp>

namespace anpr {

namespace {
constexpr int kMog2History = 300;
constexpr double kMog2VarThreshold = 25.0;
constexpr int kWarmupFrames = 15; ///< Process unconditionally while the model settles.
} // namespace

MotionGate::MotionGate(MotionConfig config) : config_(std::move(config)) {
    if (config_.enabled) {
        // detectShadows=false: shadows would count as motion and cost extra CPU.
        subtractor_ = cv::createBackgroundSubtractorMOG2(kMog2History, kMog2VarThreshold,
                                                         /*detectShadows=*/false);
    }
}

bool MotionGate::shouldProcess(const cv::Mat& bgr, double& motionFraction) {
    motionFraction = 0.0;
    if (!config_.enabled || bgr.empty()) {
        return true;
    }

    const double scale = config_.downscaleWidth / static_cast<double>(bgr.cols);
    cv::resize(bgr, small_, cv::Size(), scale, scale, cv::INTER_NEAREST);
    cv::cvtColor(small_, small_, cv::COLOR_BGR2GRAY);
    subtractor_->apply(small_, mask_);

    // Suppress single-pixel noise before counting.
    cv::morphologyEx(mask_, mask_, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    motionFraction = cv::countNonZero(mask_) / static_cast<double>(mask_.total());

    if (warmupFrames_ < kWarmupFrames) {
        ++warmupFrames_;
        return true;
    }

    const auto now = std::chrono::steady_clock::now();
    if (motionFraction >= config_.minAreaFraction) {
        lastMotionAt_ = now;
        sawMotion_ = true;
        return true;
    }
    return sawMotion_ &&
           (now - lastMotionAt_) < std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                       std::chrono::duration<double>(config_.cooldownSeconds));
}

} // namespace anpr
