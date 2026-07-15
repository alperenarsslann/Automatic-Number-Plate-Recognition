/**
 * @file MotionGate.h
 * @brief Cheap motion detector that gates the expensive DNN pipeline.
 */
#pragma once

#include <chrono>

#include <opencv2/core/mat.hpp>
#include <opencv2/video/background_segm.hpp>

#include "core/Config.h"

namespace anpr {

/**
 * @brief Frigate-style motion gating: background subtraction (MOG2) on a
 *        small grayscale copy decides whether a frame deserves inference.
 *
 * Cost per frame is a resize + subtractor update at ~320px — orders of
 * magnitude cheaper than a DNN forward. On a static scene the whole ALPR
 * chain is skipped; when motion appears, frames flow for at least
 * `cooldownSeconds` after the last movement so slowing/stopping vehicles
 * are not cut off mid-read.
 *
 * Not thread-safe; one instance per camera, owned by its capture thread.
 */
class MotionGate {
public:
    explicit MotionGate(MotionConfig config);

    /**
     * @brief Decide whether this frame should be processed.
     * @param bgr Full-resolution BGR frame.
     * @param[out] motionFraction Fraction of pixels in motion (diagnostics/HUD).
     * @return true when motion is present or the cooldown window is active.
     *         Always true when gating is disabled in the config.
     */
    bool shouldProcess(const cv::Mat& bgr, double& motionFraction);

private:
    MotionConfig config_;
    cv::Ptr<cv::BackgroundSubtractor> subtractor_;
    cv::Mat small_, mask_;
    std::chrono::steady_clock::time_point lastMotionAt_;
    bool sawMotion_ = false;
    int warmupFrames_ = 0; ///< MOG2 needs a few frames to build a background.
};

} // namespace anpr
