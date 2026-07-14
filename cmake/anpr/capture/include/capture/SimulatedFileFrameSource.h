/**
 * @file SimulatedFileFrameSource.h
 * @brief IFrameSource implementation that replays a video file.
 */
#pragma once

#include <chrono>
#include <memory>
#include <string>

#include "core/Config.h"
#include "core/interfaces/IFrameSource.h"

namespace cv {
class VideoCapture;
}

namespace anpr {

/**
 * @brief Frame source that reads from a video file via cv::VideoCapture,
 *        standing in for the real camera during development.
 *
 * Behavior is fully driven by SimulationConfig:
 *  - realtime/targetFps: pace frames to wall clock (as a live camera would)
 *    or deliver them as fast as they can be decoded.
 *  - frameSkip: skip N frames after each delivered frame.
 *  - loop: rewind and continue at end of file instead of reporting EOS.
 *  - startTimeSeconds: seek before delivering the first frame.
 *
 * NOTE: Skeleton in step 1; implemented in step 2.
 */
class SimulatedFileFrameSource final : public IFrameSource {
public:
    explicit SimulatedFileFrameSource(SimulationConfig config);
    ~SimulatedFileFrameSource() override;

    bool open() override;
    bool getNextFrame(Frame& frame) override;
    void close() override;
    bool isOpen() const override;
    std::string id() const override;
    std::string lastError() const override;

private:
    SimulationConfig config_;
    std::unique_ptr<cv::VideoCapture> capture_;
    std::uint64_t nextSequence_ = 0;
    double nativeFps_ = 0.0;
    std::chrono::steady_clock::time_point nextFrameDue_{};
    std::string lastError_;
};

} // namespace anpr
