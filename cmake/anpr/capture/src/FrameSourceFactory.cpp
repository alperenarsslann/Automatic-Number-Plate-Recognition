#include "capture/FrameSourceFactory.h"

#include <stdexcept>

#include "capture/HikvisionRtspFrameSource.h"
#include "capture/SimulatedFileFrameSource.h"

namespace anpr {

std::unique_ptr<IFrameSource> createFrameSource(const CaptureConfig& config) {
    if (config.source == "simulation") {
        return std::make_unique<SimulatedFileFrameSource>(config.simulation);
    }
    if (config.source == "camera") {
        return std::make_unique<HikvisionRtspFrameSource>(config.camera);
    }
    throw std::runtime_error("unknown capture.source: \"" + config.source +
                             "\" (expected \"simulation\" or \"camera\")");
}

} // namespace anpr
