#include "processing/PlateRecognizerFactory.h"

#include <stdexcept>

#include "processing/OpenCvDnnPlateRecognizer.h"

namespace anpr {

std::unique_ptr<IPlateRecognizer> createPlateRecognizer(const ProcessingConfig& config) {
    const auto it = config.modelProfiles.find(config.activeModelProfile);
    if (it == config.modelProfiles.end()) {
        throw std::runtime_error("active_model_profile \"" + config.activeModelProfile +
                                 "\" is not defined in processing.model_profiles");
    }
    const ModelProfile& profile = it->second;

    if (profile.engine == "opencv_dnn") {
        return std::make_unique<OpenCvDnnPlateRecognizer>(config.activeModelProfile, profile);
    }
    // Future engines ("onnxruntime", "rknn", ...) plug in here.
    throw std::runtime_error("unknown recognizer engine \"" + profile.engine +
                             "\" in model profile \"" + config.activeModelProfile + '"');
}

} // namespace anpr
