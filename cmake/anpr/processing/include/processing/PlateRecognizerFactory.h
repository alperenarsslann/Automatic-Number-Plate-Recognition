/**
 * @file PlateRecognizerFactory.h
 * @brief Creates the IPlateRecognizer selected by the active model profile.
 */
#pragma once

#include <memory>

#include "core/Config.h"
#include "core/interfaces/IPlateRecognizer.h"

namespace anpr {

/**
 * @brief Instantiate the recognizer for the active model profile.
 *
 * Reads ProcessingConfig::activeModelProfile, looks up the profile and
 * dispatches on ModelProfile::engine (e.g. "opencv_dnn" ->
 * OpenCvDnnPlateRecognizer). All model parameters (paths, tensor sizes,
 * preprocessing, thresholds, charset) are injected from the profile —
 * switching to a fine-tuned Turkish model is a config change only.
 *
 * NOTE: Declared in step 1; implemented in step 3 together with the first
 * engine.
 *
 * @throws std::runtime_error on unknown profile or engine type.
 */
std::unique_ptr<IPlateRecognizer> createPlateRecognizer(const ProcessingConfig& config);

} // namespace anpr
