/**
 * @file IPlateRecognizer.h
 * @brief Abstract interface for license plate detection + OCR engines.
 */
#pragma once

#include <string>

#include "core/Types.h"

namespace anpr {

/**
 * @brief Abstract plate recognition engine (detection + OCR).
 *
 * Implementations are constructed by a factory that reads the active model
 * profile from the configuration (model paths, input sizes, preprocessing
 * parameters, thresholds, charset). This keeps engines swappable — e.g. an
 * OpenCV-DNN/ONNX implementation today, an ONNX Runtime or RKNN/NPU
 * implementation later — without touching upper layers, and lets fine-tuned
 * models (e.g. Turkish plates) be deployed via a config change only.
 *
 * Implementations are not required to be thread-safe; the processing thread
 * is the sole caller. Deduplication of repeated plates across consecutive
 * frames is NOT this interface's job; it is handled by a separate component
 * in the processing layer so it works uniformly for every engine.
 */
class IPlateRecognizer {
public:
    virtual ~IPlateRecognizer() = default;

    /**
     * @brief Load models and allocate resources.
     * @return true on success; on failure lastError() describes the cause.
     */
    virtual bool initialize() = 0;

    /**
     * @brief Detect and read all license plates in a frame.
     * @param frame Input frame (BGR).
     * @return Zero or more detections that passed the configured
     *         confidence thresholds.
     */
    virtual PlateDetectionList recognize(const Frame& frame) = 0;

    /// Human-readable engine/profile identifier used in logs.
    virtual std::string id() const = 0;

    /// Description of the last error, or empty string if none occurred.
    virtual std::string lastError() const = 0;
};

} // namespace anpr
