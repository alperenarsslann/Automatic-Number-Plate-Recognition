/**
 * @file Types.h
 * @brief Common data types shared between all layers.
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace anpr {

/// Wall-clock timestamp type used throughout the system.
using Timestamp = std::chrono::system_clock::time_point;

/**
 * @brief A single video frame with acquisition metadata.
 *
 * Produced by the capture layer (IFrameSource) and consumed by the
 * processing layer. The image is always BGR, 8-bit, 3 channels.
 */
struct Frame {
    cv::Mat image;              ///< BGR image data (CV_8UC3).
    std::uint64_t sequence = 0; ///< Monotonic frame counter per source.
    Timestamp timestamp;        ///< Acquisition time (wall clock).
    std::string sourceId;       ///< Identifier of the producing source.
};

/**
 * @brief A single recognized license plate.
 *
 * Produced by the processing layer (IPlateRecognizer) and consumed by the
 * network layer, which serializes it into the wire JSON schema.
 */
struct PlateDetection {
    std::string text;               ///< Normalized plate text (e.g. "34ABC123").
    float detectionConfidence = 0;  ///< Plate detector confidence in [0, 1].
    float ocrConfidence = 0;        ///< OCR confidence in [0, 1].
    cv::Rect boundingBox;           ///< Plate location in the source frame.
    std::uint64_t frameSequence = 0;///< Sequence number of the source frame.
    Timestamp timestamp;            ///< Acquisition time of the source frame.
    std::string sourceId;           ///< Identifier of the producing source.
};

/// A batch of detections found in one frame (may be empty).
using PlateDetectionList = std::vector<PlateDetection>;

} // namespace anpr
