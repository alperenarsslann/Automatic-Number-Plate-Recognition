/**
 * @file OpenCvDnnPlateRecognizer.h
 * @brief IPlateRecognizer implementation running ONNX models via OpenCV DNN.
 */
#pragma once

#include <string>
#include <vector>

#include <opencv2/dnn.hpp>

#include "core/Config.h"
#include "core/interfaces/IPlateRecognizer.h"

namespace anpr {

/**
 * @brief Plate recognizer running a detection ONNX + an OCR ONNX with the
 *        OpenCV `dnn` module (no extra runtime dependency).
 *
 * Everything model-specific is injected via ModelProfile (paths, tensor
 * sizes, preprocessing, thresholds, charset), so a fine-tuned Turkish model
 * is deployed by editing the config only.
 *
 * Supported model families:
 *  - Detection: YOLO-style single-tensor output, either [1, 4+nc, N]
 *    (YOLOv8/v9/v11) or [1, N, 4+nc] (YOLOv5) — auto-detected. Boxes are
 *    letterboxed in and mapped back to source coordinates; final boxes go
 *    through OpenCV NMS with the configured IoU threshold.
 *  - OCR: fixed-slot classification heads (fast-plate-ocr style): output
 *    reshaped to [slots, charset]; per-slot argmax, pad characters dropped,
 *    confidence = mean of per-slot max probabilities.
 */
class OpenCvDnnPlateRecognizer final : public IPlateRecognizer {
public:
    explicit OpenCvDnnPlateRecognizer(std::string profileName, ModelProfile profile);

    bool initialize() override;
    PlateDetectionList recognize(const Frame& frame) override;
    std::string id() const override;
    std::string lastError() const override;

private:
    /// Candidate plate region produced by the detection stage.
    struct Candidate {
        cv::Rect box;
        float confidence = 0;
    };

    std::vector<Candidate> detectPlates(const cv::Mat& image);
    /// Runs OCR on one plate crop; returns false if below the OCR threshold.
    bool readPlate(const cv::Mat& crop, std::string& text, float& confidence);

    std::string profileName_;
    ModelProfile profile_;
    cv::dnn::Net detectionNet_;
    cv::dnn::Net ocrNet_;
    std::vector<char> charset_;
    std::string lastError_;
    bool initialized_ = false;
};

} // namespace anpr
