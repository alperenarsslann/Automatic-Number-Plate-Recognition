#include "processing/OpenCvDnnPlateRecognizer.h"

#include <algorithm>
#include <fstream>

#include <opencv2/imgproc.hpp>

#include "core/Logger.h"

namespace anpr {

namespace {

constexpr const char* kComponent = "ALPR";

/// Letterbox parameters needed to map detections back to source coordinates.
struct Letterbox {
    double scale = 1.0;
    int padX = 0;
    int padY = 0;
};

/// Resize preserving aspect ratio, padding the borders (YOLO convention).
cv::Mat letterboxImage(const cv::Mat& src, int dstW, int dstH, Letterbox& lb) {
    lb.scale = std::min(dstW / static_cast<double>(src.cols),
                        dstH / static_cast<double>(src.rows));
    const int scaledW = static_cast<int>(std::round(src.cols * lb.scale));
    const int scaledH = static_cast<int>(std::round(src.rows * lb.scale));
    lb.padX = (dstW - scaledW) / 2;
    lb.padY = (dstH - scaledH) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(scaledW, scaledH));
    cv::Mat boxed(dstH, dstW, src.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(boxed(cv::Rect(lb.padX, lb.padY, scaledW, scaledH)));
    return boxed;
}

/// Scale + mean/std normalization on a float image, per the model's config.
void normalizeInPlace(cv::Mat& imageF32, const PreprocessConfig& pp) {
    imageF32 *= pp.scale;
    if (imageF32.channels() == 3) {
        imageF32 -= cv::Scalar(pp.mean[0], pp.mean[1], pp.mean[2]);
        cv::divide(imageF32, cv::Scalar(pp.stddev[0], pp.stddev[1], pp.stddev[2]), imageF32);
    } else {
        imageF32 -= pp.mean[0];
        imageF32 /= pp.stddev[0];
    }
}

/**
 * @brief Build an input tensor from a BGR image according to a model's
 *        preprocessing config (channel count, color order, layout, scale,
 *        mean/std). Supports NCHW and NHWC layouts.
 */
cv::Mat makeBlob(const cv::Mat& bgr, int dstW, int dstH, const PreprocessConfig& pp) {
    cv::Mat prepared;
    if (pp.channels == 1) {
        cv::cvtColor(bgr, prepared, cv::COLOR_BGR2GRAY);
    } else if (pp.colorOrder == "RGB") {
        cv::cvtColor(bgr, prepared, cv::COLOR_BGR2RGB);
    } else {
        prepared = bgr;
    }

    if (prepared.size() != cv::Size(dstW, dstH)) {
        cv::resize(prepared, prepared, cv::Size(dstW, dstH));
    }

    prepared.convertTo(prepared, CV_32F);
    normalizeInPlace(prepared, pp);

    if (pp.layout == "nhwc") {
        const int sizes[4] = {1, dstH, dstW, pp.channels};
        cv::Mat blob(4, sizes, CV_32F);
        std::memcpy(blob.data, prepared.data,
                    static_cast<std::size_t>(dstH) * dstW * pp.channels * sizeof(float));
        return blob;
    }
    return cv::dnn::blobFromImage(prepared); // HWC float -> NCHW, no extra scaling.
}

} // namespace

OpenCvDnnPlateRecognizer::OpenCvDnnPlateRecognizer(std::string profileName, ModelProfile profile)
    : profileName_(std::move(profileName)), profile_(std::move(profile)) {}

bool OpenCvDnnPlateRecognizer::initialize() {
    try {
        detectionNet_ = cv::dnn::readNetFromONNX(profile_.detection.modelPath);
    } catch (const cv::Exception& e) {
        lastError_ = "cannot load detection model '" + profile_.detection.modelPath +
                     "': " + e.what();
        return false;
    }
    try {
        ocrNet_ = cv::dnn::readNetFromONNX(profile_.ocr.modelPath);
    } catch (const cv::Exception& e) {
        lastError_ = "cannot load OCR model '" + profile_.ocr.modelPath + "': " + e.what();
        return false;
    }

    std::ifstream charsetFile(profile_.ocr.charsetPath);
    if (!charsetFile) {
        lastError_ = "cannot open charset file: " + profile_.ocr.charsetPath;
        return false;
    }
    std::string line;
    std::getline(charsetFile, line);
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
        line.pop_back();
    }
    charset_.assign(line.begin(), line.end());
    if (charset_.empty()) {
        lastError_ = "charset file is empty: " + profile_.ocr.charsetPath;
        return false;
    }

    initialized_ = true;
    LOG_INFO(kComponent, "initialized profile '", profileName_,
             "' (detector: ", profile_.detection.modelPath,
             ", ocr: ", profile_.ocr.modelPath, ", charset: ", charset_.size(), " symbols)");
    return true;
}

std::vector<OpenCvDnnPlateRecognizer::Candidate>
OpenCvDnnPlateRecognizer::detectPlates(const cv::Mat& image) {
    const auto& det = profile_.detection;

    Letterbox lb;
    const cv::Mat boxed = letterboxImage(image, det.inputWidth, det.inputHeight, lb);
    detectionNet_.setInput(makeBlob(boxed, det.inputWidth, det.inputHeight, det.preprocess));
    cv::Mat out = detectionNet_.forward();

    // Accept [1, 4+nc, N] (YOLOv8 family) and [1, N, 4+nc] (YOLOv5 family).
    // The attribute axis is much smaller than the anchor axis, which makes
    // the orientation unambiguous.
    CV_Assert(out.dims == 3);
    cv::Mat pred(out.size[1], out.size[2], CV_32F, out.ptr<float>());
    if (out.size[1] < out.size[2]) {
        pred = pred.t(); // -> [N, 4+nc]
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    for (int i = 0; i < pred.rows; ++i) {
        const float* row = pred.ptr<float>(i);
        // Single-class models: row[4] is the score; multi-class: take the max.
        float score = row[4];
        for (int c = 5; c < pred.cols; ++c) {
            score = std::max(score, row[c]);
        }
        if (score < static_cast<float>(det.confidenceThreshold)) {
            continue;
        }
        // cx, cy, w, h in letterboxed input pixels -> source pixels.
        const double cx = (row[0] - lb.padX) / lb.scale;
        const double cy = (row[1] - lb.padY) / lb.scale;
        const double w = row[2] / lb.scale;
        const double h = row[3] / lb.scale;
        const cv::Rect box(static_cast<int>(cx - w / 2), static_cast<int>(cy - h / 2),
                           static_cast<int>(w), static_cast<int>(h));
        const cv::Rect clipped = box & cv::Rect(0, 0, image.cols, image.rows);
        if (clipped.area() <= 0) {
            continue;
        }
        boxes.push_back(clipped);
        scores.push_back(score);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, static_cast<float>(det.confidenceThreshold),
                      static_cast<float>(det.nmsThreshold), keep);

    std::vector<Candidate> result;
    result.reserve(keep.size());
    for (const int idx : keep) {
        result.push_back({boxes[idx], scores[idx]});
    }
    return result;
}

bool OpenCvDnnPlateRecognizer::readPlate(const cv::Mat& crop, std::string& text,
                                         float& confidence) {
    const auto& ocr = profile_.ocr;

    ocrNet_.setInput(makeBlob(crop, ocr.inputWidth, ocr.inputHeight, ocr.preprocess));
    cv::Mat out = ocrNet_.forward();

    // Fixed-slot head: total elements = slots * charset size, already softmaxed.
    const int vocab = static_cast<int>(charset_.size());
    const int total = static_cast<int>(out.total());
    if (total % vocab != 0) {
        lastError_ = "OCR output size " + std::to_string(total) +
                     " is not divisible by charset size " + std::to_string(vocab) +
                     " (wrong charset file?)";
        return false;
    }
    const int slots = total / vocab;
    const cv::Mat probs(slots, vocab, CV_32F, out.ptr<float>());

    const char padChar = ocr.padChar.empty() ? '_' : ocr.padChar.front();
    text.clear();
    double confidenceSum = 0.0;
    for (int s = 0; s < slots; ++s) {
        const float* row = probs.ptr<float>(s);
        const int best =
            static_cast<int>(std::max_element(row, row + vocab) - row);
        confidenceSum += row[best];
        if (charset_[best] != padChar) {
            text += charset_[best];
        }
    }
    confidence = static_cast<float>(confidenceSum / slots);
    return !text.empty() && confidence >= static_cast<float>(ocr.confidenceThreshold);
}

PlateDetectionList OpenCvDnnPlateRecognizer::recognize(const Frame& frame) {
    PlateDetectionList detections;
    if (!initialized_ || frame.image.empty()) {
        return detections;
    }

    for (const Candidate& candidate : detectPlates(frame.image)) {
        std::string text;
        float ocrConfidence = 0;
        if (!readPlate(frame.image(candidate.box), text, ocrConfidence)) {
            LOG_DEBUG(kComponent, "plate candidate at ", candidate.box.x, ",", candidate.box.y,
                      " rejected by OCR (conf=", ocrConfidence, ")");
            continue;
        }

        PlateDetection detection;
        detection.text = std::move(text);
        detection.detectionConfidence = candidate.confidence;
        detection.ocrConfidence = ocrConfidence;
        detection.boundingBox = candidate.box;
        detection.frameSequence = frame.sequence;
        detection.timestamp = frame.timestamp;
        detection.sourceId = frame.sourceId;
        detections.push_back(std::move(detection));
    }
    return detections;
}

std::string OpenCvDnnPlateRecognizer::id() const {
    return "opencv_dnn:" + profileName_;
}

std::string OpenCvDnnPlateRecognizer::lastError() const {
    return lastError_;
}

} // namespace anpr
