/**
 * @file Config.h
 * @brief Typed, JSON-backed configuration for all layers.
 *
 * Single source of truth for every tunable in the system. Nothing here may
 * be hardcoded in the layers: model paths, input tensor sizes, preprocessing
 * parameters, thresholds, charset files, network endpoints — all flow from
 * one JSON file (see config/anpr.json) into these structs, which are then
 * injected into the layer implementations by factories. Deploying a new
 * fine-tuned model or pointing at a different server is a config edit +
 * restart, never a recompile.
 */
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace anpr {

/// Input tensor preparation for a DNN model. Different training pipelines
/// use different schemes, so every field is per-model and config-driven.
struct PreprocessConfig {
    std::string colorOrder = "BGR";           ///< "BGR" or "RGB" channel order expected by the model.
    int channels = 3;                         ///< Input channels: 3 (color) or 1 (grayscale).
    std::string layout = "nchw";              ///< Tensor layout: "nchw" or "nhwc".
    double scale = 1.0 / 255.0;               ///< Pixel scale factor applied first.
    std::array<double, 3> mean = {0, 0, 0};   ///< Per-channel mean subtracted after scaling.
    std::array<double, 3> stddev = {1, 1, 1}; ///< Per-channel std divided after mean subtraction.
};

/// Plate detection model parameters.
struct DetectionModelConfig {
    std::string modelPath;              ///< Path to the .onnx detection model.
    int inputWidth = 640;               ///< Model input tensor width.
    int inputHeight = 640;              ///< Model input tensor height.
    double confidenceThreshold = 0.5;   ///< Minimum detection confidence.
    double nmsThreshold = 0.45;         ///< Non-max suppression IoU threshold.
    PreprocessConfig preprocess;
};

/// Plate OCR model parameters.
struct OcrModelConfig {
    std::string modelPath;              ///< Path to the .onnx OCR model.
    int inputWidth = 160;               ///< Model input tensor width.
    int inputHeight = 32;               ///< Model input tensor height.
    double confidenceThreshold = 0.5;   ///< Minimum mean character confidence.
    std::string charsetPath;            ///< File mapping model outputs to characters.
    std::string padChar = "_";          ///< Charset symbol that marks an empty slot.
    PreprocessConfig preprocess;
};

/**
 * @brief A named, self-contained model configuration.
 *
 * Multiple profiles (e.g. "generic" and "turkish_finetuned") coexist in the
 * config file; ProcessingConfig::activeModelProfile selects one at startup.
 */
struct ModelProfile {
    std::string engine = "opencv_dnn";  ///< Recognizer implementation to instantiate.
    DetectionModelConfig detection;
    OcrModelConfig ocr;
};

/// Simulated (video file) frame source settings.
struct SimulationConfig {
    std::string videoPath;              ///< Input video file.
    bool realtime = true;               ///< true: pace to FPS; false: as fast as possible.
    double targetFps = 0.0;             ///< Override pacing FPS; 0 = use the video's native FPS.
    int frameSkip = 0;                  ///< Process every (frameSkip+1)-th frame.
    bool loop = true;                   ///< Rewind at end of file.
    double startTimeSeconds = 0.0;      ///< Seek position before the first frame.
};

/// Real camera (RTSP) frame source settings.
struct CameraConfig {
    std::string rtspUrl;                ///< e.g. rtsp://user:pass@host:554/Streaming/Channels/101
    std::string transport = "tcp";      ///< RTP transport: "tcp" or "udp".
    double reconnectIntervalSeconds = 5.0;
};

/// Capture layer settings.
struct CaptureConfig {
    std::string source = "simulation";  ///< Active source: "simulation" or "camera".
    SimulationConfig simulation;
    CameraConfig camera;
    std::size_t queueCapacity = 8;      ///< Bounded frame queue toward processing.
};

/// Processing layer settings.
struct ProcessingConfig {
    std::string activeModelProfile = "generic"; ///< Key into modelProfiles.
    std::map<std::string, ModelProfile> modelProfiles;
    double dedupWindowSeconds = 10.0;   ///< Same plate re-reported only after this window.
    std::size_t queueCapacity = 32;     ///< Bounded detection queue toward network.
    int processEveryNFrames = 1;        ///< Run ALPR on every Nth frame (CPU relief).
    int numThreads = 0;                 ///< OpenCV thread cap; 0 = library default (all cores).
    int maxFrameWidth = 0;              ///< Downscale frames wider than this right after
                                        ///< capture (0 = off). Big CPU saver for 4K sources.
};

/// Live display settings (development/verification aid).
struct DisplayConfig {
    bool enabled = false;               ///< Show annotated video in a window.
    int maxWidth = 1280;                ///< Downscale the window if wider than this.
};

/// Network layer settings.
struct NetworkConfig {
    bool enabled = true;                ///< Allows running capture+processing only.
    std::string serverHost = "127.0.0.1";
    std::uint16_t serverPort = 9000;
    double reconnectIntervalSeconds = 5.0;
    std::size_t sendQueueCapacity = 256; ///< Outgoing messages buffered while disconnected.
};

/// Logging settings.
struct LoggingConfig {
    std::string level = "info";         ///< debug | info | warn | error
};

/// Root configuration object (mirrors config/anpr.json).
struct AppConfig {
    int version = 1;                    ///< Config schema version.
    LoggingConfig logging;
    CaptureConfig capture;
    ProcessingConfig processing;
    NetworkConfig network;
    DisplayConfig display;

    /**
     * @brief Load and validate a JSON config file.
     *
     * Missing keys fall back to the defaults defined in these structs;
     * structural errors (bad JSON, wrong types, unknown active profile)
     * throw std::runtime_error with a descriptive message.
     */
    static AppConfig loadFromFile(const std::string& path);

    /// @return The active model profile, or nullopt if the key is unknown.
    std::optional<ModelProfile> activeProfile() const;
};

} // namespace anpr
