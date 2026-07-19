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
#include <vector>

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

/**
 * @brief Optional vehicle detection stage that runs BEFORE plate detection.
 *
 * When enabled, plates are searched only inside detected vehicle regions and
 * at most one plate (the best candidate) is reported per vehicle. This kills
 * false positives on background structures and stops one physical plate from
 * being reported as several.
 */
struct VehicleModelConfig {
    bool enabled = false;
    std::string modelPath;              ///< COCO-style YOLO .onnx (or custom).
    int inputWidth = 640;
    int inputHeight = 640;
    double confidenceThreshold = 0.4;
    double nmsThreshold = 0.45;
    std::vector<int> classIds = {2, 3, 5, 7}; ///< COCO: car, motorcycle, bus, truck.
    double roiExpand = 0.15;            ///< Grow the vehicle box by this fraction
                                        ///< before searching for the plate.
    PreprocessConfig preprocess;
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
    /// DNN compute backend: "cpu" (default), "opencl" / "opencl_fp16" (GPU via
    /// OpenCL, works with stock OpenCV), "cuda" / "cuda_fp16" (requires an
    /// OpenCV build with CUDA support). Unavailable backends fall back to CPU
    /// with a warning instead of failing.
    std::string dnnBackend = "cpu";
    VehicleModelConfig vehicle;         ///< Optional vehicle-first stage.
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
    std::string rtspUrl;                ///< Main stream, e.g. rtsp://user:pass@host:554/Streaming/Channels/101
    std::string substreamUrl;           ///< Optional low-res substream (e.g. .../102). When set,
                                        ///< analysis decodes this instead of the main stream —
                                        ///< the standard NVR trick for high camera counts.
    std::string transport = "tcp";      ///< RTP transport: "tcp" or "udp".
    double reconnectIntervalSeconds = 5.0;
};

/**
 * @brief Motion gating (Frigate-style): a very cheap background-subtraction
 *        pass decides whether a frame is worth sending to the DNN pipeline.
 *
 * Static scenes then cost almost nothing: no vehicle/plate/OCR inference runs
 * until pixels actually change. This is the single biggest CPU lever for
 * always-on cameras.
 */
struct MotionConfig {
    bool enabled = true;
    int downscaleWidth = 320;           ///< Motion analysis resolution (grayscale).
    double minAreaFraction = 0.005;     ///< Fraction of pixels that must change to count as motion.
    double cooldownSeconds = 2.0;       ///< Keep processing this long after the last motion.
};

/// Capture layer settings (LEGACY single-camera block; still accepted and
/// converted to a one-element camera list when "cameras" is absent).
struct CaptureConfig {
    std::string source = "simulation";  ///< Active source: "simulation" or "camera".
    SimulationConfig simulation;
    CameraConfig camera;
    std::size_t queueCapacity = 8;      ///< Bounded frame queue toward processing.
};

/// One camera in the multi-camera pipeline (up to 64).
struct CameraInstanceConfig {
    std::string id;                     ///< Unique key, used in reports and the API.
    std::string name;                   ///< Human-readable label for displays.
    bool enabled = true;                ///< Camera runs at all.
    bool alprEnabled = true;            ///< Plate recognition on/off (runtime-togglable via API/UI).
    std::string source = "simulation";  ///< "simulation" or "camera".
    SimulationConfig simulation;
    CameraConfig camera;
    MotionConfig motion;
};

/// Embedded REST API for status/control (per-camera ALPR toggles, plates feed).
struct ApiConfig {
    bool enabled = true;
    std::string bindAddress = "127.0.0.1"; ///< "0.0.0.0" to serve other hosts (then set apiKeys!).
    std::uint16_t port = 8088;
    /// Accepted API keys. When empty, the API is UNAUTHENTICATED (fine for a
    /// localhost dev bind). When non-empty, every request must present a valid
    /// key via `X-API-Key: <key>` or `Authorization: Bearer <key>`.
    std::vector<std::string> apiKeys;
};

/**
 * @brief Temporal consolidation ("one final result per plate").
 *
 * A passing vehicle yields many noisy OCR reads of the SAME plate across
 * frames (e.g. K619879 / 1619879 / V619879). Instead of reporting each,
 * readings are grouped into a track and, once the plate leaves view, a
 * single consolidated result is emitted via per-character confidence voting.
 */
struct ConsolidationConfig {
    bool enabled = true;
    double finalizeAfterSeconds = 1.5;  ///< Idle time before a track is finalized/reported.
    double maxTrackSeconds = 10.0;      ///< Safety cap: finalize even if still visible.
    int maxEditDistance = 2;            ///< Group readings within this edit distance.
    double maxCenterDistance = 0.2;     ///< ...and whose boxes are within this fraction
                                        ///< of the frame diagonal (spatial gate).
    int minSightings = 2;               ///< Require at least this many reads to report.
};

/// Processing layer settings.
struct ProcessingConfig {
    std::string activeModelProfile = "generic"; ///< Key into modelProfiles.
    std::map<std::string, ModelProfile> modelProfiles;
    ConsolidationConfig consolidation;
    double dedupWindowSeconds = 10.0;   ///< Same plate re-reported only after this window
                                        ///< (fallback path when consolidation is disabled).
    std::size_t queueCapacity = 32;     ///< Bounded detection queue toward network.
    int processEveryNFrames = 1;        ///< Run ALPR on every Nth frame (CPU relief).
    int numThreads = 0;                 ///< OpenCV threads PER inference. 0 = auto-balance so
                                        ///< workerCount * threads ~= CPU cores (avoids the
                                        ///< oversubscription that wastes CPU); >0 = fixed cap.
    int maxFrameWidth = 0;              ///< Downscale frames wider than this right after
                                        ///< capture (0 = off). Big CPU saver for 4K sources.
    int workerCount = 0;                ///< ALPR worker threads shared by ALL cameras. Each
                                        ///< worker owns its own model instances; cameras feed
                                        ///< one bounded queue (drop-oldest), so N cameras never
                                        ///< cost more inference CPU than workerCount allows.
                                        ///< 0 = auto (derived from CPU cores).
};

/// Live display settings: a grid wall of all enabled cameras (1x1 for a
/// single camera, up to 8x8 for 64). Clicking a tile toggles that camera's
/// ALPR on/off.
struct DisplayConfig {
    bool enabled = false;               ///< Show the annotated camera grid window.
    int maxWidth = 1280;                ///< Total window width; tiles share it.
};

/// Save an annotated JPG for every reported (consolidated) plate — a visual
/// verification record. One image per final result, not per raw frame.
struct DetectionOutputConfig {
    bool enabled = false;               ///< Master on/off.
    std::string directory = "detections"; ///< Output folder (created if missing).
    bool drawTimestamp = true;          ///< Burn a timestamp banner into the image.
    int jpegQuality = 90;               ///< 1..100.
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
    CaptureConfig capture;              ///< Legacy single-camera block (see cameras).
    std::vector<CameraInstanceConfig> cameras; ///< Multi-camera list (max 64). When empty,
                                               ///< synthesized from the legacy capture block.
    ProcessingConfig processing;
    NetworkConfig network;
    DisplayConfig display;
    DetectionOutputConfig detectionOutput;
    ApiConfig api;

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
