#include "core/Config.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace anpr {

namespace {

using nlohmann::json;

/// Read a key into @p target if present, leaving the struct default otherwise.
template <typename T>
void readIfPresent(const json& j, const char* key, T& target) {
    if (j.contains(key)) {
        j.at(key).get_to(target);
    }
}

void readPreprocess(const json& j, PreprocessConfig& cfg) {
    readIfPresent(j, "color_order", cfg.colorOrder);
    readIfPresent(j, "channels", cfg.channels);
    readIfPresent(j, "layout", cfg.layout);
    readIfPresent(j, "scale", cfg.scale);
    readIfPresent(j, "mean", cfg.mean);
    readIfPresent(j, "std", cfg.stddev);
    if (cfg.colorOrder != "BGR" && cfg.colorOrder != "RGB") {
        throw std::runtime_error("preprocess.color_order must be \"BGR\" or \"RGB\", got \"" +
                                 cfg.colorOrder + '"');
    }
    if (cfg.channels != 1 && cfg.channels != 3) {
        throw std::runtime_error("preprocess.channels must be 1 or 3");
    }
    if (cfg.layout != "nchw" && cfg.layout != "nhwc") {
        throw std::runtime_error("preprocess.layout must be \"nchw\" or \"nhwc\", got \"" +
                                 cfg.layout + '"');
    }
}

void readDetectionModel(const json& j, DetectionModelConfig& cfg) {
    readIfPresent(j, "model_path", cfg.modelPath);
    readIfPresent(j, "input_width", cfg.inputWidth);
    readIfPresent(j, "input_height", cfg.inputHeight);
    readIfPresent(j, "confidence_threshold", cfg.confidenceThreshold);
    readIfPresent(j, "nms_threshold", cfg.nmsThreshold);
    if (j.contains("preprocess")) readPreprocess(j.at("preprocess"), cfg.preprocess);
}

void readOcrModel(const json& j, OcrModelConfig& cfg) {
    readIfPresent(j, "model_path", cfg.modelPath);
    readIfPresent(j, "input_width", cfg.inputWidth);
    readIfPresent(j, "input_height", cfg.inputHeight);
    readIfPresent(j, "confidence_threshold", cfg.confidenceThreshold);
    readIfPresent(j, "charset_path", cfg.charsetPath);
    readIfPresent(j, "pad_char", cfg.padChar);
    if (j.contains("preprocess")) readPreprocess(j.at("preprocess"), cfg.preprocess);
}

void readModelProfile(const json& j, ModelProfile& profile) {
    readIfPresent(j, "engine", profile.engine);
    if (j.contains("detection")) readDetectionModel(j.at("detection"), profile.detection);
    if (j.contains("ocr")) readOcrModel(j.at("ocr"), profile.ocr);
}

void readSimulation(const json& j, SimulationConfig& cfg) {
    readIfPresent(j, "video_path", cfg.videoPath);
    readIfPresent(j, "realtime", cfg.realtime);
    readIfPresent(j, "target_fps", cfg.targetFps);
    readIfPresent(j, "frame_skip", cfg.frameSkip);
    readIfPresent(j, "loop", cfg.loop);
    readIfPresent(j, "start_time_seconds", cfg.startTimeSeconds);
}

void readCamera(const json& j, CameraConfig& cfg) {
    readIfPresent(j, "rtsp_url", cfg.rtspUrl);
    readIfPresent(j, "transport", cfg.transport);
    readIfPresent(j, "reconnect_interval_seconds", cfg.reconnectIntervalSeconds);
}

void readCapture(const json& j, CaptureConfig& cfg) {
    readIfPresent(j, "source", cfg.source);
    if (cfg.source != "simulation" && cfg.source != "camera") {
        throw std::runtime_error("capture.source must be \"simulation\" or \"camera\", got \"" +
                                 cfg.source + '"');
    }
    if (j.contains("simulation")) readSimulation(j.at("simulation"), cfg.simulation);
    if (j.contains("camera")) readCamera(j.at("camera"), cfg.camera);
    readIfPresent(j, "queue_capacity", cfg.queueCapacity);
}

void readProcessing(const json& j, ProcessingConfig& cfg) {
    readIfPresent(j, "active_model_profile", cfg.activeModelProfile);
    readIfPresent(j, "dedup_window_seconds", cfg.dedupWindowSeconds);
    readIfPresent(j, "queue_capacity", cfg.queueCapacity);
    readIfPresent(j, "process_every_n_frames", cfg.processEveryNFrames);
    readIfPresent(j, "num_threads", cfg.numThreads);
    readIfPresent(j, "max_frame_width", cfg.maxFrameWidth);
    if (cfg.processEveryNFrames < 1) {
        throw std::runtime_error("processing.process_every_n_frames must be >= 1");
    }
    if (j.contains("model_profiles")) {
        for (const auto& [name, value] : j.at("model_profiles").items()) {
            ModelProfile profile;
            readModelProfile(value, profile);
            cfg.modelProfiles.emplace(name, std::move(profile));
        }
    }
    if (!cfg.modelProfiles.empty() && !cfg.modelProfiles.count(cfg.activeModelProfile)) {
        throw std::runtime_error("processing.active_model_profile \"" + cfg.activeModelProfile +
                                 "\" is not defined in processing.model_profiles");
    }
}

void readDisplay(const json& j, DisplayConfig& cfg) {
    readIfPresent(j, "enabled", cfg.enabled);
    readIfPresent(j, "max_width", cfg.maxWidth);
}

void readNetwork(const json& j, NetworkConfig& cfg) {
    readIfPresent(j, "enabled", cfg.enabled);
    readIfPresent(j, "server_host", cfg.serverHost);
    readIfPresent(j, "server_port", cfg.serverPort);
    readIfPresent(j, "reconnect_interval_seconds", cfg.reconnectIntervalSeconds);
    readIfPresent(j, "send_queue_capacity", cfg.sendQueueCapacity);
}

} // namespace

AppConfig AppConfig::loadFromFile(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open config file: " + path);
    }

    json j;
    try {
        j = json::parse(file);
    } catch (const json::parse_error& e) {
        throw std::runtime_error("config parse error in " + path + ": " + e.what());
    }

    try {
        AppConfig cfg;
        readIfPresent(j, "version", cfg.version);
        if (j.contains("logging")) readIfPresent(j.at("logging"), "level", cfg.logging.level);
        if (j.contains("capture")) readCapture(j.at("capture"), cfg.capture);
        if (j.contains("processing")) readProcessing(j.at("processing"), cfg.processing);
        if (j.contains("network")) readNetwork(j.at("network"), cfg.network);
        if (j.contains("display")) readDisplay(j.at("display"), cfg.display);
        return cfg;
    } catch (const json::exception& e) {
        throw std::runtime_error("config type error in " + path + ": " + e.what());
    }
}

std::optional<ModelProfile> AppConfig::activeProfile() const {
    const auto it = processing.modelProfiles.find(processing.activeModelProfile);
    if (it == processing.modelProfiles.end()) return std::nullopt;
    return it->second;
}

} // namespace anpr
