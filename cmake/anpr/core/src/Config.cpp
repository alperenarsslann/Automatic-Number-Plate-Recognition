#include "core/Config.h"

#include <fstream>
#include <set>
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

void readVehicleModel(const json& j, VehicleModelConfig& cfg) {
    readIfPresent(j, "enabled", cfg.enabled);
    readIfPresent(j, "model_path", cfg.modelPath);
    readIfPresent(j, "input_width", cfg.inputWidth);
    readIfPresent(j, "input_height", cfg.inputHeight);
    readIfPresent(j, "confidence_threshold", cfg.confidenceThreshold);
    readIfPresent(j, "nms_threshold", cfg.nmsThreshold);
    readIfPresent(j, "class_ids", cfg.classIds);
    readIfPresent(j, "roi_expand", cfg.roiExpand);
    if (j.contains("preprocess")) readPreprocess(j.at("preprocess"), cfg.preprocess);
}

void readModelProfile(const json& j, ModelProfile& profile) {
    readIfPresent(j, "engine", profile.engine);
    readIfPresent(j, "dnn_backend", profile.dnnBackend);
    static const std::set<std::string> kBackends = {"cpu", "opencl", "opencl_fp16", "cuda",
                                                    "cuda_fp16"};
    if (!kBackends.count(profile.dnnBackend)) {
        throw std::runtime_error("unknown dnn_backend \"" + profile.dnnBackend +
                                 "\" (expected cpu, opencl, opencl_fp16, cuda or cuda_fp16)");
    }
    if (j.contains("vehicle_detection")) readVehicleModel(j.at("vehicle_detection"), profile.vehicle);
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
    readIfPresent(j, "substream_url", cfg.substreamUrl);
    readIfPresent(j, "transport", cfg.transport);
    readIfPresent(j, "reconnect_interval_seconds", cfg.reconnectIntervalSeconds);
}

void readMotion(const json& j, MotionConfig& cfg) {
    readIfPresent(j, "enabled", cfg.enabled);
    readIfPresent(j, "downscale_width", cfg.downscaleWidth);
    readIfPresent(j, "min_area_fraction", cfg.minAreaFraction);
    readIfPresent(j, "cooldown_seconds", cfg.cooldownSeconds);
}

void readCameraInstance(const json& j, CameraInstanceConfig& cfg) {
    readIfPresent(j, "id", cfg.id);
    readIfPresent(j, "name", cfg.name);
    readIfPresent(j, "enabled", cfg.enabled);
    readIfPresent(j, "alpr_enabled", cfg.alprEnabled);
    readIfPresent(j, "source", cfg.source);
    if (cfg.source != "simulation" && cfg.source != "camera") {
        throw std::runtime_error("camera \"" + cfg.id +
                                 "\": source must be \"simulation\" or \"camera\"");
    }
    if (j.contains("simulation")) readSimulation(j.at("simulation"), cfg.simulation);
    if (j.contains("camera")) readCamera(j.at("camera"), cfg.camera);
    if (j.contains("motion")) readMotion(j.at("motion"), cfg.motion);
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
    if (j.contains("consolidation")) {
        const auto& c = j.at("consolidation");
        readIfPresent(c, "enabled", cfg.consolidation.enabled);
        readIfPresent(c, "finalize_after_seconds", cfg.consolidation.finalizeAfterSeconds);
        readIfPresent(c, "max_track_seconds", cfg.consolidation.maxTrackSeconds);
        readIfPresent(c, "max_edit_distance", cfg.consolidation.maxEditDistance);
        readIfPresent(c, "max_center_distance", cfg.consolidation.maxCenterDistance);
        readIfPresent(c, "min_sightings", cfg.consolidation.minSightings);
    }
    readIfPresent(j, "dedup_window_seconds", cfg.dedupWindowSeconds);
    readIfPresent(j, "queue_capacity", cfg.queueCapacity);
    readIfPresent(j, "process_every_n_frames", cfg.processEveryNFrames);
    readIfPresent(j, "num_threads", cfg.numThreads);
    readIfPresent(j, "max_frame_width", cfg.maxFrameWidth);
    readIfPresent(j, "worker_count", cfg.workerCount);
    if (cfg.workerCount < 0 || cfg.workerCount > 16) {
        throw std::runtime_error("processing.worker_count must be between 0 (auto) and 16");
    }
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
        if (j.contains("cameras")) {
            for (const auto& item : j.at("cameras")) {
                CameraInstanceConfig cam;
                readCameraInstance(item, cam);
                if (cam.id.empty()) {
                    cam.id = "cam" + std::to_string(cfg.cameras.size() + 1);
                }
                if (cam.name.empty()) cam.name = cam.id;
                cfg.cameras.push_back(std::move(cam));
            }
            if (cfg.cameras.size() > 64) {
                throw std::runtime_error("at most 64 cameras are supported, got " +
                                         std::to_string(cfg.cameras.size()));
            }
            std::set<std::string> ids;
            for (const auto& cam : cfg.cameras) {
                if (!ids.insert(cam.id).second) {
                    throw std::runtime_error("duplicate camera id: " + cam.id);
                }
            }
        } else {
            // Legacy single-camera config: synthesize one camera entry.
            CameraInstanceConfig cam;
            cam.id = "cam1";
            cam.name = "Camera 1";
            cam.source = cfg.capture.source;
            cam.simulation = cfg.capture.simulation;
            cam.camera = cfg.capture.camera;
            cam.motion.enabled = false; // Preserve old behavior: process everything.
            cfg.cameras.push_back(std::move(cam));
        }
        if (j.contains("processing")) readProcessing(j.at("processing"), cfg.processing);
        if (j.contains("network")) readNetwork(j.at("network"), cfg.network);
        if (j.contains("display")) readDisplay(j.at("display"), cfg.display);
        if (j.contains("detection_output")) {
            const auto& d = j.at("detection_output");
            readIfPresent(d, "enabled", cfg.detectionOutput.enabled);
            readIfPresent(d, "directory", cfg.detectionOutput.directory);
            readIfPresent(d, "draw_timestamp", cfg.detectionOutput.drawTimestamp);
            readIfPresent(d, "jpeg_quality", cfg.detectionOutput.jpegQuality);
        }
        if (j.contains("api")) {
            readIfPresent(j.at("api"), "enabled", cfg.api.enabled);
            readIfPresent(j.at("api"), "bind_address", cfg.api.bindAddress);
            readIfPresent(j.at("api"), "port", cfg.api.port);
            readIfPresent(j.at("api"), "api_keys", cfg.api.apiKeys);
        }
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
