/**
 * @file main.cpp
 * @brief Application entry point (Control Layer).
 *
 * Modes:
 *  - default: multi-camera pipeline (see MultiCameraPipeline.h) — up to 64
 *    cameras, motion-gated ALPR worker pool, optional grid display and
 *    embedded REST API.
 *  - --image: one-shot recognition on a single picture (verification aid).
 */
#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "MultiCameraPipeline.h"
#include "capture/HikvisionDiscovery.h"
#include "core/Config.h"
#include "core/Logger.h"
#include "core/Types.h"
#include "processing/PlateRecognizerFactory.h"

namespace {

constexpr const char* kComponent = "Control";

/// Set by the SIGINT/SIGTERM handler; checked by the pipeline loops.
std::atomic<bool> g_stopRequested{false};

void signalHandler(int /*signum*/) {
    g_stopRequested.store(true);
}

struct CliOptions {
    std::string configPath = "config/anpr.json";
    std::string logLevel;   ///< Overrides config when non-empty.
    std::string source;     ///< Overrides camera[0] source when non-empty.
    std::string videoPath;  ///< Overrides camera[0] simulation video when non-empty.
    std::string imagePath;  ///< One-shot mode: recognize a single image and exit.
    std::string display;    ///< "on"/"off" overrides display.enabled when non-empty.
    bool discover = false;  ///< Scan the LAN for Hikvision cameras and exit.
    bool showHelp = false;
};

void printUsage() {
    std::cout <<
        "Usage: anpr [options]\n"
        "  --config=<path>      Config file (default: config/anpr.json)\n"
        "  --source=<name>      Override first camera's source: simulation | camera\n"
        "  --video=<path>       Override first camera's simulation video file\n"
        "  --image=<path>       One-shot: recognize a single image, save an\n"
        "                       annotated copy next to it, then exit\n"
        "  --display[=on|off]   Override the camera-wall display window\n"
        "  --discover           Scan the LAN for Hikvision cameras (SADP) and exit\n"
        "  --log-level=<level>  Override log level: debug | info | warn | error\n"
        "  --help               Show this help\n"
        "\n"
        "Examples:\n"
        "  anpr                                   (all cameras from the config)\n"
        "  anpr --video=traffic.mp4               (quick single-video run)\n"
        "  anpr --image=car.jpg --log-level=debug\n";
}

/// Parse --key=value style arguments; throws on unknown options.
CliOptions parseArgs(int argc, char** argv) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto eq = arg.find('=');
        const std::string key = arg.substr(0, eq);
        const std::string value = (eq == std::string::npos) ? "" : arg.substr(eq + 1);

        if (key == "--help" || key == "-h") opts.showHelp = true;
        else if (key == "--config") opts.configPath = value;
        else if (key == "--source") opts.source = value;
        else if (key == "--video") opts.videoPath = value;
        else if (key == "--image") opts.imagePath = value;
        else if (key == "--display") opts.display = value.empty() ? "on" : value;
        else if (key == "--discover") opts.discover = true;
        else if (key == "--log-level") opts.logLevel = value;
        else throw std::runtime_error("unknown option: " + arg);
    }
    return opts;
}

/**
 * @brief One-shot mode: recognize a single image, log the results and write
 *        an annotated copy (<name>_anpr.<ext>) for visual verification.
 */
int runImageTest(const anpr::ProcessingConfig& processingConfig, const std::string& imagePath) {
    std::unique_ptr<anpr::IPlateRecognizer> recognizer;
    try {
        recognizer = anpr::createPlateRecognizer(processingConfig);
    } catch (const std::exception& e) {
        LOG_ERROR(kComponent, "cannot create recognizer: ", e.what());
        return 1;
    }
    if (!recognizer->initialize()) {
        LOG_ERROR(kComponent, "recognizer init failed: ", recognizer->lastError());
        return 1;
    }

    anpr::Frame frame;
    frame.image = cv::imread(imagePath);
    if (frame.image.empty()) {
        LOG_ERROR(kComponent, "cannot read image: ", imagePath);
        return 1;
    }
    frame.timestamp = std::chrono::system_clock::now();
    frame.sourceId = "image";

    const auto started = std::chrono::steady_clock::now();
    const auto detections = recognizer->recognize(frame);
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - started)
                               .count();

    LOG_INFO(kComponent, detections.size(), " plate(s) in ", imagePath, " (", elapsedMs, " ms)");
    if (detections.empty()) {
        return 0;
    }

    cv::Mat annotated = frame.image.clone();
    for (const auto& d : detections) {
        std::ostringstream oss;
        oss << "PLATE " << d.text << "  det=" << std::fixed << std::setprecision(2)
            << d.detectionConfidence << " ocr=" << d.ocrConfidence << " box=["
            << d.boundingBox.x << ',' << d.boundingBox.y << ' ' << d.boundingBox.width << 'x'
            << d.boundingBox.height << ']';
        LOG_INFO(kComponent, oss.str());
        cv::rectangle(annotated, d.boundingBox, cv::Scalar(0, 255, 0), 2);
        cv::putText(annotated, d.text, cv::Point(d.boundingBox.x, std::max(d.boundingBox.y - 8, 16)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }

    const auto dot = imagePath.find_last_of('.');
    const std::string outPath = (dot == std::string::npos)
                                    ? imagePath + "_anpr.jpg"
                                    : imagePath.substr(0, dot) + "_anpr" + imagePath.substr(dot);
    if (cv::imwrite(outPath, annotated)) {
        LOG_INFO(kComponent, "annotated image written to ", outPath);
    }
    return 0;
}

} // namespace

/**
 * @brief LAN scan mode: probe for Hikvision cameras via SADP and print them.
 *
 * Besides the human-readable table, each device is printed as a single
 * machine-parsable "DISCOVERED ..." line consumed by AnprStudio.
 */
int runDiscovery() {
    std::cout << "Scanning the local network for Hikvision cameras (SADP, 3 s)...\n";
    std::string error;
    const auto cameras = anpr::discoverHikvisionCameras(3.0, error);
    if (!error.empty()) {
        std::cerr << "Error: " << error << std::endl;
        return 1;
    }
    if (cameras.empty()) {
        std::cout << "No cameras answered. Check that the devices are powered, on the same\n"
                     "subnet, and that the Windows firewall allows UDP replies (port 37020).\n";
        return 0;
    }
    for (const auto& cam : cameras) {
        std::cout << "DISCOVERED ip=" << cam.ipv4 << " http=" << cam.httpPort
                  << " sdk=" << cam.sdkPort << " desc=" << cam.description
                  << " serial=" << cam.serialNumber << " mac=" << cam.mac
                  << " fw=" << cam.firmwareVersion << "\n";
        std::cout << "  RTSP main stream: " << cam.mainStreamUrl() << "\n";
    }
    std::cout << cameras.size() << " camera(s) found.\n";
    return 0;
}

int main(int argc, char** argv) {
    CliOptions opts;
    try {
        opts = parseArgs(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        printUsage();
        return 2;
    }
    if (opts.showHelp) {
        printUsage();
        return 0;
    }
    if (opts.discover) {
        return runDiscovery();
    }

    anpr::AppConfig config;
    try {
        config = anpr::AppConfig::loadFromFile(opts.configPath);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // CLI overrides take precedence over the config file.
    if (!opts.source.empty() && !config.cameras.empty()) config.cameras[0].source = opts.source;
    if (!opts.videoPath.empty() && !config.cameras.empty()) {
        config.cameras[0].simulation.videoPath = opts.videoPath;
    }
    if (!opts.logLevel.empty()) config.logging.level = opts.logLevel;
    if (!opts.display.empty()) config.display.enabled = (opts.display == "on");

    anpr::Logger::instance().setLevel(anpr::logLevelFromString(config.logging.level));

    // Silence OpenCV's internal INFO chatter (backend probing etc.);
    // real problems still surface as warnings/errors.
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    // For one-shot --image, cap OpenCV threads directly when configured; the
    // multi-camera pipeline manages this itself (auto-balances against the
    // ALPR worker count), so we leave it alone in that path.
    if (!opts.imagePath.empty() && config.processing.numThreads > 0) {
        cv::setNumThreads(config.processing.numThreads);
    }

    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    if (!opts.imagePath.empty()) {
        LOG_INFO(kComponent, "one-shot image mode (config: ", opts.configPath,
                 ", profile: ", config.processing.activeModelProfile, ")");
        return runImageTest(config.processing, opts.imagePath);
    }

    LOG_INFO(kComponent, "ANPR system starting (config: ", opts.configPath,
             ", cameras: ", config.cameras.size(),
             ", profile: ", config.processing.activeModelProfile, ")");
    return anpr::runMultiCameraPipeline(config, g_stopRequested);
}
