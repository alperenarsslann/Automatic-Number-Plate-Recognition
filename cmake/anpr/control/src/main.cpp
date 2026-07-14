/**
 * @file main.cpp
 * @brief Application entry point (Control Layer).
 *
 * Step 3 scope: frames from the configured IFrameSource flow through the
 * ALPR recognizer (built by the factory from the active model profile) and
 * deduplicated plate reports are logged. A one-shot --image mode recognizes
 * a single picture and writes an annotated copy for visual verification.
 * The network layer and the threaded pipeline arrive in the next steps.
 */
#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "capture/FrameSourceFactory.h"
#include "core/BoundedQueue.h"
#include "core/Config.h"
#include "core/Logger.h"
#include "core/Types.h"
#include "processing/PlateDeduplicator.h"
#include "processing/PlateRecognizerFactory.h"

namespace {

constexpr const char* kComponent = "Control";

/// Set by the SIGINT/SIGTERM handler; checked by the processing loop.
std::atomic<bool> g_stopRequested{false};

void signalHandler(int /*signum*/) {
    g_stopRequested.store(true);
}

struct CliOptions {
    std::string configPath = "config/anpr.json";
    std::string logLevel;   ///< Overrides config when non-empty.
    std::string source;     ///< Overrides capture.source when non-empty.
    std::string videoPath;  ///< Overrides capture.simulation.video_path when non-empty.
    std::string imagePath;  ///< One-shot mode: recognize a single image and exit.
    std::string display;    ///< "on"/"off" overrides display.enabled when non-empty.
    bool showHelp = false;
};

void printUsage() {
    std::cout <<
        "Usage: anpr [options]\n"
        "  --config=<path>      Config file (default: config/anpr.json)\n"
        "  --source=<name>      Override frame source: simulation | camera\n"
        "  --video=<path>       Override simulation video file\n"
        "  --image=<path>       One-shot: recognize a single image, save an\n"
        "                       annotated copy next to it, then exit\n"
        "  --display[=on|off]   Override the live annotated video window\n"
        "  --log-level=<level>  Override log level: debug | info | warn | error\n"
        "  --help               Show this help\n"
        "\n"
        "Examples:\n"
        "  anpr --source=simulation --video=traffic.mp4\n"
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
        else if (key == "--log-level") opts.logLevel = value;
        else throw std::runtime_error("unknown option: " + arg);
    }
    return opts;
}

std::string formatDetection(const anpr::PlateDetection& d) {
    std::ostringstream oss;
    oss << "PLATE " << d.text
        << "  det=" << std::fixed << std::setprecision(2) << d.detectionConfidence
        << " ocr=" << d.ocrConfidence
        << " box=[" << d.boundingBox.x << ',' << d.boundingBox.y << ' '
        << d.boundingBox.width << 'x' << d.boundingBox.height << ']'
        << " seq=" << d.frameSequence;
    return oss.str();
}

void drawDetection(cv::Mat& canvas, const anpr::PlateDetection& d) {
    cv::rectangle(canvas, d.boundingBox, cv::Scalar(0, 255, 0), 2);
    std::ostringstream label;
    label << d.text << ' ' << std::fixed << std::setprecision(2) << d.ocrConfidence;
    const int y = std::max(d.boundingBox.y - 8, 16);
    cv::putText(canvas, label.str(), cv::Point(d.boundingBox.x, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
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
    frame.sourceId = "image:" + imagePath;

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
        LOG_INFO(kComponent, formatDetection(d));
        drawDetection(annotated, d);
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

/// Cross-thread state shared between the capture, processing and display
/// threads. Counters are atomics; the detection overlay is mutex-guarded.
struct PipelineState {
    std::atomic<std::uint64_t> totalFrames{0};
    std::atomic<std::uint64_t> totalSightings{0}; ///< Detections before dedup.
    std::atomic<std::uint64_t> totalReports{0};   ///< Detections after dedup.
    std::atomic<std::uint64_t> framesDropped{0};  ///< Overflowed ALPR queue.

    std::mutex overlayMutex;
    anpr::PlateDetectionList lastDetections; ///< Most recent ALPR result (for drawing).
    double fps = 0.0;                        ///< Capture rate, 5 s window.
    double alprMs = 0.0;                     ///< Mean ALPR latency, 5 s window.
};

/**
 * @brief Video mode: threaded pipeline.
 *
 * capture thread  -> frame queue (drop-oldest) -> processing thread (ALPR)
 *                 -> display queue (drop-oldest, size 2) -> main thread (UI)
 *
 * Each stage runs at its own pace: the source paces to the video FPS (the
 * capture thread sleeps, it does not spin), ALPR consumes the newest frame
 * whenever it is free (a slow model drops backlog instead of stalling the
 * video), and the display shows every captured frame with the most recent
 * detections overlaid.
 */
int runPipeline(const anpr::AppConfig& config) {
    using Clock = std::chrono::steady_clock;
    using namespace std::chrono_literals;

    std::unique_ptr<anpr::IFrameSource> source;
    std::unique_ptr<anpr::IPlateRecognizer> recognizer;
    try {
        source = anpr::createFrameSource(config.capture);
        recognizer = anpr::createPlateRecognizer(config.processing);
    } catch (const std::exception& e) {
        LOG_ERROR(kComponent, "pipeline setup failed: ", e.what());
        return 1;
    }

    if (!recognizer->initialize()) {
        LOG_ERROR(kComponent, "recognizer init failed: ", recognizer->lastError());
        return 1;
    }
    if (!source->open()) {
        LOG_ERROR(kComponent, "cannot open frame source '", source->id(), "': ",
                  source->lastError());
        return 1;
    }

    const bool display = config.display.enabled;
    const int everyN = config.processing.processEveryNFrames;
    const int maxWidth = config.processing.maxFrameWidth;
    constexpr const char* kWindowName = "ANPR";

    LOG_INFO(kComponent, "pipeline running (source: ", source->id(),
             ", recognizer: ", recognizer->id(),
             ", dedup window: ", config.processing.dedupWindowSeconds, " s",
             ", alpr every ", everyN, " frame(s)",
             maxWidth > 0 ? ", max frame width " + std::to_string(maxWidth) : "",
             display ? ", display on) — press q/ESC in the window or Ctrl+C to stop"
                     : ") — press Ctrl+C to stop");

    PipelineState state;
    anpr::BoundedQueue<anpr::Frame> alprQueue(config.capture.queueCapacity);
    anpr::BoundedQueue<anpr::Frame> displayQueue(2);
    bool sourceFailed = false;

    // ---- capture thread: read, downscale, fan out ------------------------
    std::thread captureThread([&] {
        anpr::Frame frame;
        std::uint64_t windowFrames = 0;
        auto windowStartedAt = Clock::now();

        while (!g_stopRequested.load()) {
            if (!source->getNextFrame(frame)) {
                if (!source->lastError().empty()) {
                    LOG_ERROR(kComponent, "frame source failed: ", source->lastError());
                    sourceFailed = true;
                }
                break; // Clean end-of-stream (loop disabled) or error.
            }

            if (maxWidth > 0 && frame.image.cols > maxWidth) {
                const double s = maxWidth / static_cast<double>(frame.image.cols);
                cv::Mat resized;
                cv::resize(frame.image, resized, cv::Size(), s, s, cv::INTER_AREA);
                frame.image = std::move(resized);
            }

            const auto count = ++state.totalFrames;
            ++windowFrames;

            if ((count - 1) % static_cast<std::uint64_t>(everyN) == 0) {
                if (alprQueue.pushDropOldest(frame)) {
                    ++state.framesDropped;
                }
            }
            if (display) {
                displayQueue.pushDropOldest(frame);
            }

            const auto now = Clock::now();
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - windowStartedAt);
            if (elapsed >= 5s) {
                std::lock_guard<std::mutex> lock(state.overlayMutex);
                state.fps = windowFrames * 1000.0 / elapsed.count();
                windowFrames = 0;
                windowStartedAt = now;
            }
        }
        alprQueue.close();
        displayQueue.close();
    });

    // ---- processing thread: ALPR + dedup + reporting ---------------------
    std::thread processingThread([&] {
        anpr::PlateDeduplicator dedup(config.processing.dedupWindowSeconds);
        std::uint64_t windowProcessed = 0;
        double windowMs = 0.0;
        auto windowStartedAt = Clock::now();

        while (true) {
            auto frame = alprQueue.waitPop(100ms);
            if (!frame) {
                if (alprQueue.isClosed()) break;
                continue;
            }

            const auto started = Clock::now();
            auto detections = recognizer->recognize(*frame);
            windowMs += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
            ++windowProcessed;

            for (const auto& d : detections) {
                ++state.totalSightings;
                if (dedup.shouldReport(d.text)) {
                    ++state.totalReports;
                    LOG_INFO(kComponent, formatDetection(d));
                } else {
                    LOG_DEBUG(kComponent, "suppressed duplicate: ", d.text,
                              " (seq=", frame->sequence, ")");
                }
            }

            {
                std::lock_guard<std::mutex> lock(state.overlayMutex);
                state.lastDetections = std::move(detections);
            }

            const auto now = Clock::now();
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - windowStartedAt);
            if (elapsed >= 5s) {
                double fps;
                {
                    std::lock_guard<std::mutex> lock(state.overlayMutex);
                    state.alprMs = windowMs / windowProcessed;
                    fps = state.fps;
                }
                LOG_INFO(kComponent, "stats: seq=", frame->sequence, " fps=", fps,
                         " alpr_ms=", windowMs / windowProcessed,
                         " dropped=", state.framesDropped.load(),
                         " sightings=", state.totalSightings.load(),
                         " reports=", state.totalReports.load());
                windowProcessed = 0;
                windowMs = 0.0;
                windowStartedAt = now;
            }
        }
    });

    // ---- main thread: display (or idle wait) -----------------------------
    const auto startedAt = Clock::now();
    if (display) {
        while (!g_stopRequested.load()) {
            auto frame = displayQueue.waitPop(50ms);
            if (!frame) {
                if (displayQueue.isClosed()) break;
            } else {
                cv::Mat canvas = frame->image.clone();
                double fps;
                double alprMs;
                {
                    std::lock_guard<std::mutex> lock(state.overlayMutex);
                    for (const auto& d : state.lastDetections) {
                        drawDetection(canvas, d);
                    }
                    fps = state.fps;
                    alprMs = state.alprMs;
                }
                if (canvas.cols > config.display.maxWidth) {
                    const double s = config.display.maxWidth / static_cast<double>(canvas.cols);
                    cv::resize(canvas, canvas, cv::Size(), s, s, cv::INTER_AREA);
                }
                std::ostringstream hud;
                hud << std::fixed << std::setprecision(1) << fps << " fps  "
                    << std::setprecision(0) << alprMs << " ms/alpr  "
                    << "reports: " << state.totalReports.load() << "  [q/ESC quits]";
                cv::rectangle(canvas, cv::Rect(0, 0, canvas.cols, 28), cv::Scalar(32, 32, 32),
                              cv::FILLED);
                cv::putText(canvas, hud.str(), cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX,
                            0.55, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                cv::imshow(kWindowName, canvas);
            }

            const int key = cv::waitKey(1); // Also keeps the window responsive.
            if (key == 'q' || key == 27 /*ESC*/) {
                g_stopRequested.store(true);
            }
        }
    } else {
        // Headless: just wait for shutdown (Ctrl+C) or end-of-stream.
        while (!g_stopRequested.load() && !alprQueue.isClosed()) {
            std::this_thread::sleep_for(200ms);
        }
    }

    // Shutdown: stop capture first, queues close, workers drain and join.
    g_stopRequested.store(true);
    captureThread.join();
    processingThread.join();
    if (display) {
        cv::destroyAllWindows();
    }
    source->close();

    const auto totalElapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - startedAt);
    LOG_INFO(kComponent, sourceFailed ? "stopped after source error" : "finished",
             ": ", state.totalFrames.load(), " frames in ", totalElapsed.count() / 1000.0,
             " s, ", state.totalSightings.load(), " sightings -> ",
             state.totalReports.load(), " unique reports");
    return sourceFailed ? 1 : 0;
}

} // namespace

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

    anpr::AppConfig config;
    try {
        config = anpr::AppConfig::loadFromFile(opts.configPath);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // CLI overrides take precedence over the config file.
    if (!opts.source.empty()) config.capture.source = opts.source;
    if (!opts.videoPath.empty()) config.capture.simulation.videoPath = opts.videoPath;
    if (!opts.logLevel.empty()) config.logging.level = opts.logLevel;
    if (!opts.display.empty()) config.display.enabled = (opts.display == "on");

    anpr::Logger::instance().setLevel(anpr::logLevelFromString(config.logging.level));

    // Silence OpenCV's internal INFO chatter (backend probing etc.);
    // real problems still surface as warnings/errors.
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    // Cap OpenCV's internal parallelism (decode + DNN) when configured;
    // 0 keeps the library default (all cores).
    if (config.processing.numThreads > 0) {
        cv::setNumThreads(config.processing.numThreads);
        LOG_INFO(kComponent, "OpenCV thread cap: ", config.processing.numThreads);
    }

    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    if (!opts.imagePath.empty()) {
        LOG_INFO(kComponent, "one-shot image mode (config: ", opts.configPath,
                 ", profile: ", config.processing.activeModelProfile, ")");
        return runImageTest(config.processing, opts.imagePath);
    }

    LOG_INFO(kComponent, "ANPR system starting (config: ", opts.configPath,
             ", source: ", config.capture.source,
             ", profile: ", config.processing.activeModelProfile, ")");
    return runPipeline(config);
}
