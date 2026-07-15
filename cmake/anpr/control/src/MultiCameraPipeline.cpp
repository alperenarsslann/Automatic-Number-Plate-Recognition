#include "MultiCameraPipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <deque>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "capture/FrameSourceFactory.h"
#include "core/BoundedQueue.h"
#include "core/Logger.h"
#include "core/Types.h"
#include "processing/MotionGate.h"
#include "processing/PlateDeduplicator.h"
#include "processing/PlateRecognizerFactory.h"

namespace anpr {

namespace {

constexpr const char* kComponent = "Pipeline";
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

std::string isoTimestamp(Timestamp tp) {
    const std::time_t t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    return buf;
}

/// One accepted plate report, kept in a ring buffer for the API.
struct PlateRecord {
    std::string cameraId;
    std::string text;
    float detectionConfidence = 0;
    float ocrConfidence = 0;
    std::string timestamp;
    std::uint64_t frameSequence = 0;
};

/// Per-camera runtime state shared between threads.
struct CameraRuntime {
    CameraInstanceConfig cfg;
    std::size_t index = 0;

    std::atomic<bool> alprEnabled{true};
    std::atomic<bool> online{false};
    std::atomic<bool> motionActive{false};
    std::atomic<std::uint64_t> frames{0};
    std::atomic<std::uint64_t> alprSubmitted{0};
    std::atomic<std::uint64_t> alprDropped{0};
    std::atomic<std::uint64_t> reports{0};

    std::mutex tileMutex;
    cv::Mat tile;                 ///< Latest frame downscaled for the grid.
    cv::Size frameSize;           ///< Size the detections below refer to.
    PlateDetectionList lastDetections;
    Clock::time_point lastDetectionAt{};

    std::thread thread;
};

struct AlprJob {
    std::size_t cameraIndex = 0;
    Frame frame;
};

/// Everything the worker/API/display threads share.
struct PipelineShared {
    std::vector<std::unique_ptr<CameraRuntime>> cameras;
    BoundedQueue<AlprJob> alprQueue;
    std::atomic<std::size_t> activeCaptures{0};
    std::atomic<bool> fatalError{false};

    std::mutex dedupMutex;
    PlateDeduplicator dedup;

    std::mutex platesMutex;
    std::deque<PlateRecord> recentPlates; ///< Newest first, bounded.

    std::atomic<std::uint64_t> totalReports{0};
    std::atomic<std::uint64_t> alprMicros{0};
    std::atomic<std::uint64_t> alprRuns{0};

    PipelineShared(std::size_t queueCapacity, double dedupWindow)
        : alprQueue(queueCapacity), dedup(dedupWindow) {}
};

// --------------------------------------------------------------------- capture

void captureLoop(CameraRuntime& cam, PipelineShared& shared, const AppConfig& config,
                 std::atomic<bool>& stop, int tileWidth) {
    CaptureConfig capture;
    capture.source = cam.cfg.source;
    capture.simulation = cam.cfg.simulation;
    capture.camera = cam.cfg.camera;

    std::unique_ptr<IFrameSource> source;
    try {
        source = createFrameSource(capture);
    } catch (const std::exception& e) {
        LOG_ERROR(kComponent, "[", cam.cfg.id, "] cannot create source: ", e.what());
        return;
    }

    // Real cameras keep retrying (network hiccups); simulations fail fast.
    while (!stop.load() && !source->open()) {
        LOG_ERROR(kComponent, "[", cam.cfg.id, "] open failed: ", source->lastError());
        if (cam.cfg.source != "camera") return;
        std::this_thread::sleep_for(std::chrono::duration<double>(
            cam.cfg.camera.reconnectIntervalSeconds));
    }
    if (stop.load()) return;

    cam.online.store(true);
    MotionGate motionGate(cam.cfg.motion);
    const int everyN = std::max(1, config.processing.processEveryNFrames);
    const int maxWidth = config.processing.maxFrameWidth;

    Frame frame;
    std::uint64_t counter = 0;
    while (!stop.load()) {
        if (!source->getNextFrame(frame)) {
            if (!source->lastError().empty()) {
                LOG_ERROR(kComponent, "[", cam.cfg.id, "] source failed: ", source->lastError());
            } else {
                LOG_INFO(kComponent, "[", cam.cfg.id, "] end of stream");
            }
            break;
        }
        frame.sourceId = cam.cfg.id; // Reports carry the camera id.

        if (maxWidth > 0 && frame.image.cols > maxWidth) {
            const double s = maxWidth / static_cast<double>(frame.image.cols);
            cv::Mat resized;
            cv::resize(frame.image, resized, cv::Size(), s, s, cv::INTER_AREA);
            frame.image = std::move(resized);
        }

        cam.frames.fetch_add(1);
        ++counter;

        double motionFraction = 0.0;
        const bool motion = motionGate.shouldProcess(frame.image, motionFraction);
        cam.motionActive.store(motion);

        if (motion && cam.alprEnabled.load() && (counter - 1) % everyN == 0) {
            if (shared.alprQueue.pushDropOldest(AlprJob{cam.index, frame})) {
                cam.alprDropped.fetch_add(1);
            }
            cam.alprSubmitted.fetch_add(1);
        }

        // Tile is kept fresh unconditionally: both the display wall and the
        // API snapshot endpoint (embedded Studio wall) read from it.
        {
            const double s = tileWidth / static_cast<double>(frame.image.cols);
            std::lock_guard<std::mutex> lock(cam.tileMutex);
            cam.frameSize = frame.image.size();
            cv::resize(frame.image, cam.tile, cv::Size(), s, s, cv::INTER_AREA);
        }
    }

    cam.online.store(false);
    source->close();
    if (shared.activeCaptures.fetch_sub(1) == 1) {
        shared.alprQueue.close(); // Last capture thread out closes the queue.
    }
}

// --------------------------------------------------------------------- workers

void workerLoop(int workerId, PipelineShared& shared, const AppConfig& config,
                std::atomic<bool>& stop) {
    std::unique_ptr<IPlateRecognizer> recognizer;
    try {
        recognizer = createPlateRecognizer(config.processing);
    } catch (const std::exception& e) {
        LOG_ERROR(kComponent, "worker ", workerId, ": cannot create recognizer: ", e.what());
        shared.fatalError.store(true);
        stop.store(true);
        return;
    }
    if (!recognizer->initialize()) {
        LOG_ERROR(kComponent, "worker ", workerId, ": init failed: ", recognizer->lastError());
        shared.fatalError.store(true);
        stop.store(true);
        return;
    }

    while (!stop.load()) {
        auto job = shared.alprQueue.waitPop(100ms);
        if (!job) {
            if (shared.alprQueue.isClosed()) break;
            continue;
        }
        CameraRuntime& cam = *shared.cameras[job->cameraIndex];

        const auto started = Clock::now();
        auto detections = recognizer->recognize(job->frame);
        shared.alprMicros.fetch_add(static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - started)
                .count()));
        shared.alprRuns.fetch_add(1);

        for (const auto& d : detections) {
            bool report;
            {
                std::lock_guard<std::mutex> lock(shared.dedupMutex);
                report = shared.dedup.shouldReport(d.sourceId + "|" + d.text);
            }
            if (!report) continue;

            cam.reports.fetch_add(1);
            shared.totalReports.fetch_add(1);
            std::ostringstream oss;
            oss << "PLATE " << d.text << "  cam=" << d.sourceId << " det=" << std::fixed
                << std::setprecision(2) << d.detectionConfidence << " ocr=" << d.ocrConfidence
                << " box=[" << d.boundingBox.x << ',' << d.boundingBox.y << ' '
                << d.boundingBox.width << 'x' << d.boundingBox.height << "] seq="
                << d.frameSequence;
            LOG_INFO(kComponent, oss.str());

            std::lock_guard<std::mutex> lock(shared.platesMutex);
            shared.recentPlates.push_front(PlateRecord{d.sourceId, d.text,
                                                       d.detectionConfidence, d.ocrConfidence,
                                                       isoTimestamp(d.timestamp),
                                                       d.frameSequence});
            if (shared.recentPlates.size() > 1000) shared.recentPlates.pop_back();
        }

        {
            std::lock_guard<std::mutex> lock(cam.tileMutex);
            cam.lastDetections = std::move(detections);
            cam.lastDetectionAt = Clock::now();
        }
    }
}

// ------------------------------------------------------------------- rendering

/// Draw detection boxes, the status banner and the state border onto an
/// image whose content came from the camera's tile. Shared by the OpenCV
/// display wall and the API snapshot endpoint (embedded Studio wall).
void decorate(cv::Mat& img, CameraRuntime& cam, const cv::Size& frameSize,
              const PlateDetectionList& dets, bool fresh) {
    if (fresh && frameSize.width > 0) {
        const double sx = img.cols / static_cast<double>(frameSize.width);
        const double sy = img.rows / static_cast<double>(frameSize.height);
        for (const auto& d : dets) {
            const cv::Rect box(static_cast<int>(d.boundingBox.x * sx),
                               static_cast<int>(d.boundingBox.y * sy),
                               static_cast<int>(d.boundingBox.width * sx),
                               static_cast<int>(d.boundingBox.height * sy));
            cv::rectangle(img, box, cv::Scalar(0, 255, 0), 1);
            cv::putText(img, d.text, cv::Point(box.x, std::max(box.y - 3, 10)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }
    }

    const bool alprOn = cam.alprEnabled.load();
    cv::rectangle(img, cv::Rect(0, 0, img.cols, 16), cv::Scalar(32, 32, 32), cv::FILLED);
    std::string label = cam.cfg.name + (alprOn ? "  [ALPR]" : "  [off]");
    if (!cam.online.load()) label += "  OFFLINE";
    cv::putText(img, label, cv::Point(4, 12), cv::FONT_HERSHEY_SIMPLEX, 0.35,
                alprOn ? cv::Scalar(0, 255, 0) : cv::Scalar(160, 160, 160), 1, cv::LINE_AA);
    if (cam.motionActive.load()) {
        cv::circle(img, cv::Point(img.cols - 8, 8), 4, cv::Scalar(0, 215, 255), cv::FILLED);
    }
    cv::rectangle(img, cv::Rect(0, 0, img.cols, img.rows),
                  cam.online.load() ? (alprOn ? cv::Scalar(0, 160, 0) : cv::Scalar(90, 90, 90))
                                    : cv::Scalar(0, 0, 160),
                  1);
}

/// Latest decorated frame for one camera, or empty if none arrived yet.
cv::Mat renderSnapshot(CameraRuntime& cam) {
    cv::Mat img;
    cv::Size frameSize;
    PlateDetectionList dets;
    bool fresh = false;
    {
        std::lock_guard<std::mutex> lock(cam.tileMutex);
        if (cam.tile.empty()) return {};
        img = cam.tile.clone();
        frameSize = cam.frameSize;
        dets = cam.lastDetections;
        fresh = (Clock::now() - cam.lastDetectionAt) < 3s;
    }
    decorate(img, cam, frameSize, dets, fresh);
    return img;
}

// ------------------------------------------------------------------------- api

class ApiServer {
public:
    ApiServer(const ApiConfig& config, PipelineShared& shared) : config_(config), shared_(shared) {
        server_.Get("/api/status", [this](const httplib::Request&, httplib::Response& res) {
            nlohmann::json j;
            std::uint64_t frames = 0;
            std::size_t online = 0;
            for (const auto& cam : shared_.cameras) {
                frames += cam->frames.load();
                online += cam->online.load() ? 1 : 0;
            }
            j["cameras_total"] = shared_.cameras.size();
            j["cameras_online"] = online;
            j["frames_total"] = frames;
            j["alpr_runs"] = shared_.alprRuns.load();
            j["alpr_mean_ms"] = shared_.alprRuns.load()
                                    ? shared_.alprMicros.load() / 1000.0 / shared_.alprRuns.load()
                                    : 0.0;
            j["alpr_queue_depth"] = shared_.alprQueue.size();
            j["reports_total"] = shared_.totalReports.load();
            res.set_content(j.dump(2), "application/json");
        });

        server_.Get("/api/cameras", [this](const httplib::Request&, httplib::Response& res) {
            nlohmann::json j = nlohmann::json::array();
            for (const auto& cam : shared_.cameras) {
                j.push_back({{"id", cam->cfg.id},
                             {"name", cam->cfg.name},
                             {"online", cam->online.load()},
                             {"alpr_enabled", cam->alprEnabled.load()},
                             {"motion_active", cam->motionActive.load()},
                             {"motion_gating", cam->cfg.motion.enabled},
                             {"frames", cam->frames.load()},
                             {"alpr_submitted", cam->alprSubmitted.load()},
                             {"alpr_dropped", cam->alprDropped.load()},
                             {"reports", cam->reports.load()}});
            }
            res.set_content(j.dump(2), "application/json");
        });

        // POST /api/cameras/<id>/alpr  body: {"enabled": true|false}
        server_.Post(R"(/api/cameras/([^/]+)/alpr)",
                     [this](const httplib::Request& req, httplib::Response& res) {
                         const std::string id = req.matches[1];
                         for (const auto& cam : shared_.cameras) {
                             if (cam->cfg.id != id) continue;
                             try {
                                 const auto body = nlohmann::json::parse(req.body);
                                 const bool enabled = body.at("enabled").get<bool>();
                                 cam->alprEnabled.store(enabled);
                                 LOG_INFO(kComponent, "[", id, "] ALPR ",
                                          enabled ? "enabled" : "disabled", " via API");
                                 res.set_content(nlohmann::json{{"id", id}, {"alpr_enabled", enabled}}.dump(),
                                                 "application/json");
                             } catch (const std::exception& e) {
                                 res.status = 400;
                                 res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                                                 "application/json");
                             }
                             return;
                         }
                         res.status = 404;
                         res.set_content(nlohmann::json{{"error", "unknown camera: " + id}}.dump(),
                                         "application/json");
                     });

        // GET /api/cameras/<id>/snapshot — latest decorated frame as JPEG.
        // Polled by the embedded camera wall in AnprStudio.
        server_.Get(R"(/api/cameras/([^/]+)/snapshot)",
                    [this](const httplib::Request& req, httplib::Response& res) {
                        const std::string id = req.matches[1];
                        for (const auto& cam : shared_.cameras) {
                            if (cam->cfg.id != id) continue;
                            const cv::Mat img = renderSnapshot(*cam);
                            if (img.empty()) {
                                res.status = 503;
                                res.set_content(nlohmann::json{{"error", "no frame yet"}}.dump(),
                                                "application/json");
                                return;
                            }
                            std::vector<uchar> jpeg;
                            cv::imencode(".jpg", img, jpeg, {cv::IMWRITE_JPEG_QUALITY, 75});
                            res.set_content(reinterpret_cast<const char*>(jpeg.data()),
                                            jpeg.size(), "image/jpeg");
                            return;
                        }
                        res.status = 404;
                        res.set_content(nlohmann::json{{"error", "unknown camera: " + id}}.dump(),
                                        "application/json");
                    });

        server_.Get("/api/plates", [this](const httplib::Request& req, httplib::Response& res) {
            std::size_t limit = 50;
            if (req.has_param("limit")) {
                limit = std::clamp<std::size_t>(std::stoul(req.get_param_value("limit")), 1, 1000);
            }
            nlohmann::json j = nlohmann::json::array();
            std::lock_guard<std::mutex> lock(shared_.platesMutex);
            for (const auto& p : shared_.recentPlates) {
                if (j.size() >= limit) break;
                j.push_back({{"camera", p.cameraId},
                             {"plate", p.text},
                             {"detection_confidence", p.detectionConfidence},
                             {"ocr_confidence", p.ocrConfidence},
                             {"timestamp", p.timestamp},
                             {"frame", p.frameSequence}});
            }
            res.set_content(j.dump(2), "application/json");
        });
    }

    bool start() {
        if (!server_.bind_to_port(config_.bindAddress, config_.port)) {
            LOG_ERROR(kComponent, "API cannot bind ", config_.bindAddress, ":", config_.port);
            return false;
        }
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        LOG_INFO(kComponent, "API listening on http://", config_.bindAddress, ":", config_.port,
                 " (endpoints: /api/status /api/cameras /api/plates, POST /api/cameras/<id>/alpr)");
        return true;
    }

    void stop() {
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }

private:
    ApiConfig config_;
    PipelineShared& shared_;
    httplib::Server server_;
    std::thread thread_;
};

// --------------------------------------------------------------------- display

struct GridLayout {
    int cols = 1;
    int rows = 1;
    int tileWidth = 640;
    int tileHeight = 360;
};

GridLayout makeGrid(std::size_t cameraCount, int maxWidth) {
    GridLayout g;
    g.cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(cameraCount))));
    g.rows = static_cast<int>(std::ceil(cameraCount / static_cast<double>(g.cols)));
    g.tileWidth = std::max(96, maxWidth / g.cols);
    g.tileHeight = g.tileWidth * 9 / 16;
    return g;
}

struct MouseContext {
    PipelineShared* shared = nullptr;
    GridLayout grid;
};

void onMouse(int event, int x, int y, int /*flags*/, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    auto* ctx = static_cast<MouseContext*>(userdata);
    const std::size_t index = static_cast<std::size_t>(y / ctx->grid.tileHeight) * ctx->grid.cols +
                              static_cast<std::size_t>(x / ctx->grid.tileWidth);
    if (index >= ctx->shared->cameras.size()) return;
    CameraRuntime& cam = *ctx->shared->cameras[index];
    const bool enabled = !cam.alprEnabled.load();
    cam.alprEnabled.store(enabled);
    LOG_INFO(kComponent, "[", cam.cfg.id, "] ALPR ", enabled ? "enabled" : "disabled",
             " via display click");
}

void drawTile(cv::Mat& cell, CameraRuntime& cam) {
    cv::Mat tile;
    cv::Size frameSize;
    PlateDetectionList dets;
    bool fresh = false;
    {
        std::lock_guard<std::mutex> lock(cam.tileMutex);
        if (!cam.tile.empty()) tile = cam.tile.clone();
        frameSize = cam.frameSize;
        dets = cam.lastDetections;
        fresh = (Clock::now() - cam.lastDetectionAt) < 3s;
    }

    if (tile.empty()) {
        cell.setTo(cv::Scalar(24, 24, 24));
    } else {
        cv::resize(tile, cell, cell.size(), 0, 0, cv::INTER_LINEAR);
    }
    decorate(cell, cam, frameSize, dets, fresh && !tile.empty());
}

} // namespace

int runMultiCameraPipeline(const AppConfig& config, std::atomic<bool>& stopRequested) {
    // Only enabled cameras take part.
    std::vector<CameraInstanceConfig> enabledCameras;
    for (const auto& cam : config.cameras) {
        if (cam.enabled) enabledCameras.push_back(cam);
    }
    if (enabledCameras.empty()) {
        LOG_ERROR(kComponent, "no enabled cameras in the configuration");
        return 1;
    }

    PipelineShared shared(config.processing.queueCapacity, config.processing.dedupWindowSeconds);
    for (std::size_t i = 0; i < enabledCameras.size(); ++i) {
        auto cam = std::make_unique<CameraRuntime>();
        cam->cfg = enabledCameras[i];
        cam->index = i;
        cam->alprEnabled.store(cam->cfg.alprEnabled);
        shared.cameras.push_back(std::move(cam));
    }

    const GridLayout grid = makeGrid(shared.cameras.size(), config.display.maxWidth);
    LOG_INFO(kComponent, "starting: ", shared.cameras.size(), " camera(s), ",
             config.processing.workerCount, " ALPR worker(s), queue capacity ",
             config.processing.queueCapacity,
             config.display.enabled
                 ? ", grid " + std::to_string(grid.cols) + "x" + std::to_string(grid.rows) +
                       " (click tile = toggle ALPR, q/ESC = quit)"
                 : " (headless)");

    // Workers first (models load once per worker), then captures.
    std::vector<std::thread> workers;
    for (int w = 0; w < config.processing.workerCount; ++w) {
        workers.emplace_back(workerLoop, w, std::ref(shared), std::cref(config),
                             std::ref(stopRequested));
    }
    shared.activeCaptures.store(shared.cameras.size());
    for (auto& cam : shared.cameras) {
        cam->thread = std::thread(captureLoop, std::ref(*cam), std::ref(shared),
                                  std::cref(config), std::ref(stopRequested), grid.tileWidth);
    }

    ApiServer api(config.api, shared);
    const bool apiRunning = config.api.enabled && api.start();

    // Periodic stats, printed from whichever loop below is active.
    auto lastStatsAt = Clock::now();
    std::uint64_t lastFrames = 0;
    const auto maybeLogStats = [&] {
        const auto now = Clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastStatsAt);
        if (elapsed < 5s) return;
        std::uint64_t frames = 0;
        std::size_t motion = 0;
        std::uint64_t submitted = 0;
        for (const auto& cam : shared.cameras) {
            frames += cam->frames.load();
            submitted += cam->alprSubmitted.load();
            motion += cam->motionActive.load() ? 1 : 0;
        }
        const double fps = (frames - lastFrames) * 1000.0 / elapsed.count();
        const auto runs = shared.alprRuns.load();
        LOG_INFO(kComponent, "stats: cams=", shared.cameras.size(), " fps_total=", fps,
                 " motion_active=", motion, " alpr_submitted=", submitted, " alpr_runs=", runs,
                 " alpr_mean_ms=", runs ? shared.alprMicros.load() / 1000.0 / runs : 0.0,
                 " queue=", shared.alprQueue.size(), " reports=", shared.totalReports.load());
        lastFrames = frames;
        lastStatsAt = now;
    };

    const auto allCapturesDone = [&] { return shared.alprQueue.isClosed(); };

    if (config.display.enabled) {
        constexpr const char* kWindow = "ANPR — camera wall";
        cv::namedWindow(kWindow, cv::WINDOW_AUTOSIZE);
        MouseContext mouseCtx{&shared, grid};
        cv::setMouseCallback(kWindow, onMouse, &mouseCtx);
        cv::Mat canvas(grid.rows * grid.tileHeight, grid.cols * grid.tileWidth, CV_8UC3);

        while (!stopRequested.load() && !allCapturesDone()) {
            canvas.setTo(cv::Scalar(16, 16, 16));
            for (std::size_t i = 0; i < shared.cameras.size(); ++i) {
                const int cx = static_cast<int>(i) % grid.cols;
                const int cy = static_cast<int>(i) / grid.cols;
                cv::Mat cell = canvas(cv::Rect(cx * grid.tileWidth, cy * grid.tileHeight,
                                               grid.tileWidth, grid.tileHeight));
                drawTile(cell, *shared.cameras[i]);
            }
            cv::imshow(kWindow, canvas);
            const int key = cv::waitKey(33); // ~30 fps wall refresh.
            if (key == 'q' || key == 27) stopRequested.store(true);
            maybeLogStats();
        }
        cv::destroyAllWindows();
    } else {
        while (!stopRequested.load() && !allCapturesDone()) {
            std::this_thread::sleep_for(200ms);
            maybeLogStats();
        }
    }

    stopRequested.store(true);
    shared.alprQueue.close();
    for (auto& cam : shared.cameras) {
        if (cam->thread.joinable()) cam->thread.join();
    }
    for (auto& worker : workers) {
        worker.join();
    }
    if (apiRunning) api.stop();

    std::uint64_t frames = 0;
    for (const auto& cam : shared.cameras) frames += cam->frames.load();
    LOG_INFO(kComponent, shared.fatalError.load() ? "stopped after fatal error" : "finished",
             ": ", frames, " frames across ", shared.cameras.size(), " camera(s), ",
             shared.alprRuns.load(), " ALPR runs, ", shared.totalReports.load(),
             " plate reports");
    return shared.fatalError.load() ? 1 : 0;
}

} // namespace anpr
