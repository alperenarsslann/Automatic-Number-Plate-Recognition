#include "MultiCameraPipeline.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
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
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "capture/FrameSourceFactory.h"
#include "core/BoundedQueue.h"
#include "core/Logger.h"
#include "core/Types.h"
#include "core/interfaces/INetworkTransport.h"
#include "network/MessageSchema.h"
#include "network/NetworkTransportFactory.h"
#include "processing/MotionGate.h"
#include "processing/PlateDeduplicator.h"
#include "processing/PlateRecognizerFactory.h"
#include "processing/PlateTracker.h"

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

/// Filesystem-safe timestamp, e.g. "2026-07-18_20-16-44-123".
std::string fileTimestamp(Timestamp tp) {
    using namespace std::chrono;
    const auto ms = duration_cast<milliseconds>(tp.time_since_epoch()) % 1000;
    const std::time_t t = system_clock::to_time_t(tp);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", &tm);
    char out[40];
    std::snprintf(out, sizeof(out), "%s-%03d", buf, static_cast<int>(ms.count()));
    return out;
}

/**
 * @brief Write an annotated JPG for one reported plate (visual verification).
 *
 * Draws the plate box + recognized text on the representative frame and,
 * when enabled, burns a timestamp/plate banner. Filename:
 *   <dir>/<timestamp>_<camera>_<plate>.jpg
 */
void saveDetectionImage(const DetectionOutputConfig& cfg, const PlateDetection& d,
                        const cv::Mat& frame) {
    if (frame.empty()) {
        LOG_DEBUG(kComponent, "detection image skipped for ", d.text, " (no frame captured)");
        return;
    }
    cv::Mat canvas = frame.clone();
    cv::rectangle(canvas, d.boundingBox, cv::Scalar(0, 255, 0), 2);

    std::ostringstream label;
    label << d.text << "  " << std::fixed << std::setprecision(2) << d.ocrConfidence;
    cv::putText(canvas, label.str(), cv::Point(d.boundingBox.x, std::max(d.boundingBox.y - 8, 18)),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

    if (cfg.drawTimestamp) {
        std::ostringstream banner;
        banner << isoTimestamp(d.timestamp) << "  cam=" << d.sourceId << "  plate=" << d.text;
        cv::rectangle(canvas, cv::Rect(0, canvas.rows - 24, canvas.cols, 24),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(canvas, banner.str(), cv::Point(6, canvas.rows - 7),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    std::error_code ec;
    std::filesystem::create_directories(cfg.directory, ec);
    if (ec) {
        LOG_WARN(kComponent, "cannot create detection dir '", cfg.directory, "': ", ec.message());
        return;
    }

    // Sanitize the plate text for a filename (alnum only).
    std::string safePlate;
    for (const char c : d.text) {
        if (std::isalnum(static_cast<unsigned char>(c))) safePlate.push_back(c);
    }
    const std::string path = cfg.directory + "/" + fileTimestamp(d.timestamp) + "_" + d.sourceId +
                             "_" + (safePlate.empty() ? "plate" : safePlate) + ".jpg";
    const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY,
                                     std::clamp(cfg.jpegQuality, 1, 100)};
    if (cv::imwrite(path, canvas, params)) {
        LOG_DEBUG(kComponent, "saved detection image ", path);
    } else {
        LOG_WARN(kComponent, "failed to write detection image ", path);
    }
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

    /// Temporal consolidation of this camera's readings (one final per plate).
    std::mutex trackerMutex;
    std::unique_ptr<PlateTracker> tracker;

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

    /// Optional uplink to the site server (null when networking disabled).
    /// Workers push plate messages; inbound commands are applied to cameras.
    INetworkTransport* transport = nullptr;

    PipelineShared(std::size_t queueCapacity, double dedupWindow)
        : alprQueue(queueCapacity), dedup(dedupWindow) {}

    CameraRuntime* findCamera(const std::string& id) {
        for (auto& cam : cameras) {
            if (cam->cfg.id == id) return cam.get();
        }
        return nullptr;
    }
};

// --------------------------------------------------------------------- capture

void processFrame(CameraRuntime& cam, PipelineShared& shared, Frame& frame,
                  MotionGate& motionGate, std::uint64_t& counter, int everyN, int maxWidth,
                  int tileWidth);

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

    // Live cameras never give up: open failures AND mid-stream errors both
    // lead back here after reconnectIntervalSeconds. Simulations fail fast
    // and end on end-of-stream.
    const bool isLiveCamera = cam.cfg.source == "camera";
    const auto reconnectDelay =
        std::chrono::duration<double>(cam.cfg.camera.reconnectIntervalSeconds);
    MotionGate motionGate(cam.cfg.motion);
    const int everyN = std::max(1, config.processing.processEveryNFrames);
    const int maxWidth = config.processing.maxFrameWidth;

    Frame frame;
    std::uint64_t counter = 0;
    bool finished = false;
    while (!stop.load() && !finished) {
        if (!source->open()) {
            LOG_ERROR(kComponent, "[", cam.cfg.id, "] open failed: ", source->lastError());
            if (!isLiveCamera) break;
            std::this_thread::sleep_for(reconnectDelay);
            continue;
        }
        cam.online.store(true);

        while (!stop.load()) {
            if (!source->getNextFrame(frame)) {
                if (!source->lastError().empty()) {
                    LOG_ERROR(kComponent, "[", cam.cfg.id, "] source failed: ",
                              source->lastError());
                } else {
                    LOG_INFO(kComponent, "[", cam.cfg.id, "] end of stream");
                }
                break;
            }
            processFrame(cam, shared, frame, motionGate, counter, everyN, maxWidth, tileWidth);
        }

        cam.online.store(false);
        source->close();
        if (!isLiveCamera) {
            finished = true;
        } else if (!stop.load()) {
            LOG_WARN(kComponent, "[", cam.cfg.id, "] reconnecting in ",
                     cam.cfg.camera.reconnectIntervalSeconds, " s");
            std::this_thread::sleep_for(reconnectDelay);
        }
    }

    cam.online.store(false);
    if (shared.activeCaptures.fetch_sub(1) == 1) {
        shared.alprQueue.close(); // Last capture thread out closes the queue.
    }
}

/// Per-frame work shared by the capture loop: downscale, motion gate,
/// fan out to the ALPR queue and refresh the display/snapshot tile.
void processFrame(CameraRuntime& cam, PipelineShared& shared, Frame& frame,
                  MotionGate& motionGate, std::uint64_t& counter, int everyN, int maxWidth,
                  int tileWidth) {
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

// --------------------------------------------------------------------- workers

/// Emit ONE plate report: counters, log line, site-server uplink, API buffer.
/// Called for finalized (consolidated) plates, or per-reading in the fallback
/// path when consolidation is disabled.
void reportPlate(PipelineShared& shared, CameraRuntime& cam, const PlateDetection& d) {
    cam.reports.fetch_add(1);
    shared.totalReports.fetch_add(1);

    std::ostringstream oss;
    oss << "PLATE " << d.text << "  cam=" << d.sourceId << " det=" << std::fixed
        << std::setprecision(2) << d.detectionConfidence << " ocr=" << d.ocrConfidence
        << " box=[" << d.boundingBox.x << ',' << d.boundingBox.y << ' ' << d.boundingBox.width
        << 'x' << d.boundingBox.height << "] seq=" << d.frameSequence;
    LOG_INFO(kComponent, oss.str());

    if (shared.transport) {
        shared.transport->send(buildPlateMessage(d));
    }

    std::lock_guard<std::mutex> lock(shared.platesMutex);
    shared.recentPlates.push_front(PlateRecord{d.sourceId, d.text, d.detectionConfidence,
                                               d.ocrConfidence, isoTimestamp(d.timestamp),
                                               d.frameSequence});
    if (shared.recentPlates.size() > 1000) shared.recentPlates.pop_back();
}

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

        const bool saveImages = config.detectionOutput.enabled;
        if (config.processing.consolidation.enabled) {
            // Defer reporting: feed the tracker (with the frame, kept only for
            // the strongest reading); finals emitted by the finalization pass.
            std::lock_guard<std::mutex> lock(cam.trackerMutex);
            for (const auto& d : detections) {
                cam.tracker->add(d, saveImages ? job->frame.image : cv::Mat{});
            }
        } else {
            // Fallback: exact-text dedup, report immediately.
            for (const auto& d : detections) {
                bool report;
                {
                    std::lock_guard<std::mutex> lock(shared.dedupMutex);
                    report = shared.dedup.shouldReport(d.sourceId + "|" + d.text);
                }
                if (report) {
                    reportPlate(shared, cam, d);
                    if (saveImages) saveDetectionImage(config.detectionOutput, d, job->frame.image);
                }
            }
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
        // API-key gate: runs before every route. When apiKeys is empty the
        // API is open (localhost dev); otherwise a valid key is required.
        server_.set_pre_routing_handler(
            [this](const httplib::Request& req, httplib::Response& res) {
                if (authorized(req)) {
                    return httplib::Server::HandlerResponse::Unhandled; // continue to route
                }
                res.status = 401;
                res.set_header("WWW-Authenticate", "Bearer");
                res.set_content(nlohmann::json{{"error", "missing or invalid API key"}}.dump(),
                                "application/json");
                return httplib::Server::HandlerResponse::Handled; // stop here
            });

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
        const bool localhostOnly =
            config_.bindAddress == "127.0.0.1" || config_.bindAddress == "localhost";
        if (config_.apiKeys.empty() && !localhostOnly) {
            LOG_WARN(kComponent, "API bound to ", config_.bindAddress,
                     " with NO api_keys — anyone on the network has full control. "
                     "Set api.api_keys to require authentication.");
        }
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        LOG_INFO(kComponent, "API listening on http://", config_.bindAddress, ":", config_.port,
                 config_.apiKeys.empty() ? " (no auth)" : " (API key required)",
                 " (endpoints: /api/status /api/cameras /api/plates /api/cameras/<id>/snapshot, "
                 "POST /api/cameras/<id>/alpr)");
        return true;
    }

    void stop() {
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }

private:
    /// True when the request carries an accepted key, or when auth is disabled.
    bool authorized(const httplib::Request& req) const {
        if (config_.apiKeys.empty()) {
            return true; // Auth disabled (empty key list).
        }
        std::string provided;
        if (req.has_header("X-API-Key")) {
            provided = req.get_header_value("X-API-Key");
        } else if (req.has_header("Authorization")) {
            const std::string value = req.get_header_value("Authorization");
            constexpr const char* kBearer = "Bearer ";
            if (value.rfind(kBearer, 0) == 0) {
                provided = value.substr(std::strlen(kBearer));
            }
        }
        if (provided.empty()) return false;
        // Constant-ish time comparison against each configured key.
        for (const auto& key : config_.apiKeys) {
            if (key.size() == provided.size()) {
                unsigned char diff = 0;
                for (std::size_t i = 0; i < key.size(); ++i) {
                    diff |= static_cast<unsigned char>(key[i] ^ provided[i]);
                }
                if (diff == 0) return true;
            }
        }
        return false;
    }

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
        cam->tracker = std::make_unique<PlateTracker>(config.processing.consolidation);
        shared.cameras.push_back(std::move(cam));
    }

    // Drain every camera's finalized (consolidated) plates and report them.
    const auto finalizePlates = [&shared, &config] {
        for (auto& cam : shared.cameras) {
            std::vector<PlateTracker::Finalized> finals;
            {
                std::lock_guard<std::mutex> lock(cam->trackerMutex);
                finals = cam->tracker->collectFinalized();
            }
            for (const auto& f : finals) {
                reportPlate(shared, *cam, f.detection);
                if (config.detectionOutput.enabled) {
                    saveDetectionImage(config.detectionOutput, f.detection, f.frame);
                }
            }
        }
    };

    // --- Threading policy (config-driven, with auto defaults) -------------
    // The pipeline is fully multi-threaded already (per-camera capture,
    // an ALPR worker pool, display, network, API). The knobs that matter for
    // CPU efficiency are how many ALPR workers run and how many threads each
    // inference uses — their product should stay near the core count so the
    // cores stay busy WITHOUT oversubscription (which wastes CPU on context
    // switches). Both auto-tune when left at 0.
    const int cores = std::max(1u, std::thread::hardware_concurrency());
    const int maxWorkers = std::max(2, cores - 1); // Leave a core for capture/display/OS.
    int workerCount = config.processing.workerCount;
    if (workerCount <= 0) {
        // Auto: concurrency comes from the cameras (each is an independent job
        // source). One worker per camera (at least two so a single camera can
        // still overlap bursts), capped below the core count. This is the
        // knob that prevents the real CPU waster: many cameras each spawning
        // an all-cores inference (N x cores threads => context-switch thrash).
        workerCount = std::clamp(static_cast<int>(shared.cameras.size()), 2, maxWorkers);
    }
    if (config.processing.numThreads > 0) {
        cv::setNumThreads(config.processing.numThreads);
    } else {
        // Auto-balance: share the cores across the workers so concurrent
        // inferences don't fight over them. Capped at 6 because OpenCV's DNN
        // scales poorly past that on these small models (sync overhead > gain).
        cv::setNumThreads(std::clamp(cores / std::max(1, workerCount), 1, 6));
    }
    LOG_INFO(kComponent, "threading: ", workerCount, " ALPR worker(s) x ",
             cv::getNumThreads(), " OpenCV thread(s) each (", cores, " cores)");

    const GridLayout grid = makeGrid(shared.cameras.size(), config.display.maxWidth);
    LOG_INFO(kComponent, "starting: ", shared.cameras.size(), " camera(s), ",
             workerCount, " ALPR worker(s), queue capacity ",
             config.processing.queueCapacity,
             config.display.enabled
                 ? ", grid " + std::to_string(grid.cols) + "x" + std::to_string(grid.rows) +
                       " (click tile = toggle ALPR, q/ESC = quit)"
                 : " (headless)");

    // Site-server uplink: plate reports flow out, commands (e.g. remote ALPR
    // toggle) flow back in and are applied to the matching camera.
    auto transport = createNetworkTransport(config.network);
    if (transport) {
        transport->setMessageHandler([&shared](const std::string& message) {
            const ServerCommand cmd = parseServerCommand(message);
            if (!cmd.valid) return;
            if (cmd.name == "set_alpr") {
                if (CameraRuntime* cam = shared.findCamera(cmd.cameraId)) {
                    cam->alprEnabled.store(cmd.boolArg);
                    LOG_INFO(kComponent, "[", cmd.cameraId, "] ALPR ",
                             cmd.boolArg ? "enabled" : "disabled", " via server command");
                } else {
                    LOG_WARN(kComponent, "server command for unknown camera: ", cmd.cameraId);
                }
            } else {
                LOG_DEBUG(kComponent, "unhandled server command: ", cmd.name);
            }
        });
        transport->setStateHandler([](ConnectionState state) {
            LOG_INFO(kComponent, "site server link: ",
                     state == ConnectionState::Connected      ? "connected"
                     : state == ConnectionState::Connecting    ? "connecting"
                                                               : "disconnected");
        });
        transport->start();
        shared.transport = transport.get();
    }

    // Workers first (models load once per worker), then captures.
    std::vector<std::thread> workers;
    for (int w = 0; w < workerCount; ++w) {
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
            finalizePlates();
            maybeLogStats();
        }
        cv::destroyAllWindows();
    } else {
        while (!stopRequested.load() && !allCapturesDone()) {
            std::this_thread::sleep_for(200ms);
            finalizePlates();
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

    // Emit any plates still being tracked when we stopped.
    if (config.processing.consolidation.enabled) {
        for (auto& cam : shared.cameras) {
            std::lock_guard<std::mutex> lock(cam->trackerMutex);
            for (const auto& f : cam->tracker->flushAll()) {
                reportPlate(shared, *cam, f.detection);
                if (config.detectionOutput.enabled) {
                    saveDetectionImage(config.detectionOutput, f.detection, f.frame);
                }
            }
        }
    }

    if (transport) {
        shared.transport = nullptr; // No more sends after this point.
        transport->stop();
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
