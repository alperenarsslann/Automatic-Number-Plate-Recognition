/**
 * @file PlateTracker.h
 * @brief Temporal consolidation of noisy plate reads into one final result.
 */
#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "core/Config.h"
#include "core/Types.h"

namespace anpr {

/**
 * @brief Groups per-frame plate readings of the same physical plate and
 *        emits a single consolidated result when the plate leaves view.
 *
 * A passing car produces many jittery OCR strings for one plate
 * (K619879 / 1619879 / V619879 ...). Reporting each is noise; the operator
 * wants the final answer. This tracker:
 *  - assigns each reading to an existing track when it is close in both text
 *    (edit distance) and image position (box center), else opens a new track;
 *  - accumulates all readings with their OCR confidences;
 *  - finalizes a track after it has been idle for finalizeAfterSeconds (car
 *    gone) or lived longer than maxTrackSeconds, producing ONE detection via
 *    per-character, confidence-weighted majority voting.
 *
 * One instance per camera; not thread-safe (caller serializes access).
 */
class PlateTracker {
public:
    using Clock = std::chrono::steady_clock;

    /// A finalized plate plus the representative frame it was best seen in
    /// (empty frame when none was supplied). The frame lets the caller save a
    /// visual verification image annotated with the consolidated result.
    struct Finalized {
        PlateDetection detection;
        cv::Mat frame;
    };

    explicit PlateTracker(ConsolidationConfig config);

    /**
     * @brief Feed a fresh reading (one plate detected in one frame).
     * @param reading The detection.
     * @param frame   The frame it came from (kept only if this reading is the
     *                track's strongest, for later verification imagery). May
     *                be empty when imagery is not needed.
     */
    void add(const PlateDetection& reading, const cv::Mat& frame = {});

    /// Return consolidated results for tracks that have just finalized.
    std::vector<Finalized> collectFinalized();

    /// Finalize and return every remaining track (call on shutdown).
    std::vector<Finalized> flushAll();

private:
    struct Reading {
        std::string text;
        float ocrConfidence;
        float detectionConfidence;
        cv::Rect box;
    };
    struct Track {
        std::vector<Reading> readings;
        cv::Rect lastBox;
        cv::Point2f lastCenter;
        Timestamp firstTimestamp;
        std::uint64_t lastFrame = 0;
        std::string sourceId;
        Clock::time_point firstSeen;
        Clock::time_point lastSeen;
        cv::Mat bestFrame;         ///< Frame of the highest-confidence reading.
        float bestFrameConf = -1;  ///< Its OCR confidence.
    };

    /// Confidence-weighted per-position character vote across a track.
    PlateDetection consolidate(const Track& track) const;

    ConsolidationConfig config_;
    std::vector<Track> tracks_;
};

/// Levenshtein edit distance (exposed for testing/reuse).
int editDistance(const std::string& a, const std::string& b);

} // namespace anpr
