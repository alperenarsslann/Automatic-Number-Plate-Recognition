#include "processing/PlateTracker.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>

namespace anpr {

namespace {

cv::Point2f centerOf(const cv::Rect& box) {
    return {box.x + box.width * 0.5f, box.y + box.height * 0.5f};
}

} // namespace

int editDistance(const std::string& a, const std::string& b) {
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    if (n == 0) return static_cast<int>(m);
    if (m == 0) return static_cast<int>(n);
    std::vector<int> prev(m + 1);
    std::vector<int> curr(m + 1);
    std::iota(prev.begin(), prev.end(), 0);
    for (std::size_t i = 1; i <= n; ++i) {
        curr[0] = static_cast<int>(i);
        for (std::size_t j = 1; j <= m; ++j) {
            const int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost});
        }
        std::swap(prev, curr);
    }
    return prev[m];
}

PlateTracker::PlateTracker(ConsolidationConfig config) : config_(std::move(config)) {}

void PlateTracker::add(const PlateDetection& reading, const cv::Mat& frame) {
    const cv::Point2f center = centerOf(reading.boundingBox);
    // Spatial gate scales with the frame; we do not know the frame size here,
    // so treat maxCenterDistance as a fraction of a nominal 1080p diagonal.
    // Reads of one plate on consecutive frames stay well within this.
    const double diagonal = std::hypot(1920.0, 1080.0);
    const double maxDist = config_.maxCenterDistance * diagonal;

    Track* best = nullptr;
    int bestScore = config_.maxEditDistance + 1;
    for (auto& track : tracks_) {
        if (track.sourceId != reading.sourceId) continue;
        const double dist = std::hypot(center.x - track.lastCenter.x,
                                       center.y - track.lastCenter.y);
        if (dist > maxDist) continue;
        // Compare against the track's most recent reading text.
        const int d = editDistance(reading.text, track.readings.back().text);
        if (d < bestScore) {
            bestScore = d;
            best = &track;
        }
    }

    const auto now = Clock::now();
    if (best) {
        best->readings.push_back({reading.text, reading.ocrConfidence,
                                  reading.detectionConfidence, reading.boundingBox});
        best->lastBox = reading.boundingBox;
        best->lastCenter = center;
        best->lastFrame = reading.frameSequence;
        best->lastSeen = now;
        if (!frame.empty() && reading.ocrConfidence > best->bestFrameConf) {
            best->bestFrame = frame.clone();
            best->bestFrameConf = reading.ocrConfidence;
        }
        return;
    }

    Track track;
    track.readings.push_back({reading.text, reading.ocrConfidence,
                              reading.detectionConfidence, reading.boundingBox});
    track.lastBox = reading.boundingBox;
    track.lastCenter = center;
    track.firstTimestamp = reading.timestamp;
    track.lastFrame = reading.frameSequence;
    track.sourceId = reading.sourceId;
    track.firstSeen = now;
    track.lastSeen = now;
    if (!frame.empty()) {
        track.bestFrame = frame.clone();
        track.bestFrameConf = reading.ocrConfidence;
    }
    tracks_.push_back(std::move(track));
}

PlateDetection PlateTracker::consolidate(const Track& track) const {
    // 1) Pick the target length: the confidence-weighted most common length.
    std::map<std::size_t, double> lengthWeight;
    for (const auto& r : track.readings) {
        lengthWeight[r.text.size()] += r.ocrConfidence;
    }
    const std::size_t targetLen =
        std::max_element(lengthWeight.begin(), lengthWeight.end(),
                         [](const auto& a, const auto& b) { return a.second < b.second; })
            ->first;

    // 2) Per-position, confidence-weighted character vote among readings that
    //    share the target length.
    std::string consolidated;
    double confidenceAccum = 0.0;
    for (std::size_t pos = 0; pos < targetLen; ++pos) {
        std::map<char, double> charWeight;
        for (const auto& r : track.readings) {
            if (r.text.size() == targetLen) {
                charWeight[r.text[pos]] += r.ocrConfidence;
            }
        }
        const auto winner = std::max_element(
            charWeight.begin(), charWeight.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        consolidated.push_back(winner->first);
        confidenceAccum += winner->second;
    }

    // 3) Build the final detection. Confidence is the mean winning-vote weight
    //    normalized by the number of contributing readings.
    const auto sameLen = std::count_if(
        track.readings.begin(), track.readings.end(),
        [&](const Reading& r) { return r.text.size() == targetLen; });
    const double denom = static_cast<double>(std::max<std::size_t>(1, sameLen)) * targetLen;

    PlateDetection result;
    result.text = consolidated;
    result.ocrConfidence = static_cast<float>(confidenceAccum / std::max(1.0, denom));
    // Representative detection confidence + box: the strongest single reading.
    const auto bestReading = std::max_element(
        track.readings.begin(), track.readings.end(),
        [](const Reading& a, const Reading& b) { return a.ocrConfidence < b.ocrConfidence; });
    result.detectionConfidence = bestReading->detectionConfidence;
    result.boundingBox = bestReading->box;
    result.frameSequence = track.lastFrame;
    result.timestamp = track.firstTimestamp;
    result.sourceId = track.sourceId;
    return result;
}

std::vector<PlateTracker::Finalized> PlateTracker::collectFinalized() {
    std::vector<Finalized> finalized;
    const auto now = Clock::now();
    const auto idleLimit = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(config_.finalizeAfterSeconds));
    const auto lifeLimit = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(config_.maxTrackSeconds));

    for (auto it = tracks_.begin(); it != tracks_.end();) {
        const bool idle = (now - it->lastSeen) >= idleLimit;
        const bool tooOld = (now - it->firstSeen) >= lifeLimit;
        if (idle || tooOld) {
            if (static_cast<int>(it->readings.size()) >= config_.minSightings) {
                finalized.push_back({consolidate(*it), it->bestFrame});
            }
            it = tracks_.erase(it);
        } else {
            ++it;
        }
    }
    return finalized;
}

std::vector<PlateTracker::Finalized> PlateTracker::flushAll() {
    std::vector<Finalized> finalized;
    for (const auto& track : tracks_) {
        if (static_cast<int>(track.readings.size()) >= config_.minSightings) {
            finalized.push_back({consolidate(track), track.bestFrame});
        }
    }
    tracks_.clear();
    return finalized;
}

} // namespace anpr
