/**
 * @file PlateDeduplicator.h
 * @brief Time-window based suppression of repeated plate reports.
 */
#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

namespace anpr {

/**
 * @brief Suppresses re-reports of the same plate text within a time window.
 *
 * A plate seen on consecutive frames is reported once; it is reported again
 * only after `windowSeconds` (config: processing.dedup_window_seconds) have
 * passed since the last *accepted* report. Sightings inside the window
 * refresh nothing, so a car parked in view reports once per window, not once
 * per frame.
 *
 * Engine-agnostic by design: lives outside IPlateRecognizer so every engine
 * gets identical dedup behavior. Not thread-safe; owned by the processing
 * thread.
 */
class PlateDeduplicator {
public:
    using Clock = std::chrono::steady_clock;

    explicit PlateDeduplicator(double windowSeconds)
        : window_(std::chrono::duration_cast<Clock::duration>(
              std::chrono::duration<double>(windowSeconds))) {}

    /**
     * @brief Decide whether a sighting should be reported.
     * @return true if this plate was not reported within the window (the
     *         sighting is then recorded as reported).
     */
    bool shouldReport(const std::string& plateText) {
        const auto now = Clock::now();
        pruneExpired(now);
        const auto [it, inserted] = lastReported_.try_emplace(plateText, now);
        if (inserted) {
            return true;
        }
        if (now - it->second >= window_) {
            it->second = now;
            return true;
        }
        return false;
    }

    /// Number of plates currently inside their suppression window.
    std::size_t activeCount() const { return lastReported_.size(); }

private:
    void pruneExpired(Clock::time_point now) {
        for (auto it = lastReported_.begin(); it != lastReported_.end();) {
            if (now - it->second >= window_) {
                it = lastReported_.erase(it);
            } else {
                ++it;
            }
        }
    }

    Clock::duration window_;
    std::unordered_map<std::string, Clock::time_point> lastReported_;
};

} // namespace anpr
