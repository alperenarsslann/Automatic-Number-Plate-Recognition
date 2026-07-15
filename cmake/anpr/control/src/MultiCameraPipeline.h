/**
 * @file MultiCameraPipeline.h
 * @brief Orchestrates up to 64 cameras through a shared ALPR worker pool.
 */
#pragma once

#include <atomic>

#include "core/Config.h"

namespace anpr {

/**
 * @brief Multi-camera pipeline (Frigate/DeepStream-style architecture).
 *
 * Per camera: one capture thread (decode -> optional downscale -> motion
 * gate -> tile update). Frames that pass the motion gate AND the camera's
 * ALPR switch enter ONE shared bounded queue (drop-oldest). A fixed pool of
 * ALPR worker threads (processing.worker_count) drains that queue, so total
 * inference CPU is capped regardless of camera count — cameras compete for
 * inference instead of multiplying it.
 *
 * Optional extras handled here:
 *  - Grid display: an N×N wall of all cameras (click a tile to toggle that
 *    camera's ALPR at runtime; q/ESC quits).
 *  - Embedded REST API: status, camera list, per-camera ALPR toggle and a
 *    recent-plates feed (see docs/ARCHITECTURE.md for endpoints).
 */
int runMultiCameraPipeline(const AppConfig& config, std::atomic<bool>& stopRequested);

} // namespace anpr
