/**
 * @file FrameSourceFactory.h
 * @brief Creates the configured IFrameSource implementation.
 */
#pragma once

#include <memory>

#include "core/Config.h"
#include "core/interfaces/IFrameSource.h"

namespace anpr {

/**
 * @brief Instantiate the frame source selected by CaptureConfig::source
 *        ("simulation" -> SimulatedFileFrameSource,
 *         "camera"     -> HikvisionRtspFrameSource).
 *
 * Adding a new source type means adding a new IFrameSource implementation
 * and one branch here — no changes in upper layers.
 *
 * @throws std::runtime_error on unknown source type.
 */
std::unique_ptr<IFrameSource> createFrameSource(const CaptureConfig& config);

} // namespace anpr
