/**
 * @file IFrameSource.h
 * @brief Abstract interface for any video frame producer.
 */
#pragma once

#include <string>

#include "core/Types.h"

namespace anpr {

/**
 * @brief Abstract source of video frames.
 *
 * Implementations:
 *  - SimulatedFileFrameSource: reads frames from a video file (development /
 *    testing without physical hardware).
 *  - HikvisionRtspFrameSource: reads frames from the Hikvision IP camera over
 *    RTSP (real deployment; skeleton until the hardware is available).
 *
 * Upper layers depend only on this interface and must not care which
 * implementation is active. Implementations are not required to be
 * thread-safe; the capture thread is the sole caller.
 */
class IFrameSource {
public:
    virtual ~IFrameSource() = default;

    /**
     * @brief Open the underlying source (file, RTSP stream, ...).
     * @return true on success; on failure lastError() describes the cause.
     */
    virtual bool open() = 0;

    /**
     * @brief Retrieve the next frame, blocking until one is available.
     *
     * @param[out] frame Filled with image data and metadata on success.
     * @return true if a frame was produced; false on end-of-stream or error
     *         (distinguish via isOpen()/lastError()). A looping simulated
     *         source never reports end-of-stream.
     */
    virtual bool getNextFrame(Frame& frame) = 0;

    /// Release the underlying source. Safe to call multiple times.
    virtual void close() = 0;

    /// @return true while the source is open and able to produce frames.
    virtual bool isOpen() const = 0;

    /// Human-readable identifier used in logs and Frame::sourceId.
    virtual std::string id() const = 0;

    /// Description of the last error, or empty string if none occurred.
    virtual std::string lastError() const = 0;
};

} // namespace anpr
