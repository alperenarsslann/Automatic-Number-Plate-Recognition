/**
 * @file Logger.h
 * @brief Minimal thread-safe leveled logger.
 *
 * Deliberately dependency-free (no spdlog etc.) to keep the embedded build
 * light. Output format:
 *   2026-07-15T14:03:12.345 [INFO ] [Capture] message text
 */
#pragma once

#include <mutex>
#include <sstream>
#include <string>

namespace anpr {

/// Log severity levels, ordered from most to least verbose.
enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

/// Parse a level name ("debug", "info", "warn", "error"); defaults to Info.
LogLevel logLevelFromString(const std::string& name);

/**
 * @brief Process-wide logger singleton.
 *
 * Thread-safe: concurrent log() calls from capture/processing/network
 * threads are serialized with a mutex. Writes to stderr so stdout stays
 * free for the interactive CLI.
 */
class Logger {
public:
    static Logger& instance();

    /// Set the minimum level that will be emitted (from config / CLI).
    void setLevel(LogLevel level);
    LogLevel level() const;

    /**
     * @brief Emit a log record if @p level passes the configured threshold.
     * @param level     Severity of this record.
     * @param component Short subsystem tag, e.g. "Capture", "Network".
     * @param message   Preformatted message text.
     */
    void log(LogLevel level, const std::string& component, const std::string& message);

private:
    Logger() = default;

    mutable std::mutex mutex_;
    LogLevel level_ = LogLevel::Info;
};

namespace detail {
/// Fold arbitrary streamable arguments into one string.
template <typename... Args>
std::string concat(Args&&... args) {
    std::ostringstream oss;
    (oss << ... << std::forward<Args>(args));
    return oss.str();
}
} // namespace detail

} // namespace anpr

/// Convenience macros: LOG_INFO("Capture", "opened ", path, " fps=", fps);
#define LOG_DEBUG(component, ...) \
    ::anpr::Logger::instance().log(::anpr::LogLevel::Debug, component, ::anpr::detail::concat(__VA_ARGS__))
#define LOG_INFO(component, ...) \
    ::anpr::Logger::instance().log(::anpr::LogLevel::Info, component, ::anpr::detail::concat(__VA_ARGS__))
#define LOG_WARN(component, ...) \
    ::anpr::Logger::instance().log(::anpr::LogLevel::Warn, component, ::anpr::detail::concat(__VA_ARGS__))
#define LOG_ERROR(component, ...) \
    ::anpr::Logger::instance().log(::anpr::LogLevel::Error, component, ::anpr::detail::concat(__VA_ARGS__))
