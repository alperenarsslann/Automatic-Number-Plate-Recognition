/**
 * @file BoundedQueue.h
 * @brief Bounded thread-safe queue for producer-consumer pipelines.
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace anpr {

/**
 * @brief Bounded MPMC queue built on std::mutex / std::condition_variable.
 *
 * Connects the pipeline threads (capture -> processing -> network). The
 * capacity bound prevents unbounded memory growth when a consumer falls
 * behind. Two overflow policies are offered per call site:
 *  - pushDropOldest(): evicts the oldest element (right choice for frames —
 *    stale video is worthless, the newest frame matters).
 *  - tryPush(): rejects the new element (right choice when older items must
 *    not be lost, e.g. plate reports awaiting transmission).
 *
 * close() wakes all blocked consumers so threads can shut down gracefully.
 */
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity ? capacity : 1) {}

    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;

    /**
     * @brief Push, evicting the oldest element if the queue is full.
     * @return true if an element was evicted to make room.
     */
    bool pushDropOldest(T value) {
        bool dropped = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) return false;
            if (queue_.size() >= capacity_) {
                queue_.pop_front();
                dropped = true;
            }
            queue_.push_back(std::move(value));
        }
        notEmpty_.notify_one();
        return dropped;
    }

    /**
     * @brief Push only if there is room.
     * @return false if the queue was full or closed (element not enqueued).
     */
    bool tryPush(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_ || queue_.size() >= capacity_) return false;
            queue_.push_back(std::move(value));
        }
        notEmpty_.notify_one();
        return true;
    }

    /**
     * @brief Block until an element is available, the timeout expires, or the
     *        queue is closed.
     * @return The element, or std::nullopt on timeout / closed-and-empty.
     */
    std::optional<T> waitPop(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        notEmpty_.wait_for(lock, timeout, [this] { return !queue_.empty() || closed_; });
        if (queue_.empty()) return std::nullopt;
        T value = std::move(queue_.front());
        queue_.pop_front();
        return value;
    }

    /// Non-blocking pop.
    std::optional<T> tryPop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return std::nullopt;
        T value = std::move(queue_.front());
        queue_.pop_front();
        return value;
    }

    /// Mark the queue closed and wake all waiting consumers.
    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        notEmpty_.notify_all();
    }

    bool isClosed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    std::size_t capacity() const { return capacity_; }

private:
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable notEmpty_;
    std::deque<T> queue_;
    bool closed_ = false;
};

} // namespace anpr
