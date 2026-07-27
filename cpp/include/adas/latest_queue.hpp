#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>

namespace adas {

template <typename T>
class LatestQueue {
public:
    explicit LatestQueue(std::size_t maxsize = 1) : maxsize_(maxsize == 0 ? 1 : maxsize) {}

    void put(T item) {
        {
            std::lock_guard<std::mutex> lock(m_);
            if (closed_) return;
            if (buf_.size() >= maxsize_) {
                buf_.pop_front();
                ++dropped_;
            }
            buf_.push_back(std::move(item));
        }
        cv_.notify_one();
    }

    std::optional<T> get(std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
        std::unique_lock<std::mutex> lock(m_);
        if (!cv_.wait_for(lock, timeout, [&] { return !buf_.empty() || closed_; })) {
            return std::nullopt;
        }
        if (buf_.empty()) return std::nullopt;
        T v = std::move(buf_.front());
        buf_.pop_front();
        return v;
    }

    std::optional<T> try_get() {
        std::lock_guard<std::mutex> lock(m_);
        if (buf_.empty()) return std::nullopt;
        T v = std::move(buf_.front());
        buf_.pop_front();
        return v;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(m_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(m_);
        return buf_.size();
    }

    std::uint64_t dropped() const { return dropped_.load(); }
    bool closed() const {
        std::lock_guard<std::mutex> lock(m_);
        return closed_;
    }

private:
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<T> buf_;
    std::size_t maxsize_;
    bool closed_{false};
    std::atomic<std::uint64_t> dropped_{0};
};

}  // namespace adas
