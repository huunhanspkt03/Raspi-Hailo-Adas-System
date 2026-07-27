#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace adas {

class StageTimer {
public:
    explicit StageTimer(bool enabled = false, double ema_alpha = 0.1)
        : enabled_(enabled), alpha_(ema_alpha) {}

    bool enabled() const { return enabled_; }
    void set_enabled(bool v) { enabled_ = v; }

    void record(const std::string& name, double ms) {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lock(m_);
        auto it = ema_ms_.find(name);
        if (it == ema_ms_.end()) {
            ema_ms_[name] = ms;
        } else {
            it->second = alpha_ * ms + (1.0 - alpha_) * it->second;
        }
        double& mx = max_ms_[name];
        if (ms > mx) mx = ms;
        ++count_[name];
    }

    void frame_end() {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lock(m_);
        ++frame_count_;
    }

    std::uint64_t frame_count() const {
        std::lock_guard<std::mutex> lock(m_);
        return frame_count_;
    }

    bool should_log(int every_n_frames) const {
        if (!enabled_ || every_n_frames <= 0) return false;
        std::lock_guard<std::mutex> lock(m_);
        return frame_count_ > 0 &&
               (frame_count_ % static_cast<std::uint64_t>(every_n_frames)) == 0;
    }

    std::string format_summary() const {
        if (!enabled_) return "";
        std::lock_guard<std::mutex> lock(m_);
        if (ema_ms_.empty()) return "stage_timer: no samples yet";
        std::vector<std::string> stages;
        stages.reserve(ema_ms_.size());
        std::size_t width = 5;  // len("stage")
        for (const auto& kv : ema_ms_) {
            stages.push_back(kv.first);
            width = std::max(width, kv.first.size());
        }
        std::sort(stages.begin(), stages.end());

        std::ostringstream os;
        os << "stage_timer (frames=" << frame_count_
           << ", alpha=" << std::fixed << std::setprecision(2) << alpha_ << "):\n";
        os << "  " << std::left << std::setw(static_cast<int>(width)) << "stage"
           << "    ema_ms     max_ms   count\n";
        for (const auto& s : stages) {
            double ema = ema_ms_.at(s);
            double mx  = max_ms_.at(s);
            std::uint64_t cnt = count_.at(s);
            os << "  " << std::left << std::setw(static_cast<int>(width)) << s
               << "  " << std::right << std::fixed << std::setprecision(2)
               << std::setw(8) << ema
               << "  " << std::setw(8) << mx
               << "  " << std::setw(6) << cnt << "\n";
        }
        std::string out = os.str();
        if (!out.empty() && out.back() == '\n') out.pop_back();
        return out;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(m_);
        ema_ms_.clear();
        max_ms_.clear();
        count_.clear();
        frame_count_ = 0;
    }

private:
    bool enabled_;
    double alpha_;
    mutable std::mutex m_;
    std::map<std::string, double> ema_ms_;
    std::map<std::string, double> max_ms_;
    std::map<std::string, std::uint64_t> count_;
    std::uint64_t frame_count_{0};
};

// RAII scope timer. Records elapsed ms on destruction. When the underlying
// timer is disabled the constructor/destructor degrade to one bool check
// each — no heap allocation, no clock read.
class ScopedStage {
public:
    ScopedStage(StageTimer& t, const char* name)
        : t_(t), name_(name) {
        if (t_.enabled()) t0_ = std::chrono::steady_clock::now();
    }
    ~ScopedStage() {
        if (!t_.enabled()) return;
        auto dt = std::chrono::steady_clock::now() - t0_;
        double ms = std::chrono::duration<double, std::milli>(dt).count();
        t_.record(name_, ms);
    }
    ScopedStage(const ScopedStage&) = delete;
    ScopedStage& operator=(const ScopedStage&) = delete;

private:
    StageTimer& t_;
    const char* name_;
    std::chrono::steady_clock::time_point t0_{};
};

}  // namespace adas
