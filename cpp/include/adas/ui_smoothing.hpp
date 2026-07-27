#pragma once

#include <array>
#include <cmath>

namespace adas::ui {
template <int N>
class StickyLevel {
public:
    int update(int raw) {
        buf_[head_] = raw;
        head_ = (head_ + 1) % N;
        if (count_ < N) ++count_;
        if (raw >= shown_) {           // instant escalate toward danger
            shown_ = raw;
            return shown_;
        }
        // De-escalate to the highest level still backed by a window majority.
        while (shown_ > 0 && support(shown_) * 2 <= count_) --shown_;
        return shown_;
    }
    int value() const { return shown_; }
    void reset() { head_ = 0; count_ = 0; shown_ = 0; buf_.fill(0); }

private:
    int support(int level) const {     // frames in the window with value >= level
        int c = 0;
        for (int i = 0; i < count_; ++i) if (buf_[i] >= level) ++c;
        return c;
    }
    std::array<int, N> buf_{};
    int head_{0};
    int count_{0};
    int shown_{0};
};

class SteerStatus {
public:
    // keep / depart: the |steer| thresholds (e.g. 0.10 / 0.22). margin: the
    // hysteresis half-width (e.g. 0.03). Writes the smoothed level + direction.
    void update(float steer, float keep, float depart, float margin,
                int* out_level, int* out_dir) {
        const float mag = std::fabs(steer);
        int lv = level_;
        if (lv <= 1 && mag >= depart + margin)      lv = 2;   // -> depart
        else if (lv == 2 && mag <= depart - margin) lv = 1;   // depart -> turn
        if (lv == 0 && mag >= keep + margin)        lv = 1;   // keep -> turn
        else if (lv == 1 && mag <= keep - margin)   lv = 0;   // turn -> keep
        level_ = lv;

        if (steer <= -(keep + margin))      dir_ = -1;
        else if (steer >= (keep + margin))  dir_ = +1;
        else if (mag <= keep - margin)      dir_ = 0;
        // else: inside the hysteresis band -> hold the previous direction.

        *out_level = level_;
        *out_dir = dir_;
    }
    void reset() { level_ = 0; dir_ = 0; }

private:
    int level_{0};
    int dir_{0};
};

}  // namespace adas::ui
