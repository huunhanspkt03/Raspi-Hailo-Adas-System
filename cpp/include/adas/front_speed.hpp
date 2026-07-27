#pragma once

#include "adas/types.hpp"

namespace adas {

class FrontSpeedEstimator {
public:
    void update(BehaviorPacket& bp);

    // Drop all internal state. Called automatically on reset
    // conditions; exposed so callers can force-clear on video restart.
    void reset();

private:
    bool      has_prev_{false};
    FrameId   prev_object_frame_id_{0};
    TimePoint prev_time_{};            // object-packet timestamp of prev sample
    float     prev_distance_m_{0.0f};
    int       prev_track_id_{-1};      // lead identity the samples belong to
    bool      ema_ready_{false};
    float     ema_mps_{0.0f};       // positive = closing rate, m/s
    // Periodic log dedup: emit on first valid estimate and on closure
    // changes > kLogChangeKmh; emit a single "cleared" line on reset.
    bool  logged_active_{false};
    float last_logged_closing_kmh_{0.0f};
};

}  // namespace adas
