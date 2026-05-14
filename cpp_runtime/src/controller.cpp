#include "adas/controller.hpp"

#include <algorithm>
#include <cmath>

namespace adas {

Controller::Controller(int deadband_px, float max_steering, float ema_alpha)
    : deadband_px_(deadband_px),
      max_steering_(max_steering),
      ema_alpha_(ema_alpha) {}

SteeringOutput Controller::compute(int lane_center_x, int frame_w) {
    SteeringOutput out;
    if (lane_center_x < 0 || frame_w <= 0) {
        out.smoothed = has_prev_ ? prev_smoothed_ : 0.0f;
        return out;
    }
    int err = lane_center_x - frame_w / 2;
    if (std::abs(err) <= deadband_px_) err = 0;
    float s = static_cast<float>(err) / std::max(frame_w / 2.0f, 1.0f);
    s = std::max(-max_steering_, std::min(max_steering_, s));
    out.error_px = err;
    out.steering = s;
    out.smoothed = has_prev_ ? (1.0f - ema_alpha_) * prev_smoothed_ + ema_alpha_ * s : s;
    prev_smoothed_ = out.smoothed;
    has_prev_ = true;
    return out;
}

void Controller::reset() {
    has_prev_ = false;
    prev_smoothed_ = 0.0f;
}

}  // namespace adas
