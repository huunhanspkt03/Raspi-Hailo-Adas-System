#pragma once

namespace adas {

struct SteeringOutput {
    int error_px = 0;
    float steering = 0.0f;      // [-max_steering, max_steering]
    float smoothed = 0.0f;      // EMA-smoothed steering
};

class Controller {
public:
    Controller(int deadband_px = 20,
               float max_steering = 1.0f,
               float ema_alpha = 0.4f);

    // Returns a non-valid (zeroed) result when lane_center_x < 0.
    SteeringOutput compute(int lane_center_x, int frame_w);

    void reset();

private:
    int deadband_px_;
    float max_steering_;
    float ema_alpha_;
    bool has_prev_ = false;
    float prev_smoothed_ = 0.0f;
};

}  // namespace adas
