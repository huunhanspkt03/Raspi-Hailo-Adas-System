#pragma once

#include "adas/deeplab_postprocess.hpp"

#include <opencv2/opencv.hpp>

namespace adas {

// Image-space lane-center estimator. Uses lane-line classes (solid/dotted,
// yellow/white) as boundaries when available, and falls back to the
// lane_area centroid when markings are missing. Lightweight enough for Pi.
class LaneCenterEstimator {
public:
    struct Config {
        int   row_step_div    = 40;    // sample every H / row_step_div rows
        int   min_valid_rows  = 6;     // below this, result is marked invalid
        float nominal_width_f = 0.35f; // initial nominal lane width (fraction of W)
        float min_width_ratio = 0.5f;  // reject markings narrower than median * this
        float max_width_ratio = 1.8f;  // reject markings wider than median * this
        float ema_alpha_ctrl  = 0.35f; // EMA on control_x
        float ema_alpha_width = 0.10f; // slow EMA on lane width
        int   max_jump_px     = 120;   // clip per-frame control_x jump
    };

    LaneCenterEstimator();
    explicit LaneCenterEstimator(Config cfg);
    void reset();

    // Mutates lane.fitted_path, lane.control_x/y, lane.confidence, lane.valid.
    // lane.class_masks and lane.lane_area_mask must already be populated.
    void estimate(LaneResult& lane, int frame_w, int frame_h);

private:
    Config cfg_;
    bool   has_prev_ctrl_ = false;
    float  prev_ctrl_x_   = 0.0f;
    float  nominal_width_ = 0.0f;
};

}  // namespace adas
