#pragma once
ack.

#include <opencv2/core.hpp>

#include "adas/types.hpp"

namespace adas {

struct LaneGeometryConfig {
    float roi_top_frac{0.55f};
    float roi_bottom_frac{1.00f};
    float render_top_frac{0.45f};
    float render_bottom_frac{0.72f};
    int   num_render_points{24};

    // Per-row line-pixel acceptance (cluster of >=N within 4-px window).
    int   min_line_pixels_per_row{2};

    // Per-side fit acceptance — minimum sample rows per side.
    int   min_rows_per_side{8};
    float min_lane_width_px{40.0f};
    float corridor_far_frac{0.04f};
    float corridor_near_frac{0.25f};
    float min_inner_gap_px{100.0f};  // DEFAULT_MIN_INNER_GAP_PX
};

float compute_steering(const LaneGeometry& geom, int frame_w, int frame_h);

}  // namespace adas
