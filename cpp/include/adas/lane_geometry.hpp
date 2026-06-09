#pragma once

// Steering from the lane geometry produced by the python_parity path
// (lane_parity.hpp / lane_parity_render.hpp), plus the LaneGeometryConfig
// knobs shared with that path's signature.
//
// NOTE: the former image-mode compute_lane_geometry() (per-frame line scan
// + quadratic fit, no tracker) was REMOVED — this build runs python_parity
// exclusively. See git history if it ever needs to come back.

#include <opencv2/core.hpp>

#include "adas/types.hpp"

namespace adas {

struct LaneGeometryConfig {
    // Sampling ROI (fraction of frame_h).
    //   * Detection rows live in [roi_top_frac, roi_bottom_frac]. Matches
    //     Python features/lane_boundaries.DEFAULT_TOP_FRAC = 0.55.
    //   * Render rows live in [render_top_frac, render_bottom_frac]. Matches
    //     features/lane_corridor_render_constants:
    //       DEFAULT_ROI_TOP_FRAC=0.45, DEFAULT_ROI_BOTTOM_FRAC=0.72.
    //     The polynomial is *extrapolated* above the detection top into the
    //     render top — this is what keeps the yellow above the hood line
    //     and prevents it from spilling below 72% of frame_h into the
    //     bumper/hood region.
    float roi_top_frac{0.55f};
    float roi_bottom_frac{1.00f};
    float render_top_frac{0.45f};
    float render_bottom_frac{0.72f};
    int   num_render_points{24};

    // Per-row line-pixel acceptance (cluster of >=N within 4-px window).
    int   min_line_pixels_per_row{2};

    // Per-side fit acceptance — minimum sample rows per side.
    int   min_rows_per_side{8};

    // Lane-width gate at render rows. Mirrors Python
    //   features/lane_corridor_render_constants.DEFAULT_MIN_LANE_WIDTH_PX = 40
    // (a median check, NOT a per-row separation check — the per-row gate
    // we used previously triggered "corridor_too_narrow" on healthy frames
    // because vanishing-region rows naturally converge below the per-row
    // floor).
    float min_lane_width_px{40.0f};

    // Yellow-corridor perspective-aware half-width (fraction of frame_w),
    // linearly interpolated from far (top) to near (bottom). Mirrors
    //   DEFAULT_FIXED_YELLOW_HALF_WIDTH_FAR_FRAC  = 0.040
    //   DEFAULT_FIXED_YELLOW_HALF_WIDTH_NEAR_FRAC = 0.250
    // floored by half of min_inner_gap_px (matches Python's max(...) floor).
    float corridor_far_frac{0.04f};
    float corridor_near_frac{0.25f};
    float min_inner_gap_px{100.0f};  // DEFAULT_MIN_INNER_GAP_PX
};

// Lookahead-style steering value in [-1, 1]. Positive = steer right.
// Uses the fitted polyline if available, otherwise returns 0.
float compute_steering(const LaneGeometry& geom, int frame_w, int frame_h);

}  // namespace adas
