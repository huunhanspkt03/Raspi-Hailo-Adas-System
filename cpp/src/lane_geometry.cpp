#include "adas/lane_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// The image-mode compute_lane_geometry() implementation was removed: this
// build runs the python_parity lane path exclusively (see lane_parity.hpp /
// lane_parity_tracker.cpp). compute_steering stays here because it consumes
// the LaneGeometry produced by the parity path.

namespace adas {

namespace {

float red_center_x_at_y(const std::vector<cv::Point2f>& red_center, float y) {
    if (red_center.empty()) return std::numeric_limits<float>::quiet_NaN();
    if (red_center.size() == 1) return red_center.front().x;

    const auto& first = red_center.front();
    const auto& last = red_center.back();
    if (y <= first.y) return first.x;
    if (y >= last.y) return last.x;

    for (std::size_t i = 1; i < red_center.size(); ++i) {
        const auto& a = red_center[i - 1];
        const auto& b = red_center[i];
        if ((a.y <= y && y <= b.y) || (b.y <= y && y <= a.y)) {
            const float dy = b.y - a.y;
            if (std::fabs(dy) < 1e-6f) return b.x;
            const float t = (y - a.y) / dy;
            return a.x + t * (b.x - a.x);
        }
    }
    return last.x;
}

// Steering lane-evaluation ROI expressed as image-height fractions (mirrors
// LaneGeometryConfig::roi_top_frac). Steering follows the red centerline only
// inside this band, which keeps the servo on the near, reliable rows instead of
// the noisy vanishing region near the top of the rendered polyline:
//   * kSteerEvalTopFrac  — start evaluating the lane at 0.55H from the image
//                          top (the lookahead row).
//   * kSteerEvalBandFrac — the band extends ~0.40H further down toward the
//                          bottom for the near row. The near sample is clamped
//                          to the geometry's actual near end (≈0.72H render
//                          bottom) by red_center_x_at_y, so this only widens
//                          the band as far as the existing geometry provides.
constexpr float kSteerEvalTopFrac{0.55f};
constexpr float kSteerEvalBandFrac{0.40f};

}  // namespace

float compute_steering(const LaneGeometry& geom, int frame_w, int frame_h) {
    if (!geom.valid || geom.fitted_pts_img.empty()) return 0.0f;
    const float fit_top  = geom.fitted_pts_img.front().y;  // far / top of fit
    const float fit_near = geom.fitted_pts_img.back().y;   // near / bottom of fit
    const float h = static_cast<float>(frame_h);
    // Lookahead row at 0.55H, clamped into the fitted span. Near row at
    // 0.55H+0.40H=0.95H, clamped to the fit's near end by red_center_x_at_y.
    const float y_look = std::clamp(kSteerEvalTopFrac * h, fit_top, fit_near);
    const float y_near = std::clamp(
        (kSteerEvalTopFrac + kSteerEvalBandFrac) * h, y_look, fit_near);
    const float x_look = red_center_x_at_y(geom.fitted_pts_img, y_look);
    const float x_near = red_center_x_at_y(geom.fitted_pts_img, y_near);
    if (!std::isfinite(x_look) || !std::isfinite(x_near)) return 0.0f;

    // Steering follows only the red-center shape: a vertical/straight red
    // center produces zero steering even if it is offset in the camera view.
    // Curvature is represented by how far the lookahead point moves left/right
    // from the near red-center point.
    const float shape_dx = x_look - x_near;
    // Stronger lane-centering: 0.08 * frame_w (was 0.10 -> 0.15). Smaller scale
    // = more steering per pixel of red-center offset, so the servo reaches a
    // larger angle for the same correction and saturates (full lock) sooner.
    // The physical MAX angle is still set by servo_left_us/servo_right_us
    // (RcConfig, 50/130 deg); this only changes how hard it drives toward it.
    // No deadband here (intentionally reverted), so it also reacts to small
    // offsets.
    const float scale = std::max(1.0f, 0.08f * static_cast<float>(frame_w));
    return std::clamp(shape_dx / scale, -1.0f, 1.0f);
}

}  // namespace adas
