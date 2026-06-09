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

}  // namespace

float compute_steering(const LaneGeometry& geom, int frame_w, int frame_h) {
    if (!geom.valid || geom.fitted_pts_img.empty()) return 0.0f;
    (void)frame_h;
    const float y_look = geom.fitted_pts_img.front().y;
    const float y_near = geom.fitted_pts_img.back().y;
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
