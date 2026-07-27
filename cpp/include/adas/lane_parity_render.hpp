#pragma once
#include <limits>

#include <opencv2/core.hpp>

#include "adas/lane_parity.hpp"      // LaneSource / LaneState / BoundaryTrackResult
#include "adas/types.hpp"

namespace adas {


enum class CenterSource {
    Lost,
    LaneAreaOnly,
    PurpleOnly,        // purple usable, no lane_area
    PurplePrimary,     // both purple detected + lane_area present
    PurpleTracked,     // purple usable but not both freshly detected
};
const char* to_string(CenterSource s);

enum class RoadType { Unknown, Straight, CurveLeft, CurveRight };
const char* to_string(RoadType s);

struct RedCenter {
    bool         valid{false};
    CenterSource source{CenterSource::Lost};
    RoadType     road_type{RoadType::Unknown};
    float        center_confidence{0.0f};
    float        purple_confidence{0.0f};
    float        lane_area_confidence{0.0f};
    // Sampled per-row centerline (in image px).
    std::vector<cv::Point2f> center_pts_img;
    // Purple per-row midpoint (only when both sides DETECTED + separable).
    std::vector<cv::Point2f> purple_center_pts_img;
    // lane_area row-centroid polyline used by the lookahead resolver when
    // SOURCE == LaneAreaOnly.
    std::vector<cv::Point2f> lane_area_center_pts_img;
    // Quadratic fit of the resolved centerline. For LaneAreaOnly the
    // tracker forces deg=1 (a = 0).
    float poly_a{0.0f};
    float poly_b{0.0f};
    float poly_c{0.0f};
    const char* reason{"no evidence"};
};

// ----------------------------------------------------------------------------
// Lookahead target + ego-anchored path
// ----------------------------------------------------------------------------

struct LaneTargetState {
    bool  has_prev{false};
    float prev_target_x{0.0f};
    void reset() { has_prev = false; prev_target_x = 0.0f; }
};

struct LookaheadTarget {
    float target_x{0.0f};       // smoothed; defaults to ego_x on cold start
    float target_y{0.0f};
    bool  raw_present{false};
    float raw_target_x{0.0f};
    bool  accepted{false};
    // ego_anchored_path clamps for this source.
    float ego_max_curvature_a{0.0f};
    float ego_deadband_a{0.0f};
    float ego_max_top_disp_frac_w{0.0f};
};

LookaheadTarget resolve_lookahead_target(const RedCenter& red, int frame_h,
                                         int frame_w, LaneTargetState& state);

struct EgoPathResult {
    std::vector<float> xs;     // per-y rendered xs (same length as input ys)
    float a{0.0f};
    float anchor_y{0.0f};
    float lookahead_y{0.0f};
    float control_x{0.0f};
    int   control_y{0};
    bool  applied{false};      // a != 0
    bool  top_disp_clamped{false};
    bool  curvature_capped{false};  // |a_raw| > max_curvature_a was soft-capped
    const char* reason{"no_targets"};
};

EgoPathResult build_ego_anchored_path(const std::vector<cv::Point2f>& target_pts,
                                      const std::vector<float>& ys,
                                      int frame_h, int frame_w,
                                      float max_curvature_a,
                                      float deadband_a,
                                      float max_top_disp_frac_w);

// ----------------------------------------------------------------------------
// CorridorCurveState (purple-curvature fallback when ego-anchored target
// is unavailable). Mirrors lane_corridor_render_curve.CorridorCurveState.
// ----------------------------------------------------------------------------

struct CorridorCurveState {
    bool  has_prev{false};
    float prev_aC{0.0f};
    void reset() { has_prev = false; prev_aC = 0.0f; }
};

// ----------------------------------------------------------------------------
// Render corridor result
// ----------------------------------------------------------------------------

struct RenderCorridor {
    bool valid{false};
    // Per-row sampled ys for all polylines.
    std::vector<float> ys;
    // Centerline xs after ego-anchor / purple curvature.
    std::vector<float> red_xs;
    // Yellow corridor offsets along red's normal.
    std::vector<float> yellow_left_xs;
    std::vector<float> yellow_right_xs;
    // Purple polylines — empty per side when not detected.
    std::vector<float> purple_left_xs;
    std::vector<float> purple_right_xs;
    LaneState  lane_state{LaneState::Lost};
    const char* reason{"no detected purple line"};
    bool used_ego_anchor{false};
    // Diagnostic fields (populated for the periodic log line; not consumed
    // by the overlay). NaN/zero when not available.
    CenterSource red_source{CenterSource::Lost};
    float        purple_confidence{0.0f};
    float        lane_area_confidence{0.0f};
    float        center_confidence{0.0f};
    float        raw_target_x{std::numeric_limits<float>::quiet_NaN()};
    float        target_x{std::numeric_limits<float>::quiet_NaN()};
    float        ego_max_curvature_a{0.0f};   // effective clamp used
    bool         ego_curvature_capped{false}; // soft-cap triggered
};

RenderCorridor build_render_corridor(const BoundaryTrackResult& track,
                                     const RedCenter& red,
                                     int frame_h, int frame_w,
                                     LaneTargetState& target_state,
                                     CorridorCurveState& curve_state);

// ----------------------------------------------------------------------------
// Top-level (red center + render corridor) used by lane_parity_tracker.cpp
// ----------------------------------------------------------------------------

RedCenter compute_red_centerline(const cv::Mat& lane_area,
                                 const BoundaryTrackResult& track,
                                 int frame_h, int frame_w);

LaneGeometry make_fixed_guidance_geometry(int frame_h, int frame_w);

}  // namespace adas
