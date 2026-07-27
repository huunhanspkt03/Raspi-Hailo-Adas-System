#pragma once


#include <opencv2/core.hpp>

#include "adas/lane_geometry.hpp"   // LaneGeometryConfig, LaneGeometry
#include "adas/types.hpp"

namespace adas {

// Per-side tracker source — mirrors lane_boundary_tracker_contract.py.
enum class LaneSource { Detected, Held, Lost };
const char* to_string(LaneSource s);

enum class LaneState { BothDetected, OneSideDetected, HeldPrevious, Lost };
const char* to_string(LaneState s);

// Quadratic in y: x = a*y^2 + b*y + c. Same convention as Python coeffs
// returned by np.polyfit(deg=2) (highest-order first), repackaged.
struct LaneCoeffs {
    float a{0.0f};
    float b{0.0f};
    float c{0.0f};
    bool  valid{false};
};

struct PerSideResult {
    LaneSource source{LaneSource::Lost};
    LaneCoeffs coeffs;            // current (possibly EMA-blended) coeffs
    float confidence_final{0.0f}; // 0..1 (decayed when HELD)
    int   n_pixels{0};            // raw frame-coord pixel count when DETECTED
    int   held_streak{0};
    int   raw_missing_streak{0};
};

struct BoundaryTrackResult {
    PerSideResult left;
    PerSideResult right;
    LaneState     lane_state{LaneState::Lost};
    float         boundary_confidence{0.0f}; // min(left.cf, right.cf)
    // Sampled polylines in image coordinates (frame_w x frame_h). Empty
    // when source is LOST.
    std::vector<cv::Point2f> left_pts_img;
    std::vector<cv::Point2f> right_pts_img;
};

// Forward declaration — defined in lane_parity_tracker.cpp. Header-only
// for the tracker would balloon this header; the class lives in the
// translation unit and we expose only its construct / update / reset.
class LaneBoundaryTracker;

// Build / reset a tracker instance with defaults matching the Python
// LaneBoundaryTracker constructor.
LaneBoundaryTracker* make_lane_boundary_tracker();
void destroy_lane_boundary_tracker(LaneBoundaryTracker* t);
void reset_lane_boundary_tracker(LaneBoundaryTracker* t);

// One-shot tracker update for the current frame. ``line_union`` and
// ``lane_area`` are the STDC1 0/1 masks at frame resolution.
BoundaryTrackResult tracker_update(LaneBoundaryTracker* t,
                                   const cv::Mat& line_union,
                                   const cv::Mat& lane_area,
                                   int frame_h, int frame_w);

// Top-level Phase-1 geometry entry. Wraps tracker_update + a temporary
// midpoint centerline (placeholder until Phase 2 ports the ego-anchored
// red center). Returns a LaneGeometry compatible with the existing
// overlay code: left_corridor/right_corridor = purple polylines,
// fitted_pts_img = midpoint, control_x/y = bottom of midpoint.
LaneGeometry compute_lane_geometry_python_parity(
    LaneBoundaryTracker* tracker,
    const cv::Mat& lane_area,
    const cv::Mat& line_union,
    const LaneGeometryConfig& cfg = {});

// Process-wide gate for [lane_parity] diagnostic / fallback / invalid
// stderr messages. Set once at startup from --quiet. Errors (e.g. a
// genuinely first-time fallback or first invalid) still print so a real
// problem can't be silenced entirely.
void set_lane_parity_quiet_logs(bool quiet);

}  // namespace adas
