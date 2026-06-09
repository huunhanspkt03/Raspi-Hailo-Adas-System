#pragma once

// Common typed payloads passed between threaded stages (capture / lane /
// object / fusion / render). Mirrors src/pipeline/packets.py at the
// C++ level; field names match where practical so cross-language parity
// reviews are easy.

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace adas {

using FrameId = std::uint64_t;
using TimePoint = std::chrono::steady_clock::time_point;

inline TimePoint now() { return std::chrono::steady_clock::now(); }

struct FramePacket {
    FrameId frame_id{0};
    TimePoint timestamp{};
    cv::Mat frame;     // BGR uint8 — owned (shallow refcount semantics).
    int frame_h{0};
    int frame_w{0};
};

// Hard-class label IDs from the STDC1 semantic-mask dataset; identical to
// src/features/lane_semantic_postprocess.py::_CLASS_KEY_TO_ID.
namespace lane_class {
constexpr std::uint8_t kBackground = 0;
constexpr std::uint8_t kDottedLine = 1;
constexpr std::uint8_t kLaneArea = 2;
constexpr std::uint8_t kSolidLine = 3;
constexpr int kNumClasses = 4;
}  // namespace lane_class

struct LaneGeometry {
    bool valid{false};
    int control_x{0};       // bottom-of-fit x in source-frame px
    int control_y{0};       // bottom-of-fit y in source-frame px
    float confidence{0.0f};
    std::vector<cv::Point2f> fitted_pts_img;  // red centerline polyline
    std::vector<cv::Point2f> left_corridor;   // yellow corridor left edge
    std::vector<cv::Point2f> right_corridor;  // yellow corridor right edge
    // Strict lane-membership gate used by FCWS/front-object selection.
    // Built from the red centerline plus learned purple lane width, so it
    // tracks the driven lane instead of the wider UI/drive corridor.
    std::vector<cv::Point2f> strict_lane_left;
    std::vector<cv::Point2f> strict_lane_right;
    // Quadratic centerline coefficients [a, b, c]: x(y) = a*y^2 + b*y + c.
    float poly_a{0.0f};
    float poly_b{0.0f};
    float poly_c{0.0f};

    // Optional separate purple boundary polylines (python_parity path).
    // Empty when the source path doesn't compute them (image path or when
    // the tracker reports the side as LOST). When present these are
    // drawn in the Python solid_line color (magenta) underneath the
    // yellow corridor; the overlay leaves them alone when they're empty.
    std::vector<cv::Point2f> purple_left_pts_img;
    std::vector<cv::Point2f> purple_right_pts_img;
};

struct LanePacket {
    FrameId frame_id{0};
    TimePoint timestamp{};
    cv::Mat source_frame;     // shallow ref to the capture frame
    cv::Mat lane_area;        // HxW uint8, 0/1
    cv::Mat line_union;       // HxW uint8, 0/1
    LaneGeometry geometry;
    float steering{0.0f};
};

struct Detection {
    int class_id{-1};
    std::string label;
    cv::Rect2f bbox_img;      // source-frame px
    float score{0.0f};
    int track_id{-1};

    // Monocular distance estimate — mirrors the fields populated by
    // src/features/object_localizer_pose.py::estimate_road_object_pose.
    // pose_valid means an estimate ran successfully; distance_display_valid
    // gates HUD/label rendering.
    //   z_m       : raw camera-relative forward distance (diagnostic)
    //   distance_m: raw camera-relative slant range hypot(x_m, z_m)
    //   display_distance_m: bumper-relative slant range that the operator
    //     UI consumes. This subtracts the camera→front-bumper mount
    //     offset (DistanceCameraConfig::camera_to_front_bumper_m) from
    //     the forward component only; lateral offset (x_m) and bearing
    //     are unchanged. Clamped to >= 0.
    bool  pose_valid{false};
    bool  distance_display_valid{false};
    float distance_m{-1.0f};
    float display_distance_m{-1.0f};
    float z_m{0.0f};               // a.k.a. ground_distance_m
    float x_m{0.0f};
    float bearing_deg{0.0f};
    float distance_confidence{0.0f};

    // Relative speed of the FUSED front-object only (display/log only —
    // never feeds motor/RC). Populated by FrontSpeedEstimator after two
    // in-corridor distance samples with a sane dt. Sign convention:
    //   relative_speed_*  = d(distance)/dt  — positive when the front
    //                       vehicle is moving away from ego
    //   closing_speed_kmh = -relative_speed_kmh — positive when closing
    // relative_speed_valid is left false on every other detection, even
    // when that detection's bbox matches the front_object copy: only the
    // optional<Detection> front_object carries the estimate.
    bool  relative_speed_valid{false};
    float relative_speed_mps{0.0f};
    float relative_speed_kmh{0.0f};
    float closing_speed_kmh{0.0f};
    // Estimated absolute speed of the fused front vehicle. Valid only when
    // the ego speed source (GPS) is valid:
    //   front_speed_kmh = ego_speed_kmh + relative_speed_kmh
    // Display/log only; never feeds motor/RC.
    bool  front_speed_valid{false};
    float front_speed_kmh{0.0f};

    // Time gap for the fused front object (seconds), derived by
    // FrontSpeedEstimator as distance / signed_relative_rate in BOTH
    // directions. SIGNED: ttc_s > 0 = closing (seconds to collision);
    // ttc_s < 0 = opening (gap growing, no collision). Left invalid only when
    // the |rate| is below the noise floor (kTtcMinClosingMps, ~stationary) or
    // distance is unavailable. Display / log / HUD only — NEVER feeds motor/RC
    // (the motor is distance-only; see FollowController).
    bool  ttc_valid{false};
    float ttc_s{0.0f};
};

struct ObjectPacket {
    FrameId frame_id{0};
    TimePoint timestamp{};
    std::vector<Detection> detections;
    std::optional<Detection> front_object;  // closest in-corridor detection
    // Speed/TTC source consumed by FrontSpeedEstimator — NEVER feeds motor /
    // warning zone. Fusion assigns it the SAME selection as front_object
    // (the separate looser-gate pick was removed because it ran the
    // identical strict gate); kept as its own field so the estimator's
    // contract is explicit.
    std::optional<Detection> speed_object;
};

// Motor policy is keyed off this action (see runtime render_loop). It maps
// from the front object's collision warning zone:
//   Red    -> Stop    -> throttle 0
//   Orange -> Caution -> reduced throttle
//   Green  -> Monitor -> full commanded throttle
// No object in corridor -> Go -> full commanded throttle.
// Lane/perception validity must NOT influence the motor — only steering
// falls back to neutral when lane geometry is unavailable.
enum class BehaviorAction { Go, Follow, Stop, Observe, Caution, Monitor };

inline const char* to_string(BehaviorAction a) {
    switch (a) {
        case BehaviorAction::Go:      return "go";
        case BehaviorAction::Follow:  return "follow";
        case BehaviorAction::Stop:    return "stop";
        case BehaviorAction::Observe: return "observe";
        case BehaviorAction::Caution: return "caution";
        case BehaviorAction::Monitor: return "monitor";
    }
    return "?";
}

struct ControlCommand {
    float steering{0.0f};
    float throttle{0.0f};
    bool rc_enabled{false};
    bool safe{true};
};

// Display-only GPS readout copied from GpsReader::latest(). Surfaced on the
// HUD; never feeds steering, throttle, or behavior. Absent (nullopt on the
// packet) when --gps is not enabled.
struct GpsHud {
    bool        valid{false};      // fresh fix within stale_timeout
    double      speed_kmh{0.0};
    int         satellites{0};
    double      age_s{0.0};
    std::string source{"no_fix"};  // rmc / gga / stale / no_fix / ...
};

// Speed-limit sign ceiling the motor is currently applying (display-only;
// populated by FollowController every frame). The UI reads this instead of
// recomputing, so the panel can never disagree with the throttle cap the motor
// uses. ``source``: Default (no sign ever seen), Detected (a sign is in this
// frame), Held (a sign was seen and is held after leaving the frame).
struct SpeedLimitHud {
    enum class Source { Default, Detected, Held };
    int    kmh{60};                  // 60 default; 50/60/70/80 for a sign
    // Throttle ceiling (== the applied sign cap). Default matches
    // FollowConfig::default_sign_cap (0.8 == the speed_limit_60 cap) so a
    // packet rendered before FollowController stamps it cannot disagree.
    float  cap{0.8f};
    Source source{Source::Default};
};

struct BehaviorPacket {
    FrameId frame_id{0};
    TimePoint timestamp{};
    cv::Mat frame;                       // shallow ref
    std::optional<LanePacket> lane;
    std::optional<ObjectPacket> objects;
    BehaviorAction behavior_action{BehaviorAction::Go};
    std::string behavior_reason;
    ControlCommand control;
    std::optional<GpsHud> gps;           // display-only; set when --gps on
    SpeedLimitHud speed_limit;           // display-only; set by FollowController
};

}  // namespace adas
