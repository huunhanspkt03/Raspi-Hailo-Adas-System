#pragma once

// Minimal monocular distance estimator for road-vehicle detections on the
// C++ runtime. Mirrors the road-object branch of:
//   src/features/object_localizer.py
//   src/features/object_localizer_pose.py
//   src/features/object_localizer_depth.py
//   src/features/object_localizer_priors.py
//
// Only the four vehicle labels are supported (bus / car / motorcycle /
// truck); temporal smoothing from the Python reference is intentionally
// not ported here.

#include <cstdint>
#include <vector>

#include "adas/types.hpp"

namespace adas {

// Camera defaults match the Pi Cam Module 3 Wide capture used for this
// project (rpicam-vid --mode 1536:864:10:P --width 1280 --height 864).
// Resolutions are *not* hard-coded — the focal length is derived from
// hfov_deg and the current frame_w at annotate time.
struct DistanceCameraConfig {
    float hfov_deg{97.0f};           // effective horizontal FOV after CSI crop
    float vfov_deg{-1.0f};           // <=0 means "use fx for fy"
    float camera_height_m{1.35f};    // used by off-axis ground-plane correction
    float pitch_deg{-1.5f};          // negative = camera pointed down
    // Longitudinal offset between the camera and the front bumper /
    // vehicle nose. The camera is mounted this far BEHIND the bumper,
    // so reported HUD/label distances should be smaller than the
    // camera-relative forward distance by this amount. Subtracted only
    // from the forward (z) component, not from the lateral component or
    // bearing_deg. Result is clamped to >= 0. This is a *longitudinal*
    // value only — it is intentionally independent of vehicle_width_m
    // below (the two must not be conflated even if numerically similar).
    float camera_to_front_bumper_m{0.20f};
    // Full vehicle width. The camera is centered, so half this value is
    // the lateral distance from the optical axis to either side of the
    // body. Display distance for off-axis objects measures the gap to
    // the nearest vehicle EDGE, i.e. lateral_gap = max(0, |x_m| - width/2),
    // not the full camera-center lateral offset |x_m|. A *lateral* value
    // only — independent of camera_to_front_bumper_m above.
    float vehicle_width_m{1.8f};
    // Optional display calibration after the bumper/edge-relative range is
    // computed. Use this to correct a consistent under/over estimate without
    // changing the geometric cues:
    //   corrected = raw_display * display_distance_scale + display_distance_offset_m
    float display_distance_scale{1.0f};
    float display_distance_offset_m{0.0f};
    // Optional single-point ground-plane calibration. When calib_z_m > 0 and
    // calib_foot_y >= 0, the effective downward pitch is DERIVED so that a
    // bbox foot at source-frame row calib_foot_y maps to camera-forward depth
    // calib_z_m, and pitch_deg is ignored for the ground-plane model:
    //   pitch_down_eff = atan(camera_height_m / calib_z_m)
    //                    - atan2(calib_foot_y - cy, fy)
    // This re-anchors the horizon row from one measured correspondence,
    // absorbing principal-point offset, bbox-foot bias (box bottom above the
    // true tire contact / partial hood occlusion) and pitch error into a
    // single calibrated angle. It calibrates the horizon (offset) only; the
    // depth scale still rides on fy and camera_height_m, so it is exact at
    // calib_z_m and most accurate near it. calib_foot_y is in SOURCE-frame
    // pixels (same space as the detection bbox). calib_z_m is the
    // CAMERA-forward distance to that foot point (bumper offset is subtracted
    // later for display): e.g. a motorcycle 2.0 m ahead of a nose that is
    // 1.8 m ahead of the camera -> calib_z_m = 3.8. <=0 disables.
    float calib_foot_y{-1.0f};
    float calib_z_m{-1.0f};
};

// Annotate every detection in-place with pose_valid / z_m / x_m /
// distance_m / bearing_deg / distance_confidence /
// distance_display_valid. Detections whose label has no prior, or whose
// bbox is degenerate, are left with pose_valid=false (default). Cheap to
// call per frame — no heap allocations.
// frame_id is used only for the optional per-detection distance trace
// (enabled by setting the ADAS_DEBUG_DISTANCE environment variable); pass -1
// when no id is available.
void annotate_distances(std::vector<Detection>& dets,
                        int frame_w, int frame_h,
                        const DistanceCameraConfig& cfg,
                        std::int64_t frame_id = -1);

// Override the ground-plane single-point calibration fields from the
// environment so the camera can be re-calibrated without rebuilding:
//   ADAS_CALIB_FOOT_Y          -> calib_foot_y   (source-frame pixels)
//   ADAS_CALIB_Z_M             -> calib_z_m      (camera-forward meters)
//   ADAS_CAMERA_TO_BUMPER_M    -> camera_to_front_bumper_m
//   ADAS_DISTANCE_SCALE        -> display_distance_scale
//   ADAS_DISTANCE_OFFSET_M     -> display_distance_offset_m
// Each var is independent and only overrides its field when set to a parseable
// number; unset/invalid leaves the compiled default (calibration disabled).
// The env values are parsed once on first call and cached, so this is cheap to
// call every frame. Header defaults stay disabled (-1) by design.
void apply_distance_calibration_env(DistanceCameraConfig& cfg);

// Inverse of the ground-plane foot model: the source-frame image ROW where a
// flat-ground point at camera-forward distance z_cam projects, using the same
// intrinsics and effective (calibrated) pitch as the depth estimator. Lets the
// overlay paint warning bands at metric distances so the colored corridor
// matches the motor/HUD decision. Returns < 0 on bad input. Note z_cam is
// CAMERA-forward: for a nose distance d add camera_to_front_bumper_m.
float ground_plane_row_for_z(float z_cam, int frame_w, int frame_h,
                             const DistanceCameraConfig& cfg);

// Clear per-track temporal pose-smoothing state (e.g. on a video/stream
// switch) so distances do not carry over a cut. Pair with
// reset_object_tracker() in object_detection.hpp.
void reset_distance_smoother();

}  // namespace adas
