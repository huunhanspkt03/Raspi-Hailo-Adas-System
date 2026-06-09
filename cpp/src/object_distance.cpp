#include "adas/object_distance.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "adas/object_distance_smooth.hpp"

namespace adas {

namespace {

struct VehiclePrior {
    const char* label;
    float height_m;
    float min_width_m;  // plausible real-world width range, used to gate the
    float max_width_m;  // size-height cue (implausible aspect -> low conf)
};

// Mirrors _REF_PRIORS_M in features/object_localizer_priors.py for the
// four road-vehicle labels (height + width range). EXCEPTION: the motorcycle
// height is raised here to include the upright rider. The size-height cue
// (z = prior_h*fy/bbox_h) otherwise reads ~30% low because 0.9 m is the
// bare-bike height, while a moving bike+rider bbox is ~1.2-1.3 m tall. This
// is the dominant cue beyond ~15 m where the ground cue is dropped. Python
// prior is unchanged (audit if the two paths are ever reconciled).
constexpr VehiclePrior kVehiclePriors[] = {
    {"bus",        3.2f, 2.2f, 2.8f},
    {"car",        1.5f, 1.55f, 2.1f},
    {"motorcycle", 1.2f, 0.45f, 1.15f},   // +rider (was 0.9 = bare bike)
    {"truck",      3.4f, 2.0f, 2.8f},
};

constexpr float kMinDepthM = 0.3f;
constexpr float kMaxDepthM = 80.0f;
constexpr float kMinConfidence = 0.05f;
constexpr int   kClipMarginPx = 3;
constexpr float kGroundMinDownAngleRad = 1.0f * 3.14159265358979323846f / 180.0f;
constexpr float kGroundMaxDepthM = 15.0f;
constexpr float kGroundConfidence = 0.80f;
// Ground cue stays fully trusted out to this forward depth; beyond it the foot
// sits near the horizon where a 1 px error is several metres, so its weight
// ramps to zero by kGroundMaxDepthM.
constexpr float kGroundReliableMaxM = 8.0f;

const VehiclePrior* find_prior(const std::string& label) {
    for (const auto& p : kVehiclePriors) {
        if (label == p.label) return &p;
    }
    return nullptr;
}

float select_foot_x(float x1, float x2, float cx) {
    // Mirrors object_localizer_depth.select_foot_u: closest bbox-bottom
    // point to the camera forward axis.
    if (x1 <= cx && cx <= x2) return cx;
    if (x2 < cx) return x2;
    return x1;
}

float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Parse an environment variable as a float; returns NaN when unset, empty, or
// non-numeric so the caller can leave the compiled default untouched.
float read_env_float(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') {
        return std::numeric_limits<float>::quiet_NaN();
    }
    char* end = nullptr;
    const float parsed = std::strtof(v, &end);
    if (end == v || !std::isfinite(parsed)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return parsed;
}

// Bumper/edge-relative slant range for the operator UI.
//   * Forward: subtract the longitudinal camera->bumper mount offset from z.
//   * Lateral: the camera is centered, so the gap to the nearest vehicle
//     EDGE is max(0, |x_m| - half_width), not the full camera-center
//     offset |x_m|. An object straddling the body (|x_m| <= half_width)
//     contributes no lateral term, so a centered object's display equals
//     its bumper-relative forward distance (preserves the on-axis case).
// Both clamps are independent: vehicle_width_m is lateral only,
// camera_to_front_bumper_m is longitudinal only.
float edge_relative_display(float x_m, float z, const DistanceCameraConfig& cfg) {
    const float half_w = std::max(0.0f, 0.5f * cfg.vehicle_width_m);
    const float lateral_gap = std::max(0.0f, std::fabs(x_m) - half_w);
    const float z_bumper = std::max(0.0f, z - cfg.camera_to_front_bumper_m);
    const float raw = std::sqrt(lateral_gap * lateral_gap + z_bumper * z_bumper);
    return std::max(
        0.0f,
        raw * std::max(0.01f, cfg.display_distance_scale) +
            cfg.display_distance_offset_m);
}

// Calibrated ground-plane foot depth. Forward distance to the point where
// the bbox bottom edge meets the road, from the flat-ground pinhole model:
//   ray_down = atan2(foot_y - cy, fy) + pitch_down
//   z        = camera_height / tan(ray_down)
// This is INDEPENDENT of the object's real-world height, so unlike the
// size-height estimate it scales correctly with distance — a far vehicle
// whose bbox is a few px too tall does not collapse the estimate. Valid
// only when the foot sits below the horizon by a usable margin
// (kGroundMinDownAngleRad) and the recovered depth is within the range
// where pitch/foot-pixel noise stays bounded (<= kGroundMaxDepthM). The
// caller must additionally reject clipped feet (bbox bottom at the frame
// edge / occluded by the hood), where foot_y is meaningless.
// pitch_down_rad is the effective downward tilt of the optical axis (the
// caller supplies either the configured pitch or the value derived from the
// single-point calibration); ray_down_deg/out_z are echoed for the debug
// trace. ray_down is reported via *out_ray_down_rad even on a soft reject so
// the trace can show why a foot was discarded.
bool ground_plane_z(float foot_y, float fy, float cy, float pitch_down_rad,
                    float camera_height_m, float* out_z,
                    float* out_ray_down_rad) {
    if (out_z == nullptr || camera_height_m <= 0.0f) return false;
    const float ray_down_rad =
        std::atan2(foot_y - cy, std::max(fy, 1e-3f)) + pitch_down_rad;
    if (out_ray_down_rad != nullptr) *out_ray_down_rad = ray_down_rad;
    if (!std::isfinite(ray_down_rad) || ray_down_rad <= kGroundMinDownAngleRad) {
        return false;
    }
    const float z = camera_height_m / std::tan(ray_down_rad);
    if (!std::isfinite(z) || z <= kMinDepthM || z > kGroundMaxDepthM) {
        return false;
    }
    *out_z = z;
    return true;
}

// Effective optical-axis downward tilt (radians) for the ground-plane model.
// Uses the single-point calibration when configured, else the static pitch.
float effective_pitch_down_rad(const DistanceCameraConfig& cfg,
                               float fy, float cy) {
    constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
    if (cfg.calib_z_m > 0.0f && cfg.calib_foot_y >= 0.0f &&
        cfg.camera_height_m > 0.0f) {
        const float theta = std::atan2(cfg.camera_height_m, cfg.calib_z_m);
        const float delta =
            std::atan2(cfg.calib_foot_y - cy, std::max(fy, 1e-3f));
        return theta - delta;  // may be < 0 if the calib foot is above horizon
    }
    return std::max(0.0f, -cfg.pitch_deg * kDegToRad);
}

// Fused forward-depth estimate + diagnostics for one detection.
struct DepthResult {
    float z;
    float confidence;
    const char* source;
    float z_height;
    float z_ground;       // -1 when the ground cue was unavailable
    float conf_height;
    float conf_ground;
    float ray_down_rad;   // 0 when the ground cue was not attempted
};

// Multi-cue forward-depth fusion. Two complementary monocular cues plus a
// near-field guardrail, each with a physically-motivated confidence; fused in
// inverse-depth space so neither cue's failure mode dominates:
//   * size-height  (z = prior_h*fy/bbox_h): fails for partial/angled/rider
//     boxes. Down-weighted when clipped, for motorcycles, and when the implied
//     real-world width is outside the class range (aspect guardrail).
//   * ground-plane (z from bbox foot): fails when the bbox bottom is not the
//     true tire contact (hood occlusion / box above the wheel). Disabled when
//     clipped; down-weighted for far feet near the horizon.
// On large cue disagreement the fused confidence is reduced so the behavior
// stop gate stays cautious. Returns false when neither cue is usable.
bool fuse_depth_cues(const VehiclePrior& prior, const std::string& label,
                     float bbox_w, float bbox_h, float fx, float fy, float cy,
                     float foot_y, bool clipped, float bottom_ratio,
                     float pitch_down_rad, float camera_height_m,
                     DepthResult* out) {
    const float z_height = prior.height_m * fy / std::max(bbox_h, 1e-3f);
    float conf_height = clipped ? 0.35f : 0.55f;
    if (label == "motorcycle") conf_height *= 0.85f;
    if (std::isfinite(z_height) && z_height > 0.0f) {
        const float implied_w = bbox_w * z_height / std::max(fx, 1e-3f);
        if (implied_w < prior.min_width_m || implied_w > prior.max_width_m) {
            conf_height *= 0.5f;  // bbox shape inconsistent with the class
        }
    } else {
        conf_height = 0.0f;
    }

    float z_ground = -1.0f;
    float ray_down_rad = 0.0f;
    float conf_ground = 0.0f;
    if (!clipped &&
        ground_plane_z(foot_y, fy, cy, pitch_down_rad, camera_height_m,
                       &z_ground, &ray_down_rad)) {
        conf_ground = kGroundConfidence;
        if (z_ground > kGroundReliableMaxM) {
            conf_ground *= clampf(
                (kGroundMaxDepthM - z_ground) /
                    (kGroundMaxDepthM - kGroundReliableMaxM), 0.0f, 1.0f);
        }
    }

    const bool h_ok =
        conf_height > 0.0f && std::isfinite(z_height) && z_height > 0.0f;
    const bool g_ok =
        conf_ground > 0.0f && std::isfinite(z_ground) && z_ground > 0.0f;

    float z;
    float confidence;
    const char* source;
    if (h_ok && g_ok) {
        const float inv = (conf_height / z_height + conf_ground / z_ground) /
                          (conf_height + conf_ground);
        z = 1.0f / inv;
        const float disagree =
            std::fabs(z_height - z_ground) / std::max(z_height, z_ground);
        confidence = std::max(conf_height, conf_ground) *
                     clampf(1.0f - disagree, 0.4f, 1.0f);
        source = "fused";
    } else if (g_ok) {
        z = z_ground;
        confidence = conf_ground;
        source = "ground";
    } else if (h_ok) {
        z = z_height;
        confidence = conf_height;
        source = "size";
    } else {
        return false;
    }

    if (bottom_ratio >= 0.78f) {
        const float t =
            clampf((bottom_ratio - 0.78f) / (0.95f - 0.78f), 0.0f, 1.0f);
        const float bottom_cap =
            std::max(kMinDepthM, 8.0f * (1.0f - t) + 0.6f * t);
        if (bottom_cap < z) {
            z = bottom_cap;
            confidence = std::max(confidence, 0.70f);
            source = "bottom_cap";
        }
    }

    out->z = clampf(z, kMinDepthM, kMaxDepthM);
    out->confidence = clampf(confidence, kMinConfidence, 0.95f);
    out->source = source;
    out->z_height = z_height;
    out->z_ground = g_ok ? z_ground : -1.0f;
    out->conf_height = conf_height;
    out->conf_ground = conf_ground;
    out->ray_down_rad = ray_down_rad;
    return true;
}

// Per-track temporal pose state, persisted across frames. Single-threaded
// (object worker thread only); cleared by reset_distance_smoother().
PoseSmoother g_pose_smoother;

}  // namespace

void apply_distance_calibration_env(DistanceCameraConfig& cfg) {
    // Parse once and cache; thread-safe static init (object_loop is the only
    // caller thread). Each field is overridden only when its var is set.
    static const float env_foot_y = read_env_float("ADAS_CALIB_FOOT_Y");
    static const float env_z_m = read_env_float("ADAS_CALIB_Z_M");
    static const float env_bumper_m = read_env_float("ADAS_CAMERA_TO_BUMPER_M");
    static const float env_scale = read_env_float("ADAS_DISTANCE_SCALE");
    static const float env_offset_m = read_env_float("ADAS_DISTANCE_OFFSET_M");
    if (std::isfinite(env_foot_y)) cfg.calib_foot_y = env_foot_y;
    if (std::isfinite(env_z_m)) cfg.calib_z_m = env_z_m;
    if (std::isfinite(env_bumper_m)) {
        cfg.camera_to_front_bumper_m = std::max(0.0f, env_bumper_m);
    }
    if (std::isfinite(env_scale)) {
        cfg.display_distance_scale = std::max(0.01f, env_scale);
    }
    if (std::isfinite(env_offset_m)) {
        cfg.display_distance_offset_m = env_offset_m;
    }
}

float ground_plane_row_for_z(float z_cam, int frame_w, int frame_h,
                             const DistanceCameraConfig& cfg) {
    if (frame_w <= 1 || frame_h <= 1 || z_cam <= 0.0f ||
        cfg.camera_height_m <= 0.0f) {
        return -1.0f;
    }
    constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
    const float half_hfov_rad = std::max(cfg.hfov_deg * 0.5f * kDegToRad, 1e-3f);
    const float fx = (0.5f * static_cast<float>(frame_w)) / std::tan(half_hfov_rad);
    float fy;
    if (cfg.vfov_deg > 0.0f) {
        const float half_vfov_rad =
            std::max(cfg.vfov_deg * 0.5f * kDegToRad, 1e-3f);
        fy = (0.5f * static_cast<float>(frame_h)) / std::tan(half_vfov_rad);
    } else {
        fy = fx;
    }
    const float cy = 0.5f * static_cast<float>(frame_h - 1);
    const float pitch_down = effective_pitch_down_rad(cfg, fy, cy);
    // ray_down = atan(H / z); foot_y - cy = fy * tan(ray_down - pitch_down).
    const float ray_down = std::atan2(cfg.camera_height_m, z_cam);
    const float row = cy + fy * std::tan(ray_down - pitch_down);
    return std::isfinite(row) ? row : -1.0f;
}

void annotate_distances(std::vector<Detection>& dets,
                        int frame_w, int frame_h,
                        const DistanceCameraConfig& cfg,
                        std::int64_t frame_id) {
    if (frame_w <= 1 || frame_h <= 1) return;

    // Per-detection distance trace, off unless ADAS_DEBUG_DISTANCE is set.
    static const bool kDebugDistance =
        std::getenv("ADAS_DEBUG_DISTANCE") != nullptr;
    if (kDebugDistance) {
        // One-time banner: report the active ground-plane calibration so the
        // trace is self-describing (env override vs compiled pitch_deg).
        static bool calib_banner = false;
        if (!calib_banner) {
            calib_banner = true;
            if (cfg.calib_z_m > 0.0f && cfg.calib_foot_y >= 0.0f) {
                std::fprintf(stderr,
                    "[dist] calibration ACTIVE: foot_y=%.1f z_m=%.2f "
                    "bumper=%.2f scale=%.3f offset=%.2f "
                    "(pitch_deg overridden)\n",
                    cfg.calib_foot_y, cfg.calib_z_m,
                    cfg.camera_to_front_bumper_m,
                    cfg.display_distance_scale,
                    cfg.display_distance_offset_m);
            } else {
                std::fprintf(stderr,
                    "[dist] calibration DISABLED: using pitch_deg=%.2f "
                    "bumper=%.2f scale=%.3f offset=%.2f\n",
                    cfg.pitch_deg, cfg.camera_to_front_bumper_m,
                    cfg.display_distance_scale,
                    cfg.display_distance_offset_m);
            }
        }
    }

    constexpr float kRadToDeg = 180.0f / 3.14159265358979323846f;
    constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
    const float half_hfov_rad = std::max(cfg.hfov_deg * 0.5f * kDegToRad, 1e-3f);
    const float fx = (0.5f * static_cast<float>(frame_w)) / std::tan(half_hfov_rad);
    float fy;
    if (cfg.vfov_deg > 0.0f) {
        const float half_vfov_rad =
            std::max(cfg.vfov_deg * 0.5f * kDegToRad, 1e-3f);
        fy = (0.5f * static_cast<float>(frame_h)) / std::tan(half_vfov_rad);
    } else {
        fy = fx;
    }
    const float cx = 0.5f * static_cast<float>(frame_w - 1);
    const float cy = 0.5f * static_cast<float>(frame_h - 1);
    // Effective optical-axis tilt: the single-point calibration when set,
    // otherwise the static pitch. Computed once per frame (depends only on
    // cy/fy, which are constant across detections).
    const float pitch_down_eff = effective_pitch_down_rad(cfg, fy, cy);

    std::vector<int> seen_ids;
    seen_ids.reserve(dets.size());

    for (Detection& d : dets) {
        d.pose_valid = false;
        d.distance_display_valid = false;
        d.distance_m = -1.0f;
        d.display_distance_m = -1.0f;
        d.z_m = 0.0f;
        d.x_m = 0.0f;
        d.bearing_deg = 0.0f;
        d.distance_confidence = 0.0f;

        const VehiclePrior* prior = find_prior(d.label);
        if (prior == nullptr) continue;
        const float bbox_w = d.bbox_img.width;
        const float bbox_h = d.bbox_img.height;
        if (!(bbox_w > 1.0f) || !(bbox_h > 1.0f)) continue;

        const float x1 = d.bbox_img.x;
        const float x2 = x1 + bbox_w;
        const float y2 = d.bbox_img.y + bbox_h;
        const bool clipped =
            y2 >= static_cast<float>(frame_h - kClipMarginPx);

        const float foot_x = select_foot_x(x1, x2, cx);
        const float bottom_ratio =
            y2 / static_cast<float>(std::max(frame_h, 1));

        // Multi-cue forward-depth fusion (size-height + ground-plane foot +
        // near-field cap), each confidence-weighted; see fuse_depth_cues.
        DepthResult dr;
        if (!fuse_depth_cues(*prior, d.label, bbox_w, bbox_h, fx, fy, cy, y2,
                             clipped, bottom_ratio, pitch_down_eff,
                             cfg.camera_height_m, &dr)) {
            continue;
        }

        float z = dr.z;
        // x_m: lateral offset of the foot column at this depth (camera optical
        // axis). z_m/x_m/bearing stay camera-relative; only display maps to
        // the vehicle edge via edge_relative_display.
        float x_m = ((foot_x - cx) / std::max(fx, 1e-3f)) * z;
        float confidence = dr.confidence;
        // Per-track temporal smoothing — confidence-adaptive EMA with a depth
        // step limiter (object_localizer_smooth.smooth_pose). No-op until the
        // tracker assigns a track_id, so it never harms the untracked case.
        g_pose_smoother.smooth(d.track_id, &z, &x_m, &confidence);
        if (d.track_id >= 0) seen_ids.push_back(d.track_id);

        const float dist = std::sqrt(x_m * x_m + z * z);
        const float bearing_deg =
            std::atan2(x_m, z) * (180.0f / 3.14159265358979323846f);
        const float display_dist = edge_relative_display(x_m, z, cfg);

        if (kDebugDistance) {
            std::fprintf(stderr,
                "[dist] frame=%lld %s tid=%d foot_y=%.1f cy=%.1f fy=%.1f "
                "pitch_eff_deg=%.2f ray_down_deg=%.2f z_ground=%.2f(c=%.2f) "
                "z_height=%.2f(c=%.2f) z=%.2f(%s) x_m=%.2f display=%.2f "
                "conf=%.2f%s\n",
                static_cast<long long>(frame_id), d.label.c_str(), d.track_id,
                y2, cy, fy, pitch_down_eff * kRadToDeg,
                dr.ray_down_rad * kRadToDeg, dr.z_ground, dr.conf_ground,
                dr.z_height, dr.conf_height, z, dr.source, x_m, display_dist,
                confidence, clipped ? " CLIPPED" : "");
        }

        d.pose_valid = true;
        d.distance_display_valid = true;
        d.z_m = z;
        d.x_m = x_m;
        d.distance_m = dist;
        d.display_distance_m = display_dist;
        d.bearing_deg = bearing_deg;
        d.distance_confidence = confidence;
    }

    // Drop temporal state for tracks not seen this frame (mirrors
    // localize()'s prune-to-seen-tracks); keeps the smoother map bounded.
    g_pose_smoother.prune(seen_ids);
}

void reset_distance_smoother() { g_pose_smoother.reset(); }

}  // namespace adas
