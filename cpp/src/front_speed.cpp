#include "adas/front_speed.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>

namespace adas {

namespace {

constexpr float kEmaAlpha     = 0.30f;
constexpr float kMinDt        = 0.030f;   // 30 ms — below this dt is unreliable
constexpr float kMaxDt        = 1.000f;   // 1 s — too stale, treat as a reset
constexpr float kMaxRateMps   = 30.0f;    // ~108 km/h closure rate cap
constexpr float kLogChangeKmh = 5.0f;     // change-only logging threshold
constexpr float kMaxFrontKmh  = 180.0f;   // reject impossible display values
constexpr float kTtcMinClosingMps = 0.30f;  // ~1 km/h — |rate| below this is noise
constexpr float kTtcMaxS          = 60.0f;  // clamp |displayed TTC| for sanity

// Choose the most stable distance source. Mirrors the priority used by
// the warning HUD label: display_distance_m (bumper-relative) when valid,
// otherwise the raw forward z_m. The bumper-offset estimate matches the
// number the operator reads, so the speed they see derives from the same
// quantity they see on the panel.
bool resolve_distance(const Detection& d, float* out) {
    if (d.distance_display_valid && d.display_distance_m >= 0.0f) {
        *out = d.display_distance_m;
        return true;
    }
    if (d.pose_valid && d.z_m > 0.0f) {
        *out = d.z_m;
        return true;
    }
    return false;
}

void stamp_speed(Detection& fo, float ema_mps, float cur_distance,
                 bool ego_valid, float ego_kmh, Detection* mirror) {
    fo.relative_speed_valid = true;
    fo.relative_speed_mps   = -ema_mps;            // positive = opening
    fo.relative_speed_kmh   = -ema_mps * 3.6f;
    fo.closing_speed_kmh    =  ema_mps * 3.6f;     // positive = closing
    fo.front_speed_valid    = false;
    fo.front_speed_kmh      = 0.0f;
    // Time gap = distance over the SIGNED relative rate, computed in BOTH
    // directions (display/log only — never feeds the motor). Sign convention
    // matches closing_speed_kmh:
    //   ema_mps > 0 (closing) -> ttc_s > 0  = seconds to collision.
    //   ema_mps < 0 (opening) -> ttc_s < 0  = the gap is growing (no collision;
    //                                         the negative sign marks "opening").
    // Gated on the rate MAGNITUDE so a near-stationary lead (|rate| below the
    // noise floor) stays invalid — there the quotient explodes. Same resolved
    // distance the rate came from, clamped to a sane band.
    fo.ttc_valid = false;
    fo.ttc_s     = 0.0f;
    if (std::fabs(ema_mps) > kTtcMinClosingMps && cur_distance >= 0.0f) {
        const float ttc = cur_distance / ema_mps;
        if (std::isfinite(ttc)) {
            fo.ttc_valid = true;
            fo.ttc_s     = std::clamp(ttc, -kTtcMaxS, kTtcMaxS);
        }
    }
    if (ego_valid) {
        const float front_kmh = ego_kmh + fo.relative_speed_kmh;
        if (std::isfinite(front_kmh) &&
            front_kmh >= 0.0f && front_kmh <= kMaxFrontKmh) {
            fo.front_speed_valid = true;
            fo.front_speed_kmh = front_kmh;
        }
    }
    // Mirror onto the corridor front_object when it is the SAME vehicle, so the
    // HUD (which reads front_object) still shows the speed estimated on the
    // lane-relevant speed_object. Different vehicle -> leave front_object alone.
    if (mirror != nullptr && mirror != &fo &&
        fo.track_id >= 0 && mirror->track_id == fo.track_id) {
        mirror->relative_speed_valid = fo.relative_speed_valid;
        mirror->relative_speed_mps   = fo.relative_speed_mps;
        mirror->relative_speed_kmh   = fo.relative_speed_kmh;
        mirror->closing_speed_kmh    = fo.closing_speed_kmh;
        mirror->front_speed_valid    = fo.front_speed_valid;
        mirror->front_speed_kmh      = fo.front_speed_kmh;
        mirror->ttc_valid            = fo.ttc_valid;
        mirror->ttc_s                = fo.ttc_s;
    }
}

}  // namespace

void FrontSpeedEstimator::reset() {
    if (logged_active_) {
        std::cerr << "[front_speed] cleared\n";
    }
    has_prev_                = false;
    prev_object_frame_id_    = 0;
    prev_distance_m_         = 0.0f;
    prev_track_id_           = -1;
    ema_ready_               = false;
    ema_mps_                 = 0.0f;
    logged_active_           = false;
    last_logged_closing_kmh_ = 0.0f;
}

void FrontSpeedEstimator::update(BehaviorPacket& bp) {
    // Speed/TTC tracks the lane-relevant vehicle (speed_object) when present,
    // else the corridor front_object. mirror copies the result onto
    // front_object when it is the same vehicle so the HUD still shows speed.
    if (!bp.objects.has_value()) { reset(); return; }
    std::optional<Detection>& speed_opt = bp.objects->speed_object;
    std::optional<Detection>& front_opt = bp.objects->front_object;
    Detection* target = speed_opt.has_value() ? &(*speed_opt)
                      : front_opt.has_value() ? &(*front_opt)
                      : nullptr;
    if (target == nullptr) { reset(); return; }
    if (!bp.lane.has_value() || !bp.lane->geometry.valid) {
        // Without a valid corridor the selection is unsound — drop state so we
        // don't compare across lane-loss gaps.
        reset();
        return;
    }
    Detection& fo = *target;
    Detection* mirror = (speed_opt.has_value() && front_opt.has_value())
                        ? &(*front_opt) : nullptr;
    const bool ego_valid = bp.gps.has_value() && bp.gps->valid;
    const float ego_kmh = ego_valid
        ? static_cast<float>(bp.gps->speed_kmh)
        : 0.0f;
    float cur_distance = 0.0f;
    if (!resolve_distance(fo, &cur_distance)) {
        reset();
        return;
    }
    const FrameId cur_obj_fid = bp.objects->frame_id;
    // dt must be measured between OBJECT packet timestamps: bp.timestamp is
    // the render/lane frame's capture time and can lag the object frame by
    // several frames in the alternate schedule.
    const TimePoint cur_obj_time = bp.objects->timestamp;
    const int cur_tid = fo.track_id;

    if (!has_prev_) {
        prev_object_frame_id_ = cur_obj_fid;
        prev_time_            = cur_obj_time;
        prev_distance_m_      = cur_distance;
        prev_track_id_        = cur_tid;
        has_prev_             = true;
        return;
    }

    // Lead identity changed (selection switched to a different vehicle, or
    // the track was lost and re-minted). Differentiating distances of two
    // DIFFERENT vehicles produces a bogus closing rate / TTC that can pass
    // the kMaxRateMps guard, so restart estimation from this sample and do
    // NOT stamp the stale EMA — it belonged to the previous vehicle.
    //
    // This MUST run before the cached-frame check below: one cached object
    // packet is paired with several lane frames, and because lane geometry
    // updates every lane frame, pick_front_object can select a DIFFERENT
    // track from the same cached packet between consecutive lane frames.
    // Checking identity first stops the cached path from stamping the
    // previous vehicle's EMA onto the new one.
    if (cur_tid < 0 || cur_tid != prev_track_id_) {
        prev_object_frame_id_ = cur_obj_fid;
        prev_time_            = cur_obj_time;
        prev_distance_m_      = cur_distance;
        prev_track_id_        = cur_tid;
        ema_ready_            = false;
        ema_mps_              = 0.0f;
        return;
    }

    // Same lead, cached ObjectPacket reused across render packets. Same
    // measurement = no new info, but keep the last EMA visible so the HUD
    // doesn't strobe.
    if (cur_obj_fid == prev_object_frame_id_) {
        if (ema_ready_) stamp_speed(fo, ema_mps_, cur_distance, ego_valid, ego_kmh, mirror);
        return;
    }

    const double dt = std::chrono::duration<double>(
                          cur_obj_time - prev_time_).count();
    if (dt < static_cast<double>(kMinDt) ||
        dt > static_cast<double>(kMaxDt)) {
        // Sample stream paused or jumped. Re-anchor on the current sample
        // but keep the last EMA stamped for one packet so a single late
        // frame doesn't drop the HUD readout.
        if (ema_ready_) stamp_speed(fo, ema_mps_, cur_distance, ego_valid, ego_kmh, mirror);
        prev_object_frame_id_ = cur_obj_fid;
        prev_time_            = cur_obj_time;
        prev_distance_m_      = cur_distance;
        return;
    }

    const float raw_mps = (prev_distance_m_ - cur_distance) /
                          static_cast<float>(dt);
    if (std::fabs(raw_mps) > kMaxRateMps) {
        // Likely a distance-estimator glitch (bbox flicker or label
        // mismatch). Re-anchor without poisoning the EMA.
        prev_object_frame_id_ = cur_obj_fid;
        prev_time_            = cur_obj_time;
        prev_distance_m_      = cur_distance;
        return;
    }

    if (!ema_ready_) {
        ema_mps_   = raw_mps;
        ema_ready_ = true;
    } else {
        ema_mps_ = kEmaAlpha * raw_mps + (1.0f - kEmaAlpha) * ema_mps_;
    }
    stamp_speed(fo, ema_mps_, cur_distance, ego_valid, ego_kmh, mirror);

    prev_object_frame_id_ = cur_obj_fid;
    prev_time_            = cur_obj_time;
    prev_distance_m_      = cur_distance;

    // Concise change-only logging. Local stream so std::cerr's global
    // formatting flags stay untouched.
    const float closing_kmh = fo.closing_speed_kmh;
    if (!logged_active_ ||
        std::fabs(closing_kmh - last_logged_closing_kmh_) >= kLogChangeKmh) {
        std::ostringstream os;
        os.setf(std::ios::fixed);
        os.precision(1);
        os << "[front_speed] closing=" << closing_kmh << " km/h";
        if (fo.ttc_valid) os << " ttc=" << fo.ttc_s << " s";
        else              os << " ttc=-- s";
        if (fo.front_speed_valid) os << " front=" << fo.front_speed_kmh << " km/h";
        else                      os << " front=-- km/h";
        os << "  distance=" << cur_distance << " m  (display/log only)";
        std::cerr << os.str() << "\n";
        logged_active_           = true;
        last_logged_closing_kmh_ = closing_kmh;
    }
}

}  // namespace adas
