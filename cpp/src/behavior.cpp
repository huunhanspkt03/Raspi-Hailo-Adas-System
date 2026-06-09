#include "adas/behavior.hpp"
#include "adas/object_detection.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>

// Lightweight milestone behavior policy. Python reference:
//   src/features/behavior_logic.py     — BehaviorSmoother, decide_action
//   src/features/behavior_runner.py    — run_behavior pipeline glue
//   src/features/control_runner.py     — RC actuator wrapper
//
// TODO(adas-cpp/behavior): port front-object distance/braking ladder,
// behavior smoother window, and release-frames hysteresis once
// object detection is wired through.

namespace adas {

namespace {

// Speed-limit sign throttle cap. Applied as an UPPER BOUND on the
// behavior throttle (std::min), so it can never raise speed and never
// overrides a STOP. The mapping is fixed in object_detection.cpp:
//   speed_limit_50 -> 0.60    speed_limit_60 -> 0.80
//   speed_limit_70 -> 0.80    speed_limit_80 -> 1.00
// The cap is an additive safety layer (sign-limited speed), not a
// replacement for STOP/CAUTION. It runs in front of SafetyGovernor so
// the smoother / fail-closed still owns the final word; RcActuator only
// applies its fixed [0,1] duty clamp downstream.
// Best speed-limit sign throttle cap among detections, or 1.0 (no cap). Mapping
// fixed in object_detection.cpp: 50->0.60 60->0.80 70->0.80 80->1.00. It can
// only ever LOWER throttle (applied via std::min) and never overrides a STOP.
// Reports the matched sign class / km-h for the reason string + logging.
float speed_sign_cap(const std::optional<ObjectPacket>& objects,
                     int* sign_class_out, int* sign_kmh_out) {
    if (sign_class_out) *sign_class_out = -1;
    if (sign_kmh_out)   *sign_kmh_out   = 0;
    if (!objects) return 1.0f;
    const Detection* best = nullptr;
    for (const auto& d : objects->detections) {
        if (!yolo26_is_traffic_sign(d.class_id)) continue;
        if (best == nullptr || d.score > best->score) best = &d;
    }
    if (best == nullptr) return 1.0f;
    if (sign_class_out) *sign_class_out = best->class_id;
    if (sign_kmh_out)
        *sign_kmh_out = static_cast<int>(yolo26_speed_limit_kmh(best->class_id));
    return yolo26_speed_limit_throttle_cap(best->class_id);
}

// Ego-speed-tiered caution distance (m): the distance at which the throttle
// reaches full cruise. Below it the throttle ramps down to 0 at the hard-stop
// margin. ego > ego_fast_kmh -> caution_far_m; otherwise (incl. no GPS fix,
// where ego speed is unknown) caution_near_m.
float caution_distance_for(const FollowConfig& cfg, bool ego_valid,
                           float ego_kmh) {
    if (ego_valid && ego_kmh > cfg.ego_fast_kmh) return cfg.caution_far_m;
    return cfg.caution_near_m;
}

}  // namespace

BehaviorState decide_behavior(const std::optional<LanePacket>& lane,
                              const std::optional<ObjectPacket>& objects) {
    // Lane validity must never gate the motor (see header) — the parameter
    // stays in the signature for parity with the Python decide_action shape.
    (void)lane;
    BehaviorState s;

    // PROVISIONAL only. When there IS an in-corridor front object, the
    // authoritative distance-only FollowController (render loop) overrides
    // action + throttle entirely, so the value here just needs to be a safe
    // "slow down" default. With NO front object this is the final base the
    // FollowController only caps by the speed-limit sign ceiling.
    if (objects && objects->front_object) {
        s.action = BehaviorAction::Caution;
        s.reason = "front_object";
        s.throttle = 0.5f;
        return s;
    }

    s.action = BehaviorAction::Go;
    s.reason = "no_front_object";
    s.throttle = 1.0f;
    return s;
}

void SafetyGovernor::apply(BehaviorPacket& bp, bool object_fresh) {
    const BehaviorAction raw = bp.behavior_action;

    // --- STOP hysteresis: engage immediately on a Stop action, then hold for
    // stop_release_frames further (non-stop) packets before releasing. The
    // hold is re-armed every Stop frame. ---
    bool stopping;
    if (raw == BehaviorAction::Stop) {
        stop_hold_ = cfg_.stop_release_frames;
        stopping = true;
    } else {
        stopping = stop_hold_ > 0;
        if (stop_hold_ > 0) --stop_hold_;
    }

    // --- CAUTION hysteresis (only evaluated when not stopping). ---
    bool cautioning = false;
    if (!stopping) {
        if (raw == BehaviorAction::Caution) {
            caution_hold_ = cfg_.caution_release_frames;
            cautioning = true;
        } else {
            cautioning = caution_hold_ > 0;
            if (caution_hold_ > 0) --caution_hold_;
        }
    }

    BehaviorAction eff = raw;
    std::string reason = bp.behavior_reason;
    float throttle = bp.control.throttle;
    bool safe = bp.control.safe;

    if (stopping) {
        eff = BehaviorAction::Stop;
        throttle = 0.0f;
        if (raw != BehaviorAction::Stop) reason = "stop_hold";
    } else if (cautioning) {
        eff = BehaviorAction::Caution;
        throttle = std::min(throttle, cfg_.caution_throttle);
        if (raw != BehaviorAction::Caution) reason = "caution_hold";
    }

    // --- Fail-closed supervision. ---
    // Sustained loss of lane geometry. A short dropout is tolerated by the
    // grace window so a single bad frame does not brake the car.
    //
    // VALID requires the strict lane gate + confidence only. Purple polylines
    // are deliberately NOT required: the parity tracker leaves them empty on
    // HELD frames (boundary briefly occluded — often by the very lead vehicle
    // we are following), and HELD is the tracker bridging a dropout, not a
    // lost lane. The fixed-guidance fallback still reads invalid here because
    // it carries confidence 0 and no strict gate.
    bool lane_valid = false;
    if (bp.lane.has_value()) {
        const LaneGeometry& g = bp.lane->geometry;
        const bool has_strict_gate =
            g.strict_lane_left.size() >= 2 &&
            g.strict_lane_right.size() >= 2;
        lane_valid = g.valid && has_strict_gate &&
                     g.confidence >= cfg_.min_lane_confidence;
    }
    auto close_front_object_during_lane_loss = [&]() -> bool {
        if (!bp.objects.has_value() || !bp.objects->front_object) return false;
        const Detection& d = *bp.objects->front_object;
        float dist = -1.0f;
        if (d.distance_display_valid && d.display_distance_m >= 0.0f) {
            dist = d.display_distance_m;
        } else if (d.pose_valid && d.z_m > 0.0f) {
            dist = d.z_m;
        }
        return dist >= 0.0f && dist <= cfg_.lane_loss_stop_distance_m;
    };
    lane_invalid_streak_ = lane_valid ? 0 : (lane_invalid_streak_ + 1);
    if (lane_invalid_streak_ > cfg_.lane_invalid_grace &&
        close_front_object_during_lane_loss()) {
        safe = false;
        reason = "lane_lost_near_object";
    }

    // Sustained absence of fresh object data.
    object_stale_streak_ = object_fresh ? 0 : (object_stale_streak_ + 1);
    if (object_stale_streak_ > cfg_.object_stale_grace) {
        safe = false;
        reason = "object_stale";
    }

    if (!safe) throttle = 0.0f;

    bp.behavior_action = eff;
    bp.behavior_reason = reason;
    bp.control.throttle = throttle;
    bp.control.safe = safe;

    // Concise logging: only on a change of (action, safe, reason).
    if (!logged_once_ || eff != last_action_ || safe != last_safe_ ||
        reason != last_reason_) {
        std::cerr << "[safety] action=" << to_string(eff)
                  << " safe=" << (safe ? 1 : 0)
                  << " throttle=" << throttle
                  << " reason=" << reason << "\n";
        logged_once_ = true;
        last_action_ = eff;
        last_safe_ = safe;
        last_reason_ = reason;
    }
}

// ---------------------------------------------------------------------------
// FollowController — front_object throttle.
// ---------------------------------------------------------------------------
void FollowController::reset() {
    has_prev_             = false;
    prev_throttle_        = 0.0f;
    logged_once_          = false;
    last_logged_throttle_ = 0.0f;
    held_sign_cap_   = cfg_.default_sign_cap;
    held_sign_kmh_   = cfg_.default_sign_kmh;
    ever_seen_sign_  = false;
    sign_detected_   = false;
}

// Speed-limit sign ceiling: adopted IMMEDIATELY on detection and HELD
// INDEFINITELY until a DIFFERENT sign is seen (it persists after the sign leaves
// the frame). Before any sign the 60 km/h default applies. A ceiling can only
// REDUCE throttle, so early adoption is safe. Updates held_sign_* + the source
// flags consumed by the UI panel, the reason string, and the [motor] log.
float FollowController::sign_ceiling(const std::optional<ObjectPacket>& objects) {
    int cls = -1, kmh = 0;
    const float seen = speed_sign_cap(objects, &cls, &kmh);  // cls < 0 -> no sign
    if (cls >= 0) {
        held_sign_cap_  = seen;        // adopt immediately
        held_sign_kmh_  = kmh;
        ever_seen_sign_ = true;
        sign_detected_  = true;
    } else {
        sign_detected_  = false;       // hold last sign (or default if never seen)
    }
    return held_sign_cap_;
}

// Asymmetric throttle slew (brake faster than accelerate) toward `target`,
// shared by the early-return paths so a ceiling change never steps the throttle.
float FollowController::slew(float target, const BehaviorPacket& bp) const {
    if (!has_prev_) return target;
    const double dt = std::chrono::duration<double>(bp.timestamp - prev_time_).count();
    if (!(dt > 0.0 && dt < 1.0)) return target;
    const float up = cfg_.slew_up_per_s   * static_cast<float>(dt);
    const float dn = cfg_.slew_down_per_s * static_cast<float>(dt);
    return std::clamp(target, prev_throttle_ - dn, prev_throttle_ + up);
}

void FollowController::update(BehaviorPacket& bp) {
    // Speed-limit sign ceiling = the motor's MAX throttle. Advanced EVERY frame
    // (incl. the early-return modes) so the hold timer stays consistent; it can
    // only REDUCE the follow/TTC/distance result, never raise it.
    const float sign_cap = sign_ceiling(bp.objects);
    // Expose the applied ceiling to the UI (display-only); the overlay never
    // recomputes it, so the panel always matches the motor's cap.
    bp.speed_limit.kmh    = held_sign_kmh_;
    bp.speed_limit.cap    = held_sign_cap_;
    bp.speed_limit.source = !ever_seen_sign_ ? SpeedLimitHud::Source::Default
                          : sign_detected_   ? SpeedLimitHud::Source::Detected
                                             : SpeedLimitHud::Source::Held;
    auto sign_note = [&](std::string& reason, float pre_cap) {
        if (sign_cap < pre_cap - 1e-4f) {           // ceiling actually bound
            std::ostringstream tag; tag.setf(std::ios::fixed); tag.precision(2);
            if (ever_seen_sign_)
                tag << "+cap(sign" << held_sign_kmh_ << "=" << held_sign_cap_ << ")";
            else
                tag << "+cap(default" << cfg_.default_sign_kmh << "=" << held_sign_cap_ << ")";
            reason += tag.str();
        }
    };

    // Detection-only forces Observe/throttle 0 upstream; lane-only leaves
    // decide_behavior's base. Either way the sign ceiling is the hard max.
    if (bp.behavior_action == BehaviorAction::Observe || !bp.objects.has_value()) {
        const float base = std::clamp(bp.control.throttle, 0.0f, 1.0f);
        bp.control.throttle = slew(std::min(base, sign_cap), bp);
        sign_note(bp.behavior_reason, base);
        prev_throttle_ = bp.control.throttle;
        prev_time_     = bp.timestamp;
        has_prev_      = true;
        return;
    }
    const std::optional<Detection>& front_opt = bp.objects->front_object;
    const std::optional<Detection>& speed_opt = bp.objects->speed_object;
    // Nothing in or near the lane -> decide_behavior's base, capped by the sign
    // ceiling (default 0.6 / 60 km/h when no sign is held).
    if (!front_opt.has_value() && !speed_opt.has_value()) {
        const float base = std::clamp(bp.control.throttle, 0.0f, 1.0f);
        bp.control.throttle = slew(std::min(base, sign_cap), bp);
        sign_note(bp.behavior_reason, base);
        prev_throttle_ = bp.control.throttle;
        prev_time_     = bp.timestamp;
        has_prev_      = true;
        return;
    }

    const bool  ego_valid  = bp.gps.has_value() && bp.gps->valid;
    const float ego_kmh    = ego_valid ? static_cast<float>(bp.gps->speed_kmh) : 0.0f;

    // Distance the UI shows: bumper-relative display preferred, else z_m.
    auto resolve_dist = [](const Detection& d) -> float {
        if (d.distance_display_valid && d.display_distance_m >= 0.0f)
            return d.display_distance_m;
        if (d.pose_valid && d.z_m > 0.0f) return d.z_m;
        return -1.0f;
    };

    // === Distance-only longitudinal control for the in-corridor front_object.
    //   < hard_stop_distance_m (3 m)        -> STOP, unconditional (lead moving
    //                                          or not; GPS-independent).
    //   [3 m, caution_dist)                 -> CAUTION, throttle ramps linearly
    //                                          0 (at 3 m) -> max cruise (at caution_dist).
    //   >= caution_dist                     -> MONITOR, full cruise.
    // caution_dist is ego-speed tiered: ego > 40 km/h -> 15 m, else (incl. no
    // GPS fix) -> 10 m. No TTC / closing-rate term — pure distance (per spec).
    const char*    mode      = "no_front";
    float          raw       = bp.control.throttle;   // decide_behavior provisional
    bool           hard_stop = false;
    float          caution_dist = 0.0f;
    float          front_dist = -1.0f;
    BehaviorAction action = bp.behavior_action;
    std::string    reason  = bp.behavior_reason;

    if (front_opt.has_value()) {
        const Detection& fo = *front_opt;
        const bool dv = fo.distance_display_valid && fo.display_distance_m >= 0.0f;
        front_dist = resolve_dist(fo);
        const bool have_dist = front_dist >= 0.0f;

        const bool hard_stop_zone =
            dv && fo.display_distance_m <= cfg_.hard_stop_distance_m;

        if (hard_stop_zone) {
            // STOP — absolute priority, GPS-independent, regardless of whether
            // the lead is moving.
            raw = 0.0f; hard_stop = true;
            action = BehaviorAction::Stop; reason = "hard_stop"; mode = "hard_stop";
        } else if (have_dist) {
            caution_dist = caution_distance_for(cfg_, ego_valid, ego_kmh);
            const float span =
                std::max(caution_dist - cfg_.hard_stop_distance_m, 0.1f);
            raw = cfg_.caution_max_throttle *
                  std::clamp((front_dist - cfg_.hard_stop_distance_m) / span,
                             0.0f, 1.0f);
            if (front_dist >= caution_dist) {
                action = BehaviorAction::Monitor; reason = "clear"; mode = "cruise";
            } else if (raw <= 0.01f) {
                action = BehaviorAction::Stop; reason = "caution_stop"; mode = "caution";
            } else {
                action = BehaviorAction::Caution; reason = "caution_follow"; mode = "caution";
            }
        } else {
            // In corridor but no usable distance estimate: conservative, never
            // a hard stop.
            raw = 0.5f; action = BehaviorAction::Caution;
            mode = "no_range"; reason = "front_object_unranged";
        }
    }

    // speed_object is a lane-relevant display/speed source only.  Do not let it
    // reduce motor throttle: adjacent-lane motorcycles/cars can sit close to the
    // corridor edge and otherwise make the car slow even when there is no strict
    // in-corridor front_object.

    // Speed-limit sign ceiling (final upper bound; never raises throttle). The
    // distance-ramp result `raw` can only be reduced below it.
    const float capped = std::min(raw, sign_cap);

    // Asymmetric PWM slew (brake faster than accelerate); hard stop bypasses it.
    float t_cmd = capped;
    if (!hard_stop && has_prev_) {
        const double dt = std::chrono::duration<double>(
                              bp.timestamp - prev_time_).count();
        if (dt > 0.0 && dt < 1.0) {
            const float up = cfg_.slew_up_per_s   * static_cast<float>(dt);
            const float dn = cfg_.slew_down_per_s * static_cast<float>(dt);
            t_cmd = std::clamp(capped, prev_throttle_ - dn, prev_throttle_ + up);
        }
    }
    // Following any in-/near-lane object never exceeds the max cruise throttle.
    t_cmd = std::clamp(t_cmd, 0.0f, cfg_.caution_max_throttle);

    sign_note(reason, raw);   // append +cap(...) to the reason when it bound `raw`

    bp.control.throttle = t_cmd;
    bp.behavior_action  = action;
    bp.behavior_reason  = reason;
    prev_throttle_ = t_cmd;
    prev_time_     = bp.timestamp;
    has_prev_      = true;

    // Change-only [motor] log. SafetyGovernor emits its own [safety] line when
    // it forces throttle 0 (stop hold / lane-invalid / object-stale). Formats
    // into a local stream so std::cerr's global flags are left untouched.
    if (!logged_once_ || std::fabs(t_cmd - last_logged_throttle_) >= 0.05f) {
        std::ostringstream os;
        os.setf(std::ios::fixed); os.precision(2);
        os << "[motor] mode=" << mode;
        if (front_dist >= 0.0f) os << " d=" << front_dist << "m";
        else                    os << " d=--";
        if (caution_dist > 0.0f) os << " caution=" << caution_dist << "m";
        if (ego_valid)  os << " ego=" << ego_kmh << "km/h"; else os << " ego=--";
        os << " raw=" << raw << " final=" << t_cmd;
        if (sign_cap < raw - 1e-4f) os << " cap=" << sign_cap;
        std::cerr << os.str() << "\n";
        logged_once_          = true;
        last_logged_throttle_ = t_cmd;
    }
}

}  // namespace adas
