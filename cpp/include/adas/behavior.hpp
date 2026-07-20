#pragma once

// Lightweight behavior policy. Full behavior logic is still a port target of:
//   src/features/behavior_logic.py     (BehaviorSmoother, decide_action)
//   src/features/behavior_runner.py    (run_behavior)
//   src/features/control_runner.py     (send_rc_command)
//
// Motor policy is authoritatively the FollowController (below), a distance-only
// longitudinal controller on the fused in-corridor front object:
//   < 3 m                 -> Stop(0), unconditional (lead moving or not).
//   [3 m, caution_dist)   -> Caution, throttle ramps linearly 0 -> 0.8.
//   >= caution_dist       -> Monitor, full cruise (0.8).
// caution_dist is ego-speed tiered: 10 m at <=40 km/h or no GPS fix, 15 m above.
// decide_behavior() only produces a provisional action used when there is no
// front object; with a front object the FollowController overrides it. Lane
// validity is reserved for steering/overlay and the corridor membership test;
// it must not gate motor throttle on its own.

#include <string>

#include "adas/types.hpp"

namespace adas {

struct BehaviorState {
    BehaviorAction action{BehaviorAction::Go};
    std::string reason{"no_front_object"};
    float throttle{1.0f};
};

BehaviorState decide_behavior(const std::optional<LanePacket>& lane,
                              const std::optional<ObjectPacket>& objects);

// Tunables for the SafetyGovernor. Frame counts are in render packets.
struct SafetyConfig {
    int stop_release_frames{5};     // consecutive non-stop frames before STOP releases
    int caution_release_frames{3};  // consecutive non-caution frames before CAUTION releases
    // Consecutive invalid-lane packets tolerated before the close-object
    // stop arms. Kept short so a near lead vehicle that hides the lane/purple
    // cues triggers fail-safe promptly while still ignoring brief single-frame
    // geometry dropouts.
    int lane_invalid_grace{4};
    int object_stale_grace{10};     // packets without a fresh object before unsafe
    float caution_throttle{0.8f};   // throttle cap while CAUTION is held
    float min_lane_confidence{0.25f}; // below this, lane is not safe for motor
    float lane_loss_stop_distance_m{4.0f}; // stop only when lane is lost AND front_object is this close
};

// Behavior smoother + fail-closed safety supervisor. Mutates a BehaviorPacket
// in place just before it is handed to the RC actuator.
//
// Smoothing (hysteresis):
//   * STOP engages immediately on a Stop action and is held for at least
//     stop_release_frames further packets, so a single dropped detection
//     cannot release the brake.
//   * CAUTION is similarly held for caution_release_frames to reduce flicker.
//
// Supervision (fail closed — when uncertain, throttle 0 / safe=false):
//   * effective Stop forces throttle 0.
//   * lane geometry invalid/low-confidence/missing strict gate does not stop
//     the motor by itself. It marks the packet unsafe only when the invalid
//     streak outlasts the tracker's HELD bridging window AND the selected
//     front_object is very close, because that likely means a near in-lane
//     object is occluding the lane. HELD frames (purple polylines absent but
//     strict gate + confidence intact) count as VALID — the tracker holds
//     boundaries across short dropouts by design.
//   * no fresh object for > object_stale_grace packets marks the packet
//     unsafe.
//   * any unsafe packet forces throttle 0.
//
// NOT thread-safe by design: construct and call apply() from a single thread
// (the render loop), which is the only producer of RC commands.
class SafetyGovernor {
public:
    explicit SafetyGovernor(SafetyConfig cfg = {}) : cfg_(cfg) {}

    // `object_fresh` is true when this packet carried a recent object packet.
    // (Both pipeline branches are always active in this build.)
    void apply(BehaviorPacket& bp, bool object_fresh);

private:
    SafetyConfig cfg_;
    int stop_hold_{0};
    int caution_hold_{0};
    int lane_invalid_streak_{0};
    int object_stale_streak_{0};
    // Concise change-only logging state.
    bool logged_once_{false};
    BehaviorAction last_action_{BehaviorAction::Go};
    bool last_safe_{true};
    std::string last_reason_;
};

// Tunables for the longitudinal throttle controller (distance-only). Throttle
// is RC [0,1] duty; distances in meters. Behavior for the selected in-corridor
// front_object:
//   < hard_stop_distance_m          -> STOP (throttle 0), unconditional.
//   [hard_stop, caution_dist)       -> CAUTION, linear ramp 0 -> caution_max_throttle.
//   >= caution_dist                 -> MONITOR, full cruise.
// caution_dist is ego-speed tiered (caution_far_m if ego > ego_fast_kmh, else
// caution_near_m — which also covers the no-GPS-fix case). No TTC term.
struct FollowConfig {
    // Hard safety stop on the displayed near distance, for the selected
    // front_object only (speed_object never hard-stops). 3.0 m absolute margin,
    // GPS-independent, applied whether or not the lead is moving.
    float hard_stop_distance_m{3.0f};

    // Throttle reaches full cruise at the caution distance and ramps linearly
    // down to 0 at the hard-stop margin. Caution distance by ego speed:
    float caution_near_m{10.0f};   // ego <= ego_fast_kmh, or GPS no fix
    float caution_far_m{15.0f};    // ego > ego_fast_kmh
    float ego_fast_kmh{40.0f};     // ego speed tier boundary
    float caution_max_throttle{0.8f};

    // PWM slew (brake faster than accelerate); a hard stop bypasses it -> 0.
    float slew_up_per_s{0.6f};
    float slew_down_per_s{2.0f};

    // --- Speed-limit sign ceiling ---------------------------------------------
    // A speed-limit sign sets the motor's MAX throttle: the follow / TTC /
    // safe-distance rules can only reduce BELOW it, never exceed it. Before any
    // sign is seen the ceiling is default_sign_cap (60 km/h). Once a sign is
    // detected it is adopted IMMEDIATELY and HELD INDEFINITELY until a different
    // sign is detected — it persists after the sign leaves the frame. Per-sign
    // caps come from object_detection.cpp::yolo26_speed_limit_throttle_cap
    // (50->0.6 60->0.8 70->0.8 80->1.0).
    float default_sign_cap{0.8f};   // 60 km/h default ceiling (== speed_limit_60 cap)
    int   default_sign_kmh{60};     // default km/h shown before any sign
};

// Longitudinal throttle controller (distance-only). Runs in the render loop
// BEFORE the SafetyGovernor (the final authority).
//   In-corridor front_object: STOP < 3 m (unconditional); CAUTION with a
//   linear throttle ramp between 3 m and the ego-speed-tiered caution distance
//   (10 m at <=40 km/h or no GPS fix, 15 m above 40 km/h); MONITOR / full
//   cruise beyond it.
//   speed_object is display/speed only — it must not lower throttle or stop
//   the car.
// Final throttle is capped at the max cruise (0.8) and the speed-limit sign
// ceiling. With no front object it only applies the sign ceiling.
// Single-thread (render loop) only.
class FollowController {
public:
    explicit FollowController(FollowConfig cfg = {})
        : cfg_(cfg), held_sign_cap_(cfg.default_sign_cap),
          held_sign_kmh_(cfg.default_sign_kmh) {}

    // Set bp.control.throttle/behavior_action in place from the front object
    // (GPS-speed safe-distance follow / TTC / distance, + hard stop), then apply
    // the speed-limit sign ceiling as the throttle MAX in EVERY path (incl. when
    // there is no front/speed object — the default 60 km/h ceiling still binds).
    void update(BehaviorPacket& bp);

    // Drop frame-to-frame state (call on video restart / stream gaps).
    void reset();

private:
    // Held speed-limit sign ceiling, advanced every frame (see behavior.cpp).
    float sign_ceiling(const std::optional<ObjectPacket>& objects);
    // Asymmetric throttle slew toward target, shared by the early-return paths.
    float slew(float target, const BehaviorPacket& bp) const;

    FollowConfig cfg_;
    bool      has_prev_{false};
    TimePoint prev_time_{};
    float     prev_throttle_{0.0f};
    // change-only logging
    bool  logged_once_{false};
    float last_logged_throttle_{0.0f};
    // Held speed-limit sign ceiling state. Adopted on detection, held
    // indefinitely until a different sign is seen.
    float held_sign_cap_{0.8f};
    int   held_sign_kmh_{60};
    bool  ever_seen_sign_{false};   // false -> default ceiling; true -> detected/held
    bool  sign_detected_{false};    // a sign is present in the current frame
};

}  // namespace adas
