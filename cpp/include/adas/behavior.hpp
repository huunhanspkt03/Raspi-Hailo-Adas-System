#pragma once
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
struct SafetyConfig {
    int stop_release_frames{5};     // consecutive non-stop frames before STOP releases
    int caution_release_frames{3};  // consecutive non-caution frames before CAUTION releases
    int lane_invalid_grace{4};
    int object_stale_grace{10};     // packets without a fresh object before unsafe
    float caution_throttle{0.8f};   // throttle cap while CAUTION is held
    float min_lane_confidence{0.25f}; // below this, lane is not safe for motor
    float lane_loss_stop_distance_m{4.0f}; // stop only when lane is lost AND front_object is this close
};
class SafetyGovernor {
public:
    explicit SafetyGovernor(SafetyConfig cfg = {}) : cfg_(cfg) {}
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
struct FollowConfig {
    float hard_stop_distance_m{3.0f};
    float caution_near_m{10.0f};   // ego <= ego_fast_kmh, or GPS no fix
    float caution_far_m{15.0f};    // ego > ego_fast_kmh
    float ego_fast_kmh{40.0f};     // ego speed tier boundary
    float caution_max_throttle{0.8f};

    // PWM slew (brake faster than accelerate); a hard stop bypasses it -> 0.
    float slew_up_per_s{0.6f};
    float slew_down_per_s{2.0f};
    float default_sign_cap{0.8f};   // 60 km/h default ceiling (== speed_limit_60 cap)
    int   default_sign_kmh{60};     // default km/h shown before any sign
};
class FollowController {
public:
    explicit FollowController(FollowConfig cfg = {})
        : cfg_(cfg), held_sign_cap_(cfg.default_sign_cap),
          held_sign_kmh_(cfg.default_sign_kmh) {}
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
