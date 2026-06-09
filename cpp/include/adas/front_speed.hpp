#pragma once

// Single-vehicle speed estimator for the FUSED front_object.
// Strictly display/log only — does not feed motor/RC. Runs once per
// BehaviorPacket from a single thread (Runtime::render_loop), so no
// internal locking is needed.
//
// The estimator differentiates the front_object's already-annotated
// distance over wall-clock time, with EMA smoothing and reject-on-glitch
// guardrails. From the same smoothed signed rate it also derives a time gap
// for the warning HUD, in BOTH directions:
//   ttc_s = distance / signed_rate
//     > 0 closing (seconds to collision), < 0 opening (gap growing).
//   valid whenever |rate| > noise floor (the HUD red alert still only fires
//   on a positive TTC under threshold).
// When bp.gps is valid it also estimates the front vehicle's
// absolute speed as:
//   front_speed_kmh = ego_gps_speed_kmh + relative_speed_kmh
// It resets on:
//   * absent front_object (no in-corridor vehicle this frame)
//   * invalid lane geometry (front_object selection is unsound)
//   * missing distance source (pose_valid + z_m both unavailable)
//   * a change of lead vehicle identity (track_id) — distances of two
//     DIFFERENT vehicles must never be differentiated against each other
//   * large dt gap (object stream paused) — kept in [kMinDt, kMaxDt]
// and rejects implausible single-sample rates (> kMaxRateMps) without
// poisoning the EMA.
//
// dt is measured between OBJECT-packet timestamps (bp.objects->timestamp),
// not render-packet timestamps: in the alternate schedule the same object
// packet is re-paired with several render packets, and the distance sample's
// true age is the object frame's capture time.

#include "adas/types.hpp"

namespace adas {

class FrontSpeedEstimator {
public:
    // Annotate bp.objects->front_object with relative-speed fields and,
    // when GPS ego speed is valid, front_speed_kmh. No-op otherwise.
    // Idempotent on repeated cached ObjectPackets — the last EMA is
    // re-stamped onto the new BehaviorPacket so the HUD stays continuous.
    void update(BehaviorPacket& bp);

    // Drop all internal state. Called automatically on reset
    // conditions; exposed so callers can force-clear on video restart.
    void reset();

private:
    bool      has_prev_{false};
    FrameId   prev_object_frame_id_{0};
    TimePoint prev_time_{};            // object-packet timestamp of prev sample
    float     prev_distance_m_{0.0f};
    int       prev_track_id_{-1};      // lead identity the samples belong to
    bool      ema_ready_{false};
    float     ema_mps_{0.0f};       // positive = closing rate, m/s
    // Periodic log dedup: emit on first valid estimate and on closure
    // changes > kLogChangeKmh; emit a single "cleared" line on reset.
    bool  logged_active_{false};
    float last_logged_closing_kmh_{0.0f};
};

}  // namespace adas
