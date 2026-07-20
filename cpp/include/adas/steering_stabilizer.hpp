#pragma once

// Temporal stabilization for the scalar lane-follow steering command.
//
// WHY: the python_parity lane path already temporally smooths the *displayed*
// red centerline inside the tracker, but the C++ runtime fed compute_steering's
// RAW per-frame value straight to the servo (fusion.cpp: control.steering =
// lane.steering). So the rendered path looked stable while the servo jumped on
// every per-frame red-center wobble. This restores the smoothing chain the
// Python single-threaded loop already runs (src/pipeline_loop.py +
// src/features/controller.py: gate_by_confidence -> rate_limit -> smooth_value),
// so the displayed path and the servo command come from the same stabilized
// signal instead of disagreeing.
//
// Robust combination — each knob is a clearly named constant:
//   1. Validity gate  — too few valid path points => HOLD the last command.
//   2. Median window  — rejects single-frame outlier spikes (frame buffer).
//   3. Confidence gate— low lane confidence => blend toward / hold previous.
//   4. Rate limit     — bounds |Δsteer| per frame (no single-frame lurch).
//   5. EMA            — final smoothing toward the rate-limited target.
//
// Single-threaded: owned by the lane worker, which calls update() once per
// lane frame. No internal locking.

#include <algorithm>
#include <array>
#include <cmath>

namespace adas {

struct SteeringStabilizerConfig {
    // Median history window (frame buffer). Rejects single-frame spikes before
    // any EMA/rate-limit is applied. Must be <= kMaxHistory.
    int   history_frames{5};
    // Minimum valid red-center points before steering is trusted. When the
    // fitted polyline is shorter (lane unstable / fixed-guidance fallback) the
    // last valid command is HELD instead of snapping to a fresh bad frame.
    // This is the confidence-free validity check the task asks for.
    int   min_valid_path_points{4};

    // --- Parity with src/features/controller.py + config/base.yaml ---
    // gate_by_confidence(): below hold => hold previous; above soft => pass
    // through; between => linear blend prev->raw.
    float confidence_hold_thresh{0.25f};
    float confidence_soft_thresh{0.45f};
    // rate_limit(): max per-frame steering change, scaled by max(conf, hold)
    // exactly as pipeline_loop.py does (max_steering_rate_per_frame).
    float max_steering_rate_per_frame{0.08f};
    // smooth_value(): EMA weight on the fresh value (config smoothing_alpha).
    float smoothing_alpha{0.25f};
};

class SteeringStabilizer {
public:
    explicit SteeringStabilizer(SteeringStabilizerConfig cfg = {}) : cfg_(cfg) {}

    // raw_steer          : per-frame compute_steering() output in [-1, 1].
    // valid_path_points  : number of red-centerline points this frame (0 when
    //                      the geometry is invalid).
    // confidence         : LaneGeometry::confidence in [0, 1].
    // Returns the stabilized steering command in [-1, 1].
    float update(float raw_steer, int valid_path_points, float confidence) {
        // 1. Validity gate: hold the last valid command on an unstable frame so
        // the servo neither snaps to a bad detection nor lurches to neutral.
        if (valid_path_points < cfg_.min_valid_path_points) return output_;

        // 2. Median over the bounded history window — kills single-frame spikes.
        const int n = std::clamp(cfg_.history_frames, 1, kMaxHistory);
        hist_[head_] = std::clamp(raw_steer, -1.0f, 1.0f);
        head_ = (head_ + 1) % n;
        if (count_ < n) ++count_;
        const float med = median();

        // 3. Confidence gate (controller.py::gate_by_confidence).
        const float prev = have_output_ ? output_ : 0.0f;
        const float gated = gate_by_confidence(med, prev, confidence);

        // 4. Rate limit (controller.py::rate_limit), per-frame slew bounded by
        // max_steering_rate * max(conf, hold) — confident frames may move
        // faster, noisy ones are pinned near the previous command.
        const float max_delta = cfg_.max_steering_rate_per_frame *
                                std::max(confidence, cfg_.confidence_hold_thresh);
        const float limited = have_output_
            ? std::clamp(gated, prev - max_delta, prev + max_delta)
            : gated;

        // 5. EMA smoothing (controller.py::smooth_value).
        output_ = have_output_
            ? (1.0f - cfg_.smoothing_alpha) * output_ + cfg_.smoothing_alpha * limited
            : limited;
        output_ = std::clamp(output_, -1.0f, 1.0f);
        have_output_ = true;
        return output_;
    }

    float value() const { return output_; }
    void reset() {
        head_ = 0; count_ = 0; output_ = 0.0f; have_output_ = false;
        hist_.fill(0.0f);
    }

private:
    float gate_by_confidence(float raw, float prev, float conf) const {
        if (conf >= cfg_.confidence_soft_thresh) return raw;
        if (conf <= cfg_.confidence_hold_thresh) return prev;
        const float span = std::max(
            cfg_.confidence_soft_thresh - cfg_.confidence_hold_thresh, 1e-6f);
        const float w = (conf - cfg_.confidence_hold_thresh) / span;
        return (1.0f - w) * prev + w * raw;
    }

    float median() const {
        std::array<float, kMaxHistory> tmp{};
        for (int i = 0; i < count_; ++i) tmp[i] = hist_[i];
        std::sort(tmp.begin(), tmp.begin() + count_);
        return tmp[count_ / 2];
    }

    static constexpr int kMaxHistory = 9;
    SteeringStabilizerConfig cfg_;
    std::array<float, kMaxHistory> hist_{};
    int   head_{0};
    int   count_{0};
    float output_{0.0f};
    bool  have_output_{false};
};

}  // namespace adas
