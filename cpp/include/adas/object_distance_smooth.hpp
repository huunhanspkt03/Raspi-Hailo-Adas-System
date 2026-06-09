#pragma once

// Per-track temporal smoothing for monocular pose estimates. Direct port of
//   src/features/object_localizer_smooth.py::smooth_pose
// plus the prune-to-seen-tracks step from object_localizer.localize. Keyed on
// the detection track_id assigned by the object tracker (update_tracker). When
// a detection has no track (track_id < 0) smoothing is a no-op, mirroring the
// Python `if tid is None: return z, x_m, confidence`.
//
// State is per-track {z, x, confidence}. Each frame: smooth() every tracked
// detection, then prune() to the ids actually seen so dropped tracks do not
// leak. Single-threaded use only (the object worker thread).

#include <unordered_map>
#include <vector>

namespace adas {

struct PoseSmoothConfig {
    bool  enabled{true};
    float min_alpha{0.25f};        // EMA gain at confidence 0
    float max_alpha{0.70f};        // EMA gain at confidence 1
    float max_depth_step_m{8.0f};  // per-frame z step budget (scaled by conf)
};

class PoseSmoother {
public:
    explicit PoseSmoother(PoseSmoothConfig cfg = {}) : cfg_(cfg) {}

    // Smooth (z, x, confidence) in place for track_id. No-op (leaves inputs
    // unchanged) when smoothing is disabled or track_id < 0.
    void smooth(int track_id, float* z, float* x, float* confidence);

    // Drop state for any track not present in seen_ids. Call once per frame
    // after all smooth() calls (mirrors localize()'s seen_tracks prune).
    void prune(const std::vector<int>& seen_ids);

    // Clear all temporal state (e.g. on a video/stream switch).
    void reset();

private:
    struct Pose { float z; float x; float confidence; };
    PoseSmoothConfig cfg_;
    std::unordered_map<int, Pose> tracks_;
};

}  // namespace adas
