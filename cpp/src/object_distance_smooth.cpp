#include "adas/object_distance_smooth.hpp"

#include <algorithm>
#include <unordered_set>

namespace adas {

namespace {
float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
}  // namespace

void PoseSmoother::smooth(int track_id, float* z, float* x, float* confidence) {
    if (!cfg_.enabled || track_id < 0 || z == nullptr || x == nullptr ||
        confidence == nullptr) {
        return;  // mirrors Python `if not enabled or tid is None: return`
    }
    const float conf = *confidence;
    // Confidence-adaptive EMA gain: trust high-confidence measurements faster.
    const float alpha = clamp01(
        cfg_.min_alpha + (cfg_.max_alpha - cfg_.min_alpha) * clamp01(conf));

    auto it = tracks_.find(track_id);
    if (it == tracks_.end()) {
        tracks_.emplace(track_id, Pose{*z, *x, conf});
        return;  // first sample for this track passes through unchanged
    }

    Pose& prev = it->second;
    // Per-frame depth step budget — wider when the measurement is confident,
    // so a confident jump is not over-damped but a noisy one is rate-limited.
    const float max_step = std::max(cfg_.max_depth_step_m * (0.5f + conf), 0.5f);
    const float z_limited =
        std::min(std::max(*z, prev.z - max_step), prev.z + max_step);
    const float z_smooth = prev.z + alpha * (z_limited - prev.z);
    const float x_smooth = prev.x + alpha * (*x - prev.x);
    // Confidence decays slowly so a single low-confidence frame does not make
    // a previously-stable track jumpy.
    const float conf_smooth = std::max(conf, 0.85f * prev.confidence);

    prev = Pose{z_smooth, x_smooth, conf_smooth};
    *z = z_smooth;
    *x = x_smooth;
    *confidence = conf_smooth;
}

void PoseSmoother::prune(const std::vector<int>& seen_ids) {
    if (tracks_.empty()) return;
    std::unordered_set<int> seen(seen_ids.begin(), seen_ids.end());
    for (auto it = tracks_.begin(); it != tracks_.end();) {
        it = seen.count(it->first) ? std::next(it) : tracks_.erase(it);
    }
}

void PoseSmoother::reset() { tracks_.clear(); }

}  // namespace adas
