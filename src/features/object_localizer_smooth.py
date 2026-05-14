"""Per-track temporal smoothing for object pose estimates."""
from __future__ import annotations

from features.object_localizer_priors import _TrackPose


def smooth_pose(
    tracks: dict[int, _TrackPose],
    det: dict,
    z: float,
    x_m: float,
    confidence: float,
    smoothing_enabled: bool,
    smoothing_min_alpha: float,
    smoothing_max_alpha: float,
    max_depth_step_m: float,
) -> tuple[float, float, float]:
    tid = det.get("track_id")
    if not smoothing_enabled or tid is None:
        return z, x_m, confidence

    try:
        track_id = int(tid)
    except (TypeError, ValueError):
        return z, x_m, confidence

    prev = tracks.get(track_id)
    alpha_span = smoothing_max_alpha - smoothing_min_alpha
    alpha = smoothing_min_alpha + alpha_span * min(max(confidence, 0.0), 1.0)
    alpha = min(max(alpha, 0.0), 1.0)

    if prev is None:
        tracks[track_id] = _TrackPose(z, x_m, confidence)
        return z, x_m, confidence

    max_step = max(max_depth_step_m * (0.5 + confidence), 0.5)
    z_limited = min(max(z, prev.z_m - max_step), prev.z_m + max_step)
    z_smooth = prev.z_m + alpha * (z_limited - prev.z_m)
    x_smooth = prev.x_m + alpha * (x_m - prev.x_m)
    conf_smooth = max(confidence, 0.85 * prev.confidence)
    tracks[track_id] = _TrackPose(z_smooth, x_smooth, conf_smooth)
    return z_smooth, x_smooth, conf_smooth
