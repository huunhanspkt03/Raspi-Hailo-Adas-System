"""
Controller — steering computation and smoothing for RC lane-centering.

Translates lane center position into a normalized steering value [-1.0, 1.0].
Positive steering → lane center is right of frame center → steer right.
Negative steering → lane center is left of frame center → steer left.
"""
from __future__ import annotations


def compute_steering(
    lane_center_x: int,
    frame_w: int,
    deadband_px: int,
    max_steering_value: float,
) -> tuple[int, float]:
    """
    Compute steering from lane center vs frame center.

    Returns:
        error_px  — signed pixel offset (0 if within deadband)
        steering  — normalized value in [-max_steering_value, max_steering_value]
    """
    error_px = int(lane_center_x - frame_w // 2)
    if abs(error_px) <= deadband_px:
        error_px = 0

    steering = error_px / max(frame_w / 2.0, 1.0)
    steering = max(-max_steering_value, min(max_steering_value, steering))
    return error_px, float(steering)


def smooth_value(prev_value: float | None, new_value: float, alpha: float) -> float:
    """
    Exponential moving average smoothing.
    alpha=1.0 → no smoothing (pass-through).
    alpha=0.1 → heavy smoothing.
    """
    if prev_value is None:
        return new_value
    return float((1.0 - alpha) * prev_value + alpha * new_value)


def gate_by_confidence(
    raw_steer: float,
    prev_steer: float | None,
    confidence: float,
    hold_thresh: float = 0.25,
    soft_thresh: float = 0.45,
) -> float:
    """Scale steering by lane-path confidence.

    - confidence ≥ soft_thresh  → raw command passes through.
    - hold_thresh ≤ conf < soft → linear blend prev → raw.
    - confidence < hold_thresh  → hold previous command (avoid acting on noise).

    On the first frame ``prev_steer`` is None; substitute 0.0 so the hold
    branch freezes at neutral instead of leaking an uninitialized value.
    """
    prev = 0.0 if prev_steer is None else float(prev_steer)
    raw  = float(raw_steer)
    if confidence >= soft_thresh:
        return raw
    if confidence <= hold_thresh:
        return prev
    span = max(soft_thresh - hold_thresh, 1e-6)
    w = (confidence - hold_thresh) / span
    return (1.0 - w) * prev + w * raw


def rate_limit(
    new_steer: float,
    prev_steer: float | None,
    max_delta: float,
) -> float:
    """Clamp per-frame change in steering to ±max_delta.

    Prevents single-frame lurches on curve entry or source-fallback flips
    without introducing phase lag the way a deeper EMA would.
    """
    if prev_steer is None or max_delta <= 0.0:
        return float(new_steer)
    lo = float(prev_steer) - float(max_delta)
    hi = float(prev_steer) + float(max_delta)
    return max(lo, min(hi, float(new_steer)))
