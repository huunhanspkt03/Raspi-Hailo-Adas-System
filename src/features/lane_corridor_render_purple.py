"""Purple side resolution for the render corridor.

Samples purple sides on the ROI row grid from the tracker's
``SOURCE_DETECTED`` coeffs only. No synthesis, no inference, no
lane-area substitution: the renderer shows the purple line(s) the
model actually produced.

The corridor activates when at least one side is detected; the other
side stays ``None`` and is skipped downstream. When both sides are
detected, a swap+min-width invariant catches degenerate crossing
pairs.
"""
from __future__ import annotations

import numpy as np

from features.lane_boundary_tracker_contract import SOURCE_DETECTED, SOURCE_LOST


def polyval(coeffs: tuple[float, float, float], ys: np.ndarray) -> np.ndarray:
    a, b, c = coeffs
    return (a * ys * ys + b * ys + c).astype(np.float32)


def resolve_purple_xs(
    track: dict, ys: np.ndarray, frame_w: int,
    min_lane_width_px: float,
    max_lane_width_frac_w: float,
) -> tuple[np.ndarray | None, np.ndarray | None, str, str, bool, str]:
    """Sample DETECTED purple sides on the ROI grid.

    Returns ``(left_xs, right_xs, left_render_source, right_render_source,
    provisional, reason)``. Either ``left_xs`` or ``right_xs`` may be
    ``None`` when only one side is detected; both ``None`` means no
    detected purple line and the caller should fall back to the
    camera-anchored corridor. ``provisional`` is always ``False`` — the
    legacy lane-area-bootstrap render path was removed with synthesis.
    """
    _ = max_lane_width_frac_w  # kept for back-compat with callers
    l_coeffs = track.get("left_coeffs")
    r_coeffs = track.get("right_coeffs")
    l_src = track.get("left_source", SOURCE_LOST)
    r_src = track.get("right_source", SOURCE_LOST)
    l_shape_ok = l_src == SOURCE_DETECTED and l_coeffs is not None
    r_shape_ok = r_src == SOURCE_DETECTED and r_coeffs is not None
    if not l_shape_ok and not r_shape_ok:
        return None, None, l_src, r_src, False, "no detected purple line"

    left_xs = polyval(l_coeffs, ys) if l_shape_ok else None
    right_xs = polyval(r_coeffs, ys) if r_shape_ok else None
    l_render, r_render = l_src, r_src
    provisional = False

    # Both sides present: enforce the crossing / min-width / clip
    # invariants. Single-side corridors skip these — they can't cross,
    # and the corridor width comes from yellow's perspective profile.
    if left_xs is not None and right_xs is not None:
        swap = left_xs > right_xs
        if bool(np.any(swap)):
            l_copy, r_copy = left_xs.copy(), right_xs.copy()
            left_xs[swap], right_xs[swap] = r_copy[swap], l_copy[swap]
        if float(np.median(right_xs - left_xs)) < float(min_lane_width_px):
            return None, None, l_render, r_render, provisional, "corridor too narrow"

    if left_xs is not None:
        left_xs = np.clip(left_xs, 0.0, float(frame_w - 1)).astype(np.float32)
    if right_xs is not None:
        right_xs = np.clip(right_xs, 0.0, float(frame_w - 1)).astype(np.float32)
    return left_xs, right_xs, l_render, r_render, provisional, "ok"
