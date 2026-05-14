"""Ego-anchored red path: tail pinned at vehicle center, head bends toward
the lane target ahead.

The red path is a desired ego trajectory in image space. It is built each
frame from per-row lane-center targets (typically the fused red center from
``red_center_from_boundaries``) and obeys two anchor constraints:

* ``x(y_tail) = ego_x``      — tail pinned at the camera/vehicle column
* ``dx/dy(y_tail) = 0``      — zero slope at the tail (no kink at the camera)

The free parameter ``a`` is fit by least squares to the lane-center samples,
so the head/upper portion bends toward the lane target. The lookahead point
sits a configurable fraction of frame height up from the bottom and is what
the controller should consume — never the fixed tail.

This module is stateless. Frame-to-frame smoothing of ``a`` and the
controller's existing confidence gate / rate limit still apply downstream.
"""
from __future__ import annotations

import numpy as np

DEFAULT_LOOKAHEAD_FRAC = 0.70
# Hard reject extreme curvatures (top would jump > half the frame).
DEFAULT_MAX_CURVATURE_A = 8e-3
# Clamp top-row displacement to this fraction of frame_w so the curve
# stays inside the frame even on aggressive lane shifts.
DEFAULT_MAX_TOP_DISP_FRAC_W = 0.40
# Below this damped magnitude the path is treated as straight.
DEFAULT_CURVATURE_DEADBAND_A = 1e-5

__all__ = (
    "DEFAULT_LOOKAHEAD_FRAC",
    "DEFAULT_MAX_CURVATURE_A",
    "DEFAULT_MAX_TOP_DISP_FRAC_W",
    "DEFAULT_CURVATURE_DEADBAND_A",
    "fit_ego_anchored_a",
    "build_ego_anchored_path",
    "ego_path_dxdy_per_y",
    "lookahead_y",
)


def lookahead_y(frame_h: int, lookahead_frac: float = DEFAULT_LOOKAHEAD_FRAC) -> float:
    """Image-space lookahead row. Smaller value = further ahead in world."""
    fh = float(int(frame_h))
    return float(np.clip(fh * float(lookahead_frac), 0.0, fh - 1.0))


def fit_ego_anchored_a(
    target_pts: np.ndarray | None,
    y_tail: float,
    ego_x: float,
    *,
    max_curvature_a: float = DEFAULT_MAX_CURVATURE_A,
    deadband_a: float = DEFAULT_CURVATURE_DEADBAND_A,
    frame_w: int | None = None,
    max_top_disp_frac_w: float = DEFAULT_MAX_TOP_DISP_FRAC_W,
    y_top: float | None = None,
) -> tuple[float, dict]:
    """Least-squares fit of ``a`` in ``x = ego_x + a*(y - y_tail)^2``.

    The closed form is ``a = sum(dx_i * dy_i^2) / sum(dy_i^4)`` where
    ``dy_i = y_i - y_tail`` and ``dx_i = x_i - ego_x``. The tail-anchor and
    zero-slope-at-tail constraints are baked into the parameterization.

    Applies (in order): hard-reject on raw magnitude, deadband, top
    displacement clamp. Returns ``(a, diag)``; ``diag`` is a dict that
    documents what happened for the caller.
    """
    diag: dict = {
        "ego_path_a_raw": 0.0,
        "ego_path_a": 0.0,
        "ego_path_n_targets": 0,
        "ego_path_fit_valid": False,
        "ego_path_reason": "no_targets",
        "ego_path_top_disp_clamped": False,
    }
    if target_pts is None:
        return 0.0, diag
    arr = np.asarray(target_pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 2:
        return 0.0, diag

    ys_t = arr[:, 1].astype(np.float64)
    xs_t = arr[:, 0].astype(np.float64)
    dy = ys_t - float(y_tail)
    dx = xs_t - float(ego_x)
    denom = float(np.sum(dy ** 4))
    if denom < 1e-3:
        diag["ego_path_reason"] = "degenerate_denominator"
        return 0.0, diag

    a_raw = float(np.sum(dx * dy * dy) / denom)
    diag["ego_path_a_raw"] = a_raw
    diag["ego_path_n_targets"] = int(arr.shape[0])

    if abs(a_raw) > float(max_curvature_a):
        diag["ego_path_reason"] = "curvature_too_large"
        return 0.0, diag

    a = a_raw if abs(a_raw) >= float(deadband_a) else 0.0
    if a == 0.0 and a_raw != 0.0:
        diag["ego_path_reason"] = "below_deadband"

    if a != 0.0 and frame_w is not None and y_top is not None:
        max_disp = float(max_top_disp_frac_w) * float(frame_w)
        dy_top = float(y_top) - float(y_tail)
        if max_disp > 0.0 and dy_top * dy_top > 1.0:
            max_a_by_top = max_disp / (dy_top * dy_top)
            if abs(a) > max_a_by_top:
                a = (max_a_by_top if a > 0.0 else -max_a_by_top)
                diag["ego_path_top_disp_clamped"] = True

    diag["ego_path_a"] = a
    diag["ego_path_fit_valid"] = a != 0.0 or diag["ego_path_n_targets"] > 0
    if diag["ego_path_reason"] == "no_targets":
        diag["ego_path_reason"] = "ok"
    return a, diag


def build_ego_anchored_path(
    target_pts: np.ndarray | None,
    ys: np.ndarray,
    frame_h: int,
    frame_w: int,
    *,
    lookahead_frac: float = DEFAULT_LOOKAHEAD_FRAC,
    max_curvature_a: float = DEFAULT_MAX_CURVATURE_A,
    deadband_a: float = DEFAULT_CURVATURE_DEADBAND_A,
    max_top_disp_frac_w: float = DEFAULT_MAX_TOP_DISP_FRAC_W,
) -> tuple[np.ndarray, dict]:
    """Build the ego-anchored quadratic centerline sampled at ``ys``.

    Path is ``x(y) = ego_x + a*(y - y_tail)^2`` with ``y_tail = ys[-1]`` and
    ``ego_x = frame_w/2``. The tail is exactly ``ego_x`` with zero slope;
    the head bends toward the lane targets according to the LSQ-fit ``a``.

    Returns ``(xs, diag)``. ``diag`` carries ``ego_path_control_x`` and
    ``ego_path_control_y`` at the lookahead row — these are what the
    controller should consume. ``xs[-1]`` is always ``ego_x`` (the
    rendered tail).
    """
    fw = int(frame_w)
    fh = int(frame_h)
    ego_x = 0.5 * float(fw)
    if ys is None or ys.size == 0:
        diag = {
            "ego_path_a": 0.0, "ego_path_anchor_y": float(fh - 1),
            "ego_path_lookahead_y": lookahead_y(fh, lookahead_frac),
            "ego_path_control_x": float(ego_x),
            "ego_path_control_y": int(round(lookahead_y(fh, lookahead_frac))),
            "ego_path_n_targets": 0, "ego_path_fit_valid": False,
            "ego_path_reason": "empty_ys", "ego_path_top_disp_clamped": False,
        }
        return np.zeros((0,), dtype=np.float32), diag

    ys_f = ys.astype(np.float32)
    y_tail = float(ys_f[-1])
    y_top = float(ys_f[0])
    y_la = lookahead_y(fh, lookahead_frac)

    a, diag = fit_ego_anchored_a(
        target_pts, y_tail, ego_x,
        max_curvature_a=max_curvature_a,
        deadband_a=deadband_a,
        frame_w=fw,
        max_top_disp_frac_w=max_top_disp_frac_w,
        y_top=y_top,
    )

    dy_path = ys_f - np.float32(y_tail)
    xs_path = (np.float32(a) * dy_path * dy_path + np.float32(ego_x)).astype(np.float32)
    xs_path = np.clip(xs_path, 0.0, float(fw - 1)).astype(np.float32)

    dy_la = y_la - y_tail
    control_x = float(np.clip(a * dy_la * dy_la + ego_x, 0.0, fw - 1))

    diag["ego_path_anchor_y"] = y_tail
    diag["ego_path_lookahead_y"] = y_la
    diag["ego_path_control_x"] = control_x
    diag["ego_path_control_y"] = int(round(y_la))
    return xs_path, diag


def ego_path_dxdy_per_y(
    ys: np.ndarray, a: float, y_tail: float,
) -> np.ndarray:
    """Per-row ``dx/dy`` for the ego-anchored path: ``2 * a * (y - y_tail)``.

    Slope is zero at the tail by construction. Used by the yellow corridor
    builder so yellow follows red's normal exactly.
    """
    if ys is None or ys.size == 0 or a == 0.0:
        return np.zeros_like(ys, dtype=np.float32) if ys is not None else np.zeros(0, np.float32)
    return (2.0 * float(a)
            * (ys.astype(np.float32) - np.float32(y_tail))).astype(np.float32)
