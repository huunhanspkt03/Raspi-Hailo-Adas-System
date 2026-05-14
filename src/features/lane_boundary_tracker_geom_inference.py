"""Side-inference and lane-area width geometry for ``LaneBoundaryTracker``.

Pure functions. Missing-side inference uses ``inferred_xs = ref_xs +/-
width_profile[y]`` at the same y rows as the detected line — no
normal-vector offset, no linear ``m*y + k`` reduction. The renderer and
the tracker output both get the per-row xs directly; quadratic coeffs
are polyfitted from those xs only as a contract shim for downstream
consumers that still expect ``(a, b, c)``.
"""
from __future__ import annotations

import numpy as np


def _interp_edge(edge_pts: np.ndarray | None, ys: np.ndarray) -> np.ndarray | None:
    if edge_pts is None or len(edge_pts) < 2:
        return None
    sorted_edge = edge_pts[np.argsort(edge_pts[:, 1])]
    return np.interp(ys, sorted_edge[:, 1], sorted_edge[:, 0])


def _roi_ys(frame_h: int, top_frac: float, num_points: int) -> np.ndarray:
    return np.linspace(max(0.0, frame_h * float(top_frac)),
                       float(frame_h - 1), num=int(num_points),
                       dtype=np.float32)


def _polyfit_quadratic(ys: np.ndarray, xs: np.ndarray,
                       ) -> tuple[float, float, float] | None:
    """Quadratic ``a*y^2 + b*y + c`` through ``(ys, xs)``. None on failure.

    Used only to keep the ``(a, b, c)`` contract alive for consumers that
    still inspect coeffs (red center, validation). The renderer uses the
    raw ``xs`` directly, so any approximation error here does not affect
    the drawn polyline.
    """
    if ys.size < 3 or xs.size != ys.size:
        return None
    try:
        a, b, c = np.polyfit(ys.astype(np.float64), xs.astype(np.float64), 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    return float(a), float(b), float(c)


def _mean_width_10_rows(widths: np.ndarray) -> np.ndarray:
    """Simple row-window average for lane_area width.

    The input is already sampled on the ROI row grid. We intentionally do
    not run leak / jump / max-width gates here: lane_area is used only as
    a distance ruler, and the requested policy is ``x_max - x_min`` with
    a 10-row mean.
    """
    if widths.size == 0:
        return widths.astype(np.float32)
    w = widths.astype(np.float32, copy=False)
    out = np.empty_like(w, dtype=np.float32)
    radius = 5
    for i in range(w.size):
        lo = max(0, i - radius)
        hi = min(w.size, i + radius)
        out[i] = float(np.mean(w[lo:hi]))
    return out.astype(np.float32)


def shifted_parallel_coeffs(
    reference: tuple[float, float, float] | None,
    width: float,
    *, infer_for: str,
) -> tuple[float, float, float] | None:
    """Return ``reference`` shifted by a scalar width.

    Kept for the legacy code path: when only a scalar width is available,
    shift ``c`` by +/- width. The per-row inference path no longer routes
    through here.
    """
    if reference is None or width <= 0.0:
        return None
    a, b, c = reference
    sign = -1.0 if infer_for == "left" else 1.0
    return float(a), float(b), float(c + sign * float(width))


def infer_side_xs_from_width_profile(
    ref_xs: np.ndarray | None,
    width_profile: np.ndarray | None,
    *, infer_for: str, min_width_px: float,
) -> np.ndarray | None:
    """Per-row horizontal offset: ``ref_xs +/- width_profile`` element-wise.

    Returns ``None`` if shapes mismatch or the profile lacks enough
    positive samples. No polyfit, no normal projection — the missing
    line lives at the same y rows as the reference and is shifted only
    in x by the learned width at each row.
    """
    if (ref_xs is None or width_profile is None
            or ref_xs.size == 0 or ref_xs.size != width_profile.size):
        return None
    valid = width_profile > float(min_width_px)
    if int(np.sum(valid)) < 6:
        return None
    sign = -1.0 if infer_for == "left" else 1.0
    return (ref_xs.astype(np.float32)
            + np.float32(sign) * width_profile.astype(np.float32)
            ).astype(np.float32)


def infer_missing_side_coeffs(
    fit: dict, *, infer_for: str, reference: tuple[float, float, float],
    frame_h: int, top_frac: float, num_points: int, min_width_px: float,
) -> tuple[tuple[float, float, float], np.ndarray, np.ndarray] | None:
    """Per-row inference from *live* lane_area edges (bootstrap fallback).

    Returns ``(coeffs, xs, ys)`` so the caller can keep the coeffs
    contract alive while rendering uses the raw per-row xs. Returns
    ``None`` when lane_area edges are missing or too sparse.
    """
    le_pts = fit.get("lane_area_left_pts")
    re_pts = fit.get("lane_area_right_pts")
    ys = _roi_ys(frame_h, top_frac, num_points)
    le_x = _interp_edge(le_pts, ys)
    re_x = _interp_edge(re_pts, ys)
    if le_x is None or re_x is None:
        return None
    width_profile = (re_x - le_x).astype(np.float32)
    a, b, c = reference
    ref_xs = (a * ys * ys + b * ys + c).astype(np.float32)
    inferred_xs = infer_side_xs_from_width_profile(
        ref_xs, width_profile, infer_for=infer_for, min_width_px=min_width_px,
    )
    if inferred_xs is None:
        return None
    coeffs = _polyfit_quadratic(ys, inferred_xs)
    if coeffs is None:
        return None
    return coeffs, inferred_xs, ys


def sample_width_profile(
    left_coeffs: tuple[float, float, float] | None,
    right_coeffs: tuple[float, float, float] | None,
    frame_h: int, *, top_frac: float, num_points: int,
) -> np.ndarray | None:
    """Per-row purple width ``right_x(y) - left_x(y)`` on the ROI grid."""
    if left_coeffs is None or right_coeffs is None:
        return None
    ys = _roi_ys(frame_h, top_frac, num_points)
    a, b, c = left_coeffs
    lx = (a * ys * ys + b * ys + c).astype(np.float32)
    a, b, c = right_coeffs
    rx = (a * ys * ys + b * ys + c).astype(np.float32)
    return (rx - lx).astype(np.float32)


def infer_side_from_width_profile(
    other_coeffs: tuple[float, float, float] | None,
    width_profile: np.ndarray | None,
    *, infer_for: str, frame_h: int, top_frac: float, num_points: int,
    min_width_px: float,
) -> tuple[tuple[float, float, float], np.ndarray, np.ndarray] | None:
    """Per-row inference using a stored width-x(y) profile.

    Same ys as the detected line (the tracker ROI grid). Returns
    ``(coeffs, xs, ys)``. Coeffs are polyfitted from xs only so
    downstream coefficient consumers still work; the renderer uses the
    raw xs.
    """
    if (other_coeffs is None or width_profile is None
            or width_profile.size != int(num_points)):
        return None
    ys = _roi_ys(frame_h, top_frac, num_points)
    a, b, c = other_coeffs
    ref_xs = (a * ys * ys + b * ys + c).astype(np.float32)
    inferred_xs = infer_side_xs_from_width_profile(
        ref_xs, width_profile, infer_for=infer_for, min_width_px=min_width_px,
    )
    if inferred_xs is None:
        return None
    coeffs = _polyfit_quadratic(ys, inferred_xs)
    if coeffs is None:
        return None
    return coeffs, inferred_xs, ys


def lane_area_width_profile(
    fit: dict, *, frame_h: int, top_frac: float, num_points: int,
    min_width_px: float,
) -> np.ndarray | None:
    """Per-row lane_area width sampled on the ROI grid."""
    le_pts = fit.get("lane_area_left_pts")
    re_pts = fit.get("lane_area_right_pts")
    ys = np.linspace(max(0.0, frame_h * top_frac), float(frame_h - 1),
                     num=num_points, dtype=np.float32)
    le_x = _interp_edge(le_pts, ys)
    re_x = _interp_edge(re_pts, ys)
    if le_x is None or re_x is None:
        return None
    _ = min_width_px
    widths = (re_x - le_x).astype(np.float32)
    if not bool(np.any(widths > 0.0)):
        return None
    return _mean_width_10_rows(widths)


def lane_width_diagnostic(
    fit: dict, *, frame_h: int, top_frac: float, num_points: int, min_width_px: float,
) -> tuple[float | None, bool, str]:
    """Median lane_area width across the bottom half of the ROI."""
    le_pts = fit.get("lane_area_left_pts")
    re_pts = fit.get("lane_area_right_pts")
    ys = np.linspace(max(0.0, frame_h * top_frac), float(frame_h - 1),
                     num=num_points, dtype=np.float32)
    le_x = _interp_edge(le_pts, ys)
    re_x = _interp_edge(re_pts, ys)
    if le_x is None or re_x is None:
        return None, False, "unavailable"
    _ = min_width_px
    widths = _mean_width_10_rows((re_x - le_x).astype(np.float32))
    valid = widths > 0.0
    if not bool(np.any(valid)):
        return None, False, "lane_area_edges"
    bottom = widths[-min(10, widths.size):]
    pool = bottom[bottom > 0.0] if bool(np.any(bottom > 0.0)) else widths[valid]
    return float(np.mean(pool)), True, "lane_area_edges"
