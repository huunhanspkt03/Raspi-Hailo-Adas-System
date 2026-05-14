"""Pure geometry helpers for ``red_center_from_boundaries``.

Stateless math: polyline interpolation, purple midpoint extraction,
lane_area row-centroid fit, lane_area row clipping, and a lightweight
quadratic-curvature road classifier. Split out so the fusion module
stays focused on policy/weighting.
"""
from __future__ import annotations

import numpy as np

from features.dynamic_corridor import compute_dynamic_corridor
from features.lane_boundary_tracker_contract import SOURCE_DETECTED, SOURCE_LOST

ROAD_STRAIGHT = "STRAIGHT"
ROAD_CURVE_LEFT = "CURVE_LEFT"
ROAD_CURVE_RIGHT = "CURVE_RIGHT"
ROAD_UNKNOWN = "UNKNOWN"


def empty_pts() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def interp_polyline(pts: np.ndarray | None, ys: np.ndarray) -> np.ndarray | None:
    if pts is None or len(pts) < 2:
        return None
    s = pts[np.argsort(pts[:, 1])]
    return np.interp(ys, s[:, 1], s[:, 0]).astype(np.float32)


def purple_midpoint(track: dict | None, ys: np.ndarray, src_weights: dict
                    ) -> tuple[np.ndarray | None, float, str, str,
                               np.ndarray | None, np.ndarray | None]:
    """Per-y midpoint of purple boundaries when BOTH sides are DETECTED.

    Returns ``(xs, conf, left_source, right_source, left_xs, right_xs)``.
    ``xs`` is ``None`` if either side is not a current model detection or
    the polylines cannot be interpolated on ``ys``. HELD / legacy INFERRED
    sides are deliberately ignored so hidden synthetic/stale boundaries
    cannot pull the red center.
    """
    if not isinstance(track, dict):
        return None, 0.0, SOURCE_LOST, SOURCE_LOST, None, None
    l_src = track.get("left_source", SOURCE_LOST)
    r_src = track.get("right_source", SOURCE_LOST)
    if l_src != SOURCE_DETECTED or r_src != SOURCE_DETECTED:
        return None, 0.0, l_src, r_src, None, None
    lx = interp_polyline(track.get("left_pts_img"), ys)
    rx = interp_polyline(track.get("right_pts_img"), ys)
    if lx is None or rx is None:
        return None, 0.0, l_src, r_src, None, None
    mid = (0.5 * (lx + rx)).astype(np.float32)
    lc = float(track.get("left_confidence_final", 0.0))
    rc = float(track.get("right_confidence_final", 0.0))
    lw = src_weights.get(l_src, 0.0)
    rw = src_weights.get(r_src, 0.0)
    return mid, float(0.5 * (lw * lc + rw * rc)), l_src, r_src, lx, rx


def purple_sides_separable(left_xs: np.ndarray | None,
                           right_xs: np.ndarray | None,
                           *, min_gap_px: float = 1.0) -> bool:
    """True iff per-y purple left strictly precedes purple right with at
    least ``min_gap_px`` of clearance everywhere on the sampling grid.
    Used as the safety gate for "purple is primary"."""
    if left_xs is None or right_xs is None:
        return False
    if left_xs.size == 0 or right_xs.size != left_xs.size:
        return False
    return bool(np.all(right_xs - left_xs >= float(min_gap_px)))


def clip_between(xs: np.ndarray, left_xs: np.ndarray, right_xs: np.ndarray,
                 *, eps_px: float = 1.0) -> np.ndarray:
    """Clamp ``xs`` to ``[left_xs + eps, right_xs - eps]`` per row."""
    eps = float(eps_px)
    lo = (left_xs + eps).astype(np.float32)
    hi = (right_xs - eps).astype(np.float32)
    return np.minimum(np.maximum(xs.astype(np.float32), lo), hi)


def lane_area_evidence(
    lane_area, frame_h: int, frame_w: int, ys: np.ndarray,
    *, top_frac: float, num_points: int, min_pixels: int,
) -> tuple[np.ndarray | None, float, tuple[float, float, float] | None]:
    """Quadratic fit of lane_area row centroids, sampled at common ys."""
    if lane_area is None or getattr(lane_area, "size", 0) == 0:
        return None, 0.0, None
    if int(np.count_nonzero(lane_area)) < int(min_pixels):
        return None, 0.0, None
    corr = compute_dynamic_corridor(
        lane_area, frame_h, frame_w,
        top_frac=top_frac, num_points=num_points,
    )
    if corr is None:
        return None, 0.0, None
    a, b, c = corr["coeffs"]
    xs = (a * ys * ys + b * ys + c).astype(np.float32)
    return xs, float(corr["confidence"]), corr["coeffs"]


def clip_to_lane_area(xs: np.ndarray, ys: np.ndarray, lane_area: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Clamp each x to its row's leftmost / rightmost lane_area pixel.

    Rows where lane_area is empty produce ``valid=False`` so the caller
    can drop them before the polynomial refit.
    """
    h = lane_area.shape[0]
    out = xs.astype(np.float64).copy()
    valid = np.ones_like(ys, dtype=bool)
    for i, y in enumerate(ys):
        ri = int(round(float(y)))
        if ri < 0 or ri >= h:
            valid[i] = False
            continue
        cols = np.where(lane_area[ri] > 0)[0]
        if cols.size == 0:
            valid[i] = False
            continue
        out[i] = float(np.clip(out[i], float(cols[0]), float(cols[-1])))
    return out.astype(np.float32), valid


def classify_road(coeffs: tuple[float, float, float] | None, thresh: float) -> str:
    """Convention (image y-down, x rightward): ``a > thresh`` → curve-left
    (road veers left in distance because x decreases as y decreases);
    ``a < -thresh`` → curve-right. Diagnostic only, no temporal smoothing."""
    if coeffs is None:
        return ROAD_UNKNOWN
    a = float(coeffs[0])
    if a > thresh:
        return ROAD_CURVE_LEFT
    if a < -thresh:
        return ROAD_CURVE_RIGHT
    return ROAD_STRAIGHT
