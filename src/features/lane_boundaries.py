"""Fit purple outer lane boundaries from YOLO26-seg dotted/solid line masks.

Line-object based: the ``dotted_line``/``solid_line`` semantic masks are unioned and
fed straight into connected-component analysis + side split + robust
polynomial fit. ``lane_area`` is **only** consulted for diagnostics
(per-row left/right edge points exposed in the result dict, used by the
tracker for missing-side inference and lane-width); it is no longer a
hard mask filter that removes dotted/solid pixels, and it does not reject
components that sit far from the lane_area edges.
"""
from __future__ import annotations

import numpy as np

from features.lane_boundary_components import (
    analyze_components, assign_sides_by_split,
    collect_side_pixels, extract_lane_area_edges,
)
from features.lane_boundary_fit import (
    REJECT_GEOMETRY_CROSSING, REJECT_GEOMETRY_WIDTH,
    STATUS_EMPTY, STATUS_OK, STATUS_WEAK,
    enforce_geometry, fit_side,
)

__all__ = (
    "LINE_CLASS_KEYS",
    "REJECT_GEOMETRY_CROSSING", "REJECT_GEOMETRY_WIDTH",
    "STATUS_EMPTY", "STATUS_OK", "STATUS_WEAK",
    "fit_lane_boundaries",
)

DEFAULT_TOP_FRAC = 0.55
DEFAULT_NUM_POINTS = 24
DEFAULT_MIN_PIXELS = 30
DEFAULT_RESIDUAL_TRIM_FRAC = 0.20
DEFAULT_MAX_RESIDUAL_PX = 30.0
DEFAULT_MIN_AREA = 25
DEFAULT_MIN_HEIGHT = 18
# Real perspective lane lines that curve across the frame can have wide
# bbox aspect (e.g., a near-edge stripe spans ~400px laterally over ~250px
# vertically). Angle/aspect pre-filters here just kill obvious crosswalks
# / arrows / text. With lane_area no longer gating components, side
# assignment leans on ``side_split_x`` (explicit > tracker memory >
# lane_area median > frame center).
DEFAULT_MAX_ASPECT = 4.0
DEFAULT_MAX_ANGLE_DEG = 80.0
DEFAULT_MIN_BOTTOM_WIDTH_FRAC = 0.10
DEFAULT_MAX_BOTTOM_WIDTH_FRAC = 0.85
LINE_CLASS_KEYS = ("dotted_line", "solid_line")


def _union_line_masks(semantics: dict[str, np.ndarray]) -> np.ndarray | None:
    first = next(iter(semantics.values()), None)
    if first is None:
        return None
    h, w = first.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for key in LINE_CLASS_KEYS:
        m = semantics.get(key)
        if m is not None:
            out |= (m > 0).astype(np.uint8)
    return out


def _sample_edge(edge: np.ndarray, top_cut: int, num_points: int) -> np.ndarray:
    if edge.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    ys_full = np.arange(top_cut, top_cut + edge.size, dtype=np.float32)
    valid = ~np.isnan(edge)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float32)
    pts = np.stack([edge[valid], ys_full[valid]], axis=1)
    if pts.shape[0] > num_points:
        idx = np.linspace(0, pts.shape[0] - 1, num_points).astype(np.int32)
        pts = pts[idx]
    return pts.astype(np.float32)


def _empty_result(frame_w: int) -> dict:
    e = np.empty((0, 2), dtype=np.float32)
    return {"left_coeffs": None, "right_coeffs": None,
            "left_pts_img": e, "right_pts_img": e,
            "left_confidence": 0.0, "right_confidence": 0.0,
            "n_left": 0, "n_right": 0,
            "side_split_x_used": int(frame_w // 2),
            "status": {"left": STATUS_EMPTY, "right": STATUS_EMPTY},
            "components": [], "lane_area_left_pts": e.copy(),
            "lane_area_right_pts": e.copy(), "geometry_reject_reason": None}


def fit_lane_boundaries(
    semantics, lane_area, frame_h, frame_w,
    *,
    side_split_x=None,
    prev_side_split_x=None,
    top_frac=DEFAULT_TOP_FRAC,
    min_pixels=DEFAULT_MIN_PIXELS,
    min_component_area=DEFAULT_MIN_AREA,
    min_component_height=DEFAULT_MIN_HEIGHT,
    max_aspect=DEFAULT_MAX_ASPECT,
    max_angle_deg=DEFAULT_MAX_ANGLE_DEG,
    max_residual_px=DEFAULT_MAX_RESIDUAL_PX,
    min_bottom_width_frac=DEFAULT_MIN_BOTTOM_WIDTH_FRAC,
    max_bottom_width_frac=DEFAULT_MAX_BOTTOM_WIDTH_FRAC,
    residual_trim_frac=DEFAULT_RESIDUAL_TRIM_FRAC,
    num_points=DEFAULT_NUM_POINTS,
):
    """Fit purple outer boundaries from dotted/solid line semantics.

    ``lane_area`` is optional; it is only used for the diagnostic edge
    points and as a final fallback for ``side_split_x``. The dotted/solid
    masks are NOT intersected with lane_area, and components are not
    rejected by distance to lane_area edges.

    ``side_split_x`` priority:
        explicit ``side_split_x`` >  ``prev_side_split_x`` (tracker memory)
        > lane_area median        >  frame center.

    Returns a dict with the per-side fit (``left_coeffs``, ``right_coeffs``,
    ``*_pts_img``, ``*_confidence``, ``n_*``, ``status``), diagnostic
    fields (``side_split_x_used``, ``components``, ``lane_area_*_pts``,
    ``geometry_reject_reason``).
    """
    if semantics is None:
        return _empty_result(frame_w)
    union = _union_line_masks(semantics)
    if union is None or not np.any(union):
        return _empty_result(frame_w)
    top_cut = int(round(frame_h * top_frac))
    union_roi = union.copy()
    union_roi[:top_cut] = 0
    components = analyze_components(
        union_roi,
        min_area=min_component_area, min_height=min_component_height,
        max_aspect=max_aspect, max_angle_deg=max_angle_deg,
    )
    lane_left, lane_right, lane_center = extract_lane_area_edges(lane_area, top_cut)
    if side_split_x is not None:
        split_used = int(side_split_x)
    elif prev_side_split_x is not None:
        split_used = int(round(float(prev_side_split_x)))
    elif lane_left.size > 0:
        valid = ~np.isnan(lane_center)
        split_used = (int(round(float(np.median(lane_center[valid]))))
                      if np.any(valid) else int(frame_w // 2))
    else:
        split_used = int(frame_w // 2)
    assign_sides_by_split(components, split_used)
    ly, lx = collect_side_pixels(components, "left")
    ry, rx = collect_side_pixels(components, "right")
    fk = dict(frame_h=frame_h, frame_w=frame_w, top_cut=top_cut,
              min_pixels=min_pixels, residual_trim_frac=residual_trim_frac,
              max_residual_px=max_residual_px, num_points=num_points, top_frac=top_frac)
    left, right = fit_side(ly, lx, **fk), fit_side(ry, rx, **fk)
    left, right, geom_reason = enforce_geometry(
        left, right,
        min_bottom_width=min_bottom_width_frac * frame_w,
        max_bottom_width=max_bottom_width_frac * frame_w,
    )
    return {
        "left_coeffs": left.coeffs, "right_coeffs": right.coeffs,
        "left_pts_img": left.pts_img, "right_pts_img": right.pts_img,
        "left_confidence": left.confidence, "right_confidence": right.confidence,
        "n_left": left.n_pixels, "n_right": right.n_pixels,
        "side_split_x_used": split_used,
        "status": {"left": left.status, "right": right.status},
        "components": components,
        "lane_area_left_pts": _sample_edge(lane_left, top_cut, num_points),
        "lane_area_right_pts": _sample_edge(lane_right, top_cut, num_points),
        "geometry_reject_reason": geom_reason,
    }
