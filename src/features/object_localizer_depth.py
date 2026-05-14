"""Depth estimation and bbox-foot helpers for monocular object localization."""
from __future__ import annotations

import math

from features.object_localizer_priors import (
    ObjectPrior,
    _MAX_DEPTH_M,
    _MIN_CONFIDENCE,
    _MIN_DEPTH_M,
)


def select_foot_u(x1: float, x2: float, cx: float) -> tuple[float, str]:
    """Closest point on the bbox bottom edge to the camera forward axis."""
    if x1 <= cx <= x2:
        return cx, "bottom_axis_intersection"
    if x2 < cx:
        return x2, "bottom_right_corner"
    return x1, "bottom_left_corner"


def bbox_intersects_center_gate(
    x1: float, x2: float, cx: float, frame_w: int, center_gate_ratio: float,
) -> bool:
    center_gate_px = center_gate_ratio * float(frame_w)
    return x2 >= cx - center_gate_px and x1 <= cx + center_gate_px


def depth_estimates(
    label: str,
    prior: ObjectPrior,
    bbox_w: float,
    bbox_h: float,
    fy: float,
    clipped: bool,
    bottom_y: float,
    frame_h: int,
    size_focal_px: float | None,
) -> dict[str, tuple[float, float]]:
    estimates: dict[str, tuple[float, float]] = {}
    focal = size_focal_px if size_focal_px is not None else fy
    z_height = prior.height_m * focal / bbox_h
    confidence = 0.55 if not clipped else 0.35
    if label == "motorcycle":
        confidence *= 0.85
    estimates["size_height"] = (z_height, confidence)
    bottom_ratio = bottom_y / max(float(frame_h), 1.0)
    if bottom_ratio >= 0.78:
        t = min(max((bottom_ratio - 0.78) / (0.95 - 0.78), 0.0), 1.0)
        # Heuristic cap: lower bbox foot means closer to the camera.
        # 78% frame height -> <= 8m, 95%+ -> <= 0.6m.
        bottom_cap = 8.0 * (1.0 - t) + 0.6 * t
        estimates["bottom_cap"] = (max(_MIN_DEPTH_M, bottom_cap), 0.75)
    return estimates


def fuse_depth(estimates: dict[str, tuple[float, float]]) -> tuple[float, float, str]:
    candidates = {
        name: (float(z), float(w))
        for name, (z, w) in estimates.items()
        if name not in {"prior_min", "prior_max"} and math.isfinite(z) and z > 0.0
    }
    if not candidates:
        return -1.0, 0.0, "none"

    z, confidence = candidates.get("size_height", next(iter(candidates.values())))
    source = "size_height"
    bottom_cap = candidates.get("bottom_cap")
    if bottom_cap is not None and bottom_cap[0] < z:
        z, confidence = bottom_cap
        source = "bottom_cap"
    z = min(max(z, _MIN_DEPTH_M), _MAX_DEPTH_M)
    confidence = max(_MIN_CONFIDENCE, min(0.95, confidence))
    return z, confidence, source
