"""Step-4 red target center line.

Stateless single-frame fusion. Inputs:

* ``lane_area``      - H*W binary mask (drivable surface from YOLO26-seg).
* ``boundary_track`` - dict from ``LaneBoundaryTracker`` (per-side source,
  ``*_pts_img``, ``*_confidence_final``, ``lane_state``, ``boundary_confidence``).

Source policy (descending priority):

1. **Purple is primary.** When both purple boundaries are freshly DETECTED,
   interpolable on the ROI grid, and not crossing, the red center is the
   per-row midpoint of the purple polylines. ``lane_area`` is *not* fused
   in and does *not* clip the result - a wrong / too-narrow ``lane_area``
   cannot pull the red center off the purple corridor midpoint anymore.
2. **Lane_area fallback.** When at most one purple side is usable (or the
   purple sides cross), the red center is the lane_area row-centroid fit,
   clipped to lane_area rows for safety.
3. **LOST.** Neither signal is usable.

The green safe corridor (Step 5) is intentionally NOT computed here.
Pure helpers live in ``red_center_geom``.
"""
from __future__ import annotations

import numpy as np

from features.lane_boundary_tracker_contract import LANE_LOST, SOURCE_DETECTED
from features.red_center_geom import (
    ROAD_CURVE_LEFT, ROAD_CURVE_RIGHT, ROAD_STRAIGHT, ROAD_UNKNOWN,
    classify_road, clip_between, clip_to_lane_area,
    lane_area_evidence, purple_midpoint, purple_sides_separable,
)
from features.red_center_source import (
    PURPLE_GEOM_SOURCES,
    SRC_HELD_PREVIOUS, SRC_LANE_AREA_ASSISTED, SRC_LANE_AREA_ONLY, SRC_LOST,
    SRC_PURPLE_ONLY, SRC_PURPLE_PRIMARY, SRC_PURPLE_TRACKED,
    classify_purple_center_source, empty_red_center_result,
)

# ---- Defaults ------------------------------------------------------------
DEFAULT_TOP_FRAC = 0.55
DEFAULT_NUM_POINTS = 24
DEFAULT_MIN_LANE_AREA_PIXELS = 600
DEFAULT_DETECTED_WEIGHT = 1.0
DEFAULT_INFERRED_WEIGHT = 0.55
DEFAULT_HELD_WEIGHT = 0.25
DEFAULT_MAX_JUMP_FRAC_W = 0.10
DEFAULT_CURVE_A_THRESH = 5e-5

__all__ = [
    "compute_red_centerline",
    "SRC_LANE_AREA_ONLY", "SRC_PURPLE_ONLY", "SRC_PURPLE_PRIMARY",
    "SRC_PURPLE_TRACKED", "SRC_LANE_AREA_ASSISTED",
    "SRC_HELD_PREVIOUS", "SRC_LOST",
    "ROAD_STRAIGHT", "ROAD_CURVE_LEFT", "ROAD_CURVE_RIGHT", "ROAD_UNKNOWN",
    "DEFAULT_TOP_FRAC", "DEFAULT_NUM_POINTS",
]


def compute_red_centerline(
    lane_area,
    boundary_track,
    frame_h,
    frame_w,
    *,
    top_frac=DEFAULT_TOP_FRAC,
    num_points=DEFAULT_NUM_POINTS,
    min_lane_area_pixels=DEFAULT_MIN_LANE_AREA_PIXELS,
    detected_weight=DEFAULT_DETECTED_WEIGHT,
    inferred_weight=DEFAULT_INFERRED_WEIGHT,
    held_weight=DEFAULT_HELD_WEIGHT,
    max_jump_frac_w=DEFAULT_MAX_JUMP_FRAC_W,
    curve_a_thresh=DEFAULT_CURVE_A_THRESH,
) -> dict:
    """Build the red center using purple midpoint as primary; fall back to
    lane_area when purple is not usable. See module docstring for policy."""
    _ = inferred_weight, held_weight
    src_w = {SOURCE_DETECTED: float(detected_weight)}
    fh, fw = int(frame_h), int(frame_w)
    lane_state = (boundary_track or {}).get("lane_state", LANE_LOST)
    ys = np.linspace(max(0.0, fh * top_frac), float(fh - 1),
                     num=int(num_points), dtype=np.float32)

    area_xs, area_conf, _ = lane_area_evidence(
        lane_area, fh, fw, ys, top_frac=top_frac,
        num_points=num_points, min_pixels=min_lane_area_pixels)
    purp_xs, purp_conf, _l, _r, lx, rx = purple_midpoint(
        boundary_track, ys, src_w)
    purple_usable = (purp_xs is not None
                     and purple_sides_separable(lx, rx, min_gap_px=1.0))
    have_a = area_xs is not None

    out = empty_red_center_result("no evidence", lane_state=lane_state)
    out["lane_area_confidence"] = float(area_conf) if have_a else 0.0
    out["purple_confidence"] = float(purp_conf) if purp_xs is not None else 0.0
    if have_a:
        out["lane_area_center_pts_img"] = np.stack(
            [area_xs, ys], axis=1).astype(np.float32)
    if purp_xs is not None:
        out["purple_center_pts_img"] = np.stack(
            [purp_xs, ys], axis=1).astype(np.float32)

    # Pick the primary signal. Purple beats lane_area only when both sides
    # are current DETECTED model outputs. HELD and legacy INFERRED sides
    # are ignored by ``purple_midpoint`` so synthetic/stale boundaries do
    # not pull the red path.
    if purple_usable:
        fused = purp_xs.copy()
        center_source = classify_purple_center_source(
            boundary_track, have_lane_area=have_a)
        valid = np.ones_like(ys, dtype=bool)
    elif have_a:
        fused = area_xs.copy()
        center_source = SRC_LANE_AREA_ONLY
        if lane_area is not None:
            fused, valid = clip_to_lane_area(fused, ys, lane_area)
        else:
            valid = np.ones_like(ys, dtype=bool)
    else:
        out["reason"] = "lane_area unusable and purple LOST"
        return out

    if int(np.sum(valid)) < 6:
        out["reason"] = "insufficient valid samples for fit"
        return out

    try:
        coeffs = np.polyfit(ys[valid].astype(np.float64),
                            fused[valid].astype(np.float64), 2)
    except (np.linalg.LinAlgError, ValueError):
        out["reason"] = "polyfit failed"
        return out
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    final_xs = (a * ys * ys + b * ys + c).astype(np.float32)

    # Safety: when current DETECTED purple drives the geometry, clamp the
    # smoothed centerline to sit strictly between the purple polylines on
    # every row of the ROI grid.
    if center_source in PURPLE_GEOM_SOURCES \
            and lx is not None and rx is not None:
        final_xs = clip_between(final_xs, lx, rx, eps_px=1.0)

    jumps = np.abs(np.diff(final_xs))
    jump_penalty = (0.5 if (jumps.size and float(np.max(jumps)) > max_jump_frac_w * fw)
                    else 1.0)
    if center_source in PURPLE_GEOM_SOURCES:
        center_conf = float(purp_conf)
    else:
        center_conf = float(area_conf)
    center_conf = float(np.clip(center_conf * jump_penalty, 0.0, 1.0))

    out["center_coeffs"] = (a, b, c)
    out["center_pts_img"] = np.stack([final_xs, ys], axis=1).astype(np.float32)
    out["center_confidence"] = center_conf
    out["center_source"] = center_source
    out["road_type"] = classify_road((a, b, c), curve_a_thresh)
    out["valid"] = True
    out["reason"] = "ok"
    return out
