"""Render-corridor builder - purple + ego-anchored curved corridor.

Single-frame (with a small frame-to-frame curve EMA stored in a
module-level singleton; see ``reset_render_corridor_state``).

* ``boundary_track`` - dict from ``LaneBoundaryTracker``: per-side
  ``*_coeffs``, ``*_source``, ``lane_state``. Only ``SOURCE_DETECTED``
  sides drive purple/red/yellow shape.
* ``red_center``     - accepted for back-compat but unused; purple
  coefficients drive red's shape directly.

Returns a dict shaped to drop into ``draw_fixed_guidance`` (the
``fixed_corridor_*`` and ``fixed_centerline_pts_img`` keys), plus the
``purple_left_pts_img`` / ``purple_right_pts_img`` polylines.

Activation rule:

* At least one purple side ``SOURCE_DETECTED``  → corridor activates
* Neither side detected                         → ``valid=False`` and
  the caller falls back to ``fixed_guidance_template``.

Missing purple sides are **not synthesized**: only the detected side(s)
populate ``purple_*_pts_img``; the other stays empty so the boundary
renderer simply does not draw it.

Red/yellow geometry:

* Red anchors at ``ego_x = frame_w / 2`` at the bottom ROI row with
  zero slope there. Curvature comes from the detected side(s)' ``a``
  (averaged when both detected) after damping, EMA, deadband, and a
  top-displacement clamp. Purple's slope ``b`` is ignored.
* Yellow is offset symmetrically along red's normal with a fixed
  perspective half-width profile (frame-size driven). For a straight
  red this reduces to a horizontal offset.
"""
from __future__ import annotations

import numpy as np

from features.lane_boundary_tracker_contract import LANE_LOST, SOURCE_LOST
from features.lane_corridor_render_constants import (
    DEFAULT_ASSUMED_LANE_WIDTH_M,
    DEFAULT_EGO_VEHICLE_WIDTH_M,
    DEFAULT_FIXED_YELLOW_HALF_WIDTH_FAR_FRAC,
    DEFAULT_FIXED_YELLOW_HALF_WIDTH_NEAR_FRAC,
    DEFAULT_MAX_LANE_WIDTH_FRAC_W,
    DEFAULT_MIN_INNER_GAP_PX,
    DEFAULT_MIN_LANE_WIDTH_PX,
    DEFAULT_NUM_POINTS,
    DEFAULT_ROI_BOTTOM_FRAC,
    DEFAULT_ROI_TOP_FRAC,
    DEFAULT_TARGET_SMOOTH_TAPS,
    DEFAULT_VEHICLE_LANE_WIDTH_RATIO_MAX,
    DEFAULT_VEHICLE_LANE_WIDTH_RATIO_MIN,
    DEFAULT_YELLOW_PURPLE_MARGIN_PX,
    RENDER_SRC_LANE_AREA_BOOTSTRAP,
    RENDER_SRC_PURPLE,
)
from features.lane_corridor_render_curve import (
    CorridorCurveState, compute_red_xs_from_purple, red_dxdy_per_y,
)
from features.lane_corridor_render_purple import resolve_purple_xs
from features.lane_corridor_render_yellow import yellow_corridor_from_center
from features.ego_anchored_path import (
    build_ego_anchored_path, ego_path_dxdy_per_y,
)

# Singleton state for frame-to-frame curve delta gating. The pipeline
# resets this together with the lane tracker so a fresh video does not
# inherit the previous run's accepted curve.
_CURVE_STATE = CorridorCurveState()


def reset_render_corridor_state() -> None:
    """Reset the frame-to-frame corridor curve gate state."""
    _CURVE_STATE.reset()


__all__ = [
    "build_render_corridor",
    "reset_render_corridor_state",
    "DEFAULT_ROI_TOP_FRAC", "DEFAULT_ROI_BOTTOM_FRAC", "DEFAULT_NUM_POINTS",
    "DEFAULT_EGO_VEHICLE_WIDTH_M",
    "DEFAULT_ASSUMED_LANE_WIDTH_M",
    "DEFAULT_VEHICLE_LANE_WIDTH_RATIO_MIN",
    "DEFAULT_VEHICLE_LANE_WIDTH_RATIO_MAX",
    "DEFAULT_FIXED_YELLOW_HALF_WIDTH_FAR_FRAC",
    "DEFAULT_FIXED_YELLOW_HALF_WIDTH_NEAR_FRAC",
    "DEFAULT_YELLOW_PURPLE_MARGIN_PX",
    "DEFAULT_TARGET_SMOOTH_TAPS",
    "DEFAULT_MIN_INNER_GAP_PX", "DEFAULT_MAX_LANE_WIDTH_FRAC_W",
    "RENDER_SRC_PURPLE", "RENDER_SRC_LANE_AREA_BOOTSTRAP",
]


def _empty_pts() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def _ego_red_xs(
    red_center: dict | None,
    ys: np.ndarray,
    frame_h: int,
    frame_w: int,
    *,
    provisional: bool,
) -> tuple[np.ndarray, dict, bool]:
    """Fit the ego-anchored red xs from the fused red center samples.

    Returns ``(xs, diag, used)``. ``used=False`` when red_center is missing,
    invalid, or provisional — the caller then falls back to the legacy
    purple-curvature path so a transient red-center dropout doesn't blank
    the displayed corridor.
    """
    if provisional or not isinstance(red_center, dict):
        return np.zeros((0,), dtype=np.float32), {}, False
    if not red_center.get("valid"):
        return np.zeros((0,), dtype=np.float32), {}, False
    target = red_center.get("center_pts_img")
    if target is None or len(target) < 2:
        return np.zeros((0,), dtype=np.float32), {}, False
    xs, diag = build_ego_anchored_path(target, ys, frame_h, frame_w)
    return xs, diag, True


def _empty_result(reason: str, lane_state: str) -> dict:
    e = _empty_pts()
    return {
        "fixed_corridor_left_pts_img": e, "fixed_corridor_right_pts_img": e,
        "fixed_corridor_polygon_img": e, "fixed_centerline_pts_img": e,
        "purple_left_pts_img": e, "purple_right_pts_img": e,
        "yellow_left_pts_img": e, "yellow_right_pts_img": e,
        "purple_left_source": SOURCE_LOST, "purple_right_source": SOURCE_LOST,
        "render_corridor_source": RENDER_SRC_PURPLE, "provisional": False,
        "lane_state": lane_state, "is_render_corridor": True, "valid": False,
        "reason": reason,
    }


def build_render_corridor(
    boundary_track: dict | None,
    frame_h: int,
    frame_w: int,
    *,
    red_center: dict | None = None,
    curve_state: CorridorCurveState | None = None,
    roi_top_frac: float = DEFAULT_ROI_TOP_FRAC,
    roi_bottom_frac: float = DEFAULT_ROI_BOTTOM_FRAC,
    num_points: int = DEFAULT_NUM_POINTS,
    ego_vehicle_width_m: float = DEFAULT_EGO_VEHICLE_WIDTH_M,
    assumed_lane_width_m: float = DEFAULT_ASSUMED_LANE_WIDTH_M,
    vehicle_lane_ratio_min: float = DEFAULT_VEHICLE_LANE_WIDTH_RATIO_MIN,
    vehicle_lane_ratio_max: float = DEFAULT_VEHICLE_LANE_WIDTH_RATIO_MAX,
    fixed_yellow_half_width_far_frac: float = DEFAULT_FIXED_YELLOW_HALF_WIDTH_FAR_FRAC,
    fixed_yellow_half_width_near_frac: float = DEFAULT_FIXED_YELLOW_HALF_WIDTH_NEAR_FRAC,
    yellow_purple_margin_px: float = DEFAULT_YELLOW_PURPLE_MARGIN_PX,
    target_smooth_taps: int = DEFAULT_TARGET_SMOOTH_TAPS,
    min_inner_gap_px: float = DEFAULT_MIN_INNER_GAP_PX,
    min_lane_width_px: float = DEFAULT_MIN_LANE_WIDTH_PX,
    max_lane_width_frac_w: float = DEFAULT_MAX_LANE_WIDTH_FRAC_W,
) -> dict:
    """Build purple + yellow + centerline polylines through a fixed lower ROI.

    The output dict is drop-in for ``draw_fixed_guidance`` (it carries the
    legacy ``fixed_corridor_*`` and ``fixed_centerline_pts_img`` keys).
    """
    fh, fw = int(frame_h), int(frame_w)
    lane_state = (boundary_track or {}).get("lane_state", LANE_LOST)
    if not isinstance(boundary_track, dict):
        return _empty_result("no boundary track", lane_state)

    y_top = max(0.0, fh * float(roi_top_frac))
    y_bot = min(float(fh - 1), fh * float(roi_bottom_frac) - 1.0)
    if y_bot <= y_top:
        y_top, y_bot = max(0.0, fh - 2.0), float(fh - 1)
    ys = np.linspace(y_top, y_bot, num=int(num_points), dtype=np.float32)

    left_xs, right_xs, l_render, r_render, provisional, reason = resolve_purple_xs(
        boundary_track, ys, fw,
        min_lane_width_px=min_lane_width_px,
        max_lane_width_frac_w=max_lane_width_frac_w,
    )
    # Corridor activates as soon as ONE purple side is model-detected;
    # the other side simply won't be drawn (no synthesis).
    if left_xs is None and right_xs is None:
        return _empty_result(reason, lane_state)

    # Red centerline: ego-anchored at the bottom (x = ego_x, dx/dy = 0).
    # Preferred shape source is the fused red center (lane_area + tracked
    # purple), which lets the head bend toward where the lane actually
    # sits even when purple has no measurable curvature (e.g. straight
    # road with lateral offset). When red_center is unusable we fall back
    # to the legacy purple-curvature path so the corridor never goes dark.
    ego_cxs, ego_diag, ego_used = _ego_red_xs(
        red_center, ys, fh, fw, provisional=bool(provisional),
    )
    if ego_used:
        cxs = ego_cxs
        anchor_y = float(ego_diag["ego_path_anchor_y"])
        aC = float(ego_diag["ego_path_a"])
        red_dxdy = ego_path_dxdy_per_y(ys, aC, anchor_y)
        shape_src = "red_center_ego_anchored"
        shape_reason = ego_diag.get("ego_path_reason") or "ok"
        curve_diag = {
            "corridor_curve_applied": aC != 0.0,
            "corridor_curve_aC": aC if aC != 0.0 else None,
            "corridor_curve_top_disp_px": float(cxs[0] - 0.5 * fw) if cxs.size else None,
            "corridor_curve_top_disp_clamped": bool(
                ego_diag.get("ego_path_top_disp_clamped", False)),
            "corridor_curve_reject_reason": (
                None if aC != 0.0 else (ego_diag.get("ego_path_reason") or "below_deadband")),
        }
    else:
        cxs, curve_diag = compute_red_xs_from_purple(
            boundary_track, ys, fw,
            state=curve_state if curve_state is not None else _CURVE_STATE,
            provisional=bool(provisional),
        )
        aC = curve_diag.get("corridor_curve_aC") if curve_diag.get(
            "corridor_curve_applied") else None
        anchor_y = curve_diag.get("corridor_curve_anchor_y") if curve_diag.get(
            "corridor_curve_applied") else None
        red_dxdy = red_dxdy_per_y(ys, aC, anchor_y)
        shape_src = "purple_curve" if curve_diag.get(
            "corridor_curve_applied") else "fixed_center"
        shape_reason = curve_diag.get("corridor_curve_reject_reason") or "ok"

    # Kept for API/back-compat diagnostics only. Yellow width no longer
    # depends on purple lane width or live lane-area width.
    lane_m = max(float(assumed_lane_width_m), 1e-3)
    raw_ratio = float(ego_vehicle_width_m) / lane_m
    ratio = float(np.clip(raw_ratio,
                          float(vehicle_lane_ratio_min),
                          float(vehicle_lane_ratio_max)))
    _ = yellow_purple_margin_px

    yl, yr, realized_near_px, realized_far_px = yellow_corridor_from_center(
        left_xs, right_xs, cxs, fw,
        fixed_half_width_far_frac=float(fixed_yellow_half_width_far_frac),
        fixed_half_width_near_frac=float(fixed_yellow_half_width_near_frac),
        min_inner_gap_px=float(min_inner_gap_px),
        smooth_taps=int(target_smooth_taps),
        red_dxdy=red_dxdy,
    )

    # Missing-side purples stay empty — only the model-detected side is
    # filled. The renderer / boundary draw filters by source, so an
    # empty polyline simply doesn't draw.
    purple_left = (np.stack([left_xs, ys], axis=1).astype(np.float32)
                   if left_xs is not None else _empty_pts())
    purple_right = (np.stack([right_xs, ys], axis=1).astype(np.float32)
                    if right_xs is not None else _empty_pts())
    yellow_left = np.stack([yl, ys], axis=1).astype(np.float32)
    yellow_right = np.stack([yr, ys], axis=1).astype(np.float32)
    centerline = np.stack([cxs, ys], axis=1).astype(np.float32)
    polygon = np.vstack([yellow_left, yellow_right[::-1]]).astype(np.float32)

    return {
        # Drop-in for draw_fixed_guidance:
        "fixed_corridor_left_pts_img": yellow_left,
        "fixed_corridor_right_pts_img": yellow_right,
        "fixed_corridor_polygon_img": polygon,
        "fixed_centerline_pts_img": centerline,
        # Extra fields for purple rendering and diagnostics:
        "purple_left_pts_img": purple_left,
        "purple_right_pts_img": purple_right,
        "yellow_left_pts_img": yellow_left,
        "yellow_right_pts_img": yellow_right,
        "ego_vehicle_width_m": float(ego_vehicle_width_m),
        "assumed_lane_width_m": float(assumed_lane_width_m),
        "vehicle_lane_width_ratio": float(ratio),
        "fixed_yellow_half_width_far_frac": float(fixed_yellow_half_width_far_frac),
        "fixed_yellow_half_width_near_frac": float(fixed_yellow_half_width_near_frac),
        "yellow_width_near_px": float(realized_near_px),
        "yellow_width_far_px": float(realized_far_px),
        "ego_path_shape_source": shape_src,
        "ego_path_shape_reason": shape_reason,
        "purple_left_source": l_render,
        "purple_right_source": r_render,
        # ``provisional`` is always False since lane-area-bootstrap
        # synthesis is gone; kept on the output dict for back-compat.
        "render_corridor_source": RENDER_SRC_PURPLE,
        "provisional": False,
        "lane_state": lane_state,
        "is_render_corridor": True,
        "valid": True,
        "reason": "ok",
        "corridor_curve_applied": bool(curve_diag.get("corridor_curve_applied")),
        "corridor_curve_aC": curve_diag.get("corridor_curve_aC"),
        "corridor_curve_top_disp_px": curve_diag.get("corridor_curve_top_disp_px"),
        "corridor_curve_top_disp_clamped": bool(
            curve_diag.get("corridor_curve_top_disp_clamped")),
        "corridor_curve_reject_reason": curve_diag.get(
            "corridor_curve_reject_reason"),
    }
