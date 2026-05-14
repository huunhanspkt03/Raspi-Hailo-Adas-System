"""Ego-lane geometry — fixed 1.8 m guidance lane built around the ego.

The ego-lane is a *virtual* guidance corridor:

  * width is fixed at 1.8 m in BEV pixels
  * lateral position is anchored to the ego/camera reference column
    (the BEV column directly below the camera) at the bottom row
  * curvature/orientation is taken from the polynomial fit through
    detected lane-line geometry — the detected lines do **not** become
    the rendered ego-lane edges, they only shape the ego-lane's curve.

Two small primitives live here:

  * ``ego_centered_polynomial(shape_fit, y_anchor, ego_x_at_anchor)``
    DC-shifts a shape polynomial so it passes through ``ego_x_at_anchor``
    at ``y_anchor``. Slope at every y is unchanged (only the constant
    term is rewritten), so curvature/orientation cues from the detected
    lines are preserved exactly while the lane re-anchors to the ego.

  * ``ego_lane_from_center(bev_ys, ego_center_xs, *, bev_w)`` builds
    rendered left/right at ±0.9 m around an already-ego-centered
    centerline. Used by both the per-frame estimator (after the DC
    shift) and the tracker prediction path (whose stored polynomial
    is ego-centered by construction).

Width is enforced via ``BEV_PX_PER_METER`` once at module level so
overlays, tracker, and tests all read the same metric scale.
"""
from __future__ import annotations

import cv2
import numpy as np

LANE_WIDTH_M = 1.8

# Pixels per meter on the BEV produced by ``_build_bev_transform``.
# The BEV trapezoid bottom (~0.9 * img_w in image space) maps to the
# full ``_BEV_W`` (320 px) and is assumed to span ~4 m of road in
# front of the car, giving 80 BEV px/m. Update once a real camera-to-
# ground calibration exists; everything else here is exact geometry.
BEV_PX_PER_METER = 80.0
LANE_HALF_WIDTH_BEV = 0.5 * LANE_WIDTH_M * BEV_PX_PER_METER  # 72 px


def ego_centered_polynomial(
    shape_fit: np.ndarray,
    y_anchor: float,
    ego_x_at_anchor: float,
) -> np.ndarray:
    """Return ``shape_fit`` DC-shifted to pass through the ego column.

    Slope at every y is preserved — only the constant term is rewritten.
    The detected line's curvature/orientation therefore drives the
    ego-lane's bend, while the rendered lane stays anchored to the ego
    reference column at ``y_anchor``.
    """
    coeffs = np.asarray(shape_fit, dtype=np.float32).copy()
    if coeffs.size == 0:
        return coeffs
    shift = float(ego_x_at_anchor) - float(np.polyval(coeffs, float(y_anchor)))
    coeffs[-1] = coeffs[-1] + shift
    return coeffs


def ego_lane_from_center(
    bev_ys: np.ndarray,
    ego_center_xs: np.ndarray,
    *,
    bev_w: int,
) -> dict:
    """Build the 1.8 m rendered ego-lane around an ego-centered centerline."""
    bev_ys = np.asarray(bev_ys, dtype=np.float32)
    center = np.asarray(ego_center_xs, dtype=np.float32)
    left_xs = np.clip(center - LANE_HALF_WIDTH_BEV, 0.0, bev_w - 1.0)
    right_xs = np.clip(center + LANE_HALF_WIDTH_BEV, 0.0, bev_w - 1.0)
    return {
        "bev_ys": bev_ys,
        "bev_left_xs": left_xs.astype(np.float32, copy=False),
        "bev_right_xs": right_xs.astype(np.float32, copy=False),
        "bev_center_xs": center.astype(np.float32, copy=False),
    }


def project_ego_lane_to_image(
    ego: dict,
    Minv: np.ndarray,
    img_h: int,
    img_w: int,
) -> dict:
    """Project the BEV ego-lane to image-space polylines."""
    ys = ego["bev_ys"]
    return {
        "left_pts_img": _project(ego["bev_left_xs"], ys, Minv, img_h, img_w),
        "right_pts_img": _project(ego["bev_right_xs"], ys, Minv, img_h, img_w),
        "center_pts_img": _project(ego["bev_center_xs"], ys, Minv, img_h, img_w),
    }


def _project(
    xs: np.ndarray, ys: np.ndarray, Minv: np.ndarray, img_h: int, img_w: int,
) -> np.ndarray:
    pts = np.column_stack([xs, ys]).reshape(1, -1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, Minv)[0]
    out[:, 0] = np.clip(out[:, 0], 0, img_w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, img_h - 1)
    return out
