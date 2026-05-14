"""Camera-anchored guidance template.

A single builder, ``fixed_guidance_template``, produces a stable
perspective trapezoid that depends only on ``(frame_h, frame_w)`` and
a few tunable constants. The red centerline + yellow corridor + closed
polygon are all derived from this template; nothing from the lane
mask, semantic markings, or any per-frame perception input shapes the
displayed guidance geometry.

Path-relevance helpers (``mark_in_path``, ``reclassify_detections``,
``point_in_corridor``, ``bbox_intersects_corridor``) live in
``features.path_relevance`` and are re-exported here for back-compat.
"""
from __future__ import annotations

import numpy as np

from features.path_relevance import (  # re-exported for back-compat
    bbox_intersects_corridor,
    mark_in_path,
    point_in_corridor,
    reclassify_detections,
)

__all__ = [
    "fixed_guidance_template",
    "mark_in_path",
    "reclassify_detections",
    "point_in_corridor",
    "bbox_intersects_corridor",
    "DEFAULT_GUIDANCE_TOP_FRAC",
    "DEFAULT_GUIDANCE_BOTTOM_FRAC",
    "DEFAULT_HALF_WIDTH_FAR_FRAC",
    "DEFAULT_HALF_WIDTH_NEAR_FRAC",
    "DEFAULT_CENTER_FRAC",
]

# Perspective trapezoid in image space, anchored to the camera view.
# ``DEFAULT_GUIDANCE_TOP_FRAC`` controls where the corridor's top
# endpoint sits as a fraction of frame height (0 = top of frame, 1 =
# bottom). Higher values pull the top of the overlay lower in the
# image, keeping it on the near/mid road surface and out of the upper
# background.
DEFAULT_GUIDANCE_TOP_FRAC = 0.65
DEFAULT_GUIDANCE_BOTTOM_FRAC = 1.0
DEFAULT_HALF_WIDTH_FAR_FRAC = 0.04
DEFAULT_HALF_WIDTH_NEAR_FRAC = 0.24
DEFAULT_CENTER_FRAC = 0.5
_NUM_POINTS = 24
_MIN_HALF_WIDTH_PX = 12


def fixed_guidance_template(
    frame_h: int,
    frame_w: int,
    *,
    top_frac: float = DEFAULT_GUIDANCE_TOP_FRAC,
    bottom_frac: float = DEFAULT_GUIDANCE_BOTTOM_FRAC,
    half_width_far_frac: float = DEFAULT_HALF_WIDTH_FAR_FRAC,
    half_width_near_frac: float = DEFAULT_HALF_WIDTH_NEAR_FRAC,
    center_frac: float = DEFAULT_CENTER_FRAC,
) -> dict[str, np.ndarray]:
    """Camera-referenced perspective trapezoid + vertical centerline.

    Width grows linearly with y so the corridor is a clean perspective
    trapezoid: narrow in the far field, wide near the bumper. Output
    dict keys: ``fixed_centerline_pts_img``,
    ``fixed_corridor_left_pts_img``, ``fixed_corridor_right_pts_img``,
    ``fixed_corridor_polygon_img``.
    """
    cx = float(frame_w) * center_frac
    y_top = max(0.0, float(frame_h) * top_frac)
    y_bot = min(float(frame_h - 1), float(frame_h) * bottom_frac - 1.0)
    if y_bot <= y_top:
        y_bot = float(frame_h - 1)
        y_top = max(0.0, y_bot - 1.0)
    ys = np.linspace(y_top, y_bot, num=_NUM_POINTS, dtype=np.float32)
    far = max(half_width_far_frac * float(frame_w), _MIN_HALF_WIDTH_PX)
    near = max(half_width_near_frac * float(frame_w), _MIN_HALF_WIDTH_PX)
    t = (ys - y_top) / max(y_bot - y_top, 1.0)
    half_widths = (far + (near - far) * t).astype(np.float32)
    cxs = np.full_like(ys, cx, dtype=np.float32)
    centerline = np.stack([cxs, ys], axis=1)
    left_x = np.clip(cxs - half_widths, 0.0, float(frame_w - 1))
    right_x = np.clip(cxs + half_widths, 0.0, float(frame_w - 1))
    left = np.stack([left_x, ys], axis=1)
    right = np.stack([right_x, ys], axis=1)
    polygon = np.vstack([left, right[::-1]]).astype(np.float32)
    return {
        "fixed_centerline_pts_img": centerline,
        "fixed_corridor_left_pts_img": left,
        "fixed_corridor_right_pts_img": right,
        "fixed_corridor_polygon_img": polygon,
    }
