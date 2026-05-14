"""Road-object (non-traffic-light) pose estimation."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from features.lane_corridor import mark_in_path
from features.object_localizer_depth import (
    bbox_intersects_center_gate,
    depth_estimates,
    fuse_depth,
    select_foot_u,
)
from features.object_localizer_priors import ObjectPrior, _CLIP_MARGIN_PX

if TYPE_CHECKING:
    from features.object_localizer import ObjectLocalizer


def estimate_road_object_pose(
    loc: "ObjectLocalizer",
    det: dict,
    prior: ObjectPrior,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    bbox_w: float,
    bbox_h: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    frame_w: int,
    frame_h: int,
    corridor_polygon: np.ndarray | None,
) -> bool:
    label = det.get("label", "")
    foot_u, foot_source = select_foot_u(x1, x2, cx)
    det["foot_x"] = int(round(foot_u))
    det["foot_source"] = foot_source
    v = float(det["bottom_center_y"])
    clipped = det["bottom_center_y"] >= frame_h - _CLIP_MARGIN_PX
    if clipped:
        fill = bbox_h / max(float(frame_h), 1.0)
        v += fill * bbox_h * 1.5
    det["foot_y"] = int(round(v))

    estimates = depth_estimates(
        label, prior, bbox_w, bbox_h, fy, clipped, y2, frame_h, loc.size_focal_px,
    )
    z, confidence, source = fuse_depth(estimates)
    if not math.isfinite(z) or z <= 0.0:
        return False

    u = float(det["foot_x"])
    x_m = ((u - cx) / max(fx, 1e-3)) * z
    z, x_m, confidence = loc._smooth_pose(det, z, x_m, confidence)

    det["pose_valid"] = True
    det["x_m"] = float(x_m)
    det["z_m"] = float(z)
    det["distance_m"] = float(math.hypot(x_m, z))
    det["ground_distance_m"] = float(z)
    det["distance_display_valid"] = True
    det["bearing_deg"] = float(math.degrees(math.atan2(x_m, z)))
    det["distance_confidence"] = float(confidence)
    det["distance_source"] = source
    det["distance_raw_m"] = {
        name: float(value)
        for name, (value, _) in estimates.items()
        if math.isfinite(value) and value > 0.0
    }
    if corridor_polygon is not None:
        mark_in_path(
            det, corridor_polygon,
            frame_w=frame_w, frame_h=frame_h,
            fallback_center_gate_ratio=loc.center_gate_ratio,
            min_bottom_ratio=loc.min_bottom_ratio,
        )
        if det["in_path"] and confidence < 0.20:
            det["in_path"] = False
    else:
        det["in_path"] = (
            bbox_intersects_center_gate(x1, x2, cx, frame_w, loc.center_gate_ratio)
            and v >= loc.min_bottom_ratio * float(frame_h)
            and confidence >= 0.20
        )
    return True
