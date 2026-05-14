"""Lane + object worker classes for the threaded runtime.

Each worker reuses the existing perception helpers and owns its own
state. Inference is guarded by a shared lock because the Hailo / ONNX
backends are not safe to re-enter from multiple Python threads.
"""
from __future__ import annotations

import time
from threading import Lock
from typing import Any

import numpy as np

from pipeline.packets import FramePacket, LanePacket, ObjectPacket
from pipeline.preprocessor import preprocess_segmentation, preprocess_yolo26
from features.lane_corridor import fixed_guidance_template
from features.dynamic_corridor_tracker import DynamicCorridorTracker
from features.lane_boundary_tracker import LaneBoundaryTracker
from features.lane_postprocess import decode_lane_components
from features.lane_path import mask_centerline
from features.object_detection import decode_yolo26
from features.object_localizer import ObjectLocalizer
from features.object_tracker import ByteTracker
from features.red_center_from_boundaries import compute_red_centerline
from features.ego_anchored_path import build_ego_anchored_path
from features.state_types import LaneState


class LaneWorker:
    """Consumes FramePacket, emits LanePacket."""

    def __init__(
        self,
        infer: Any,
        infer_lock: Lock,
        *,
        segmentation_size: int | tuple[int, int],
        min_lane_w: int = 0,
        steering_source: str = "lane_marking_center",
    ) -> None:
        self._infer = infer
        self._lock = infer_lock
        self._seg_size = segmentation_size
        # min_lane_w / steering_source kept for call-site compatibility with
        # the legacy stack but are not used by the mask-centerline path.
        self._min_lane_w = min_lane_w
        self._steering_source = steering_source
        self._corridor = DynamicCorridorTracker()
        self._boundaries = LaneBoundaryTracker()

    def process(self, pkt: FramePacket) -> LanePacket:
        prep = preprocess_segmentation(pkt.frame, self._seg_size)
        with self._lock:
            raw = self._infer.infer("lane_segmentation", prep)
        _, class_probs, semantics, lane_area = decode_lane_components(
            raw, pkt.frame_h, pkt.frame_w,
        )
        lane_path = mask_centerline(lane_area)
        boundary_track = self._boundaries.update(semantics, lane_area, pkt.frame_h, pkt.frame_w)
        red_center = compute_red_centerline(lane_area, boundary_track, pkt.frame_h, pkt.frame_w)
        lane_bounds = dict(boundary_track)
        lane_bounds["red_center"] = red_center
        lane_bounds["lane_area_path"] = lane_path
        red_path = _lane_path_from_red_center(red_center, pkt.frame_h, pkt.frame_w)
        if red_path is not None:
            lane_path = red_path
        if lane_path is not None:
            lane_path["display_lane_area_img"] = lane_area
            if red_center.get("valid"):
                lane_path["red_center"] = red_center
                lane_path["center_source"] = red_center.get("center_source")
                lane_path["road_type"] = red_center.get("road_type")
        corridor = self._corridor.update(lane_area, pkt.frame_h, pkt.frame_w)
        if corridor is not None and lane_path is not None and lane_path.get("is_red_center"):
            # Match the main-loop semantics: the displayed centerline is the
            # ego-anchored red path, not the raw lane-center samples.
            corridor = dict(corridor)
            corridor["fixed_centerline_pts_img"] = lane_path["fitted_pts_img"]
        lane_state = LaneState.from_path_result(lane_path, None, 0.0)
        return LanePacket(
            frame_id=pkt.frame_id, timestamp=pkt.timestamp,
            lane_state=lane_state, source_frame=pkt.frame,
            lane_area=lane_area, semantics=semantics,
            class_probs=class_probs, lane_path=lane_path, lane_bounds=lane_bounds,
            corridor=corridor,
        )

    def set_prev_detections(self, dets: list[dict]) -> None:
        # Legacy hook; the mask-centerline path does not use detections.
        del dets

    def reset(self) -> None:
        self._corridor.reset()
        self._boundaries.reset()


def _lane_path_from_red_center(
    red_center: dict, frame_h: int, frame_w: int,
) -> dict | None:
    """Threaded-runtime twin of pipeline_loop._lane_path_from_red_center.

    Same ego-anchored quadratic + lookahead control point: the tail of
    ``fitted_pts_img`` is pinned at ``frame_w/2`` and the head bends
    toward the lane-center targets in ``red_center["center_pts_img"]``.
    """
    if not red_center.get("valid"):
        return None
    pts = red_center.get("center_pts_img")
    if pts is None or len(pts) == 0:
        return None
    ys = np.asarray(pts[:, 1], dtype=np.float32)
    xs_ego, diag = build_ego_anchored_path(pts, ys, frame_h, frame_w)
    ego_pts = np.stack([xs_ego, ys], axis=1).astype(np.float32)
    return {
        "is_red_center": True,
        "control_x": int(round(float(diag["ego_path_control_x"]))),
        "control_y": int(round(float(diag["ego_path_control_y"]))),
        "fitted_pts_img": ego_pts,
        "lane_target_pts_img": pts,
        "ego_path_a": float(diag["ego_path_a"]),
        "ego_path_anchor_y": float(diag["ego_path_anchor_y"]),
        "ego_path_lookahead_y": float(diag["ego_path_lookahead_y"]),
        "ego_path_top_disp_clamped": bool(diag.get("ego_path_top_disp_clamped", False)),
        "confidence": float(red_center.get("center_confidence", 0.0)),
        "poly_coeffs": tuple(red_center.get("center_coeffs") or ()),
        "row_count": int(len(ego_pts)),
    }


class ObjectWorker:
    """Consumes FramePacket, emits ObjectPacket."""

    def __init__(
        self,
        infer: Any,
        infer_lock: Lock,
        *,
        yolo_size: int,
        conf_thresh: float,
    ) -> None:
        self._infer = infer
        self._lock = infer_lock
        self._yolo_size = yolo_size
        self._conf_thresh = conf_thresh
        self._tracker = ByteTracker()
        self._localizer = ObjectLocalizer()

    def process(self, pkt: FramePacket) -> ObjectPacket:
        prep = preprocess_yolo26(pkt.frame, self._yolo_size)
        with self._lock:
            raw = self._infer.infer("yolo26", prep)
        dets = decode_yolo26(raw, self._conf_thresh, pkt.frame_w, pkt.frame_h,
                             self._yolo_size, self._yolo_size)
        dets = self._tracker.update(dets)
        polygon = fixed_guidance_template(pkt.frame_h, pkt.frame_w)["fixed_corridor_polygon_img"]
        front = self._localizer.localize(
            dets, pkt.frame_w, pkt.frame_h, corridor_polygon=polygon,
        )
        front_obj = front.det if front.valid else None
        return ObjectPacket(
            frame_id=pkt.frame_id, timestamp=pkt.timestamp,
            source_frame=pkt.frame,
            detections=dets, front_object=front_obj,
        )

    def reset(self) -> None:
        self._tracker.reset()


def now_ts() -> float:
    return time.perf_counter()
