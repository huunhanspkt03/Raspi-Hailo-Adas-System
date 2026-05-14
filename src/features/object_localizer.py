"""Monocular object localization for Windows/ONNX testing.

This module estimates object pose from a single camera only.  It is still an
approximation, but the implementation avoids relying on one fragile cue:

* object-size depth from bbox height, matching the simple single-camera
  method used by Vehicle-CV-ADAS,
* a bottom-of-frame cap so objects touching the lower image area are treated
  as close even when the detected box is partial,
* per-track temporal filtering to suppress YOLO bbox jitter,
* confidence/source metadata so downstream code can avoid treating every
  estimate as equally reliable.
"""
from __future__ import annotations

import math

import numpy as np

from features.object_localizer_pose import estimate_road_object_pose
from features.object_localizer_priors import (
    FrontObject,
    ObjectPrior,
    _REF_PRIORS_M,
    _TRAFFIC_LIGHT_LABELS,
    _TrackPose,
)
from features.object_localizer_smooth import smooth_pose
from features.object_localizer_traffic import (
    estimate_traffic_light_pose,
    fuse_traffic_light_groups,
)


__all__ = ["FrontObject", "ObjectLocalizer", "ObjectPrior"]


class ObjectLocalizer:
    """Approximate object pose in camera coordinates from 2D detections.

    The default camera priors match the older runtime closely enough for ONNX
    debug videos, while adding calibration hooks and temporal stability.
    """

    def __init__(
        self,
        hfov_deg: float = 73.0,
        vfov_deg: float | None = None,
        camera_height_m: float = 1.0,
        pitch_deg: float = 5.0,
        center_gate_ratio: float = 0.18,
        min_bottom_ratio: float = 0.45,
        fx_px: float | None = None,
        fy_px: float | None = None,
        cx_px: float | None = None,
        cy_px: float | None = None,
        smoothing_enabled: bool = True,
        smoothing_min_alpha: float = 0.25,
        smoothing_max_alpha: float = 0.70,
        max_depth_step_m: float = 8.0,
        size_focal_px: float | None = None,
        traffic_light_mount_height_m: float = 5.5,
        traffic_light_low_mount_height_m: float | None = 2.5,
        traffic_light_min_distance_m: float = 3.0,
        traffic_light_max_distance_m: float = 80.0,
        traffic_light_size_fallback_min_m: float = 8.0,
        traffic_light_smoothing_alpha: float = 0.25,
        traffic_light_max_depth_step_m: float = 4.0,
        traffic_light_mount_switch_ratio: float = 3.0,
    ) -> None:
        self.hfov_deg = float(hfov_deg)
        self.vfov_deg = None if vfov_deg is None else float(vfov_deg)
        self.camera_height_m = float(camera_height_m)
        # Positive pitch means the camera is tilted upward; negative means
        # downward. The traffic-light geometry uses the same convention.
        self.pitch_rad = math.radians(float(pitch_deg))
        self.center_gate_ratio = float(center_gate_ratio)
        self.min_bottom_ratio = float(min_bottom_ratio)
        self.fx_px = None if fx_px is None else float(fx_px)
        self.fy_px = None if fy_px is None else float(fy_px)
        self.cx_px = None if cx_px is None else float(cx_px)
        self.cy_px = None if cy_px is None else float(cy_px)
        self.smoothing_enabled = bool(smoothing_enabled)
        self.smoothing_min_alpha = float(smoothing_min_alpha)
        self.smoothing_max_alpha = float(smoothing_max_alpha)
        self.max_depth_step_m = float(max_depth_step_m)
        self.size_focal_px = None if size_focal_px is None else float(size_focal_px)
        self.traffic_light_mount_height_m = float(traffic_light_mount_height_m)
        self.traffic_light_low_mount_height_m = (
            None if traffic_light_low_mount_height_m is None
            else float(traffic_light_low_mount_height_m)
        )
        self.traffic_light_min_distance_m = float(traffic_light_min_distance_m)
        self.traffic_light_max_distance_m = float(traffic_light_max_distance_m)
        self.traffic_light_size_fallback_min_m = float(traffic_light_size_fallback_min_m)
        self.traffic_light_smoothing_alpha = float(traffic_light_smoothing_alpha)
        self.traffic_light_max_depth_step_m = float(traffic_light_max_depth_step_m)
        self.traffic_light_mount_switch_ratio = float(traffic_light_mount_switch_ratio)
        self._tracks: dict[int, _TrackPose] = {}
        self._traffic_light_tracks: dict[str, dict] = {}
        self._traffic_light_frame_seq = 0

    def _make_intrinsics(self, frame_w: int, frame_h: int) -> tuple[float, float, float, float]:
        if self.fx_px is not None:
            fx = self.fx_px
        else:
            half_hfov_rad = math.radians(self.hfov_deg * 0.5)
            fx = (0.5 * float(frame_w)) / max(math.tan(half_hfov_rad), 1e-3)
        if self.fy_px is not None:
            fy = self.fy_px
        elif self.vfov_deg is None:
            fy = fx
        else:
            half_vfov_rad = math.radians(self.vfov_deg * 0.5)
            fy = (0.5 * float(frame_h)) / max(math.tan(half_vfov_rad), 1e-3)
        cx = self.cx_px if self.cx_px is not None else 0.5 * float(frame_w - 1)
        cy = self.cy_px if self.cy_px is not None else 0.5 * float(frame_h - 1)
        return fx, fy, cx, cy

    def reset(self) -> None:
        """Clear temporal pose state, e.g. when switching videos."""
        self._tracks.clear()
        self._traffic_light_tracks.clear()
        self._traffic_light_frame_seq = 0

    def _smooth_pose(
        self, det: dict, z: float, x_m: float, confidence: float,
    ) -> tuple[float, float, float]:
        return smooth_pose(
            self._tracks, det, z, x_m, confidence,
            self.smoothing_enabled, self.smoothing_min_alpha,
            self.smoothing_max_alpha, self.max_depth_step_m,
        )

    def _estimate_pose(
        self,
        det: dict,
        frame_w: int,
        frame_h: int,
        corridor_polygon: np.ndarray | None = None,
    ) -> bool:
        x1, y1, x2, y2 = det["box"]
        bbox_w = float(x2 - x1)
        bbox_h = float(y2 - y1)
        det["bottom_center_x"] = int(round(x1 + bbox_w * 0.5))
        det["bottom_center_y"] = int(round(y2))
        det["foot_x"] = det["bottom_center_x"]
        det["foot_y"] = det["bottom_center_y"]
        det["foot_source"] = "bottom_center"
        det["pose_valid"] = False
        det["in_path"] = False
        det["distance_m"] = -1.0
        det["x_m"] = 0.0
        det["z_m"] = 0.0
        det["bearing_deg"] = 0.0
        det["distance_confidence"] = 0.0
        det["distance_source"] = "none"
        det["distance_raw_m"] = {}
        det["distance_display_valid"] = False

        label = det.get("label", "")
        prior = _REF_PRIORS_M.get(label)
        if prior is None or bbox_w <= 1.0 or bbox_h <= 1.0:
            return False
        fx, fy, cx, cy = self._make_intrinsics(frame_w, frame_h)
        if label in _TRAFFIC_LIGHT_LABELS:
            return estimate_traffic_light_pose(
                self, det, prior, x1, y1, x2, y2, bbox_w, bbox_h, fx, fy, cx, cy,
            )
        return estimate_road_object_pose(
            self, det, prior, x1, y1, x2, y2, bbox_w, bbox_h,
            fx, fy, cx, cy, frame_w, frame_h, corridor_polygon,
        )

    def localize(
        self,
        detections: list[dict],
        frame_w: int,
        frame_h: int,
        *,
        corridor_polygon: np.ndarray | None = None,
    ) -> FrontObject:
        self._traffic_light_frame_seq += 1
        front = FrontObject()
        seen_tracks: set[int] = set()
        for det in detections:
            if not self._estimate_pose(det, frame_w, frame_h, corridor_polygon=corridor_polygon):
                continue
            tid = det.get("track_id")
            if tid is not None:
                try:
                    seen_tracks.add(int(tid))
                except (TypeError, ValueError):
                    pass
            if not det["in_path"]:
                continue
            if not front.valid or det["z_m"] < float(front.det["z_m"]):
                front = FrontObject(valid=True, det=det)
        fuse_traffic_light_groups(self, detections, frame_w, frame_h)
        self._traffic_light_tracks = {
            key: state for key, state in self._traffic_light_tracks.items()
            if self._traffic_light_frame_seq - int(state.get("frame", 0)) <= 45
        }
        if seen_tracks:
            self._tracks = {
                tid: state for tid, state in self._tracks.items()
                if tid in seen_tracks
            }
        return front
