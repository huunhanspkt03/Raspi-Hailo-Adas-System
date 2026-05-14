"""Prototype ego-speed estimator from front camera (image-space scale).

Estimates **own-vehicle** speed (NOT other-vehicle speed) by measuring
vertical optical flow over the road ROI defined by ``lane_area``. Object
bounding boxes / tracker IDs are intentionally not used: ego motion is
read from how the road texture moves between frames.

Scale constant
--------------
``EGO_SPEED_IMAGE_PX_PER_METER = 160.0`` is a dedicated, ego-speed-only
image-space scale. It was measured manually by drawing a 1 m reference
on the road surface in a real frame and reading off ~160 pixels along
that mark near the lower road ROI. Notes:

* This is **not** a calibrated BEV/IPM scale.
* It is only valid near the lower road ROI; the pixel-per-meter ratio
  changes with row because of perspective foreshortening.
* It deliberately differs from ``ego_lane_geometry.BEV_PX_PER_METER``
  (= 80.0), which belongs to the older prototype BEV-trapezoid geometry.
* For real accuracy, replace this with a calibrated camera-to-ground
  transform or fuse another sensor (wheel speed, GPS, IMU).

Every output dict carries ``scale_source = "prototype_image_space_scale"``
so downstream code does not mistake it for a calibrated value.
"""
from __future__ import annotations

import numpy as np

from features.ego_speed_flow import (
    compute_median_dy, prepare_gray_and_mask, roi_from_mask,
)

# Image-space pixels-per-meter for the lower road ROI. Manually measured
# from a 1 m road reference in a real frame (~160 px). Prototype only -
# see module docstring for caveats.
EGO_SPEED_IMAGE_PX_PER_METER = 160.0

DEFAULT_ROI_TOP_FRAC = 0.55
DEFAULT_DOWNSCALE = 0.5
DEFAULT_EMA_ALPHA = 0.3
DEFAULT_MIN_FLOW_PX = 0.5
DEFAULT_MAX_FLOW_PX = 60.0
DEFAULT_MIN_POINTS = 200
_SCALE_SOURCE = "prototype_image_space_scale"

__all__ = ["EgoSpeedEstimator", "EGO_SPEED_IMAGE_PX_PER_METER"]


def _invalid(reason: str) -> dict:
    return {
        "speed_mps": 0.0,
        "speed_kmh": 0.0,
        "valid": False,
        "confidence": 0.0,
        "dy_px": 0.0,
        "n_flow_points": 0,
        "reason": reason,
        "scale_source": _SCALE_SOURCE,
    }


class EgoSpeedEstimator:
    """Image-space optical-flow ego-speed estimator (prototype scale).

    The output ``speed_mps`` / ``speed_kmh`` is **not** a calibrated
    value. By default it uses ``EGO_SPEED_IMAGE_PX_PER_METER`` (~160
    px/m), a manually measured image-space scale from a 1 m road
    reference near the lower ROI. Treat the output as approximate.
    """

    def __init__(
        self,
        *,
        roi_top_frac: float = DEFAULT_ROI_TOP_FRAC,
        downscale: float = DEFAULT_DOWNSCALE,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        min_flow_px: float = DEFAULT_MIN_FLOW_PX,
        max_flow_px: float = DEFAULT_MAX_FLOW_PX,
        min_points: int = DEFAULT_MIN_POINTS,
        px_per_meter: float = EGO_SPEED_IMAGE_PX_PER_METER,
    ) -> None:
        self.roi_top_frac = float(roi_top_frac)
        ds = float(downscale)
        self.downscale = ds if 0.0 < ds <= 1.0 else 1.0
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.min_flow_px = float(min_flow_px)
        self.max_flow_px = float(max_flow_px)
        self.min_points = int(min_points)
        self.px_per_meter = float(px_per_meter)
        self._prev_gray: np.ndarray | None = None
        self._prev_mask: np.ndarray | None = None
        self._smoothed_mps: float | None = None

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_mask = None
        self._smoothed_mps = None

    def update(
        self,
        frame: np.ndarray | None,
        lane_area: np.ndarray | None,
        *,
        fps: float | None = None,
        dt: float | None = None,
    ) -> dict:
        if frame is None or getattr(frame, "size", 0) == 0:
            return _invalid("no_frame")
        if lane_area is None or getattr(lane_area, "size", 0) == 0:
            self._prev_gray = None
            self._prev_mask = None
            return _invalid("no_mask")
        eff_fps = (1.0 / float(dt)) if (dt is not None and dt > 0.0) else fps
        if eff_fps is None or float(eff_fps) <= 0.0:
            return _invalid("invalid_fps")

        gray_ds, mask_ds, scale = prepare_gray_and_mask(frame, lane_area, self.downscale)
        roi = roi_from_mask(mask_ds, self.roi_top_frac)
        roi_count = int(np.count_nonzero(roi))
        if roi_count < self.min_points:
            self._prev_gray, self._prev_mask = gray_ds, roi
            return _invalid("roi_too_small")

        if (self._prev_gray is None or self._prev_mask is None
                or self._prev_gray.shape != gray_ds.shape):
            self._prev_gray, self._prev_mask = gray_ds, roi
            return _invalid("no_previous")

        median_dy, n_used, mad = compute_median_dy(
            self._prev_gray, gray_ds, self._prev_mask, roi,
            min_flow_px=self.min_flow_px, max_flow_px=self.max_flow_px,
            min_points=self.min_points,
        )
        if median_dy is None:
            self._prev_gray, self._prev_mask = gray_ds, roi
            return _invalid("not_enough_flow")

        median_dy_full = median_dy / max(scale, 1e-6)
        speed_mps = (
            abs(median_dy_full) * float(eff_fps)
            / max(self.px_per_meter, 1e-6)
        )
        if self._smoothed_mps is None:
            self._smoothed_mps = speed_mps
        else:
            self._smoothed_mps = (
                self.ema_alpha * speed_mps
                + (1.0 - self.ema_alpha) * self._smoothed_mps
            )

        confidence = self._confidence(
            n_used, mad, median_dy,
            roi_count, mask_ds.shape[0] * mask_ds.shape[1],
        )

        self._prev_gray, self._prev_mask = gray_ds, roi
        smoothed = float(self._smoothed_mps)
        return {
            "speed_mps": smoothed,
            "speed_kmh": smoothed * 3.6,
            "valid": True,
            "confidence": confidence,
            "dy_px": float(median_dy_full),
            "n_flow_points": int(n_used),
            "reason": "ok",
            "scale_source": _SCALE_SOURCE,
            "raw_speed_mps": float(speed_mps),
        }

    @staticmethod
    def _confidence(
        n_used: int, mad: float, median_dy: float,
        roi_count: int, frame_pixels: int,
    ) -> float:
        point_score = float(min(1.0, n_used / 800.0))
        denom = max(1.0, abs(median_dy) + 1.0)
        stability = float(np.exp(-mad / denom))
        roi_score = float(min(1.0, roi_count / max(1.0, 0.10 * frame_pixels)))
        return float(np.clip(
            0.5 * point_score + 0.3 * stability + 0.2 * roi_score,
            0.0, 1.0,
        ))
