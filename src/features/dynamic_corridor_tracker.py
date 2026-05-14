"""Stateful EMA wrapper around ``compute_dynamic_corridor``.

Holds the last accepted coefficients so weak/missing masks hold the
previous corridor for ``lost_frame_limit`` frames before falling back to
the camera-anchored template. Emits ``LANE_OK`` (fresh accept),
``LANE_WEAK`` (held), or ``LANE_LOST`` (held too long).
"""
from __future__ import annotations

import numpy as np

from features.dynamic_corridor import (
    DEFAULT_BOTTOM_FRAC, DEFAULT_HALF_WIDTH_FAR_FRAC,
    DEFAULT_HALF_WIDTH_NEAR_FRAC, DEFAULT_NUM_POINTS, DEFAULT_TOP_FRAC,
    build_corridor_dict, compute_dynamic_corridor,
)

DEFAULT_EMA_ALPHA = 0.35
DEFAULT_LOST_FRAME_LIMIT = 8
DEFAULT_ACCEPT_CONFIDENCE = 0.18
DEFAULT_MAX_CENTER_JUMP_FRAC = 0.18
DEFAULT_MAX_ABS_CURVATURE = 3.0e-3
DEFAULT_MAX_BOTTOM_CENTER_OFFSET_FRAC = 0.28

LANE_OK = "LANE_OK"
LANE_WEAK = "LANE_WEAK"
LANE_LOST = "LANE_LOST"

__all__ = [
    "DynamicCorridorTracker", "DEFAULT_EMA_ALPHA", "DEFAULT_LOST_FRAME_LIMIT",
    "DEFAULT_ACCEPT_CONFIDENCE", "DEFAULT_MAX_CENTER_JUMP_FRAC",
    "DEFAULT_MAX_ABS_CURVATURE", "DEFAULT_MAX_BOTTOM_CENTER_OFFSET_FRAC",
    "LANE_OK", "LANE_WEAK", "LANE_LOST",
]


class DynamicCorridorTracker:
    def __init__(
        self,
        *,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        lost_frame_limit: int = DEFAULT_LOST_FRAME_LIMIT,
        accept_confidence: float = DEFAULT_ACCEPT_CONFIDENCE,
        top_frac: float = DEFAULT_TOP_FRAC,
        bottom_frac: float = DEFAULT_BOTTOM_FRAC,
        half_width_near_frac: float = DEFAULT_HALF_WIDTH_NEAR_FRAC,
        half_width_far_frac: float = DEFAULT_HALF_WIDTH_FAR_FRAC,
        num_points: int = DEFAULT_NUM_POINTS,
        max_center_jump_frac: float = DEFAULT_MAX_CENTER_JUMP_FRAC,
        max_abs_curvature: float = DEFAULT_MAX_ABS_CURVATURE,
        max_bottom_center_offset_frac: float = (
            DEFAULT_MAX_BOTTOM_CENTER_OFFSET_FRAC),
    ) -> None:
        self.ema_alpha = float(ema_alpha)
        self.lost_frame_limit = int(lost_frame_limit)
        self.accept_confidence = float(accept_confidence)
        self.top_frac = float(top_frac)
        self.bottom_frac = float(bottom_frac)
        self.half_width_near_frac = float(half_width_near_frac)
        self.half_width_far_frac = float(half_width_far_frac)
        self.num_points = int(num_points)
        self.max_center_jump_frac = float(max_center_jump_frac)
        self.max_abs_curvature = float(max_abs_curvature)
        self.max_bottom_center_offset_frac = float(max_bottom_center_offset_frac)
        self._coeffs: tuple[float, float, float] | None = None
        self._last_confidence: float = 0.0
        self._last_n_points: int = 0
        self._last_rms: float = 0.0
        self._missing_streak: int = 0
        self._last_reject_reason: str | None = None

    def reset(self) -> None:
        self._coeffs = None
        self._last_confidence = 0.0
        self._last_n_points = 0
        self._last_rms = 0.0
        self._missing_streak = 0
        self._last_reject_reason = None

    @property
    def has_state(self) -> bool:
        return self._coeffs is not None

    def update(
        self,
        lane_area: np.ndarray | None,
        frame_h: int,
        frame_w: int,
    ) -> dict | None:
        """Return guidance dict with ``lane_state``. ``None`` only when no
        cached coeffs exist and this frame produced no fit."""
        result = compute_dynamic_corridor(
            lane_area, frame_h, frame_w,
            top_frac=self.top_frac, bottom_frac=self.bottom_frac,
            half_width_near_frac=self.half_width_near_frac,
            half_width_far_frac=self.half_width_far_frac,
            num_points=self.num_points,
        )
        accept, reject_reason = self._accept_result(result, frame_h, frame_w)
        if accept and result is not None:
            new_coeffs = result["coeffs"]
            if self._coeffs is None:
                self._coeffs = new_coeffs
            else:
                a = self.ema_alpha
                self._coeffs = (
                    (1.0 - a) * self._coeffs[0] + a * new_coeffs[0],
                    (1.0 - a) * self._coeffs[1] + a * new_coeffs[1],
                    (1.0 - a) * self._coeffs[2] + a * new_coeffs[2],
                )
            self._last_confidence = float(result["confidence"])
            self._last_n_points = int(result["n_points"])
            self._last_rms = float(result["rms_residual_px"])
            self._missing_streak = 0
            self._last_reject_reason = None
            return self._emit(self._coeffs, frame_h, frame_w,
                              state=LANE_OK, from_cache=False)

        self._missing_streak += 1
        self._last_reject_reason = reject_reason
        if self._coeffs is None:
            return None
        state = LANE_LOST if self._missing_streak > self.lost_frame_limit else LANE_WEAK
        return self._emit(self._coeffs, frame_h, frame_w,
                          state=state, from_cache=True)

    def _emit(
        self, coeffs: tuple[float, float, float],
        frame_h: int, frame_w: int, *, state: str, from_cache: bool,
    ) -> dict:
        half_near = max(8.0, self.half_width_near_frac * float(frame_w))
        half_far = max(6.0, self.half_width_far_frac * float(frame_w))
        out = build_corridor_dict(
            coeffs,
            frame_h=frame_h, frame_w=frame_w,
            top_frac=self.top_frac, bottom_frac=self.bottom_frac,
            half_width_near_px=half_near, half_width_far_px=half_far,
            num_points=self.num_points,
        )
        decay = max(0.0, 1.0 - 0.10 * float(self._missing_streak)) if from_cache else 1.0
        out["coeffs"] = coeffs
        out["half_width_near_px"] = half_near
        out["half_width_far_px"] = half_far
        out["is_dynamic"] = True
        out["lane_state"] = state
        out["from_cache"] = from_cache
        out["confidence"] = self._last_confidence * decay
        out["n_points"] = self._last_n_points
        out["rms_residual_px"] = self._last_rms
        out["missing_streak"] = self._missing_streak
        out["reject_reason"] = self._last_reject_reason
        return out

    def _accept_result(
        self,
        result: dict | None,
        frame_h: int,
        frame_w: int,
    ) -> tuple[bool, str | None]:
        """Gate single-frame lane_area fits before they can update EMA.

        Reject low confidence, excessive curvature, sudden center jumps,
        and bottom-center offset that would let occlusion holes commit a
        bogus fit. Caller holds the previous corridor on rejection.
        """
        if result is None:
            return False, "no_fit"
        if float(result.get("confidence", 0.0)) < self.accept_confidence:
            return False, "low_confidence"
        coeffs = result.get("coeffs")
        if coeffs is None:
            return False, "no_coeffs"
        if abs(float(coeffs[0])) > self.max_abs_curvature:
            return False, "curvature_jump"
        y_bottom = float(int(frame_h) - 1)
        bottom_x = float(_sample_coeffs(coeffs, np.array([y_bottom],
                                                         dtype=np.float32))[0])
        camera_x = 0.5 * float(int(frame_w) - 1)
        bottom_offset = abs(bottom_x - camera_x)
        if bottom_offset > self.max_bottom_center_offset_frac * float(frame_w):
            return False, "bottom_center_offset"
        if self._coeffs is not None:
            y_top = max(0.0, int(frame_h) * self.top_frac)
            ys = np.linspace(y_top, float(int(frame_h) - 1),
                             num=self.num_points, dtype=np.float32)
            new_x = _sample_coeffs(coeffs, ys)
            old_x = _sample_coeffs(self._coeffs, ys)
            max_jump = float(np.max(np.abs(new_x - old_x)))
            if max_jump > self.max_center_jump_frac * float(frame_w):
                return False, "center_jump"
        return True, None


def _sample_coeffs(coeffs: tuple[float, float, float],
                   ys: np.ndarray) -> np.ndarray:
    a, b, c = coeffs
    return (a * ys * ys + b * ys + c).astype(np.float32)
