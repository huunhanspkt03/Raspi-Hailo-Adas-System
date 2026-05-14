"""Stateful per-side purple-boundary tracker.

Per-side resolution order (see ``lane_boundary_tracker_resolve``):
``DETECTED -> HELD -> LOST``. Missing-side inference is disabled, so a
single detected marking no longer creates a synthetic opposite boundary.

Candidates are gated by ``lane_boundary_tracker_validate`` before
EMA-blending: a frame-jump / width-memory / corridor-pinch / crossing
failure drops the candidate so resolve falls through to a safe held or
lost output. Per-side streaks (``raw_missing`` / ``held`` /
``output_missing``) and aggregate ``lane_state`` /
``boundary_confidence`` ride on every output dict.
"""
from __future__ import annotations

from dataclasses import dataclass

from features.lane_area_width_profile import LaneAreaWidthMemory
from features.lane_boundary_tracker_contract import (
    HELD_DECAY_FRAMES,
)
from features.lane_boundary_tracker_geom import ema
from features.lane_boundary_tracker_memory import (
    DEFAULT_MEMORY_WARMUP_LANE_AREA_TOLERANCE_PX,
)
from features.lane_boundary_tracker_resolve import (
    DEFAULT_WIDTH_EMA_ALPHA, PurpleWidthMemory,
)
from features.lane_boundary_tracker_update import run_tracker_update
from features.lane_boundary_tracker_validate import (
    DEFAULT_MAX_EXPECTED_FROM_OPPOSITE_PX, DEFAULT_MAX_FRAME_JUMP_PX,
    DEFAULT_MAX_OPPOSITE_MEMORY_ERROR_PX,
    DEFAULT_MAX_PARALLEL_CURVATURE_DELTA, DEFAULT_MAX_PARALLEL_SLOPE_DELTA,
    DEFAULT_MAX_WIDTH_ERROR_PX, DEFAULT_MIN_STRONG_SUPPORT_PIXELS,
    DEFAULT_PINCH_TOP_WIDTH_FRAC, DEFAULT_WEAK_SUPPORT_PIXELS,
)

DEFAULT_EMA_ALPHA = 0.40
DEFAULT_ACCEPT_CONFIDENCE = 0.18
DEFAULT_NUM_POINTS = 24
DEFAULT_TOP_FRAC = 0.55
DEFAULT_MIN_INFERENCE_WIDTH_PX = 20.0
# Kept for constructor/back-compat only; missing-side inference is disabled.
DEFAULT_MAX_INFERRED_STREAK = 0
DEFAULT_HELD_FRAME_LIMIT = HELD_DECAY_FRAMES


@dataclass(slots=True)
class _SideState:
    coeffs: tuple[float, float, float] | None = None
    last_detected_confidence: float = 0.0
    last_n_pixels: int = 0
    raw_missing_streak: int = 0
    inferred_streak: int = 0
    held_streak: int = 0
    output_missing_streak: int = 0


class LaneBoundaryTracker:
    def __init__(
        self,
        *,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        accept_confidence: float = DEFAULT_ACCEPT_CONFIDENCE,
        num_points: int = DEFAULT_NUM_POINTS,
        top_frac: float = DEFAULT_TOP_FRAC,
        min_inference_width_px: float = DEFAULT_MIN_INFERENCE_WIDTH_PX,
        max_inferred_streak: int = DEFAULT_MAX_INFERRED_STREAK,
        held_frame_limit: int = DEFAULT_HELD_FRAME_LIMIT,
        width_ema_alpha: float = DEFAULT_WIDTH_EMA_ALPHA,
        max_frame_jump_px: float = DEFAULT_MAX_FRAME_JUMP_PX,
        max_width_error_px: float = DEFAULT_MAX_WIDTH_ERROR_PX,
        pinch_top_width_frac: float = DEFAULT_PINCH_TOP_WIDTH_FRAC,
        max_opposite_memory_error_px: float = DEFAULT_MAX_OPPOSITE_MEMORY_ERROR_PX,
        weak_support_pixels: int = DEFAULT_WEAK_SUPPORT_PIXELS,
        min_strong_support_pixels: int = DEFAULT_MIN_STRONG_SUPPORT_PIXELS,
        max_parallel_curvature_delta: float = DEFAULT_MAX_PARALLEL_CURVATURE_DELTA,
        max_parallel_slope_delta: float = DEFAULT_MAX_PARALLEL_SLOPE_DELTA,
        max_expected_from_opposite_px: float = DEFAULT_MAX_EXPECTED_FROM_OPPOSITE_PX,
        memory_warmup_lane_area_tolerance_px: float = (
            DEFAULT_MEMORY_WARMUP_LANE_AREA_TOLERANCE_PX),
    ) -> None:
        self.ema_alpha = float(ema_alpha)
        self.accept_confidence = float(accept_confidence)
        self.num_points = int(num_points)
        self.top_frac = float(top_frac)
        self.min_inference_width_px = float(min_inference_width_px)
        self.max_inferred_streak = int(max_inferred_streak)
        self.held_frame_limit = int(held_frame_limit)
        self.max_frame_jump_px = float(max_frame_jump_px)
        self.max_width_error_px = float(max_width_error_px)
        self.pinch_top_width_frac = float(pinch_top_width_frac)
        self.max_opposite_memory_error_px = float(max_opposite_memory_error_px)
        self.weak_support_pixels = int(weak_support_pixels)
        self.min_strong_support_pixels = int(min_strong_support_pixels)
        self.max_parallel_curvature_delta = float(max_parallel_curvature_delta)
        self.max_parallel_slope_delta = float(max_parallel_slope_delta)
        self.max_expected_from_opposite_px = float(max_expected_from_opposite_px)
        self.memory_warmup_lane_area_tolerance_px = float(
            memory_warmup_lane_area_tolerance_px)
        self._left = _SideState()
        self._right = _SideState()
        self._width_memory = PurpleWidthMemory(
            top_frac=self.top_frac, num_points=self.num_points,
            ema_alpha=float(width_ema_alpha),
        )
        self._lane_area_width_memory = LaneAreaWidthMemory()
        self._frame_id = -1

    def reset(self) -> None:
        self._left = _SideState()
        self._right = _SideState()
        self._width_memory.reset()
        self._lane_area_width_memory.reset()
        self._frame_id = -1

    @property
    def has_state(self) -> bool:
        return self._left.coeffs is not None or self._right.coeffs is not None

    def update(self, semantics, lane_area, frame_h, frame_w, **fit_kwargs) -> dict:
        return run_tracker_update(self, semantics, lane_area, frame_h, frame_w, **fit_kwargs)

    def _previous_side_split_x(self, frame_h: int) -> float | None:
        """Bottom-row midpoint of cached coeffs. None unless BOTH sides
        have memory - single-side memory leaves the split unbiased so the
        lane_area / frame-center fallback in ``fit_lane_boundaries`` wins."""
        if self._left.coeffs is None or self._right.coeffs is None:
            return None
        y = float(frame_h - 1)
        la, lb, lc = self._left.coeffs
        ra, rb, rc = self._right.coeffs
        return 0.5 * ((la * y * y + lb * y + lc) + (ra * y * y + rb * y + rc))

    def _try_detect(
        self, side: _SideState,
        new_coeffs: tuple[float, float, float] | None,
        new_conf: float, new_n: int,
    ) -> bool:
        """EMA-blend on accept; return whether the fit was accepted."""
        if new_coeffs is None or float(new_conf) < self.accept_confidence:
            return False
        side.coeffs = (new_coeffs if side.coeffs is None
                       else ema(side.coeffs, new_coeffs, self.ema_alpha))
        side.last_detected_confidence = float(new_conf)
        side.last_n_pixels = int(new_n)
        return True
