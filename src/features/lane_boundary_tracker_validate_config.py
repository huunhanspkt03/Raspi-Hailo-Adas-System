"""Validation thresholds, reject labels, and result/config dataclasses."""
from __future__ import annotations

from dataclasses import dataclass

# Tightened thresholds - replay metrics on test3/test4 showed bad sides
# slipping through the previous 120 / 120 / 80 limits with corridor that
# was clearly non-parallel to the opposite purple.
DEFAULT_MAX_FRAME_JUMP_PX = 140.0     # max per-y x delta vs previous coeffs
DEFAULT_MAX_WIDTH_ERROR_PX = 80.0     # was 120; per-y |cand width - memory|
DEFAULT_PINCH_TOP_WIDTH_FRAC = 0.55   # candidate top width / memory top width
DEFAULT_MIN_CROSSING_GAP_PX = 1.0     # min gap candidate <-> opposite previous
DEFAULT_MAX_OPPOSITE_MEMORY_ERROR_PX = 70.0   # was 120
DEFAULT_WEAK_SUPPORT_PIXELS = 140     # was 80; below this = "weak"
DEFAULT_MIN_STRONG_SUPPORT_PIXELS = 200  # at/above this = "strong" anchor
# Parallel-shape gate. Lanes in perspective have nearly equal quadratic
# curvature on both sides. Anything past these thresholds means one side
# is fitting independent geometry - usually the bogus one.
DEFAULT_MAX_PARALLEL_CURVATURE_DELTA = 2.0e-3
DEFAULT_MAX_PARALLEL_SLOPE_DELTA = 1.0
# Opposite-anchored override. ``|cand - (opposite +/- width_profile)|``.
DEFAULT_MAX_EXPECTED_FROM_OPPOSITE_PX = 80.0

REJECT_FRAME_JUMP = "frame_jump"
REJECT_WIDTH_MEMORY_MISMATCH = "width_memory_mismatch"
REJECT_CORRIDOR_PINCH = "corridor_pinch"
REJECT_CROSSING = "crossing_other_side"
REJECT_OPPOSITE_MEMORY_MISMATCH = "opposite_memory_mismatch"
REJECT_WEAK_SUPPORT = "weak_support"
REJECT_STRONG_SIDE_COPY = "strong_side_copy"
REJECT_NOT_PARALLEL = "not_parallel_to_opposite"
REJECT_OPPOSITE_ANCHORED_OVERRIDE = "opposite_anchored_override"


@dataclass(slots=True)
class ValidationConfig:
    """Per-tracker thresholds passed to ``validate_candidate``."""
    top_frac: float
    num_points: int
    max_frame_jump_px: float = DEFAULT_MAX_FRAME_JUMP_PX
    max_width_error_px: float = DEFAULT_MAX_WIDTH_ERROR_PX
    pinch_top_width_frac: float = DEFAULT_PINCH_TOP_WIDTH_FRAC
    min_crossing_gap_px: float = DEFAULT_MIN_CROSSING_GAP_PX
    max_opposite_memory_error_px: float = DEFAULT_MAX_OPPOSITE_MEMORY_ERROR_PX
    weak_support_pixels: int = DEFAULT_WEAK_SUPPORT_PIXELS
    min_strong_support_pixels: int = DEFAULT_MIN_STRONG_SUPPORT_PIXELS
    max_parallel_curvature_delta: float = DEFAULT_MAX_PARALLEL_CURVATURE_DELTA
    max_parallel_slope_delta: float = DEFAULT_MAX_PARALLEL_SLOPE_DELTA
    max_expected_from_opposite_px: float = DEFAULT_MAX_EXPECTED_FROM_OPPOSITE_PX


@dataclass(slots=True)
class CandidateValidationResult:
    """Outcome of validating one candidate. ``accepted=False`` tells the
    tracker to drop the candidate; diagnostic numbers are populated when
    computable regardless of accept/reject."""
    accepted: bool
    reason: str | None = None
    memory_error_px: float | None = None
    purple_width_error_px: float | None = None
    pinch_detected: bool = False
    opposite_memory_error_px: float | None = None
    support_weak: bool = False
    parallel_curvature_error: float | None = None
    parallel_slope_error: float | None = None
    expected_from_opposite_error_px: float | None = None
    parallel_reject: bool = False
