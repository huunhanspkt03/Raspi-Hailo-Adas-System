"""Per-side candidate validation (frame jump, width, pinch, crossing)."""
from __future__ import annotations

import numpy as np

from features.lane_boundary_tracker_geom import (
    max_abs_delta, profile_error, sample_coeffs,
)
from features.lane_boundary_tracker_resolve import PurpleWidthMemory
from features.lane_boundary_tracker_validate_config import (
    REJECT_CORRIDOR_PINCH, REJECT_CROSSING, REJECT_FRAME_JUMP,
    REJECT_WIDTH_MEMORY_MISMATCH,
    CandidateValidationResult, ValidationConfig,
)


def _candidate_width(side: str, cand_xs: np.ndarray,
                     other_xs: np.ndarray) -> np.ndarray:
    return (other_xs - cand_xs) if side == "left" else (cand_xs - other_xs)


def _crosses(side: str, cand_xs: np.ndarray, other_xs: np.ndarray,
             min_gap_px: float) -> bool:
    if side == "left":
        return bool(np.any(cand_xs >= other_xs - float(min_gap_px)))
    return bool(np.any(other_xs >= cand_xs - float(min_gap_px)))


def validate_candidate(
    *,
    side: str,
    candidate: tuple[float, float, float] | None,
    previous_coeffs: tuple[float, float, float] | None,
    other_previous_coeffs: tuple[float, float, float] | None,
    width_memory: PurpleWidthMemory,
    frame_h: int,
    config: ValidationConfig,
) -> CandidateValidationResult:
    """Validate a freshly fit candidate. ``None`` candidate is a passthrough
    (caller treats the side as missing, no rejection reason recorded)."""
    if candidate is None:
        return CandidateValidationResult(accepted=True)

    ys = np.linspace(max(0.0, frame_h * config.top_frac), float(frame_h - 1),
                     num=int(config.num_points), dtype=np.float32)
    cand_xs = sample_coeffs(candidate, ys)
    prev_xs = sample_coeffs(previous_coeffs, ys)
    other_prev_xs = sample_coeffs(other_previous_coeffs, ys)

    memory_error_px = max_abs_delta(cand_xs, prev_xs)
    purple_width_error_px: float | None = None
    pinch = False
    crossing = False

    if other_prev_xs is not None and cand_xs is not None:
        crossing = _crosses(side, cand_xs, other_prev_xs, config.min_crossing_gap_px)
        if width_memory.is_valid(frame_h) and width_memory.profile is not None \
                and width_memory.profile.size:
            cand_width = _candidate_width(side, cand_xs, other_prev_xs)
            purple_width_error_px = profile_error(
                cand_width.astype(np.float32), width_memory.profile,
            )
            top_mem = float(width_memory.profile[0])
            if top_mem > 0.0:
                pinch = float(cand_width[0]) < float(config.pinch_top_width_frac) * top_mem

    def _reject(reason: str) -> CandidateValidationResult:
        return CandidateValidationResult(
            accepted=False, reason=reason,
            memory_error_px=memory_error_px,
            purple_width_error_px=purple_width_error_px,
            pinch_detected=pinch,
        )

    if crossing:
        return _reject(REJECT_CROSSING)
    if memory_error_px is not None and memory_error_px > float(config.max_frame_jump_px):
        return _reject(REJECT_FRAME_JUMP)
    if (purple_width_error_px is not None
            and purple_width_error_px > float(config.max_width_error_px)):
        return _reject(REJECT_WIDTH_MEMORY_MISMATCH)
    if pinch:
        return _reject(REJECT_CORRIDOR_PINCH)
    return CandidateValidationResult(
        accepted=True, reason=None,
        memory_error_px=memory_error_px,
        purple_width_error_px=purple_width_error_px,
        pinch_detected=pinch,
    )


def replace_reject(result: CandidateValidationResult, reason: str,
                   *, mark_parallel_reject: bool = False,
                   ) -> CandidateValidationResult:
    """Return a copy of ``result`` flipped to rejected with the given reason,
    preserving diagnostic numbers and the soft flags. Sets
    ``parallel_reject`` when the rejection came from one of the new
    parallel/opposite-anchored gates."""
    return CandidateValidationResult(
        accepted=False, reason=reason,
        memory_error_px=result.memory_error_px,
        purple_width_error_px=result.purple_width_error_px,
        pinch_detected=result.pinch_detected,
        opposite_memory_error_px=result.opposite_memory_error_px,
        support_weak=result.support_weak,
        parallel_curvature_error=result.parallel_curvature_error,
        parallel_slope_error=result.parallel_slope_error,
        expected_from_opposite_error_px=result.expected_from_opposite_error_px,
        parallel_reject=bool(mark_parallel_reject or result.parallel_reject),
    )
