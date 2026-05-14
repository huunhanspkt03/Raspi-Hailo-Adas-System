"""Cross-side candidate validation for ``LaneBoundaryTracker``.

Per-side gates (frame jump, width vs memory, pinch, crossing) live in
``lane_boundary_tracker_validate_candidate``. This module composes
``validate_pair`` (cross-side memory, weak support, strong-side copy,
and parallel/opposite-anchored gates) and re-exports the public surface.
Pure: no state, no I/O.
"""
from __future__ import annotations

import numpy as np

from features.lane_boundary_tracker_geom import profile_error, sample_coeffs
from features.lane_boundary_tracker_resolve import PurpleWidthMemory
from features.lane_boundary_tracker_validate_candidate import (
    replace_reject as _replace_reject,
    validate_candidate,
)
from features.lane_boundary_tracker_validate_config import (
    DEFAULT_MAX_EXPECTED_FROM_OPPOSITE_PX, DEFAULT_MAX_FRAME_JUMP_PX,
    DEFAULT_MAX_OPPOSITE_MEMORY_ERROR_PX, DEFAULT_MAX_PARALLEL_CURVATURE_DELTA,
    DEFAULT_MAX_PARALLEL_SLOPE_DELTA, DEFAULT_MAX_WIDTH_ERROR_PX,
    DEFAULT_MIN_CROSSING_GAP_PX, DEFAULT_MIN_STRONG_SUPPORT_PIXELS,
    DEFAULT_PINCH_TOP_WIDTH_FRAC, DEFAULT_WEAK_SUPPORT_PIXELS,
    REJECT_CORRIDOR_PINCH, REJECT_CROSSING, REJECT_FRAME_JUMP,
    REJECT_NOT_PARALLEL, REJECT_OPPOSITE_ANCHORED_OVERRIDE,
    REJECT_OPPOSITE_MEMORY_MISMATCH, REJECT_STRONG_SIDE_COPY,
    REJECT_WEAK_SUPPORT, REJECT_WIDTH_MEMORY_MISMATCH,
    CandidateValidationResult, ValidationConfig,
)
from features.lane_boundary_tracker_validate_diagnostics import (
    merge_validation_diagnostics,
)
from features.lane_boundary_tracker_validate_parallel import (
    apply_opposite_anchored_gate, apply_parallel_shape_gate,
    apply_strong_side_copy_gate,
)

__all__ = (
    "DEFAULT_MAX_FRAME_JUMP_PX", "DEFAULT_MAX_WIDTH_ERROR_PX",
    "DEFAULT_PINCH_TOP_WIDTH_FRAC", "DEFAULT_MIN_CROSSING_GAP_PX",
    "DEFAULT_MAX_OPPOSITE_MEMORY_ERROR_PX", "DEFAULT_WEAK_SUPPORT_PIXELS",
    "DEFAULT_MIN_STRONG_SUPPORT_PIXELS",
    "DEFAULT_MAX_PARALLEL_CURVATURE_DELTA", "DEFAULT_MAX_PARALLEL_SLOPE_DELTA",
    "DEFAULT_MAX_EXPECTED_FROM_OPPOSITE_PX",
    "REJECT_FRAME_JUMP", "REJECT_WIDTH_MEMORY_MISMATCH",
    "REJECT_CORRIDOR_PINCH", "REJECT_CROSSING",
    "REJECT_OPPOSITE_MEMORY_MISMATCH", "REJECT_WEAK_SUPPORT",
    "REJECT_STRONG_SIDE_COPY", "REJECT_NOT_PARALLEL",
    "REJECT_OPPOSITE_ANCHORED_OVERRIDE",
    "CandidateValidationResult", "ValidationConfig",
    "validate_candidate", "validate_pair", "merge_validation_diagnostics",
)


def validate_pair(
    *,
    left_candidate: tuple[float, float, float] | None,
    right_candidate: tuple[float, float, float] | None,
    left_previous: tuple[float, float, float] | None,
    right_previous: tuple[float, float, float] | None,
    left_n_pixels: int,
    right_n_pixels: int,
    width_memory: PurpleWidthMemory,
    frame_h: int,
    config: ValidationConfig,
    lane_area_width_profile: np.ndarray | None = None,
) -> tuple[CandidateValidationResult, CandidateValidationResult]:
    """Validate both sides together so cross-side rules can catch a false
    DETECTED that the per-side rules accept.

    Pass 1 runs ``validate_candidate`` per side (frame jump, width vs
    opposite previous + memory, pinch, crossing). Pass 2 then applies, in
    order:

    * **Opposite-memory mismatch** - pair-width vs purple memory; drops
      the weaker-supported side.
    * **Weak support** - thin/fragmented side dropped when opposite is
      solid.
    * **Strong-side copy** - exactly one strong side drops the weaker
      side immediately; the resolver then copies/shifts from the strong
      anchor using purple memory or lane_area bootstrap width.
    * **Parallel / opposite-anchored diagnostics** - shape and
      expected-position errors are still populated for metrics.
    """
    left = validate_candidate(
        side="left", candidate=left_candidate,
        previous_coeffs=left_previous, other_previous_coeffs=right_previous,
        width_memory=width_memory, frame_h=frame_h, config=config,
    )
    right = validate_candidate(
        side="right", candidate=right_candidate,
        previous_coeffs=right_previous, other_previous_coeffs=left_previous,
        width_memory=width_memory, frame_h=frame_h, config=config,
    )

    if left_candidate is None or right_candidate is None:
        return left, right

    ys = np.linspace(max(0.0, frame_h * config.top_frac), float(frame_h - 1),
                     num=int(config.num_points), dtype=np.float32)
    cand_lx = sample_coeffs(left_candidate, ys)
    cand_rx = sample_coeffs(right_candidate, ys)

    has_memory = (
        width_memory.is_valid(frame_h)
        and width_memory.profile is not None
        and width_memory.profile.size > 0
    )
    if has_memory:
        cand_width = (cand_rx - cand_lx).astype(np.float32)
        err = profile_error(cand_width, width_memory.profile)
        left.opposite_memory_error_px = err
        right.opposite_memory_error_px = err

    l_n = int(left_n_pixels or 0)
    r_n = int(right_n_pixels or 0)
    weak_n = int(config.weak_support_pixels)
    left.support_weak = l_n < weak_n
    right.support_weak = r_n < weak_n

    # The original memory-based gates (opposite_memory_mismatch and
    # weak_support rejection reason) only fire when purple memory is valid.
    # ``support_weak`` itself is still populated on every frame so metrics
    # and memory-update gating can see thin/fragmented support consistently.
    if has_memory:
        # 1. Cross-side memory mismatch - drop the weaker side first.
        if left.accepted and right.accepted:
            err = left.opposite_memory_error_px
            if (err is not None
                    and err > float(config.max_opposite_memory_error_px)):
                if l_n <= r_n:
                    left = _replace_reject(left, REJECT_OPPOSITE_MEMORY_MISMATCH)
                else:
                    right = _replace_reject(right, REJECT_OPPOSITE_MEMORY_MISMATCH)

        # 2. Weak support - only when OPPOSITE is still accepted and strong.
        l_strong_now = left.accepted and l_n >= weak_n
        r_strong_now = right.accepted and r_n >= weak_n
        if left.accepted and l_n < weak_n and r_strong_now:
            left = _replace_reject(left, REJECT_WEAK_SUPPORT)
        if right.accepted and r_n < weak_n and l_strong_now:
            right = _replace_reject(right, REJECT_WEAK_SUPPORT)

    # 3. Strong-side copy override. This is intentionally stricter than
    # the shape gates below: one strong side is enough to drop the weaker
    # opposite side so resolution copies/shifts from the strong anchor.
    left, right = apply_strong_side_copy_gate(
        left, right, l_n=l_n, r_n=r_n, config=config,
        replace_reject=_replace_reject,
        reject_reason=REJECT_STRONG_SIDE_COPY,
    )

    # 4. Parallel-shape gate.
    left, right = apply_parallel_shape_gate(
        left, right, left_candidate=left_candidate,
        right_candidate=right_candidate, l_n=l_n, r_n=r_n, ys=ys,
        config=config, replace_reject=_replace_reject,
        reject_reason=REJECT_NOT_PARALLEL,
    )

    # 5. Opposite-anchored override. Width source: purple memory if
    # valid, else the caller-supplied lane_area_width_profile.
    width_profile_for_check: np.ndarray | None = None
    if has_memory:
        width_profile_for_check = width_memory.profile
    elif (lane_area_width_profile is not None
          and lane_area_width_profile.size == int(config.num_points)):
        width_profile_for_check = lane_area_width_profile

    left, right = apply_opposite_anchored_gate(
        left, right, cand_lx=cand_lx, cand_rx=cand_rx,
        l_n=l_n, r_n=r_n, width_profile=width_profile_for_check,
        config=config, replace_reject=_replace_reject,
        reject_reason=REJECT_OPPOSITE_ANCHORED_OVERRIDE,
    )

    return left, right


