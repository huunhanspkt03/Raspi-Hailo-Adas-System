"""Purple width-memory update gating.

Pure decision function. The tracker calls this each frame to decide
whether the current candidate pair is clean enough to refresh
``PurpleWidthMemory``. Split out from ``lane_boundary_tracker`` so the
gating policy is independently testable and the tracker module stays
focused on the per-frame state machine.

The headline rules — derived from replay metrics that showed the
previous "any BOTH_DETECTED frame updates memory" policy contaminating
the profile from fake/weak/lane_area-proxy pairs:

* Both sides must be freshly DETECTED.
* No ``support_weak`` or ``pinch_detected`` diagnostic.
* On the FIRST update (no memory yet), cross-check the candidate pair
  width against ``lane_area_width_px``. Without this warm-up a single
  bad first frame can anchor the profile.
"""
from __future__ import annotations

import numpy as np

from features.lane_boundary_tracker_geom import sample_width_profile
from features.lane_boundary_tracker_resolve import PurpleWidthMemory

__all__ = (
    "DEFAULT_MEMORY_WARMUP_LANE_AREA_TOLERANCE_PX",
    "MemoryUpdateConfig",
    "evaluate_memory_update",
)

DEFAULT_MEMORY_WARMUP_LANE_AREA_TOLERANCE_PX = 100.0


class MemoryUpdateConfig:
    """Per-tracker thresholds for the warm-up cross-check."""
    __slots__ = ("top_frac", "num_points", "warmup_lane_area_tolerance_px")

    def __init__(self, *, top_frac: float, num_points: int,
                 warmup_lane_area_tolerance_px: float) -> None:
        self.top_frac = float(top_frac)
        self.num_points = int(num_points)
        self.warmup_lane_area_tolerance_px = float(warmup_lane_area_tolerance_px)


def evaluate_memory_update(
    *,
    l_detected: bool, r_detected: bool,
    l_validation, r_validation,
    frame_h: int,
    left_coeffs, right_coeffs,
    width_memory: PurpleWidthMemory,
    lane_area_width_px: float | None,
    lane_area_width_valid: bool,
    config: MemoryUpdateConfig,
) -> tuple[bool, str | None]:
    """Return ``(allowed, reject_reason)``. The caller is responsible for
    actually mutating ``width_memory`` when ``allowed`` is True."""
    if not (l_detected and r_detected):
        return False, "not_both_detected"
    # Both DETECTED implies validations were accepted (the tracker rejects
    # un-accepted candidates upstream of ``_try_detect``); we still check
    # for soft diagnostic flags that survive an accept.
    if l_validation.support_weak or r_validation.support_weak:
        return False, "weak_support"
    if l_validation.pinch_detected or r_validation.pinch_detected:
        return False, "corridor_pinch"
    # Warm-up cross-check on the first update. Once memory is valid the
    # cross-side gates in ``validate_pair`` carry the load.
    if not width_memory.is_valid(frame_h):
        if not (lane_area_width_valid and lane_area_width_px is not None):
            return False, "warmup_no_lane_area"
        pair_profile = sample_width_profile(
            left_coeffs, right_coeffs, frame_h,
            top_frac=config.top_frac, num_points=config.num_points,
        )
        if pair_profile is None or pair_profile.size == 0:
            return False, "warmup_no_pair_profile"
        # Match ``lane_width_diagnostic``'s bottom-half median so the
        # comparison is apples-to-apples (lane_area_width_px is the
        # bottom-half median across the lane_area edge grid).
        half = pair_profile[len(pair_profile) // 2:]
        pair_width = float(np.median(half if half.size else pair_profile))
        if (abs(pair_width - float(lane_area_width_px))
                > config.warmup_lane_area_tolerance_px):
            return False, "warmup_disagree_lane_area"
    return True, None
