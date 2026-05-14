"""Per-side state machine resolution + purple lane width memory.

Per-frame, per-side resolution order:

    DETECTED
      -> HELD
      -> LOST

Missing-side inference is intentionally disabled. If the model sees only
one lane marking, the other side remains ``LOST`` unless that same side
can be briefly ``HELD`` from its own recent detections. This prevents a
synthetic second line from pulling the red center path.

``PurpleWidthMemory`` still stores the EMA-smoothed per-row purple width
profile from clean BOTH_DETECTED frames. It is kept for diagnostics and
validation gates, but no longer generates an inferred side.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from features.lane_boundary_tracker_contract import (
    SOURCE_DETECTED, SOURCE_HELD, SOURCE_LOST,
    detected_confidence, held_confidence, lost_confidence,
)
from features.lane_boundary_tracker_geom import sample_width_profile

DEFAULT_WIDTH_EMA_ALPHA = 0.30

# Legacy labels kept for older metrics readers. New tracker output does not
# synthesize missing sides.
INFERENCE_KIND_PURPLE = "purple_memory"
INFERENCE_KIND_WIDTH_X_PROFILE = "width_x_profile"
INFERENCE_KIND_LANE_AREA_BOOTSTRAP = "lane_area_bootstrap"
BOOTSTRAP_CONFIDENCE_SCALE = 0.6


@dataclass(slots=True)
class PurpleWidthMemory:
    """EMA-smoothed per-row purple width on the tracker ROI grid."""

    top_frac: float
    num_points: int
    ema_alpha: float = DEFAULT_WIDTH_EMA_ALPHA
    profile: np.ndarray | None = None
    profile_frame_h: int | None = None

    def reset(self) -> None:
        self.profile = None
        self.profile_frame_h = None

    def is_valid(self, frame_h: int) -> bool:
        return (self.profile is not None
                and self.profile_frame_h == int(frame_h)
                and self.profile.size == self.num_points)

    def median_px(self) -> float | None:
        if self.profile is None:
            return None
        return float(np.median(self.profile))

    def update(self, left_coeffs, right_coeffs, frame_h: int) -> None:
        profile = sample_width_profile(
            left_coeffs, right_coeffs, frame_h,
            top_frac=self.top_frac, num_points=self.num_points,
        )
        if profile is None:
            return
        if not self.is_valid(frame_h):
            self.profile = profile.copy()
            self.profile_frame_h = int(frame_h)
            return
        a = float(self.ema_alpha)
        self.profile = ((1.0 - a) * self.profile + a * profile).astype(np.float32)


@dataclass(slots=True)
class ResolveConfig:
    """Per-tracker config snapshot passed to ``resolve_side``."""

    top_frac: float
    num_points: int
    min_inference_width_px: float
    max_inferred_streak: int
    held_frame_limit: int


@dataclass(slots=True)
class ResolvedSide:
    """Return value of ``resolve_side``."""

    source: str
    coeffs: tuple[float, float, float] | None
    confidence: float
    inference_kind: str | None


def resolve_side(
    side, infer_for: str, *,
    was_detected: bool, other_state, other_detected: bool,
    fit: dict, frame_h: int,
    width_memory: PurpleWidthMemory,
    width_x_learner,
    config: ResolveConfig,
) -> ResolvedSide:
    """Resolve a side without synthesizing missing boundaries.

    The opposite side, purple width memory, width-x learner, and live
    lane_area edges are intentionally ignored for missing-side geometry.
    They remain in the signature so the tracker API stays stable.
    """
    del infer_for, other_state, other_detected, fit, frame_h, width_memory
    del width_x_learner

    if was_detected:
        side.raw_missing_streak = 0
        side.inferred_streak = 0
        side.held_streak = 0
        side.output_missing_streak = 0
        return ResolvedSide(
            source=SOURCE_DETECTED, coeffs=side.coeffs,
            confidence=detected_confidence(side.last_detected_confidence),
            inference_kind=None,
        )

    side.raw_missing_streak += 1
    side.inferred_streak = 0

    if (side.coeffs is not None and side.last_detected_confidence > 0.0
            and side.held_streak < int(config.held_frame_limit)):
        side.held_streak += 1
        side.output_missing_streak = 0
        return ResolvedSide(
            source=SOURCE_HELD, coeffs=side.coeffs,
            confidence=held_confidence(
                side.last_detected_confidence, side.held_streak,
                decay_frames=config.held_frame_limit,
            ),
            inference_kind=None,
        )

    side.output_missing_streak += 1
    return ResolvedSide(
        source=SOURCE_LOST, coeffs=None,
        confidence=lost_confidence(), inference_kind=None,
    )
