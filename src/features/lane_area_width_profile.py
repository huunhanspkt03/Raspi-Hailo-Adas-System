"""100-frame per-row lane_area width-x learner.

Owned by ``LaneBoundaryTracker``. During the first ``learning_frames``
frames that supply a valid lane_area row-width profile, each sample is
appended to a per-row history. Once enough samples are collected, the
median per row is frozen as the stable ``profile``.

Missing-side inference is disabled, so this learner is diagnostic-only
for now. After freeze, the profile is held; fresh per-frame lane_area
edges no longer overwrite it.

If the profile is not ready, callers can report readiness/skip diagnostics
without synthesizing a boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = (
    "DEFAULT_LEARNING_FRAMES",
    "LaneAreaWidthLearner",
    # Back-compat alias kept so legacy imports still resolve.
    "LaneAreaWidthMemory",
)

DEFAULT_LEARNING_FRAMES = 100


@dataclass(slots=True)
class LaneAreaWidthLearner:
    """Per-row width-x(y) profile learned over ``learning_frames`` frames.

    Collects valid lane_area edge widths frame-by-frame, then freezes a
    stable median-per-row profile. Once frozen, the profile is held; new
    samples are ignored unless ``reset()`` is called.
    """
    learning_frames: int = DEFAULT_LEARNING_FRAMES
    profile: np.ndarray | None = None
    samples_collected: int = 0
    last_skip_reason: str | None = None
    _samples: list[np.ndarray] = field(default_factory=list)

    def reset(self) -> None:
        self.profile = None
        self.samples_collected = 0
        self.last_skip_reason = None
        self._samples = []

    def is_valid(self) -> bool:
        return self.profile is not None and self.profile.size > 0

    def is_ready(self) -> bool:
        return self.is_valid()

    @property
    def age(self) -> int:
        return int(self.samples_collected)

    def median_px(self) -> float | None:
        return float(np.median(self.profile)) if self.is_valid() else None

    def min_px(self) -> float | None:
        return float(np.min(self.profile)) if self.is_valid() else None

    def max_px(self) -> float | None:
        return float(np.max(self.profile)) if self.is_valid() else None

    def maybe_update(
        self,
        raw_profile: np.ndarray | None,
        frame_id: int,
        frame_w: int,
    ) -> tuple[bool, str | None]:
        """Append ``raw_profile`` to the learning buffer and freeze once
        enough samples are collected.

        Returns ``(changed, reject_reason)`` where ``changed`` is True
        only on the frame that promotes the buffer to a frozen profile.
        ``reject_reason`` describes why the buffer was not appended.
        """
        _ = frame_id, frame_w
        if self.is_ready():
            self.last_skip_reason = "frozen"
            return False, "frozen"
        if raw_profile is None or raw_profile.size == 0:
            self.last_skip_reason = "no_lane_area_profile"
            return False, "no_lane_area_profile"
        sample = raw_profile.astype(np.float32, copy=True)
        if not bool(np.any(sample > 0.0)):
            self.last_skip_reason = "no_lane_area_profile"
            return False, "no_lane_area_profile"
        if self._samples and sample.size != self._samples[0].size:
            self.last_skip_reason = "shape_mismatch"
            return False, "shape_mismatch"
        self._samples.append(sample)
        self.samples_collected = len(self._samples)
        self.last_skip_reason = None
        if self.samples_collected >= int(self.learning_frames):
            stacked = np.stack(self._samples, axis=0).astype(np.float32)
            self.profile = np.median(stacked, axis=0).astype(np.float32)
            self._samples = []
            return True, None
        return False, "learning"


# Back-compat alias. Older modules import ``LaneAreaWidthMemory``; both
# names point at the same learner now.
LaneAreaWidthMemory = LaneAreaWidthLearner
