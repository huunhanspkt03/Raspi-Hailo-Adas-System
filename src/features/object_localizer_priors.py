"""Priors, constants, and small data classes for monocular object localization."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ObjectPrior:
    """Soft real-world size prior for one detection class."""

    height_m: float
    width_m: float
    min_height_m: float
    max_height_m: float
    min_width_m: float
    max_width_m: float


@dataclass(slots=True)
class _TrackPose:
    z_m: float
    x_m: float
    confidence: float


@dataclass(slots=True)
class FrontObject:
    valid: bool = False
    det: dict | None = None


_REF_PRIORS_M: dict[str, ObjectPrior] = {
    "bus": ObjectPrior(3.2, 2.5, 2.7, 3.8, 2.2, 2.8),
    "car": ObjectPrior(1.5, 1.8, 1.25, 1.9, 1.55, 2.1),
    "green_light": ObjectPrior(0.6, 0.3, 0.35, 0.9, 0.18, 0.45),
    # Effective motorcycle bbox dimensions include rider + visual footprint.
    "motorcycle": ObjectPrior(1.0, 0.8, 0.75, 1.45, 0.45, 1.15),
    "red_light": ObjectPrior(0.6, 0.3, 0.35, 0.9, 0.18, 0.45),
    "truck": ObjectPrior(3.4, 2.5, 2.4, 4.2, 2.0, 2.8),
}
_COMPACT_MOTORCYCLE_PRIOR_M = ObjectPrior(0.62, 0.36, 0.45, 0.90, 0.25, 0.55)

_MIN_DEPTH_M = 0.3
_MAX_DEPTH_M = 80.0
_CLIP_MARGIN_PX = 3
_HORIZON_GUARD_PX = 8.0
_TRAFFIC_LIGHT_STABLE_DY_PX = 40.0
_TRAFFIC_LIGHT_GROUP_GAP_RATIO = 0.18
_TRAFFIC_LIGHT_GROUP_MIN_GAP_PX = 120.0
_MIN_CONFIDENCE = 0.05
_TRAFFIC_LIGHT_LABELS = {"red_light", "green_light"}
