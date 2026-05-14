"""Typed packets that flow between threaded runtime workers.

Keeps the contract between capture / lane / object / fusion / output
explicit, so workers can be replaced or tested in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from features.state_types import LaneState


@dataclass(slots=True)
class FramePacket:
    """Raw capture output shared with lane + object threads."""
    frame_id: int
    timestamp: float
    frame: np.ndarray
    frame_h: int
    frame_w: int

    @staticmethod
    def make(frame_id: int, timestamp: float, frame: np.ndarray) -> "FramePacket":
        h, w = frame.shape[:2]
        return FramePacket(frame_id=frame_id, timestamp=timestamp, frame=frame,
                           frame_h=h, frame_w=w)


@dataclass(slots=True)
class LanePacket:
    """Lane-branch result for one frame.

    Carries a reference to the source frame so downstream workers can
    render without reaching back to the capture thread. ``corridor`` is
    the lane-mask-driven dynamic guidance dict (centerline + yellow
    boundaries + polygon) when the lane mask is usable, ``None`` otherwise.
    """
    frame_id: int
    timestamp: float
    lane_state: LaneState
    source_frame: np.ndarray
    lane_area: np.ndarray | None = None
    semantics: dict[str, np.ndarray] | None = None
    class_probs: np.ndarray | None = None
    lane_path: dict | None = None
    lane_bounds: dict | None = None
    corridor: dict | None = None


@dataclass(slots=True)
class ObjectPacket:
    """Detection-branch result for one frame."""
    frame_id: int
    timestamp: float
    source_frame: np.ndarray
    detections: list[dict] = field(default_factory=list)
    front_object: dict | None = None


@dataclass(slots=True)
class ControlCommand:
    """Final actuator command produced by fusion + safety supervisor."""
    steering: float = 0.0
    throttle: float = 0.0
    rc_enabled: bool = False
    safe: bool = True


@dataclass(slots=True)
class BehaviorPacket:
    """Fused world-model + behavior decision for one frame.

    Carries everything the output thread needs to draw, save, and
    actuate without reaching back into other workers' state.
    """
    frame_id: int
    timestamp: float
    frame: np.ndarray
    lane: LanePacket | None
    objects: ObjectPacket | None
    behavior_action: str
    behavior_reason: str
    control: ControlCommand
    extras: dict[str, Any] = field(default_factory=dict)
