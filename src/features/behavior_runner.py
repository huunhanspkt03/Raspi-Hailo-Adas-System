"""Small wrapper to run behavior logic outside the perception pipeline.

This module centralizes the behavior decision call so pipeline_loop no
longer performs behavior/control directly.
"""
from __future__ import annotations

from features.behavior_logic import compute_behavior_state
from features.state_types import BehaviorState


def run_behavior(detections: list, ctrl_cfg: dict, frame_w: int, behavior_smoother) -> BehaviorState:
    """Run behavior decision and apply optional smoothing.

    Returns a BehaviorState instance.
    """
    raw_state = compute_behavior_state(detections, ctrl_cfg, frame_w)
    state = behavior_smoother.update(raw_state) if behavior_smoother else raw_state
    return state
