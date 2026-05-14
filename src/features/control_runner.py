"""Wrapper for RC/control actuation so perception stays separated.
"""
from __future__ import annotations


def send_rc_command(send_rc_fn, lane_state, throttle: float, rc_cfg: dict) -> None:
    enabled = bool(rc_cfg.get("enabled", False)) if rc_cfg is not None else False
    # send_rc_fn signature: (steering: float, throttle: float, enabled: bool)
    send_rc_fn(lane_state.steering if lane_state is not None else 0.0, throttle, enabled)
