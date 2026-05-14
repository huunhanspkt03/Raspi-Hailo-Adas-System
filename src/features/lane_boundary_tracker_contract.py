"""Source enum, aggregate-state policy, and confidence-decay rules.

Pure constants + pure functions, no state and no I/O. Imported by the
``LaneBoundaryTracker`` and the test suite so the Step-3 output contract is
explicit and independently testable. Step 4 (red center fusion) reads from
this contract — it must NOT inspect the tracker's private streak counters.
"""
from __future__ import annotations

# ---- Per-side source ------------------------------------------------------
# Where a side's final coefficients came from this frame.
SOURCE_DETECTED = "DETECTED"   # accepted from current YOLO line-mask fit
SOURCE_HELD = "HELD"           # cached coefficients from a previous DETECTED
SOURCE_LOST = "LOST"           # no usable coefficients (output is unusable)

ALL_SOURCES = (SOURCE_DETECTED, SOURCE_HELD, SOURCE_LOST)

# ---- Aggregate lane state (the per-frame headline) ------------------------
LANE_BOTH_DETECTED = "BOTH_DETECTED"
LANE_ONE_SIDE_DETECTED = "ONE_SIDE_DETECTED"
LANE_HELD_PREVIOUS = "HELD_PREVIOUS"
LANE_LOST = "LOST"

# ---- Confidence policy ----------------------------------------------------
# DETECTED: pass-through of the raw fit confidence.
# HELD:     last-detected confidence × linear streak decay over HELD_DECAY_FRAMES.
# LOST:     0.0.
HELD_DECAY_FRAMES = 8


def aggregate_lane_state(left_source: str, right_source: str) -> str:
    """Per-frame aggregate of two per-side sources. See enum above."""
    if left_source == SOURCE_DETECTED and right_source == SOURCE_DETECTED:
        return LANE_BOTH_DETECTED
    if left_source == SOURCE_DETECTED and right_source in (SOURCE_HELD, SOURCE_LOST):
        return LANE_ONE_SIDE_DETECTED
    if right_source == SOURCE_DETECTED and left_source in (SOURCE_HELD, SOURCE_LOST):
        return LANE_ONE_SIDE_DETECTED
    if left_source == SOURCE_LOST and right_source == SOURCE_LOST:
        return LANE_LOST
    return LANE_HELD_PREVIOUS


def detected_confidence(raw_confidence: float) -> float:
    return float(max(0.0, min(1.0, raw_confidence)))


def held_confidence(last_detected_confidence: float, held_streak: int,
                    decay_frames: int = HELD_DECAY_FRAMES) -> float:
    """Linear decay to 0 by ``decay_frames`` consecutive held frames."""
    decay = max(0.0, 1.0 - float(held_streak) / float(max(1, decay_frames)))
    return float(max(0.0, min(1.0, last_detected_confidence * decay)))


def lost_confidence() -> float:
    return 0.0


def boundary_confidence(left_conf_final: float, right_conf_final: float) -> float:
    """Conservative combined score — the weakest important side."""
    return float(min(left_conf_final, right_conf_final))
