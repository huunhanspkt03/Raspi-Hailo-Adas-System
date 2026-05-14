"""Deterministic validation for the ego-anchored red path.

Covers the contract from `_lane_path_from_red_center` end-to-end:

* tail of the ego path is always exactly ``ego_x = frame_w / 2``
* slope at the tail is zero (path leaves the camera column vertically)
* with a centered lane target, control_x at the lookahead row sits at ego_x
  (lateral error ~ 0)
* with a left/right-shifted lane target, control_x at the lookahead row
  shifts in the same direction as the target — i.e. the controller will
  steer toward the lane center, not freeze at neutral
* control_x is not the tail point

Run::

    .venv/Scripts/python.exe tools/validate_ego_anchored_path.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from features.ego_anchored_path import (  # noqa: E402
    DEFAULT_LOOKAHEAD_FRAC,
    build_ego_anchored_path,
    ego_path_dxdy_per_y,
    lookahead_y,
)
from features.red_center_from_boundaries import DEFAULT_TOP_FRAC  # noqa: E402

FRAME_H = 480
FRAME_W = 640
NUM_POINTS = 24
EGO_X = 0.5 * FRAME_W


def _make_target_pts(shift_at_top_px: float) -> np.ndarray:
    """Linearly varying lane target: 0 shift at the bottom, ``shift_at_top_px``
    at the top of the ROI. Mimics a lane that diverges from ego with depth."""
    ys = np.linspace(FRAME_H * DEFAULT_TOP_FRAC, FRAME_H - 1,
                     num=NUM_POINTS, dtype=np.float32)
    t = (FRAME_H - 1 - ys) / max(FRAME_H - 1 - ys[0], 1e-3)  # 0 at bottom, 1 at top
    xs = EGO_X + shift_at_top_px * t
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _build(target_pts: np.ndarray) -> tuple[np.ndarray, dict]:
    ys = target_pts[:, 1].astype(np.float32)
    return build_ego_anchored_path(target_pts, ys, FRAME_H, FRAME_W)


def _check(label: str, condition: bool, detail: str = "") -> bool:
    flag = "OK " if condition else "FAIL"
    print(f"[{flag}] {label}{(' — ' + detail) if detail else ''}")
    return condition


def main() -> int:
    failures = 0
    print(f"frame={FRAME_W}x{FRAME_H}  ego_x={EGO_X:.1f}  "
          f"lookahead_frac={DEFAULT_LOOKAHEAD_FRAC}  "
          f"top_frac={DEFAULT_TOP_FRAC}")
    print()

    # 1. Centered lane target
    target = _make_target_pts(0.0)
    xs, diag = _build(target)
    if not _check("centered: tail x == ego_x",
                  abs(float(xs[-1]) - EGO_X) < 1e-3,
                  f"tail x={xs[-1]:.2f}"):
        failures += 1
    if not _check("centered: control_x within 1px of ego_x",
                  abs(diag["ego_path_control_x"] - EGO_X) < 1.0,
                  f"control_x={diag['ego_path_control_x']:.2f}"):
        failures += 1
    if not _check("centered: a == 0",
                  diag["ego_path_a"] == 0.0,
                  f"a={diag['ego_path_a']:.3e}"):
        failures += 1

    # 2. Right-shifted lane target (head 80px right of ego)
    target = _make_target_pts(80.0)
    xs, diag = _build(target)
    a = diag["ego_path_a"]
    if not _check("right shift: tail x == ego_x",
                  abs(float(xs[-1]) - EGO_X) < 1e-3,
                  f"tail x={xs[-1]:.2f}"):
        failures += 1
    if not _check("right shift: control_x > ego_x",
                  diag["ego_path_control_x"] > EGO_X + 5.0,
                  f"control_x={diag['ego_path_control_x']:.2f}"):
        failures += 1
    if not _check("right shift: control_x != tail x",
                  abs(diag["ego_path_control_x"] - float(xs[-1])) > 5.0,
                  f"|ctrl - tail|={abs(diag['ego_path_control_x'] - float(xs[-1])):.2f}"):
        failures += 1
    if not _check("right shift: head bends right (xs[0] > ego_x)",
                  float(xs[0]) > EGO_X + 1.0,
                  f"head x={xs[0]:.2f}"):
        failures += 1
    if not _check("right shift: zero slope at tail",
                  abs(float(ego_path_dxdy_per_y(np.array([xs.shape[0] and float(target[-1, 1])],
                                                          dtype=np.float32), a,
                                                 float(target[-1, 1]))[0])) < 1e-6,
                  "dx/dy at y_tail"):
        failures += 1

    # 3. Left-shifted lane target (head 80px left of ego)
    target = _make_target_pts(-80.0)
    xs, diag = _build(target)
    if not _check("left shift: tail x == ego_x",
                  abs(float(xs[-1]) - EGO_X) < 1e-3,
                  f"tail x={xs[-1]:.2f}"):
        failures += 1
    if not _check("left shift: control_x < ego_x",
                  diag["ego_path_control_x"] < EGO_X - 5.0,
                  f"control_x={diag['ego_path_control_x']:.2f}"):
        failures += 1
    if not _check("left shift: head bends left (xs[0] < ego_x)",
                  float(xs[0]) < EGO_X - 1.0,
                  f"head x={xs[0]:.2f}"):
        failures += 1

    # 4a. Moderate-large shift triggers the top-displacement clamp without
    #     blowing past frame width.
    target = _make_target_pts(300.0)
    xs, diag = _build(target)
    if not _check("clampable shift: tail x == ego_x",
                  abs(float(xs[-1]) - EGO_X) < 1e-3):
        failures += 1
    if not _check("clampable shift: top displacement clamped",
                  bool(diag.get("ego_path_top_disp_clamped", False)),
                  f"a={diag.get('ego_path_a'):.3e}, "
                  f"reason={diag.get('ego_path_reason', '')}"):
        failures += 1
    if not _check("clampable shift: head x within frame",
                  0.0 <= float(xs[0]) < float(FRAME_W),
                  f"head x={xs[0]:.2f}"):
        failures += 1

    # 4b. Absurd shift is hard-rejected so the curve collapses to straight
    #     ego — failsafe path remains at vehicle center.
    target = _make_target_pts(2000.0)
    xs, diag = _build(target)
    if not _check("absurd shift: hard-reject → a == 0",
                  diag["ego_path_a"] == 0.0
                  and diag.get("ego_path_reason") == "curvature_too_large"):
        failures += 1
    if not _check("absurd shift: head stays at ego_x",
                  abs(float(xs[0]) - EGO_X) < 1e-3):
        failures += 1

    # 5. Lookahead row matches DEFAULT_LOOKAHEAD_FRAC
    y_la_expected = FRAME_H * DEFAULT_LOOKAHEAD_FRAC
    if not _check("lookahead y matches frac * frame_h",
                  abs(lookahead_y(FRAME_H) - y_la_expected) < 1.0):
        failures += 1

    # 6. End-to-end: _lane_path_from_red_center (main runtime)
    from pipeline_loop import _lane_path_from_red_center as main_path  # noqa: E402
    target = _make_target_pts(80.0)
    red_center = {
        "valid": True,
        "center_pts_img": target,
        "center_confidence": 0.7,
        "center_coeffs": (0.0, 0.0, 0.0),
    }
    lp = main_path(red_center, FRAME_H, FRAME_W)
    if not _check("main runtime: control_x is the lookahead value",
                  lp is not None
                  and abs(lp["control_x"] - int(round(float(target[-1, 0])))) > 5,
                  f"control_x={lp and lp.get('control_x')}, target tail x={int(round(float(target[-1, 0])))}"):
        failures += 1
    if not _check("main runtime: tail of fitted_pts == ego_x",
                  lp is not None
                  and abs(float(lp["fitted_pts_img"][-1, 0]) - EGO_X) < 1e-3):
        failures += 1
    if not _check("main runtime: control_y matches lookahead row",
                  lp is not None
                  and abs(lp["control_y"] - int(round(y_la_expected))) <= 1):
        failures += 1

    # 7. End-to-end: threaded runtime twin
    from pipeline.thread_workers import _lane_path_from_red_center as threaded_path  # noqa: E402
    lpt = threaded_path(red_center, FRAME_H, FRAME_W)
    if not _check("threaded runtime parity: same control_x",
                  lpt is not None and lp is not None
                  and lpt["control_x"] == lp["control_x"]):
        failures += 1
    if not _check("threaded runtime parity: same fitted_pts tail",
                  lpt is not None
                  and abs(float(lpt["fitted_pts_img"][-1, 0]) - EGO_X) < 1e-3):
        failures += 1

    print()
    print(f"failures: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
