"""Debug renderer for the purple-boundary fit + accepted/rejected components.

Used by ``tools/preview_purple_boundaries.py`` and unit tests. NOT wired
into the live overlay path — that is step 5.
"""
from __future__ import annotations

import cv2
import numpy as np

from features.lane_boundary_tracker_contract import SOURCE_DETECTED

PURPLE_BGR = (200, 80, 200)
ACCEPT_BGR_LEFT = (255, 200, 0)        # cyan-ish boxes for left-accepted
ACCEPT_BGR_RIGHT = (0, 200, 255)       # orange-ish boxes for right-accepted
REJECT_BGR = (60, 60, 220)             # red-ish boxes for rejected
LANE_LEFT_EDGE_BGR = (200, 100, 255)   # pink lane_area left-edge polyline
LANE_RIGHT_EDGE_BGR = (255, 200, 100)  # sky-blue lane_area right-edge polyline
SHADOW_BGR = (255, 255, 255)


def render_boundary_polylines(out: np.ndarray, fit: dict, *, thickness: int = 2) -> np.ndarray:
    """Draw model-detected purple boundaries onto ``out`` (in place).

    Per spec: only sides marked ``SOURCE_DETECTED`` are drawn. Inferred,
    held, and lost sides are skipped — the displayed purple matches what
    the lane segmentation model actually produced this frame.
    """
    sides = (("left_source", "left_pts_img"), ("right_source", "right_pts_img"))
    has_source_keys = "left_source" in fit or "right_source" in fit
    for src_key, pts_key in sides:
        if has_source_keys and fit.get(src_key) != SOURCE_DETECTED:
            continue
        pts = fit.get(pts_key)
        if pts is None or len(pts) < 2:
            continue
        curve = np.round(np.asarray(pts)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [curve], False, SHADOW_BGR, thickness + 2, cv2.LINE_AA)
        cv2.polylines(out, [curve], False, PURPLE_BGR, thickness, cv2.LINE_AA)
    return out


def render_lane_area_edges(out: np.ndarray, fit: dict, *, thickness: int = 1) -> np.ndarray:
    """Draw lane_area left/right edge polylines (debug)."""
    for key, color in (("lane_area_left_pts", LANE_LEFT_EDGE_BGR),
                       ("lane_area_right_pts", LANE_RIGHT_EDGE_BGR)):
        pts = fit.get(key)
        if pts is None or len(pts) < 2:
            continue
        curve = np.round(np.asarray(pts)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [curve], False, color, thickness, cv2.LINE_AA)
    return out


def render_components(
    out: np.ndarray, fit: dict, *, draw_text: bool = True,
) -> np.ndarray:
    """Draw bbox per component, color-coded by side or reject reason."""
    for comp in fit.get("components") or ():
        x, y, w, h = comp.bbox
        if comp.rejected_reason is not None:
            color, label = REJECT_BGR, comp.rejected_reason[:6]
        elif comp.side == "left":
            color, label = ACCEPT_BGR_LEFT, "L"
        elif comp.side == "right":
            color, label = ACCEPT_BGR_RIGHT, "R"
        else:
            color, label = (180, 180, 180), "?"
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)
        if draw_text:
            cv2.putText(out, label, (x, max(10, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def render_full_debug(out: np.ndarray, fit: dict) -> np.ndarray:
    """Convenience: components + lane edges + purple polylines."""
    render_components(out, fit)
    render_lane_area_edges(out, fit)
    render_boundary_polylines(out, fit)
    return out
