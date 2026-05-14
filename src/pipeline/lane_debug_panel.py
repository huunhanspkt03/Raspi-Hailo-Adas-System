"""2x2 lane debug panel — minimal, no BEV.

Tile layout:
  top-left     original frame
  top-right    lane_area mask, color-coded
  bottom-left  semantic markings overlay
  bottom-right original frame + centerline + control point
"""
from __future__ import annotations

import cv2
import numpy as np

from features.lane_path import render_raw_mask_overlay

_LANE_AREA_COLOR = (90, 220, 110)
_CENTERLINE_COLOR = (0, 0, 255)
_CONTROL_RING = (0, 0, 0)
_CONTROL_DOT = (255, 255, 255)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    width = max(160, 12 + 10 * len(text))
    cv2.rectangle(out, (0, 0), (width, 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _lane_area_view(frame_shape: tuple[int, int, int], lane_area: np.ndarray | None) -> np.ndarray:
    h, w = frame_shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if lane_area is not None and np.any(lane_area):
        canvas[lane_area > 0] = _LANE_AREA_COLOR
    return canvas


def _markings_view(frame: np.ndarray, semantics: dict[str, np.ndarray] | None) -> np.ndarray:
    if semantics is None:
        return frame.copy()
    return render_raw_mask_overlay(frame, np.zeros(frame.shape[:2], dtype=np.uint8), semantics)


def _path_view(frame: np.ndarray, lane_path: dict | None) -> np.ndarray:
    out = frame.copy()
    if lane_path is None:
        return out
    pts = lane_path.get("fitted_pts_img")
    if pts is not None and len(pts) >= 2:
        curve = np.round(np.asarray(pts)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [curve], False, _CENTERLINE_COLOR, 3, cv2.LINE_AA)
    cx = lane_path.get("control_x")
    cy = lane_path.get("control_y")
    if cx is not None and cy is not None:
        cv2.circle(out, (int(cx), int(cy)), 6, _CONTROL_RING, 2, cv2.LINE_AA)
        cv2.circle(out, (int(cx), int(cy)), 4, _CONTROL_DOT, -1, cv2.LINE_AA)
    conf = float(lane_path.get("confidence", 0.0)) if lane_path else 0.0
    cv2.putText(out, f"conf={conf:.2f}", (12, out.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_lane_debug_panel(
    frame: np.ndarray,
    semantics: dict[str, np.ndarray] | None,
    lane_path: dict | None,
) -> np.ndarray:
    """Compose a 2x2 debug panel from the current lane inputs and result."""
    lane_area = None if semantics is None else semantics.get("lane_area")
    top_left = _label(frame.copy(), "original")
    top_right = _label(_lane_area_view(frame.shape, lane_area), "lane_area")
    bottom_left = _label(_markings_view(frame, semantics), "markings")
    bottom_right = _label(_path_view(frame, lane_path), "centerline")
    top = np.hstack([top_left, top_right])
    bottom = np.hstack([bottom_left, bottom_right])
    return np.vstack([top, bottom])
