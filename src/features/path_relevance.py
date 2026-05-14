"""Path-relevance helpers — decide which detections sit in the corridor.

Pure geometry: point-in-polygon, bbox-vs-polygon overlap, and the
``mark_in_path`` / ``reclassify_detections`` entry points used by
``ObjectLocalizer`` and ``FusionWorker``. The corridor polygon itself
is built in ``features.lane_corridor`` (fixed or lane-area-aware).
"""
from __future__ import annotations

import cv2
import numpy as np

_FALLBACK_CENTER_GATE_RATIO = 0.18


def _polygon_for_test(polygon: np.ndarray) -> np.ndarray:
    return np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)


def point_in_corridor(polygon: np.ndarray | None, x: float, y: float) -> bool:
    if polygon is None or len(polygon) < 3:
        return False
    poly = _polygon_for_test(polygon)
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def _segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """Strict 2D segment intersection test (excludes pure touch)."""
    def _orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def bbox_intersects_corridor(
    polygon: np.ndarray | None,
    x1: float, y1: float, x2: float, y2: float,
) -> bool:
    """True if the axis-aligned bbox overlaps the corridor polygon."""
    if polygon is None or len(polygon) < 3:
        return False
    poly = _polygon_for_test(polygon)
    corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    if any(cv2.pointPolygonTest(poly, c, False) >= 0 for c in corners):
        return True
    poly_arr = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    inside_bbox = (
        (poly_arr[:, 0] >= x1) & (poly_arr[:, 0] <= x2)
        & (poly_arr[:, 1] >= y1) & (poly_arr[:, 1] <= y2)
    )
    if bool(np.any(inside_bbox)):
        return True
    bbox_edges = (
        (np.array([x1, y1]), np.array([x2, y1])),
        (np.array([x2, y1]), np.array([x2, y2])),
        (np.array([x2, y2]), np.array([x1, y2])),
        (np.array([x1, y2]), np.array([x1, y1])),
    )
    n = len(poly_arr)
    for i in range(n):
        a = poly_arr[i]
        b = poly_arr[(i + 1) % n]
        for c, d in bbox_edges:
            if _segments_cross(a, b, c, d):
                return True
    return False


def mark_in_path(
    det: dict,
    polygon: np.ndarray | None,
    *,
    frame_w: int,
    frame_h: int,
    fallback_center_gate_ratio: float = _FALLBACK_CENTER_GATE_RATIO,
    min_bottom_ratio: float = 0.45,
) -> bool:
    """Set ``det['in_path']`` from corridor geometry, falling back to center-gate.

    The fallback path mirrors the original ``ObjectLocalizer`` heuristic so
    behavior stays sane when no polygon is supplied (e.g., callers that
    intentionally skip corridor reasoning). Returns the new ``in_path``.
    """
    box = det.get("box")
    if box is None:
        det["in_path"] = False
        return False
    x1, y1, x2, y2 = (float(v) for v in box)
    if polygon is not None and len(polygon) >= 3:
        foot_x = float(det.get("foot_x", det.get("bottom_center_x", (x1 + x2) * 0.5)))
        foot_y = float(det.get("foot_y", det.get("bottom_center_y", y2)))
        if point_in_corridor(polygon, foot_x, foot_y):
            det["in_path"] = True
            return True
        if bbox_intersects_corridor(polygon, x1, y1, x2, y2):
            det["in_path"] = True
            return True
        det["in_path"] = False
        return False
    cx = 0.5 * float(frame_w - 1)
    gate_px = fallback_center_gate_ratio * float(frame_w)
    bottom = max(y2, det.get("foot_y", y2))
    near_center = (x2 >= cx - gate_px) and (x1 <= cx + gate_px)
    deep_enough = bottom >= min_bottom_ratio * float(frame_h)
    in_path = bool(near_center and deep_enough)
    det["in_path"] = in_path
    return in_path


def reclassify_detections(
    detections: list[dict],
    polygon: np.ndarray | None,
    frame_w: int,
    frame_h: int,
    *,
    skip_labels: frozenset[str] = frozenset({"red_light", "green_light"}),
) -> None:
    """Update ``in_path`` on each detection in-place using the corridor."""
    for det in detections:
        if det.get("label") in skip_labels:
            det["in_path"] = False
            continue
        mark_in_path(det, polygon, frame_w=frame_w, frame_h=frame_h)
