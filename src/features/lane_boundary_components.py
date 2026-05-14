"""Connected-component analysis + lane-edge association for lane_boundaries.

For each connected component of the line-mask union we compute its bbox,
area, principal-axis angle (PCA via cv2.fitLine), and assign it to the
``left`` or ``right`` side based on median lateral distance from the
per-row ``lane_area`` edges. Components that are too small, too short,
mostly horizontal, or too far from any lane edge are rejected with a
``rejected_reason`` code that the debug renderer picks up.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

REJECT_AREA = "small_area"
REJECT_HEIGHT = "short"
REJECT_ASPECT = "horizontal_box"
REJECT_ANGLE = "horizontal_axis"
REJECT_FAR_FROM_LANE = "far_from_lane"
REJECT_NO_LANE_OVERLAP = "no_lane_overlap"

REJECT_REASONS = (REJECT_AREA, REJECT_HEIGHT, REJECT_ASPECT, REJECT_ANGLE,
                  REJECT_FAR_FROM_LANE, REJECT_NO_LANE_OVERLAP)


@dataclass(slots=True)
class ComponentInfo:
    """Per-component summary + side assignment + reject reason."""
    label: int
    bbox: tuple[int, int, int, int]   # (x, y, w, h) in frame coords
    n_pixels: int
    centroid: tuple[float, float]      # (x, y)
    aspect_ratio: float                # bbox_w / max(1, bbox_h)
    angle_from_vertical_deg: float     # 0=vertical, 90=horizontal
    pixels_y: np.ndarray = field(repr=False)
    pixels_x: np.ndarray = field(repr=False)
    side: str = "unknown"              # "left" | "right" | "unknown"
    edge_distance_px: float = float("inf")
    rejected_reason: str | None = None


def extract_lane_area_edges(
    lane_area: np.ndarray | None, top_cut: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row leftmost/rightmost ``lane_area`` x within the ROI rows.

    Returns ``(left, right, center)`` length ``frame_h - top_cut``, with
    NaN entries where the row had no lane_area pixel.
    """
    if lane_area is None:
        return np.empty(0, np.float32), np.empty(0, np.float32), np.empty(0, np.float32)
    region = (lane_area[top_cut:] > 0).astype(np.uint8)
    h, w = region.shape
    left = np.full(h, np.nan, dtype=np.float32)
    right = np.full(h, np.nan, dtype=np.float32)
    has = region.any(axis=1)
    if np.any(has):
        first = region.argmax(axis=1).astype(np.float32)
        last = (w - 1 - region[:, ::-1].argmax(axis=1)).astype(np.float32)
        left[has] = first[has]
        right[has] = last[has]
    center = 0.5 * (left + right)
    return left, right, center


def _angle_from_vertical_deg(pts_xy: np.ndarray) -> float:
    """Principal-axis angle from vertical via cv2.fitLine. 0=vertical, 90=horizontal."""
    if pts_xy.shape[0] < 4:
        return 90.0
    line = cv2.fitLine(pts_xy.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(line[0, 0]), float(line[1, 0])
    return float(np.degrees(np.arctan2(abs(vx), abs(vy) + 1e-9)))


def analyze_components(
    mask: np.ndarray,
    *,
    min_area: int,
    min_height: int,
    max_aspect: float,
    max_angle_deg: float,
) -> list[ComponentInfo]:
    """Per-component geometry + an early reject for non-line shapes."""
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8,
    )
    out: list[ComponentInfo] = []
    for label in range(1, n_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx, cy = float(centroids[label, 0]), float(centroids[label, 1])
        sub = labels[y:y + h, x:x + w]
        ys_local, xs_local = np.where(sub == label)
        if ys_local.size == 0:
            continue
        ys = (ys_local + y).astype(np.int32)
        xs = (xs_local + x).astype(np.float32)
        aspect = float(w) / max(1.0, float(h))
        ang = _angle_from_vertical_deg(np.column_stack([xs, ys.astype(np.float32)]))
        reason: str | None = None
        if area < min_area:
            reason = REJECT_AREA
        elif h < min_height:
            reason = REJECT_HEIGHT
        elif aspect > max_aspect:
            reason = REJECT_ASPECT
        elif ang > max_angle_deg:
            reason = REJECT_ANGLE
        out.append(ComponentInfo(
            label=label, bbox=(x, y, w, h), n_pixels=area, centroid=(cx, cy),
            aspect_ratio=aspect, angle_from_vertical_deg=ang,
            pixels_y=ys, pixels_x=xs,
            rejected_reason=reason,
        ))
    return out


def assign_sides_by_lane_edges(
    components: list[ComponentInfo],
    lane_left: np.ndarray,
    lane_right: np.ndarray,
    top_cut: int,
    max_edge_dist_px: float,
) -> None:
    """Mutate components in place: side='left'|'right' or REJECT_FAR_FROM_LANE."""
    if lane_left.size == 0:
        # No lane_area — caller falls back to a constant centroid split.
        for comp in components:
            if comp.rejected_reason is None:
                comp.side = "unknown"
        return
    for comp in components:
        if comp.rejected_reason is not None:
            continue
        ys_local = comp.pixels_y - top_cut
        valid = (ys_local >= 0) & (ys_local < lane_left.size)
        if not np.any(valid):
            comp.rejected_reason = REJECT_NO_LANE_OVERLAP
            continue
        ysv = ys_local[valid]
        xsv = comp.pixels_x[valid]
        le = lane_left[ysv]
        ri = lane_right[ysv]
        defined = ~np.isnan(le) & ~np.isnan(ri)
        if not np.any(defined):
            comp.rejected_reason = REJECT_NO_LANE_OVERLAP
            continue
        med_dl = float(np.median(np.abs(xsv[defined] - le[defined])))
        med_dr = float(np.median(np.abs(xsv[defined] - ri[defined])))
        if med_dl <= med_dr:
            if med_dl > max_edge_dist_px:
                comp.rejected_reason = REJECT_FAR_FROM_LANE
                continue
            comp.side = "left"
            comp.edge_distance_px = med_dl
        else:
            if med_dr > max_edge_dist_px:
                comp.rejected_reason = REJECT_FAR_FROM_LANE
                continue
            comp.side = "right"
            comp.edge_distance_px = med_dr


def assign_sides_by_split(
    components: list[ComponentInfo], side_split_x: float,
) -> None:
    """Assign side purely by centroid x vs ``side_split_x`` (line-object based,
    no lane_area distance gate)."""
    split = float(side_split_x)
    for comp in components:
        if comp.rejected_reason is not None:
            continue
        comp.side = "left" if comp.centroid[0] < split else "right"


def collect_side_pixels(
    components: list[ComponentInfo], side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate (ys, xs) from accepted components on one side."""
    yss: list[np.ndarray] = []
    xss: list[np.ndarray] = []
    for comp in components:
        if comp.rejected_reason is None and comp.side == side:
            yss.append(comp.pixels_y)
            xss.append(comp.pixels_x)
    if not yss:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    return (np.concatenate(yss).astype(np.int32),
            np.concatenate(xss).astype(np.float32))
