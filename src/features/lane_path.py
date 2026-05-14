"""Minimal lane-path estimator and raw-mask overlay.

Replaces the legacy fitted-geometry / BEV / occlusion-tracker stack
with a single producer:

    mask_centerline(lane_area) -> dict | None

The result carries enough fields for the controller (``control_x``,
``control_y``), the renderer (``fitted_pts_img``, ``confidence``), and
the raw-mask flag the legacy debug code keys on (``is_raw_mask``).
"""
from __future__ import annotations

import cv2
import numpy as np

# Render constants for the debug overlay.
_LANE_FILL = np.array((90, 220, 110), dtype=np.float32)
_MARKING_COLORS = {
    "dotted_line": np.array((255, 255, 0), dtype=np.float32),
    "solid_line": np.array((255, 0, 255), dtype=np.float32),
}
_MIN_ROW_PIXELS = 6
_MIN_ROWS_FOR_FIT = 6
_TOP_FRACTION = 0.45  # only consider the lower half of the frame


def _row_centroids(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row centroid x, plus the row index and per-row pixel counts."""
    h, w = mask.shape
    bin_mask = (mask > 0).astype(np.float32)
    counts = bin_mask.sum(axis=1)
    valid = counts >= _MIN_ROW_PIXELS
    if not np.any(valid):
        return (np.empty(0, np.int32), np.empty(0, np.float32), np.empty(0, np.float32))
    cols = np.arange(w, dtype=np.float32)
    sums = bin_mask @ cols
    centroids = np.zeros_like(counts)
    centroids[valid] = sums[valid] / counts[valid]
    rows = np.where(valid)[0].astype(np.int32)
    return rows, centroids[valid].astype(np.float32), counts[valid].astype(np.float32)


def mask_centerline(lane_area: np.ndarray) -> dict | None:
    """Fit a per-row centroid centerline to a binary lane mask.

    Returns ``None`` if the mask has no usable rows. Otherwise returns
    a dict carrying control point, fitted polyline, and a confidence
    score derived from row coverage.
    """
    if lane_area is None or lane_area.size == 0:
        return None
    h, w = lane_area.shape
    top_cut = int(h * _TOP_FRACTION)
    region = lane_area[top_cut:]
    rows, cx, counts = _row_centroids(region)
    if rows.size < _MIN_ROWS_FOR_FIT:
        return None
    rows_full = rows + top_cut

    # Fit a quadratic centerline across the surviving rows; fall back to
    # mean-x if the system is degenerate.
    try:
        coeffs = np.polyfit(rows_full.astype(np.float32), cx, 2)
        ys = np.linspace(rows_full[0], h - 1, num=max(8, len(rows_full)))
        xs = np.polyval(coeffs, ys)
    except (np.linalg.LinAlgError, ValueError):
        coeffs = (0.0, 0.0, float(cx.mean()))
        ys = np.linspace(rows_full[0], h - 1, num=max(8, len(rows_full)))
        xs = np.full_like(ys, cx.mean(), dtype=np.float32)

    xs = np.clip(xs, 0.0, w - 1.0)
    fitted_pts = np.stack([xs, ys], axis=1).astype(np.float32)

    # Control point: bottom of the fitted line, biased toward the
    # region the model was most confident about (highest row coverage).
    control_x = int(round(float(xs[-1])))
    control_y = int(round(float(ys[-1])))

    coverage = float(rows.size) / max(1, h - top_cut)
    density = float(counts.mean()) / max(1.0, w)
    confidence = float(np.clip(0.5 * coverage + 0.5 * density, 0.0, 1.0))

    return {
        "is_raw_mask": True,
        "control_x": control_x,
        "control_y": control_y,
        "fitted_pts_img": fitted_pts,
        "confidence": confidence,
        "poly_coeffs": tuple(float(c) for c in coeffs),
        "row_count": int(rows.size),
    }


def _blend_overlay(out_f: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    if not np.any(mask):
        return
    nz = mask > 0
    out_f[nz] = out_f[nz] * (1.0 - alpha) + color * alpha


def render_raw_mask_overlay(
    frame: np.ndarray,
    lane_area: np.ndarray,
    semantics: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Lightweight diagnostic overlay used by tests and the debug panel."""
    out_f = frame.astype(np.float32, copy=True)
    _blend_overlay(out_f, lane_area.astype(np.uint8), _LANE_FILL, alpha=0.30)
    if semantics is not None:
        for key, color in _MARKING_COLORS.items():
            mask = semantics.get(key)
            if mask is None:
                continue
            _blend_overlay(out_f, mask.astype(np.uint8), color, alpha=0.85)
    np.clip(out_f, 0, 255, out=out_f)
    return out_f.astype(np.uint8)
