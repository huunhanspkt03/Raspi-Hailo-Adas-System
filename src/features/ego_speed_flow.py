"""Optical-flow helpers for the ego-speed estimator.

Pure helpers extracted from ``EgoSpeedEstimator``: grayscale/mask
preparation, lower-ROI extraction, and dense Farneback flow + outlier
trimming for vertical motion.
"""
from __future__ import annotations

import cv2
import numpy as np

_PERCENTILE_LO = 25.0
_PERCENTILE_HI = 75.0


def prepare_gray_and_mask(
    frame: np.ndarray, lane_area: np.ndarray, downscale: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape[:2]
    if downscale != 1.0:
        new_size = (max(1, int(w * downscale)), max(1, int(h * downscale)))
        gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(
            lane_area.astype(np.uint8), new_size,
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        mask = lane_area.astype(np.uint8, copy=False)
    return gray, mask, downscale


def roi_from_mask(mask: np.ndarray, roi_top_frac: float) -> np.ndarray:
    roi = np.zeros_like(mask)
    top_cut = int(round(mask.shape[0] * roi_top_frac))
    if top_cut < mask.shape[0]:
        roi[top_cut:] = mask[top_cut:]
    return roi


def compute_median_dy(
    prev_gray: np.ndarray, gray: np.ndarray,
    prev_mask: np.ndarray, roi: np.ndarray,
    *, min_flow_px: float, max_flow_px: float, min_points: int,
) -> tuple[float | None, int, float]:
    """Return ``(median_dy, n_used, mad)`` or ``(None, 0, 0.0)`` when no
    flow can be derived. Vertical-only signal: ``dy`` is ``flow[..., 1]``."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=2, winsize=15,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
    )
    valid = (prev_mask > 0) & (roi > 0)
    if int(np.count_nonzero(valid)) < min_points:
        return None, 0, 0.0

    dy_all = flow[..., 1][valid]
    dx_all = flow[..., 0][valid]
    mag = np.sqrt(dx_all * dx_all + dy_all * dy_all)
    keep = (mag >= min_flow_px) & (mag <= max_flow_px)
    dy = dy_all[keep]
    if dy.size < min_points:
        return None, 0, 0.0

    lo, hi = np.percentile(dy, [_PERCENTILE_LO, _PERCENTILE_HI])
    core = dy[(dy >= lo) & (dy <= hi)]
    if core.size < max(20, min_points // 4):
        core = dy
    median_dy = float(np.median(core))
    mad = float(np.median(np.abs(core - median_dy)))
    return median_dy, int(core.size), mad
