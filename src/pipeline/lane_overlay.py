"""Internal blending helpers for the lane-area + semantic-marking layer.

These power the green ``lane_area`` fill and the semantic line markings
drawn underneath the guidance overlay. The current YOLO26s-seg export has
3 model classes: dotted_line, lane_area, and solid_line.
"""
from __future__ import annotations

import cv2
import numpy as np

_LANE_FILL = (30, 190, 45)
LANE_FILL_ARR = np.array(_LANE_FILL, dtype=np.float32).reshape(1, 1, 3)
MARKING_COLORS: dict[str, tuple[int, int, int]] = {
    "dotted_line": (255, 255, 0),
    "solid_line": (255, 0, 255),
}
MARKING_COLOR_ARRAYS = {
    key: np.array(color, dtype=np.float32).reshape(1, 1, 3)
    for key, color in MARKING_COLORS.items()
}
_MARKING_DILATE_KERNEL = np.ones((3, 3), np.uint8)
_CLASS_KEY_TO_ID = {
    "dotted_line": 1,
    "lane_area": 2,
    "solid_line": 3,
}


def _soft_blend_mask(
    out_f: np.ndarray,
    mask: np.ndarray,
    color_arr: np.ndarray,
    sigma: float,
    alpha: float,
) -> None:
    if not np.any(mask):
        return
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    np.clip(soft, 0.0, 1.0, out=soft)
    soft *= alpha
    blend = soft[:, :, np.newaxis]
    out_f *= 1.0 - blend
    out_f += color_arr * blend
    np.clip(out_f, 0, 255, out=out_f)


def _draw_markings(out_f: np.ndarray, semantics: dict[str, np.ndarray]) -> None:
    sigma = max(1.6, out_f.shape[1] / 520.0)
    for key in MARKING_COLORS:
        mask = semantics.get(key)
        if mask is None or not np.any(mask):
            continue
        thick_mask = cv2.dilate(mask.astype(np.uint8), _MARKING_DILATE_KERNEL, iterations=1)
        _soft_blend_mask(out_f, thick_mask, MARKING_COLOR_ARRAYS[key], sigma=sigma, alpha=0.92)


def _visualize_lane_area_prob(class_probs: np.ndarray) -> np.ndarray:
    num_classes = int(class_probs.shape[-1])
    lane_idx = 1 if num_classes <= 2 else _CLASS_KEY_TO_ID["lane_area"]
    lane_prob = class_probs[:, :, lane_idx].copy()
    bg_prob = class_probs[:, :, 0]
    lane_prob[lane_prob <= bg_prob] = 0.0
    lane_prob[lane_prob < 0.25] = 0.0
    return lane_prob.astype(np.float32, copy=False)


def _draw_lane_area(
    out_f: np.ndarray,
    lane_area_mask: np.ndarray,
    lane_area_prob: np.ndarray | None,
) -> None:
    sigma = max(2.4, out_f.shape[1] / 320.0)
    if lane_area_prob is not None:
        masked_prob = lane_area_prob * lane_area_mask.astype(np.float32)
        _soft_blend_mask(out_f, masked_prob, LANE_FILL_ARR, sigma=sigma, alpha=0.42)
        return
    _soft_blend_mask(out_f, lane_area_mask.astype(np.uint8), LANE_FILL_ARR, sigma=sigma, alpha=0.48)


def render_lane_overlay(
    frame: np.ndarray,
    lane_area_mask: np.ndarray,
    semantics: dict[str, np.ndarray],
    class_probs: np.ndarray | None,
) -> np.ndarray:
    """Soft-blended lane fill + per-class semantic markings."""
    out_f = frame.astype(np.float32)
    lane_prob = None if class_probs is None else _visualize_lane_area_prob(class_probs)
    _draw_lane_area(out_f, lane_area_mask, lane_prob)
    _draw_markings(out_f, semantics)
    return out_f.astype(np.uint8)


def _blend_fast_mask(
    out_f: np.ndarray, mask: np.ndarray, color_arr: np.ndarray, alpha: float,
) -> None:
    if not np.any(mask):
        return
    nz = mask > 0
    out_f[nz] = out_f[nz] * (1.0 - alpha) + color_arr.ravel() * alpha


def render_lane_overlay_fast(
    frame: np.ndarray,
    lane_area_mask: np.ndarray,
    semantics: dict[str, np.ndarray],
) -> np.ndarray:
    """Hard-blended lane fill + markings — used in the low-CPU path."""
    out_f = frame.astype(np.float32)
    _blend_fast_mask(out_f, lane_area_mask.astype(np.uint8), LANE_FILL_ARR, alpha=0.48)
    for key in MARKING_COLORS:
        mask = semantics.get(key)
        if mask is None:
            continue
        _blend_fast_mask(out_f, mask.astype(np.uint8), MARKING_COLOR_ARRAYS[key], alpha=0.92)
    return out_f.astype(np.uint8)
