"""Raw segmentation overlay — faithful argmax debug visualization."""
from __future__ import annotations

import cv2
import numpy as np

_MARKING_OPEN_KERNEL = np.ones((3, 3), np.uint8)

_RAW_CLASS_COLORS = np.array(
    [
        (0, 0, 0),        # background
        (255, 255, 0),    # dotted_line
        (90, 220, 110),   # lane_area
        (255, 80, 255),   # solid_line
    ],
    dtype=np.float32,
)
_RAW_CLASS_ALPHA = np.array([0.0, 0.92, 0.18, 0.92], dtype=np.float32)

_RAW_BINARY_CLASS_COLORS = np.array(
    [
        (0, 0, 0),       # background
        (90, 220, 110),  # lane_area
    ],
    dtype=np.float32,
)
_RAW_BINARY_CLASS_ALPHA = np.array([0.0, 0.30], dtype=np.float32)


def draw_raw_overlay(
    frame: np.ndarray,
    mask: np.ndarray | None,
    class_probs: np.ndarray | None = None,
    *,
    fast: bool = False,
) -> np.ndarray:
    """Faithful raw segmentation overlay — argmax classes, no amplification.

    Shows exactly what the model predicts per pixel. Isolated single-pixel
    noise is removed with morphological opening so the debug view is usable.
    """
    if class_probs is not None:
        argmax = class_probs.argmax(axis=-1).astype(np.uint8)
        non_bg = (argmax > 0).astype(np.uint8)
        non_bg = cv2.morphologyEx(non_bg, cv2.MORPH_OPEN, _MARKING_OPEN_KERNEL)
        argmax[non_bg == 0] = 0
        out_f = frame.astype(np.float32)
        if class_probs.shape[-1] <= 2:
            color = _RAW_BINARY_CLASS_COLORS[argmax]
            alpha = _RAW_BINARY_CLASS_ALPHA[argmax][:, :, np.newaxis]
        else:
            color = _RAW_CLASS_COLORS[argmax]
            alpha = _RAW_CLASS_ALPHA[argmax][:, :, np.newaxis]
        out_f[:] = out_f * (1.0 - alpha) + color * alpha
        return out_f.astype(np.uint8)

    if mask is None:
        raise ValueError("draw_raw_overlay requires either class_probs or mask.")

    out_f = frame.astype(np.float32)
    if int(mask.max()) <= 1:
        color = _RAW_BINARY_CLASS_COLORS[mask]
        alpha = _RAW_BINARY_CLASS_ALPHA[mask][:, :, np.newaxis]
    else:
        color = _RAW_CLASS_COLORS[mask]
        alpha = _RAW_CLASS_ALPHA[mask][:, :, np.newaxis]
    out_f[:] = out_f * (1.0 - alpha) + color * alpha
    return out_f.astype(np.uint8)
