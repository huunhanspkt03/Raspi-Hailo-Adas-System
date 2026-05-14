"""
Preprocessor — frame preparation for Hailo inference.
Handles YOLO26 detection (640x640 letterbox) and YOLO26s_seg lane
segmentation (640x640 letterbox).

Both models expect RGB uint8 input — do NOT divide by 255 on Pi.
On x86 ONNX path, OnnxFallbackInference normalizes to float32 automatically.
"""
from __future__ import annotations

import cv2
import numpy as np

YOLO_INPUT_SIZE = 640
YOLO_PAD_VALUE = 114
SEGMENTATION_INPUT_SIZE = 640  # YOLO26s_seg letterbox input.


def _resolve_hw(input_size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(input_size, int):
        return input_size, input_size
    if isinstance(input_size, (tuple, list)) and len(input_size) == 2:
        return int(input_size[0]), int(input_size[1])
    raise ValueError(f"Unsupported input_size format: {input_size!r}")


def resolve_runtime_input_size(
    input_shape: tuple[int, ...] | list[int],
    fallback: int | tuple[int, int] | list[int],
) -> int | tuple[int, int]:
    """Infer model input HxW from ONNX/Hailo shape metadata.

    Supports:
    - ONNX NCHW: (1, 3, H, W)
    - Hailo/array HWC: (H, W, 3)
    - NHWC: (1, H, W, 3)

    Returns the original fallback when the shape is dynamic/unknown.
    """
    shape = tuple(int(dim) for dim in input_shape)
    h = w = None

    if len(shape) == 4:
        if shape[1] in (1, 3) and shape[2] > 0 and shape[3] > 0:
            h, w = shape[2], shape[3]
        elif shape[3] in (1, 3) and shape[1] > 0 and shape[2] > 0:
            h, w = shape[1], shape[2]
    elif len(shape) == 3:
        if shape[2] in (1, 3) and shape[0] > 0 and shape[1] > 0:
            h, w = shape[0], shape[1]
        elif shape[0] in (1, 3) and shape[1] > 0 and shape[2] > 0:
            h, w = shape[1], shape[2]

    if h is None or w is None or h <= 0 or w <= 0:
        if isinstance(fallback, int):
            return int(fallback)
        fb_h, fb_w = _resolve_hw(fallback)
        return fb_h if fb_h == fb_w else (fb_h, fb_w)

    return h if h == w else (h, w)


def preprocess_rgb_uint8(frame_bgr: np.ndarray, input_size: int | tuple[int, int] | list[int]) -> np.ndarray:
    """Resize and convert a BGR frame to RGB uint8 input."""
    input_h, input_w = _resolve_hw(input_size)
    resized = cv2.resize(frame_bgr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def preprocess_letterbox_rgb_uint8(
    frame_bgr: np.ndarray,
    input_size: int,
    pad_value: int = YOLO_PAD_VALUE,
) -> np.ndarray:
    """Letterbox a BGR frame to square RGB uint8 input with centered padding."""
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Input frame has invalid shape for letterbox preprocessing")

    scale = min(input_size / w, input_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((input_size, input_size, 3), pad_value, dtype=np.uint8)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def preprocess_yolo26(frame_bgr: np.ndarray, input_size: int = YOLO_INPUT_SIZE) -> np.ndarray:
    """Letterbox and convert BGR frame to YOLO26 input: NxN RGB uint8."""
    return preprocess_letterbox_rgb_uint8(frame_bgr, input_size, pad_value=YOLO_PAD_VALUE)


def preprocess_segmentation(
    frame_bgr: np.ndarray,
    input_size: int | tuple[int, int] | list[int] = SEGMENTATION_INPUT_SIZE,
) -> np.ndarray:
    """Preprocess BGR frame for YOLO26s_seg: 640x640 letterboxed RGB uint8."""
    h, w = _resolve_hw(input_size)
    if h == w:
        return preprocess_letterbox_rgb_uint8(frame_bgr, h, pad_value=YOLO_PAD_VALUE)
    return preprocess_rgb_uint8(frame_bgr, input_size)
