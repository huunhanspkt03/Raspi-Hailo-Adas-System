"""Decode YOLO26s_seg ONNX outputs into lane-area + semantic masks.

Contract for ``decode_lane_components`` is the 4-tuple every downstream
consumer expects:

    (mask, class_probs, semantics, lane_area)

- ``mask``       : H x W uint8 argmax over [bg, dotted_line, lane_area,
                   solid_line].
- ``class_probs``: H x W x 4 float32 score stack used by raw-overlay debug.
- ``semantics``  : dict with binary uint8 masks at frame size for the
                   model classes ``dotted_line``, ``lane_area``, and
                   ``solid_line``.
- ``lane_area``  : H x W uint8 binary mask (alias of semantics["lane_area"]).

The model exports with NMS, so ``output0`` rows are
``[x1, y1, x2, y2, score, class_id, *32_coeffs]`` in 640x640 letterbox
space. ``output1`` is the (32, 160, 160) prototype used for mask synthesis.
"""
from __future__ import annotations

import cv2
import numpy as np

MODEL_CLASS_NAMES = {
    0: "dotted_line",
    1: "lane_area",
    2: "solid_line",
}

_CLASS_KEY_TO_ID = {
    "dotted_line": 1,
    "lane_area": 2,
    "solid_line": 3,
}

# yolo26s_seg.onnx metadata currently matches MODEL_CLASS_NAMES.
_YOLO_TO_SEM = {
    0: ("dotted_line", 1),
    1: ("lane_area", 2),
    2: ("solid_line", 3),
}
NUM_CLASSES = 4  # bg + 3 foreground
INPUT_SIZE = 640
PROTO_SIZE = 160
SCORE_THRESH = 0.25
MASK_THRESH = 0.5


def _extract_outputs(raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pull (detections, proto) from the raw output dict by shape."""
    det = proto = None
    for arr in raw.values():
        a = np.asarray(arr)
        if a.ndim == 4 and a.shape[1] == 32 and a.shape[-1] == a.shape[-2]:
            proto = a[0]  # (32, P, P)
        elif a.ndim == 3 and a.shape[-1] >= 6 + 32:
            det = a[0]    # (N, 6 + 32)
        elif a.ndim == 2 and a.shape[-1] >= 6 + 32:
            det = a       # already squeezed
    if det is None or proto is None:
        raise RuntimeError(
            "lane_postprocess: could not find YOLO-seg outputs "
            f"(have shapes={[np.asarray(v).shape for v in raw.values()]})"
        )
    return det, proto


def _letterbox_params(frame_h: int, frame_w: int) -> tuple[float, int, int]:
    """Return (scale, pad_x, pad_y) used to letterbox to INPUT_SIZE."""
    scale = min(INPUT_SIZE / frame_w, INPUT_SIZE / frame_h)
    new_w = max(1, int(round(frame_w * scale)))
    new_h = max(1, int(round(frame_h * scale)))
    pad_x = (INPUT_SIZE - new_w) // 2
    pad_y = (INPUT_SIZE - new_h) // 2
    return scale, pad_x, pad_y


def _per_detection_mask(coefs: np.ndarray, proto_flat: np.ndarray) -> np.ndarray:
    """Sigmoid(coefs @ proto) reshaped to (PROTO_SIZE, PROTO_SIZE)."""
    logits = coefs @ proto_flat
    np.clip(logits, -30.0, 30.0, out=logits)
    mask = 1.0 / (1.0 + np.exp(-logits))
    return mask.reshape(PROTO_SIZE, PROTO_SIZE)


def _crop_to_box(mask_proto: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Zero out mask outside the detection box (box in 640 letterbox space)."""
    s = PROTO_SIZE / INPUT_SIZE
    px1 = int(np.clip(np.floor(x1 * s), 0, PROTO_SIZE - 1))
    py1 = int(np.clip(np.floor(y1 * s), 0, PROTO_SIZE - 1))
    px2 = int(np.clip(np.ceil(x2 * s), 0, PROTO_SIZE))
    py2 = int(np.clip(np.ceil(y2 * s), 0, PROTO_SIZE))
    out = np.zeros_like(mask_proto)
    if px2 > px1 and py2 > py1:
        out[py1:py2, px1:px2] = mask_proto[py1:py2, px1:px2]
    return out


def _unletterbox(mask_input: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
    """Resize a 640-space mask back to (frame_h, frame_w), undoing the pad."""
    scale, pad_x, pad_y = _letterbox_params(frame_h, frame_w)
    new_w = max(1, int(round(frame_w * scale)))
    new_h = max(1, int(round(frame_h * scale)))
    cropped = mask_input[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    if cropped.size == 0:
        return np.zeros((frame_h, frame_w), dtype=mask_input.dtype)
    return cv2.resize(cropped, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)


def _empty_components(frame_h: int, frame_w: int) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    class_probs = np.zeros((frame_h, frame_w, NUM_CLASSES), dtype=np.float32)
    class_probs[:, :, 0] = 1.0
    sem = {key: np.zeros((frame_h, frame_w), dtype=np.uint8) for key in _CLASS_KEY_TO_ID}
    return mask, class_probs, sem, sem["lane_area"]


def decode_lane_components(
    raw: dict,
    frame_h: int,
    frame_w: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    det, proto = _extract_outputs(raw)
    proto_flat = proto.reshape(32, -1).astype(np.float32, copy=False)

    # Per-class accumulator at INPUT_SIZE resolution.
    per_class_640 = np.zeros((NUM_CLASSES, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    any_kept = False
    for row in det:
        score = float(row[4])
        if score < SCORE_THRESH:
            continue
        cls_id = int(row[5])
        if cls_id not in _YOLO_TO_SEM:
            continue
        key, sem_id = _YOLO_TO_SEM[cls_id]
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        coefs = row[6:6 + 32].astype(np.float32, copy=False)
        mask_proto = _per_detection_mask(coefs, proto_flat)
        mask_proto = _crop_to_box(mask_proto, x1, y1, x2, y2)
        mask_640 = cv2.resize(mask_proto, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        weighted = mask_640 * score
        np.maximum(per_class_640[sem_id], weighted, out=per_class_640[sem_id])
        any_kept = True

    if not any_kept:
        return _empty_components(frame_h, frame_w)

    # Background score: 1 - max(foreground). Keeps argmax sane.
    fg_max = per_class_640[1:].max(axis=0)
    per_class_640[0] = np.clip(1.0 - fg_max, 0.0, 1.0)

    # Unletterbox each channel back to frame size, then build outputs.
    class_probs = np.zeros((frame_h, frame_w, NUM_CLASSES), dtype=np.float32)
    for c in range(NUM_CLASSES):
        class_probs[:, :, c] = _unletterbox(per_class_640[c], frame_h, frame_w)

    mask = class_probs.argmax(axis=-1).astype(np.uint8)
    semantics: dict[str, np.ndarray] = {}
    for key, sem_id in _CLASS_KEY_TO_ID.items():
        bin_mask = (class_probs[:, :, sem_id] >= MASK_THRESH).astype(np.uint8)
        # Tighten via mask agreement so a soft >=0.5 region only counts as
        # this class when argmax also picked it.
        bin_mask &= (mask == sem_id).astype(np.uint8)
        semantics[key] = bin_mask
    lane_area = semantics["lane_area"]
    return mask, class_probs, semantics, lane_area


def decode_segmentation_probabilities(
    raw: dict,
    frame_h: int,
    frame_w: int,
) -> np.ndarray:
    """Probability stack for the raw-overlay debug renderer."""
    _, class_probs, _, _ = decode_lane_components(raw, frame_h, frame_w)
    return class_probs
