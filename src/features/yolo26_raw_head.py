"""
YOLO26 raw-head decoder — Hailo 3-box + 3-class tensor decode pipeline.

Handles tensor normalization, anchor generation, dist2bbox, and top-k selection
for the one2one raw-head output format from Hailo-compiled YOLO26.

Hailo HEF outputs are NHWC (float32, dequantized). The decoder accepts both
NCHW and NHWC layouts. When ``--skip-sigmoid`` is used during Hailo compilation,
the HEF absorbs the sigmoid — set ``YOLO26_HAILO_SIGMOID_ABSORBED=1`` env var
to skip the Python-side sigmoid on class logits.
"""
from __future__ import annotations

import os
import re

import numpy as np

from features.detection_filters import CLASS_ID_TO_NAME

YOLO26_STRIDES = (8, 16, 32)
YOLO26_MAX_DET = 300

_RE_CV2 = re.compile(r'one2one_cv2[._](\d+)')
_RE_CV3 = re.compile(r'one2one_cv3[._](\d+)')
_SIGMOID_ABSORBED = os.getenv("YOLO26_HAILO_SIGMOID_ABSORBED", "0") == "1"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _normalize_raw_head_tensor(output: np.ndarray, expected_channels: int) -> np.ndarray:
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(f"Unsupported raw-head tensor shape: {arr.shape}")

    if arr.shape[0] == expected_channels:
        return arr
    if arr.shape[-1] == expected_channels:
        return arr.transpose(2, 0, 1)

    raise ValueError(f"Cannot infer channel axis for raw-head tensor shape: {arr.shape}")


def _raw_head_spatial_size(output: np.ndarray) -> int:
    arr = np.asarray(output)
    shape = arr.shape[1:] if arr.ndim == 4 and arr.shape[0] == 1 else arr.shape
    if len(shape) != 3:
        raise ValueError(f"Unsupported raw-head tensor shape for ordering: {arr.shape}")

    if shape[0] in (4, len(CLASS_ID_TO_NAME)):
        h, w = shape[1], shape[2]
    elif shape[-1] in (4, len(CLASS_ID_TO_NAME)):
        h, w = shape[0], shape[1]
    else:
        h, w = sorted(shape)[-2:]
    return int(h) * int(w)


def _extract_raw_head_groups(raw_output: dict) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    box_entries: list[tuple[int, np.ndarray]] = []
    cls_entries: list[tuple[int, np.ndarray]] = []

    for name, value in raw_output.items():
        m = _RE_CV2.search(name)
        if m:
            box_entries.append((int(m.group(1)), np.asarray(value)))
            continue
        m = _RE_CV3.search(name)
        if m:
            cls_entries.append((int(m.group(1)), np.asarray(value)))

    if not box_entries or not cls_entries:
        arrays = list(raw_output.values())
        if len(arrays) != 6:
            return None
        guessed_boxes = []
        guessed_classes = []
        for value in arrays:
            arr = np.asarray(value)
            shape = arr.shape[1:] if arr.ndim == 4 and arr.shape[0] == 1 else arr.shape
            if 4 in shape:
                guessed_boxes.append(value)
            elif len(CLASS_ID_TO_NAME) in shape:
                guessed_classes.append(value)
        if len(guessed_boxes) == 3 and len(guessed_classes) == 3:
            guessed_boxes.sort(key=_raw_head_spatial_size, reverse=True)
            guessed_classes.sort(key=_raw_head_spatial_size, reverse=True)
            box_entries = list(enumerate(guessed_boxes))
            cls_entries = list(enumerate(guessed_classes))
        else:
            return None

    box_entries.sort(key=lambda item: item[0])
    cls_entries.sort(key=lambda item: item[0])
    return [item[1] for item in box_entries], [item[1] for item in cls_entries]


def _make_anchor_points(width: int, height: int) -> np.ndarray:
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32) + 0.5,
        np.arange(width, dtype=np.float32) + 0.5,
        indexing="ij",
    )
    return np.stack((xs, ys), axis=-1).reshape(-1, 2)


def _dist2bbox_xyxy(distance: np.ndarray, anchor_points: np.ndarray) -> np.ndarray:
    lt = distance[:, :2]
    rb = distance[:, 2:]
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return np.concatenate((x1y1, x2y2), axis=1)


def _topk_raw_head_predictions(
    boxes_xyxy: np.ndarray,
    cls_scores: np.ndarray,
    max_det: int = YOLO26_MAX_DET,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_anchors, num_classes = cls_scores.shape
    k = min(max_det, num_anchors)
    anchor_order = np.argpartition(cls_scores.max(axis=1), -k)[-k:]
    candidate_scores = cls_scores[anchor_order].reshape(-1)
    top_count = min(max_det, candidate_scores.shape[0])
    top_flat = np.argpartition(candidate_scores, -top_count)[-top_count:]
    top_sorted = top_flat[np.argsort(candidate_scores[top_flat])[::-1]]
    anchor_indices = anchor_order[top_sorted // num_classes]
    class_ids = top_sorted % num_classes
    scores = candidate_scores[top_sorted]
    return boxes_xyxy[anchor_indices], scores, class_ids.astype(np.int32)


def decode_raw_head_output(
    raw_output: dict,
    conf_thresh: float,
    orig_w: int,
    orig_h: int,
    model_w: int,
    model_h: int,
    scale_fn,
    clip_fn,
) -> list[dict]:
    """Decode Hailo raw-head tensors into detection dicts.

    Parameters
    ----------
    scale_fn : callable
        (boxes_xyxy, orig_w, orig_h, model_w, model_h) -> scaled boxes
    clip_fn : callable
        (box, width, height) -> clipped [x1,y1,x2,y2]
    """
    grouped = _extract_raw_head_groups(raw_output)
    if grouped is None:
        raise ValueError("Could not identify YOLO26 raw-head outputs in inference result")

    box_heads, cls_heads = grouped
    all_boxes = []
    all_scores = []
    num_classes = len(CLASS_ID_TO_NAME)

    for stride, box_head, cls_head in zip(YOLO26_STRIDES, box_heads, cls_heads):
        box_chw = _normalize_raw_head_tensor(box_head, expected_channels=4)
        cls_chw = _normalize_raw_head_tensor(cls_head, expected_channels=num_classes)
        _, h, w = box_chw.shape
        anchors = _make_anchor_points(w, h)
        box_dist = box_chw.reshape(4, -1).transpose(1, 0)
        cls_logits = cls_chw.reshape(num_classes, -1).transpose(1, 0)
        boxes_xyxy = _dist2bbox_xyxy(box_dist, anchors) * float(stride)
        cls_scores = cls_logits if _SIGMOID_ABSORBED else _sigmoid(cls_logits)
        all_boxes.append(boxes_xyxy)
        all_scores.append(cls_scores)

    boxes_xyxy = np.concatenate(all_boxes, axis=0)
    cls_scores = np.concatenate(all_scores, axis=0)
    boxes_xyxy, scores, class_ids = _topk_raw_head_predictions(boxes_xyxy, cls_scores)
    boxes_xyxy = scale_fn(boxes_xyxy, orig_w, orig_h, model_w, model_h)

    detections = []
    for box, score, cid in zip(boxes_xyxy, scores, class_ids):
        if score < conf_thresh:
            continue
        label = CLASS_ID_TO_NAME.get(int(cid))
        if label is None:
            continue
        detections.append({
            "box": clip_fn(box, orig_w, orig_h),
            "score": float(score),
            "class_id": int(cid),
            "label": label,
        })
    return detections
