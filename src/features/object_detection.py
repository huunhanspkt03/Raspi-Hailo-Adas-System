"""
Object Detection — YOLO26 decode for RC lane-centering.

Supports both:
- Ultralytics end-to-end output: `(1, N, 6)` with `[x1, y1, x2, y2, score, class_id]`
- Hailo raw-head output: 3 box tensors + 3 class tensors parsed from YOLO26 one-to-one heads
"""
from __future__ import annotations

import numpy as np

from features.detection_filters import (
    CLASS_ID_TO_NAME,
    DIRECTIONAL_SIGN_LABELS,
    PROHIBITION_SIGN_LABELS,
    SPEED_LIMIT_LABELS,
    TRAFFIC_LIGHT_LABELS,
    TRAFFIC_PARTICIPANT_LABELS,
    filter_by_labels,
    filter_directional_signs,
    filter_prohibition_signs,
    filter_speed_limit_signs,
    filter_traffic_lights,
    filter_traffic_participants,
    find_leading_vehicle,
    is_box_near_image_center,
    resolve_speed_sign,
)
from features.yolo26_raw_head import YOLO26_MAX_DET, decode_raw_head_output

__all__ = [
    "CLASS_ID_TO_NAME",
    "TRAFFIC_PARTICIPANT_LABELS",
    "TRAFFIC_LIGHT_LABELS",
    "DIRECTIONAL_SIGN_LABELS",
    "PROHIBITION_SIGN_LABELS",
    "SPEED_LIMIT_LABELS",
    "decode_yolo26",
    "filter_by_labels",
    "filter_traffic_participants",
    "filter_traffic_lights",
    "filter_directional_signs",
    "filter_prohibition_signs",
    "filter_speed_limit_signs",
    "is_box_near_image_center",
    "find_leading_vehicle",
    "resolve_speed_sign",
]


def _clip_box_xyxy(box: np.ndarray, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width - 1))
    y2 = max(0, min(int(round(y2)), height - 1))
    return [x1, y1, x2, y2]


def _letterbox_params(orig_w: int, orig_h: int, model_w: int, model_h: int) -> tuple[float, float, float]:
    scale = min(model_w / float(orig_w), model_h / float(orig_h))
    resized_w = orig_w * scale
    resized_h = orig_h * scale
    pad_x = (model_w - resized_w) / 2.0
    pad_y = (model_h - resized_h) / 2.0
    return scale, pad_x, pad_y


def _scale_xyxy_from_model_to_image(
    boxes_xyxy: np.ndarray,
    orig_w: int,
    orig_h: int,
    model_w: int,
    model_h: int,
) -> np.ndarray:
    scale, pad_x, pad_y = _letterbox_params(orig_w, orig_h, model_w, model_h)
    boxes = boxes_xyxy.copy().astype(np.float32)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)
    return boxes


def _decode_end2end_output(
    output: np.ndarray,
    conf_thresh: float,
    orig_w: int,
    orig_h: int,
    model_w: int,
    model_h: int,
) -> list[dict]:
    preds = output[0] if output.ndim == 3 else output
    boxes_xyxy = _scale_xyxy_from_model_to_image(preds[:, :4], orig_w, orig_h, model_w, model_h)
    scores = preds[:, 4]
    class_ids = preds[:, 5].astype(np.int32)

    detections = []
    for box, score, cid in zip(boxes_xyxy, scores, class_ids):
        if score < conf_thresh:
            continue
        label = CLASS_ID_TO_NAME.get(int(cid))
        if label is None:
            continue
        detections.append({
            "box": _clip_box_xyxy(box, orig_w, orig_h),
            "score": float(score),
            "class_id": int(cid),
            "label": label,
        })
    return detections


def decode_yolo26(
    raw_output: dict,
    conf_thresh: float = 0.30,
    orig_w: int = 1280,
    orig_h: int = 720,
    model_w: int = 640,
    model_h: int = 640,
    max_det: int = YOLO26_MAX_DET,
) -> list[dict]:
    """Decode either end-to-end or raw-head YOLO26 outputs into detection dicts."""
    output = next(iter(raw_output.values()))
    output_arr = np.asarray(output)

    if output_arr.ndim >= 2 and output_arr.shape[-1] == 6:
        detections = _decode_end2end_output(output_arr, conf_thresh, orig_w, orig_h, model_w, model_h)
    else:
        detections = decode_raw_head_output(
            raw_output, conf_thresh, orig_w, orig_h, model_w, model_h,
            scale_fn=_scale_xyxy_from_model_to_image,
            clip_fn=_clip_box_xyxy,
        )

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    return detections[:max_det]
