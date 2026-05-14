"""Detection filter and query utilities for YOLO26 outputs."""
from __future__ import annotations


CLASS_ID_TO_NAME: dict[int, str] = {
    0:  "bus",
    1:  "car",
    2:  "green_light",
    3:  "motorcycle",
    4:  "red_light",
    5:  "speed_limit_40",
    6:  "speed_limit_50",
    7:  "speed_limit_60",
    8:  "speed_limit_70",
    9:  "speed_limit_80",
    10: "truck",
}

TRAFFIC_PARTICIPANT_LABELS = {"car", "motorcycle", "bus", "truck"}
TRAFFIC_LIGHT_LABELS       = {"red_light", "green_light"}
DIRECTIONAL_SIGN_LABELS: set[str] = set()
PROHIBITION_SIGN_LABELS: set[str] = set()
SPEED_LIMIT_LABELS         = {
    "speed_limit_40",
    "speed_limit_50",
    "speed_limit_60",
    "speed_limit_70",
    "speed_limit_80",
}


def filter_by_labels(detections: list[dict], labels: set[str]) -> list[dict]:
    return [d for d in detections if d["label"] in labels]


def filter_traffic_participants(detections: list[dict]) -> list[dict]:
    return filter_by_labels(detections, TRAFFIC_PARTICIPANT_LABELS)


def filter_traffic_lights(detections: list[dict]) -> list[dict]:
    return filter_by_labels(detections, TRAFFIC_LIGHT_LABELS)


def filter_directional_signs(detections: list[dict]) -> list[dict]:
    return filter_by_labels(detections, DIRECTIONAL_SIGN_LABELS)


def filter_prohibition_signs(detections: list[dict]) -> list[dict]:
    return filter_by_labels(detections, PROHIBITION_SIGN_LABELS)


def filter_speed_limit_signs(detections: list[dict]) -> list[dict]:
    return filter_by_labels(detections, SPEED_LIMIT_LABELS)


def is_box_near_image_center(box: list[int], img_w: int, tolerance: float = 0.2) -> bool:
    cx = (box[0] + box[2]) / 2
    return abs(cx - img_w / 2) <= img_w * tolerance


def find_leading_vehicle(
    detections: list[dict],
    img_w: int,
    min_track_age: int = 2,
) -> dict | None:
    """Return the most likely vehicle ahead.

    Preference order:
      1. Vehicles flagged ``in_path=True`` (corridor-derived path relevance).
      2. Vehicles whose bbox center sits near the image center (legacy
         heuristic for callers with no corridor available).
    Within the chosen pool, tracked candidates whose ``track_age`` has
    reached ``min_track_age`` are preferred over flicker; the largest box
    wins as a proxy for "closest".
    """
    vehicle_labels = {"car", "motorcycle", "bus", "truck"}
    vehicles = [d for d in detections if d["label"] in vehicle_labels]
    in_path = [d for d in vehicles if d.get("in_path")]
    if in_path:
        candidates = in_path
    else:
        candidates = [d for d in vehicles if is_box_near_image_center(d["box"], img_w)]
    if not candidates:
        return None
    confirmed = [d for d in candidates if d.get("track_age", 0) >= min_track_age]
    pool = confirmed if confirmed else candidates
    return max(pool, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))


def resolve_speed_sign(detections: list[dict]) -> str | None:
    """Return the highest-confidence active speed-limit sign label, or None."""
    speed_dets = filter_speed_limit_signs(detections)
    if not speed_dets:
        return None
    return max(speed_dets, key=lambda d: d["score"])["label"]
