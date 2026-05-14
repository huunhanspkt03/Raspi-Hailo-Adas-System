from __future__ import annotations

from typing import Any


MODE_ALIASES = {
    "lane": "lane",
    "segmentation": "lane",
    "detection": "detection",
    "full": "full",
    "combined": "full",
}

RUNTIME_MODE_CHOICES = tuple(sorted(MODE_ALIASES))

MODE_FEATURE_OVERRIDES = {
    "lane": {
        "lane_detection": True,
        "object_detection": False,
        "behavior_logic": False,
    },
    "detection": {
        "lane_detection": False,
        "object_detection": True,
        "behavior_logic": False,
        "rc_control": False,
    },
    "full": {
        "lane_detection": True,
        "object_detection": True,
    },
}


def canonicalize_runtime_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    key = str(mode).strip().lower()
    if not key:
        return None
    if key not in MODE_ALIASES:
        valid = ", ".join(RUNTIME_MODE_CHOICES)
        raise ValueError(f"Unknown runtime mode '{mode}'. Valid values: {valid}")
    return MODE_ALIASES[key]


def apply_runtime_mode(cfg: dict[str, Any], mode: str | None) -> dict[str, Any]:
    cfg = dict(cfg)
    project = dict(cfg.get("project", {}))
    features = dict(cfg.get("features", {}))

    selected_mode = canonicalize_runtime_mode(mode or project.get("mode"))
    if selected_mode is None:
        return cfg

    project["mode"] = selected_mode
    features.update(MODE_FEATURE_OVERRIDES[selected_mode])

    cfg["project"] = project
    cfg["features"] = features
    return cfg


def sanitize_runtime_features(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    cfg = dict(cfg)
    features = dict(cfg.get("features", {}))
    notes: list[str] = []

    lane_enabled = bool(features.get("lane_detection", False))
    object_enabled = bool(features.get("object_detection", False))
    behavior_enabled = bool(features.get("behavior_logic", False))
    rc_enabled = bool(features.get("rc_control", False))

    if behavior_enabled and not object_enabled:
        behavior_enabled = False
        notes.append("Disabled behavior_logic because object_detection is off.")

    if rc_enabled and not lane_enabled and not behavior_enabled:
        rc_enabled = False
        notes.append("Disabled rc_control because no steering/throttle source is active in this mode.")

    if not lane_enabled and not object_enabled:
        raise ValueError(
            "Invalid runtime configuration: both lane_detection and object_detection are disabled."
        )

    features["lane_detection"] = lane_enabled
    features["object_detection"] = object_enabled
    features["behavior_logic"] = behavior_enabled
    features["rc_control"] = rc_enabled
    cfg["features"] = features
    return cfg, notes
