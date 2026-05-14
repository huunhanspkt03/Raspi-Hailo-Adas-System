"""CLI argument parsing and config loading for the ADAS pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from runtime_mode import RUNTIME_MODE_CHOICES

LANE_SEGMENTATION_HEF_CANDIDATES = (
    "src/models/yolo26s_seg.hef",
    "src/models/yolo26n_seg.hef",
)

YOLO26_HEF_CANDIDATES = (
    "src/models/yolo26_adas_2404.hef",
    "src/models/yolo26_adas_1404.hef",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RC Lane-Centering Pipeline — RPi5 + Hailo-8")
    p.add_argument("--config", default="config/base.yaml")
    p.add_argument(
        "--mode",
        choices=RUNTIME_MODE_CHOICES,
        default=None,
        help="Force a runtime mode: lane/segmentation, detection, or full.",
    )
    p.add_argument("--display",    action="store_true",
                   help="Force display window on. Shows frames live but does not save video by itself.")
    p.add_argument("--no-display", action="store_true", help="Force display window off")
    p.add_argument("--onnx",   action="store_true", help="Force ONNX Runtime fallback")
    p.add_argument("--camera", choices=["csi", "usb"], default=None)
    p.add_argument("--video",  default=None, help="Path to a video file (skips camera)")
    p.add_argument("--loop-video", dest="loop_video", action="store_true",
                   help="When using --video, loop back to start at EOF.")
    p.add_argument("--no-loop-video", dest="loop_video", action="store_false",
                   help="When using --video, stop at EOF (default).")
    p.add_argument("--lane-onnx", default=None,
                   help="Override the lane-segmentation model path when running with --onnx.")
    p.add_argument("--conf",   type=float, default=None, help="Override confidence threshold")
    p.add_argument("--raw-overlay", action="store_true",
                   help="Raw overlay mode: segmentation mask colormap only, no smoothing or HUD.")
    p.add_argument("--fast-overlay", action="store_true",
                   help="Use cheaper overlay rendering to reduce CPU cost on Raspberry Pi.")
    p.add_argument("--perf", action="store_true",
                   help="Aggressive Pi perf mode: low-latency capture, smaller display, fast overlay.")
    p.add_argument("--display-width", type=int, default=None,
                   help="Override output display width in pixels.")
    p.add_argument("--save-video", default=None,
                   help="Optional output video path (.mp4). This is the flag that writes annotated output video.")
    p.add_argument("--branch-threading", dest="branch_threading", action="store_true",
                   help="Parallelize lane/detection CPU branches in full mode.")
    p.add_argument("--no-branch-threading", dest="branch_threading", action="store_false",
                   help="Disable lane/detection CPU branch threading.")
    p.add_argument("--branch-workers", type=int, default=None,
                   help="Override number of worker threads for branch threading.")
    p.add_argument("--lane-debug", action="store_true",
                   help="Render the 2x2 lane BEV debug panel (original, BEV markings, "
                        "BEV construction, final polygon) and save/display that instead.")
    p.add_argument("--threaded", action="store_true",
                   help="Run the decoupled multithreaded runtime "
                        "(capture/lane/object/fusion/output threads).")
    p.add_argument("--metrics-out", default=None,
                   help="Optional path for per-frame lane validation "
                        "metrics (JSONL / CSV side-channel logger). "
                        "Off by default.")
    p.add_argument("--metrics-format", choices=("jsonl", "csv"),
                   default="jsonl",
                   help="Format for --metrics-out (default: jsonl).")
    p.set_defaults(branch_threading=None, loop_video=None)
    return p.parse_args()


def load_config(path: str) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(cfg)
    camera = dict(cfg.get("camera", {}))
    vis    = dict(cfg.get("visualization", cfg.get("display", {})))
    infer  = dict(cfg.get("inference", {}))
    pipeline = dict(cfg.get("pipeline", {}))

    if args.camera == "csi":
        camera["use_gstreamer"] = True
    elif args.camera == "usb":
        camera["use_gstreamer"] = False
    if args.display:
        vis["show_window"] = True
    if args.no_display:
        vis["show_window"] = False
    if args.conf is not None:
        infer["conf_threshold"] = args.conf
    if args.lane_onnx is not None:
        infer["lane_segmentation_hef"] = args.lane_onnx
    if args.perf:
        camera["prefer_low_latency"] = True
        camera["drop_frames"] = True
        camera["threaded_capture"] = True
        camera["width"] = 640
        camera["height"] = 360
        vis["fast_overlay"] = True
        vis["display_width"] = 640
    if args.fast_overlay:
        vis["fast_overlay"] = True
    if args.display_width is not None:
        vis["display_width"] = args.display_width
    if args.branch_threading is not None:
        pipeline["branch_threading"] = bool(args.branch_threading)
    if args.branch_workers is not None:
        pipeline["branch_workers"] = max(1, int(args.branch_workers))

    cfg["camera"] = camera
    cfg["visualization"] = vis
    cfg["inference"] = infer
    cfg["pipeline"] = pipeline
    return cfg


def _resolve_lane_segmentation_hef(configured_path: str | None) -> str:
    if configured_path:
        cfg_path = Path(configured_path)
        if cfg_path.exists() or configured_path not in LANE_SEGMENTATION_HEF_CANDIDATES:
            return configured_path
    for candidate in LANE_SEGMENTATION_HEF_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return configured_path or LANE_SEGMENTATION_HEF_CANDIDATES[0]


def _resolve_yolo26_hef(configured_path: str | None) -> str:
    if configured_path:
        cfg_path = Path(configured_path)
        if cfg_path.exists() or configured_path not in YOLO26_HEF_CANDIDATES:
            return configured_path
    for candidate in YOLO26_HEF_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return configured_path or YOLO26_HEF_CANDIDATES[0]


def _parse_segmentation_size(value: Any) -> int | tuple[int, int]:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, str):
        text = value.strip().lower()
        for sep in ("x", ","):
            if sep in text:
                left, right = text.split(sep, 1)
                return int(left), int(right)
        return int(text)
    raise ValueError(f"Unsupported lane_segmentation_input_size: {value!r}")


def extract_pipeline_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Pull all scalar pipeline parameters out of the nested config dict."""
    infer_cfg    = cfg.get("inference", {})
    ctrl_cfg     = cfg.get("control", {})
    lane_cfg     = cfg.get("lane_detection", {})
    vis_cfg      = cfg.get("visualization", cfg.get("display", {}))
    pipeline_cfg = cfg.get("pipeline", {})
    features_cfg = cfg.get("features", {})
    camera_cfg   = cfg.get("camera", {})

    lane_enabled   = bool(features_cfg.get("lane_detection", True))
    object_enabled = bool(features_cfg.get("object_detection", True))

    model_paths = {}
    if lane_enabled:
        model_paths["lane_segmentation"] = _resolve_lane_segmentation_hef(
            infer_cfg.get("lane_segmentation_hef")
        )
    if object_enabled:
        model_paths["yolo26"] = _resolve_yolo26_hef(
            infer_cfg.get("yolo26_hef")
        )

    show_window = bool(vis_cfg.get("show_window", False))

    return {
        "model_paths":      model_paths,
        "conf_thresh":      float(infer_cfg.get("conf_threshold", 0.45)),
        "yolo_size":        int(infer_cfg.get("yolo_input_size", 640)),
        "segmentation_size": _parse_segmentation_size(
            infer_cfg.get("lane_segmentation_input_size", 640)
        ),
        "deadband_px":      int(ctrl_cfg.get("steering_deadband_px", 10)),
        "max_steer":        float(ctrl_cfg.get("max_steering_value", 1.0)),
        "smooth_alpha":     float(ctrl_cfg.get("smoothing_alpha", 0.25)),
        "smoothing":        bool(ctrl_cfg.get("smoothing_enabled", True)),
        "min_lane_w":       int(lane_cfg.get("min_lane_width_px", 40)),
        "show_window":      show_window,
        "window_title":     vis_cfg.get("window_title", "RC Lane-Centering"),
        "display_width":    int(vis_cfg.get("display_width", camera_cfg.get("width", 1280))),
        "fast_overlay":     bool(vis_cfg.get("fast_overlay", False)),
        "display_wait_ms":  max(1, int(vis_cfg.get("wait_key_ms", 1))),
        "default_throttle": float(ctrl_cfg.get("default_throttle", 0.35)),
        "lane_enabled":     lane_enabled,
        "object_enabled":   object_enabled,
        "behavior_enabled": bool(features_cfg.get("behavior_logic", True)),
        "rc_enabled":       bool(features_cfg.get("rc_control", True)),
        "branch_threading": bool(pipeline_cfg.get("branch_threading", False)),
        "branch_workers":   max(2, int(pipeline_cfg.get("branch_workers", 2))),
        "runtime_mode":     cfg.get("project", {}).get("mode", "custom"),
        "ctrl_cfg":         ctrl_cfg,
        "rc_cfg":           cfg.get("rc_vehicle", {}),
        "camera_cfg":       camera_cfg,
    }


def resolve_camera_overrides(
    cam_cfg: dict[str, Any], video: str | None, save_video: str | None,
    loop_video_override: bool | None = None,
) -> dict[str, Any]:
    """Compute camera flags adjusted for video-render modes."""
    save_render = bool(video and save_video)
    sequential = bool(video)
    if sequential:
        # Video testing should be finite by default so direct checks
        # terminate without requiring an output writer.
        loop_video = False
        if loop_video_override is not None:
            loop_video = bool(loop_video_override)
    else:
        loop_video = bool(cam_cfg.get("loop_video", True))
    return {
        "use_gstreamer": False if sequential else bool(cam_cfg.get("use_gstreamer", True)),
        "prefer_low_latency": False if sequential else bool(cam_cfg.get("prefer_low_latency", True)),
        "drop_frames": False if sequential else bool(cam_cfg.get("drop_frames", True)),
        "threaded_capture": False if sequential else bool(cam_cfg.get("threaded_capture", True)),
        "loop_video": False if save_render else loop_video,
        "save_video_render": save_render,
    }
