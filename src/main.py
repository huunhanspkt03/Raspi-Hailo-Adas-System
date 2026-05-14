#!/usr/bin/env python3
"""
RC Lane-Centering Pipeline — Raspberry Pi 5 + Hailo-8 AI HAT+ (26 TOPS)

Thin orchestration layer:
  1. Load config + open camera
  2. YOLO26s_seg lane segmentation → lane center-path → steering
  3. YOLO26 traffic detection   → behavior constraints
  4. Combine → RC control output + optional debug display
"""
from __future__ import annotations

import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from loguru import logger

from config_loader import parse_args, load_config, apply_cli_overrides, extract_pipeline_params, resolve_camera_overrides
from pipeline.camera import CameraSource
from pipeline.inference import create_inference
from pipeline.preprocessor import resolve_runtime_input_size
from pipeline.postprocessor import draw_hud
from pipeline_loop import run_raw_overlay_frame, run_pipeline_frame
from pipeline.threaded_runtime import run_threaded
from features.behavior_logic import BehaviorSmoother
from features.behavior_runner import run_behavior
from features.control_runner import send_rc_command
from features.validation_metrics_logger import MetricsLogger
from utils.fps_counter import FPSCounter
from runtime_mode import apply_runtime_mode, sanitize_runtime_features


def send_rc_control(steering: float, throttle: float, enabled: bool) -> None:
    _ = steering, throttle, enabled


def main() -> None:
    args = parse_args()
    cfg  = apply_cli_overrides(load_config(args.config), args)

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=cfg.get("logging", {}).get("level", "INFO"))

    cfg = apply_runtime_mode(cfg, args.mode)
    cfg, runtime_notes = sanitize_runtime_features(cfg)
    for note in runtime_notes:
        logger.warning(note)

    p = extract_pipeline_params(cfg)
    if args.raw_overlay and not p["lane_enabled"]:
        raise ValueError("--raw-overlay requires lane/segmentation mode.")

    save_video_path = args.save_video
    render_output   = p["show_window"] or bool(save_video_path)
    writer: cv2.VideoWriter | None = None

    stop = {"v": False}
    signal.signal(signal.SIGINT,  lambda *_: stop.update({"v": True}))
    signal.signal(signal.SIGTERM, lambda *_: stop.update({"v": True}))

    source_desc = f"video:{args.video}" if args.video else (
        "csi" if p["camera_cfg"].get("use_gstreamer", True) else "usb"
    )
    logger.info(
        "Starting ADAS pipeline | mode={} | lane={} | detection={} | source={} | onnx={} | display={} | branch_threading={}",
        p["runtime_mode"], p["lane_enabled"], p["object_enabled"], source_desc,
        args.onnx, p["show_window"], p["branch_threading"] and p["lane_enabled"] and p["object_enabled"],
    )
    if p["show_window"] and save_video_path:
        logger.info("Display + save mode: showing live output and writing video to {}", save_video_path)
    elif p["show_window"]:
        logger.info("Display-only mode: showing live output without saving video. Add --save-video <path> to record.")
    elif save_video_path:
        logger.info("Headless render mode: saving annotated output to {} without opening a window", save_video_path)

    fps              = FPSCounter()
    steering_smooth: float | None = None
    branch_executor: ThreadPoolExecutor | None = None
    behavior_smoother = None
    metrics_logger = MetricsLogger(
        path=getattr(args, "metrics_out", None),
        fmt=getattr(args, "metrics_format", "jsonl"),
    )
    if metrics_logger.enabled:
        logger.info("Per-frame metrics → {} ({})", args.metrics_out, args.metrics_format)

    if p["branch_threading"] and p["lane_enabled"] and p["object_enabled"]:
        branch_executor = ThreadPoolExecutor(max_workers=p["branch_workers"], thread_name_prefix="adas-branch")
        logger.info("Branch threading enabled | workers={}", p["branch_workers"])

    if p["behavior_enabled"]:
        behavior_smoother = BehaviorSmoother(
            window=max(1, int(p["ctrl_cfg"].get("behavior_smoothing_window", 5))),
            release_frames=max(1, int(p["ctrl_cfg"].get("behavior_release_frames", 3))),
        )

    if p["show_window"]:
        cv2.namedWindow(p["window_title"], cv2.WINDOW_NORMAL)
        cv2.resizeWindow(p["window_title"], p["display_width"], p["display_width"] * 9 // 16)

    cam_cfg = p["camera_cfg"]
    cam_ov = resolve_camera_overrides(
        cam_cfg, args.video, save_video_path,
        loop_video_override=getattr(args, "loop_video", None),
    )
    if cam_ov["save_video_render"]:
        logger.info("Video render mode: sequential file playback enabled to preserve all source frames.")

    try:
        with CameraSource(
            width=int(cam_cfg.get("width", 1280)),
            height=int(cam_cfg.get("height", 720)),
            fps=int(cam_cfg.get("fps", 30)),
            use_gstreamer=cam_ov["use_gstreamer"],
            device_index=int(cam_cfg.get("usb_device_index", 0)),
            video_path=args.video,
            prefer_low_latency=cam_ov["prefer_low_latency"],
            drop_frames=cam_ov["drop_frames"],
            threaded_capture=cam_ov["threaded_capture"],
            loop_video=cam_ov["loop_video"],
        ) as cam, create_inference(p["model_paths"], force_onnx=args.onnx) as infer:

            hailo_active = not args.onnx and Path("/dev/hailo0").exists()
            segmentation_size = p["segmentation_size"]
            yolo_size = p["yolo_size"]
            if p["lane_enabled"]:
                segmentation_size = resolve_runtime_input_size(
                    infer.get_input_shape("lane_segmentation"), segmentation_size,
                )
            if p["object_enabled"]:
                yolo_size = resolve_runtime_input_size(
                    infer.get_input_shape("yolo26"), yolo_size,
                )
            logger.info(
                "Resolved branch inputs | lane={} | yolo={}",
                segmentation_size if p["lane_enabled"] else "off",
                yolo_size if p["object_enabled"] else "off",
            )

            if args.threaded and not args.raw_overlay:
                run_threaded(
                    cam, infer, p, args,
                    send_rc_fn=send_rc_control,
                    render_output=render_output,
                    save_video_path=save_video_path,
                    should_stop=lambda: stop["v"],
                    behavior_smoother=behavior_smoother,
                    hailo_active=hailo_active,
                    segmentation_size=segmentation_size,
                    yolo_size=yolo_size,
                )
                return

            while not stop["v"]:
                loop_t0 = time.perf_counter()
                frame = cam.read_frame()
                cam_t1 = time.perf_counter()
                if frame is None:
                    if args.video and hasattr(cam, "is_exhausted") and cam.is_exhausted():
                        logger.info("Reached end of video source.")
                        break
                    continue

                orig_h, orig_w = frame.shape[:2]
                proc_w = p["display_width"] if p["show_window"] else orig_w
                proc_h = proc_w * orig_h // orig_w
                out = frame.copy() if render_output else None

                if args.raw_overlay:
                    out, ms_pre, ms_inf, ms_dec, ms_ov = run_raw_overlay_frame(
                        frame, out, cam_t1, loop_t0, infer, fps,
                        segmentation_size, p["fast_overlay"], render_output,
                    )
                else:
                    out, steering_smooth, perception = run_pipeline_frame(
                        frame, out, infer, fps,
                        lane_enabled=p["lane_enabled"], object_enabled=p["object_enabled"],
                        behavior_enabled=p["behavior_enabled"], rc_enabled=p["rc_enabled"],
                        segmentation_size=segmentation_size, yolo_size=yolo_size,
                        conf_thresh=p["conf_thresh"], frame_w=orig_w, frame_h=orig_h,
                        min_lane_w=p["min_lane_w"], deadband_px=p["deadband_px"],
                        max_steer=p["max_steer"], smooth_alpha=p["smooth_alpha"], smoothing=p["smoothing"],
                        default_throttle=p["default_throttle"], fast_overlay=p["fast_overlay"],
                        render_output=render_output, branch_executor=branch_executor,
                        behavior_smoother=behavior_smoother, ctrl_cfg=p["ctrl_cfg"],
                        hailo_active=hailo_active, steering_smooth=steering_smooth,
                        send_rc_fn=send_rc_control, rc_cfg=p["rc_cfg"],
                        lane_debug=getattr(args, "lane_debug", False),
                        metrics_logger=metrics_logger,
                    )
                    # Compute behavior (moved out of perception) and send RC if enabled
                    if p["behavior_enabled"]:
                        state = run_behavior(perception.detections or [], p["ctrl_cfg"], orig_w, behavior_smoother)
                        behavior_action = state.action
                        behavior_reason = state.reason
                        throttle = float(state.throttle)
                    else:
                        if p["lane_enabled"] and not p["object_enabled"]:
                            behavior_action, behavior_reason = "go", "lane_only"
                        elif p["object_enabled"] and not p["lane_enabled"]:
                            behavior_action, behavior_reason = "observe", "detection_only"
                        else:
                            behavior_action, behavior_reason = "go", "full_stack"
                        throttle = float(p["default_throttle"])

                    if p["rc_enabled"]:
                        # Actuation moved to control layer wrapper
                        send_rc_command(send_rc_control, perception.lane_state, throttle, p["rc_cfg"])

                    if render_output and out is not None:
                        out = draw_hud(
                            out, fps.get(), perception.lane_state,
                            throttle, behavior_action, behavior_reason,
                            hailo_active,
                            perception.object_state.get("leading_vehicle")
                            if perception.object_state else None,
                            copy_frame=False,
                            ego_speed=perception.ego_speed,
                        )

                    if fps.should_log(100):
                        logger.info(
                            "FPS={:.1f} | dets={} | steer={:+.3f} | throttle={:.2f} | {}",
                            fps.get(), len(perception.detections or []), perception.lane_state.steering, throttle, behavior_action,
                        )

                if render_output and out is not None and (out.shape[1] != proc_w or out.shape[0] != proc_h):
                    out = cv2.resize(out, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

                if save_video_path and out is not None:
                    if writer is None:
                        out_path = Path(save_video_path)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        output_fps = (
                            float(cam.get_source_fps()) if args.video and hasattr(cam, "get_source_fps")
                            else float(cam_cfg.get("fps", 30))
                        )
                        writer = cv2.VideoWriter(
                            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                            output_fps, (out.shape[1], out.shape[0]),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot open VideoWriter: {out_path}")
                        logger.info("Saving output video to {} @ {:.3f} fps", out_path, output_fps)
                    writer.write(out)

                if p["show_window"] and out is not None:
                    cv2.imshow(p["window_title"], out)
                    if cv2.waitKey(p["display_wait_ms"]) & 0xFF == ord("q"):
                        break

                if args.raw_overlay and fps.should_log(30):
                    logger.info(
                        "raw-overlay | FPS={:.1f} | cam={:.0f}ms | pre={:.0f}ms | inf={:.0f}ms | dec={:.0f}ms | ov={:.0f}ms | loop={:.0f}ms",
                        fps.get(), (cam_t1 - loop_t0) * 1e3, ms_pre, ms_inf, ms_dec, ms_ov,
                        (time.perf_counter() - loop_t0) * 1e3,
                    )
    finally:
        if branch_executor is not None:
            branch_executor.shutdown(wait=True)
        if writer is not None:
            writer.release()
            logger.info("Saved output video: {}", save_video_path)
        metrics_logger.close()
        cv2.destroyAllWindows()
        logger.info("Pipeline stopped cleanly.")


if __name__ == "__main__":
    main()
