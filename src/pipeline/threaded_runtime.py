"""Threaded ADAS runtime — implements the 12_threading_architecture flow.

Thread topology:

    Capture ──► lane_q  ──► Lane  ──┐
         │                           ├─► Fusion ──► behavior_q ──► Output
         └──► object_q ─► Object ──┘

Design rules matched to the flowchart:
  - Fan-out: capture pushes each frame into *both* lane_q and object_q
    (per-consumer latest-value queues). Lane and Object never compete
    for frames.
  - Drop-stale: every queue is bounded (maxsize=1) and drops the oldest
    on put. Low latency > completeness.
  - Fusion synchronizes lane/object packets by ``frame_id`` with a small
    stale tolerance. In lane-only / detection-only modes it passes the
    available side through immediately.
  - Graceful shutdown: stop Event + thread.join, queues closed, output
    thread owns display/VideoWriter/RC hand-off.
"""
from __future__ import annotations

import signal
from threading import Event, Lock, Thread
from typing import Any, Callable

import cv2
from loguru import logger

from pipeline.fusion_worker import (
    FusionWorker, drain_state_queues, emit_fused, safety_supervisor,
)
from pipeline.latest_queue import LatestQueue
from pipeline.output_renderer import (
    open_video_writer, render_behavior_packet, resize_for_display,
)
from pipeline.packets import BehaviorPacket, FramePacket, LanePacket, ObjectPacket
from pipeline.thread_workers import LaneWorker, ObjectWorker, now_ts
from utils.fps_counter import FPSCounter

STALE_FRAME_TOLERANCE = 8  # frames — fusion gives up matching after this gap


class ThreadedRuntime:
    """Owns all threads and queues for the decoupled pipeline."""

    def __init__(
        self,
        cam: Any,
        infer: Any,
        params: dict,
        args: Any,
        *,
        send_rc_fn: Callable[[float, float, bool], None],
        render_output: bool,
        save_video_path: str | None,
        should_stop: Callable[[], bool] | None = None,
        behavior_smoother: Any | None = None,
        hailo_active: bool = False,
        segmentation_size: int | tuple[int, int] = 512,
        yolo_size: int = 640,
    ) -> None:
        self._cam = cam
        self._infer = infer
        self._p = params
        self._args = args
        self._send_rc = send_rc_fn
        self._render_output = render_output
        self._save_video_path = save_video_path
        self._should_stop_cb = should_stop
        self._hailo_active = hailo_active

        self._stop = Event()
        self._infer_lock = Lock()
        self._lane_q: LatestQueue[FramePacket] = LatestQueue(maxsize=1)
        self._object_q: LatestQueue[FramePacket] = LatestQueue(maxsize=1)
        self._lane_state_q: LatestQueue[LanePacket] = LatestQueue(maxsize=2)
        self._object_state_q: LatestQueue[ObjectPacket] = LatestQueue(maxsize=2)
        self._behavior_q: LatestQueue[BehaviorPacket] = LatestQueue(maxsize=1)

        self._threads: list[Thread] = []
        self._fps_capture = FPSCounter()
        self._fps_output = FPSCounter()
        self._last_frame_id = 0

        self._lane_worker: LaneWorker | None = None
        self._object_worker: ObjectWorker | None = None
        if params["lane_enabled"]:
            self._lane_worker = LaneWorker(
                infer, self._infer_lock,
                segmentation_size=segmentation_size,
                min_lane_w=params["min_lane_w"],
                steering_source=str(params["ctrl_cfg"].get("steering_source", "lane_marking_center")),
            )
        if params["object_enabled"]:
            self._object_worker = ObjectWorker(
                infer, self._infer_lock,
                yolo_size=yolo_size, conf_thresh=params["conf_thresh"],
            )
        self._fusion = FusionWorker(
            ctrl_cfg=params["ctrl_cfg"], rc_cfg=params["rc_cfg"],
            deadband_px=params["deadband_px"], max_steer=params["max_steer"],
            smooth_alpha=params["smooth_alpha"], smoothing=params["smoothing"],
            default_throttle=params["default_throttle"],
            behavior_enabled=params["behavior_enabled"],
            rc_enabled=params["rc_enabled"],
            behavior_smoother=behavior_smoother,
        )

    def request_stop(self) -> None:
        self._stop.set()
        for q in (self._lane_q, self._object_q, self._lane_state_q,
                  self._object_state_q, self._behavior_q):
            q.close()

    def _should_stop(self) -> bool:
        if self._stop.is_set():
            return True
        if self._should_stop_cb is not None and self._should_stop_cb():
            self._stop.set()
            return True
        return False

    def run(self) -> None:
        self._start_threads()
        writer: cv2.VideoWriter | None = None
        window_opened = False
        p = self._p
        try:
            if p["show_window"] and self._render_output:
                cv2.namedWindow(p["window_title"], cv2.WINDOW_NORMAL)
                cv2.resizeWindow(p["window_title"],
                                 p["display_width"], p["display_width"] * 9 // 16)
                window_opened = True
            writer = self._output_loop(writer)
        finally:
            self.request_stop()
            for t in self._threads:
                t.join(timeout=2.0)
            if writer is not None:
                writer.release()
                logger.info("Saved output video: {}", self._save_video_path)
            if window_opened:
                cv2.destroyAllWindows()
            logger.info("Threaded runtime stopped cleanly.")

    # ---- worker loops --------------------------------------------------

    def _start_threads(self) -> None:
        self._threads.append(Thread(target=self._capture_loop,
                                    name="adas-capture", daemon=True))
        if self._lane_worker is not None:
            self._threads.append(Thread(target=self._lane_loop,
                                        name="adas-lane", daemon=True))
        if self._object_worker is not None:
            self._threads.append(Thread(target=self._object_loop,
                                        name="adas-object", daemon=True))
        self._threads.append(Thread(target=self._fusion_loop,
                                    name="adas-fusion", daemon=True))
        for t in self._threads:
            t.start()

    def _capture_loop(self) -> None:
        while not self._should_stop():
            frame = self._cam.read_frame()
            if frame is None:
                if getattr(self._args, "video", None) and hasattr(self._cam, "is_exhausted") \
                        and self._cam.is_exhausted():
                    logger.info("Reached end of video source.")
                    self.request_stop()
                    return
                continue
            self._last_frame_id += 1
            pkt = FramePacket.make(self._last_frame_id, now_ts(), frame)
            if self._lane_worker is not None:
                self._lane_q.put(pkt)
            if self._object_worker is not None:
                self._object_q.put(pkt)
            self._fps_capture.update()

    def _lane_loop(self) -> None:
        assert self._lane_worker is not None
        while not self._should_stop():
            pkt = self._lane_q.get(timeout=0.1)
            if pkt is None:
                continue
            try:
                lane_pkt = self._lane_worker.process(pkt)
            except Exception as exc:
                logger.error("Lane worker error: {}", exc)
                continue
            self._lane_state_q.put(lane_pkt)

    def _object_loop(self) -> None:
        assert self._object_worker is not None
        while not self._should_stop():
            pkt = self._object_q.get(timeout=0.1)
            if pkt is None:
                continue
            try:
                obj_pkt = self._object_worker.process(pkt)
            except Exception as exc:
                logger.error("Object worker error: {}", exc)
                continue
            if self._lane_worker is not None:
                self._lane_worker.set_prev_detections(obj_pkt.detections)
            self._object_state_q.put(obj_pkt)

    def _fusion_loop(self) -> None:
        pending_lane: dict[int, LanePacket] = {}
        pending_obj: dict[int, ObjectPacket] = {}
        want_lane = self._lane_worker is not None
        want_obj = self._object_worker is not None
        while not self._should_stop():
            got_any = drain_state_queues(
                self._lane_state_q, self._object_state_q,
                pending_lane, pending_obj, want_lane, want_obj,
            )
            if not got_any:
                continue
            emit_fused(
                self._fusion, self._behavior_q,
                pending_lane, pending_obj, want_lane, want_obj,
                stale_tolerance=STALE_FRAME_TOLERANCE,
            )

    # ---- output --------------------------------------------------------

    def _output_loop(self, writer: cv2.VideoWriter | None) -> cv2.VideoWriter | None:
        p = self._p
        while not self._should_stop():
            pkt = self._behavior_q.get(timeout=0.1)
            if pkt is None:
                continue
            pkt = safety_supervisor(pkt, max_steer=p["max_steer"], rc_send=self._send_rc)
            if self._render_output:
                out = render_behavior_packet(
                    pkt, fast_overlay=p["fast_overlay"],
                    hailo_active=self._hailo_active,
                    display_fps=self._fps_output.get(),
                )
                orig_h, orig_w = pkt.frame.shape[:2]
                out = resize_for_display(out, p["display_width"], orig_w, orig_h)
                writer = self._write_frame(out, writer)
                if p["show_window"]:
                    cv2.imshow(p["window_title"], out)
                    if cv2.waitKey(p["display_wait_ms"]) & 0xFF == ord("q"):
                        self.request_stop()
                        break
            self._fps_output.update()
            if self._fps_output.should_log(100):
                logger.info(
                    "threaded | out_fps={:.1f} | cap_fps={:.1f} | steer={:+.3f} | thr={:.2f} | {}",
                    self._fps_output.get(), self._fps_capture.get(),
                    pkt.control.steering, pkt.control.throttle, pkt.behavior_action,
                )
        return writer

    def _write_frame(self, out: Any, writer: cv2.VideoWriter | None) -> cv2.VideoWriter | None:
        if not self._save_video_path or out is None:
            return writer
        if writer is None:
            cam_cfg = self._p["camera_cfg"]
            output_fps = float(self._cam.get_source_fps()) \
                if hasattr(self._cam, "get_source_fps") else float(cam_cfg.get("fps", 30))
            writer = open_video_writer(
                self._save_video_path, (out.shape[1], out.shape[0]), output_fps,
            )
        writer.write(out)
        return writer


def install_signal_handler(runtime: ThreadedRuntime) -> None:
    def _handler(*_: Any) -> None:
        runtime.request_stop()
    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, AttributeError):
        pass


def run_threaded(
    cam: Any, infer: Any, params: dict, args: Any,
    *,
    send_rc_fn: Callable[[float, float, bool], None],
    render_output: bool,
    save_video_path: str | None,
    should_stop: Callable[[], bool] | None = None,
    behavior_smoother: Any | None = None,
    hailo_active: bool = False,
    segmentation_size: int | tuple[int, int] = 512,
    yolo_size: int = 640,
) -> None:
    rt = ThreadedRuntime(
        cam, infer, params, args,
        send_rc_fn=send_rc_fn, render_output=render_output,
        save_video_path=save_video_path, should_stop=should_stop,
        behavior_smoother=behavior_smoother, hailo_active=hailo_active,
        segmentation_size=segmentation_size, yolo_size=yolo_size,
    )
    rt.run()
