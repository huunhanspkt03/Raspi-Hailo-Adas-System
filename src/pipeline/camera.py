"""
CameraSource — Camera capture for Raspberry Pi 5
Supports: RPi Camera Module 3 (libcamera/GStreamer), USB cameras, and video files.
"""
from __future__ import annotations

from typing import Optional
import time
import numpy as np
import cv2
from loguru import logger

from pipeline.camera_video import open_video_capture
from pipeline.threaded_reader import ThreadedReader

GSTREAMER_CSI_PIPELINE = (
    "libcamerasrc ! "
    "video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
    "videoconvert ! "
    "appsink emit-signals=True sync=False max-buffers=1 drop=True"
)


class CameraSource:
    """
    Context-manager camera source for ADAS pipeline.

    Usage:
        with CameraSource() as cam:
            frame = cam.read_frame()
        with CameraSource(video_path="test.mp4") as cam:
            frame = cam.read_frame()
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        use_gstreamer: bool = True,
        device_index: int = 0,
        video_path: Optional[str] = None,
        prefer_low_latency: bool = True,
        drop_frames: bool = True,
        threaded_capture: bool = True,
        loop_video: bool = True,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.use_gstreamer = use_gstreamer
        self.device_index = device_index
        self.video_path = video_path
        self.prefer_low_latency = prefer_low_latency
        self.drop_frames = drop_frames
        self.threaded_capture = threaded_capture
        self.loop_video = loop_video
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_interval: float = 0.0
        self._source_fps: float = float(fps)
        self._video_uses_gstreamer = False
        self._reader: Optional[ThreadedReader] = None
        self._eof = False

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self._release()

    def _open(self):
        self._eof = False
        if self.video_path:
            self._open_video_capture()
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            src_fps = self._cap.get(cv2.CAP_PROP_FPS) or float(self.fps) or 30.0
            self._source_fps = float(src_fps)
            self._frame_interval = 1.0 / self._source_fps
            logger.info(
                f"Video open: {w}x{h} @ {src_fps:.1f}fps → "
                f"replaying at {self.width}x{self.height} @ {self._source_fps:.1f}fps"
            )
            if self.threaded_capture:
                self._start_reader()
            return

        if self.use_gstreamer:
            pipeline = GSTREAMER_CSI_PIPELINE.format(
                width=self.width, height=self.height, fps=self.fps
            )
            logger.info(f"Opening GStreamer pipeline: {pipeline[:60]}...")
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            logger.info(f"Opening USB camera at index {self.device_index}")
            self._cap = cv2.VideoCapture(self.device_index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self._cap.isOpened():
            if self.use_gstreamer:
                logger.warning("GStreamer pipeline failed. Falling back to USB camera.")
                self.use_gstreamer = False
                self.device_index = 0
                self._open()
            else:
                raise RuntimeError(
                    "Cannot open any camera source.\n"
                    "Check: libcamera-hello --list-cameras\n"
                    "Check: ls /dev/video*"
                )

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._source_fps = float(self.fps)
        logger.info(f"Camera open: {actual_w}x{actual_h} @ {self.fps}fps")
        if self.threaded_capture:
            self._start_reader()

    def _open_video_capture(self) -> None:
        logger.info(f"Opening video file: {self.video_path}")
        self._eof = False
        self._cap, self._video_uses_gstreamer = open_video_capture(
            self.video_path, self.width, self.height,
            self.use_gstreamer, self.drop_frames,
        )

    def _start_reader(self) -> None:
        if self._reader is not None:
            return
        self._reader = ThreadedReader(self._read_once)
        self._reader.start()

    def _read_once(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None

        t0 = time.monotonic()
        ret, frame = self._cap.read()
        if not ret:
            if self.video_path:
                if not self.loop_video:
                    self._eof = True
                    return None
                if self._video_uses_gstreamer:
                    self._cap.release()
                    self._open_video_capture()
                else:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    return None
            else:
                logger.warning("Frame drop — camera read failed")
                return None

        if self.video_path and (frame.shape[1] != self.width or frame.shape[0] != self.height):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        if self.video_path and not self.prefer_low_latency:
            elapsed = time.monotonic() - t0
            remaining = self._frame_interval - elapsed
            if remaining > 0.001:
                time.sleep(remaining)

        return frame

    def read_frame(self) -> Optional[np.ndarray]:
        """Read one frame. Returns None on drop. Loops video files on end."""
        if not self.threaded_capture or self._reader is None:
            return self._read_once()
        return self._reader.read()

    def get_source_fps(self) -> float:
        """Return the natural FPS of the active source."""
        return max(1e-6, float(self._source_fps))

    def is_exhausted(self) -> bool:
        """Return True when a finite video source reached EOF without looping."""
        return bool(self._eof)

    def _release(self):
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Camera released.")
