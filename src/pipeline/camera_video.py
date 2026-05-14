"""Video file capture helpers for CameraSource."""
from __future__ import annotations

from pathlib import Path

import cv2
from loguru import logger

GSTREAMER_VIDEO_PIPELINE = (
    'uridecodebin uri="{uri}" ! '
    "videoconvert ! "
    "videoscale ! "
    "video/x-raw,format=BGR,width={width},height={height} ! "
    "appsink emit-signals=True sync=false max-buffers=1 drop={drop}"
)


def open_video_capture(
    video_path: str,
    width: int,
    height: int,
    use_gstreamer: bool,
    drop_frames: bool,
) -> tuple[cv2.VideoCapture, bool]:
    """Open a video file, trying GStreamer first if requested.

    Returns:
        (capture, used_gstreamer) — the opened VideoCapture and whether
        the GStreamer pipeline was actually used.
    """
    cap = None
    used_gstreamer = False

    if use_gstreamer:
        uri = Path(video_path).resolve().as_uri()
        pipeline = GSTREAMER_VIDEO_PIPELINE.format(
            uri=uri, width=width, height=height,
            drop=str(drop_frames).lower(),
        )
        logger.info("Opening GStreamer video pipeline for low-latency decode.")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            used_gstreamer = True
        else:
            logger.warning("GStreamer video pipeline failed. Falling back to OpenCV file decode.")
            cap = None

    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    return cap, used_gstreamer
