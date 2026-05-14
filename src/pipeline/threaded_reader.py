"""Threaded frame reader — latest-frame-only delivery for low-latency capture."""
from __future__ import annotations

import time
from threading import Condition, Event, Thread
from typing import Callable, Optional

import numpy as np


class ThreadedReader:
    """Background thread that continuously reads frames and keeps only the latest.

    Parameters
    ----------
    read_fn : callable
        A zero-arg function that returns a frame (np.ndarray) or None on failure.
    """

    def __init__(self, read_fn: Callable[[], Optional[np.ndarray]]) -> None:
        self._read_fn = read_fn
        self._stop_event = Event()
        self._frame_ready = Condition()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_seq = 0
        self._delivered_seq = 0
        self._thread: Optional[Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._loop, name="camera-reader", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            frame = self._read_fn()
            if frame is None:
                time.sleep(0.005)
                continue
            with self._frame_ready:
                self._latest_frame = frame
                self._latest_seq += 1
                self._frame_ready.notify_all()

    def read(self) -> Optional[np.ndarray]:
        """Block until a new frame is available, then return it."""
        with self._frame_ready:
            while not self._stop_event.is_set() and self._latest_seq == self._delivered_seq:
                self._frame_ready.wait(timeout=0.5)

            if self._latest_frame is None:
                return None

            self._delivered_seq = self._latest_seq
            return self._latest_frame

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._frame_ready:
            self._latest_frame = None
            self._latest_seq = 0
            self._delivered_seq = 0
