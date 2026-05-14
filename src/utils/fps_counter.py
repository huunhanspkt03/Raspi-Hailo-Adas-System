from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class FPSCounter:
    """
    Simple FPS counter for real-time pipelines.

    Usage:
        fps = FPSCounter()
        ...
        fps.update()
        current_fps = fps.get()
    """

    start_time: float = field(default_factory=time.perf_counter)
    last_time: float = field(default_factory=time.perf_counter)
    frame_count: int = 0
    instant_fps: float = 0.0

    def reset(self) -> None:
        now = time.perf_counter()
        self.start_time = now
        self.last_time = now
        self.frame_count = 0
        self.instant_fps = 0.0

    def update(self) -> float:
        """
        Update frame count and return current average FPS.
        Also keeps a simple instant FPS estimate between frames.
        """
        now = time.perf_counter()
        dt = now - self.last_time

        self.frame_count += 1
        self.last_time = now

        if dt > 0:
            self.instant_fps = 1.0 / dt

        return self.get()

    def get(self) -> float:
        """
        Return average FPS since start/reset.
        """
        elapsed = max(time.perf_counter() - self.start_time, 1e-6)
        return self.frame_count / elapsed

    def get_instant(self) -> float:
        """
        Return instantaneous FPS estimated from the most recent frame interval.
        """
        return self.instant_fps

    def elapsed(self) -> float:
        """
        Return elapsed time in seconds since start/reset.
        """
        return max(time.perf_counter() - self.start_time, 0.0)

    def should_log(self, every_n_frames: int) -> bool:
        """
        Convenience helper for periodic logging.
        """
        if every_n_frames <= 0:
            return False
        return self.frame_count > 0 and self.frame_count % every_n_frames == 0
