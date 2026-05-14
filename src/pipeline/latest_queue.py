"""Bounded latest-value queue used for fan-out between runtime threads.

Semantics:
  - ``put`` never blocks; if the queue is full the OLDEST item is dropped
    (drop-oldest), matching the flowchart goal "drop stale frames/results
    instead of building backlog".
  - ``get`` blocks up to ``timeout`` waiting for a value; returns ``None``
    on timeout or after ``close()``.
  - Safe for one or many producers / consumers.

This is intentionally a very small primitive — the runtime uses one
instance per edge of the pipeline graph. Each lane/object worker reads
from its own LatestQueue so capture can fan-out without consumers
competing for frames.
"""
from __future__ import annotations

import time
from collections import deque
from threading import Condition
from typing import Generic, Iterable, TypeVar

T = TypeVar("T")


class LatestQueue(Generic[T]):
    def __init__(self, maxsize: int = 1) -> None:
        if maxsize < 1:
            raise ValueError("LatestQueue maxsize must be >= 1")
        self._maxsize = maxsize
        self._buf: deque[T] = deque(maxlen=maxsize)
        self._cond = Condition()
        self._closed = False
        self._dropped = 0

    def put(self, item: T) -> None:
        """Insert an item, evicting the oldest if the buffer is full."""
        with self._cond:
            if self._closed:
                return
            if len(self._buf) == self._maxsize:
                self._dropped += 1
            self._buf.append(item)
            self._cond.notify()

    def get(self, timeout: float | None = 0.1) -> T | None:
        """Return the next item, or ``None`` on timeout / after close."""
        with self._cond:
            deadline = None if timeout is None else time.monotonic() + timeout
            while not self._buf and not self._closed:
                if timeout is None:
                    self._cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(timeout=remaining)
            if self._buf:
                return self._buf.popleft()
            return None

    def try_get(self) -> T | None:
        """Non-blocking pop; returns ``None`` if empty."""
        with self._cond:
            if self._buf:
                return self._buf.popleft()
            return None

    def drain(self) -> Iterable[T]:
        with self._cond:
            items = list(self._buf)
            self._buf.clear()
            return items

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def clear(self) -> None:
        with self._cond:
            self._buf.clear()

    @property
    def dropped(self) -> int:
        with self._cond:
            return self._dropped

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def __len__(self) -> int:
        with self._cond:
            return len(self._buf)
