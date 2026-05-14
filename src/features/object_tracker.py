"""
Minimal ByteTrack-style detection tracker.

Two-stage IoU association (no Kalman; last observed box is the prediction):
  1. Match high-confidence detections to existing tracks.
  2. Match still-unmatched tracks against low-confidence detections — the
     ByteTrack idea that recovers occluded/low-score boxes.
  3. Unmatched high-confidence detections start new tentative tracks;
     they become ``confirmed`` after ``confirm_hits`` consecutive hits.
  4. Unmatched tracks age out after ``max_age`` missed frames.

Matching is per-class: a ``car`` track never absorbs a ``truck`` detection.

Public API:
    tracker = ByteTracker()
    tracked = tracker.update(detections)   # same dicts, with "track_id"/"track_age"
    tracker.reset()
"""
from __future__ import annotations

import numpy as np


def _iou_xyxy(a: list[int] | tuple[int, int, int, int],
              b: list[int] | tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _Track:
    __slots__ = ("id", "label", "box", "score", "age", "missed", "hits", "confirmed")

    def __init__(self, tid: int, label: str, box, score: float) -> None:
        self.id = tid
        self.label = label
        self.box = list(box)
        self.score = float(score)
        self.age = 1
        self.missed = 0
        self.hits = 1
        self.confirmed = False

    def observe(self, det: dict, confirm_hits: int) -> None:
        self.box = list(det["box"])
        self.score = float(det["score"])
        self.hits += 1
        self.missed = 0
        self.age += 1
        if self.hits >= confirm_hits:
            self.confirmed = True


class ByteTracker:
    def __init__(
        self,
        high_thresh: float = 0.50,
        low_thresh: float = 0.10,
        iou_thresh_high: float = 0.30,
        iou_thresh_low: float = 0.20,
        max_age: int = 20,
        confirm_hits: int = 3,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh_high = iou_thresh_high
        self.iou_thresh_low = iou_thresh_low
        self.max_age = max_age
        self.confirm_hits = confirm_hits
        self._tracks: list[_Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1

    def _match(
        self, tracks: list[_Track], dets: list[dict], iou_thresh: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))

        iou_mat = np.zeros((len(tracks), len(dets)), dtype=np.float32)
        for ti, t in enumerate(tracks):
            for di, d in enumerate(dets):
                if d["label"] != t.label:
                    continue
                iou_mat[ti, di] = _iou_xyxy(t.box, d["box"])

        matches: list[tuple[int, int]] = []
        used_t: set[int] = set()
        used_d: set[int] = set()
        while True:
            ti, di = np.unravel_index(int(np.argmax(iou_mat)), iou_mat.shape)
            if iou_mat[ti, di] < iou_thresh:
                break
            matches.append((int(ti), int(di)))
            used_t.add(int(ti))
            used_d.add(int(di))
            iou_mat[ti, :] = 0.0
            iou_mat[:, di] = 0.0

        um_t = [i for i in range(len(tracks)) if i not in used_t]
        um_d = [i for i in range(len(dets)) if i not in used_d]
        return matches, um_t, um_d

    def update(self, detections: list[dict]) -> list[dict]:
        """Assign ``track_id`` and ``track_age`` to each high/low-conf detection.

        Returns the same list (dicts mutated in place). Detections below
        ``low_thresh`` are not tracked and receive no track fields.
        """
        if not detections:
            for t in self._tracks:
                t.missed += 1
                t.age += 1
            self._tracks = [t for t in self._tracks if t.missed <= self.max_age]
            return detections

        high = [d for d in detections if d["score"] >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d["score"] < self.high_thresh]

        # Stage 1 — high-conf dets to existing tracks.
        m1, um_t1, um_d1 = self._match(self._tracks, high, self.iou_thresh_high)
        for ti, di in m1:
            self._tracks[ti].observe(high[di], self.confirm_hits)
            high[di]["track_id"] = self._tracks[ti].id
            high[di]["track_age"] = self._tracks[ti].age

        # Stage 2 — still-unmatched tracks to low-conf dets (ByteTrack recovery).
        rem_tracks = [self._tracks[i] for i in um_t1]
        m2, um_t_local, _ = self._match(rem_tracks, low, self.iou_thresh_low)
        matched_low: set[int] = set()
        for ti_local, di in m2:
            t = rem_tracks[ti_local]
            t.observe(low[di], self.confirm_hits)
            low[di]["track_id"] = t.id
            low[di]["track_age"] = t.age
            matched_low.add(di)

        # Age out fully unmatched tracks.
        for i_local in um_t_local:
            t = rem_tracks[i_local]
            t.missed += 1
            t.age += 1
        self._tracks = [t for t in self._tracks if t.missed <= self.max_age]

        # Spawn new tracks from unmatched high-conf dets.
        for di in um_d1:
            d = high[di]
            t = _Track(self._next_id, d["label"], d["box"], d["score"])
            self._next_id += 1
            self._tracks.append(t)
            d["track_id"] = t.id
            d["track_age"] = t.age

        return detections

    def active_track_ids(self) -> list[int]:
        return [t.id for t in self._tracks if t.confirmed and t.missed == 0]
