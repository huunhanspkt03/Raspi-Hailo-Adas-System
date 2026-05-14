"""Robust ego-path shape ``(a, b)`` derived from purple lane coefficients.

Pulls the curvature/slope terms from one or both purple polynomials so
the corridor stays visually centered. Both DETECTED + parallel
curvatures average ``(a, b)`` (perspective slopes cancel); single-side
fallback keeps only the curvature hint and forces ``b = 0``. A top-row
drift cap rejects anything that would slide the top of the ROI away
from the ego column. Rejected frames hold the last shape for up to
``max_hold_frames`` then fall back to ``(0, 0)``.
"""
from __future__ import annotations

from features.lane_boundary_tracker_contract import SOURCE_DETECTED, SOURCE_HELD

__all__ = ["LaneShapeSmoother", "is_reliable_side"]


def is_reliable_side(source: str | None, kind: str | None) -> bool:
    """Reliable shape source = DETECTED or short-term HELD."""
    _ = kind
    if source in (SOURCE_DETECTED, SOURCE_HELD):
        return True
    return False


class LaneShapeSmoother:
    """Stateful ``(a, b)`` shape estimator for the ego path.

    Defaults are tuned for 720-row source frames; tune ``max_curvature_a``
    and ``max_top_disp_px`` for very different image scales.
    """

    def __init__(
        self,
        *,
        max_curvature_a: float = 1.5e-3,
        max_slope_b: float = 2.0,
        max_da_per_frame: float = 1.0e-3,
        max_db_per_frame: float = 0.5,
        parallel_da: float = 8.0e-4,
        ema_alpha: float = 0.30,
        max_hold_frames: int = 6,
        # Strict cap on how far the top of the ROI may drift from the
        # ego column after DC-shifting. 30 px on a 1280-wide frame is
        # ≈2% of the image width — the corridor stays visually centered
        # even on detected curves with small but non-zero average b.
        max_top_disp_px: float = 30.0,
    ) -> None:
        self.max_curvature_a = float(max_curvature_a)
        self.max_slope_b = float(max_slope_b)
        self.max_da_per_frame = float(max_da_per_frame)
        self.max_db_per_frame = float(max_db_per_frame)
        self.parallel_da = float(parallel_da)
        self.ema_alpha = float(ema_alpha)
        self.max_hold_frames = int(max_hold_frames)
        self.max_top_disp_px = float(max_top_disp_px)
        self._last: tuple[float, float] | None = None
        self._held_frames = 0
        self._last_reason: str = "init"

    def reset(self) -> None:
        self._last = None
        self._held_frames = 0
        self._last_reason = "reset"

    def update(
        self,
        left_coeffs: tuple[float, float, float] | None,
        right_coeffs: tuple[float, float, float] | None,
        *,
        left_source: str | None,
        right_source: str | None,
        left_kind: str | None = None,
        right_kind: str | None = None,
        anchor_y: float,
        top_y: float,
        purple_width_trusted: bool = True,
    ) -> tuple[tuple[float, float], str, str]:
        """Return ``((a, b), source_label, reason)`` for the ego path.

        ``source_label`` is one of ``both``, ``left_curvature_only``,
        ``right_curvature_only``, ``held``, ``straight``. Useful for
        diagnostics and overlays.

        ``purple_width_trusted`` is the ``purple_lane_width_valid`` flag
        from the boundary tracker. When ``False`` (e.g., warmup or
        ``warmup_disagree_lane_area`` reject), the smoother does not
        commit a bent shape from fresh purple — it falls back to
        held/straight. Once purple memory validates, normal shape
        acceptance resumes.
        """
        l_ok = (left_coeffs is not None
                and is_reliable_side(left_source, left_kind))
        r_ok = (right_coeffs is not None
                and is_reliable_side(right_source, right_kind))

        proposal = self._proposal(left_coeffs, right_coeffs, l_ok, r_ok)
        if proposal is None:
            return self._held_or_straight("no_reliable_side")

        # Even with both sides DETECTED + parallel curvatures, a "fresh"
        # corridor must not bend until purple memory has validated lane
        # geometry. This catches the warmup_disagree_lane_area case
        # where purple_lane_width_valid is False.
        if not purple_width_trusted:
            return self._held_or_straight("purple_width_untrusted")

        a, b, src = proposal
        if abs(a) > self.max_curvature_a or abs(b) > self.max_slope_b:
            return self._held_or_straight("out_of_bounds")

        # Top-row displacement gate after DC-shifting to ego_x at anchor_y:
        # x_top - ego_x = a*(top² - anchor²) + b*(top - anchor).
        top_disp = (a * (top_y * top_y - anchor_y * anchor_y)
                    + b * (top_y - anchor_y))
        if abs(top_disp) > self.max_top_disp_px:
            return self._held_or_straight("excessive_top_disp")

        if self._last is not None:
            la, lb = self._last
            if (abs(a - la) > self.max_da_per_frame
                    or abs(b - lb) > self.max_db_per_frame):
                return self._held_or_straight("frame_jump")
            a = self.ema_alpha * a + (1.0 - self.ema_alpha) * la
            b = self.ema_alpha * b + (1.0 - self.ema_alpha) * lb

        self._last = (float(a), float(b))
        self._held_frames = 0
        self._last_reason = f"ok:{src}"
        return self._last, src, self._last_reason

    def _proposal(
        self,
        l_coeffs: tuple[float, float, float] | None,
        r_coeffs: tuple[float, float, float] | None,
        l_ok: bool, r_ok: bool,
    ) -> tuple[float, float, str] | None:
        if l_ok and r_ok and l_coeffs is not None and r_coeffs is not None:
            la, lb, _ = l_coeffs
            ra, rb, _ = r_coeffs
            # Real-lane curvatures should agree. When they do, average
            # both coefficients — the perspective-induced slopes b_L and
            # b_R cancel on a centered straight road and the residual
            # is the centerline's true bend.
            if abs(float(la) - float(ra)) <= self.parallel_da:
                return (
                    (float(la) + float(ra)) * 0.5,
                    (float(lb) + float(rb)) * 0.5,
                    "both",
                )
            # Curvature mismatch — neither side is trustworthy as the
            # ego shape source. Fall back rather than picking one and
            # risking a slanted corridor.
            return None
        # Single-side fallback: keep only the curvature hint; drop the
        # perspective slope b that would otherwise pull the corridor
        # sideways. The downstream top-disp gate further suppresses
        # anything but very small curvatures.
        if l_ok and l_coeffs is not None:
            return (float(l_coeffs[0]), 0.0, "left_curvature_only")
        if r_ok and r_coeffs is not None:
            return (float(r_coeffs[0]), 0.0, "right_curvature_only")
        return None

    def _held_or_straight(
        self, reason: str,
    ) -> tuple[tuple[float, float], str, str]:
        if (self._last is not None
                and self._held_frames < self.max_hold_frames):
            self._held_frames += 1
            self._last_reason = f"held({reason})"
            return self._last, "held", self._last_reason
        self._last = None
        self._held_frames = 0
        self._last_reason = f"straight({reason})"
        return (0.0, 0.0), "straight", self._last_reason
