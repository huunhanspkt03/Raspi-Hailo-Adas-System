"""Robust polynomial fit + post-fit geometry sanity for lane boundaries.

Splits per-side polynomial fitting and the cross-/width sanity check out of
``lane_boundaries`` so each module stays small and testable. Pure functions
that consume pixel arrays from the component pipeline.

Two guards keep the fit honest when dotted-line samples are sparse and
spread vertically (the user-reported "purple bends inward toward
lane_area" case):

* Low vertical coverage → force a linear (degree-1) fit so a couple of
  far-apart dotted clusters cannot drive a quadratic that bows between
  them.
* Quadratic curvature cap → if a degree-2 fit returns ``|a|`` above
  ``DEFAULT_MAX_CURVATURE``, refit linearly. ``DEFAULT_MAX_CURVATURE``
  is generous enough for realistic curved lanes (the existing curved
  test uses ``a ≈ 1e-3``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATUS_OK, STATUS_WEAK, STATUS_EMPTY = "OK", "WEAK", "EMPTY"
REJECT_GEOMETRY_CROSSING = "crossing"
REJECT_GEOMETRY_WIDTH = "width_oob"

DEFAULT_MAX_CURVATURE = 5.0e-3
DEFAULT_LOW_COVERAGE_THRESHOLD = 0.30


@dataclass(slots=True)
class SideFit:
    coeffs: tuple[float, float, float] | None
    pts_img: np.ndarray
    confidence: float
    n_pixels: int
    status: str


def _empty_pts() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def _weak(n: int, status: str = STATUS_WEAK) -> SideFit:
    return SideFit(None, _empty_pts(), 0.0, n, status)


def _fit_poly_capped(
    y_f: np.ndarray, x_f: np.ndarray, *,
    force_linear: bool, max_curvature: float,
) -> np.ndarray:
    """Fit ``x = a*y^2 + b*y + c`` with degree-1 fallback.

    Returns a 3-element ``[a, b, c]`` array (linear fits use ``a=0``).
    Falls back to linear when ``force_linear`` is True or when a degree-2
    fit yields ``|a| > max_curvature``.
    """
    if force_linear:
        b, c = np.polyfit(y_f, x_f, 1)
        return np.array([0.0, float(b), float(c)])
    coeffs = np.polyfit(y_f, x_f, 2)
    if abs(float(coeffs[0])) > float(max_curvature):
        b, c = np.polyfit(y_f, x_f, 1)
        return np.array([0.0, float(b), float(c)])
    return coeffs


def fit_side(
    ys: np.ndarray, xs: np.ndarray, *,
    frame_h: int, frame_w: int, top_cut: int,
    min_pixels: int, residual_trim_frac: float, max_residual_px: float,
    num_points: int, top_frac: float,
    max_curvature: float = DEFAULT_MAX_CURVATURE,
    low_coverage_threshold: float = DEFAULT_LOW_COVERAGE_THRESHOLD,
) -> SideFit:
    """Robust polynomial fit with curvature cap and one residual-trim pass."""
    if ys.size < min_pixels:
        return _weak(int(ys.size), STATUS_EMPTY if ys.size == 0 else STATUS_WEAK)
    y_f = ys.astype(np.float64)
    x_f = xs.astype(np.float64)
    total_rows = max(1, frame_h - top_cut)
    coverage = float(np.unique(ys).size) / float(total_rows)
    force_linear = coverage < float(low_coverage_threshold)
    try:
        coeffs = _fit_poly_capped(y_f, x_f, force_linear=force_linear,
                                  max_curvature=max_curvature)
    except (np.linalg.LinAlgError, ValueError):
        return _weak(int(ys.size))
    residuals = np.abs(np.polyval(coeffs, y_f) - x_f)
    rows_used = ys.size
    if residuals.size > min_pixels and residual_trim_frac > 0.0:
        trim = max(1, int(round(residuals.size * residual_trim_frac)))
        keep = np.argsort(residuals)[:-trim]
        if keep.size >= min_pixels:
            try:
                coeffs = _fit_poly_capped(
                    y_f[keep], x_f[keep],
                    force_linear=force_linear, max_curvature=max_curvature,
                )
                residuals = np.abs(np.polyval(coeffs, y_f[keep]) - x_f[keep])
                rows_used = keep.size
            except (np.linalg.LinAlgError, ValueError):
                pass
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    if rms > max_residual_px:
        return _weak(int(ys.size))
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    density = float(rows_used) / max(1.0, float(frame_w))
    fit_quality = float(np.exp(-rms / max(1.0, 0.04 * frame_w)))
    confidence = float(np.clip(0.45 * coverage + 0.35 * density + 0.20 * fit_quality, 0.0, 1.0))
    sy = np.linspace(max(0.0, frame_h * top_frac), float(frame_h - 1),
                     num=num_points, dtype=np.float32)
    sx = (a * sy * sy + b * sy + c).astype(np.float32)
    return SideFit((a, b, c), np.stack([sx, sy], axis=1).astype(np.float32),
                   confidence, int(ys.size), STATUS_OK)


def enforce_geometry(
    left: SideFit, right: SideFit,
    *, min_bottom_width: float, max_bottom_width: float,
) -> tuple[SideFit, SideFit, str | None]:
    """Drop the worse side if the two purple curves cross or width is OOB.

    Uses already-sampled ``pts_img`` so the geometry check shares the y-grid
    with downstream consumers.
    """
    if left.coeffs is None or right.coeffs is None:
        return left, right, None
    lx, rx = left.pts_img[:, 0], right.pts_img[:, 0]
    n = min(lx.size, rx.size)
    crossing = bool(np.any(lx[:n] >= rx[:n]))
    bottom_width = float(rx[n - 1] - lx[n - 1])
    bad_width = (bottom_width < min_bottom_width) or (bottom_width > max_bottom_width)
    if not crossing and not bad_width:
        return left, right, None
    reason = REJECT_GEOMETRY_CROSSING if crossing else REJECT_GEOMETRY_WIDTH
    if left.confidence < right.confidence:
        return _weak(left.n_pixels), right, reason
    return left, _weak(right.n_pixels), reason
