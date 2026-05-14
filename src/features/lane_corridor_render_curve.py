"""Apply purple curvature (only) to the red centerline + yellow corridor.

Spec: red and the two yellow lines stay vertical at frame center even
when the detected purple is a tilted straight line. Only sharp purple
*curvature* bends red; gentle curves and tilts are ignored.

Mechanics:

* Red is anchored at ``ego_x`` at the bottom ROI row **and** has zero
  slope there, so it never tilts away from vertical at the camera.
* The curve form is ``red_x(y) = aC * (y - anchor_y)^2 + ego_x`` -
  purple's slope ``bC`` is intentionally dropped.
* A curvature deadband (``curvature_deadband_a``) keeps red vertical
  when ``|aC|`` is small (slight purple curve).
* The existing max-curvature, top-displacement, and frame-delta gates
  still reject runaway values; the angle / slope-b gates are gone
  because red no longer inherits purple's tilt.

Yellow consumes red's per-row ``dx/dy`` via ``red_dxdy_per_y`` and is
offset along red's normal. So when red is vertical, yellow is a
symmetric vertical offset; when red curves, yellow follows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from features.lane_boundary_tracker_contract import SOURCE_DETECTED, SOURCE_LOST

__all__ = (
    "DEFAULT_MAX_CURVATURE_A",
    "DEFAULT_CURVATURE_DEADBAND_A",
    "DEFAULT_CURVATURE_DAMP_FACTOR",
    "DEFAULT_CURVATURE_EMA_ALPHA",
    "DEFAULT_MAX_TOP_DISP_FRAC_W",
    "CorridorCurveState",
    "compute_red_xs_from_purple",
    "red_dxdy_per_y",
)

# Hard reject above this magnitude — treat as noise.
DEFAULT_MAX_CURVATURE_A = 5e-3
# Damp purple's curvature before applying it to red. Half-strength keeps
# the visual subtler than the underlying purple fit, which absorbs some
# fitting noise and over-fit curvature.
DEFAULT_CURVATURE_DAMP_FACTOR = 0.5
# Temporal EMA on the smoothed curvature. 0 = freeze, 1 = no smoothing.
# 0.25 settles in ~10 frames (~0.3s @ 30 fps), enough to follow lane
# changes without per-frame jitter.
DEFAULT_CURVATURE_EMA_ALPHA = 0.25
# Pure noise floor — below this damped magnitude red is treated as
# vertical. Set low so slight real curves still show.
DEFAULT_CURVATURE_DEADBAND_A = 1e-4
# Clamp top displacement to this fraction of frame_w instead of falling
# back to straight: the curve still shows, it just stops growing.
DEFAULT_MAX_TOP_DISP_FRAC_W = 0.35


@dataclass(slots=True)
class CorridorCurveState:
    """Last accepted curvature for the frame-to-frame delta gate."""
    prev_aC: float | None = None

    def reset(self) -> None:
        self.prev_aC = None


def _reliable_side(src: str, kind: str | None) -> bool:
    _ = kind
    if src == SOURCE_LOST:
        return False
    # Shape may come only from model-detected purple line classes
    # (solid_line / dotted_line). lane_area-derived, inferred, and held
    # sides are allowed to help rendering width elsewhere, but never
    # determine red/corridor curvature.
    return src == SOURCE_DETECTED


def _reliable_a(track: dict | None) -> float | None:
    """Average the curvature 'a' across reliable purple sides.

    The slope 'b' is intentionally NOT returned. Red must stay vertical
    at ego_x regardless of how tilted the detected purple is; only the
    quadratic curvature is allowed to bend the rendered centerline.

    Returns ``None`` when no reliable side has coefficients.
    """
    if not isinstance(track, dict):
        return None
    l_src = track.get("left_source", SOURCE_LOST)
    r_src = track.get("right_source", SOURCE_LOST)
    l_kind = track.get("left_inference_kind")
    r_kind = track.get("right_inference_kind")
    l_coeffs = track.get("left_coeffs")
    r_coeffs = track.get("right_coeffs")
    l_ok = _reliable_side(l_src, l_kind) and l_coeffs is not None
    r_ok = _reliable_side(r_src, r_kind) and r_coeffs is not None
    if l_ok and r_ok:
        return float(0.5 * (l_coeffs[0] + r_coeffs[0]))
    if l_ok:
        return float(l_coeffs[0])
    if r_ok:
        return float(r_coeffs[0])
    return None


def _straight(n: int, ego_x: float) -> np.ndarray:
    return np.full(int(n), float(ego_x), dtype=np.float32)


def compute_red_xs_from_purple(
    track: dict | None,
    ys: np.ndarray,
    frame_w: int,
    *,
    state: CorridorCurveState | None = None,
    provisional: bool = False,
    max_curvature_a: float = DEFAULT_MAX_CURVATURE_A,
    curvature_damp_factor: float = DEFAULT_CURVATURE_DAMP_FACTOR,
    curvature_ema_alpha: float = DEFAULT_CURVATURE_EMA_ALPHA,
    curvature_deadband_a: float = DEFAULT_CURVATURE_DEADBAND_A,
    max_top_disp_frac_w: float = DEFAULT_MAX_TOP_DISP_FRAC_W,
) -> tuple[np.ndarray, dict]:
    """Curve red from purple's ``a`` only — damped, smoothed, top-clamped.

    Shape: ``red_x(y) = aC_eff * (y - anchor_y)^2 + ego_x``. Red is
    exactly ``ego_x`` and exactly vertical at the bottom ROI row.

    Pipeline applied to the raw ``a`` from ``_reliable_a``:

      1. **Hard reject** if ``|a| > max_curvature_a`` (treated as 0).
      2. **Damp**: multiply by ``curvature_damp_factor`` so red bends
         less aggressively than purple.
      3. **Deadband**: damped values below ``curvature_deadband_a``
         collapse to 0 (noise floor).
      4. **Temporal EMA**: ``smoothed = alpha * target + (1 - alpha) *
         prev_smoothed``, stored in ``state``.
      5. **Top-displacement clamp**: shrink ``|aC_eff|`` so the curve
         at the top of the ROI stays inside
         ``max_top_disp_frac_w * frame_w``.

    When purple is unreliable or provisional, the smoothing target is
    0 — red decays toward straight rather than snapping back. Returns
    ``(red_xs, diagnostics)``; ``state.prev_aC`` is always updated to
    the smoothed value (even when 0).
    """
    ego_x = 0.5 * float(frame_w)
    n = int(ys.shape[0])
    anchor_y = float(ys[-1]) if n > 0 else 0.0
    diag: dict = {
        "corridor_curve_applied": False,
        "corridor_curve_aC": None,
        "corridor_curve_anchor_y": anchor_y,
        "corridor_curve_top_disp_px": None,
        "corridor_curve_a_raw": None,
        "corridor_curve_a_damped": None,
        "corridor_curve_a_smoothed": None,
        "corridor_curve_top_disp_clamped": False,
        "corridor_curve_reject_reason": None,
    }
    if n == 0:
        return np.zeros((0,), dtype=np.float32), diag

    # Step 1-2: pick the raw curvature and decide the EMA target.
    a_raw = _reliable_a(track)
    target_a = 0.0
    if provisional:
        diag["corridor_curve_reject_reason"] = "provisional_source"
    elif a_raw is None:
        diag["corridor_curve_reject_reason"] = "no_reliable_purple"
    elif abs(float(a_raw)) > float(max_curvature_a):
        diag["corridor_curve_a_raw"] = float(a_raw)
        diag["corridor_curve_reject_reason"] = "curvature_too_large"
    else:
        diag["corridor_curve_a_raw"] = float(a_raw)
        damped = float(a_raw) * float(curvature_damp_factor)
        diag["corridor_curve_a_damped"] = damped
        if abs(damped) < float(curvature_deadband_a):
            diag["corridor_curve_reject_reason"] = "below_deadband"
        else:
            target_a = damped

    # Step 4: temporal EMA — always run so the curve decays smoothly
    # when purple becomes unreliable instead of snapping to straight.
    prev = float(state.prev_aC) if (state is not None
                                    and state.prev_aC is not None) else 0.0
    alpha = float(curvature_ema_alpha)
    smoothed = alpha * float(target_a) + (1.0 - alpha) * prev
    diag["corridor_curve_a_smoothed"] = smoothed

    # Step 5: top-displacement clamp. Shrink smoothed so the corner
    # row stays inside the allowed band; the curve still renders.
    top_y_offset = abs(anchor_y - float(ys[0]))
    max_top_disp_px = float(max_top_disp_frac_w) * float(frame_w)
    if top_y_offset > 1.0 and max_top_disp_px > 0.0:
        max_a_by_top = max_top_disp_px / (top_y_offset * top_y_offset)
        if abs(smoothed) > max_a_by_top:
            sign = 1.0 if smoothed >= 0.0 else -1.0
            smoothed = sign * max_a_by_top
            diag["corridor_curve_top_disp_clamped"] = True

    if state is not None:
        state.prev_aC = float(smoothed)

    # If the smoothed value is below the deadband, render straight.
    if abs(smoothed) < float(curvature_deadband_a):
        diag["corridor_curve_aC"] = float(smoothed)
        diag["corridor_curve_top_disp_px"] = 0.0
        return _straight(n, ego_x), diag

    aC = float(smoothed)
    dy = ys.astype(np.float32) - np.float32(anchor_y)
    red_xs = (np.float32(aC) * dy * dy + np.float32(ego_x)).astype(np.float32)
    diag["corridor_curve_applied"] = True
    diag["corridor_curve_aC"] = aC
    diag["corridor_curve_top_disp_px"] = float(red_xs[0] - ego_x)
    return red_xs, diag


def red_dxdy_per_y(
    ys: np.ndarray, aC: float | None, anchor_y: float | None,
) -> np.ndarray:
    """Per-y first derivative of red ``= 2 * aC * (y - anchor_y)``.

    Slope is zero at ``anchor_y`` (red is vertical there). Returns zeros
    when the curve was not applied so yellow falls back to a straight
    symmetric offset.
    """
    if aC is None or anchor_y is None:
        return np.zeros_like(ys, dtype=np.float32)
    return (2.0 * float(aC)
            * (ys.astype(np.float32) - np.float32(anchor_y))
            ).astype(np.float32)
