"""Per-frame update body for ``LaneBoundaryTracker``."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from features.lane_boundaries import fit_lane_boundaries
from features.lane_boundary_tracker_contract import (
    aggregate_lane_state, boundary_confidence,
)
from features.lane_boundary_tracker_geom import (
    lane_area_width_profile, lane_width_diagnostic, sample_polyline,
)
from features.lane_boundary_tracker_memory import (
    MemoryUpdateConfig, evaluate_memory_update,
)
from features.lane_boundary_tracker_resolve import (
    ResolveConfig, ResolvedSide, resolve_side,
)
from features.lane_boundary_tracker_validate import (
    ValidationConfig, merge_validation_diagnostics, validate_pair,
)

# Render-source tag attached to each side in the tracker output. Missing-side
# inference is disabled, so only DETECTED / HELD / LOST are emitted.
RENDER_SRC_DETECTED = "DETECTED"
RENDER_SRC_HELD = "HELD"
RENDER_SRC_LOST = "LOST"

if TYPE_CHECKING:
    from features.lane_boundary_tracker import LaneBoundaryTracker


def _make_validate_config(t: "LaneBoundaryTracker") -> ValidationConfig:
    return ValidationConfig(
        top_frac=t.top_frac, num_points=t.num_points,
        max_frame_jump_px=t.max_frame_jump_px,
        max_width_error_px=t.max_width_error_px,
        pinch_top_width_frac=t.pinch_top_width_frac,
        max_opposite_memory_error_px=t.max_opposite_memory_error_px,
        weak_support_pixels=t.weak_support_pixels,
        min_strong_support_pixels=t.min_strong_support_pixels,
        max_parallel_curvature_delta=t.max_parallel_curvature_delta,
        max_parallel_slope_delta=t.max_parallel_slope_delta,
        max_expected_from_opposite_px=t.max_expected_from_opposite_px,
    )


def _render_src_from_resolved(r: ResolvedSide) -> str:
    return {"DETECTED": RENDER_SRC_DETECTED,
            "HELD": RENDER_SRC_HELD,
            "LOST": RENDER_SRC_LOST}.get(r.source, RENDER_SRC_LOST)


def _pts_for_side(
    r: ResolvedSide, frame_h: int, top_frac: float, num_points: int,
) -> np.ndarray:
    """Sample the resolved side from its own coeffs, if available."""
    return sample_polyline(r.coeffs, frame_h,
                           top_frac=top_frac, num_points=num_points)


def run_tracker_update(
    t: "LaneBoundaryTracker", semantics, lane_area, frame_h, frame_w, **fit_kwargs,
) -> dict:
    if "side_split_x" not in fit_kwargs and "prev_side_split_x" not in fit_kwargs:
        prev = t._previous_side_split_x(frame_h)
        if prev is not None:
            fit_kwargs["prev_side_split_x"] = prev
    t._frame_id += 1
    fit = fit_lane_boundaries(semantics, lane_area, frame_h, frame_w, **fit_kwargs)
    validate_cfg = _make_validate_config(t)
    # Per-row lane_area-edges width profile. It still feeds validation
    # gates, but no longer synthesizes a missing side.
    la_width_profile = lane_area_width_profile(
        fit, frame_h=frame_h, top_frac=t.top_frac,
        num_points=t.num_points, min_width_px=t.min_inference_width_px,
    )
    la_cache_updated = False
    la_cache_reject = "inference_disabled"
    l_validation, r_validation = validate_pair(
        left_candidate=fit["left_coeffs"],
        right_candidate=fit["right_coeffs"],
        left_previous=t._left.coeffs,
        right_previous=t._right.coeffs,
        left_n_pixels=int(fit.get("n_left") or 0),
        right_n_pixels=int(fit.get("n_right") or 0),
        width_memory=t._width_memory, frame_h=frame_h,
        config=validate_cfg,
        lane_area_width_profile=la_width_profile,
    )
    l_cand = fit["left_coeffs"] if l_validation.accepted else None
    r_cand = fit["right_coeffs"] if r_validation.accepted else None
    l_detected = t._try_detect(t._left, l_cand,
                               fit["left_confidence"], fit["n_left"])
    r_detected = t._try_detect(t._right, r_cand,
                               fit["right_confidence"], fit["n_right"])
    # Refresh purple width memory only when BOTH sides are clean DETECTED.
    lw_px, lw_valid, lw_src = lane_width_diagnostic(
        fit, frame_h=frame_h, top_frac=t.top_frac,
        num_points=t.num_points, min_width_px=t.min_inference_width_px,
    )
    memory_update_allowed, memory_update_reject_reason = evaluate_memory_update(
        l_detected=l_detected, r_detected=r_detected,
        l_validation=l_validation, r_validation=r_validation,
        frame_h=frame_h,
        left_coeffs=t._left.coeffs, right_coeffs=t._right.coeffs,
        width_memory=t._width_memory,
        lane_area_width_px=lw_px, lane_area_width_valid=lw_valid,
        config=MemoryUpdateConfig(
            top_frac=t.top_frac, num_points=t.num_points,
            warmup_lane_area_tolerance_px=t.memory_warmup_lane_area_tolerance_px,
        ),
    )
    if memory_update_allowed:
        t._width_memory.update(t._left.coeffs, t._right.coeffs, frame_h)
    resolve_cfg = ResolveConfig(
        top_frac=t.top_frac, num_points=t.num_points,
        min_inference_width_px=t.min_inference_width_px,
        max_inferred_streak=t.max_inferred_streak,
        held_frame_limit=t.held_frame_limit,
    )
    learner = t._lane_area_width_memory
    l_resolved = resolve_side(
        t._left, "left", was_detected=l_detected,
        other_state=t._right, other_detected=r_detected,
        fit=fit, frame_h=frame_h,
        width_memory=t._width_memory,
        width_x_learner=learner,
        config=resolve_cfg,
    )
    r_resolved = resolve_side(
        t._right, "right", was_detected=r_detected,
        other_state=t._left, other_detected=l_detected,
        fit=fit, frame_h=frame_h,
        width_memory=t._width_memory,
        width_x_learner=learner,
        config=resolve_cfg,
    )
    wm_valid = t._width_memory.is_valid(frame_h)
    # Width-x profile debug diagnostics. Surfaced on every frame so the
    # metrics consumer does not have to handle missing keys, but inference
    # from this profile is disabled.
    wx_ready = learner.is_ready()
    wx_skip_reason = "inference_disabled"
    out = {
        **fit,
        "left_coeffs": l_resolved.coeffs, "right_coeffs": r_resolved.coeffs,
        "left_pts_img": _pts_for_side(
            l_resolved, frame_h, t.top_frac, t.num_points),
        "right_pts_img": _pts_for_side(
            r_resolved, frame_h, t.top_frac, t.num_points),
        "left_source": l_resolved.source, "right_source": r_resolved.source,
        "left_inference_kind": l_resolved.inference_kind,
        "right_inference_kind": r_resolved.inference_kind,
        "left_render_src": _render_src_from_resolved(l_resolved),
        "right_render_src": _render_src_from_resolved(r_resolved),
        "lane_state": aggregate_lane_state(l_resolved.source, r_resolved.source),
        "left_confidence_final": l_resolved.confidence,
        "right_confidence_final": r_resolved.confidence,
        "boundary_confidence": boundary_confidence(
            l_resolved.confidence, r_resolved.confidence),
        "lane_width_px": lw_px, "lane_width_valid": lw_valid,
        "lane_width_source": lw_src,
        # Explicit lane_area-only width aliases so consumers that only
        # care about lane_area-derived width don't have to know the
        # ``lane_width_source == "lane_area_edges"`` convention.
        "lane_area_width_px": lw_px if lw_valid else None,
        "lane_area_width_valid": bool(lw_valid),
        "purple_lane_width_valid": wm_valid,
        "purple_lane_width_px": (
            t._width_memory.median_px() if wm_valid else None),
        "purple_lane_width_source": "purple_memory" if wm_valid else "unset",
        "purple_memory_update_allowed": bool(memory_update_allowed),
        "purple_memory_update_reject_reason": memory_update_reject_reason,
        "lane_area_width_profile_px": (
            learner.profile.copy() if learner.is_valid() else None),
        "lane_area_width_profile_valid": bool(learner.is_valid()),
        "lane_area_width_profile_update_allowed": bool(la_cache_updated),
        "lane_area_width_profile_reject_reason": la_cache_reject,
        # Width-x learner diagnostics for HUD / metrics / replay logs.
        "width_profile_ready": bool(wx_ready),
        "width_profile_age": int(learner.age),
        "width_profile_learning_frames": int(learner.learning_frames),
        "width_profile_median_px": learner.median_px(),
        "width_profile_min_px": learner.min_px(),
        "width_profile_max_px": learner.max_px(),
        "width_profile_skip_reason": wx_skip_reason,
        "frame_id": int(t._frame_id),
    }
    for prefix, state in (("left", t._left), ("right", t._right)):
        out[f"{prefix}_raw_missing_streak"] = state.raw_missing_streak
        out[f"{prefix}_inferred_streak"] = state.inferred_streak
        out[f"{prefix}_held_streak"] = state.held_streak
        out[f"{prefix}_output_missing_streak"] = state.output_missing_streak
    merge_validation_diagnostics(out, l_validation, r_validation)
    return out
