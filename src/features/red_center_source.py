"""Center-source labels and small helpers for the red center fusion."""
from __future__ import annotations

from features.lane_boundary_tracker_contract import LANE_LOST, SOURCE_DETECTED
from features.lane_boundary_tracker_resolve import INFERENCE_KIND_LANE_AREA_BOOTSTRAP
from features.red_center_geom import ROAD_UNKNOWN, empty_pts

# Center-source values surfaced on the red-center result dict.
SRC_LANE_AREA_ONLY = "LANE_AREA_ONLY"
# Purple usable, no lane_area present.
SRC_PURPLE_ONLY = "PURPLE_ONLY"
# Both purple sides freshly DETECTED, lane_area also present - strongest.
SRC_PURPLE_PRIMARY = "PURPLE_PRIMARY"
# Both purple sides usable, but at least one is HELD rather than freshly
# detected. Legacy INFERRED labels are still recognized for old metrics, but
# the current tracker no longer synthesizes missing sides.
SRC_PURPLE_TRACKED = "PURPLE_TRACKED"
# Legacy: one side came from lane_area_bootstrap inference. Current tracker
# no longer emits this, but the label remains for old logs/tools.
SRC_LANE_AREA_ASSISTED = "LANE_AREA_ASSISTED"
SRC_HELD_PREVIOUS = "HELD_PREVIOUS"      # reserved for future temporal layer
SRC_LOST = "LOST"

PURPLE_GEOM_SOURCES = (
    SRC_PURPLE_PRIMARY, SRC_PURPLE_ONLY,
    SRC_PURPLE_TRACKED, SRC_LANE_AREA_ASSISTED,
)


def classify_purple_center_source(
    boundary_track: dict | None, *, have_lane_area: bool,
) -> str:
    """Pick the right ``SRC_*`` for a purple-usable frame.

    The label is derived from per-side ``*_source`` and ``*_inference_kind``
    so red center never claims ``PURPLE_PRIMARY`` when one side is actually
    a legacy lane_area_bootstrap proxy.
    """
    if not isinstance(boundary_track, dict):
        return SRC_PURPLE_PRIMARY if have_lane_area else SRC_PURPLE_ONLY
    l_src = boundary_track.get("left_source")
    r_src = boundary_track.get("right_source")
    l_kind = boundary_track.get("left_inference_kind")
    r_kind = boundary_track.get("right_inference_kind")
    has_bootstrap = (
        l_kind == INFERENCE_KIND_LANE_AREA_BOOTSTRAP
        or r_kind == INFERENCE_KIND_LANE_AREA_BOOTSTRAP
    )
    if has_bootstrap:
        return SRC_LANE_AREA_ASSISTED
    if l_src == SOURCE_DETECTED and r_src == SOURCE_DETECTED:
        return SRC_PURPLE_PRIMARY if have_lane_area else SRC_PURPLE_ONLY
    return SRC_PURPLE_TRACKED if have_lane_area else SRC_PURPLE_ONLY


def has_lane_area_bootstrap_side(boundary_track: dict | None) -> bool:
    if not isinstance(boundary_track, dict):
        return False
    return (
        boundary_track.get("left_inference_kind") == INFERENCE_KIND_LANE_AREA_BOOTSTRAP
        or boundary_track.get("right_inference_kind") == INFERENCE_KIND_LANE_AREA_BOOTSTRAP
    )


def empty_red_center_result(reason: str, lane_state: str = LANE_LOST) -> dict:
    return {
        "center_coeffs": None, "center_pts_img": empty_pts(),
        "center_confidence": 0.0, "center_source": SRC_LOST,
        "lane_area_center_pts_img": empty_pts(),
        "purple_center_pts_img": empty_pts(),
        "lane_area_confidence": 0.0, "purple_confidence": 0.0,
        "lane_state": lane_state, "road_type": ROAD_UNKNOWN,
        "valid": False, "reason": reason,
    }
