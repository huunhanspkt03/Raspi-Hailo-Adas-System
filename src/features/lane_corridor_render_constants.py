"""Defaults and labels shared by the render-corridor builder."""
from __future__ import annotations

from features.ego_lane_geometry import LANE_WIDTH_M as _EGO_LANE_WIDTH_M

DEFAULT_ROI_TOP_FRAC = 0.55          # fixed lower ROI: [H * top, H - 1]
DEFAULT_ROI_BOTTOM_FRAC = 1.0
DEFAULT_NUM_POINTS = 24
# Real-world ego-vehicle width in meters. Reuses the existing project
# constant from ``ego_lane_geometry`` so all overlays speak in the same
# physical units. Override per-call if a vehicle-specific value exists.
DEFAULT_EGO_VEHICLE_WIDTH_M = float(_EGO_LANE_WIDTH_M)  # = 1.8 m
# Assumed real-world width of the visible road lane in meters. 3.5 m is
# the typical highway/urban lane in US/EU and is the working value when
# no calibration is available; lane-width-based scale calibration is
# only as accurate as this assumption.
DEFAULT_ASSUMED_LANE_WIDTH_M = 3.5
# Sane bounds for the ratio ``ego_width_m / lane_width_m``. A passenger
# car is typically 0.45-0.55 of a 3.5 m lane; clamping prevents a noisy
# purple side or an off-spec lane assumption from blowing up the
# corridor.
DEFAULT_VEHICLE_LANE_WIDTH_RATIO_MIN = 0.30
DEFAULT_VEHICLE_LANE_WIDTH_RATIO_MAX = 0.65
# Fixed ego corridor half-width profile as a fraction of frame width.
# Yellow/cam lines are symmetric around the red centerline: x = cx +/- w(y).
DEFAULT_FIXED_YELLOW_HALF_WIDTH_FAR_FRAC = 0.045
DEFAULT_FIXED_YELLOW_HALF_WIDTH_NEAR_FRAC = 0.24
# Legacy floor retained as a minimum yellow gap.
DEFAULT_YELLOW_PURPLE_MARGIN_PX = 6.0
# Rolling-mean window across ROI rows to keep yellow width stable.
DEFAULT_TARGET_SMOOTH_TAPS = 3
DEFAULT_MIN_INNER_GAP_PX = 100.0     # min yellow_right - yellow_left at any sampled y
DEFAULT_MIN_LANE_WIDTH_PX = 40.0     # below this, lane is too narrow to render
# Upper bound on a provisional lane_area-derived width before we treat the
# value as nonsense. Lane widths approaching frame_w are usually a leaked /
# split lane_area mask, not a real lane.
DEFAULT_MAX_LANE_WIDTH_FRAC_W = 0.85

# Render-corridor source labels surfaced in the output dict.
RENDER_SRC_PURPLE = "purple"
# ``provisional`` is no longer produced (lane-area-bootstrap synthesis
# was removed), but this label is still emitted on legacy code paths
# and consumed by replay-log parsers.
RENDER_SRC_LANE_AREA_BOOTSTRAP = "lane_area_bootstrap"
