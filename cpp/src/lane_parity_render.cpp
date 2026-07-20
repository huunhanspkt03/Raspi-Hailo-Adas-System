// Phase-2/3 implementation of the Python parity render pipeline:
//   red center  -> ego-anchored path  -> purple+yellow render corridor.
//
// See lane_parity_render.hpp for the per-module mapping back to Python.

#include "adas/lane_parity_render.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace adas {

// ===========================================================================
// Enum -> string helpers
// ===========================================================================

const char* to_string(CenterSource s) {
    switch (s) {
        case CenterSource::Lost:           return "LOST";
        case CenterSource::LaneAreaOnly:   return "LANE_AREA_ONLY";
        case CenterSource::PurpleOnly:     return "PURPLE_ONLY";
        case CenterSource::PurplePrimary:  return "PURPLE_PRIMARY";
        case CenterSource::PurpleTracked:  return "PURPLE_TRACKED";
    }
    return "?";
}

const char* to_string(RoadType s) {
    switch (s) {
        case RoadType::Unknown:    return "UNKNOWN";
        case RoadType::Straight:   return "STRAIGHT";
        case RoadType::CurveLeft:  return "CURVE_LEFT";
        case RoadType::CurveRight: return "CURVE_RIGHT";
    }
    return "?";
}

// ===========================================================================
// Shared constants (matched to lane_corridor_render_constants.py)
// ===========================================================================

namespace {

constexpr float kRedCenterTopFrac    = 0.55f;
constexpr int   kRedCenterNumPoints  = 24;
constexpr int   kRedMinLaneAreaPixels = 600;
constexpr float kRedMaxJumpFracW     = 0.10f;
constexpr float kRoadCurveAThresh    = 5e-5f;

constexpr float kRenderRoiTopFrac    = 0.40f;
constexpr float kRenderRoiBottomFrac = 0.72f;
constexpr int   kRenderNumPoints     = 24;
// Yellow corridor half-width as a fraction of frame width, interpolated from
// far (top) to near (bottom). Narrowed to sit inside the ego lane, then nudged
// up slightly so an object right at the lane edge still counts as in-corridor.
// pick_front_object (object_detection.cpp) tests the bbox bottom-centre against
// this SAME polygon, so paint and behaviour stay consistent. The near end gets
// the bigger nudge (edge objects sit low); far stays conservative so distant
// adjacent-lane vehicles near the vanishing point are not swept in.
constexpr float kYellowHalfFarFrac   = 0.034f;   // 0.040 -> 0.033 -> 0.034 (+~3%)
constexpr float kYellowHalfNearFrac  = 0.190f;   // 0.250 -> 0.180 -> 0.190 (+~5.5%)
constexpr float kMinInnerGapPx       = 80.0f;    // was 100; far-end floor (half = 40px)
constexpr float kMinLaneWidthPx      = 40.0f;
constexpr int   kSmoothTaps          = 3;

// Ego-anchored path defaults (ego_anchored_path.py)
constexpr float kEgoLookaheadFrac     = 0.50f;
constexpr float kEgoMaxCurvatureA     = 8e-3f;
constexpr float kEgoMaxTopDispFracW   = 0.40f;
constexpr float kEgoDeadbandA         = 1e-5f;

// LaneAreaOnly clamps (lane_target_resolver.py)
constexpr float kLaneAreaMaxCurvatureA   = 2e-3f;
constexpr float kLaneAreaMaxTopDispFracW = 0.10f;
constexpr float kLaneAreaDeadbandA       = 5e-5f;

// LaneTargetState EMA params (lane_target_resolver.py)
constexpr float kTargetConfHold = 0.25f;
constexpr float kTargetEmaAlpha = 0.20f;

// CorridorCurveState (lane_corridor_render_curve.py)
constexpr float kCurveMaxCurvatureA   = 5e-3f;
constexpr float kCurveDampFactor      = 0.25f;
constexpr float kCurveEmaAlpha        = 0.25f;
constexpr float kCurveDeadbandA       = 2e-4f;
constexpr float kCurveMaxTopDispFracW = 0.06f;

// --- Helpers ---------------------------------------------------------------

inline float eval_q(float a, float b, float c, float y) {
    return a * y * y + b * y + c;
}

bool polyfit1(const std::vector<double>& ys, const std::vector<double>& xs,
              double* b, double* c) {
    if (ys.size() < 2) return false;
    double S0=0, S1=0, S2=0, Tx=0, Txy=0;
    for (std::size_t i = 0; i < ys.size(); ++i) {
        S0 += 1.0; S1 += ys[i]; S2 += ys[i] * ys[i];
        Tx += xs[i]; Txy += xs[i] * ys[i];
    }
    const double det = S2 * S0 - S1 * S1;
    if (std::fabs(det) < 1e-12) return false;
    *b = (Txy * S0 - Tx * S1) / det;
    *c = (Tx * S2 - Txy * S1) / det;
    return true;
}

bool polyfit2(const std::vector<double>& ys, const std::vector<double>& xs,
              double* a, double* b, double* c) {
    if (ys.size() < 3) return false;
    double S0=0,S1=0,S2=0,S3=0,S4=0,Tx=0,Txy=0,Txy2=0;
    for (std::size_t i = 0; i < ys.size(); ++i) {
        double y = ys[i], x = xs[i];
        double y2 = y * y;
        S0 += 1.0;   S1 += y;    S2 += y2;
        S3 += y2*y;  S4 += y2*y2;
        Tx += x;     Txy += x*y; Txy2 += x*y2;
    }
    double m[3][3] = {{S4,S3,S2},{S3,S2,S1},{S2,S1,S0}};
    double v[3]    = {Txy2, Txy, Tx};
    for (int i = 0; i < 3; ++i) {
        int piv = i;
        for (int k = i + 1; k < 3; ++k)
            if (std::fabs(m[k][i]) > std::fabs(m[piv][i])) piv = k;
        if (std::fabs(m[piv][i]) < 1e-12) return false;
        if (piv != i) { std::swap(m[piv], m[i]); std::swap(v[piv], v[i]); }
        const double inv = 1.0 / m[i][i];
        for (int j = 0; j < 3; ++j) m[i][j] *= inv;
        v[i] *= inv;
        for (int k = 0; k < 3; ++k) {
            if (k == i) continue;
            const double f = m[k][i];
            for (int j = 0; j < 3; ++j) m[k][j] -= f * m[i][j];
            v[k] -= f * v[i];
        }
    }
    *a = v[0]; *b = v[1]; *c = v[2];
    return true;
}

// Interpolate a polyline (sorted-by-y) at the given ys. Returns false when
// the polyline is empty or has fewer than 2 points; otherwise writes
// out_xs of length ys.size().
bool interp_polyline_at_ys(const std::vector<cv::Point2f>& pts,
                           const std::vector<float>& ys,
                           std::vector<float>* out_xs) {
    if (pts.size() < 2) return false;
    std::vector<cv::Point2f> sorted = pts;
    std::sort(sorted.begin(), sorted.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
    out_xs->resize(ys.size());
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < ys.size(); ++i) {
        const float y = ys[i];
        if (y <= sorted.front().y) { (*out_xs)[i] = sorted.front().x; continue; }
        if (y >= sorted.back().y)  { (*out_xs)[i] = sorted.back().x;  continue; }
        while (cursor + 1 < sorted.size() && sorted[cursor + 1].y < y) ++cursor;
        const auto& A = sorted[cursor];
        const auto& B = sorted[cursor + 1];
        const float t = (y - A.y) / std::max(1e-6f, (B.y - A.y));
        (*out_xs)[i] = A.x + t * (B.x - A.x);
    }
    return true;
}

bool both_purple_detected(const BoundaryTrackResult& track) {
    return track.left.source  == LaneSource::Detected &&
           track.right.source == LaneSource::Detected;
}

bool purple_sides_separable(const std::vector<float>& lx,
                            const std::vector<float>& rx,
                            float min_gap_px) {
    if (lx.empty() || lx.size() != rx.size()) return false;
    for (std::size_t i = 0; i < lx.size(); ++i) {
        if (rx[i] - lx[i] < min_gap_px) return false;
    }
    return true;
}

}  // namespace

// ===========================================================================
// compute_red_centerline (red_center_from_boundaries.py)
// ===========================================================================

// Lightweight lane_area row-centroid quadratic fit. C++ analogue of
// red_center_geom.lane_area_evidence (which itself wraps
// features/dynamic_corridor.compute_dynamic_corridor). We approximate the
// dynamic-corridor weighting with a uniform-weight quadratic fit over the
// per-row lane_area centroids in the ROI; this matches Python output in
// straight scenes.
namespace {

struct LaneAreaFitResult {
    bool valid{false};
    float a{0.0f};
    float b{0.0f};
    float c{0.0f};
    float confidence{0.0f};
    std::vector<float> centroid_xs;  // per ROI row (matches ys grid)
    std::vector<float> ys;
    // Per-row left/right lane_area edge, indexed by (row - top_cut), captured
    // in the SAME scan that computes the centroids so clip_to_fit_edges below
    // never re-scans the mask. -1 marks an empty row. top_cut is the absolute
    // row the arrays start at.
    int top_cut{0};
    std::vector<int> row_left;
    std::vector<int> row_right;
};

// `total_nonzero` is the caller's cv::countNonZero(lane_area) — passed in so
// the full-mask pass runs once per frame instead of twice.
LaneAreaFitResult fit_lane_area_centerline(const cv::Mat& lane_area,
                                           int frame_h, int frame_w,
                                           const std::vector<float>& sample_ys,
                                           int min_pixels, int total_nonzero) {
    LaneAreaFitResult res;
    if (lane_area.empty()) return res;
    if (total_nonzero < min_pixels) return res;
    const int top_cut = static_cast<int>(std::round(frame_h * kRedCenterTopFrac));
    const int h = lane_area.rows;
    std::vector<double> ys, xs;
    ys.reserve(h - top_cut); xs.reserve(h - top_cut);
    res.top_cut = top_cut;
    res.row_left.assign(std::max(0, h - top_cut), -1);
    res.row_right.assign(std::max(0, h - top_cut), -1);
    for (int y = top_cut; y < h; ++y) {
        const std::uint8_t* row = lane_area.ptr<std::uint8_t>(y);
        long long sum_x = 0, cnt = 0;
        int first = -1, last = -1;
        // Single scan: centroid (sum_x/cnt) AND left/right edge (first/last
        // nonzero x), so the downstream clip reuses these instead of a
        // second per-row pass.
        for (int x = 0; x < frame_w; ++x) {
            if (row[x]) {
                if (first < 0) first = x;
                last = x;
                sum_x += x; ++cnt;
            }
        }
        res.row_left[y - top_cut]  = first;
        res.row_right[y - top_cut] = last;
        if (cnt > 0) {
            ys.push_back(static_cast<double>(y));
            xs.push_back(static_cast<double>(sum_x) / static_cast<double>(cnt));
        }
    }
    if (static_cast<int>(ys.size()) < 8) return res;

    double a=0, b=0, c=0;
    bool ok2 = polyfit2(ys, xs, &a, &b, &c);
    if (!ok2) {
        a = 0; double bb=0, cc=0;
        if (!polyfit1(ys, xs, &bb, &cc)) return res;
        b = bb; c = cc;
    }
    res.valid = true;
    res.a = static_cast<float>(a);
    res.b = static_cast<float>(b);
    res.c = static_cast<float>(c);
    // Confidence: row coverage in the detection ROI.
    const float coverage = static_cast<float>(ys.size()) /
                           static_cast<float>(std::max(1, h - top_cut));
    res.confidence = std::clamp(coverage, 0.0f, 1.0f);

    // Sample at requested ys.
    res.ys = sample_ys;
    res.centroid_xs.resize(sample_ys.size());
    for (std::size_t i = 0; i < sample_ys.size(); ++i) {
        res.centroid_xs[i] = res.a * sample_ys[i] * sample_ys[i]
                           + res.b * sample_ys[i] + res.c;
    }
    return res;
}

// Per-row clip to lane_area columns, using the left/right edges captured by
// fit_lane_area_centerline's scan (no re-scan of the mask). Rows with no
// lane_area edge (or outside the captured ROI) are marked invalid.
void clip_to_fit_edges(std::vector<float>* xs,
                       const std::vector<float>& ys,
                       const LaneAreaFitResult& fit,
                       std::vector<bool>* valid_out) {
    valid_out->assign(ys.size(), true);
    const int n = static_cast<int>(fit.row_left.size());
    for (std::size_t i = 0; i < ys.size(); ++i) {
        const int idx = static_cast<int>(std::round(ys[i])) - fit.top_cut;
        if (idx < 0 || idx >= n || fit.row_left[idx] < 0) {
            (*valid_out)[i] = false;
            continue;
        }
        (*xs)[i] = std::clamp((*xs)[i],
                              static_cast<float>(fit.row_left[idx]),
                              static_cast<float>(fit.row_right[idx]));
    }
}

}  // namespace

RedCenter compute_red_centerline(const cv::Mat& lane_area,
                                 const BoundaryTrackResult& track,
                                 int frame_h, int frame_w) {
    RedCenter out;
    out.source = CenterSource::Lost;
    out.road_type = RoadType::Unknown;
    out.reason = "no evidence";

    const float y_top = std::max(0.0f, frame_h * kRedCenterTopFrac);
    const float y_bot = static_cast<float>(frame_h - 1);
    std::vector<float> ys(kRedCenterNumPoints);
    for (int i = 0; i < kRedCenterNumPoints; ++i) {
        const float t = static_cast<float>(i) /
                        static_cast<float>(kRedCenterNumPoints - 1);
        ys[i] = y_top + (y_bot - y_top) * t;
    }

    // Purple midpoint when BOTH sides freshly DETECTED + separable. Mirrors
    // red_center_geom.purple_midpoint (HELD/LOST sides ignored).
    std::vector<float> lx, rx, purp_xs;
    bool purple_usable = false;
    float purp_conf = 0.0f;
    if (both_purple_detected(track) &&
        interp_polyline_at_ys(track.left_pts_img, ys, &lx) &&
        interp_polyline_at_ys(track.right_pts_img, ys, &rx)) {
        purp_xs.resize(ys.size());
        for (std::size_t i = 0; i < ys.size(); ++i)
            purp_xs[i] = 0.5f * (lx[i] + rx[i]);
        if (purple_sides_separable(lx, rx, 1.0f)) {
            purple_usable = true;
            // src_weights[DETECTED]=1.0 in Python; 0.5*(left_conf+right_conf).
            purp_conf = 0.5f * (track.left.confidence_final +
                                track.right.confidence_final);
        }
        out.purple_center_pts_img.reserve(ys.size());
        for (std::size_t i = 0; i < ys.size(); ++i)
            out.purple_center_pts_img.emplace_back(purp_xs[i], ys[i]);
    }
    out.purple_confidence = purple_usable ? purp_conf : 0.0f;

    const int lane_area_nonzero =
        lane_area.empty() ? 0 : cv::countNonZero(lane_area);
    const bool have_lane_area = lane_area_nonzero >= kRedMinLaneAreaPixels;

    std::vector<float> fused;
    std::vector<bool>  valid_mask(ys.size(), true);
    int fit_deg = 1;
    LaneAreaFitResult area_fit;

    if (purple_usable) {
        fused = purp_xs;
        out.source = have_lane_area ? CenterSource::PurplePrimary
                                    : CenterSource::PurpleOnly;
        fit_deg = 2;
    } else if (have_lane_area) {
        area_fit = fit_lane_area_centerline(
            lane_area, frame_h, frame_w, ys, kRedMinLaneAreaPixels,
            lane_area_nonzero);
        if (!area_fit.valid) {
            out.reason = "lane_area unusable and purple LOST";
            return out;
        }
        out.lane_area_confidence = area_fit.confidence;
        out.lane_area_center_pts_img.reserve(ys.size());
        for (std::size_t i = 0; i < ys.size(); ++i)
            out.lane_area_center_pts_img.emplace_back(area_fit.centroid_xs[i], ys[i]);
        fused = area_fit.centroid_xs;
        out.source = CenterSource::LaneAreaOnly;
        fit_deg = 1;
        // Clip to lane_area columns using edges from the fit scan (no re-scan);
        // rows w/o lane_area mark invalid.
        clip_to_fit_edges(&fused, ys, area_fit, &valid_mask);
    } else {
        out.reason = "lane_area unusable and purple LOST";
        return out;
    }

    // Drop invalid rows.
    std::vector<double> ys_valid, xs_valid;
    ys_valid.reserve(ys.size()); xs_valid.reserve(ys.size());
    for (std::size_t i = 0; i < ys.size(); ++i) {
        if (valid_mask[i]) {
            ys_valid.push_back(ys[i]);
            xs_valid.push_back(fused[i]);
        }
    }
    if (ys_valid.size() < 6) {
        out.reason = "insufficient valid samples for fit";
        return out;
    }
    double a=0, b=0, c=0;
    if (fit_deg == 2) {
        if (!polyfit2(ys_valid, xs_valid, &a, &b, &c)) {
            out.reason = "polyfit failed"; return out;
        }
    } else {
        a = 0; double bb=0, cc=0;
        if (!polyfit1(ys_valid, xs_valid, &bb, &cc)) {
            out.reason = "polyfit failed"; return out;
        }
        b = bb; c = cc;
    }
    std::vector<float> final_xs(ys.size());
    for (std::size_t i = 0; i < ys.size(); ++i)
        final_xs[i] = static_cast<float>(a * ys[i] * ys[i] + b * ys[i] + c);

    // Clip the smoothed centerline strictly between the purple sides.
    // purple_sides_separable only guarantees a gap >= 1 px, so for rows
    // narrower than 2*eps the [lo, hi] interval would invert — std::clamp
    // with lo > hi is UB. Fall back to the row midpoint there.
    if (out.source == CenterSource::PurplePrimary ||
        out.source == CenterSource::PurpleOnly ||
        out.source == CenterSource::PurpleTracked) {
        constexpr float eps = 1.0f;
        for (std::size_t i = 0; i < ys.size(); ++i) {
            const float lo = lx[i] + eps;
            const float hi = rx[i] - eps;
            final_xs[i] = (lo <= hi) ? std::clamp(final_xs[i], lo, hi)
                                     : 0.5f * (lx[i] + rx[i]);
        }
    }

    // Jump penalty on max neighbour delta.
    float max_jump = 0.0f;
    for (std::size_t i = 1; i < final_xs.size(); ++i) {
        max_jump = std::max(max_jump, std::fabs(final_xs[i] - final_xs[i-1]));
    }
    const float jump_penalty = (max_jump > kRedMaxJumpFracW * frame_w) ? 0.5f : 1.0f;
    const float base_conf = purple_usable ? purp_conf : area_fit.confidence;
    out.center_confidence = std::clamp(base_conf * jump_penalty, 0.0f, 1.0f);

    out.poly_a = static_cast<float>(a);
    out.poly_b = static_cast<float>(b);
    out.poly_c = static_cast<float>(c);
    out.center_pts_img.reserve(ys.size());
    for (std::size_t i = 0; i < ys.size(); ++i)
        out.center_pts_img.emplace_back(final_xs[i], ys[i]);

    // road_type from a.
    if (a >  kRoadCurveAThresh) out.road_type = RoadType::CurveLeft;
    else if (a < -kRoadCurveAThresh) out.road_type = RoadType::CurveRight;
    else out.road_type = RoadType::Straight;

    out.valid = true;
    out.reason = "ok";
    return out;
}

// ===========================================================================
// Lookahead target + ego-anchored path (lane_target_resolver.py, ego_anchored_path.py)
// ===========================================================================

namespace {

float lookahead_y_for(int frame_h, float lookahead_frac) {
    const float fh = static_cast<float>(frame_h);
    return std::clamp(fh * lookahead_frac, 0.0f, fh - 1.0f);
}

// Confidence-scaled clamps. When the source is LaneAreaOnly and the
// lane_area is strong, allow ego-like curvature/displacement so the red
// path can bend toward the actual lane center; otherwise stay conservative.
// Below ~0.30 confidence we keep the original tight LaneAreaOnly clamps;
// above ~0.70 we use the same clamps the purple-primary path uses.
// `area_conf` is ignored for non-LaneAreaOnly sources.
void ego_path_clamps_for_source(CenterSource s,
                                float area_conf,
                                float* max_curvature_a,
                                float* deadband_a,
                                float* max_top_disp_frac_w) {
    if (s == CenterSource::LaneAreaOnly) {
        const float t = std::clamp((area_conf - 0.30f) / 0.40f, 0.0f, 1.0f);
        *max_curvature_a    = kLaneAreaMaxCurvatureA +
                              t * (kEgoMaxCurvatureA   - kLaneAreaMaxCurvatureA);
        *max_top_disp_frac_w = kLaneAreaMaxTopDispFracW +
                              t * (kEgoMaxTopDispFracW - kLaneAreaMaxTopDispFracW);
        *deadband_a         = kLaneAreaDeadbandA;
    } else {
        *max_curvature_a    = kEgoMaxCurvatureA;
        *deadband_a         = kEgoDeadbandA;
        *max_top_disp_frac_w = kEgoMaxTopDispFracW;
    }
}

float interp_polyline_at_y(const std::vector<cv::Point2f>& pts, float y) {
    if (pts.size() < 2) return std::numeric_limits<float>::quiet_NaN();
    std::vector<cv::Point2f> sorted = pts;
    std::sort(sorted.begin(), sorted.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
    if (y <= sorted.front().y) return sorted.front().x;
    if (y >= sorted.back().y)  return sorted.back().x;
    for (std::size_t i = 0; i + 1 < sorted.size(); ++i) {
        if (sorted[i+1].y >= y) {
            const float t = (y - sorted[i].y) /
                            std::max(1e-6f, (sorted[i+1].y - sorted[i].y));
            return sorted[i].x + t * (sorted[i+1].x - sorted[i].x);
        }
    }
    return sorted.back().x;
}

}  // namespace

LookaheadTarget resolve_lookahead_target(const RedCenter& red, int frame_h,
                                         int frame_w, LaneTargetState& state) {
    LookaheadTarget out;
    const float ego_x = 0.5f * static_cast<float>(frame_w);
    out.target_y = lookahead_y_for(frame_h, kEgoLookaheadFrac);
    ego_path_clamps_for_source(red.source,
                               red.lane_area_confidence,
                               &out.ego_max_curvature_a,
                               &out.ego_deadband_a,
                               &out.ego_max_top_disp_frac_w);
    out.target_x = state.has_prev ? state.prev_target_x : ego_x;
    if (!red.valid) return out;

    float raw = std::numeric_limits<float>::quiet_NaN();
    if (red.source == CenterSource::LaneAreaOnly) {
        raw = interp_polyline_at_y(red.lane_area_center_pts_img, out.target_y);
    } else if (red.source != CenterSource::Lost) {
        // Purple: use the resolved center polynomial first; fall back to
        // sampled polyline.
        raw = red.poly_a * out.target_y * out.target_y
            + red.poly_b * out.target_y + red.poly_c;
        if (!std::isfinite(raw)) raw = interp_polyline_at_y(red.center_pts_img, out.target_y);
    }
    if (!std::isfinite(raw)) return out;
    raw = std::clamp(raw, 0.0f, static_cast<float>(frame_w - 1));
    out.raw_target_x = raw;
    out.raw_present = true;

    if (red.center_confidence < kTargetConfHold) return out;

    const float smoothed = state.has_prev
        ? kTargetEmaAlpha * raw + (1.0f - kTargetEmaAlpha) * state.prev_target_x
        : raw;
    state.prev_target_x = smoothed;
    state.has_prev = true;
    out.target_x = smoothed;
    out.accepted = true;
    return out;
}

EgoPathResult build_ego_anchored_path(const std::vector<cv::Point2f>& target_pts,
                                      const std::vector<float>& ys,
                                      int frame_h, int frame_w,
                                      float max_curvature_a,
                                      float deadband_a,
                                      float max_top_disp_frac_w) {
    EgoPathResult out;
    const float ego_x = 0.5f * static_cast<float>(frame_w);
    out.lookahead_y = lookahead_y_for(frame_h, kEgoLookaheadFrac);
    out.control_x = ego_x;
    out.control_y = static_cast<int>(std::round(out.lookahead_y));
    if (ys.empty()) { out.reason = "empty_ys"; return out; }
    out.anchor_y = ys.back();
    out.xs.assign(ys.size(), ego_x);

    if (target_pts.size() < 2) { out.reason = "no_targets"; return out; }
    const float y_top = ys.front();
    double num = 0.0, denom = 0.0;
    for (const auto& p : target_pts) {
        const double dy = static_cast<double>(p.y) - out.anchor_y;
        const double dx = static_cast<double>(p.x) - ego_x;
        num   += dx * dy * dy;
        denom += dy * dy * dy * dy;
    }
    if (denom < 1e-3) { out.reason = "degenerate_denominator"; return out; }
    double a_raw = num / denom;
    // INTENTIONAL DIVERGENCE FROM PYTHON: ego_anchored_path.py bails out
    // and returns a=0 when |a_raw| exceeds max_curvature_a, which here
    // collapsed the whole path back to ego_x whenever the LaneAreaOnly
    // (tight) clamp triggered — leaving the red center vertical at frame
    // center even on frames where lane_area clearly shows a shifted
    // lane. We soft-cap |a| to the per-source maximum instead, so the
    // curve bends as far as the clamp permits while still respecting
    // the hard ceiling. Combined with the confidence-scaled LaneAreaOnly
    // clamps in ego_path_clamps_for_source, this lets a good lane_area
    // mask actually steer the red center.
    if (std::fabs(a_raw) > max_curvature_a) {
        a_raw = (a_raw > 0.0 ? max_curvature_a : -max_curvature_a);
        out.curvature_capped = true;
        out.reason = "curvature_capped";
    }
    double a = (std::fabs(a_raw) >= deadband_a) ? a_raw : 0.0;
    if (a == 0.0 && a_raw != 0.0 && !out.curvature_capped) {
        out.reason = "below_deadband";
    }

    if (a != 0.0) {
        const double max_disp = static_cast<double>(max_top_disp_frac_w) * frame_w;
        const double dy_top = static_cast<double>(y_top) - out.anchor_y;
        if (max_disp > 0.0 && dy_top * dy_top > 1.0) {
            const double max_a_by_top = max_disp / (dy_top * dy_top);
            if (std::fabs(a) > max_a_by_top) {
                a = (a > 0.0 ? max_a_by_top : -max_a_by_top);
                out.top_disp_clamped = true;
            }
        }
    }
    out.a = static_cast<float>(a);
    for (std::size_t i = 0; i < ys.size(); ++i) {
        const float dy = ys[i] - out.anchor_y;
        out.xs[i] = std::clamp(static_cast<float>(a) * dy * dy + ego_x,
                               0.0f, static_cast<float>(frame_w - 1));
    }
    const float dy_la = out.lookahead_y - out.anchor_y;
    out.control_x = std::clamp(static_cast<float>(a) * dy_la * dy_la + ego_x,
                               0.0f, static_cast<float>(frame_w - 1));
    out.applied = (a != 0.0);
    if (out.reason && std::string("no_targets") == out.reason) out.reason = "ok";
    return out;
}

// ===========================================================================
// Render corridor: build_render_corridor + yellow_corridor_from_center +
// compute_red_xs_from_purple (when ego anchor is unavailable).
// ===========================================================================

namespace {

void smooth_widths_inplace(std::vector<float>* w, int taps) {
    if (taps <= 1 || w->size() < 3) return;
    const int pad = taps / 2;
    std::vector<float> padded(w->size() + 2 * pad);
    for (int i = 0; i < pad; ++i) padded[i] = (*w)[0];
    for (std::size_t i = 0; i < w->size(); ++i) padded[pad + i] = (*w)[i];
    for (int i = 0; i < pad; ++i) padded[pad + w->size() + i] = w->back();
    const float invk = 1.0f / static_cast<float>(taps);
    std::vector<float> tmp(w->size());
    for (std::size_t i = 0; i < w->size(); ++i) {
        float s = 0.0f;
        for (int k = 0; k < taps; ++k) s += padded[i + k];
        tmp[i] = s * invk;
    }
    *w = std::move(tmp);
}

void yellow_corridor_from_center(const std::vector<float>& cxs, int frame_w,
                                 const std::vector<float>& red_dxdy,
                                 std::vector<float>* yl,
                                 std::vector<float>* yr) {
    const int n = static_cast<int>(cxs.size());
    yl->resize(n); yr->resize(n);
    const float far  = std::max(kYellowHalfFarFrac  * frame_w, 0.5f * kMinInnerGapPx);
    const float near = std::max(kYellowHalfNearFrac * frame_w, 0.5f * kMinInnerGapPx);
    std::vector<float> half(n);
    for (int i = 0; i < n; ++i) {
        const float t = n <= 1 ? 0.0f
                       : static_cast<float>(i) / static_cast<float>(n - 1);
        half[i] = far + (near - far) * t;
    }
    smooth_widths_inplace(&half, kSmoothTaps);

    const bool has_normal = (static_cast<int>(red_dxdy.size()) == n);
    for (int i = 0; i < n; ++i) {
        float offset = half[i];
        if (has_normal) {
            const float inv_norm = 1.0f /
                std::sqrt(1.0f + red_dxdy[i] * red_dxdy[i]);
            offset *= inv_norm;
        }
        (*yl)[i] = std::clamp(cxs[i] - offset, 0.0f,
                              static_cast<float>(frame_w - 1));
        (*yr)[i] = std::clamp(cxs[i] + offset, 0.0f,
                              static_cast<float>(frame_w - 1));
    }
}

// Purple-curvature fallback. Mirrors lane_corridor_render_curve.
void compute_red_xs_from_purple(const BoundaryTrackResult& track,
                                const std::vector<float>& ys,
                                int frame_w,
                                CorridorCurveState* state,
                                std::vector<float>* red_xs_out,
                                float* aC_out, float* anchor_y_out) {
    const float ego_x = 0.5f * static_cast<float>(frame_w);
    const int n = static_cast<int>(ys.size());
    red_xs_out->assign(n, ego_x);
    *aC_out = 0.0f;
    if (n == 0) return;
    const float anchor_y = ys.back();
    *anchor_y_out = anchor_y;

    // Pick reliable-side curvature(s): only SOURCE_DETECTED contributes.
    float a_raw = 0.0f;
    int  used = 0;
    if (track.left.source  == LaneSource::Detected && track.left.coeffs.valid) {
        a_raw += track.left.coeffs.a; ++used;
    }
    if (track.right.source == LaneSource::Detected && track.right.coeffs.valid) {
        a_raw += track.right.coeffs.a; ++used;
    }
    float target_a = 0.0f;
    if (used > 0) {
        a_raw /= static_cast<float>(used);
        if (std::fabs(a_raw) <= kCurveMaxCurvatureA) {
            const float damped = a_raw * kCurveDampFactor;
            if (std::fabs(damped) >= kCurveDeadbandA) target_a = damped;
        }
    }

    const float prev = (state && state->has_prev) ? state->prev_aC : 0.0f;
    float smoothed = kCurveEmaAlpha * target_a + (1.0f - kCurveEmaAlpha) * prev;

    const float top_y_offset = std::fabs(anchor_y - ys.front());
    const float max_top_disp = kCurveMaxTopDispFracW * frame_w;
    if (top_y_offset > 1.0f && max_top_disp > 0.0f) {
        const float max_a_by_top = max_top_disp / (top_y_offset * top_y_offset);
        if (std::fabs(smoothed) > max_a_by_top) {
            smoothed = (smoothed >= 0.0f ? max_a_by_top : -max_a_by_top);
        }
    }
    if (state) { state->prev_aC = smoothed; state->has_prev = true; }

    if (std::fabs(smoothed) < kCurveDeadbandA) { *aC_out = smoothed; return; }
    *aC_out = smoothed;
    for (int i = 0; i < n; ++i) {
        const float dy = ys[i] - anchor_y;
        (*red_xs_out)[i] = smoothed * dy * dy + ego_x;
    }
}

}  // namespace

RenderCorridor build_render_corridor(const BoundaryTrackResult& track,
                                     const RedCenter& red,
                                     int frame_h, int frame_w,
                                     LaneTargetState& target_state,
                                     CorridorCurveState& curve_state) {
    RenderCorridor out;
    out.lane_state = track.lane_state;
    // ROI: 45% -> 72% of frame_h, 24 sample rows.
    const float y_top = std::max(0.0f, frame_h * kRenderRoiTopFrac);
    const float y_bot = std::min(static_cast<float>(frame_h - 1),
                                  frame_h * kRenderRoiBottomFrac - 1.0f);
    if (y_bot <= y_top) { out.reason = "roi_collapsed"; return out; }
    out.ys.resize(kRenderNumPoints);
    for (int i = 0; i < kRenderNumPoints; ++i) {
        const float t = static_cast<float>(i) /
                        static_cast<float>(kRenderNumPoints - 1);
        out.ys[i] = y_top + (y_bot - y_top) * t;
    }

    // resolve_purple_xs: sample DETECTED sides on the ROI grid. We don't
    // synthesize anything — single-side stays one-sided.
    auto sample_side = [&](const PerSideResult& side,
                           std::vector<float>* xs) -> bool {
        if (side.source != LaneSource::Detected || !side.coeffs.valid)
            return false;
        xs->resize(out.ys.size());
        for (std::size_t i = 0; i < out.ys.size(); ++i) {
            (*xs)[i] = std::clamp(
                eval_q(side.coeffs.a, side.coeffs.b, side.coeffs.c, out.ys[i]),
                0.0f, static_cast<float>(frame_w - 1));
        }
        return true;
    };
    const bool have_left  = sample_side(track.left,  &out.purple_left_xs);
    const bool have_right = sample_side(track.right, &out.purple_right_xs);
    if (!have_left && !have_right) {
        out.reason = "no detected purple line";
        return out;
    }
    // Crossing/min-width invariant on detected pairs.
    if (have_left && have_right) {
        bool any_swap = false;
        for (std::size_t i = 0; i < out.ys.size(); ++i) {
            if (out.purple_left_xs[i] > out.purple_right_xs[i]) {
                std::swap(out.purple_left_xs[i], out.purple_right_xs[i]);
                any_swap = true;
            }
        }
        (void)any_swap;
        // median width gate
        std::vector<float> widths(out.ys.size());
        for (std::size_t i = 0; i < widths.size(); ++i)
            widths[i] = out.purple_right_xs[i] - out.purple_left_xs[i];
        std::nth_element(widths.begin(), widths.begin() + widths.size() / 2,
                         widths.end());
        if (widths[widths.size() / 2] < kMinLaneWidthPx) {
            out.purple_left_xs.clear();
            out.purple_right_xs.clear();
            out.reason = "corridor too narrow";
            return out;
        }
    }

    // Ego-anchored red xs from the lookahead target (preferred) else fall
    // back to compute_red_xs_from_purple.
    LookaheadTarget tgt = resolve_lookahead_target(red, frame_h, frame_w,
                                                   target_state);
    out.red_xs.assign(out.ys.size(), 0.5f * static_cast<float>(frame_w));
    std::vector<float> red_dxdy(out.ys.size(), 0.0f);
    float aC = 0.0f;
    float anchor_y_curve = 0.0f;
    bool use_ego = false;

    // Diagnostics — populated regardless of which branch produced red.
    out.red_source           = red.source;
    out.purple_confidence    = red.purple_confidence;
    out.lane_area_confidence = red.lane_area_confidence;
    out.center_confidence    = red.center_confidence;
    out.raw_target_x         = tgt.raw_present ? tgt.raw_target_x
                                : std::numeric_limits<float>::quiet_NaN();
    out.target_x             = tgt.accepted ? tgt.target_x
                                : std::numeric_limits<float>::quiet_NaN();
    out.ego_max_curvature_a  = tgt.ego_max_curvature_a;

    const float y_tail = out.ys.back();
    if (red.valid && tgt.raw_present && y_tail > tgt.target_y) {
        // Two target points: (target_x, target_y), (ego_x, y_tail).
        std::vector<cv::Point2f> targets = {
            {tgt.target_x, tgt.target_y},
            {0.5f * static_cast<float>(frame_w), y_tail},
        };
        EgoPathResult ego = build_ego_anchored_path(
            targets, out.ys, frame_h, frame_w,
            tgt.ego_max_curvature_a, tgt.ego_deadband_a,
            tgt.ego_max_top_disp_frac_w);
        out.red_xs = ego.xs;
        aC = ego.a;
        anchor_y_curve = ego.anchor_y;
        use_ego = true;
        out.ego_curvature_capped = ego.curvature_capped;
        // red_dxdy from ego path: 2*a*(y - anchor_y)
        for (std::size_t i = 0; i < out.ys.size(); ++i)
            red_dxdy[i] = 2.0f * ego.a * (out.ys[i] - ego.anchor_y);
    } else {
        compute_red_xs_from_purple(track, out.ys, frame_w, &curve_state,
                                    &out.red_xs, &aC, &anchor_y_curve);
        if (aC != 0.0f) {
            for (std::size_t i = 0; i < out.ys.size(); ++i)
                red_dxdy[i] = 2.0f * aC * (out.ys[i] - anchor_y_curve);
        }
    }
    out.used_ego_anchor = use_ego;

    yellow_corridor_from_center(out.red_xs, frame_w, red_dxdy,
                                 &out.yellow_left_xs, &out.yellow_right_xs);
    out.valid = true;
    out.reason = "ok";
    return out;
}

// ===========================================================================
// Fixed-guidance fallback (features/lane_corridor.py::fixed_guidance_template)
// ===========================================================================

namespace {

constexpr float kFixedTopFrac        = 0.45f;
constexpr float kFixedBottomFrac     = 0.72f;
// Share the real-corridor width config so the fallback never renders a
// different width than the live lane corridor (single source of truth).
constexpr float kFixedHalfFarFrac    = kYellowHalfFarFrac;
constexpr float kFixedHalfNearFrac   = kYellowHalfNearFrac;
constexpr float kFixedCenterFrac     = 0.5f;
constexpr int   kFixedNumPoints      = 24;
constexpr float kFixedMinHalfWidthPx = 12.0f;

}  // namespace

LaneGeometry make_fixed_guidance_geometry(int frame_h, int frame_w) {
    LaneGeometry geom;
    if (frame_h <= 1 || frame_w <= 1) return geom;

    const float fw = static_cast<float>(frame_w);
    const float fh = static_cast<float>(frame_h);
    const float cx = fw * kFixedCenterFrac;
    float y_top = std::max(0.0f, fh * kFixedTopFrac);
    float y_bot = std::min(fh - 1.0f, fh * kFixedBottomFrac - 1.0f);
    if (y_bot <= y_top) {
        y_bot = fh - 1.0f;
        y_top = std::max(0.0f, y_bot - 1.0f);
    }
    const float span = std::max(y_bot - y_top, 1.0f);
    const float far  = std::max(kFixedHalfFarFrac  * fw, kFixedMinHalfWidthPx);
    const float near = std::max(kFixedHalfNearFrac * fw, kFixedMinHalfWidthPx);

    geom.fitted_pts_img.reserve(kFixedNumPoints);
    geom.left_corridor.reserve(kFixedNumPoints);
    geom.right_corridor.reserve(kFixedNumPoints);
    for (int i = 0; i < kFixedNumPoints; ++i) {
        const float t = static_cast<float>(i) /
                        static_cast<float>(kFixedNumPoints - 1);
        const float y = y_top + (y_bot - y_top) * t;
        const float half = far + (near - far) * ((y - y_top) / span);
        const float lx = std::clamp(cx - half, 0.0f, fw - 1.0f);
        const float rx = std::clamp(cx + half, 0.0f, fw - 1.0f);
        geom.fitted_pts_img.emplace_back(cx, y);
        geom.left_corridor.emplace_back(lx, y);
        geom.right_corridor.emplace_back(rx, y);
    }
    geom.poly_a = 0.0f;
    geom.poly_b = 0.0f;
    geom.poly_c = cx;
    geom.control_x = static_cast<int>(std::round(cx));
    geom.control_y = static_cast<int>(std::round(y_bot));
    // Low confidence so behavior layers can distinguish fallback from a
    // real lane lock. valid=true so the overlay still draws guidance.
    geom.confidence = 0.0f;
    geom.valid = true;
    // Intentionally leave purple_left/right_pts_img empty — fallback must
    // never synthesize boundary lines.
    return geom;
}

}  // namespace adas
