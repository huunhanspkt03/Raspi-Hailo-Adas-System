
#include "adas/lane_parity.hpp"
#include "adas/lane_parity_render.hpp"
#include "lane_parity_internal.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace adas {



const char* to_string(LaneSource s) {
    switch (s) {
        case LaneSource::Detected: return "DETECTED";
        case LaneSource::Held:     return "HELD";
        case LaneSource::Lost:     return "LOST";
    }
    return "?";
}
const char* to_string(LaneState s) {
    switch (s) {
        case LaneState::BothDetected:    return "BOTH_DETECTED";
        case LaneState::OneSideDetected: return "ONE_SIDE_DETECTED";
        case LaneState::HeldPrevious:    return "HELD_PREVIOUS";
        case LaneState::Lost:            return "LOST";
    }
    return "?";
}

namespace lane_parity {


constexpr float kEmaAlpha           = 0.40f;
constexpr float kAcceptConfidence   = 0.18f;
constexpr int   kHeldDecayFrames    = 8;     // == HELD_DECAY_FRAMES
constexpr int   kHeldFrameLimit     = 8;
constexpr float kRenderTopFrac      = 0.45f;
constexpr float kRenderBottomFrac   = 0.72f;
constexpr int   kRenderNumPoints    = 24;
constexpr float kTrackerTopFrac     = 0.55f;
constexpr int   kTrackerNumPoints   = 24;


constexpr float kMaxFrameJumpPx     = 80.0f;
constexpr float kWidthMemoryAlpha   = 0.30f; // DEFAULT_WIDTH_EMA_ALPHA

static LaneCoeffs ema_blend(const LaneCoeffs& prev, const LaneCoeffs& curr, float alpha) {
    if (!prev.valid) return curr;
    LaneCoeffs out;
    out.a = (1.0f - alpha) * prev.a + alpha * curr.a;
    out.b = (1.0f - alpha) * prev.b + alpha * curr.b;
    out.c = (1.0f - alpha) * prev.c + alpha * curr.c;
    out.valid = true;
    return out;
}

static float held_confidence_decay(float last_detected_conf, int held_streak) {
    const float decay = std::max(0.0f, 1.0f - static_cast<float>(held_streak) /
                                              static_cast<float>(std::max(1, kHeldDecayFrames)));
    return std::clamp(last_detected_conf * decay, 0.0f, 1.0f);
}

static LaneState aggregate_state(LaneSource l, LaneSource r) {
    if (l == LaneSource::Detected && r == LaneSource::Detected) return LaneState::BothDetected;
    if (l == LaneSource::Detected && (r == LaneSource::Held || r == LaneSource::Lost))
        return LaneState::OneSideDetected;
    if (r == LaneSource::Detected && (l == LaneSource::Held || l == LaneSource::Lost))
        return LaneState::OneSideDetected;
    if (l == LaneSource::Lost && r == LaneSource::Lost) return LaneState::Lost;
    return LaneState::HeldPrevious;
}



struct SideState {
    LaneCoeffs coeffs;
    float last_detected_confidence{0.0f};
    int   last_n_pixels{0};
    int   raw_missing_streak{0};
    int   held_streak{0};
    int   output_missing_streak{0};
};

struct PurpleWidthMemory {
    std::vector<float> profile;
    int frame_h{0};
    bool is_valid(int fh) const {
        return !profile.empty() && static_cast<int>(fh) == frame_h;
    }
    void update(const LaneCoeffs& l, const LaneCoeffs& r, int fh) {
        if (!l.valid || !r.valid) return;
        std::vector<float> sample(kTrackerNumPoints);
        const float y_top = std::max(0.0f, fh * kTrackerTopFrac);
        const float y_bot = static_cast<float>(fh - 1);
        for (int i = 0; i < kTrackerNumPoints; ++i) {
            const float t = static_cast<float>(i) /
                            static_cast<float>(kTrackerNumPoints - 1);
            const float y = y_top + (y_bot - y_top) * t;
            const float lx = l.a*y*y + l.b*y + l.c;
            const float rx = r.a*y*y + r.b*y + r.c;
            sample[i] = rx - lx;
        }
        if (!is_valid(fh)) { profile = sample; frame_h = fh; return; }
        for (int i = 0; i < kTrackerNumPoints; ++i) {
            profile[i] = (1.0f - kWidthMemoryAlpha) * profile[i] +
                         kWidthMemoryAlpha * sample[i];
        }
    }
    void reset() { profile.clear(); frame_h = 0; }
};

static float max_abs_delta_xs(const LaneCoeffs& cand, const LaneCoeffs& prev,
                              int frame_h) {
    if (!cand.valid || !prev.valid) return -1.0f;
    const float y_top = std::max(0.0f, frame_h * kTrackerTopFrac);
    const float y_bot = static_cast<float>(frame_h - 1);
    float max_abs = 0.0f;
    for (int i = 0; i < kTrackerNumPoints; ++i) {
        const float t = static_cast<float>(i) /
                        static_cast<float>(kTrackerNumPoints - 1);
        const float y = y_top + (y_bot - y_top) * t;
        const float cx = cand.a*y*y + cand.b*y + cand.c;
        const float px = prev.a*y*y + prev.b*y + prev.c;
        max_abs = std::max(max_abs, std::fabs(cx - px));
    }
    return max_abs;
}

}  // namespace lane_parity

// ---- LaneBoundaryTracker class --------------------------------------------

class LaneBoundaryTracker {
public:
    lane_parity::SideState left;
    lane_parity::SideState right;
    lane_parity::PurpleWidthMemory width_memory;
    LaneTargetState     target_state;
    CorridorCurveState  curve_state;
    int frame_id{-1};

    // Counters for the periodic lane-debug line.
    std::uint64_t purple_primary{0};
    std::uint64_t purple_only{0};
    std::uint64_t lane_area_only{0};
    std::uint64_t lost{0};

    void reset() {
        left  = lane_parity::SideState{};
        right = lane_parity::SideState{};
        width_memory.reset();
        target_state.reset();
        curve_state.reset();
        frame_id = -1;
        purple_primary = purple_only = lane_area_only = lost = 0;
    }

    float prev_side_split_x(int frame_h) const {
        if (!left.coeffs.valid || !right.coeffs.valid) return -1.0f;
        const float y = static_cast<float>(frame_h - 1);
        const auto& l = left.coeffs;
        const auto& r = right.coeffs;
        return 0.5f * ((l.a * y * y + l.b * y + l.c) +
                       (r.a * y * y + r.b * y + r.c));
    }
};

LaneBoundaryTracker* make_lane_boundary_tracker() { return new LaneBoundaryTracker; }
void destroy_lane_boundary_tracker(LaneBoundaryTracker* t) { delete t; }
void reset_lane_boundary_tracker(LaneBoundaryTracker* t) { if (t) t->reset(); }

namespace lane_parity {

struct SideOutcome {
    LaneSource source{LaneSource::Lost};
    LaneCoeffs coeffs;
    float confidence_final{0.0f};
    int   n_pixels{0};
    int   held_streak{0};
    int   raw_missing_streak{0};
};

// Per-side step. Frame-jump validation runs before EMA blending: a
// candidate whose sampled xs jumps more than kMaxFrameJumpPx vs the
// cached coeffs is treated as not-accepted (falls through to HELD/LOST).
static SideOutcome step_side(SideState& st, const SideFit& fit, int frame_h) {
    SideOutcome out;
    bool accept = fit.coeffs.valid && fit.confidence >= kAcceptConfidence;
    if (accept && st.coeffs.valid) {
        const float jump = max_abs_delta_xs(fit.coeffs, st.coeffs, frame_h);
        if (jump >= 0.0f && jump > kMaxFrameJumpPx) {
            accept = false;  // reject by frame-jump gate
        }
    }
    if (accept) {
        st.coeffs = ema_blend(st.coeffs, fit.coeffs, kEmaAlpha);
        st.last_detected_confidence = fit.confidence;
        st.last_n_pixels = fit.n_pixels;
        st.raw_missing_streak = 0;
        st.held_streak = 0;
        st.output_missing_streak = 0;
        out.source = LaneSource::Detected;
        out.coeffs = st.coeffs;
        out.confidence_final = fit.confidence;
        out.n_pixels = fit.n_pixels;
    } else {
        st.raw_missing_streak += 1;
        if (st.coeffs.valid && st.held_streak < kHeldFrameLimit) {
            st.held_streak += 1;
            st.output_missing_streak = 0;
            out.source = LaneSource::Held;
            out.coeffs = st.coeffs;
            out.confidence_final = held_confidence_decay(
                st.last_detected_confidence, st.held_streak);
            out.n_pixels = st.last_n_pixels;
        } else {
            st.coeffs = LaneCoeffs{};
            st.last_detected_confidence = 0.0f;
            st.last_n_pixels = 0;
            st.held_streak = 0;
            st.output_missing_streak += 1;
            out.source = LaneSource::Lost;
        }
    }
    out.held_streak = st.held_streak;
    out.raw_missing_streak = st.raw_missing_streak;
    return out;
}

static std::vector<cv::Point2f> sample_polyline_tracker(const LaneCoeffs& c, int frame_h) {
    std::vector<cv::Point2f> pts;
    if (!c.valid) return pts;
    pts.reserve(kTrackerNumPoints);
    const float y_top = std::max(0.0f, frame_h * kTrackerTopFrac);
    const float y_bot = static_cast<float>(frame_h - 1);
    for (int i = 0; i < kTrackerNumPoints; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(kTrackerNumPoints - 1);
        const float y = y_top + (y_bot - y_top) * t;
        pts.emplace_back(c.a * y * y + c.b * y + c.c, y);
    }
    return pts;
}

}  // namespace lane_parity

BoundaryTrackResult tracker_update(LaneBoundaryTracker* t,
                                   const cv::Mat& line_union,
                                   const cv::Mat& lane_area,
                                   int frame_h, int frame_w) {
    BoundaryTrackResult res;
    if (t == nullptr) return res;
    t->frame_id += 1;

    const float split = t->prev_side_split_x(frame_h);
    const float* split_ptr = (split >= 0.0f) ? &split : nullptr;

    auto fit_res = lane_parity::fit_lane_boundaries_impl(
        line_union, lane_area, frame_h, frame_w, split_ptr);

    auto L = lane_parity::step_side(t->left,  fit_res.left,  frame_h);
    auto R = lane_parity::step_side(t->right, fit_res.right, frame_h);

    res.left  = {L.source, L.coeffs, L.confidence_final, L.n_pixels,
                 L.held_streak, L.raw_missing_streak};
    res.right = {R.source, R.coeffs, R.confidence_final, R.n_pixels,
                 R.held_streak, R.raw_missing_streak};
    res.lane_state = lane_parity::aggregate_state(L.source, R.source);
    res.boundary_confidence = std::min(L.confidence_final, R.confidence_final);

    // Refresh width memory only on a clean BOTH_DETECTED frame. Mirrors
    // evaluate_memory_update's "not_both_detected" early-out; we skip the
    // weak/pinch / lane_area warmup gates (Phase 4 gaps).
    if (res.lane_state == LaneState::BothDetected) {
        t->width_memory.update(L.coeffs, R.coeffs, frame_h);
    }

    // Tracker-grid polylines (top 0.55→1.0) — used by red_center fusion.
    res.left_pts_img  = lane_parity::sample_polyline_tracker(L.coeffs, frame_h);
    res.right_pts_img = lane_parity::sample_polyline_tracker(R.coeffs, frame_h);
    return res;
}


namespace {

bool g_lane_parity_quiet = false;

float interp_width_memory(const lane_parity::PurpleWidthMemory& mem,
                          float y, int frame_h) {
    if (!mem.is_valid(frame_h) || mem.profile.size() < 2) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float y_top = std::max(0.0f, frame_h * lane_parity::kTrackerTopFrac);
    const float y_bot = static_cast<float>(frame_h - 1);
    if (y <= y_top) return mem.profile.front();
    if (y >= y_bot) return mem.profile.back();
    const float pos = (y - y_top) / std::max(y_bot - y_top, 1.0f) *
                      static_cast<float>(mem.profile.size() - 1);
    const int i0 = std::clamp(static_cast<int>(std::floor(pos)), 0,
                              static_cast<int>(mem.profile.size()) - 2);
    const float t = pos - static_cast<float>(i0);
    return mem.profile[i0] + t * (mem.profile[i0 + 1] - mem.profile[i0]);
}

void build_strict_lane_gate_from_width(const std::vector<float>& red_xs,
                                       const std::vector<float>& ys,
                                       const lane_parity::PurpleWidthMemory& mem,
                                       int frame_h, int frame_w,
                                       std::vector<cv::Point2f>* left,
                                       std::vector<cv::Point2f>* right) {
    constexpr float kMinUsableWidthPx = 40.0f;
    if (red_xs.size() != ys.size() || red_xs.size() < 2 ||
        !mem.is_valid(frame_h)) {
        return;
    }
    std::vector<cv::Point2f> l, r;
    l.reserve(ys.size());
    r.reserve(ys.size());
    for (std::size_t i = 0; i < ys.size(); ++i) {
        const float width = interp_width_memory(mem, ys[i], frame_h);
        if (!std::isfinite(width) || width < kMinUsableWidthPx) {
            l.clear();
            r.clear();
            break;
        }
        const float half = 0.5f * width;
        const float lx = std::clamp(red_xs[i] - half, 0.0f,
                                    static_cast<float>(frame_w - 1));
        const float rx = std::clamp(red_xs[i] + half, 0.0f,
                                    static_cast<float>(frame_w - 1));
        l.emplace_back(lx, ys[i]);
        r.emplace_back(rx, ys[i]);
    }
    if (l.size() >= 2 && r.size() >= 2) {
        *left = std::move(l);
        *right = std::move(r);
    }
}

void log_parity_invalid(const char* reason) {
    static unsigned long miss = 0;
    ++miss;
    // First-occurrence always prints (so a real new failure mode isn't
    // silenced); subsequent prints fold under the existing /120 throttle
    // unless quiet is on.
    if (miss == 1 || (!g_lane_parity_quiet && (miss % 120) == 0)) {
        std::cerr << "[lane_parity] invalid: " << reason
                  << " (miss=" << miss << ")\n";
    }
}

}  // namespace

void set_lane_parity_quiet_logs(bool quiet) { g_lane_parity_quiet = quiet; }

LaneGeometry compute_lane_geometry_python_parity(
    LaneBoundaryTracker* tracker,
    const cv::Mat& lane_area,
    const cv::Mat& line_union,
    const LaneGeometryConfig& /*cfg*/) {
    LaneGeometry out;
    if (tracker == nullptr || line_union.empty() || lane_area.empty()) {
        log_parity_invalid("empty_mask_or_no_tracker"); return out;
    }
    if (line_union.size() != lane_area.size()) {
        log_parity_invalid("size_mismatch"); return out;
    }

    const int frame_h = lane_area.rows;
    const int frame_w = lane_area.cols;

    BoundaryTrackResult tr = tracker_update(tracker, line_union, lane_area,
                                            frame_h, frame_w);
    RedCenter red = compute_red_centerline(lane_area, tr, frame_h, frame_w);
    RenderCorridor rc = build_render_corridor(
        tr, red, frame_h, frame_w,
        tracker->target_state, tracker->curve_state);

    // Update bookkeeping for the periodic debug line.
    switch (red.source) {
        case CenterSource::PurplePrimary:  ++tracker->purple_primary; break;
        case CenterSource::PurpleOnly:
        case CenterSource::PurpleTracked:  ++tracker->purple_only;    break;
        case CenterSource::LaneAreaOnly:   ++tracker->lane_area_only; break;
        case CenterSource::Lost:           ++tracker->lost;           break;
    }

    // Periodic one-line diagnostic so we can confirm which branch is
    // active and what the lookahead resolver is doing. Throttled to ~once
    // per second at 30 fps. Format mirrors the existing [runtime] lines.
    if (!g_lane_parity_quiet &&
        tracker->frame_id >= 0 && (tracker->frame_id % 60) == 0) {
        std::cerr << "[lane_parity] src=" << to_string(red.source)
                  << " purple_cf=" << red.purple_confidence
                  << " area_cf="   << red.lane_area_confidence
                  << " center_cf=" << red.center_confidence
                  << " raw_tx="    << rc.raw_target_x
                  << " tx="        << rc.target_x
                  << " ego_a_max=" << rc.ego_max_curvature_a
                  << " ego_capped=" << (rc.ego_curvature_capped ? 1 : 0)
                  << " used_ego="  << (rc.used_ego_anchor ? 1 : 0)
                  << " lane_state=" << to_string(tr.lane_state)
                  << "\n";
    }

    if (!rc.valid) {
        const char* reason = rc.reason ? rc.reason : "render_invalid";
        log_parity_invalid(reason);
        static unsigned long fb = 0;
        ++fb;
        if (fb == 1 || (!g_lane_parity_quiet && (fb % 60) == 0)) {
            std::cerr << "[lane_parity] fallback=fixed reason=" << reason
                      << " (fallback=" << fb << ")\n";
        }
        return make_fixed_guidance_geometry(frame_h, frame_w);
    }


    auto build_polyline = [&](const std::vector<float>& xs) {
        std::vector<cv::Point2f> pts;
        pts.reserve(xs.size());
        for (std::size_t i = 0; i < xs.size(); ++i)
            pts.emplace_back(xs[i], rc.ys[i]);
        return pts;
    };
    out.left_corridor   = build_polyline(rc.yellow_left_xs);
    out.right_corridor  = build_polyline(rc.yellow_right_xs);
    out.fitted_pts_img  = build_polyline(rc.red_xs);
    if (!rc.purple_left_xs.empty())  out.purple_left_pts_img  = build_polyline(rc.purple_left_xs);
    if (!rc.purple_right_xs.empty()) out.purple_right_pts_img = build_polyline(rc.purple_right_xs);
    build_strict_lane_gate_from_width(
        rc.red_xs, rc.ys, tracker->width_memory, frame_h, frame_w,
        &out.strict_lane_left, &out.strict_lane_right);
    if (out.strict_lane_left.empty() && !rc.purple_left_xs.empty()) {
        out.strict_lane_left = build_polyline(rc.purple_left_xs);
    }
    if (out.strict_lane_right.empty() && !rc.purple_right_xs.empty()) {
        out.strict_lane_right = build_polyline(rc.purple_right_xs);
    }

    out.control_x = static_cast<int>(std::round(rc.red_xs.back()));
    out.control_y = static_cast<int>(std::round(rc.ys.back()));

    // Steering polynomial: refit a quadratic on the rendered red xs.
    // (Mirrors the way compute_steering(geom) consumes poly_a/b/c.)
    if (out.fitted_pts_img.size() >= 3) {
        // Closed-form 3x3 normal equations on the rendered polyline.
        double S0=0,S1=0,S2=0,S3=0,S4=0,Tx=0,Txy=0,Txy2=0;
        for (const auto& p : out.fitted_pts_img) {
            const double y = p.y, x = p.x;
            const double y2 = y * y;
            S0 += 1.0; S1 += y; S2 += y2; S3 += y2*y; S4 += y2*y2;
            Tx += x; Txy += x*y; Txy2 += x*y2;
        }
        double m[3][3] = {{S4,S3,S2},{S3,S2,S1},{S2,S1,S0}};
        double v[3]    = {Txy2, Txy, Tx};
        bool ok = true;
        for (int i = 0; i < 3 && ok; ++i) {
            int piv = i;
            for (int k = i+1; k < 3; ++k)
                if (std::fabs(m[k][i]) > std::fabs(m[piv][i])) piv = k;
            if (std::fabs(m[piv][i]) < 1e-12) { ok = false; break; }
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
        if (ok) { out.poly_a = static_cast<float>(v[0]);
                  out.poly_b = static_cast<float>(v[1]);
                  out.poly_c = static_cast<float>(v[2]); }
    }

    out.confidence = red.center_confidence;
    out.valid = true;
    return out;
}

}  // namespace adas
