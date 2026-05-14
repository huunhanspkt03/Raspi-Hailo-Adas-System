#include "adas/lane_center_path.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace adas {

namespace {

struct RowSample {
    int   y = 0;
    float left   = -1.0f;
    float right  = -1.0f;
    float center = 0.0f;
    float width  = 0.0f;
};

// Build combined marking mask (OR of class_masks 1,2,4,5).
// Returns empty Mat if no line masks exist.
cv::Mat buildMarkingMask(const LaneResult& lane, int frame_w, int frame_h) {
    constexpr std::array<int, 4> kLineClasses = {
        LANE_CLASS_DOTTED_Y, LANE_CLASS_DOTTED_W,
        LANE_CLASS_SOLID_W,  LANE_CLASS_SOLID_Y,
    };
    cv::Mat marking;
    for (int id : kLineClasses) {
        if (id >= static_cast<int>(lane.class_masks.size())) continue;
        const cv::Mat& m = lane.class_masks[id];
        if (m.empty()) continue;
        if (marking.empty()) {
            marking = cv::Mat::zeros(frame_h, frame_w, CV_8UC1);
        }
        cv::bitwise_or(marking, m, marking);
    }
    return marking;
}

// Find the nearest nonzero pixel left/right of seed on a given row.
// Returns -1 if none.
inline int scanLeft(const uint8_t* row, int seed, int /*W*/) {
    for (int x = seed; x >= 0; --x) if (row[x]) return x;
    return -1;
}
inline int scanRight(const uint8_t* row, int seed, int W) {
    for (int x = seed; x < W; ++x) if (row[x]) return x;
    return -1;
}

// Weighted least-squares linear fit x = a*y + b, bottom rows weighted higher.
bool weightedLinearFit(const std::vector<RowSample>& samples,
                       int frame_h,
                       double& a, double& b) {
    double Sw = 0, Swy = 0, Swx = 0, Swyy = 0, Swxy = 0;
    for (const auto& s : samples) {
        double w = static_cast<double>(s.y) / std::max(1, frame_h);
        w = w * w;  // bias toward rows nearest the car
        Sw   += w;
        Swy  += w * s.y;
        Swx  += w * s.center;
        Swyy += w * s.y * s.y;
        Swxy += w * s.y * s.center;
    }
    const double denom = Sw * Swyy - Swy * Swy;
    if (std::abs(denom) < 1e-6 || Sw < 1e-6) {
        a = 0.0;
        b = Swx / std::max(Sw, 1e-6);
        return false;  // degenerate — still usable as a vertical line
    }
    a = (Sw  * Swxy - Swy * Swx) / denom;
    b = (Swyy * Swx - Swy * Swxy) / denom;
    return true;
}

}  // namespace

LaneCenterEstimator::LaneCenterEstimator() = default;
LaneCenterEstimator::LaneCenterEstimator(Config cfg) : cfg_(cfg) {}

void LaneCenterEstimator::reset() {
    has_prev_ctrl_ = false;
    prev_ctrl_x_   = 0.0f;
    nominal_width_ = 0.0f;
}

void LaneCenterEstimator::estimate(LaneResult& lane, int frame_w, int frame_h) {
    lane.fitted_path.clear();
    lane.control_x = -1;
    lane.control_y = -1;
    lane.confidence = 0.0f;
    lane.valid = false;

    if (frame_w <= 0 || frame_h <= 0) return;

    const cv::Mat marking = buildMarkingMask(lane, frame_w, frame_h);
    const cv::Mat& area   = lane.lane_area_mask;  // {0,1}
    const bool has_marking = !marking.empty();
    const bool has_area    = !area.empty();
    if (!has_marking && !has_area) return;

    if (nominal_width_ <= 0.0f) {
        nominal_width_ = cfg_.nominal_width_f * static_cast<float>(frame_w);
    }

    // --- Seed the per-row scan ---
    float seed_f;
    if (has_prev_ctrl_) {
        seed_f = prev_ctrl_x_;
    } else if (has_area) {
        long sx = 0, cnt = 0;
        const int y_lo = std::max(0, frame_h - std::max(1, frame_h / 10));
        for (int y = frame_h - 1; y >= y_lo; --y) {
            const uint8_t* arow = area.ptr<uint8_t>(y);
            for (int x = 0; x < frame_w; ++x) if (arow[x]) { sx += x; ++cnt; }
        }
        seed_f = (cnt > 0) ? static_cast<float>(sx) / cnt : frame_w * 0.5f;
    } else {
        seed_f = frame_w * 0.5f;
    }

    // --- Per-row boundary extraction ---
    const int y_top = frame_h / 3;
    const int step  = std::max(1, frame_h / cfg_.row_step_div);
    std::vector<RowSample> samples;
    samples.reserve(static_cast<size_t>((frame_h - y_top) / step + 1));

    for (int y = frame_h - 1; y >= y_top; y -= step) {
        RowSample s;
        s.y = y;

        int left = -1, right = -1;
        if (has_marking) {
            const uint8_t* mrow = marking.ptr<uint8_t>(y);
            int seed = std::clamp(static_cast<int>(std::round(seed_f)), 0, frame_w - 1);
            left  = scanLeft(mrow,  seed, frame_w);
            right = scanRight(mrow, seed, frame_w);
            // Reject if marking is on the same side only very close to seed
            // and the other side has a nearby mark — already handled: both
            // -1 checks drop to fallback.
        }

        float area_cx = -1.0f;
        if (has_area) {
            const uint8_t* arow = area.ptr<uint8_t>(y);
            long sum = 0, count = 0;
            for (int x = 0; x < frame_w; ++x) if (arow[x]) { sum += x; ++count; }
            if (count > 0) area_cx = static_cast<float>(sum) / count;
        }

        const float lo = nominal_width_ * cfg_.min_width_ratio;
        const float hi = nominal_width_ * cfg_.max_width_ratio;

        if (left >= 0 && right >= 0 && (right - left) >= lo && (right - left) <= hi) {
            // Case A: both boundaries.
            s.left   = static_cast<float>(left);
            s.right  = static_cast<float>(right);
            s.width  = s.right - s.left;
            s.center = 0.5f * (s.left + s.right);
        } else if (left >= 0 && right < 0) {
            // Case B: left only.
            s.left   = static_cast<float>(left);
            s.width  = nominal_width_;
            s.center = s.left + 0.5f * nominal_width_;
            s.right  = s.left + nominal_width_;
        } else if (right >= 0 && left < 0) {
            // Case B: right only.
            s.right  = static_cast<float>(right);
            s.width  = nominal_width_;
            s.center = s.right - 0.5f * nominal_width_;
            s.left   = s.right - nominal_width_;
        } else if (area_cx >= 0.0f) {
            // Case C: fallback to lane_area centroid.
            s.center = area_cx;
            s.width  = nominal_width_;
            s.left   = area_cx - 0.5f * nominal_width_;
            s.right  = area_cx + 0.5f * nominal_width_;
        } else {
            continue;
        }

        // Clamp center against lane_area — prevents markings on adjacent lanes
        // from pulling the path across the drivable region.
        if (area_cx >= 0.0f) {
            const float max_drift = 0.5f * nominal_width_;
            if (std::abs(s.center - area_cx) > max_drift) {
                s.center = area_cx + std::copysign(max_drift, s.center - area_cx);
            }
        }

        s.center = std::clamp(s.center, 0.0f, static_cast<float>(frame_w - 1));
        samples.push_back(s);
        seed_f = s.center;  // propagate seed up
    }

    if (samples.size() < static_cast<size_t>(cfg_.min_valid_rows)) return;

    // --- Outlier rejection on width (median-based) ---
    std::vector<float> widths;
    widths.reserve(samples.size());
    for (const auto& s : samples) widths.push_back(s.width);
    const size_t mid = widths.size() / 2;
    std::nth_element(widths.begin(), widths.begin() + mid, widths.end());
    const float med_w = widths[mid];
    const float lo_w  = med_w * cfg_.min_width_ratio;
    const float hi_w  = med_w * cfg_.max_width_ratio;

    std::vector<RowSample> kept;
    kept.reserve(samples.size());
    for (const auto& s : samples) {
        if (s.width >= lo_w && s.width <= hi_w) kept.push_back(s);
    }
    if (kept.size() < static_cast<size_t>(cfg_.min_valid_rows)) kept = samples;

    // Slow EMA on nominal width — lets us adapt to camera / road geometry.
    nominal_width_ = (1.0f - cfg_.ema_alpha_width) * nominal_width_
                   + cfg_.ema_alpha_width * med_w;

    // --- Weighted linear fit (bottom rows weighted higher) ---
    double a = 0.0, b = 0.0;
    weightedLinearFit(kept, frame_h, a, b);

    // Build polyline sorted top -> bottom for drawing.
    std::sort(kept.begin(), kept.end(),
              [](const RowSample& p, const RowSample& q) { return p.y < q.y; });
    std::vector<cv::Point> path;
    path.reserve(kept.size());
    for (const auto& s : kept) {
        int x = static_cast<int>(std::round(a * s.y + b));
        x = std::clamp(x, 0, frame_w - 1);
        path.emplace_back(x, s.y);
    }

    // Control point at bottom-most kept row.
    const int ctrl_y = kept.back().y;
    float ctrl_x_raw = static_cast<float>(a * ctrl_y + b);
    ctrl_x_raw = std::clamp(ctrl_x_raw, 0.0f, static_cast<float>(frame_w - 1));

    // --- Outlier rejection + EMA smoothing on control_x ---
    float ctrl_x;
    if (has_prev_ctrl_) {
        float delta = ctrl_x_raw - prev_ctrl_x_;
        if (std::abs(delta) > static_cast<float>(cfg_.max_jump_px)) {
            const float sign = delta > 0.0f ? 1.0f : -1.0f;
            ctrl_x_raw = prev_ctrl_x_ + sign * static_cast<float>(cfg_.max_jump_px);
        }
        ctrl_x = (1.0f - cfg_.ema_alpha_ctrl) * prev_ctrl_x_
               + cfg_.ema_alpha_ctrl * ctrl_x_raw;
    } else {
        ctrl_x = ctrl_x_raw;
    }
    prev_ctrl_x_   = ctrl_x;
    has_prev_ctrl_ = true;

    lane.fitted_path = std::move(path);
    lane.control_x   = static_cast<int>(std::round(ctrl_x));
    lane.control_y   = ctrl_y;
    lane.confidence  = static_cast<float>(kept.size()) /
                       static_cast<float>(std::max<size_t>(samples.size(), 1));
    lane.valid = true;
}

}  // namespace adas
