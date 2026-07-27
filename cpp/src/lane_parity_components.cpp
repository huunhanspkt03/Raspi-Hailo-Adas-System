
#include "adas/lane_parity.hpp"
#include "lane_parity_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace adas {


namespace lane_parity {

constexpr int kBoundaryDownsample = 2; 
constexpr int kDefaultMinArea     = 25;
constexpr int kDefaultMinHeight   = 18;
constexpr float kDefaultMaxAspect = 4.0f;
constexpr float kDefaultMaxAngleDeg = 80.0f;
constexpr float kDefaultMinBottomWidthFrac = 0.10f;
constexpr float kDefaultMaxBottomWidthFrac = 0.85f;
constexpr int kDefaultMinPixels = 30;
constexpr float kDefaultResidualTrimFrac = 0.20f;
constexpr float kDefaultMaxResidualPx = 30.0f;
constexpr float kDefaultMaxCurvature = 5.0e-3f;
constexpr float kDefaultLowCoverageThresh = 0.30f;
constexpr float kDefaultTopFrac = 0.55f;
constexpr int kDefaultNumPoints = 24;

struct ComponentInfo {
    cv::Rect bbox;                       // in frame coords
    int      n_pixels{0};                // full-res pixel count (after ds_sq scale)
    cv::Point2f centroid;                // in frame coords
    std::vector<int>   pixels_y;         // in frame coords
    std::vector<float> pixels_x;
    int side{0};                         // 0=unknown, 1=left, 2=right
    bool rejected{false};
};

// 2D OR-pool by integer factor. Preserves thin lane lines an averaging
// downsample would drop. Mirrors _or_pool_downsample in Python.
static cv::Mat or_pool_downsample(const cv::Mat& mask, int factor) {
    if (factor <= 1) return mask;
    const int h_ds = mask.rows / factor, w_ds = mask.cols / factor;
    cv::Mat out = cv::Mat::zeros(h_ds, w_ds, CV_8UC1);
    for (int y = 0; y < h_ds; ++y) {
        std::uint8_t* dst = out.ptr<std::uint8_t>(y);
        for (int by = 0; by < factor; ++by) {
            const std::uint8_t* src = mask.ptr<std::uint8_t>(y * factor + by);
            for (int x = 0; x < w_ds; ++x) {
                std::uint8_t acc = dst[x];
                for (int bx = 0; bx < factor; ++bx) {
                    if (src[x * factor + bx]) { acc = 1; break; }
                }
                dst[x] = acc;
            }
        }
    }
    return out;
}

// PCA via cv::fitLine — angle from vertical in degrees. 0=vertical,
// 90=horizontal. Matches _angle_from_vertical_deg in Python.
static float angle_from_vertical_deg(const std::vector<cv::Point2f>& pts) {
    if (pts.size() < 4) return 90.0f;
    cv::Vec4f line;
    cv::fitLine(pts, line, cv::DIST_L2, 0, 0.01, 0.01);
    const float vx = std::fabs(line[0]);
    const float vy = std::fabs(line[1]) + 1e-9f;
    return static_cast<float>(std::atan2(vx, vy) * 180.0 / CV_PI);
}

// Per-row leftmost / rightmost x of lane_area within the ROI rows
// (analogue of extract_lane_area_edges). Returns three vectors of length
// (frame_h - top_cut). NaN slots are encoded as <0 sentinels.
struct LaneAreaEdges {
    std::vector<float> left;
    std::vector<float> right;
    std::vector<float> center;
};

static LaneAreaEdges extract_lane_area_edges(const cv::Mat& lane_area, int top_cut) {
    LaneAreaEdges e;
    if (lane_area.empty()) return e;
    const int h = lane_area.rows - top_cut;
    const int w = lane_area.cols;
    if (h <= 0 || w <= 0) return e;
    e.left.assign(h, std::numeric_limits<float>::quiet_NaN());
    e.right.assign(h, std::numeric_limits<float>::quiet_NaN());
    e.center.assign(h, std::numeric_limits<float>::quiet_NaN());
    for (int y = 0; y < h; ++y) {
        const std::uint8_t* row = lane_area.ptr<std::uint8_t>(y + top_cut);
        int first = -1, last = -1;
        for (int x = 0; x < w; ++x) if (row[x]) { first = x; break; }
        if (first < 0) continue;
        for (int x = w - 1; x >= 0; --x) if (row[x]) { last = x; break; }
        e.left[y]   = static_cast<float>(first);
        e.right[y]  = static_cast<float>(last);
        e.center[y] = 0.5f * (e.left[y] + e.right[y]);
    }
    return e;
}

// Run cv::connectedComponentsWithStats on the downsampled mask, then
// rebuild per-component pixel lists in frame coords. Filters: area,
// height, aspect, angle. Matches analyze_components + the scale-back
// loop in fit_lane_boundaries.
static std::vector<ComponentInfo> analyze_components_frame(
    const cv::Mat& union_sliced_ds, int top_cut, int ds,
    int min_area_frame, int min_height_frame,
    float max_aspect, float max_angle_deg) {
    cv::Mat labels, stats, centroids;
    const int n_labels = cv::connectedComponentsWithStats(
        union_sliced_ds, labels, stats, centroids, 8, CV_32S);
    std::vector<ComponentInfo> out;
    out.reserve(static_cast<std::size_t>(std::max(0, n_labels - 1)));
    const int ds_sq = ds * ds;
    const int min_area_ds   = std::max(1, min_area_frame / ds_sq);
    const int min_height_ds = std::max(1, min_height_frame / ds);
    for (int label = 1; label < n_labels; ++label) {
        const int x_ds = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int y_ds = stats.at<int>(label, cv::CC_STAT_TOP);
        const int w_ds = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int h_ds = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int area_ds = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area_ds <= 0) continue;

        ComponentInfo c;
        // Pre-filter on downsampled stats to avoid the bbox scan cost on
        // tiny blobs we'll throw out anyway.
        if (area_ds   < min_area_ds)   c.rejected = true;
        else if (h_ds < min_height_ds) c.rejected = true;

        // Bbox + centroid scaled back to frame space.
        c.bbox = cv::Rect(x_ds * ds, y_ds * ds + top_cut,
                          w_ds * ds, h_ds * ds);
        c.centroid = cv::Point2f(
            static_cast<float>(centroids.at<double>(label, 0) * ds),
            static_cast<float>(centroids.at<double>(label, 1) * ds + top_cut));
        c.n_pixels = area_ds * ds_sq;

        if (!c.rejected) {
            // Aspect: frame_w/frame_h equivalent.
            const float aspect = static_cast<float>(c.bbox.width) /
                                 std::max(1.0f, static_cast<float>(c.bbox.height));
            if (aspect > max_aspect) c.rejected = true;
        }

        // Collect pixels (we need pts for angle + fit even if rejected so
        // the rejected_reason field below would be informative; for
        // Phase 1 we only need pixels for accepted components).
        if (!c.rejected) {
            const cv::Mat sub = labels(cv::Rect(x_ds, y_ds, w_ds, h_ds));
            c.pixels_y.reserve(static_cast<std::size_t>(area_ds));
            c.pixels_x.reserve(static_cast<std::size_t>(area_ds));
            std::vector<cv::Point2f> pts;
            pts.reserve(static_cast<std::size_t>(area_ds));
            for (int yy = 0; yy < h_ds; ++yy) {
                const std::int32_t* row = sub.ptr<std::int32_t>(yy);
                for (int xx = 0; xx < w_ds; ++xx) {
                    if (row[xx] == label) {
                        const int frame_y = (y_ds + yy) * ds + top_cut;
                        const float frame_x = static_cast<float>((x_ds + xx) * ds);
                        c.pixels_y.push_back(frame_y);
                        c.pixels_x.push_back(frame_x);
                        pts.emplace_back(frame_x, static_cast<float>(frame_y));
                    }
                }
            }
            const float ang = angle_from_vertical_deg(pts);
            if (ang > max_angle_deg) {
                c.rejected = true;
                c.pixels_y.clear(); c.pixels_x.clear();
            }
        }
        out.push_back(std::move(c));
    }
    return out;
}

// Side-by-split assignment (line-object based, no lane_area distance gate).
// Matches assign_sides_by_split in Python.
static void assign_sides_by_split(std::vector<ComponentInfo>& comps, float split_x) {
    for (auto& c : comps) {
        if (c.rejected) continue;
        c.side = (c.centroid.x < split_x) ? 1 : 2;
    }
}

// Concatenate per-side pixels (collect_side_pixels in Python).
static void collect_side_pixels(const std::vector<ComponentInfo>& comps, int side,
                                std::vector<float>* out_ys, std::vector<float>* out_xs) {
    out_ys->clear(); out_xs->clear();
    for (const auto& c : comps) {
        if (c.rejected || c.side != side) continue;
        out_ys->insert(out_ys->end(),
                       c.pixels_y.begin(), c.pixels_y.end());
        out_xs->insert(out_xs->end(),
                       c.pixels_x.begin(), c.pixels_x.end());
    }
    // pixels_y in ComponentInfo are int; out_ys is float (consumed by polyfit).
}

// Robust deg-2 polyfit via normal equations on a 3x3 system. Accepts
// weights (uniform here); returns false on singular system.
static bool polyfit2(const std::vector<double>& ys, const std::vector<double>& xs,
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

static bool polyfit1(const std::vector<double>& ys, const std::vector<double>& xs,
                     double* b, double* c) {
    if (ys.size() < 2) return false;
    double S0=0, S1=0, S2=0, Tx=0, Txy=0;
    for (std::size_t i = 0; i < ys.size(); ++i) {
        S0 += 1.0; S1 += ys[i]; S2 += ys[i] * ys[i];
        Tx += xs[i]; Txy += xs[i] * ys[i];
    }
    const double det = S2 * S0 - S1 * S1;
    if (std::fabs(det) < 1e-12) return false;
    *b = (Txy * S0 - Tx  * S1) / det;
    *c = (Tx  * S2 - Txy * S1) / det;
    return true;
}

static SideFit weak_fit(int n, int status) {
    SideFit f; f.n_pixels = n; f.status = status; return f;
}

// fit_side — robust polyfit with curvature cap + one residual-trim pass.
// Mirrors lane_boundary_fit.py::fit_side.
static SideFit fit_side(const std::vector<float>& ys_in,
                        const std::vector<float>& xs_in,
                        int frame_h, int frame_w, int top_cut,
                        int min_pixels, float residual_trim_frac,
                        float max_residual_px, int num_points, float top_frac,
                        float max_curvature, float low_coverage_threshold) {
    const int n = static_cast<int>(ys_in.size());
    if (n < min_pixels) return weak_fit(n, n == 0 ? 0 : 1);

    std::vector<double> ys(n), xs(n);
    for (int i = 0; i < n; ++i) { ys[i] = ys_in[i]; xs[i] = xs_in[i]; }

    // Unique-row count for coverage.
    std::vector<float> unique_ys = ys_in;
    std::sort(unique_ys.begin(), unique_ys.end());
    unique_ys.erase(std::unique(unique_ys.begin(), unique_ys.end()), unique_ys.end());
    const int total_rows = std::max(1, frame_h - top_cut);
    const float coverage = static_cast<float>(unique_ys.size()) /
                           static_cast<float>(total_rows);
    const bool force_linear = coverage < low_coverage_threshold;

    auto do_fit = [&](const std::vector<double>& ys, const std::vector<double>& xs,
                      double* a, double* b, double* c) {
        if (force_linear) {
            *a = 0.0;
            return polyfit1(ys, xs, b, c);
        }
        double aa = 0, bb = 0, cc = 0;
        if (!polyfit2(ys, xs, &aa, &bb, &cc)) return false;
        if (std::fabs(aa) > max_curvature) {
            *a = 0.0;
            return polyfit1(ys, xs, b, c);
        }
        *a = aa; *b = bb; *c = cc;
        return true;
    };

    double a = 0, b = 0, c = 0;
    if (!do_fit(ys, xs, &a, &b, &c)) return weak_fit(n, 1);

    // Residuals + one trim pass.
    std::vector<double> resid(n);
    for (int i = 0; i < n; ++i) resid[i] = std::fabs(a * ys[i] * ys[i] + b * ys[i] + c - xs[i]);
    int rows_used = n;
    if (n > min_pixels && residual_trim_frac > 0.0f) {
        const int trim = std::max(1, static_cast<int>(std::round(n * residual_trim_frac)));
        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&](int a_, int b_) { return resid[a_] < resid[b_]; });
        const int keep = n - trim;
        if (keep >= min_pixels) {
            std::vector<double> ys2(keep), xs2(keep);
            for (int i = 0; i < keep; ++i) {
                ys2[i] = ys[idx[i]];
                xs2[i] = xs[idx[i]];
            }
            double a2 = 0, b2 = 0, c2 = 0;
            if (do_fit(ys2, xs2, &a2, &b2, &c2)) {
                a = a2; b = b2; c = c2;
                resid.resize(keep);
                for (int i = 0; i < keep; ++i)
                    resid[i] = std::fabs(a * ys2[i] * ys2[i] + b * ys2[i] + c - xs2[i]);
                rows_used = keep;
            }
        }
    }

    double rms_sq = 0;
    for (double r : resid) rms_sq += r * r;
    const float rms = static_cast<float>(std::sqrt(rms_sq / std::max<std::size_t>(1, resid.size())));
    if (rms > max_residual_px) return weak_fit(n, 1);

    const float density = static_cast<float>(rows_used) /
                          std::max(1.0f, static_cast<float>(frame_w));
    const float fit_quality = std::exp(-rms / std::max(1.0f, 0.04f * frame_w));
    const float confidence = std::clamp(
        0.45f * coverage + 0.35f * density + 0.20f * fit_quality, 0.0f, 1.0f);

    SideFit f;
    f.coeffs = LaneCoeffs{static_cast<float>(a), static_cast<float>(b), static_cast<float>(c), true};
    f.pts_img.reserve(static_cast<std::size_t>(num_points));
    const float y0 = std::max(0.0f, frame_h * top_frac);
    const float y1 = static_cast<float>(frame_h - 1);
    for (int i = 0; i < num_points; ++i) {
        const float t = num_points == 1 ? 0.0f
                       : static_cast<float>(i) / static_cast<float>(num_points - 1);
        const float y = y0 + (y1 - y0) * t;
        const float x = static_cast<float>(a) * y * y + static_cast<float>(b) * y + static_cast<float>(c);
        f.pts_img.emplace_back(x, y);
    }
    f.confidence = confidence;
    f.n_pixels = n;
    f.status = 2;
    return f;
}

// Cross/width gate (enforce_geometry in Python).
struct EnforceResult {
    SideFit left;
    SideFit right;
    bool dropped_reason_set{false};
};

static EnforceResult enforce_geometry(SideFit left, SideFit right,
                                      float min_bottom_width, float max_bottom_width) {
    EnforceResult r;
    if (!left.coeffs.valid || !right.coeffs.valid) {
        r.left = left; r.right = right; return r;
    }
    const std::size_t n = std::min(left.pts_img.size(), right.pts_img.size());
    if (n == 0) { r.left = left; r.right = right; return r; }
    bool crossing = false;
    for (std::size_t i = 0; i < n; ++i) {
        if (left.pts_img[i].x >= right.pts_img[i].x) { crossing = true; break; }
    }
    const float bottom_width = right.pts_img[n - 1].x - left.pts_img[n - 1].x;
    const bool bad_width = (bottom_width < min_bottom_width) || (bottom_width > max_bottom_width);
    if (!crossing && !bad_width) { r.left = left; r.right = right; return r; }
    r.dropped_reason_set = true;
    if (left.confidence < right.confidence) {
        r.left  = weak_fit(left.n_pixels, 1);
        r.right = right;
    } else {
        r.left  = left;
        r.right = weak_fit(right.n_pixels, 1);
    }
    return r;
}

// fit_lane_boundaries orchestration (Python:
// src/features/lane_boundaries.py::fit_lane_boundaries).
// Returns left + right SideFit. side_split_x priority:
//   explicit > prev_side_split_x > lane_area median > frame center.
FitLaneBoundariesResult fit_lane_boundaries_impl(
    const cv::Mat& line_union, const cv::Mat& lane_area,
    int frame_h, int frame_w,
    const float* prev_split_x) {
    FitLaneBoundariesResult r;
    if (line_union.empty()) {
        r.left  = weak_fit(0, 0);
        r.right = weak_fit(0, 0);
        r.side_split_x_used = frame_w / 2;
        return r;
    }
    const int top_cut = static_cast<int>(std::round(frame_h * kDefaultTopFrac));
    // ROI view (no clone): or_pool_downsample only reads via row pointers,
    // which are valid on a non-contiguous row-range view.
    cv::Mat union_sliced = line_union(cv::Range(top_cut, line_union.rows),
                                       cv::Range::all());
    cv::Mat union_ds = or_pool_downsample(union_sliced, kBoundaryDownsample);
    auto comps = analyze_components_frame(
        union_ds, top_cut, kBoundaryDownsample,
        kDefaultMinArea, kDefaultMinHeight,
        kDefaultMaxAspect, kDefaultMaxAngleDeg);

    // side_split_x priority: prev_split_x > lane_area median > frame center.
    int split_used = frame_w / 2;
    if (prev_split_x != nullptr) {
        split_used = std::clamp(static_cast<int>(std::round(*prev_split_x)),
                                1, frame_w - 2);
    } else {
        const LaneAreaEdges edges = extract_lane_area_edges(lane_area, top_cut);
        std::vector<float> centers;
        centers.reserve(edges.center.size());
        for (float v : edges.center) {
            if (!std::isnan(v)) centers.push_back(v);
        }
        if (!centers.empty()) {
            std::nth_element(centers.begin(),
                             centers.begin() + centers.size() / 2,
                             centers.end());
            split_used = std::clamp(static_cast<int>(std::round(centers[centers.size() / 2])),
                                    1, frame_w - 2);
        }
    }
    r.side_split_x_used = split_used;
    assign_sides_by_split(comps, static_cast<float>(split_used));

    std::vector<float> ly, lx, ry, rx;
    collect_side_pixels(comps, 1, &ly, &lx);
    collect_side_pixels(comps, 2, &ry, &rx);

    r.left = fit_side(ly, lx, frame_h, frame_w, top_cut,
                      kDefaultMinPixels, kDefaultResidualTrimFrac,
                      kDefaultMaxResidualPx, kDefaultNumPoints, kDefaultTopFrac,
                      kDefaultMaxCurvature, kDefaultLowCoverageThresh);
    r.right = fit_side(ry, rx, frame_h, frame_w, top_cut,
                       kDefaultMinPixels, kDefaultResidualTrimFrac,
                       kDefaultMaxResidualPx, kDefaultNumPoints, kDefaultTopFrac,
                       kDefaultMaxCurvature, kDefaultLowCoverageThresh);

    auto enforced = enforce_geometry(
        std::move(r.left), std::move(r.right),
        kDefaultMinBottomWidthFrac * frame_w,
        kDefaultMaxBottomWidthFrac * frame_w);
    r.left  = std::move(enforced.left);
    r.right = std::move(enforced.right);
    return r;
}

}  // namespace lane_parity

}  // namespace adas
