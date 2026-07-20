#include "adas/object_detection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>

#include <opencv2/imgproc.hpp>

namespace adas {

// Helpers implemented in yolo26_raw_head.cpp.
namespace yolo26_internal {
// Decode ONE anchor's box (xyxy, model px). Used by the threshold-first decode
// so sub-threshold anchors skip the per-side dist softmax entirely.
cv::Vec4f decode_box_for_anchor_ext(const InferTensor&, int, int, int, bool,
                                    int stride, int y, int x);
bool resolve_3d_ext(const std::vector<int>&, int*, int*, int*, bool*);
}  // namespace yolo26_internal

namespace {

constexpr std::array<const char*, kYolo26NumClasses> kClassNames = {{
    "bus", "car", "motorcycle", "speed_limit_50",
    "speed_limit_60", "speed_limit_70", "speed_limit_80", "truck"
}};

constexpr std::array<bool, kYolo26NumClasses> kIsVehicle = {{
    /*bus*/true, /*car*/true, /*motorcycle*/true, /*sl50*/false,
    /*sl60*/false, /*sl70*/false, /*sl80*/false, /*truck*/true
}};

// Speed limit in km/h per class id (0 = not a speed-limit sign). Mirrors
// detection_filters.SPEED_LIMIT_LABELS.
constexpr std::array<int, kYolo26NumClasses> kSpeedLimitKmh = {{
    /*bus*/0, /*car*/0, /*motorcycle*/0, /*sl50*/50,
    /*sl60*/60, /*sl70*/70, /*sl80*/80, /*truck*/0
}};

// Upper-bound throttle for each sign class. Non-sign classes pin to 1.0
// so the value is safe to feed to std::min unconditionally. User-fixed
// mapping; do not retune without an explicit request.
constexpr std::array<float, kYolo26NumClasses> kSpeedLimitThrottleCap = {{
    /*bus*/1.00f, /*car*/1.00f, /*motorcycle*/1.00f, /*sl50*/0.60f,
    /*sl60*/0.80f, /*sl70*/0.80f, /*sl80*/1.00f,    /*truck*/1.00f
}};

struct LetterboxParams {
    float scale;
    float pad_x;
    float pad_y;
};

LetterboxParams letterbox_params(int orig_w, int orig_h, int model) {
    const float s = std::min(static_cast<float>(model) / orig_w,
                             static_cast<float>(model) / orig_h);
    const float rw = orig_w * s;
    const float rh = orig_h * s;
    return { s, (model - rw) * 0.5f, (model - rh) * 0.5f };
}

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline float sigmoid(float x) {
    if (x < -60.0f) x = -60.0f; else if (x > 60.0f) x = 60.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

}  // namespace

const char* yolo26_class_name(int class_id) {
    if (class_id < 0 || class_id >= kYolo26NumClasses) return nullptr;
    return kClassNames[class_id];
}

bool yolo26_is_vehicle(int class_id) {
    if (class_id < 0 || class_id >= kYolo26NumClasses) return false;
    return kIsVehicle[class_id];
}

int yolo26_speed_limit_kmh(int class_id) {
    if (class_id < 0 || class_id >= kYolo26NumClasses) return 0;
    return kSpeedLimitKmh[class_id];
}

bool yolo26_is_traffic_sign(int class_id) {
    return yolo26_speed_limit_kmh(class_id) > 0;
}

float yolo26_speed_limit_throttle_cap(int class_id) {
    if (class_id < 0 || class_id >= kYolo26NumClasses) return 1.0f;
    return kSpeedLimitThrottleCap[class_id];
}

const char* yolo26_resolve_speed_sign(const std::vector<Detection>& dets) {
    const Detection* best = nullptr;
    float best_score = -1.0f;
    for (const auto& d : dets) {
        if (!yolo26_is_traffic_sign(d.class_id)) continue;
        if (d.score > best_score) { best_score = d.score; best = &d; }
    }
    if (best == nullptr) return nullptr;
    return yolo26_class_name(best->class_id);
}

std::vector<std::uint8_t> preprocess_yolo26(const cv::Mat& frame_bgr,
                                            int input_size,
                                            std::uint8_t pad_value) {
    std::vector<std::uint8_t> out;
    cv::Mat resize_scratch;
    preprocess_yolo26(frame_bgr, input_size, pad_value, &out, &resize_scratch);
    return out;
}

void preprocess_yolo26(const cv::Mat& frame_bgr,
                       int input_size,
                       std::uint8_t pad_value,
                       std::vector<std::uint8_t>* out,
                       cv::Mat* resize_scratch) {
    const int h = frame_bgr.rows;
    const int w = frame_bgr.cols;
    const LetterboxParams lp = letterbox_params(w, h, input_size);
    const int new_w = std::max(1, static_cast<int>(std::round(w * lp.scale)));
    const int new_h = std::max(1, static_cast<int>(std::round(h * lp.scale)));

    const std::size_t need = static_cast<std::size_t>(input_size) * input_size * 3;
    if (out->size() != need) out->resize(need);

    cv::Mat canvas_rgb(input_size, input_size, CV_8UC3, out->data());
    canvas_rgb.setTo(cv::Scalar(pad_value, pad_value, pad_value));

    cv::Mat local_resize;
    cv::Mat& resized = resize_scratch ? *resize_scratch : local_resize;
    cv::resize(frame_bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    const int pad_x = (input_size - new_w) / 2;
    const int pad_y = (input_size - new_h) / 2;
    cv::Mat dst_roi = canvas_rgb(cv::Rect(pad_x, pad_y, new_w, new_h));
    cv::cvtColor(resized, dst_roi, cv::COLOR_BGR2RGB);
}

std::vector<Detection> decode_yolo26(const InferOutputs& raw,
                                     int frame_w, int frame_h,
                                     const ObjectDecodeConfig& cfg) {
    std::vector<Detection> out;
    std::vector<RawHeadTriplet> heads;
    if (!group_yolo26_raw_heads(raw, &heads)) {
        static bool warned = false;
        if (!warned) {
            std::cerr << "[object] decode_yolo26: unexpected raw-head layout "
                         "(tensors=" << raw.tensors.size() << ")\n";
            warned = true;
        }
        return out;
    }

    std::vector<cv::Vec4f> all_boxes;     // x1,y1,x2,y2 in model coords
    std::vector<float>     all_scores;    // best class score per anchor
    std::vector<int>       all_classes;
    all_boxes.reserve(8400);
    all_scores.reserve(8400);
    all_classes.reserve(8400);

    for (const auto& tr : heads) {
        // Re-resolve channel / axis for this head.
        int bh=0, bw=0, bc=0; bool b_last=true;
        int ch=0, cw=0, cc=0; bool c_last=true;
        if (!yolo26_internal::resolve_3d_ext(tr.box_head->shape, &bh, &bw, &bc, &b_last)) continue;
        if (!yolo26_internal::resolve_3d_ext(tr.cls_head->shape, &ch, &cw, &cc, &c_last)) continue;

        // Guard the unpack memcpy/indexing against short tensor buffers
        // (e.g. a stream-layout mismatch). Fail closed: drop all detections.
        const std::size_t box_need = static_cast<std::size_t>(bh) * bw * bc;
        const std::size_t cls_need = static_cast<std::size_t>(ch) * cw * cc;
        if (tr.box_head->f32_count < box_need ||
            tr.cls_head->f32_count < cls_need) {
            static bool warned = false;
            if (!warned) {
                std::cerr << "[object] decode_yolo26: raw-head tensor too small "
                             "(box have " << tr.box_head->f32_count
                          << " need " << box_need << ", cls have "
                          << tr.cls_head->f32_count << " need " << cls_need
                          << ") — dropping detections\n";
                warned = true;
            }
            return {};
        }

        // Threshold-first on RAW logits: sigmoid is monotonic, so comparing
        // each anchor's max class logit against logit(conf_thresh) gives the
        // same survivor set as sigmoiding everything — but skips ~67k exp()
        // per frame (8400 anchors x 8 classes). sigmoid runs only for the
        // handful of survivors; the box (per-side dist softmax) is also
        // decoded lazily for survivors only. Grouping guarantees the class
        // head shares the box head's grid (bh==ch, bw==cw).
        const float ct = clampf(cfg.conf_thresh, 1e-4f, 1.0f - 1e-4f);
        const float logit_thresh = std::log(ct / (1.0f - ct));
        const float* cls_src = tr.cls_head->f32;
        const std::ptrdiff_t plane = static_cast<std::ptrdiff_t>(ch) * cw;
        constexpr int C = kYolo26NumClasses;
        for (int a = 0; a < bh * bw; ++a) {
            int best = 0;
            float bl;
            if (c_last) {  // NHWC: classes contiguous per anchor
                const float* row = cls_src + static_cast<std::ptrdiff_t>(a) * C;
                bl = row[0];
                for (int c = 1; c < C; ++c) {
                    if (row[c] > bl) { bl = row[c]; best = c; }
                }
            } else {       // CHW: one plane per class
                bl = cls_src[a];
                for (int c = 1; c < C; ++c) {
                    const float v = cls_src[static_cast<std::ptrdiff_t>(c) * plane + a];
                    if (v > bl) { bl = v; best = c; }
                }
            }
            if (bl < logit_thresh) continue;
            const int y = a / bw;
            const int x = a % bw;
            all_boxes.push_back(yolo26_internal::decode_box_for_anchor_ext(
                *tr.box_head, bh, bw, bc, b_last, tr.stride, y, x));
            all_scores.push_back(sigmoid(bl));
            all_classes.push_back(best);
        }
    }

    if (all_boxes.empty()) return out;

    // Top-k by score, descending.
    std::vector<int> idx(all_boxes.size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
    const int keep_n = std::min(cfg.max_det, static_cast<int>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + keep_n, idx.end(),
        [&](int a, int b) { return all_scores[a] > all_scores[b]; });
    idx.resize(keep_n);

    const LetterboxParams lp = letterbox_params(frame_w, frame_h, cfg.yolo_input_size);
    const float inv_scale = 1.0f / lp.scale;

    out.reserve(keep_n);
    for (int k : idx) {
        const cv::Vec4f& b = all_boxes[k];
        float x1 = (b[0] - lp.pad_x) * inv_scale;
        float y1 = (b[1] - lp.pad_y) * inv_scale;
        float x2 = (b[2] - lp.pad_x) * inv_scale;
        float y2 = (b[3] - lp.pad_y) * inv_scale;
        x1 = clampf(x1, 0.f, static_cast<float>(frame_w - 1));
        y1 = clampf(y1, 0.f, static_cast<float>(frame_h - 1));
        x2 = clampf(x2, 0.f, static_cast<float>(frame_w - 1));
        y2 = clampf(y2, 0.f, static_cast<float>(frame_h - 1));
        if (x2 <= x1 || y2 <= y1) continue;
        Detection d;
        d.class_id = all_classes[k];
        const char* nm = yolo26_class_name(d.class_id);
        d.label = nm ? nm : "";
        if (d.label.empty()) continue;
        d.bbox_img = cv::Rect2f(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
        d.score = all_scores[k];
        d.track_id = -1;
        out.push_back(std::move(d));
    }
    return out;
}

namespace {

// Previous-frame track state for the greedy IoU tracker. Single-threaded
// (object worker thread); see reset_object_tracker() for video-switch reset.
struct TrackEntry {
    int        track_id;
    int        class_id;
    cv::Rect2f bbox;
};
std::vector<TrackEntry> g_prev_tracks;
int g_next_track_id = 0;

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float iw = x2 - x1;
    const float ih = y2 - y1;
    if (iw <= 0.0f || ih <= 0.0f) return 0.0f;
    const float inter = iw * ih;
    const float uni = a.width * a.height + b.width * b.height - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

constexpr float kTrackIouThresh = 0.30f;

}  // namespace

std::vector<Detection> update_tracker(std::vector<Detection> dets) {
    // Greedy IoU association against the previous frame: for each detection,
    // take the best same-class previous track with IoU >= threshold; otherwise
    // mint a new id. Each previous track is consumed at most once. Replaces the
    // former identity stub so per-track smoothing has a stable key.
    std::vector<char> prev_used(g_prev_tracks.size(), 0);
    for (Detection& d : dets) {
        int best_prev = -1;
        float best_iou = kTrackIouThresh;
        for (std::size_t p = 0; p < g_prev_tracks.size(); ++p) {
            if (prev_used[p]) continue;
            if (g_prev_tracks[p].class_id != d.class_id) continue;
            const float ov = iou(d.bbox_img, g_prev_tracks[p].bbox);
            if (ov >= best_iou) { best_iou = ov; best_prev = static_cast<int>(p); }
        }
        if (best_prev >= 0) {
            prev_used[best_prev] = 1;
            d.track_id = g_prev_tracks[best_prev].track_id;
        } else {
            d.track_id = g_next_track_id++;
        }
    }

    g_prev_tracks.clear();
    g_prev_tracks.reserve(dets.size());
    for (const Detection& d : dets) {
        g_prev_tracks.push_back({d.track_id, d.class_id, d.bbox_img});
    }
    return dets;
}

void reset_object_tracker() {
    g_prev_tracks.clear();
    g_next_track_id = 0;
}

namespace {

struct GateDecision {
    bool inside{false};
    const char* reason{"reject"};
    double bottom_center_signed_px{-1e9};
    float lane_lateral_frac{std::numeric_limits<float>::quiet_NaN()};
    float bottom_center_x{0.0f};
    float foot_y{0.0f};
};

bool debug_fcws_gate() {
    static const bool enabled = std::getenv("ADAS_DEBUG_FCWS_GATE") != nullptr;
    return enabled;
}

float boundary_x_at_y(const std::vector<cv::Point2f>& line, float y) {
    if (line.empty()) return std::numeric_limits<float>::quiet_NaN();
    if (line.size() == 1) return line.front().x;
    for (std::size_t i = 1; i < line.size(); ++i) {
        const cv::Point2f& a = line[i - 1];
        const cv::Point2f& b = line[i];
        const float ymin = std::min(a.y, b.y);
        const float ymax = std::max(a.y, b.y);
        if (y < ymin || y > ymax) continue;
        const float dy = b.y - a.y;
        if (std::fabs(dy) < 1e-3f) return 0.5f * (a.x + b.x);
        const float t = (y - a.y) / dy;
        return a.x + t * (b.x - a.x);
    }
    const cv::Point2f& first = line.front();
    const cv::Point2f& last = line.back();
    return (std::fabs(y - first.y) < std::fabs(y - last.y)) ? first.x : last.x;
}

// ---- Layered front-object selection helpers (file-local) ----
//
// Lane membership for FCWS is decided by the bbox bottom-center point against a
// strict lane gate built from the learned purple lane width around the red
// centerline (NOT the wider rendered drive corridor). Selection is layered so a
// close lead vehicle that briefly occludes the lane lines is still caught.

// Inner lane band on lane_lateral_frac = (cx - left_x)/lane_w: 0.0 == strict
// LEFT boundary, 1.0 == RIGHT boundary at foot_y. The inner 30%..90% band keeps
// a slightly off-center lead vehicle while filtering adjacent-lane vehicles.
constexpr float kInnerLaneMinFrac = 0.30f;
constexpr float kInnerLaneMaxFrac = 0.85f;

// Fixed near-ROI fail-safe bounds — close, centered, low-in-frame ONLY, so far
// adjacent-lane vehicles can never enter through this path.
constexpr float kNearRoiMaxDistM     = 6.0f;
constexpr float kNearRoiMinXFrac     = 0.25f;
constexpr float kNearRoiMaxXFrac     = 0.75f;
constexpr float kNearRoiMinFootYFrac = 0.45f;

// near_width_memory layer (layer 2): the remembered near band only ever judges
// CLOSE vehicles, so a far adjacent-lane vehicle can never be selected through a
// stale width. Match the fixed-ROI 6 m fail-safe so only truly near vehicles
// can enter through this fallback.
constexpr float kNearWidthMaxDistM   = 6.0f;
// Sanity clamp on a remembered near-lane width, as a fraction of frame width.
constexpr float kNearWidthMinFrac    = 0.04f;
constexpr float kNearWidthMaxFrac    = 0.95f;

// Concave-safe gate polygon: left boundary reversed + right boundary forward.
std::vector<cv::Point2f> build_gate_polygon(
        const std::vector<cv::Point2f>& left,
        const std::vector<cv::Point2f>& right) {
    std::vector<cv::Point2f> gate;
    if (left.size() >= 2 && right.size() >= 2) {
        gate.reserve(left.size() + right.size());
        for (auto it = left.rbegin(); it != left.rend(); ++it) gate.push_back(*it);
        for (const auto& p : right) gate.push_back(p);
    }
    return gate;
}

// Membership of one detection's bottom-center against an explicit (left,right)
// strict-lane pair. Decoupled from LaneGeometry so it works for the current and
// the held gate identically.
GateDecision eval_gate(const Detection& d,
                       const std::vector<cv::Point2f>& gate_poly,
                       const std::vector<cv::Point2f>& left,
                       const std::vector<cv::Point2f>& right) {
    GateDecision decision;
    if (gate_poly.size() < 3) { decision.reason = "invalid_gate"; return decision; }
    const cv::Rect2f& bb = d.bbox_img;
    const float cx = bb.x + bb.width * 0.5f;
    const float foot_y = bb.y + bb.height;
    decision.bottom_center_x = cx;
    decision.foot_y = foot_y;
    decision.bottom_center_signed_px =
        cv::pointPolygonTest(gate_poly, cv::Point2f(cx, foot_y), true);
    const float left_x = boundary_x_at_y(left, foot_y);
    const float right_x = boundary_x_at_y(right, foot_y);
    const float lane_w = right_x - left_x;
    if (std::isfinite(left_x) && std::isfinite(right_x) && lane_w > 1.0f) {
        decision.lane_lateral_frac = (cx - left_x) / lane_w;
    }
    if (decision.bottom_center_signed_px >= 0.0 &&
        std::isfinite(decision.lane_lateral_frac) &&
        decision.lane_lateral_frac >= kInnerLaneMinFrac &&
        decision.lane_lateral_frac <= kInnerLaneMaxFrac) {
        decision.inside = true;
        decision.reason = "inner_bottom_center";
        return decision;
    }
    decision.reason = "inner_lane_reject";
    return decision;
}

// Closest qualifying vehicle (nearest z_m, else largest bbox) inside the gate
// described by (left,right). gate_name tags the layer in the debug trace.
std::optional<Detection> select_in_gate(
        const std::vector<Detection>& dets,
        const std::vector<cv::Point2f>& left,
        const std::vector<cv::Point2f>& right,
        const char* role, const char* gate_name,
        FrameId frame_id) {
    const std::vector<cv::Point2f> gate = build_gate_polygon(left, right);
    if (gate.size() < 3) return std::nullopt;

    const Detection* best_pose = nullptr;
    float best_z = std::numeric_limits<float>::infinity();
    const Detection* best_area = nullptr;
    float best_area_val = 0.f;
    for (const auto& d : dets) {
        if (!yolo26_is_vehicle(d.class_id)) continue;
        const GateDecision gd = eval_gate(d, gate, left, right);
        if (debug_fcws_gate()) {
            const float left_x = boundary_x_at_y(left, gd.foot_y);
            const float right_x = boundary_x_at_y(right, gd.foot_y);
            const cv::Rect2f& bb = d.bbox_img;
            std::cerr << "[fcws_gate] role=" << role << " gate=" << gate_name
                      << " frame=" << static_cast<unsigned long long>(frame_id)
                      << " class=" << yolo26_class_name(d.class_id)
                      << " tid=" << d.track_id
                      << " pass=" << (gd.inside ? 1 : 0)
                      << " reason=" << gd.reason
                      << " bc=(" << gd.bottom_center_x << "," << gd.foot_y << ")"
                      << " signed_px=" << gd.bottom_center_signed_px
                      << " lane_frac=" << gd.lane_lateral_frac
                      << " gate_x_at_y=[" << left_x << "," << right_x << "]"
                      << " bbox=(" << bb.x << "," << bb.y << ","
                      << bb.width << "," << bb.height << ")"
                      << " x_m=" << d.x_m << " z_m=" << d.z_m
                      << " display=" << d.display_distance_m
                      << " dist_valid=" << (d.distance_display_valid ? 1 : 0)
                      << "\n";
        }
        if (!gd.inside) continue;
        if (d.pose_valid && d.distance_display_valid && d.z_m > 0.0f) {
            if (d.z_m < best_z) { best_z = d.z_m; best_pose = &d; }
        }
        const float area = d.bbox_img.width * d.bbox_img.height;
        if (area > best_area_val) { best_area_val = area; best_area = &d; }
    }
    const Detection* chosen = best_pose ? best_pose : best_area;
    if (chosen != nullptr && debug_fcws_gate()) {
        std::cerr << "[fcws_gate] role=" << role << " gate=" << gate_name
                  << " SELECTED tid=" << chosen->track_id
                  << " class=" << yolo26_class_name(chosen->class_id)
                  << " display=" << chosen->display_distance_m
                  << " z_m=" << chosen->z_m << "\n";
    }
    if (chosen != nullptr) return *chosen;
    return std::nullopt;
}

// Near-lane width + center from a valid strict gate, sampled at the near
// (largest-y / closest-to-camera) row both boundaries cover. Returns false if
// the geometry is degenerate. Only scalars are kept — NOT the polygon.
bool near_lane_width_center(const std::vector<cv::Point2f>& left,
                            const std::vector<cv::Point2f>& right,
                            float* width_px, float* center_x) {
    if (left.size() < 2 || right.size() < 2) return false;
    auto max_y = [](const std::vector<cv::Point2f>& l) {
        float m = -1e30f;
        for (const auto& p : l) m = std::max(m, p.y);
        return m;
    };
    const float near_y = std::min(max_y(left), max_y(right));
    const float lx = boundary_x_at_y(left, near_y);
    const float rx = boundary_x_at_y(right, near_y);
    if (!std::isfinite(lx) || !std::isfinite(rx)) return false;
    const float w = rx - lx;
    if (!(w > 1.0f)) return false;
    *width_px  = w;
    *center_x  = 0.5f * (lx + rx);
    return true;
}

// Center x for the rebuilt near band, preferring the CURRENT red center
// near-bottom, then the last remembered near center, then image center.
float pick_near_center_x(const LaneGeometry& g,
                         const FrontObjectNearLaneMemory& mem,
                         int frame_w, const char** src) {
    if (!g.fitted_pts_img.empty()) {           // current red centerline near-bottom
        const cv::Point2f* near = &g.fitted_pts_img.front();
        for (const auto& p : g.fitted_pts_img) if (p.y > near->y) near = &p;
        *src = "red_center";
        return near->x;
    }
    if (mem.usable()) { *src = "mem_center"; return mem.near_center_x(); }
    *src = "frame_center";
    return 0.5f * static_cast<float>(frame_w);
}

// Layer 2: near_width_memory. Rebuild a simple symmetric near band of the
// remembered lane width around `center_x` and accept only CLOSE vehicles whose
// bottom-center sits in the inner band. No stale polygon is replayed — just the
// scalar width. Picks the nearest accepted vehicle by display distance.
std::optional<Detection> select_near_width_gate(
        const std::vector<Detection>& dets,
        float center_x, float near_width_px, const char* center_src,
        int age, const char* role, FrameId frame_id) {
    const float half = 0.5f * near_width_px;
    if (!(half > 0.5f)) return std::nullopt;
    const float left_x  = center_x - half;
    const float right_x = center_x + half;
    const float band_w  = right_x - left_x;
    const Detection* best = nullptr;
    float best_dist = std::numeric_limits<float>::infinity();
    for (const auto& d : dets) {
        if (!yolo26_is_vehicle(d.class_id)) continue;
        const cv::Rect2f& bb = d.bbox_img;
        const float cx = bb.x + bb.width * 0.5f;
        const float foot_y = bb.y + bb.height;
        const bool close = d.distance_display_valid && d.display_distance_m >= 0.0f &&
                           d.display_distance_m <= kNearWidthMaxDistM;
        float frac = std::numeric_limits<float>::quiet_NaN();
        bool inside = false;
        const char* reason = "near_reject";
        if (!close) {
            reason = "near_too_far";
        } else {
            frac = (cx - left_x) / band_w;
            if (cx >= left_x && cx <= right_x &&
                frac >= kInnerLaneMinFrac && frac <= kInnerLaneMaxFrac) {
                inside = true;
                reason = "near_inner_band";
            }
        }
        if (debug_fcws_gate()) {
            std::cerr << "[fcws_gate] role=" << role << " gate=near_width_memory"
                      << " age=" << age
                      << " width_px=" << near_width_px
                      << " center_x=" << center_x
                      << " center_src=" << center_src
                      << " frame=" << static_cast<unsigned long long>(frame_id)
                      << " class=" << yolo26_class_name(d.class_id)
                      << " tid=" << d.track_id
                      << " pass=" << (inside ? 1 : 0)
                      << " reason=" << reason
                      << " bc=(" << cx << "," << foot_y << ")"
                      << " band=[" << left_x << "," << right_x << "]"
                      << " lane_frac=" << frac
                      << " display=" << d.display_distance_m
                      << " dist_valid=" << (d.distance_display_valid ? 1 : 0)
                      << "\n";
        }
        if (inside && d.display_distance_m < best_dist) {
            best_dist = d.display_distance_m;
            best = &d;
        }
    }
    if (best != nullptr && debug_fcws_gate()) {
        std::cerr << "[fcws_gate] role=" << role << " gate=near_width_memory"
                  << " SELECTED tid=" << best->track_id
                  << " class=" << yolo26_class_name(best->class_id)
                  << " display=" << best->display_distance_m << "\n";
    }
    if (best != nullptr) return *best;
    return std::nullopt;
}

// Layer 3: fixed near-pixel ROI fail-safe. A very close vehicle can hide
// purple/lane cues so BOTH the strict gate and the near-width memory miss it.
// Accept ONLY a close, centered, low-in-frame vehicle; nearest by distance.
std::optional<Detection> select_fixed_near_roi(
        const std::vector<Detection>& dets, int frame_w, int frame_h,
        const char* role, FrameId frame_id) {
    if (frame_w <= 0 || frame_h <= 0) return std::nullopt;
    const Detection* best = nullptr;
    float best_dist = std::numeric_limits<float>::infinity();
    for (const auto& d : dets) {
        if (!yolo26_is_vehicle(d.class_id)) continue;
        if (!(d.distance_display_valid && d.display_distance_m >= 0.0f &&
              d.display_distance_m <= kNearRoiMaxDistM)) continue;
        const cv::Rect2f& bb = d.bbox_img;
        const float cx = bb.x + bb.width * 0.5f;
        const float foot_y = bb.y + bb.height;
        const float x_frac = cx / static_cast<float>(frame_w);
        const float y_frac = foot_y / static_cast<float>(frame_h);
        if (x_frac >= kNearRoiMinXFrac && x_frac <= kNearRoiMaxXFrac &&
            y_frac >= kNearRoiMinFootYFrac && d.display_distance_m < best_dist) {
            best_dist = d.display_distance_m;
            best = &d;
        }
    }
    if (best != nullptr && debug_fcws_gate()) {
        const cv::Rect2f& bb = best->bbox_img;
        const float cx = bb.x + bb.width * 0.5f;
        const float foot_y = bb.y + bb.height;
        std::cerr << "[fcws_gate] role=" << role << " gate=fixed_near_roi"
                  << " frame=" << static_cast<unsigned long long>(frame_id)
                  << " class=" << yolo26_class_name(best->class_id)
                  << " tid=" << best->track_id
                  << " display=" << best->display_distance_m
                  << " x_frac=" << (cx / static_cast<float>(frame_w))
                  << " foot_y_frac=" << (foot_y / static_cast<float>(frame_h))
                  << " bc=(" << cx << "," << foot_y << ")\n";
    }
    if (best != nullptr) return *best;
    return std::nullopt;
}

}  // namespace

std::optional<Detection> pick_front_object(const std::vector<Detection>& dets,
                                           const LaneGeometry& lane_geom,
                                           int frame_w, int frame_h,
                                           FrontObjectNearLaneMemory& near_mem,
                                           FrameId debug_frame_id) {
    // The current strict gate is usable only when the parity path produced a
    // real strict lane gate this frame. A close lead vehicle that occludes the
    // purple lines leaves strict_lane_* empty even while lane_geom.valid holds.
    const bool current_usable =
        lane_geom.valid &&
        lane_geom.strict_lane_left.size() >= 2 &&
        lane_geom.strict_lane_right.size() >= 2;

    // Maintain the near-lane-WIDTH memory (scalars only) from the current valid
    // strict gate. Width is sanity-clamped so an absurd fit can't widen the
    // fallback band.
    if (current_usable) {
        float w_px = 0.0f, c_x = 0.0f;
        if (near_lane_width_center(lane_geom.strict_lane_left,
                                   lane_geom.strict_lane_right, &w_px, &c_x)) {
            const float min_w = kNearWidthMinFrac * static_cast<float>(frame_w);
            const float max_w = kNearWidthMaxFrac * static_cast<float>(frame_w);
            near_mem.store(std::clamp(w_px, min_w, max_w), c_x);
        } else {
            near_mem.tick_invalid();
        }
    } else {
        near_mem.tick_invalid();
    }

    // Layer 1 — current strict gate.
    if (current_usable) {
        auto sel = select_in_gate(dets, lane_geom.strict_lane_left,
                                  lane_geom.strict_lane_right,
                                  "front", "current_strict", debug_frame_id);
        if (sel) return sel;
    }

    // Layer 2 — near_width_memory: remembered near width re-centered on the
    // current red center, judging CLOSE vehicles only. Only consulted when the
    // current gate is unusable (occlusion); expires after kMaxHeldFrames, after
    // which only the fixed ROI remains.
    if (!current_usable && near_mem.usable()) {
        const char* csrc = "frame_center";
        const float center_x =
            pick_near_center_x(lane_geom, near_mem, frame_w, &csrc);
        auto sel = select_near_width_gate(dets, center_x,
                                          near_mem.near_width_px(), csrc,
                                          near_mem.age(), "front",
                                          debug_frame_id);
        if (sel) return sel;
    }

    // Layer 3 — fixed near-pixel ROI fail-safe (close + centered vehicles only).
    return select_fixed_near_roi(dets, frame_w, frame_h, "front",
                                 debug_frame_id);
}

}  // namespace adas
