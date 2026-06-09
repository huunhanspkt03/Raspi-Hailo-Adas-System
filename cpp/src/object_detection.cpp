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

// Nearest in-lane vehicle by z_m (else largest bbox). FCWS uses the strict
// lane gate built from the learned purple lane width around the red centerline,
// not the wider rendered drive corridor. Lane membership is decided only by
// the bbox bottom-center point.
std::optional<Detection> pick_strict_lane_vehicle(
        const std::vector<Detection>& dets, const LaneGeometry& lane_geom,
        const char* debug_role, FrameId debug_frame_id) {
    if (dets.empty()) return std::nullopt;

    auto build_gate = [](const std::vector<cv::Point2f>& left,
                         const std::vector<cv::Point2f>& right) {
        std::vector<cv::Point2f> gate;
        if (left.size() >= 2 && right.size() >= 2) {
            gate.reserve(left.size() + right.size());
            for (auto it = left.rbegin(); it != left.rend(); ++it) {
                gate.push_back(*it);
            }
            for (const auto& p : right) gate.push_back(p);
        }
        return gate;
    };

    std::vector<cv::Point2f> strict_gate;
    if (lane_geom.valid) {
        strict_gate = build_gate(lane_geom.strict_lane_left,
                                 lane_geom.strict_lane_right);
    }

    auto in_gate = [&](const Detection& d, const std::vector<cv::Point2f>& gate) {
        GateDecision decision;
        if (gate.size() < 3) {
            decision.reason = "invalid_gate";
            return decision;
        }
        // Inner-band acceptance on lane_lateral_frac = (cx - left_x)/lane_w,
        // so 0.0 == strict-lane LEFT boundary, 1.0 == RIGHT boundary at foot_y.
        // Band shifted right (was 0.20..0.80): the foot must be >=30% from the
        // left edge into the lane and may sit out to the right boundary (100%).
        constexpr float kInnerLaneMinFrac = 0.40f;
        constexpr float kInnerLaneMaxFrac = 0.80f;

        const cv::Rect2f& bb = d.bbox_img;
        const float cx = bb.x + bb.width * 0.5f;
        const float foot_y = bb.y + bb.height;
        const cv::Point2f bottom_center(cx, foot_y);
        decision.bottom_center_x = cx;
        decision.foot_y = foot_y;
        decision.bottom_center_signed_px =
            cv::pointPolygonTest(gate, bottom_center, true);
        const float left_x = boundary_x_at_y(lane_geom.strict_lane_left, foot_y);
        const float right_x = boundary_x_at_y(lane_geom.strict_lane_right, foot_y);
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
    };

    // Nearest qualifying vehicle by z_m when monocular distance is available,
    // else the largest bbox.
    const Detection* best_pose = nullptr;
    float best_z = std::numeric_limits<float>::infinity();
    const Detection* best_area = nullptr;
    float best_area_val = 0.f;
    for (const auto& d : dets) {
        if (!yolo26_is_vehicle(d.class_id)) continue;
        GateDecision gate_decision = in_gate(d, strict_gate);
        if (debug_fcws_gate()) {
            const cv::Rect2f& bb = d.bbox_img;
            const float left_x = boundary_x_at_y(lane_geom.strict_lane_left,
                                                 gate_decision.foot_y);
            const float right_x = boundary_x_at_y(lane_geom.strict_lane_right,
                                                  gate_decision.foot_y);
            std::cerr << "[fcws_gate] role=" << debug_role
                      << " frame=" << static_cast<unsigned long long>(debug_frame_id)
                      << " class=" << yolo26_class_name(d.class_id)
                      << " tid=" << d.track_id
                      << " pass=" << (gate_decision.inside ? 1 : 0)
                      << " reason=" << gate_decision.reason
                      << " bc=(" << gate_decision.bottom_center_x << ","
                      << gate_decision.foot_y << ")"
                      << " signed_px=" << gate_decision.bottom_center_signed_px
                      << " lane_frac=" << gate_decision.lane_lateral_frac
                      << " gate_x_at_y=[" << left_x << "," << right_x << "]"
                      << " bbox=(" << bb.x << "," << bb.y << ","
                      << bb.width << "," << bb.height << ")"
                      << " x_m=" << d.x_m
                      << " z_m=" << d.z_m
                      << " display=" << d.display_distance_m
                      << " dist_valid=" << (d.distance_display_valid ? 1 : 0)
                      << "\n";
        }
        if (!gate_decision.inside) continue;
        if (d.pose_valid && d.distance_display_valid && d.z_m > 0.0f) {
            if (d.z_m < best_z) { best_z = d.z_m; best_pose = &d; }
        }
        const float area = d.bbox_img.width * d.bbox_img.height;
        if (area > best_area_val) { best_area_val = area; best_area = &d; }
    }
    if (best_pose != nullptr) return *best_pose;
    if (best_area != nullptr) return *best_area;
    return std::nullopt;
}

}  // namespace

std::optional<Detection> pick_front_object(const std::vector<Detection>& dets,
                                           const LaneGeometry& lane_geom,
                                           int frame_w, int frame_h,
                                           FrameId debug_frame_id) {
    (void)frame_w; (void)frame_h;
    return pick_strict_lane_vehicle(dets, lane_geom, "front", debug_frame_id);
}

}  // namespace adas
