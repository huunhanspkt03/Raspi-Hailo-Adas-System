#include "adas/yolo26s_postprocess.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace adas {

namespace {

const std::array<std::string, YOLO26S_NUM_CLASSES> CLASS_NAMES = {
    "bus", "car", "green_light", "motorcycle", "red_light",
    "speed_limit_40", "speed_limit_50", "speed_limit_60",
    "speed_limit_70", "speed_limit_80", "truck",
};
const std::string UNKNOWN = "unknown";

constexpr int YOLO26S_STRIDES[3] = {8, 16, 32};

inline float sigmoid(float x) {
    if (x < -60.f) x = -60.f;
    if (x >  60.f) x =  60.f;
    return 1.0f / (1.0f + std::exp(-x));
}

inline cv::Rect scaleBoxFromLetterbox(float x1, float y1, float x2, float y2,
                                      const LetterboxInfo& lb,
                                      int orig_w, int orig_h) {
    const float inv = 1.0f / std::max(lb.scale, 1e-6f);
    float ox1 = (x1 - lb.pad_x) * inv;
    float oy1 = (y1 - lb.pad_y) * inv;
    float ox2 = (x2 - lb.pad_x) * inv;
    float oy2 = (y2 - lb.pad_y) * inv;
    const int ix1 = std::max(0, std::min(orig_w - 1, static_cast<int>(std::round(ox1))));
    const int iy1 = std::max(0, std::min(orig_h - 1, static_cast<int>(std::round(oy1))));
    const int ix2 = std::max(0, std::min(orig_w - 1, static_cast<int>(std::round(ox2))));
    const int iy2 = std::max(0, std::min(orig_h - 1, static_cast<int>(std::round(oy2))));
    return cv::Rect(cv::Point(ix1, iy1), cv::Point(ix2, iy2));
}

// Classify a tensor into box-head (C=4) or class-head (C=YOLO26S_NUM_CLASSES)
// and return (H, W, channel-first data pointer into `tmp`).
struct HeadView {
    int C = 0, H = 0, W = 0;
    std::vector<float> chw;  // size C*H*W
    int stride_hint = 0;
};

bool viewAsHead(const Tensor& t, HeadView& out) {
    if (t.shape.size() != 4 || t.shape[0] != 1) return false;
    int a = t.shape[1], b = t.shape[2], c = t.shape[3];
    int C = 0, H = 0, W = 0;
    bool nhwc = false;
    if (c == 4 || c == YOLO26S_NUM_CLASSES) {
        nhwc = true; H = a; W = b; C = c;
    } else if (a == 4 || a == YOLO26S_NUM_CLASSES) {
        nhwc = false; C = a; H = b; W = c;
    } else {
        return false;
    }
    out.C = C; out.H = H; out.W = W;
    out.chw.resize(static_cast<size_t>(C) * H * W);
    if (nhwc) {
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int cl = 0; cl < C; ++cl)
                    out.chw[(static_cast<size_t>(cl) * H + y) * W + x] =
                        t.data[((static_cast<size_t>(y) * W) + x) * C + cl];
    } else {
        std::copy(t.data.begin(), t.data.end(), out.chw.begin());
    }
    return true;
}

float iou(const cv::Rect& a, const cv::Rect& b) {
    const int inter = (a & b).area();
    const int uni = a.area() + b.area() - inter;
    return uni <= 0 ? 0.0f : static_cast<float>(inter) / static_cast<float>(uni);
}

std::vector<Detection> nms(std::vector<Detection> dets, float iou_thresh) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> kept;
    std::vector<char> removed(dets.size(), 0);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j]) continue;
            if (dets[i].class_id == dets[j].class_id &&
                iou(dets[i].box, dets[j].box) > iou_thresh) {
                removed[j] = 1;
            }
        }
    }
    if (kept.size() > YOLO26S_MAX_DET) kept.resize(YOLO26S_MAX_DET);
    return kept;
}

}  // namespace

const std::string& Yolo26sPostprocessor::className(int class_id) {
    if (class_id < 0 || class_id >= YOLO26S_NUM_CLASSES) return UNKNOWN;
    return CLASS_NAMES[class_id];
}

Yolo26sPostprocessor::Yolo26sPostprocessor(float conf_thresh, float nms_iou_thresh)
    : conf_thresh_(conf_thresh), nms_iou_thresh_(nms_iou_thresh) {}

std::vector<Detection> Yolo26sPostprocessor::decodeEnd2End(const Tensor& t,
                                                           const LetterboxInfo& lb,
                                                           int orig_w,
                                                           int orig_h) const {
    std::vector<Detection> out;
    if (t.shape.empty() || t.shape.back() != 6) return out;
    const size_t n = t.data.size() / 6;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const float* p = t.data.data() + i * 6;
        const float score = p[4];
        if (score < conf_thresh_) continue;
        const int cid = static_cast<int>(p[5]);
        if (cid < 0 || cid >= YOLO26S_NUM_CLASSES) continue;
        Detection d;
        d.box = scaleBoxFromLetterbox(p[0], p[1], p[2], p[3], lb, orig_w, orig_h);
        d.score = score;
        d.class_id = cid;
        d.label = CLASS_NAMES[cid];
        out.push_back(d);
    }
    return out;
}

std::vector<Detection> Yolo26sPostprocessor::decodeRawHead(const InferenceOutput& out,
                                                           const LetterboxInfo& lb,
                                                           int orig_w,
                                                           int orig_h) const {
    // Split the 6 tensors into 3 box-heads and 3 class-heads, ordered by
    // descending spatial size (stride 8, 16, 32).
    std::vector<HeadView> boxes, classes;
    for (const auto& t : out.tensors) {
        HeadView v;
        if (!viewAsHead(t, v)) continue;
        (v.C == 4 ? boxes : classes).push_back(std::move(v));
    }
    if (boxes.size() != 3 || classes.size() != 3) return {};
    auto by_area = [](const HeadView& a, const HeadView& b) {
        return a.H * a.W > b.H * b.W;
    };
    std::sort(boxes.begin(),   boxes.end(),   by_area);
    std::sort(classes.begin(), classes.end(), by_area);
    for (int i = 0; i < 3; ++i) {
        boxes[i].stride_hint   = YOLO26S_STRIDES[i];
        classes[i].stride_hint = YOLO26S_STRIDES[i];
    }

    std::vector<Detection> dets;
    for (int s = 0; s < 3; ++s) {
        const HeadView& bh = boxes[s];
        const HeadView& ch = classes[s];
        if (bh.H != ch.H || bh.W != ch.W) return {};
        const int H = bh.H, W = bh.W;
        const float stride = static_cast<float>(YOLO26S_STRIDES[s]);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                // box_dist: l,t,r,b at (y,x) in CHW layout
                const size_t off = static_cast<size_t>(y) * W + x;
                const float l = bh.chw[0 * H * W + off];
                const float tt = bh.chw[1 * H * W + off];
                const float r = bh.chw[2 * H * W + off];
                const float bb = bh.chw[3 * H * W + off];
                // anchor center (x+0.5, y+0.5) in grid units
                const float ax = x + 0.5f;
                const float ay = y + 0.5f;
                const float x1 = (ax - l) * stride;
                const float y1 = (ay - tt) * stride;
                const float x2 = (ax + r) * stride;
                const float y2 = (ay + bb) * stride;

                int best = 0;
                float best_logit = ch.chw[0 * H * W + off];
                for (int c = 1; c < YOLO26S_NUM_CLASSES; ++c) {
                    float v = ch.chw[static_cast<size_t>(c) * H * W + off];
                    if (v > best_logit) { best_logit = v; best = c; }
                }
                const float score = sigmoid(best_logit);
                if (score < conf_thresh_) continue;

                Detection d;
                d.box = scaleBoxFromLetterbox(x1, y1, x2, y2, lb, orig_w, orig_h);
                d.score = score;
                d.class_id = best;
                d.label = CLASS_NAMES[best];
                dets.push_back(d);
            }
        }
    }
    return nms(std::move(dets), nms_iou_thresh_);
}

std::vector<Detection> Yolo26sPostprocessor::process(const InferenceOutput& infer_out,
                                                     const LetterboxInfo& lb,
                                                     int orig_w,
                                                     int orig_h) {
    if (infer_out.tensors.empty()) return {};
    // End-to-end shape: (1,N,6) or (N,6).
    for (const auto& t : infer_out.tensors) {
        if (!t.shape.empty() && t.shape.back() == 6 &&
            (t.shape.size() == 2 || t.shape.size() == 3)) {
            return decodeEnd2End(t, lb, orig_w, orig_h);
        }
    }
    // Otherwise assume Hailo raw-head: 6 tensors.
    if (infer_out.tensors.size() == 6) {
        return decodeRawHead(infer_out, lb, orig_w, orig_h);
    }
    return {};
}

}  // namespace adas
