// Raw-head helpers for YOLO26 on Hailo. Port of:
//   src/features/yolo26_raw_head.py
//
// Hailo emits NHWC float32 after FLOAT32-output vstream params. The Python
// path tolerates both NCHW and NHWC; here we only need to handle NHWC
// because hailo_inference.cpp configures NHWC outputs. We still detect the
// channel axis from shape to stay robust if that ever changes.

#include "adas/object_detection.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace adas {

namespace {

bool is_box_channels(int c) {
    return c == 4 || c == 4 * kYolo26RegMax;
}

// Returns (channels, h, w, axis_is_last). axis_is_last==true means the
// channels are at the trailing dimension (NHWC). axis_is_last==false means
// the leading dim is channels (CHW).
bool resolve_3d(const std::vector<int>& shape,
                int* h, int* w, int* c, bool* axis_is_last) {
    std::vector<int> s = shape;
    if (s.size() == 4 && s.front() == 1) s.erase(s.begin());
    if (s.size() != 3) return false;
    // Prefer NHWC if trailing dim is the class count or a known box-channel count.
    if (s[2] == kYolo26NumClasses || is_box_channels(s[2])) {
        *h = s[0]; *w = s[1]; *c = s[2]; *axis_is_last = true;
        return true;
    }
    if (s[0] == kYolo26NumClasses || is_box_channels(s[0])) {
        *h = s[1]; *w = s[2]; *c = s[0]; *axis_is_last = false;
        return true;
    }
    return false;
}

// Decode ONE anchor's box (xyxy in *model* px) from the raw box head. Mirrors
// the old unpack_box_head + decode_stride for a single (y,x): per-side
// softmax-expectation over reg_max bins when c == 4*reg_max, else the 4 raw
// distances, then dist2bbox. Letting the caller invoke this only for anchors
// that pass the confidence threshold skips the per-side softmax (the bulk of
// decode cost) for the ~99% of anchors that are discarded anyway — output is
// identical to decoding every anchor.
cv::Vec4f decode_box_for_anchor(const InferTensor& t, int h, int w, int c,
                                bool axis_is_last, int stride, int y, int x) {
    constexpr int R = kYolo26RegMax;
    const float* src = t.f32;
    const std::ptrdiff_t a = static_cast<std::ptrdiff_t>(y) * w + x;
    float d[4];
    if (c == 4) {
        for (int b = 0; b < 4; ++b) {
            d[b] = axis_is_last
                ? src[a * 4 + b]
                : src[(static_cast<std::ptrdiff_t>(b) * h + y) * w + x];
        }
    } else {  // c == 4 * reg_max: per-side softmax-expectation
        for (int b = 0; b < 4; ++b) {
            float logits[R];
            for (int i = 0; i < R; ++i) {
                const int k = b * R + i;
                logits[i] = axis_is_last
                    ? src[a * c + k]
                    : src[(static_cast<std::ptrdiff_t>(k) * h + y) * w + x];
            }
            float m = logits[0];
            for (int i = 1; i < R; ++i) if (logits[i] > m) m = logits[i];
            float sum = 0.0f;
            float exps[R];
            for (int i = 0; i < R; ++i) {
                float v = logits[i] - m;
                if (v < -60.0f) v = -60.0f;
                exps[i] = std::exp(v);
                sum += exps[i];
            }
            if (sum < 1e-12f) sum = 1e-12f;
            float expect = 0.0f;
            for (int i = 0; i < R; ++i) expect += (exps[i] / sum) * static_cast<float>(i);
            d[b] = expect;
        }
    }
    const float ax = static_cast<float>(x) + 0.5f;
    const float ay = static_cast<float>(y) + 0.5f;
    return cv::Vec4f((ax - d[0]) * static_cast<float>(stride),
                     (ay - d[1]) * static_cast<float>(stride),
                     (ax + d[2]) * static_cast<float>(stride),
                     (ay + d[3]) * static_cast<float>(stride));
}

// (unpack_cls_head was removed: decode_yolo26 now thresholds the raw class
// logits directly — sigmoid is monotonic — and sigmoids survivors only. The
// YOLO26 HEFs in this repo do NOT absorb sigmoid; that now lives at the
// single call site in object_detection.cpp.)

}  // namespace

bool group_yolo26_raw_heads(const InferOutputs& raw,
                            std::vector<RawHeadTriplet>* out) {
    if (out == nullptr) return false;
    out->clear();
    if (raw.tensors.size() != 6) return false;

    // First pass: classify each tensor as box vs class head and capture H,W.
    struct Entry { const InferTensor* t; int h; int w; int c; bool last; bool is_box; };
    std::vector<Entry> boxes, classes;
    for (const auto& t : raw.tensors) {
        int h=0,w=0,c=0; bool last=true;
        if (!resolve_3d(t.shape, &h, &w, &c, &last)) return false;
        Entry e{&t, h, w, c, last, is_box_channels(c)};
        if (e.is_box) boxes.push_back(e);
        else if (c == kYolo26NumClasses) classes.push_back(e);
        else return false;
    }
    if (boxes.size() != 3 || classes.size() != 3) return false;

    // Sort each group by spatial size descending (largest = stride 8).
    auto by_area_desc = [](const Entry& a, const Entry& b) {
        return (a.h * a.w) > (b.h * b.w);
    };
    std::sort(boxes.begin(), boxes.end(), by_area_desc);
    std::sort(classes.begin(), classes.end(), by_area_desc);

    constexpr int kStrides[3] = {8, 16, 32};
    for (int i = 0; i < 3; ++i) {
        if (boxes[i].h != classes[i].h || boxes[i].w != classes[i].w) return false;
        RawHeadTriplet tr;
        tr.box_head = boxes[i].t;
        tr.cls_head = classes[i].t;
        tr.stride = kStrides[i];
        tr.grid_h = boxes[i].h;
        tr.grid_w = boxes[i].w;
        out->push_back(tr);
    }
    return true;
}

// ---- glue exposed to object_detection.cpp via internal linkage proxies ----
// (kept here so the heavy-math TUs are localized; object_detection.cpp
// includes these via the public hpp + extern declarations below.)
namespace yolo26_internal {
cv::Vec4f decode_box_for_anchor_ext(const InferTensor& t, int h, int w, int c,
                                    bool axis_is_last, int stride, int y, int x) {
    return decode_box_for_anchor(t, h, w, c, axis_is_last, stride, y, x);
}
bool resolve_3d_ext(const std::vector<int>& shape,
                    int* h, int* w, int* c, bool* axis_is_last) {
    return resolve_3d(shape, h, w, c, axis_is_last);
}
}  // namespace yolo26_internal

}  // namespace adas
