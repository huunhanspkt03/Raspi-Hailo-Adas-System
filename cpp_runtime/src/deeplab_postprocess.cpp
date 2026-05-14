#include "adas/deeplab_postprocess.hpp"
#include "adas/lane_center_path.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace adas {

namespace {

// Human-readable class names for log lines. Keep aligned with LANE_CLASS_*
// constants in adas/deeplab_postprocess.hpp. Index `id` -> name.
const char* kLaneClassName(int id) {
    switch (id) {
        case 0: return "background";
        case 1: return "dotted_yellow";
        case 2: return "dotted_white";
        case 3: return "lane_area";
        case 4: return "solid_white";
        case 5: return "solid_yellow";
        default: return "extra";
    }
}

// Interpret a DeepLab output tensor as either NHWC or NCHW logits and
// return (H, W, C) along with a float* that can be indexed row-major as
// logits[(y*W + x)*C + c]. If the tensor is NCHW we transpose into a
// temporary buffer.
struct HwcView {
    int H = 0, W = 0, C = 0;
    const float* data = nullptr;
    std::vector<float> storage;
};

HwcView extractHwc(const Tensor& t) {
    HwcView v;
    if (t.shape.size() != 4 || t.shape[0] != 1) return v;
    int a = t.shape[1], b = t.shape[2], c = t.shape[3];
    // NHWC: last dim is classes (<=16 and smaller than spatial dims).
    if (c <= 16 && c < a && c < b) {
        v.H = a; v.W = b; v.C = c;
        v.data = t.data.data();
        return v;
    }
    // NCHW: first dim is classes.
    if (a <= 16 && a < b && a < c) {
        v.H = b; v.W = c; v.C = a;
        v.storage.resize(static_cast<size_t>(v.H) * v.W * v.C);
        for (int y = 0; y < v.H; ++y) {
            for (int x = 0; x < v.W; ++x) {
                for (int cl = 0; cl < v.C; ++cl) {
                    const size_t src = static_cast<size_t>(cl) * v.H * v.W +
                                       static_cast<size_t>(y) * v.W + x;
                    v.storage[(static_cast<size_t>(y) * v.W + x) * v.C + cl] = t.data[src];
                }
            }
        }
        v.data = v.storage.data();
    }
    return v;
}

cv::Mat argmaxMask(const HwcView& v) {
    cv::Mat m(v.H, v.W, CV_8UC1);
    for (int y = 0; y < v.H; ++y) {
        uint8_t* row = m.ptr<uint8_t>(y);
        for (int x = 0; x < v.W; ++x) {
            const float* p = v.data + (static_cast<size_t>(y) * v.W + x) * v.C;
            int best = 0;
            float bv = p[0];
            for (int c = 1; c < v.C; ++c) {
                if (p[c] > bv) { bv = p[c]; best = c; }
            }
            row[x] = static_cast<uint8_t>(best);
        }
    }
    return m;
}

}  // namespace

DeeplabPostprocessor::DeeplabPostprocessor(bool debug)
    : debug_(debug || std::getenv("ADAS_DEEPLAB_DEBUG") != nullptr),
      estimator_(std::make_unique<LaneCenterEstimator>()) {}

DeeplabPostprocessor::~DeeplabPostprocessor() = default;

LaneResult DeeplabPostprocessor::process(const InferenceOutput& infer_out,
                                         int frame_w,
                                         int frame_h) {
    LaneResult result;
    if (infer_out.tensors.empty()) return result;

    const Tensor& t = infer_out.tensors[0];
    HwcView hwc = extractHwc(t);
    if (hwc.data == nullptr || hwc.C < 2) return result;

    // One-time shape diagnostic — raw tensor shape, layout inference, inference
    // mask size. Cheap (one line per run) and always on so failure triage is
    // straightforward without re-building with extra flags.
    if (!logged_shape_) {
        logged_shape_ = true;
        std::fprintf(stderr,
            "[deeplab] tensor name=%s raw_shape=[%d,%d,%d,%d] -> HWC=(%d,%d,%d)"
            " upscale_to=%dx%d\n",
            t.name.c_str(), t.shape[0], t.shape[1], t.shape[2], t.shape[3],
            hwc.H, hwc.W, hwc.C, frame_w, frame_h);
    }

    cv::Mat mask_small = argmaxMask(hwc);
    cv::Mat mask_full;
    cv::resize(mask_small, mask_full, cv::Size(frame_w, frame_h), 0, 0, cv::INTER_NEAREST);
    result.class_mask = mask_full;

    const bool binary = hwc.C <= 2;
    const int  active_classes = std::min(hwc.C, LANE_NUM_CLASSES);

    // Vectorized per-class binary masks. Skips background to save work.
    for (int id = 0; id < active_classes; ++id) {
        if (id == LANE_CLASS_BACKGROUND) continue;
        cv::compare(mask_full, id, result.class_masks[id], cv::CMP_EQ);
    }

    // Drivable lane area: prefer LANE_AREA in multi-class, fall back to
    // foreground class 1 in binary models.
    const int lane_id = binary ? 1 : LANE_CLASS_LANE_AREA;
    cv::Mat lane_area;
    if (lane_id < active_classes && !result.class_masks[lane_id].empty()) {
        result.class_masks[lane_id].copyTo(lane_area);
    } else {
        lane_area = (mask_full == lane_id);
    }
    lane_area /= 255;  // 0/1
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(lane_area, lane_area, cv::MORPH_OPEN,  kernel);
    cv::morphologyEx(lane_area, lane_area, cv::MORPH_CLOSE, kernel);
    result.lane_area_mask = lane_area;

    if (debug_ && (debug_frame_++ % 60 == 0)) {
        const int total = mask_full.rows * mask_full.cols;
        std::fprintf(stderr,
            "[deeplab] frame=%d mask=%dx%d classes=%d pixels:",
            debug_frame_ - 1, mask_full.cols, mask_full.rows, hwc.C);
        for (int id = 0; id < active_classes; ++id) {
            // Count from class_mask directly so id 0 (background) also shows.
            const int n = cv::countNonZero(mask_full == id);
            std::fprintf(stderr, " %s(%d)=%d", kLaneClassName(id), id, n);
        }
        std::fprintf(stderr, "\n");
    }

    // Marking-guided centerline with lane_area fallback and temporal smoothing.
    estimator_->estimate(result, frame_w, frame_h);
    return result;
}

}  // namespace adas
