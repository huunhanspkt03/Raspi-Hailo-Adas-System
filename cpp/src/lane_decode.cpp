#include "adas/lane_decode.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

#include <opencv2/imgproc.hpp>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define ADAS_LANE_DECODE_NEON 1
#else
#define ADAS_LANE_DECODE_NEON 0
#endif

namespace adas {

void preprocess_lane_semantic(const cv::Mat& frame_bgr, int size,
                              std::vector<std::uint8_t>* out) {
    // Resize directly into the caller-owned buffer, then cvtColor in-place.
    // Saves the intermediate `resized` + `rgb` Mat allocations and the
    // memcpy that the old return-by-value form required. Caller hoists the
    // vector outside its hot loop so capacity is reused across frames.
    const std::size_t need = static_cast<std::size_t>(size) * size * 3;
    if (out->size() != need) out->resize(need);
    cv::Mat dst(size, size, CV_8UC3, out->data());
    cv::resize(frame_bgr, dst, cv::Size(size, size), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
}

namespace {

// Locate the 4-class logits tensor among the model outputs. Handles both
// NHWC (1,H,W,4) — Hailo emits this for STDC1 — and NCHW (1,4,H,W) — ONNX.
// Returns false if no such tensor is present.
bool find_logits(const InferOutputs& raw, const InferTensor** out_tensor,
                 int* h, int* w, bool* is_nhwc) {
    for (const auto& t : raw.tensors) {
        const auto& s = t.shape;
        // Accept rank-3 (H,W,C / C,H,W) and rank-4 (N,...,4).
        std::vector<int> d = s;
        if (d.size() == 4 && d.front() == 1) d.erase(d.begin());
        if (d.size() != 3) continue;
        if (d[2] == lane_class::kNumClasses) {
            *out_tensor = &t;
            *h = d[0]; *w = d[1]; *is_nhwc = true;
            return true;
        }
        if (d[0] == lane_class::kNumClasses) {
            *out_tensor = &t;
            *h = d[1]; *w = d[2]; *is_nhwc = false;
            return true;
        }
    }
    return false;
}

// Argmax over the 4 classes FUSED with mask building, at model resolution.
// The per-pixel label map itself is never consumed downstream, so this
// writes the two masks directly:
//   lane_area  = (best == kLaneArea)
//   line_union = (best is kDottedLine | kSolidLine) == (best & 1)  — the
//                two line classes are exactly the odd ids (1, 3) while
//                background (0) and lane area (2) are even.
// Tie-breaking matches the previous scalar argmax: strict > replaces, so
// the FIRST max (lowest class id) wins on equality.
static_assert((lane_class::kDottedLine & 1) == 1 &&
              (lane_class::kSolidLine & 1) == 1 &&
              (lane_class::kBackground & 1) == 0 &&
              (lane_class::kLaneArea & 1) == 0,
              "line_union mask relies on line classes being the odd ids");

void argmax_masks_nhwc(const float* data, int h, int w,
                       cv::Mat& lane_out, cv::Mat& line_out) {
    lane_out.create(h, w, CV_8UC1);
    line_out.create(h, w, CV_8UC1);
    constexpr int C = lane_class::kNumClasses;
    for (int y = 0; y < h; ++y) {
        const float* in = data + static_cast<std::ptrdiff_t>(y) * w * C;
        std::uint8_t* la = lane_out.ptr<std::uint8_t>(y);
        std::uint8_t* lu = line_out.ptr<std::uint8_t>(y);
        int x = 0;
#if ADAS_LANE_DECODE_NEON
        // vld4q_f32 de-interleaves 4 channels x 4 pixels per load, so the
        // 4-way argmax becomes two max/select trees, 8 pixels per loop.
        const uint32x4_t k1 = vdupq_n_u32(1);
        const uint32x4_t k2 = vdupq_n_u32(2);
        const uint32x4_t k3 = vdupq_n_u32(3);
        for (; x + 8 <= w; x += 8) {
            uint16x4_t lane16[2];
            uint16x4_t line16[2];
            for (int half = 0; half < 2; ++half) {
                const float32x4x4_t v =
                    vld4q_f32(in + static_cast<std::ptrdiff_t>(x + half * 4) * C);
                // Pairwise max with index, then combine — strict-> semantics
                // keep the lower class id on ties, matching the scalar loop.
                const uint32x4_t  gt1   = vcgtq_f32(v.val[1], v.val[0]);
                const float32x4_t m01   = vmaxq_f32(v.val[0], v.val[1]);
                const uint32x4_t  i01   = vandq_u32(gt1, k1);          // 0|1
                const uint32x4_t  gt3   = vcgtq_f32(v.val[3], v.val[2]);
                const float32x4_t m23   = vmaxq_f32(v.val[2], v.val[3]);
                const uint32x4_t  i23   = vbslq_u32(gt3, k3, k2);      // 2|3
                const uint32x4_t  use23 = vcgtq_f32(m23, m01);
                const uint32x4_t  best  = vbslq_u32(use23, i23, i01);
                lane16[half] = vmovn_u32(vandq_u32(vceqq_u32(best, k2), k1));
                line16[half] = vmovn_u32(vandq_u32(vtstq_u32(best, k1), k1));
            }
            vst1_u8(la + x, vmovn_u16(vcombine_u16(lane16[0], lane16[1])));
            vst1_u8(lu + x, vmovn_u16(vcombine_u16(line16[0], line16[1])));
        }
#endif
        for (; x < w; ++x) {
            const float* p = in + static_cast<std::ptrdiff_t>(x) * C;
            int best = 0;
            float best_v = p[0];
            for (int c = 1; c < C; ++c) {
                if (p[c] > best_v) { best_v = p[c]; best = c; }
            }
            la[x] = (best == lane_class::kLaneArea) ? 1 : 0;
            lu[x] = (best & 1) ? 1 : 0;
        }
    }
}

void argmax_masks_nchw(const float* data, int h, int w,
                       cv::Mat& lane_out, cv::Mat& line_out) {
    lane_out.create(h, w, CV_8UC1);
    line_out.create(h, w, CV_8UC1);
    const int C = lane_class::kNumClasses;
    const std::ptrdiff_t plane = static_cast<std::ptrdiff_t>(h) * w;
    for (int y = 0; y < h; ++y) {
        std::uint8_t* la = lane_out.ptr<std::uint8_t>(y);
        std::uint8_t* lu = line_out.ptr<std::uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            const std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(y) * w + x;
            int best = 0;
            float best_v = data[idx];
            for (int c = 1; c < C; ++c) {
                float v = data[c * plane + idx];
                if (v > best_v) { best_v = v; best = c; }
            }
            la[x] = (best == lane_class::kLaneArea) ? 1 : 0;
            lu[x] = (best & 1) ? 1 : 0;
        }
    }
}

}  // namespace

bool decode_stdc1(const InferOutputs& raw, int frame_h, int frame_w,
                  LaneDecodeResult* out) {
    if (out == nullptr) return false;
    const InferTensor* logits = nullptr;
    int hm = 0, wm = 0;
    bool is_nhwc = true;
    if (!find_logits(raw, &logits, &hm, &wm, &is_nhwc)) {
        std::cerr << "[lane_decode] could not find 4-class logits tensor in outputs\n";
        return false;
    }
    if (logits->f32 == nullptr || logits->f32_count == 0) {
        std::cerr << "[lane_decode] logits tensor has no float data\n";
        return false;
    }
    // Guard the argmax reads below against a short buffer (e.g. a
    // shape/stream-layout mismatch reporting more elements than were filled).
    const std::size_t need =
        static_cast<std::size_t>(hm) * wm * lane_class::kNumClasses;
    if (logits->f32_count < need) {
        std::cerr << "[lane_decode] logits tensor too small: have "
                  << logits->f32_count << " floats, need " << need
                  << " (H=" << hm << " W=" << wm << " C=" << lane_class::kNumClasses
                  << ")\n";
        return false;
    }

    // One fused pass at MODEL resolution produces both 0/1 masks straight
    // from the logits (no intermediate label map — nothing downstream reads
    // it), then the two small masks are NEAREST-resized to frame size.
    // Replaces: argmax->label (0.41 MP) + label resize (0.92 MP) + per-pixel
    // mask build at frame size (0.92 MP). The NHWC path is NEON-vectorized
    // on the Pi (vld4q de-interleave, 8 px/iteration).
    cv::Mat lane_model, line_model;
    if (is_nhwc) {
        argmax_masks_nhwc(logits->f32, hm, wm, lane_model, line_model);
    } else {
        argmax_masks_nchw(logits->f32, hm, wm, lane_model, line_model);
    }

    if (lane_model.cols == frame_w && lane_model.rows == frame_h) {
        out->lane_area = lane_model;
        out->line_union = line_model;
    } else {
        cv::resize(lane_model, out->lane_area,
                   cv::Size(frame_w, frame_h), 0, 0, cv::INTER_NEAREST);
        cv::resize(line_model, out->line_union,
                   cv::Size(frame_w, frame_h), 0, 0, cv::INTER_NEAREST);
    }
    return true;
}

}  // namespace adas
