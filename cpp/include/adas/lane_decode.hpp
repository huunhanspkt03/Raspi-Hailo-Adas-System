#pragma once

// STDC1 lane semantic-segmentation postprocess. Mirrors
// src/features/lane_semantic_postprocess.py::decode_semantic_lane_components.
//
// STDC1 expects 640x640 stretch-resized RGB uint8 input. Output is 4-class
// logits in either NHWC (1,H,W,4) on Hailo or NCHW (1,4,H,W) on ONNX. The
// class argmax is fused with mask building at model resolution (NEON on the
// Pi), then the two 0/1 masks are resized to source-frame size with
// INTER_NEAREST. (The per-pixel label map was dropped — nothing consumed it.)

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "adas/hailo_inference.hpp"
#include "adas/types.hpp"   // lane_class constants live here

namespace adas {

constexpr int kStdc1InputSize = 640;

struct LaneDecodeResult {
    cv::Mat lane_area;   // HxW uint8 (0 or 1)
    cv::Mat line_union;  // HxW uint8 (0 or 1): dotted | solid
};

// Stretch-resize a BGR frame to (size, size) RGB uint8. Matches
// pipeline.preprocessor.preprocess_lane_semantic exactly (no /255).
// Writes into *out (resized to the right length when needed) so the caller
// can reuse the buffer across frames and avoid per-frame allocation.
void preprocess_lane_semantic(const cv::Mat& frame_bgr, int size,
                              std::vector<std::uint8_t>* out);

// Decode the Hailo / ONNX output of STDC1 into per-frame masks. Returns
// false if the output set does not contain a 4-class logits tensor.
bool decode_stdc1(const InferOutputs& raw, int frame_h, int frame_w,
                  LaneDecodeResult* out);

}  // namespace adas
