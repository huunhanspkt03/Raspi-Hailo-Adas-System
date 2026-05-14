#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace adas {

// Exactly two supported models. No generic dispatch.
enum class ModelId {
    Deeplab,  // DeepLabV3 + MobileNetV2 lane segmentation, 512x512 RGB uint8
    Yolo26s,  // YOLO26s object detection, 640x640 RGB uint8 letterbox
};

// Raw output tensor from the backend. We keep the data as float32 in NHWC
// order (or flat 1D for end2end YOLO output). The consumer knows how to
// interpret each tensor based on ModelId and shape.
struct Tensor {
    std::string name;
    std::vector<int> shape;     // as returned by the backend, e.g. {1,16,16,6}
    std::vector<float> data;    // dequantized float values
};

struct InferenceOutput {
    std::vector<Tensor> tensors;
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    // Run a single model. `rgb_uint8` must already be the correct size:
    // 512x512 for Deeplab, 640x640 for YOLO26s.
    virtual InferenceOutput infer(ModelId model,
                                  const uint8_t* rgb_uint8,
                                  int width,
                                  int height) = 0;

    virtual bool hasModel(ModelId model) const = 0;
};

}  // namespace adas
