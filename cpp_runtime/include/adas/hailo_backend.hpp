#pragma once

#include "adas/inference_backend.hpp"

#include <memory>
#include <string>

namespace adas {

// Loads at most two HEF files — one DeepLabV3 and one YOLO26s.
// When built with ADAS_USE_HAILO=0 this becomes a stub that returns empty
// outputs, so the rest of the pipeline still runs on a dev machine.
class HailoBackend : public InferenceBackend {
public:
    HailoBackend();
    ~HailoBackend() override;

    // Either path may be empty: the corresponding model is simply not loaded.
    void loadDeeplab(const std::string& hef_path);
    void loadYolo26s(const std::string& hef_path);

    InferenceOutput infer(ModelId model,
                          const uint8_t* rgb_uint8,
                          int width,
                          int height) override;

    bool hasModel(ModelId model) const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace adas
