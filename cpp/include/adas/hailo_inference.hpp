#pragma once

// Minimal HailoRT C++ wrapper. Owns the VDevice and configures one or more
// HEF models with InferVStreams, mirroring the Python backend shape in
// src/pipeline/hailo_backend.py.
//
// One InferVStreams pipeline per model name. infer(name, input) is mutexed
// per-model so multiple worker threads can submit different models
// concurrently (lane + yolo26); scheduling is left to the installed HailoRT
// runtime so the wrapper stays compatible across Pi package versions.

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace adas {

struct InferTensor {
    std::string name;
    std::vector<int> shape;   // e.g. {H, W, C}
    // Zero-copy VIEW into the backend's persistent per-model output storage
    // (outputs are requested as FLOAT32). Copying these tensors out cost
    // ~6.5 MB memcpy per lane frame alone, so decode reads them in place.
    // LIFETIME CONTRACT: the view is valid until the NEXT infer() call for
    // the SAME model name (each worker thread owns one model and decodes
    // before its next infer, so this holds by construction).
    const float* f32{nullptr};
    std::size_t  f32_count{0};
};

struct InferOutputs {
    std::vector<InferTensor> tensors;
};

class HailoInference {
public:
    HailoInference();
    ~HailoInference();
    HailoInference(const HailoInference&) = delete;
    HailoInference& operator=(const HailoInference&) = delete;

    // Returns true on success. On non-Hailo builds (ADAS_HAS_HAILO=0) this
    // is a no-op stub that returns true; infer() then returns empty tensors
    // so the rest of the pipeline can be exercised on dev machines.
    bool load(const std::string& model_name, const std::string& hef_path);

    // Run a single inference. ``input`` must already be in the model's
    // expected layout (typically HxWxC uint8 RGB for Hailo). Returns the
    // full output set so the caller can map by name or by index.
    bool infer(const std::string& model_name,
               const std::vector<std::uint8_t>& input,
               InferOutputs* out);

    std::vector<int> input_shape(const std::string& model_name) const;
    std::vector<std::vector<int>> output_shapes(const std::string& model_name) const;
    bool has_model(const std::string& model_name) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    mutable std::mutex meta_mutex_;
};

}  // namespace adas
