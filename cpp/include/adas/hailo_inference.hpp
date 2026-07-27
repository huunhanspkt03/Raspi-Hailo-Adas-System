#pragma once
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
    bool load(const std::string& model_name, const std::string& hef_path);
    
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
