#include "adas/hailo_inference.hpp"

#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#if ADAS_HAS_HAILO
#include <hailo/hailort.hpp>
#endif

namespace adas {

#if ADAS_HAS_HAILO

namespace hrt = hailort;

struct ModelCtx {
    std::string name;
    std::shared_ptr<hrt::ConfiguredNetworkGroup> network_group;
    std::unique_ptr<hrt::InferVStreams> pipeline;
    std::vector<hrt::InputVStream*> input_streams_view;
    std::vector<hrt::OutputVStream*> output_streams_view;
    std::vector<int> input_shape;
    std::vector<std::vector<int>> output_shapes;
    std::vector<std::string> output_names;
    // Persistent output buffers reused across infer() calls so the timed infer
    // path doesn't re-allocate + zero-init the (multi-MB) output tensors every
    // frame. Sized on first use / frame-size change; infer overwrites them fully.
    std::map<std::string, std::vector<std::uint8_t>> out_storage;
    std::mutex infer_mutex;
};

struct HailoInference::Impl {
    std::unique_ptr<hrt::VDevice> vdevice;
    std::unordered_map<std::string, std::unique_ptr<ModelCtx>> models;
};

HailoInference::HailoInference() : impl_(std::make_unique<Impl>()) {
    // HailoRT C++ API versions on Raspberry Pi differ from the Python API:
    // some expose VDevice::create_params(), others only expose create().
    // Use the default VDevice so this wrapper stays compatible with the Pi
    // package; scheduling is left to the installed HailoRT runtime.
    auto vd = hrt::VDevice::create();
    if (!vd) {
        std::cerr << "[hailo] VDevice::create failed: status=" << vd.status() << "\n";
        return;
    }
    impl_->vdevice = std::move(vd.value());
    std::cerr << "[hailo] VDevice initialised.\n";
}

HailoInference::~HailoInference() {
    impl_->models.clear();
    impl_->vdevice.reset();
}

bool HailoInference::load(const std::string& name, const std::string& hef_path) {
    if (!impl_->vdevice) return false;
    auto hef_exp = hrt::Hef::create(hef_path);
    if (!hef_exp) {
        std::cerr << "[hailo] Hef::create failed for " << hef_path
                  << " status=" << hef_exp.status() << "\n";
        return false;
    }
    auto cfg_params_exp = hef_exp->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
    if (!cfg_params_exp) return false;
    auto ng_exp = impl_->vdevice->configure(hef_exp.value(), cfg_params_exp.value());
    if (!ng_exp) {
        std::cerr << "[hailo] configure() failed: " << ng_exp.status() << "\n";
        return false;
    }
    auto& ng_vec = ng_exp.value();
    if (ng_vec.empty()) return false;

    auto ctx = std::make_unique<ModelCtx>();
    ctx->name = name;
    ctx->network_group = std::move(ng_vec[0]);

    auto in_params_exp = ctx->network_group->make_input_vstream_params(
        /*quantized=*/true, HAILO_FORMAT_TYPE_UINT8,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    auto out_params_exp = ctx->network_group->make_output_vstream_params(
        /*quantized=*/false, HAILO_FORMAT_TYPE_FLOAT32,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!in_params_exp || !out_params_exp) {
        std::cerr << "[hailo] vstream params failed for " << name << "\n";
        return false;
    }

    auto pipe_exp = hrt::InferVStreams::create(
        *ctx->network_group, in_params_exp.value(), out_params_exp.value());
    if (!pipe_exp) {
        std::cerr << "[hailo] InferVStreams::create failed for " << name
                  << " status=" << pipe_exp.status() << "\n";
        return false;
    }
    ctx->pipeline = std::make_unique<hrt::InferVStreams>(std::move(pipe_exp.value()));

    for (auto& s : ctx->pipeline->get_input_vstreams()) {
        ctx->input_streams_view.push_back(&s.get());
        auto info = s.get().get_info();
        ctx->input_shape = {static_cast<int>(info.shape.height),
                            static_cast<int>(info.shape.width),
                            static_cast<int>(info.shape.features)};
    }
    for (auto& s : ctx->pipeline->get_output_vstreams()) {
        ctx->output_streams_view.push_back(&s.get());
        auto info = s.get().get_info();
        ctx->output_shapes.push_back({static_cast<int>(info.shape.height),
                                      static_cast<int>(info.shape.width),
                                      static_cast<int>(info.shape.features)});
        ctx->output_names.emplace_back(info.name);
    }
    std::cerr << "[hailo] loaded model '" << name << "' from " << hef_path << "\n";
    impl_->models[name] = std::move(ctx);
    return true;
}

bool HailoInference::infer(const std::string& name,
                           const std::vector<std::uint8_t>& input,
                           InferOutputs* out) {
    if (out == nullptr) return false;
    auto it = impl_->models.find(name);
    if (it == impl_->models.end()) {
        std::cerr << "[hailo] infer(): unknown model '" << name << "'\n";
        return false;
    }
    ModelCtx& ctx = *it->second;
    std::lock_guard<std::mutex> lock(ctx.infer_mutex);

    if (ctx.input_streams_view.empty() || ctx.output_streams_view.empty()) return false;

    auto* in_stream = ctx.input_streams_view[0];
    const std::size_t expected_in = in_stream->get_frame_size();
    if (input.size() != expected_in) {
        std::cerr << "[hailo] infer('" << name << "'): input size mismatch — model expects "
                  << expected_in << " bytes, got " << input.size() << "\n";
        return false;
    }

    std::map<std::string, hrt::MemoryView> input_map;
    input_map.emplace(in_stream->name(),
                      hrt::MemoryView(const_cast<std::uint8_t*>(input.data()), input.size()));

    // Reuse ctx.out_storage — (re)allocates only on first use / frame-size
    // change, removing a per-frame multi-MB alloc + zero-init from the timed
    // infer path. infer() overwrites every byte, so stale contents are harmless.
    std::map<std::string, hrt::MemoryView> output_map;
    for (auto* os : ctx.output_streams_view) {
        std::size_t bytes = os->get_frame_size();
        auto& buf = ctx.out_storage[os->name()];
        if (buf.size() != bytes) buf.resize(bytes);
        output_map.emplace(os->name(), hrt::MemoryView(buf.data(), bytes));
    }

    auto status = ctx.pipeline->infer(input_map, output_map, 1);
    if (status != HAILO_SUCCESS) {
        std::cerr << "[hailo] infer failed: status=" << status << "\n";
        return false;
    }

    // Zero-copy: expose views into ctx.out_storage instead of copying the
    // multi-MB outputs (STDC1 logits alone are ~6.5 MB) every frame. The
    // storage vectors live in a std::map keyed by stream name, so their
    // addresses are stable across calls (resize only happens on a
    // frame-size change, which re-issues the same pointers' buffers before
    // any new view is handed out). See the lifetime contract on InferTensor.
    if (out->tensors.size() != ctx.output_streams_view.size()) {
        out->tensors.resize(ctx.output_streams_view.size());
    }
    for (std::size_t i = 0; i < ctx.output_streams_view.size(); ++i) {
        auto* os = ctx.output_streams_view[i];
        InferTensor& t = out->tensors[i];
        t.name = os->name();
        t.shape = ctx.output_shapes[i];
        const auto& buf = ctx.out_storage[os->name()];
        // Outputs requested as FLOAT32 above.
        t.f32 = reinterpret_cast<const float*>(buf.data());
        t.f32_count = buf.size() / sizeof(float);
    }
    return true;
}

std::vector<int> HailoInference::input_shape(const std::string& name) const {
    auto it = impl_->models.find(name);
    if (it == impl_->models.end()) return {};
    return it->second->input_shape;
}

std::vector<std::vector<int>> HailoInference::output_shapes(const std::string& name) const {
    auto it = impl_->models.find(name);
    if (it == impl_->models.end()) return {};
    return it->second->output_shapes;
}

bool HailoInference::has_model(const std::string& name) const {
    return impl_->models.find(name) != impl_->models.end();
}

#else  // !ADAS_HAS_HAILO  — dev-machine stub

struct HailoInference::Impl {
    std::unordered_map<std::string, std::vector<int>> shapes;
};

HailoInference::HailoInference() : impl_(std::make_unique<Impl>()) {
    std::cerr << "[hailo] stub build (ADAS_HAS_HAILO=0). infer() returns empty outputs.\n";
}
HailoInference::~HailoInference() = default;

bool HailoInference::load(const std::string& name, const std::string& hef_path) {
    (void)hef_path;
    impl_->shapes[name] = {640, 640, 3};
    return true;
}
bool HailoInference::infer(const std::string& name,
                           const std::vector<std::uint8_t>& input,
                           InferOutputs* out) {
    (void)name; (void)input;
    if (out) out->tensors.clear();
    return false;
}
std::vector<int> HailoInference::input_shape(const std::string& name) const {
    auto it = impl_->shapes.find(name);
    return it == impl_->shapes.end() ? std::vector<int>{} : it->second;
}
std::vector<std::vector<int>> HailoInference::output_shapes(const std::string&) const { return {}; }
bool HailoInference::has_model(const std::string& name) const {
    return impl_->shapes.count(name) > 0;
}

#endif  // ADAS_HAS_HAILO

}  // namespace adas
