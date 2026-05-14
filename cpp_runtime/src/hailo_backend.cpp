#include "adas/hailo_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

#if ADAS_USE_HAILO
#include <hailo/hailort.hpp>
#endif

namespace adas {

namespace {

// Env-gated verbosity. `ADAS_LOG_VERBOSE=1` enables one-time load diagnostics
// (stream count, shapes, frame_size). `ADAS_LOG_EVERY_N=<int>` (>=1) enables
// a per-inference timing + raw-output stats line every N frames. Both are
// opt-in so production runs stay quiet.
bool envFlag(const char* key) {
    const char* v = std::getenv(key);
    return v != nullptr && v[0] != 0 && v[0] != '0';
}
int envInt(const char* key, int fallback) {
    const char* v = std::getenv(key);
    if (!v || !v[0]) return fallback;
    int n = std::atoi(v);
    return n > 0 ? n : fallback;
}

}  // namespace

#if ADAS_USE_HAILO
// ---------------------------------------------------------------------------
// Real HailoRT backend (Raspberry Pi + Hailo-8/8L).
//
// Uses the legacy ConfigureNetworkGroup + input/output vstream API because
// it is well supported on the Pi HailoRT builds currently shipped. One VDevice
// is shared between the DeepLab and YOLO26s network groups.
// ---------------------------------------------------------------------------
using namespace hailort;

namespace {

std::string shapeToString(const hailo_3d_image_shape_t& s) {
    std::ostringstream os;
    os << s.height << "x" << s.width << "x" << s.features;
    return os.str();
}

// Summarize a freshly-read output buffer (float32) for sanity checks. Logs
// min/max/mean plus the first 8 samples. Cheap — O(n) once every N frames.
void logTensorStats(const char* tag, const std::vector<uint8_t>& buf) {
    const size_t n = buf.size() / sizeof(float);
    if (n == 0) {
        std::fprintf(stderr, "[hailo] %s empty output\n", tag);
        return;
    }
    const float* p = reinterpret_cast<const float*>(buf.data());
    float mn = p[0], mx = p[0];
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float v = p[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
    }
    const double mean = sum / static_cast<double>(n);
    std::fprintf(stderr, "[hailo] %s n=%zu min=%.4f max=%.4f mean=%.4f head=[",
                 tag, n, mn, mx, mean);
    const size_t head = std::min<size_t>(8, n);
    for (size_t i = 0; i < head; ++i) {
        std::fprintf(stderr, "%s%.3f", i ? "," : "", p[i]);
    }
    std::fprintf(stderr, "]\n");
}

}  // namespace

struct LoadedModel {
    std::shared_ptr<ConfiguredNetworkGroup> group;
    std::vector<InputVStream>  in_streams;
    std::vector<OutputVStream> out_streams;
    // Scratch buffers sized once after activation.
    std::vector<std::vector<uint8_t>> out_buffers;
};

struct HailoBackend::Impl {
    std::unique_ptr<VDevice> vdevice;
    std::unique_ptr<LoadedModel> deeplab;
    std::unique_ptr<LoadedModel> yolo;

    // Per-model frame counters and timing accumulators (used when
    // ADAS_LOG_EVERY_N>0). Keyed by tag so Deeplab and YOLO stats don't mix.
    struct Stats {
        size_t frames = 0;
        double ms_sum = 0.0;
        std::chrono::steady_clock::time_point window_start =
            std::chrono::steady_clock::now();
    };
    Stats deeplab_stats;
    Stats yolo_stats;

    void ensureVDevice() {
        if (vdevice) return;
        std::fprintf(stderr, "[hailo] creating VDevice\n");
        auto vd = VDevice::create();
        if (!vd) throw std::runtime_error("Failed to create Hailo VDevice");
        vdevice = std::move(vd.value());
        std::fprintf(stderr, "[hailo] VDevice ready\n");
    }

    std::unique_ptr<LoadedModel> loadHef(const std::string& path, const char* tag) {
        const auto t_begin = std::chrono::steady_clock::now();
        ensureVDevice();
        std::fprintf(stderr, "[hailo] [%s] loading HEF: %s\n", tag, path.c_str());
        auto hef = Hef::create(path);
        if (!hef) throw std::runtime_error("Hef::create failed: " + path);

        auto cfg_params = hef->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
        if (!cfg_params) throw std::runtime_error("create_configure_params failed");

        auto groups = vdevice->configure(hef.value(), cfg_params.value());
        if (!groups || groups->empty())
            throw std::runtime_error("configure() returned no network groups");

        auto group = std::move(groups.value()[0]);

        auto in_params = group->make_input_vstream_params(
            {}, HAILO_FORMAT_TYPE_UINT8,
            HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
            HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        auto out_params = group->make_output_vstream_params(
            {}, HAILO_FORMAT_TYPE_FLOAT32,
            HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
            HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        if (!in_params || !out_params)
            throw std::runtime_error("make_vstream_params failed for " + path);

        auto in_vstreams  = VStreamsBuilder::create_input_vstreams (*group, in_params.value());
        auto out_vstreams = VStreamsBuilder::create_output_vstreams(*group, out_params.value());
        if (!in_vstreams || !out_vstreams)
            throw std::runtime_error("Failed to create vstreams for " + path);

        auto model = std::make_unique<LoadedModel>();
        model->group = group;
        model->in_streams  = std::move(in_vstreams.value());
        model->out_streams = std::move(out_vstreams.value());
        for (auto& s : model->out_streams) {
            model->out_buffers.emplace_back(s.get_frame_size());
        }
        // HailoRT 4.10+ enables the network group scheduler by default when
        // using VDevice — do NOT call group->activate() manually. Writing to
        // an input vstream implicitly schedules the network group.

        const double load_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_begin).count();
        std::fprintf(stderr,
            "[hailo] [%s] loaded OK in %.1f ms | inputs=%zu outputs=%zu\n",
            tag, load_ms, model->in_streams.size(), model->out_streams.size());

        for (size_t i = 0; i < model->in_streams.size(); ++i) {
            const auto info = model->in_streams[i].get_info();
            std::fprintf(stderr,
                "[hailo] [%s]   input[%zu] name=%s shape=%s frame_size=%zu\n",
                tag, i, model->in_streams[i].name().c_str(),
                shapeToString(info.shape).c_str(),
                model->in_streams[i].get_frame_size());
        }
        for (size_t i = 0; i < model->out_streams.size(); ++i) {
            const auto info = model->out_streams[i].get_info();
            std::fprintf(stderr,
                "[hailo] [%s]   output[%zu] name=%s shape=%s frame_size=%zu\n",
                tag, i, model->out_streams[i].name().c_str(),
                shapeToString(info.shape).c_str(),
                model->out_streams[i].get_frame_size());
        }
        return model;
    }

    InferenceOutput run(LoadedModel& m, const uint8_t* rgb, size_t rgb_size,
                        const char* tag, Stats& stats) {
        if (m.in_streams.empty() || m.out_streams.empty())
            throw std::runtime_error("Model has no streams");

        const int log_every = envInt("ADAS_LOG_EVERY_N", 0);
        const auto t0 = std::chrono::steady_clock::now();

        auto status = m.in_streams[0].write(
            MemoryView(const_cast<uint8_t*>(rgb), rgb_size));
        if (status != HAILO_SUCCESS)
            throw std::runtime_error("VStream write failed");

        InferenceOutput result;
        result.tensors.resize(m.out_streams.size());
        for (size_t i = 0; i < m.out_streams.size(); ++i) {
            auto& stream = m.out_streams[i];
            auto& buf = m.out_buffers[i];
            status = stream.read(MemoryView(buf.data(), buf.size()));
            if (status != HAILO_SUCCESS)
                throw std::runtime_error("VStream read failed");

            auto& t = result.tensors[i];
            t.name = stream.name();
            auto info = stream.get_info();
            t.shape = {1,
                       static_cast<int>(info.shape.height),
                       static_cast<int>(info.shape.width),
                       static_cast<int>(info.shape.features)};
            const size_t n = buf.size() / sizeof(float);
            t.data.resize(n);
            std::memcpy(t.data.data(), buf.data(), buf.size());
        }

        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        ++stats.frames;
        stats.ms_sum += ms;

        if (log_every > 0 && (stats.frames % static_cast<size_t>(log_every)) == 0) {
            const double secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - stats.window_start).count();
            const double avg_ms = stats.ms_sum / static_cast<double>(log_every);
            const double fps = secs > 0 ? log_every / secs : 0.0;
            std::fprintf(stderr,
                "[hailo] [%s] frame=%zu avg_infer=%.2fms fps=%.1f last=%.2fms\n",
                tag, stats.frames, avg_ms, fps, ms);
            char t0tag[64];
            std::snprintf(t0tag, sizeof(t0tag), "[%s] out[0]", tag);
            logTensorStats(t0tag, m.out_buffers[0]);
            stats.ms_sum = 0.0;
            stats.window_start = std::chrono::steady_clock::now();
        }
        return result;
    }
};
#else
// ---------------------------------------------------------------------------
// Stub backend for dev machines without HailoRT. Returns empty inference
// output so the rest of the pipeline still runs and draws the camera feed.
// ---------------------------------------------------------------------------
struct HailoBackend::Impl {
    bool deeplab_loaded = false;
    bool yolo_loaded = false;
};
#endif  // ADAS_USE_HAILO

HailoBackend::HailoBackend() : impl_(std::make_unique<Impl>()) {}
HailoBackend::~HailoBackend() = default;

void HailoBackend::loadDeeplab(const std::string& hef_path) {
    if (hef_path.empty()) return;
#if ADAS_USE_HAILO
    impl_->deeplab = impl_->loadHef(hef_path, "deeplab");
#else
    impl_->deeplab_loaded = true;
    std::cerr << "[stub] loadDeeplab(" << hef_path
              << "): built without HailoRT — inference will return empty tensors\n";
#endif
}

void HailoBackend::loadYolo26s(const std::string& hef_path) {
    if (hef_path.empty()) return;
#if ADAS_USE_HAILO
    impl_->yolo = impl_->loadHef(hef_path, "yolo");
#else
    impl_->yolo_loaded = true;
    std::cerr << "[stub] loadYolo26s(" << hef_path
              << "): built without HailoRT — inference will return empty tensors\n";
#endif
}

bool HailoBackend::hasModel(ModelId model) const {
#if ADAS_USE_HAILO
    return (model == ModelId::Deeplab) ? impl_->deeplab != nullptr
                                       : impl_->yolo    != nullptr;
#else
    return (model == ModelId::Deeplab) ? impl_->deeplab_loaded
                                       : impl_->yolo_loaded;
#endif
}

InferenceOutput HailoBackend::infer(ModelId model,
                                    const uint8_t* rgb_uint8,
                                    int width,
                                    int height) {
    const size_t nbytes = static_cast<size_t>(width) * height * 3;
#if ADAS_USE_HAILO
    const bool is_deeplab = (model == ModelId::Deeplab);
    LoadedModel* m = is_deeplab ? impl_->deeplab.get() : impl_->yolo.get();
    if (!m) throw std::runtime_error("Model not loaded");
    const char* tag = is_deeplab ? "deeplab" : "yolo";
    Impl::Stats& stats = is_deeplab ? impl_->deeplab_stats : impl_->yolo_stats;
    return impl_->run(*m, rgb_uint8, nbytes, tag, stats);
#else
    (void)model; (void)rgb_uint8; (void)nbytes;
    return {};
#endif
}

}  // namespace adas
