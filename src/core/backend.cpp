#include "loom/core/backend.h"
#include "loom/loom_errors.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <mutex>

namespace loom {
namespace {

// ggml's dynamic-backend loader, called once. A statically-linked backend registers itself from its own
// translation unit and needs nothing from us; this is only for a build configured with GGML_BACKEND_DL,
// where the backends are .so files discovered next to the executable. Calling it costs a directory scan
// on the first Device::open of the process and nothing thereafter.
void ensure_backends_loaded() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_backend_load_all(); });
}

std::string lowered(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

// Every device type that is not the CPU and is not ggml's "META" aggregate -- i.e. everything worth
// resolving "gpu"/"auto" to. ACCEL is in here because an NPU/accelerator registers as one and the
// engine's reason for wanting a non-CPU device does not distinguish it from a GPU (the roadmap item is
// "GPUs and NPUs", singular in every respect that reaches this file).
bool is_offload_device(ggml_backend_dev_t dev) {
    switch (ggml_backend_dev_type(dev)) {
        case GGML_BACKEND_DEVICE_TYPE_GPU:
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return true;
        default:
            return false;
    }
}

ggml_backend_dev_t first_offload_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (is_offload_device(dev)) return dev;
    }
    return nullptr;
}

ggml_backend_dev_t cpu_device() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == nullptr) {
        // Not reachable in any build this project produces -- ggml always links its CPU backend -- but a
        // null here would otherwise surface as a crash inside ggml_backend_dev_init.
        throw Error("loom::Device: ggml reports no CPU device");
    }
    return dev;
}

std::string device_list_for_error() {
    std::string names;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (i > 0) names += ", ";
        names += ggml_backend_dev_name(ggml_backend_dev_get(i));
    }
    return names;
}

} // namespace

std::vector<DeviceInfo> available_devices() {
    ensure_backends_loaded();
    std::vector<DeviceInfo> out;
    out.reserve(ggml_backend_dev_count());
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        DeviceInfo info;
        info.name = ggml_backend_dev_name(dev);
        info.description = ggml_backend_dev_description(dev);
        info.is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
        ggml_backend_dev_memory(dev, &info.memory_free, &info.memory_total);
        out.push_back(std::move(info));
    }
    return out;
}

Device Device::open(const std::string& spec) {
    ensure_backends_loaded();

    // Explicit argument, else the environment, else autodetection -- the resolution order this project
    // uses for anything the machine can answer for itself. An empty LOOM_DEVICE counts as unset, so
    // exporting it blank is not a way to get an error.
    std::string requested = spec;
    if (requested.empty()) {
        const char* env = std::getenv("LOOM_DEVICE");
        if (env != nullptr && *env != '\0') requested = env;
    }
    if (requested.empty()) requested = "auto";
    const std::string key = lowered(requested);

    ggml_backend_dev_t dev = nullptr;
    if (key == "cpu") {
        dev = cpu_device();
    } else if (key == "gpu") {
        dev = first_offload_device();
        // Deliberately an error rather than a fallback. "auto" already means "the best you have"; a
        // caller who spelled out "gpu" is asking a question about the machine, and answering it with a
        // silent CPU run turns "there is no GPU here" into an unexplained performance number.
        if (dev == nullptr) {
            throw Error("loom::Device: no GPU/accelerator device is available -- this build has "
                        "devices [" + device_list_for_error() + "]. Configure with -DGGML_VULKAN=ON "
                        "(or -DGGML_CUDA=ON / -DGGML_METAL=ON) to compile one in, or use 'auto'.");
        }
    } else if (key == "auto") {
        dev = first_offload_device();
        if (dev == nullptr) dev = cpu_device();
    } else {
        // A device name. Matched case-insensitively against the registry rather than through
        // ggml_backend_dev_by_name, which is exact-match only -- "vulkan0" is what a person types.
        for (size_t i = 0; i < ggml_backend_dev_count() && dev == nullptr; ++i) {
            ggml_backend_dev_t candidate = ggml_backend_dev_get(i);
            if (lowered(ggml_backend_dev_name(candidate)) == key) dev = candidate;
        }
        if (dev == nullptr) {
            throw Error("loom::Device: unknown device '" + requested + "' -- available devices are [" +
                        device_list_for_error() + "], or one of 'auto', 'cpu', 'gpu'");
        }
    }

    Device device;
    device.primary_.reset(ggml_backend_dev_init(dev, nullptr));
    if (!device.primary_) {
        throw Error("loom::Device: device '" + std::string(ggml_backend_dev_name(dev)) +
                    "' failed to initialize");
    }
    device.name_ = ggml_backend_dev_name(dev);
    device.description_ = ggml_backend_dev_description(dev);

    // The CPU comes along whenever the primary is not one, and it is not optional: see backend.h for why
    // a device backend can never run this engine's ggml_map_custom nodes. ggml_backend_sched additionally
    // REQUIRES that the last backend it is given be a CPU one, so there is no hybrid arrangement in which
    // this is absent.
    if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        device.fallback_.reset(ggml_backend_dev_init(cpu_device(), nullptr));
        if (!device.fallback_) {
            throw Error("loom::Device: the CPU fallback backend failed to initialize");
        }
    }
    return device;
}

} // namespace loom
