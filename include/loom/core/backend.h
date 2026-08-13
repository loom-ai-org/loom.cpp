#pragma once

#include <ggml-backend.h>
#include <ggml-cpp.h>

#include <cstddef>
#include <string>
#include <vector>

namespace loom {

// ---------------------------------------------------------------------------------------------------
// WHICH DEVICE A GRAPH RUNS ON (BACKLOG.md P4.7)
// ---------------------------------------------------------------------------------------------------
// Everything below this line in the engine takes a `Backends`, not a `ggml_backend_t`, and the whole
// difference between the two is the second handle: where the ops the chosen device CANNOT run go.
//
// A device backend is never complete. Vulkan has no CUSTOM op at all, so the five host callbacks this
// engine registers through ggml_map_custom (RSQRT, ATAN, ATAN2, POW, SHAPE -- see primitives_basic.cpp
// for why each has no native ggml counterpart) can only ever run on a CPU, and a Kokoro or a VITS graph
// contains them by the hundred. That is not a gap to be closed by writing more shaders: `ggml_map_custom`
// takes a C function pointer, so there is nothing for a GPU backend to dispatch. The engine therefore
// carries a CPU backend alongside the device one whenever a device one is in play, and hands the pair to
// `ggml_backend_sched`, which is exactly the component that decides per node which of the two runs it.
//
// A CPU-only `Backends` (`fallback == nullptr`) is the pre-P4.7 engine unchanged, down to the allocator:
// GraphBuilder keeps its plain `ggml_gallocr` for that case rather than routing a single-backend graph
// through a scheduler that would have nothing to schedule. See graph_builder.h.
struct Backends {
    // What weights, declared inputs, the KV/conv caches and retained outputs are allocated on, and what
    // runs every op it supports. Never null in a usable Backends.
    ggml_backend_t primary = nullptr;
    // The CPU, when `primary` is not already one; null otherwise. Non-owning, like `primary`: a Device
    // (or a test's own ggml_backend_ptr) owns both, and must outlive everything holding this.
    ggml_backend_t fallback = nullptr;

    Backends() = default;
    // Implicit on purpose. Every call site that passed a bare `ggml_backend_t` before this existed --
    // which is every test in the suite, and every embedding host -- keeps compiling and keeps meaning
    // exactly what it meant: one backend, no scheduler, no fallback.
    Backends(ggml_backend_t backend) : primary(backend) {}
    Backends(ggml_backend_t compute_backend, ggml_backend_t cpu_fallback)
        : primary(compute_backend), fallback(cpu_fallback) {}

    // Whether a graph built against this needs `ggml_backend_sched` rather than a plain gallocr. The
    // `fallback != primary` half matters: `Device::open("cpu")` resolves both to the same CPU backend
    // rather than special-casing itself, and scheduling the CPU against itself is pure overhead.
    bool hybrid() const { return fallback != nullptr && fallback != primary; }
};

// One entry per device the LINKED-IN ggml backends report -- the CPU is always among them. What
// `loom_cli --list-devices` prints, and what a host needs to offer a choice.
struct DeviceInfo {
    std::string name;        // ggml's own device name, and what Device::open() accepts: "CPU", "Vulkan0", ...
    std::string description; // human-readable, e.g. "AMD Radeon Vega 3 Graphics (RADV RAVEN2)"
    bool is_cpu = false;
    // ggml reports these only for devices that track them; both are 0 when it does not.
    size_t memory_free = 0;
    size_t memory_total = 0;
};

// Every registered device, in ggml's own registration order. Safe to call before any Device exists --
// it is what initializes the backend registry.
std::vector<DeviceInfo> available_devices();

// Owns the backends a `Backends` points at. One per host process is the intended shape: the backends it
// initializes are process-wide resources (a Vulkan device and queue, a CPU thread pool), and every
// GgufModel, cache and GraphBuilder built against it holds its handles without owning them, so it must
// outlive all of them.
class Device {
public:
    // `spec` is resolved in the order this project uses everywhere for something the machine can answer
    // for itself: the explicit argument first, then the `LOOM_DEVICE` environment variable, then
    // autodetection. The accepted spellings are:
    //
    //   "auto"  (or empty, with no LOOM_DEVICE) -- the first GPU/iGPU/accelerator device registered,
    //                                             else the CPU. On a build with no device backend
    //                                             compiled in there IS no such device, so this is the
    //                                             CPU and the engine behaves exactly as it did before
    //                                             this existed.
    //   "cpu"                                   -- the CPU, whatever else is available.
    //   "gpu"                                   -- the first GPU/iGPU/accelerator; THROWS if there is
    //                                             none, because a caller who asked for one specifically
    //                                             is better served by an error than by a CPU run they
    //                                             did not ask for and cannot see.
    //   a device name ("Vulkan0", "CUDA0", ...) -- that device, matched case-insensitively against
    //                                             available_devices(); throws if absent.
    //
    // Throws loom::Error on an unresolvable spec or a device that fails to initialize.
    static Device open(const std::string& spec = "");

    Backends backends() const { return {primary_.get(), fallback_.get()}; }
    // The selected device's ggml name and human-readable description -- what a host prints to say what
    // it is actually running on.
    const std::string& name() const { return name_; }
    const std::string& description() const { return description_; }
    bool is_cpu() const { return fallback_ == nullptr; }

    Device(Device&&) = default;
    Device& operator=(Device&&) = default;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

private:
    Device() = default;

    ggml_backend_ptr primary_;
    // Held only when `primary_` is a device backend. Null for a CPU Device, which is what is_cpu() reads
    // and what keeps `backends().hybrid()` false there.
    ggml_backend_ptr fallback_;
    std::string name_;
    std::string description_;
};

} // namespace loom
