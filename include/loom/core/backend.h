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
    // Backends BETWEEN `primary` and the CPU: accelerators whose tensors live in host memory, so that
    // a node the primary cannot run has somewhere better than the CPU to land (BACKLOG.md P4.8b).
    //
    // The reason this is worth having is narrower than it looks, and worth stating so it is not
    // over-applied. A host accelerator does NOT take work away from the primary: BLAS implements
    // roughly MUL_MAT/OUT_PROD, a strict subset of what every GPU backend does, so with a GPU primary
    // it can claim nothing the GPU had not already claimed. What it does is improve the FALLBACK --
    // when a node drops out of the primary its data has to reach host memory anyway, and once it is
    // there, running a large matmul through OpenBLAS beats running it through the plain CPU kernel at
    // no extra transfer. That predicts where this pays: a primary with thin op coverage, i.e. an NPU.
    //
    // Only host-memory accelerators belong here, never a second discrete device. ggml has no general
    // peer-to-peer path, so a copy between two discrete backends goes through host memory in both
    // directions -- four transfers where falling back to the CPU costs two.
    std::vector<ggml_backend_t> assists;

    Backends() = default;
    // Implicit on purpose. Every call site that passed a bare `ggml_backend_t` before this existed --
    // which is every test in the suite, and every embedding host -- keeps compiling and keeps meaning
    // exactly what it meant: one backend, no scheduler, no fallback.
    Backends(ggml_backend_t backend) : primary(backend) {}
    Backends(ggml_backend_t compute_backend, ggml_backend_t cpu_fallback)
        : primary(compute_backend), fallback(cpu_fallback) {}
    Backends(ggml_backend_t compute_backend, std::vector<ggml_backend_t> assist_backends,
             ggml_backend_t cpu_fallback)
        : primary(compute_backend), fallback(cpu_fallback), assists(std::move(assist_backends)) {}

    // Whether a graph built against this needs `ggml_backend_sched` rather than a plain gallocr. The
    // `fallback != primary` half matters: `Device::open("cpu")` resolves both to the same CPU backend
    // rather than special-casing itself, and scheduling the CPU against itself is pure overhead.
    bool hybrid() const { return fallback != nullptr && fallback != primary; }

    // Exactly what `ggml_backend_sched_new` is handed, in the order it is handed: the primary first,
    // then any assists, and the CPU LAST -- which ggml asserts, because its split planner treats the
    // final backend as the one able to run anything. Nulls and duplicates are dropped here so that no
    // caller has to think about either.
    std::vector<ggml_backend_t> schedule_order() const {
        std::vector<ggml_backend_t> order;
        order.reserve(assists.size() + 2);
        if (primary != nullptr) order.push_back(primary);
        for (ggml_backend_t assist : assists) {
            if (assist != nullptr && assist != primary && assist != fallback) order.push_back(assist);
        }
        if (hybrid()) order.push_back(fallback);
        return order;
    }
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

// Tell the engine where to find dynamically loaded backends, for a build configured with
// GGML_BACKEND_DL (where every backend, the CPU included, is a .so found at run time rather than linked).
//
// ggml searches the executable's directory and the current directory by default. That is right for
// loom_cli and wrong for an embedded host: inside a Python interpreter the executable is `python`, and
// the current directory belongs to the caller. A host that ships backends alongside itself passes their
// directory here, before the first Device::open or available_devices() -- or after, which also works,
// since the next call sweeps whatever has been added since.
//
// Directories added here are searched BEFORE ggml's own defaults, and adding the same one twice is
// harmless. Has no effect on a build with backends linked in: those register themselves.
void add_backend_search_path(const std::string& dir);

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
    //   "auto"  (or empty, with no LOOM_DEVICE) -- the best device present, ranked by WHAT IT IS and
    //                                             never by the order ggml registered it (see below).
    //                                             On a build with no device backend compiled in the
    //                                             only candidate is the CPU, so this behaves exactly
    //                                             as it did before any of this existed.
    //   "cpu"                                   -- the CPU, whatever else is available.
    //   "gpu"                                   -- an offload device with its own memory, preferring
    //                                             one the kernel confirms is a GPU; THROWS if there is
    //                                             none, because a caller who asked for one specifically
    //                                             is better served by an error than by a CPU run they
    //                                             did not ask for and cannot see.
    //   "npu" (or "accel")                      -- ALWAYS THROWS, with a message explaining why and
    //                                             what to type instead. ggml does not report NPU
    //                                             identity; see below.
    //   a device name ("Vulkan0", "CUDA0", ...) -- that device, matched case-insensitively against
    //                                             available_devices(); throws if absent.
    //
    // THE RANKING, and why it is not "the first non-CPU device" (BACKLOG.md P4.8b, P4.8e):
    //
    //   0  an offload device with its own memory  (a split against it costs a copy)
    //   1  an offload device in host memory       (BLAS; a split against it copies nothing)
    //   2  the CPU
    //
    // "the first non-CPU device registered" was the rule until 2026-08-14, and it is not a stable
    // notion: ggml registers backends in one order when they are linked in and a DIFFERENT order when
    // they are loaded dynamically. The same source on the same machine resolved "gpu" to Vulkan0 in a
    // linked build and to BLAS in a GGML_BACKEND_DL build -- and DL is what the Python wheels ship. A
    // ranking by what the device IS removes registration order from the answer entirely.
    //
    // WHY THERE IS NO NPU TIER, which is the question this ranking gets asked most: ggml's device enum
    // documents ACCEL as "accelerator devices intended to be used together with the CPU backend (e.g.
    // BLAS or AMX)" -- rank 1 here -- and GPU as "GPU device using dedicated memory". By that taxonomy
    // a discrete NPU IS a GPU, and all three of ggml's accelerator backends agree: ggml-openvino,
    // ggml-hexagon and ggml-et all return GGML_BACKEND_DEVICE_TYPE_GPU unconditionally. An NPU is
    // therefore not distinguishable from a GPU through this API, which is why "npu" throws rather than
    // guessing, and why a caller who knows their machine should name the device.
    //
    // Ties WITHIN a rank are broken in two steps, in this order:
    //
    //   1. Ask the KERNEL, and only when it can answer: a device reporting a PCI address
    //      (ggml_backend_dev_props::device_id) whose sysfs class is 0x03 is a display controller and
    //      wins. This is a promotion on positive evidence and never a demotion on its absence --
    //      ggml-metal, ggml-sycl, ggml-opencl, ggml-webgpu and ggml-cann are real GPU backends that
    //      report no address at all, and there is no sysfs off Linux.
    //   2. Then prefer a DISCRETE GPU over an INTEGRATED one, from ggml's own device type. On a
    //      machine with an iGPU and a discrete card the first step ties -- both are class 0x03 -- and
    //      registration order was picking the iGPU while the discrete card sat idle (BACKLOG.md P4.8j).
    //
    // The kernel check has to come FIRST. A type-first key would rank a backend that merely CLAIMS to
    // be a GPU -- ggml-openvino does, while driving an NPU or a CPU -- above a genuine iGPU the kernel
    // has vouched for.
    //
    // Where nothing separates two devices, registration order still decides and a caller with two such
    // devices should name the one it wants.
    //
    // Throws loom::Error on an unresolvable spec or a device that fails to initialize.
    static Device open(const std::string& spec = "");

    Backends backends() const {
        std::vector<ggml_backend_t> assist_handles;
        assist_handles.reserve(assists_.size());
        for (const ggml_backend_ptr& assist : assists_) assist_handles.push_back(assist.get());
        return {primary_.get(), std::move(assist_handles), fallback_.get()};
    }
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
    // Host-memory accelerators, held only when `primary_` is a device with its own memory. See
    // Backends::assists for why they are worth carrying and why a second discrete device is not.
    std::vector<ggml_backend_ptr> assists_;
    std::string name_;
    std::string description_;
};

} // namespace loom
