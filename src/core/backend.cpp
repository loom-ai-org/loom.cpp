#include "loom/core/backend.h"
#include "loom/loom_errors.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <mutex>

namespace loom {
namespace {

// Function-local statics rather than namespace-scope objects, and the difference is load-bearing: a
// host may call add_backend_search_path() from a STATIC INITIALISER (tests/support/cpu_backend.h does
// exactly that, so the registry is populated before any test's main runs). A namespace-scope
// std::vector here would then be read before its own dynamic initialiser had run, across translation
// units, which is the static initialisation order fiasco and is undefined. A function-local static is
// guaranteed initialised on first use, whenever that turns out to be.
std::mutex& loader_mutex() {
    static std::mutex m;
    return m;
}
// Directories a host has declared and that have not been swept yet. Emptied by ensure_backends_loaded
// rather than kept, because ggml's registry -- not this list -- is the record of what got loaded.
std::vector<std::string>& pending_search_paths() {
    static std::vector<std::string> paths;
    return paths;
}
bool& default_swept() {
    static bool swept = false;
    return swept;
}

// ggml's dynamic-backend loader. A statically-linked backend registers itself from its own translation
// unit and needs nothing from us; all of this is for a build configured with GGML_BACKEND_DL, where the
// backends are .so files found at run time.
//
// ggml's own search looks in the executable's directory and the current directory, which is the right
// default for a CLI and the wrong one for every embedded host: inside a Python interpreter the
// "executable directory" is wherever `python` was installed, and the current directory is wherever the
// user happened to be. So a host that knows where its backends are says so through
// add_backend_search_path(), and those directories are swept BEFORE ggml's defaults -- if the same
// backend exists in both, whichever registers first is what "auto" and "gpu" resolve to, and a host
// that shipped its own copy meant that one.
//
// Sweeping is repeatable rather than once-only: a host may add a directory after a Device already
// exists (loom-py discovers its accelerator packages lazily). Re-loading is safe because ggml dedupes
// on the registration pointer -- ggml_backend_registry::register_backend returns early for a reg it
// already holds, and dlopen hands back the same handle for a path already open, so a directory swept
// twice registers nothing twice.
void ensure_backends_loaded() {
    std::lock_guard<std::mutex> lock(loader_mutex());
    for (const std::string& dir : pending_search_paths()) {
        ggml_backend_load_all_from_path(dir.c_str());
    }
    pending_search_paths().clear();
    if (!default_swept()) {
        ggml_backend_load_all();
        default_swept() = true;
    }
}

std::string lowered(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

// Whether the device's tensors live in ordinary host memory -- which decides whether a graph split
// against it costs a copy at all.
//
// Note this asks the DEFAULT BUFFER TYPE, not `ggml_backend_dev_props::caps.host_buffer`. The
// similarly-named field means "can hand out pinned host buffers for staging" and is the OPPOSITE of
// what is wanted here: measured on this machine it is true for Vulkan and false for BLAS.
//
// It is also a question about address spaces rather than about packaging. An iGPU on UMA hardware
// shares physical RAM with the CPU and still answers false, because its ggml buffer type is a device
// buffer -- so a split against it is a memcpy within RAM: cheap, but not free. That is the right
// answer for the thing this is used to decide.
bool is_host_memory(ggml_backend_dev_t dev) {
    return ggml_backend_buft_is_host(ggml_backend_dev_buffer_type(dev));
}

// HOW "auto" RANKS DEVICES (BACKLOG.md P4.8b). Lower is preferred; 4 is never selected.
//
// This exists because "the first non-CPU device the registry reports" -- what this file used to do --
// **is not a stable notion**. ggml registers in two different orders depending on how the binary was
// linked: a linked build follows the `#ifdef` sequence in ggml_backend_registry's constructor, and a
// GGML_BACKEND_DL build follows the call sequence of ggml_backend_load_all, where `blas` comes FIRST
// and `vulkan` ninth. Measured on one machine with one source tree, `Device::open("gpu")` returned
// Vulkan0 linked and BLAS dynamically loaded. Since DL is what the wheels ship, that divergence is
// not hypothetical.
//
// The ranking is by what the device IS, so registration order stops being an input:
//
//   0  a GPU or iGPU
//   1  an accelerator with its own memory -- what a discrete NPU registers as
//   2  an accelerator in host memory -- what BLAS is, and what an NPU on a UMA SoC may be
//   3  the CPU
//
// GPUs ahead of accelerators is a judgement rather than a law, and the evidence for it is P4.7d's
// support matrix: the NPU-shaped backends implement strictly FEWER ops than the GPU ones -- OpenVINO
// and Hexagon have no POOL_2D, which every GPU backend has -- so more of a graph survives on a GPU.
// Revisit it when an NPU measures better on a real graph; `"npu"` is the override until then.
int primary_rank(ggml_backend_dev_t dev) {
    switch (ggml_backend_dev_type(dev)) {
        case GGML_BACKEND_DEVICE_TYPE_GPU:
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return 0;
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return is_host_memory(dev) ? 2 : 1;
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return 3;
        default:
            return 4;
    }
}

// The best-ranked device, or null if the registry holds nothing selectable. Ties -- two GPUs, or a
// GPU and a second GPU from another backend -- are still broken by registration order, which is the
// half of this problem that is NOT solved here: nothing about a machine tells the engine that CUDA0
// should beat Vulkan0. A caller with two devices of the same kind should name the one it wants.
ggml_backend_dev_t best_device(int worst_rank_allowed) {
    ggml_backend_dev_t best = nullptr;
    int best_rank = worst_rank_allowed + 1;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const int rank = primary_rank(dev);
        if (rank <= worst_rank_allowed && rank < best_rank) {
            best = dev;
            best_rank = rank;
        }
    }
    return best;
}

// The first device matching one rank exactly -- what the specs that name a KIND of device resolve
// through, so that "gpu" cannot answer with an accelerator and "npu" cannot answer with a GPU.
ggml_backend_dev_t first_device_of_rank(int wanted) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (primary_rank(dev) == wanted) return dev;
    }
    return nullptr;
}

std::string device_list_for_error();

ggml_backend_dev_t cpu_device() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == nullptr) {
        // Reachable, and for one reason worth naming in the message. A `GGML_BACKEND_DL` build links no
        // backend at all: each is a shared library discovered at run time beside the executable, in
        // `GGML_BACKEND_DIR`, or at `$GGML_BACKEND_PATH`. When none is found the registry is EMPTY --
        // there is no CPU to fall back to, because the CPU is a plugin too -- and every spec including
        // "cpu" and "auto" arrives here. "ggml reports no CPU device" is true and useless; a deployment
        // that forgot to ship its backends needs to be told that is what happened (BACKLOG.md P4.8).
        if (ggml_backend_dev_count() == 0) {
            throw Error("loom::Device: no ggml backends are available at all. A GGML_BACKEND_DL build "
                        "loads them as shared libraries at run time -- put the ggml-*.so/.dll files "
                        "beside the executable, or point $GGML_BACKEND_PATH at one.");
        }
        throw Error("loom::Device: ggml reports no CPU device (devices: [" + device_list_for_error() +
                    "])");
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

void add_backend_search_path(const std::string& dir) {
    // An empty path is not a directory, and ggml would resolve it to one anyway -- it joins the search
    // path with the filename, so "" means the current directory rather than nothing. Dropping it here
    // makes an unset or blank entry in a host's own configuration (a trailing separator in
    // $LOOM_BACKEND_DIR, most likely) mean what it reads as.
    if (dir.empty()) return;
    std::lock_guard<std::mutex> lock(loader_mutex());
    pending_search_paths().push_back(dir);
}

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
        // A GPU or an iGPU, and NOT merely "something that is not the CPU". Answering this with an
        // accelerator is how a machine's actual GPU got silently skipped in favour of BLAS, which
        // implements roughly MUL_MAT and would have run the rest of the graph on the CPU while
        // reporting an accelerator to the caller (BACKLOG.md P4.8b).
        dev = first_device_of_rank(0);
        // Deliberately an error rather than a fallback. "auto" already means "the best you have"; a
        // caller who spelled out "gpu" is asking a question about the machine, and answering it with a
        // silent CPU run turns "there is no GPU here" into an unexplained performance number.
        if (dev == nullptr) {
            throw Error("loom::Device: no GPU device is available -- this build has devices [" +
                        device_list_for_error() + "]. Configure with -DGGML_VULKAN=ON (or "
                        "-DGGML_CUDA=ON / -DGGML_METAL=ON) to compile one in, use 'npu' for an "
                        "accelerator, or use 'auto'.");
        }
    } else if (key == "npu" || key == "accel") {
        // An accelerator with its own memory. The spelling is "npu" because that is what such a device
        // is called outside this file; ggml has no NPU concept and reports one as ACCEL, which BLAS
        // also is -- so the memory question is what separates them, not the type.
        dev = first_device_of_rank(1);
        if (dev == nullptr) {
            throw Error("loom::Device: no NPU/accelerator device with its own memory is available -- "
                        "this build has devices [" + device_list_for_error() + "]. Note that a "
                        "host-memory accelerator such as BLAS is not one; name it directly if that is "
                        "what you meant, or use 'auto'.");
        }
    } else if (key == "auto") {
        // Every rank in preference order, so this cannot fail: the CPU is rank 3 and is always there.
        dev = best_device(/*worst_rank_allowed=*/3);
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
                        device_list_for_error() + "], or one of 'auto', 'cpu', 'gpu', 'npu'");
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

        // Host-memory accelerators join the chain between the primary and the CPU, but ONLY when the
        // primary has its own memory. Two reasons for that condition, both in Backends::assists: a
        // host accelerator improves the fallback rather than the primary, so pairing it with a primary
        // that is itself in host memory buys nothing; and a discrete device must never fall back to
        // another discrete device, which the rank-2 filter below also rules out.
        //
        // A failure to initialize one is not fatal. An assist is an optimization -- the graph is
        // correct without it, because the CPU can run everything -- so a backend that declines to
        // start is skipped rather than taking the whole Device down with it.
        if (primary_rank(dev) <= 1) {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t candidate = ggml_backend_dev_get(i);
                if (candidate == dev || primary_rank(candidate) != 2) continue;
                ggml_backend_ptr assist(ggml_backend_dev_init(candidate, nullptr));
                if (assist) device.assists_.push_back(std::move(assist));
            }
        }
    }
    return device;
}

} // namespace loom
