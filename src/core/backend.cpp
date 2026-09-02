#include "loom/core/backend.h"
#include "loom/loom_errors.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <tuple>
#include <utility>

#ifdef __linux__
#include <dirent.h>
#include <sched.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

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

// HOW "auto" RANKS DEVICES (BACKLOG.md P4.8b, corrected by P4.8e). Lower is preferred; 3 is never
// selected.
//
// This exists because "the first non-CPU device the registry reports" -- what this file used to do --
// **is not a stable notion**. ggml registers in two different orders depending on how the binary was
// linked: a linked build follows the `#ifdef` sequence in ggml_backend_registry's constructor, and a
// GGML_BACKEND_DL build follows the call sequence of ggml_backend_load_all, where `blas` comes FIRST
// and `vulkan` ninth. Measured on one machine with one source tree, `Device::open("gpu")` returned
// Vulkan0 linked and BLAS dynamically loaded. Since DL is what the wheels ship, that divergence is
// not hypothetical.
//
// The ranking is by the one thing about a device that has a behavioural consequence -- where its
// tensors live -- so registration order stops being an input:
//
//   0  an offload device with its own memory  (a split against it costs a copy)
//   1  an offload device in host memory       (BLAS; a split against it copies nothing)
//   2  the CPU
//
// THERE USED TO BE A FOURTH TIER, and deleting it is P4.8e. Rank 1 was "an accelerator with its own
// memory -- what a discrete NPU registers as", sitting between GPUs and BLAS. It never had an
// inhabitant and never could have, because it was built on a misreading of ggml's own enum:
//
//     // accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
//     GGML_BACKEND_DEVICE_TYPE_ACCEL,
//
// ACCEL is DEFINED upstream as the BLAS/AMX co-processor role -- a thing used *together with* the CPU,
// which is this file's rank 1 -- and GPU is defined as "GPU device using dedicated memory". So a
// discrete NPU that runs whole graphs out of its own memory is, by upstream's taxonomy, a GPU. All
// three accelerator backends in ggml agree and return GGML_BACKEND_DEVICE_TYPE_GPU: ggml-openvino
// (:751), ggml-hexagon (:3917), and ggml-et. Verified unchanged at ggml master 8846b79e, 154 commits
// past the pinned v0.16.0, so this is upstream's position rather than an immaturity to wait out.
//
// The consequence for this file is small precisely because the deleted tier had no members: a device
// is an offload device or it is the CPU, and what separates the two offload ranks is the memory
// question that was already being asked.
int primary_rank(ggml_backend_dev_t dev) {
    switch (ggml_backend_dev_type(dev)) {
        case GGML_BACKEND_DEVICE_TYPE_GPU:
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return is_host_memory(dev) ? 1 : 0;
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return 2;
        default:
            return 3;
    }
}

// Can the KERNEL confirm this device is a GPU? Breaks ties within a rank, and is the only positive
// evidence available for the question `ggml_backend_dev_type` stopped being able to answer.
//
// Two stages, and neither trusts the backend's self-report:
//
//   1. `ggml_backend_dev_props::device_id` is the device's PCI address, or null. Null is RELIABLE
//      rather than uninitialised -- ggml_backend_dev_get_props memsets the struct before handing it to
//      the backend -- so a null means "this backend did not say which physical device it is".
//   2. The kernel then says what that address IS. PCI base class 0x03 is a display controller; the
//      Intel NPU on the workstation is 0x12, a processing accelerator, at 0000:00:0b.0. The kernel
//      maintains exactly the taxonomy ggml's backends have stopped maintaining, and it is the one
//      authority in this stack that is not a self-report.
//
// WHAT THIS DELIBERATELY DOES NOT DO is infer "not confirmed" => "not a GPU". Only ggml-cuda and
// ggml-vulkan populate device_id at all; ggml-metal, ggml-sycl, ggml-opencl, ggml-webgpu and ggml-cann
// are real GPU backends that leave it null, and there is no sysfs at all on macOS or Windows. So this
// is a POSITIVE confirmation that promotes, never a negative one that demotes: an unconfirmable device
// keeps exactly the standing it had before this function existed.
//
// What it buys is the case the workstation actually presents -- CUDA0 (confirmed 0x030000) against
// OPENVINO0 (says nothing), where the old rule picked whichever ggml registered first and could hand a
// caller an NPU-or-CPU-backed OpenVINO while a 5090 sat idle.
bool kernel_confirms_gpu(ggml_backend_dev_t dev) {
#ifdef __linux__
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    if (props.device_id == nullptr || *props.device_id == '\0') return false;

    const std::string path = std::string("/sys/bus/pci/devices/") + props.device_id + "/class";
    std::ifstream f(path);
    if (!f) return false;
    std::string cls;
    if (!(f >> cls)) return false;

    // "0x030000" -- base class is the byte after the "0x". Anything shorter is not a class code.
    if (cls.size() < 4 || cls.compare(0, 2, "0x") != 0) return false;
    return cls.compare(2, 2, "03") == 0;
#else
    (void) dev;
    return false;
#endif
}

// The accelerators the KERNEL knows about, as "accel0 (intel_vpu)", or empty. Diagnostic only -- it is
// never an input to selection, because it answers "does this machine have an accelerator" and
// selection needs "is THIS ggml device one", which nothing can answer (see the "npu" spec).
//
// /dev/accel is the Linux accel subsystem (drivers/accel: intel_vpu, habanalabs, qaic) and a GPU never
// appears there -- on the workstation accel0 is intel_vpu while both GPUs are /dev/dri/card{0,1}. So
// its presence is good evidence that a user who typed "npu" was not confused, and that is exactly the
// user this string is written for.
std::string kernel_accelerator_list() {
#ifdef __linux__
    std::string out;
    DIR* dir = opendir("/sys/class/accel");
    if (dir == nullptr) return out;
    std::vector<std::string> entries;
    while (dirent* e = readdir(dir)) {
        const std::string name = e->d_name;
        if (name == "." || name == "..") continue;
        std::string driver;
        char buf[256];
        const std::string link = "/sys/class/accel/" + name + "/device/driver";
        const ssize_t n = readlink(link.c_str(), buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';
            const std::string target(buf);
            const size_t slash = target.find_last_of('/');
            driver = slash == std::string::npos ? target : target.substr(slash + 1);
        }
        entries.push_back(driver.empty() ? name : name + " (" + driver + ")");
    }
    closedir(dir);
    std::sort(entries.begin(), entries.end());
    for (const std::string& e : entries) {
        if (!out.empty()) out += ", ";
        out += e;
    }
    return out;
#else
    return {};
#endif
}

// Within a rank: a kernel-confirmed GPU first, then a discrete one ahead of an integrated one.
// Everything else keeps registration order, which is what a strict `<` on this key preserves -- two
// devices that tie on both parts never displace each other, so the only behaviour this changes is the
// one it can prove.
//
// THE SECOND PART EXISTS BECAUSE P4.8e THREW AWAY SOMETHING IT STILL NEEDED. Collapsing the tiers
// keyed rank on where a device's memory lives and dropped `ggml_backend_dev_type` entirely, which is
// right for deciding RANKS and wrong for ordering within one. Measured on the workstation with two
// Vulkan devices present:
//
//     Vulkan0  IGPU  buft_is_host=false  0000:00:02.0  0x030000   <- Intel Arrow Lake
//     Vulkan1  GPU   buft_is_host=false  0000:02:00.0  0x030000   <- RTX 5090
//
// Neither existing signal separates them: both are non-host, so the rank ties, and both are PCI class
// 0x03, so the kernel confirmation ties. Registration order then chose the iGPU -- and `device="gpu"`
// on that machine ran 8 tokens in the time the 5090 needed for 24 (BACKLOG.md P4.8j).
//
// THE ORDER OF THE TWO PARTS IS LOAD-BEARING, and putting the type first would reintroduce the defect
// the kernel check exists to prevent. `ggml-openvino` reports GPU while driving an NPU or a CPU and
// supplies no `device_id` (P4.8d), so a type-first key would rank it above a genuine, kernel-confirmed
// iGPU. Confirmation first sorts it to (1, 0), behind anything the kernel vouches for.
std::pair<int, int> tie_break(ggml_backend_dev_t dev) {
    const int confirmed = kernel_confirms_gpu(dev) ? 0 : 1;
    int kind;
    switch (ggml_backend_dev_type(dev)) {
        case GGML_BACKEND_DEVICE_TYPE_GPU:  kind = 0; break;
        case GGML_BACKEND_DEVICE_TYPE_IGPU: kind = 1; break;
        default:                            kind = 2; break;  // ACCEL, and anything ggml adds later
    }
    return {confirmed, kind};
}

// The best device whose rank falls in [best_allowed, worst_allowed], or null if there is none.
// `"gpu"` passes a single-rank window; `"auto"` passes all of them.
ggml_backend_dev_t best_device_in_range(int best_allowed, int worst_allowed) {
    ggml_backend_dev_t best = nullptr;
    std::tuple<int, int, int> best_key{worst_allowed + 1, 0, 0};
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const int rank = primary_rank(dev);
        if (rank < best_allowed || rank > worst_allowed) continue;
        const std::pair<int, int> tb = tie_break(dev);
        const std::tuple<int, int, int> key{rank, tb.first, tb.second};
        if (key < best_key) {
            best = dev;
            best_key = key;
        }
    }
    return best;
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


#ifdef __linux__
// A Linux cpu list -- "0-3", "0,2,4", "0-1,4-5" -- appended to `out`. False on anything it cannot
// parse, rather than returning what it managed to read: a half-parsed sibling list would silently
// double a core count, and the caller needs to tell "unknown" apart from an answer.
bool parse_cpu_list(const std::string& s, std::vector<int>& out) {
    size_t i = 0;
    while (i < s.size()) {
        if (std::isdigit(static_cast<unsigned char>(s[i])) == 0) return false;
        long lo = std::strtol(s.c_str() + i, nullptr, 10);
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])) != 0) ++i;
        long hi = lo;
        if (i < s.size() && s[i] == '-') {
            ++i;
            if (i >= s.size() || std::isdigit(static_cast<unsigned char>(s[i])) == 0) return false;
            hi = std::strtol(s.c_str() + i, nullptr, 10);
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])) != 0) ++i;
        }
        if (hi < lo || hi - lo > CPU_SETSIZE) return false;
        for (long c = lo; c <= hi; ++c) out.push_back(static_cast<int>(c));
        if (i < s.size()) {
            if (s[i] != ',') return false;
            ++i;
        }
    }
    return !out.empty();
}
#endif

// The number of PHYSICAL cores this process may run on, or 0 when the machine cannot be asked.
//
// PHYSICAL rather than logical is a measurement rather than a preference, and the two machines it was
// swept on only agree once it is put that way. The Core Ultra 9 285K has 24 cores and no SMT, so every
// logical CPU is a physical one and using all of them is 1.98x on TTS, 2.41x on ASR and 1.18x on the LM
// against ggml's 4. A 2-core-plus-SMT Ryzen 3 3250U wants 2 rather than its 4 logical CPUs: the two
// extra siblings buy nothing on any of the three tasks and COST 1.19x on TTS. "Use every CPU" would be
// right on one of those machines and a regression on the other; "use every physical core" is right on
// both. The sweeps are in Epic-05 SS2.
//
// AFFINITY-aware on Linux, and that is not decoration. A process under `taskset`, or in a cgroup with a
// cpuset, has been told how much machine it has; answering with the whole socket would ignore it and
// oversubscribe exactly the deployment that took the trouble to say so. An SMT sibling group is counted
// at the lowest member THIS PROCESS MAY USE, so a mask covering one thread of a pair still counts that
// core once, and a mask covering both also counts it once.
//
// What this deliberately does not read is a cgroup CPU *quota* (`cpu.max`, docker's `--cpus=2.5`). A
// quota is a bandwidth cap rather than a set of CPUs -- it throttles a thread pool instead of confining
// it, and the right thread count under one is not a function of the quota alone. A host in that world
// sets $LOOM_N_THREADS.
int physical_core_count() {
#if defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return 0;

    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &mask)) cpus.push_back(c);
    }
    if (cpus.empty()) return 0;

    std::vector<int> leaders;
    bool topology_readable = false;
    for (int c : cpus) {
        const std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list";
        std::ifstream f(path);
        std::string line;
        std::vector<int> siblings;
        int leader = c;
        if (f && std::getline(f, line) && parse_cpu_list(line, siblings)) {
            topology_readable = true;
            for (int s : siblings) {
                if (s < leader && std::find(cpus.begin(), cpus.end(), s) != cpus.end()) leader = s;
            }
        }
        // A cpu whose own topology is unreadable counts as its own core rather than vanishing.
        if (std::find(leaders.begin(), leaders.end(), leader) == leaders.end()) leaders.push_back(leader);
    }

    // Nothing under /sys answered at all -- a container without sysfs, most likely. Report "cannot be
    // asked" rather than the logical count dressed up as a physical one: on an SMT part those differ by
    // 2x, in the direction that costs.
    if (!topology_readable) return 0;
    return static_cast<int>(leaders.size());

#elif defined(__APPLE__)
    // Apple Silicon reports its performance cores separately, and those are the ones to count: its
    // E-cores are a slower core DESIGN rather than SMT siblings, and a pool sized to include them runs
    // at the pace of its slowest member. `hw.perflevel0` is the highest-performance level and does not
    // exist on an Intel Mac, where `hw.physicalcpu` is the whole answer.
    int32_t n = 0;
    size_t len = sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &len, nullptr, 0) == 0 && n > 0) return n;
    n = 0;
    len = sizeof(n);
    if (sysctlbyname("hw.physicalcpu", &n, &len, nullptr, 0) == 0 && n > 0) return n;
    return 0;

#else
    // Every platform this project builds and tests on is covered above -- loom-py publishes manylinux
    // and macOS wheels and nothing else. Anywhere else keeps ggml's 4 and can be raised with
    // $LOOM_N_THREADS, which is a better outcome than an untested guess at a topology API.
    return 0;
#endif
}

// The CPU backend's thread count, which nothing in this engine used to set.
//
// `ggml` defaults to `GGML_DEFAULT_N_THREADS` -- **4, whatever the machine has** (`ggml.h:232`) -- so
// loom ran a 24-core workstation on four cores and said nothing about it. As of P4.30b the default is
// this machine's physical core count instead, and `$LOOM_N_THREADS` overrides that.
//
// Set through the REGISTRY's proc address rather than by calling `ggml_backend_cpu_set_n_threads`,
// which lives inside the CPU backend and is therefore unlinkable in the `GGML_BACKEND_DL` build the
// wheels ship (ADR-009) -- the same reason `tests/support/cpu_backend.h` exists. A backend that does
// not export it (every GPU one) is left alone, which is correct: their parallelism is not a host
// thread count.
void apply_cpu_threads(ggml_backend_t backend) {
    if (backend == nullptr) return;
    const int n = default_cpu_thread_count();
    if (n <= 0) return;

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) return;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) return;
    auto set_n_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
    if (set_n_threads != nullptr) set_n_threads(backend, n);
}

} // namespace

int default_cpu_thread_count() {
    // Resolution order is this project's usual one for something the machine can answer for itself
    // (see Device::open's `spec`): the environment first, then autodetection. A non-numeric or
    // non-positive $LOOM_N_THREADS falls through to autodetection rather than to ggml's 4, because
    // "LOOM_N_THREADS=oops" is a typo and not a request for four threads.
    const char* env = std::getenv("LOOM_N_THREADS");
    if (env != nullptr && *env != '\0') {
        const int n = std::atoi(env);
        if (n > 0) return n;
    }
    return physical_core_count();
}

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
        // An offload device with its own memory, preferring one the kernel confirms is a GPU. NOT
        // merely "something that is not the CPU": answering this with a host-memory accelerator is how
        // a machine's actual GPU got silently skipped in favour of BLAS, which implements roughly
        // MUL_MAT and would have run the rest of the graph on the CPU while reporting an accelerator to
        // the caller (BACKLOG.md P4.8b).
        //
        // The spelling stays "gpu" while the guarantee is now "rank 0", and the gap between those two
        // is real: an NPU backend claiming GPU satisfies this spec. That is upstream's taxonomy rather
        // than a defect here (see primary_rank), and kernel_confirms_gpu narrows it as far as anything
        // can -- where a real GPU and an NPU-shaped backend compete, the confirmed one wins.
        dev = best_device_in_range(0, 0);
        // Deliberately an error rather than a fallback. "auto" already means "the best you have"; a
        // caller who spelled out "gpu" is asking a question about the machine, and answering it with a
        // silent CPU run turns "there is no GPU here" into an unexplained performance number.
        if (dev == nullptr) {
            throw Error("loom::Device: no offload device with its own memory is available -- this "
                        "build has devices [" + device_list_for_error() + "]. Configure with "
                        "-DGGML_VULKAN=ON (or -DGGML_CUDA=ON / -DGGML_METAL=ON) to compile one in, or "
                        "use 'auto'.");
        }
    } else if (key == "npu" || key == "accel") {
        // ALWAYS THROWS, and the message is the whole feature (BACKLOG.md P4.8d/P4.8e).
        //
        // This spec used to resolve to rank 1, "an accelerator with its own memory". No such device
        // exists or can: every NPU backend in ggml registers as GPU, so an NPU is indistinguishable
        // from a GPU through the device API -- see primary_rank for the enum comment that makes that
        // upstream's definition rather than an accident.
        //
        // Kept as a recognised spelling rather than deleted, because a caller who types it has a real
        // question and deserves the answer, not "unknown device 'npu'". What that answer costs is one
        // sentence; what silently resolving it to a GPU would cost is the class of defect P4.8b exists
        // to have removed.
        std::string message =
            "loom::Device: '" + requested +
            "' cannot be resolved: ggml does not report NPU identity. Every NPU backend it ships "
            "(OpenVINO, Hexagon, ET) registers as GGML_BACKEND_DEVICE_TYPE_GPU, so an NPU is "
            "indistinguishable from a GPU through the device API. This build has devices [" +
            device_list_for_error() + "]";
        const std::string accelerators = kernel_accelerator_list();
        if (!accelerators.empty()) {
            // The kernel DOES keep the distinction ggml dropped, so if this machine has an accelerator
            // say so -- otherwise the message reads as "you have no NPU" to someone who does.
            message += ", and this machine has a kernel accelerator [" + accelerators +
                       "]. If a listed device drives it, select that device by name (and for OpenVINO "
                       "set GGML_OPENVINO_DEVICE=NPU, which is what chooses its target)";
        }
        throw Error(message + ". Use 'auto', or name a device.");
    } else if (key == "auto") {
        // Every rank in preference order, so this cannot fail: the CPU is rank 2 and is always there.
        dev = best_device_in_range(0, 2);
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
    apply_cpu_threads(device.primary_.get());

    // The CPU comes along whenever the primary is not one, and it is not optional: see backend.h for why
    // a device backend can never run this engine's ggml_map_custom nodes. ggml_backend_sched additionally
    // REQUIRES that the last backend it is given be a CPU one, so there is no hybrid arrangement in which
    // this is absent.
    if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        device.fallback_.reset(ggml_backend_dev_init(cpu_device(), nullptr));
        if (!device.fallback_) {
            throw Error("loom::Device: the CPU fallback backend failed to initialize");
        }
        apply_cpu_threads(device.fallback_.get());

        // Host-memory accelerators join the chain between the primary and the CPU, but ONLY when the
        // primary has its own memory. Two reasons for that condition, both in Backends::assists: a
        // host accelerator improves the fallback rather than the primary, so pairing it with a primary
        // that is itself in host memory buys nothing; and a discrete device must never fall back to
        // another discrete device, which the rank-1 filter below also rules out.
        //
        // The two ranks named here are exactly the two the memory question separates, which is why
        // P4.8e's collapse of the tier above left this untouched apart from the numbering.
        //
        // A failure to initialize one is not fatal. An assist is an optimization -- the graph is
        // correct without it, because the CPU can run everything -- so a backend that declines to
        // start is skipped rather than taking the whole Device down with it.
        if (primary_rank(dev) == 0) {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t candidate = ggml_backend_dev_get(i);
                if (candidate == dev || primary_rank(candidate) != 1) continue;
                ggml_backend_ptr assist(ggml_backend_dev_init(candidate, nullptr));
                if (assist) device.assists_.push_back(std::move(assist));
            }
        }
    }
    return device;
}

} // namespace loom
