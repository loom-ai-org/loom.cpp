// What `loom::Device::open()` resolves a device spec to, and what it refuses to resolve.
//
// Hermetic on purpose, and therefore asymmetric: the ONE device every build has is the CPU, so every
// assertion below that names a concrete device names that one. What a GPU-enabled build does with a real
// device is not a thing this test can know, so where the answer depends on whether one is present it
// asserts the INVARIANT rather than the outcome -- "auto" resolves to something, "gpu" either resolves
// to a non-CPU device or throws, never anything else. tests/gate/test_e2e_device_parity.cpp is where a
// real device is actually exercised.

#include "test_util.h"
#include "cpu_backend.h"

#include "loom/loom.h"


#include <algorithm>
#include <cstdlib>
#include <string>

namespace {

bool has_non_cpu_device() {
    const auto devices = loom::available_devices();
    return std::any_of(devices.begin(), devices.end(),
                        [](const loom::DeviceInfo& d) { return !d.is_cpu; });
}

template <typename Fn>
bool throws_loom_error(Fn&& fn) {
    try {
        fn();
    } catch (const loom::Error&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    // --- The registry always has a CPU, whatever else it has ---------------------------------------
    const std::vector<loom::DeviceInfo> devices = loom::available_devices();
    LOOM_CHECK(!devices.empty());
    LOOM_CHECK(std::any_of(devices.begin(), devices.end(),
                            [](const loom::DeviceInfo& d) { return d.is_cpu; }));
    for (const loom::DeviceInfo& d : devices) {
        LOOM_CHECK(!d.name.empty());
    }

    // --- "cpu" is the CPU, and a CPU Device is NOT hybrid ------------------------------------------
    // The second half is the load-bearing one: `hybrid()` is what decides whether GraphBuilder uses a
    // scheduler at all, so a CPU-only build reaching for one would silently change every existing
    // measurement in the ledger.
    {
        loom::Device cpu = loom::Device::open("cpu");
        LOOM_CHECK(cpu.is_cpu());
        LOOM_CHECK(cpu.backends().primary != nullptr);
        LOOM_CHECK(cpu.backends().fallback == nullptr);
        LOOM_CHECK(!cpu.backends().hybrid());
        LOOM_CHECK(!cpu.name().empty());
    }

    // --- Spelling is not part of the name ----------------------------------------------------------
    {
        loom::Device upper = loom::Device::open("CPU");
        LOOM_CHECK(upper.is_cpu());
    }

    // --- "auto" always resolves; whether to a device depends on the build ---------------------------
    {
        loom::Device automatic = loom::Device::open("auto");
        LOOM_CHECK(automatic.backends().primary != nullptr);
        // The whole point of "auto": it prefers a device when there is one, and never fails when there
        // is not. Both halves stated as one implication so this reads the same on either build.
        LOOM_CHECK(automatic.is_cpu() == !has_non_cpu_device());
        // A device Device is hybrid and a CPU one is not -- the two are the same fact.
        LOOM_CHECK(automatic.backends().hybrid() == !automatic.is_cpu());
        if (!automatic.is_cpu()) {
            LOOM_CHECK(automatic.backends().fallback != nullptr);
            LOOM_CHECK(automatic.backends().fallback != automatic.backends().primary);
        }
    }

    // --- "gpu" is an assertion about the machine, not a preference ----------------------------------
    if (has_non_cpu_device()) {
        loom::Device gpu = loom::Device::open("gpu");
        LOOM_CHECK(!gpu.is_cpu());
        LOOM_CHECK(gpu.backends().hybrid());
    } else {
        // Deliberately an error rather than a quiet CPU run: see Device::open's own comment.
        LOOM_CHECK(throws_loom_error([] { loom::Device::open("gpu"); }));
    }

    // --- An unknown name is an error, and says what IS available ------------------------------------
    LOOM_CHECK(throws_loom_error([] { loom::Device::open("nonexistent-device-0"); }));
    try {
        loom::Device::open("nonexistent-device-0");
        LOOM_CHECK(false);
    } catch (const loom::Error& e) {
        const std::string message = e.what();
        LOOM_CHECK(message.find("nonexistent-device-0") != std::string::npos);
        LOOM_CHECK(message.find(devices.front().name) != std::string::npos);
    }

    // --- Every device the registry lists can be opened by the name it was listed under ---------------
    for (const loom::DeviceInfo& d : devices) {
        loom::Device by_name = loom::Device::open(d.name);
        LOOM_CHECK(by_name.name() == d.name);
        LOOM_CHECK(by_name.is_cpu() == d.is_cpu);
    }

    // --- LOOM_DEVICE is consulted only when the caller named nothing ---------------------------------
    {
        ::setenv("LOOM_DEVICE", "cpu", /*overwrite=*/1);
        LOOM_CHECK(loom::Device::open().is_cpu());
        // An explicit argument outranks the environment, which is the whole reason the order is
        // argument-then-environment-then-autodetect and not the other way round.
        ::setenv("LOOM_DEVICE", "nonexistent-device-0", 1);
        LOOM_CHECK(loom::Device::open("cpu").is_cpu());
        // ...and a bad one in the environment is still an error when nothing overrides it, rather than
        // being silently ignored.
        LOOM_CHECK(throws_loom_error([] { loom::Device::open(); }));
        // Empty counts as unset, so exporting it blank is not a way to get an error.
        ::setenv("LOOM_DEVICE", "", 1);
        LOOM_CHECK(loom::Device::open().backends().primary != nullptr);
        ::unsetenv("LOOM_DEVICE");
    }

    // --- A bare ggml_backend_t still means what it always meant ---------------------------------------
    // The implicit conversion is what kept every pre-P4.4 call site compiling; this states that it also
    // kept them MEANING the same thing -- one backend, no scheduler.
    {
        // Through the shared helper, which goes via the registry rather than the CPU backend's own
        // ggml_backend_cpu_init symbol -- that one is not linked in a GGML_BACKEND_DL build. This test
        // has to work in both configurations, because a DL build is the only place the ranking above
        // can be exercised against the registration order it exists to defeat.
        ggml_backend_ptr backend(loom_test::cpu_backend());
        LOOM_CHECK(backend != nullptr);
        const loom::Backends implicit = backend.get();
        LOOM_CHECK(implicit.primary == backend.get());
        LOOM_CHECK(implicit.fallback == nullptr);
        LOOM_CHECK(!implicit.hybrid());
        // And a pair whose two halves are the same backend is not hybrid either -- scheduling a backend
        // against itself is pure overhead, so the check is `fallback != primary`, not `fallback != null`.
        const loom::Backends self_pair(backend.get(), backend.get());
        LOOM_CHECK(!self_pair.hybrid());
    }

    // --- A kind-spec answers with that kind, or not at all ------------------------------------------
    // The defect this replaces: "gpu" meant "the first device that is not the CPU", so on a machine
    // with both a GPU and a BLAS accelerator it answered with whichever ggml happened to register
    // first -- and that order differs between a linked build and a GGML_BACKEND_DL one, so the same
    // spec gave different devices from one source tree (BACKLOG.md P4.8b).
    //
    // Hermetic, so what it can assert is the invariant rather than the outcome: on a CPU-only build
    // both specs throw, and on a build with real devices each must answer with its own kind or throw.
    // The three-device build (-DGGML_VULKAN=ON -DGGML_BLAS=ON) is where this has teeth.
    {
        const auto devices = loom::available_devices();
        const bool any_non_cpu = has_non_cpu_device();

        if (!any_non_cpu) {
            LOOM_CHECK(throws_loom_error([] { loom::Device::open("gpu"); }));
            LOOM_CHECK(throws_loom_error([] { loom::Device::open("npu"); }));
        }

        // Whatever "gpu" returns, it is never the CPU and never a device "npu" would also return --
        // the two specs partition, they do not overlap.
        std::string gpu_name, npu_name;
        try {
            loom::Device gpu = loom::Device::open("gpu");
            LOOM_CHECK(!gpu.is_cpu());
            gpu_name = gpu.name();
        } catch (const loom::Error&) {
        }
        try {
            loom::Device npu = loom::Device::open("npu");
            LOOM_CHECK(!npu.is_cpu());
            npu_name = npu.name();
        } catch (const loom::Error&) {
        }
        LOOM_CHECK(gpu_name.empty() || npu_name.empty() || gpu_name != npu_name);

        // "accel" is a spelling of "npu", not a third behaviour.
        try {
            LOOM_CHECK(loom::Device::open("accel").name() == npu_name);
        } catch (const loom::Error&) {
            LOOM_CHECK(npu_name.empty());
        }

        // "auto" never fails and never ranks below what an explicit kind-spec could have found: if a
        // GPU exists, "auto" IS that GPU. This is the assertion that would have caught the defect --
        // under the old rule "auto" returned BLAS on a machine whose GPU was sitting right there.
        loom::Device automatic = loom::Device::open("auto");
        if (!gpu_name.empty()) {
            LOOM_CHECK(automatic.name() == gpu_name);
        } else if (!npu_name.empty()) {
            LOOM_CHECK(automatic.name() == npu_name);
        }
        // And with nothing else present it is the CPU, which is why it cannot throw.
        if (!any_non_cpu) LOOM_CHECK(automatic.is_cpu());

        // Repeated resolution is stable. Registration order is no longer an input, but a ranking
        // implemented with a strict `<` could still drift if it were ever made order-sensitive again.
        LOOM_CHECK(loom::Device::open("auto").name() == automatic.name());
    }

    // --- The scheduler order: primary first, CPU last, nothing repeated -----------------------------
    // `ggml_backend_sched_new` ASSERTS that the last backend it is given can run anything, i.e. that it
    // is the CPU, so this is a crash rather than a slowdown when it is wrong. It is also the invariant
    // that survives assists being added: whatever else lands in the chain, it lands in the middle.
    {
        loom::Device cpu = loom::Device::open("cpu");
        const loom::Backends cpu_backends = cpu.backends();
        LOOM_CHECK(cpu_backends.assists.empty());
        LOOM_CHECK(cpu_backends.schedule_order().size() == 1);
        LOOM_CHECK(cpu_backends.schedule_order().front() == cpu_backends.primary);

        loom::Device automatic = loom::Device::open("auto");
        const loom::Backends b = automatic.backends();
        const std::vector<ggml_backend_t> order = b.schedule_order();
        LOOM_CHECK(!order.empty());
        LOOM_CHECK(order.front() == b.primary);
        if (b.hybrid()) {
            LOOM_CHECK(order.back() == b.fallback);
            LOOM_CHECK(order.size() == b.assists.size() + 2);
        } else {
            LOOM_CHECK(order.size() == 1);
        }
        // An assist is never the primary or the CPU wearing a second hat -- handing ggml the same
        // backend twice is not a configuration, it is a bug that would show up as strange split plans.
        for (size_t i = 0; i < order.size(); ++i) {
            LOOM_CHECK(order[i] != nullptr);
            for (size_t j = i + 1; j < order.size(); ++j) LOOM_CHECK(order[i] != order[j]);
        }
    }

    // --- schedule_order() filters what a caller hands it --------------------------------------------
    // Constructed by hand rather than through a Device, because the point is that the FILTERING is in
    // schedule_order and not in Device -- an embedding host assembling its own Backends gets the same
    // guarantees. Null, the primary repeated, and the CPU repeated all disappear.
    {
        ggml_backend_ptr cpu(loom_test::cpu_backend());
        ggml_backend_ptr other(loom_test::cpu_backend());
        LOOM_CHECK(cpu != nullptr && other != nullptr);

        const loom::Backends messy(other.get(), {nullptr, other.get(), cpu.get()}, cpu.get());
        const std::vector<ggml_backend_t> order = messy.schedule_order();
        // primary, then the CPU last: the null is dropped, the repeated primary is dropped, and the
        // assist that IS the fallback is dropped rather than being scheduled twice.
        LOOM_CHECK(order.size() == 2);
        LOOM_CHECK(order.front() == other.get());
        LOOM_CHECK(order.back() == cpu.get());
    }

    // --- A backend search path never costs a device -------------------------------------------------
    // add_backend_search_path is how an embedded host points at backends ggml's own search would not
    // find (loom-py's wheel; BACKLOG.md P4.8). What it must NOT do is make things worse, and the way it
    // could is by being consulted destructively: a stale $LOOM_BACKEND_DIR, a path removed since it was
    // registered, or the same directory added twice must all leave the registry exactly as it was.
    // Directories are offered to ggml, not required of it.
    {
        const size_t before = loom::available_devices().size();
        LOOM_CHECK(before > 0);

        loom::add_backend_search_path("/definitely/not/a/directory");
        loom::add_backend_search_path("");
        LOOM_CHECK(loom::available_devices().size() == before);
        LOOM_CHECK(loom::Device::open("cpu").is_cpu());

        // Twice, because ggml dedupes on the registration pointer rather than on the path, and the
        // engine leans on that to let a host add a directory at any time without tracking what it has
        // already added. A second sweep of a directory that DID contain a backend must register nothing
        // new -- here that is asserted the only way a hermetic test can, on a path swept twice.
        const std::string cwd_path = ".";
        loom::add_backend_search_path(cwd_path);
        const size_t once = loom::available_devices().size();
        loom::add_backend_search_path(cwd_path);
        LOOM_CHECK(loom::available_devices().size() == once);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
