// What `loom::Device::open()` resolves a device spec to, and what it refuses to resolve.
//
// Hermetic on purpose, and therefore asymmetric: the ONE device every build has is the CPU, so every
// assertion below that names a concrete device names that one. What a GPU-enabled build does with a real
// device is not a thing this test can know, so where the answer depends on whether one is present it
// asserts the INVARIANT rather than the outcome -- "auto" resolves to something, "gpu" either resolves
// to a non-CPU device or throws, never anything else. tests/gate/test_e2e_device_parity.cpp is where a
// real device is actually exercised.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

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
        ggml_backend_ptr backend(ggml_backend_cpu_init());
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

    LOOM_TEST_REPORT_AND_RETURN();
}
