#pragma once

// The CPU backend, obtained the way that works in EVERY link mode.
//
// WHY THIS EXISTS (BACKLOG.md P4.8b). Every test in this suite used to call `ggml_backend_cpu_init()`
// directly -- 115 call sites across 109 files -- and that symbol lives INSIDE the CPU backend. A
// `GGML_BACKEND_DL` build does not link the CPU backend; it dlopens it at run time. So the entire
// suite failed to LINK in the one configuration the Python wheels ship, and the gate that proves the
// device layer generalises (`tests/gate/test_e2e_device_parity.cpp`) could not be run against it.
//
// Going through the registry is correct in both modes and costs nothing. In a linked build
// `ggml_backend_dev_by_type` finds the backend that registered itself from its own translation unit;
// in a DL build it finds the one loaded from disk. Neither case needs this file to know which.

#include "loom/core/backend.h"

#include <ggml-backend.h>

#include <cstdio>
#include <mutex>

namespace loom_test {

// Populate the registry. A linked build needs nothing -- backends register themselves from static
// initialisers -- but a DL build's registry is EMPTY until something sweeps for .so files, and ggml's
// own default search (the executable's directory, then the current directory) does not cover this
// build tree: the test binaries land in `tests/` and the backends in `bin/`. `LOOM_TEST_BACKEND_DIR`
// is that directory, passed in by tests/CMakeLists.txt, and it goes through the engine's own
// `add_backend_search_path` -- which incidentally makes every test a user of that API.
inline void ensure_backends() {
    static std::once_flag once;
    std::call_once(once, [] {
#ifdef LOOM_TEST_BACKEND_DIR
        loom::add_backend_search_path(LOOM_TEST_BACKEND_DIR);
#endif
        // The call that performs the sweep. The device list itself is not wanted here.
        (void)loom::available_devices();
    });
}

namespace detail {

// Registration happens BEFORE main, not on the first cpu_backend() call, and the difference is a real
// failure rather than a tidiness point. Doing it lazily works for the tests that take a CPU backend
// before anything else -- which is nearly all of them -- and breaks any test that asks the engine a
// question first: `test_device_selection` calls loom::available_devices() as its opening line, saw an
// empty registry in a DL build, and aborted. Sequencing it here makes the guarantee unconditional
// instead of dependent on what a given test happens to do first.
//
// An inline variable (C++17), so one instance across every translation unit that includes this. Safe
// against the static initialisation order fiasco because everything it reaches on the engine side is a
// function-local static -- see the note on loader_mutex() in src/core/backend.cpp, which was changed
// from namespace-scope objects for exactly this call.
struct BackendLoader {
    BackendLoader() { ensure_backends(); }
};
inline const BackendLoader g_backend_loader;

} // namespace detail

// An OWNING handle, exactly as `ggml_backend_cpu_init()` returned, so that every
// `ggml_backend_ptr backend(...)` call site kept its meaning across the rename.
//
// Null only in a build that cannot find its CPU backend at all, which in a DL build is a deployment
// problem rather than a test failure -- hence the explanation on stderr rather than a silent nullptr
// that surfaces as a crash three lines later.
inline ggml_backend_t cpu_backend() {
    ensure_backends();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == nullptr) {
        std::fprintf(stderr,
                     "loom_test::cpu_backend: no CPU device is registered. In a GGML_BACKEND_DL build "
                     "the CPU is a plugin too -- put libggml-cpu*.so where this binary can find it, "
                     "or point $GGML_BACKEND_PATH at it.\n");
        return nullptr;
    }
    return ggml_backend_dev_init(dev, nullptr);
}

} // namespace loom_test
