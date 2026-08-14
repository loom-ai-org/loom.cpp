#pragma once

// Tiny assert-based test harness shared by every test_*.cpp binary. Deliberately dependency-free (no
// gtest/catch2) to match ggml's own tests/ style and keep the build graph simple -- each test_*.cpp is
// its own `add_executable` + `add_test`, and a non-zero exit code (any failed LOOM_CHECK) fails the
// ctest run.

// cpu_backend.h is included for its SIDE EFFECT and not for anything named below, so do not remove it
// as unused (BACKLOG.md P4.8f). Its `detail::g_backend_loader` registers this build's backend directory
// before main, which in a GGML_BACKEND_DL build is the difference between a populated registry and an
// empty one. Every test file includes THIS header -- all 116 of them -- so putting it here is what makes
// the guarantee unconditional instead of per-file.
//
// It was per-file, and that failed in the worst available way: the two device-parity gates did not
// include it, so under DL they found no GPU and exited 77. ctest renders that as Skipped and the suite
// as green, which is how a build with Vulkan compiled in reported success while running neither gate.
#include "cpu_backend.h"

#include <cmath>
#include <cstdio>

namespace loom_test {
inline int g_checks = 0;
inline int g_failures = 0;
} // namespace loom_test

#define LOOM_CHECK(cond)                                                                     \
    do {                                                                                     \
        ++loom_test::g_checks;                                                               \
        if (!(cond)) {                                                                       \
            ++loom_test::g_failures;                                                         \
            std::fprintf(stderr, "CHECK FAILED at %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
        }                                                                                     \
    } while (0)

#define LOOM_CHECK_NEAR(a, b, eps)                                                                        \
    do {                                                                                                   \
        ++loom_test::g_checks;                                                                             \
        const double loom_check_near_a_ = (a), loom_check_near_b_ = (b);                                    \
        if (std::fabs(loom_check_near_a_ - loom_check_near_b_) > (eps)) {                                   \
            ++loom_test::g_failures;                                                                        \
            std::fprintf(stderr, "CHECK_NEAR FAILED at %s:%d: %s (%g) vs %s (%g)\n", __FILE__, __LINE__,    \
                          #a, loom_check_near_a_, #b, loom_check_near_b_);                                   \
        }                                                                                                    \
    } while (0)

// Runs `expr` and checks it throws exactly `ExceptionType` (or a subclass). Fails the check if it
// throws nothing, or throws something else.
#define LOOM_CHECK_THROWS(expr, ExceptionType)                                                            \
    do {                                                                                                   \
        ++loom_test::g_checks;                                                                              \
        bool loom_threw_ = false;                                                                           \
        try {                                                                                                \
            (void)(expr);                                                                                    \
        } catch (const ExceptionType&) {                                                                     \
            loom_threw_ = true;                                                                              \
        } catch (...) {                                                                                      \
        }                                                                                                     \
        if (!loom_threw_) {                                                                                   \
            ++loom_test::g_failures;                                                                           \
            std::fprintf(stderr, "CHECK_THROWS FAILED at %s:%d: expected `%s` to throw " #ExceptionType "\n", \
                          __FILE__, __LINE__, #expr);                                                          \
        }                                                                                                       \
    } while (0)

#define LOOM_TEST_REPORT_AND_RETURN()                                                          \
    do {                                                                                        \
        std::fprintf(stdout, "%d/%d checks passed\n", loom_test::g_checks - loom_test::g_failures, \
                      loom_test::g_checks);                                                      \
        return loom_test::g_failures == 0 ? 0 : 1;                                              \
    } while (0)
