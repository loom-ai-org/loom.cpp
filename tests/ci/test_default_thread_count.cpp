// What `loom::default_cpu_thread_count()` answers, and what it refuses to answer.
//
// Hermetic, and therefore asymmetric like test_device_selection.cpp: this test cannot know how many
// cores the machine running it has, so every assertion is an INVARIANT rather than a number. The one
// exception is the affinity block, which manufactures a machine it does know the answer for by pinning
// the process to a single SMT sibling group -- that group is one physical core by construction,
// whatever the box underneath is.

#include "test_util.h"
#include "cpu_backend.h"

#include "loom/loom.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <fstream>
#include <sched.h>
#endif

namespace {

void set_env(const char* value) {
    if (value == nullptr) {
        unsetenv("LOOM_N_THREADS");
    } else {
        setenv("LOOM_N_THREADS", value, 1);
    }
}

} // namespace

int main() {
    // Whatever the harness was launched with, this test owns the variable.
    set_env(nullptr);

    // --- Autodetection is a real answer, and it is a plausible one --------------------------------
    // 0 is the documented "cannot be asked", which on Linux and macOS should never happen -- both have
    // a branch and both have a topology to read. If this fires on a CI runner, the runner has no sysfs
    // and the engine is correctly falling back to ggml's 4, but say so loudly rather than passing.
    const int detected = loom::default_cpu_thread_count();
#if defined(__linux__) || defined(__APPLE__)
    LOOM_CHECK(detected > 0);
#endif

    if (detected > 0) {
        // Physical cores are never MORE than logical CPUs. This is the assertion that would have
        // caught counting sibling groups twice, which is the whole hazard in the Linux path.
        const unsigned logical = std::thread::hardware_concurrency();
        if (logical > 0) LOOM_CHECK(detected <= static_cast<int>(logical));
    }

    // Nothing about it varies between calls -- it reads the machine, not a counter.
    LOOM_CHECK(loom::default_cpu_thread_count() == detected);

    // --- $LOOM_N_THREADS wins ----------------------------------------------------------------------
    set_env("7");
    LOOM_CHECK(loom::default_cpu_thread_count() == 7);
    set_env("1");
    LOOM_CHECK(loom::default_cpu_thread_count() == 1);

    // --- A value that is not a thread count falls through to autodetection, not to ggml's 4 --------
    // "LOOM_N_THREADS=oops" is a typo. Honouring it as a request for four threads would make a typo
    // silently mean "run this 24-core box on four cores", which is the exact failure P4.30b closed.
    for (const char* junk : {"", "0", "-3", "oops", "  "}) {
        set_env(junk);
        LOOM_CHECK(loom::default_cpu_thread_count() == detected);
    }
    set_env(nullptr);
    LOOM_CHECK(loom::default_cpu_thread_count() == detected);

#ifdef __linux__
    // --- Affinity is honoured, and one sibling group is one core -----------------------------------
    // Pinning to the full SMT sibling group of one CPU builds a machine whose physical core count is
    // known to be exactly 1 without knowing anything else about the host: on an SMT part the group is
    // both threads of one core, and on a part without SMT it is that one CPU alone.
    cpu_set_t original;
    CPU_ZERO(&original);
    if (sched_getaffinity(0, sizeof(original), &original) == 0) {
        int first = -1;
        for (int c = 0; c < CPU_SETSIZE && first < 0; ++c) {
            if (CPU_ISSET(c, &original)) first = c;
        }
        LOOM_CHECK(first >= 0);

        // The engine reads this same file; reading it here independently is what makes the check an
        // integration test of the parse rather than a restatement of it.
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(first) +
                        "/topology/thread_siblings_list");
        std::string line;
        if (f && std::getline(f, line)) {
            cpu_set_t group;
            CPU_ZERO(&group);
            int n = 0;
            size_t i = 0;
            // "0,4" or "0-1" -- every number in the line names a sibling, and for a mask the ranges
            // and the commas mean the same thing, so the separators can be ignored here.
            while (i < line.size()) {
                if (std::isdigit(static_cast<unsigned char>(line[i])) != 0) {
                    const int cpu = std::atoi(line.c_str() + i);
                    if (cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &original)) {
                        CPU_SET(cpu, &group);
                        ++n;
                    }
                    while (i < line.size() && std::isdigit(static_cast<unsigned char>(line[i])) != 0) ++i;
                } else {
                    ++i;
                }
            }

            if (n > 0 && sched_setaffinity(0, sizeof(group), &group) == 0) {
                const int pinned = loom::default_cpu_thread_count();
                // Restore before asserting, so a failure does not leave the process pinned for
                // whatever the harness does next.
                LOOM_CHECK(sched_setaffinity(0, sizeof(original), &original) == 0);
                LOOM_CHECK(pinned == 1);
            }
        }
    }
#endif

    LOOM_TEST_REPORT_AND_RETURN();
}
