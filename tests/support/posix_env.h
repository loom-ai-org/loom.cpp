#pragma once

// POSIX `setenv`/`unsetenv` for the tests that drive the engine's environment knobs (LOOM_DEVICE,
// LOOM_N_THREADS, LOOM_CPU_REPACK, LOOM_FIXTURES). MSVC's CRT has neither and spells both `_putenv_s`,
// which updates the same CRT environment `std::getenv` reads -- so the engine sees the change exactly as
// it does on Linux.
//
// One difference, and why it does not matter here: `_putenv_s(name, "")` REMOVES the variable rather
// than storing an empty string. The only test that sets one empty (test_device_selection) is checking
// that empty counts as unset, so on Windows it checks the unset half of that statement.
//
// Global-namespace functions, because the call sites spell them both `setenv` and `::setenv`.
#if defined(_WIN32)
#include <cstdlib>

inline int setenv(const char* name, const char* value, int overwrite) {
    if (!overwrite && std::getenv(name) != nullptr) return 0;
    return _putenv_s(name, value);
}

inline int unsetenv(const char* name) {
    return _putenv_s(name, "");
}
#endif
