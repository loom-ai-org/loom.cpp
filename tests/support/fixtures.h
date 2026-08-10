#pragma once

// Where a gate test's real-model fixture comes from.
//
// **One variable, not sixty-nine.** Every test under `tests/gate/` compares against something a real
// checkpoint produced -- an exported GGUF, or a directory of reference tensors a HF forward pass
// wrote. Those are gigabytes and hours; they cannot live in this repo and they cannot be rebuilt in
// CI. Each test named its own environment variable for them, which meant running the gate suite was a
// matter of exporting some sixty-nine of them correctly and getting a silent skip wherever you did
// not. `LOOM_FIXTURES` names ONE directory holding all of them, so the whole suite is one variable.
//
// **The per-test variables still work, and still win.** Each is checked FIRST, because pointing one
// test at one artifact you have just rebuilt -- somewhere else, under another name -- is exactly what
// you do while working on that model, and a fixture root would be in the way of it. The root is the
// default, not the law. That ordering is also what makes this migration safe: with the old variable
// set, behaviour is bit-for-bit what it was before the root existed.
//
// **The layout is derived, not tabulated.** `LOOM_KOKORO_MIL_GGUF` is `$LOOM_FIXTURES/kokoro_mil.gguf`
// and `LOOM_KOKORO_ALBERT_REF_DIR` is `$LOOM_FIXTURES/kokoro_albert_ref/`: drop `LOOM_`, lowercase,
// and let a `_GGUF` suffix mean a file and anything else a directory. A hand-maintained table mapping
// sixty-nine variables to sixty-nine paths is a thing that drifts from the tests it describes; a rule
// cannot. `scripts/fixtures.py` applies the same rule when it populates and verifies the directory.
//
// A test that finds neither returns nullptr here, falls back to whatever default it already had, and
// exits 77 when that is not present either -- which ctest reports as Skipped. That is the whole
// contract: a missing fixture is not a failure.

#include <cstdlib>
#include <map>
#include <string>
#include <sys/stat.h>

namespace loom_test {

inline bool path_exists(const std::string& path) {
    struct stat st {};
    return !path.empty() && ::stat(path.c_str(), &st) == 0;
}

// `$LOOM_FIXTURES/<name minus LOOM_, lowercased>`, with `.gguf` restored where the variable named a
// file. Exposed for `scripts/fixtures.py`'s own test, which checks the two implementations of this
// rule against each other rather than trusting them to stay in step.
inline std::string fixture_relpath(const std::string& var) {
    auto ends_with = [](const std::string& s, const char* suffix) {
        const size_t n = std::char_traits<char>::length(suffix);
        return s.size() > n && s.compare(s.size() - n, n, suffix) == 0;
    };
    std::string stem = var.rfind("LOOM_", 0) == 0 ? var.substr(5) : var;
    const bool is_gguf = ends_with(stem, "_GGUF");
    if (is_gguf) stem.resize(stem.size() - 5);
    // `_DIR` says "this one is a directory", which the path already conveys by not being a file. It
    // is noise in a filename, and `kokoro_albert_ref/` reads better than `kokoro_albert_ref_dir/`.
    else if (ends_with(stem, "_DIR")) stem.resize(stem.size() - 4);
    for (char& c : stem) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return is_gguf ? stem + ".gguf" : stem;
}

// A drop-in for `std::getenv` on a fixture variable: the variable's own value if it is set, else the
// same artifact under `$LOOM_FIXTURES` if it is there, else nullptr.
//
// Deliberately shaped as `getenv` is, returning `const char*`, so adopting it changed nothing but the
// call itself in 78 test files -- every null check, every default and every skip around it is the one
// that was already there and already worked.
inline const char* fixture_env(const char* var) {
    if (var == nullptr) return nullptr;
    if (const char* explicit_path = std::getenv(var); explicit_path != nullptr && *explicit_path != '\0') {
        return explicit_path;
    }
    const char* root = std::getenv("LOOM_FIXTURES");
    if (root == nullptr || *root == '\0') return nullptr;

    // Stable storage for the derived path: `std::map`'s nodes do not move, so a pointer handed out
    // here stays valid for the life of the process, which is what a `getenv`-shaped return promises.
    static std::map<std::string, std::string> resolved;
    auto [entry, inserted] = resolved.try_emplace(var);
    if (inserted) {
        std::string candidate = std::string(root) + "/" + fixture_relpath(var);
        entry->second = path_exists(candidate) ? candidate : std::string{};
    }
    return entry->second.empty() ? nullptr : entry->second.c_str();
}

} // namespace loom_test
