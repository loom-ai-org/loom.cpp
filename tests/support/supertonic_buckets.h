// Picking a Supertonic text bucket, for tests that call the text topologies directly.
//
// The export traces `dp`/`ttl_text`/`vfe` once per padded text width and names them `<prefix>_<width>`;
// the embedded driver picks the smallest that fits the caller's ids (BACKLOG.md P4.6a). A test that
// builds one of those graphs itself has to make the same choice, and this is that choice in one place
// so four tests cannot drift apart on it.
//
// **Discovered, not declared.** The widths are read out of the model's own `topology_names()` rather
// than listed here, which is the difference between a test that checks the export and a test that
// agrees with itself: a bucket added, removed or renamed in `supertonic_export.py` changes what these
// tests exercise, with nothing to keep in sync.
#pragma once

#include "loom/loom.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace loom_test {

// Every width `prefix` is exported at, ascending. Empty if the model has no such topology, which is
// what a caller should skip on rather than assert about -- an older GGUF is a missing fixture, not a
// failure.
inline std::vector<uint32_t> supertonic_buckets(const loom::GgufModel& model, const std::string& prefix) {
    std::vector<uint32_t> widths;
    for (const std::string& name : model.topology_names()) {
        if (name.rfind(prefix + "_", 0) != 0) continue;
        const std::string suffix = name.substr(prefix.size() + 1);
        // `dp_32` is a bucket; a hypothetical `dp_style` is not. Digits only, so a non-numeric
        // suffix is skipped rather than parsed to garbage.
        if (suffix.empty() || suffix.find_first_not_of("0123456789") != std::string::npos) continue;
        widths.push_back(static_cast<uint32_t>(std::stoul(suffix)));
    }
    std::sort(widths.begin(), widths.end());
    return widths;
}

// The topology the driver would run for `n_ids` real ids: `<prefix>_<smallest width >= n_ids>`.
// Returns an empty string when nothing fits, which is the case the driver raises on.
inline std::string supertonic_bucket_topology(const loom::GgufModel& model, const std::string& prefix,
                                               uint32_t n_ids, uint32_t* width_out = nullptr) {
    for (uint32_t width : supertonic_buckets(model, prefix)) {
        if (width >= n_ids) {
            if (width_out != nullptr) *width_out = width;
            return prefix + "_" + std::to_string(width);
        }
    }
    return {};
}

} // namespace loom_test
