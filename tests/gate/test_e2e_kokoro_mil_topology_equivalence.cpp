// Drop-in equivalence between the MIL-traced topologies and the pre-MIL bespoke ones they replace.
//
// Kokoro's (and StyleTTS2's) MIL export is progressively taking over topologies that used to be loaded
// from the hand-built `kokoro.gguf` alongside it -- the goal being one self-contained GGUF per model.
// Each transfer needs a real numerical gate, and `test_e2e_kokoro_mil_lua_driver.cpp` is NOT one: it has
// no oracle waveform (deliberately -- see its own header), and its per-sample checks are `isfinite`
// plus an rms/max_abs range. A wrong topology that still produced a finite, plausibly-loud waveform
// would pass it.
//
// So this test compares each transferred topology against **the thing it replaces**, which is the
// strongest available reference and cheaper than a per-phase PyTorch fixture: both GGUFs are loadable,
// both declare the same inputs (they must -- the Lua driver calls them identically and is unchanged),
// so the same random inputs go into both and the outputs are compared directly.
//
// The list of topologies checked is DERIVED, not hard-coded: every named topology present in both files
// is compared. That means a phase moved to MIL is covered the moment it is exported, with no test edit,
// and `kMinShared` below stops the derivation from passing vacuously if the intersection ever collapses.
//
// Tolerance: the MIL and bespoke formulations are different arrangements of the same arithmetic (an
// LSTM cell's gate order and bias packing differ; a traced Conv1d and a hand-built MUL_MAT chain
// accumulate differently), so the target is "same function, float-accumulation apart", not bitwise.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

// The intersection must not silently shrink to nothing (or to the trivial case) -- if it does, this
// test would "pass" while checking no transfer at all.
constexpr size_t kMinShared = 4;

// A sequence length that every shared topology can be built at. Small, and not a power of two, so a
// stride bug does not accidentally cancel.
constexpr uint32_t kTokens = 7;

uint32_t resolve_dim(const std::string& spec, uint32_t n_tokens) {
    if (spec.empty()) return 1;
    if (spec[0] == '$' || spec == "n_tokens") return n_tokens;
    return static_cast<uint32_t>(std::strtoul(spec.c_str(), nullptr, 10));
}

// Fills every declared input of a built graph with deterministic pseudo-random data, and returns what
// was written so the same bytes can go into the other model's graph.
std::map<std::string, std::vector<float>> make_inputs(const loom::GraphTopology& topo,
                                                      uint32_t n_tokens, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::map<std::string, std::vector<float>> out;
    for (const loom::TensorSpec& in : topo.inputs) {
        size_t n = 1;
        for (const std::string& d : in.shape) n *= resolve_dim(d, n_tokens);
        std::vector<float> v(n);
        for (float& x : v) v.empty() ? void() : void(x = dist(rng));
        out[in.name] = std::move(v);
    }
    return out;
}

void write_inputs(const loom::GraphTopology& topo, const loom::GraphBuilder::BuildResult& r,
                  const std::map<std::string, std::vector<float>>& values, uint32_t n_tokens) {
    for (const loom::TensorSpec& in : topo.inputs) {
        const std::vector<float>& v = values.at(in.name);
        ggml_tensor* t = r.input_tensors.at(in.name);
        if (in.dtype == "i32") {
            // An integer input is a token id, not a float: map the shared random values onto a small
            // valid-looking range so both models see the SAME ids.
            std::vector<int32_t> ids(v.size());
            for (size_t i = 0; i < v.size(); ++i) {
                ids[i] = static_cast<int32_t>(std::fabs(v[i]) * 50.0f) + 1;
            }
            ggml_backend_tensor_set(t, ids.data(), 0, ids.size() * sizeof(int32_t));
        } else {
            ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
        }
    }
    (void)n_tokens;
}

std::vector<float> run(loom::GgufModel& model, const loom::GraphTopology& topo,
                       const std::map<std::string, std::vector<float>>& values,
                       ggml_backend_t backend, uint32_t n_tokens) {
    loom::GraphBuilder builder(topo, model, backend, /*kv_cache=*/nullptr);
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", n_tokens}, {"n_past", 0}});
    write_inputs(topo, r, values, n_tokens);
    ggml_backend_graph_compute(backend, r.graph);
    std::vector<float> out(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

// "$n_tokens" and "n_tokens" are the SAME symbol -- SymbolEnv accepts either, and the two producers
// happen to spell it differently (the hand-built converters write the `$` sigil, the MIL exporter does
// not). Comparing raw strings reports every transferred topology as a mismatch, which is how this was
// found; comparing the resolved symbol is the real question.
std::string canonical_dim(const std::string& d) {
    return (!d.empty() && d[0] == '$') ? d.substr(1) : d;
}

// A shared NAME is not always a transfer. Two MIL phases deliberately redefined the interface their
// same-named bespoke topology had, and the driver was rewritten to match: StyleTTS2's "albert" drops
// the host-supplied `positions`/`attn_mask` (the traced CustomAlbert derives both in-graph) and its
// "diffusion" drops `attn_mask` for the same reason -- both recorded in styletts2_driver/'s own header.
// Those are not drop-in replacements, so comparing them is meaningless. A NAMED list rather than
// "skip any declared-input difference", because every other mismatch is a real finding.
bool interface_changed_deliberately(const std::string& name) {
    return name == "albert" || name == "diffusion";
}

bool declares_same_inputs(const loom::GraphTopology& a, const loom::GraphTopology& b) {
    if (a.inputs.size() != b.inputs.size()) return false;
    for (size_t i = 0; i < a.inputs.size(); ++i) {
        if (a.inputs[i].name != b.inputs[i].name) return false;
        if (a.inputs[i].shape.size() != b.inputs[i].shape.size()) return false;
        for (size_t d = 0; d < a.inputs[i].shape.size(); ++d) {
            if (canonical_dim(a.inputs[i].shape[d]) != canonical_dim(b.inputs[i].shape[d])) return false;
        }
    }
    return true;
}

} // namespace

// One family's pair of files. Both iSTFTNet-family models went through the same transfer and share
// most of these topologies, so one test covers both rather than two near-identical ones.
struct Family {
    const char* name;
    const char* legacy_env;   // directory containing the pre-MIL gguf
    const char* legacy_file;
    const char* mil_env;      // the MIL gguf itself
};

int compare_family(const Family& fam, ggml_backend_t backend) {
    const char* legacy_dir = loom_test::fixture_env(fam.legacy_env);
    const char* mil_path = loom_test::fixture_env(fam.mil_env);
    if (legacy_dir == nullptr || mil_path == nullptr) {
        std::fprintf(stderr, "%s: skipped (set %s and %s)\n", fam.name, fam.legacy_env, fam.mil_env);
        return -1;
    }
    auto model_lua = loom::GgufModel::load(std::string(legacy_dir) + "/" + fam.legacy_file, backend);
    auto model_mil = loom::GgufModel::load(mil_path, backend);
    LOOM_CHECK(model_lua != nullptr);
    LOOM_CHECK(model_mil != nullptr);
    ggml_backend_t backend_ = backend;
    (void)backend_;

    size_t compared = 0, skipped_shape = 0;
    for (const std::string& name : model_mil->topology_names()) {
        if (!model_lua->has_topology(name)) continue;  // MIL-only phase: nothing to compare against
        if (interface_changed_deliberately(name)) {
            std::fprintf(stderr, "%-10s %-28s skipped: interface deliberately redefined\n",
                         fam.name, name.c_str());
            continue;
        }
        loom::GraphTopology mil = loom::GraphTopology::parse(model_mil->topology_json(name));
        loom::GraphTopology bespoke = loom::GraphTopology::parse(model_lua->topology_json(name));

        // A drop-in replacement must declare the same inputs -- the Lua driver calls both identically
        // and is unchanged by the transfer. A mismatch here is the finding, not a reason to skip.
        LOOM_CHECK(declares_same_inputs(mil, bespoke));

        const auto values = make_inputs(mil, kTokens, /*seed=*/1234u + static_cast<uint32_t>(compared));
        const std::vector<float> got = run(*model_mil, mil, values, backend, kTokens);
        const std::vector<float> want = run(*model_lua, bespoke, values, backend, kTokens);
        if (got.size() != want.size()) {
            std::fprintf(stderr, "%-28s SHAPE MISMATCH: mil %zu elems, bespoke %zu\n",
                         name.c_str(), got.size(), want.size());
            ++skipped_shape;
            LOOM_CHECK(false);
        }

        double max_abs = 0.0, sum_abs = 0.0, ref_max = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            max_abs = std::max(max_abs, std::fabs(static_cast<double>(got[i] - want[i])));
            sum_abs += std::fabs(static_cast<double>(got[i] - want[i]));
            ref_max = std::max(ref_max, std::fabs(static_cast<double>(want[i])));
        }
        const double mean_abs = sum_abs / static_cast<double>(got.size());
        std::fprintf(stderr, "%-10s %-28s n=%zu  mean_abs_diff=%.3g  max_abs_diff=%.3g  (ref |x|<=%.3g)\n",
                     fam.name, name.c_str(), got.size(), mean_abs, max_abs, ref_max);
        // Relative to the reference's own scale: a topology whose outputs are O(100) cannot be held to
        // the same absolute bound as one whose outputs are O(1).
        LOOM_CHECK(mean_abs < 1e-4 * std::max(1.0, ref_max));
        LOOM_CHECK(max_abs < 1e-2 * std::max(1.0, ref_max));
        ++compared;
    }

    std::fprintf(stderr, "%s: compared %zu shared topolog(ies)\n", fam.name, compared);
    LOOM_CHECK(skipped_shape == 0);
    // Guards the derivation: if the MIL export stopped producing the topologies it has taken over, the
    // loop above would silently compare nothing and this family would pass having checked no transfer.
    LOOM_CHECK(compared >= kMinShared);
    return static_cast<int>(compared);
}

int main() {
    static const Family kFamilies[] = {
        {"kokoro", "LOOM_KOKORO_LUA_DIR", "kokoro.gguf", "LOOM_KOKORO_MIL_GGUF"},
        {"styletts2", "LOOM_STYLETTS2_LUA_DIR", "styletts2.gguf", "LOOM_STYLETTS2_MIL_GGUF"},
    };

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    int families_run = 0;
    for (const Family& fam : kFamilies) {
        if (compare_family(fam, backend.get()) >= 0) ++families_run;
    }
    if (families_run == 0) {
        std::fprintf(stderr, "skipping: no family's gguf pair was configured\n");
        return 77;
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
