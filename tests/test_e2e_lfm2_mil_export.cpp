// Validates the MIL-compiler-exported LFM2-350M GGUF (export_lfm2_monolithic.py) end to end, at genuinely
// dynamic (non-fixed, unpadded) prompt lengths -- the concern EXPORT-BACKLOG.md item 3 raised: this
// script used to trace with a FIXED (1, 128) shape and pad every prompt up to it, so nothing ever
// exercised the exporter's actual dynamic-`n_tokens` codepath at a real, differing prompt length. Now
// that it traces with `ct.RangeDim`, this compares the compiled model's next-token prediction against a
// real HF forward pass at two different prompt lengths (3 and 7 tokens) -- neither padded to any fixed
// size -- so a shape-handling regression from tracing genuinely dynamically would show up as a
// wrong/crashing token, not just "does it export".
//
// Not generated at ctest time (needs the real LFM2-350M checkpoint + coremltools) -- skips cleanly if the
// fixture isn't present, same convention as test_e2e_qwen3_q8_0.cpp / test_e2e_vits_lua_driver.cpp etc.
// To (re)generate the fixture: `~/.venvs/piper/bin/python3 export_lfm2_monolithic.py` from the repo root
// (writes lfm2_350m_monolithic.gguf there), or point LOOM_LFM2_MONOLITHIC_GGUF at an existing copy.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;

bool path_exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

// Real HF top-1 token at the last sequence position, captured directly from
// `AutoModelForCausalLM.from_pretrained("/home/flavio/Dev/models/lfm2-350m")` for these exact prompts (see
// the conversation/PR description for the capture script) -- both margins over the 2nd-place token
// (0.135 and 2.873 logit units respectively) are comfortably larger than the ~0.003 max abs logit diff
// item 1's resolution measured between this exporter's output and HF, so argmax should be robust to
// ordinary fp32-rounding noise.
struct Case {
    std::vector<double> prompt;
    int32_t expected_top1;
};

const Case kCases[] = {
    {{1, 2, 3}, 3523},
    {{1, 2, 3, 4, 5, 6, 7}, 2},
};

bool run_gguf_case(const std::string& gguf_path) {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_path, backend.get());
    const std::string driver_script = model->kv_str("model.driver_script");
    LOOM_CHECK(!driver_script.empty());

    loom::LoomLuaBridge bridge(backend.get());
    // Monolithic exports have exactly one topology (named "main_topo"). Register whatever topologies
    // the file actually declares instead of assuming a single hardcoded name, so this also works
    // unmodified against a modular-profile export (one topology per prefix/layer_i/suffix_i slice).
    for (const std::string& mod_name : model->topology_names()) {
        bridge.register_module(mod_name, *model, loom::GraphTopology::parse(model->topology_json(mod_name)), nullptr);
    }
    bridge.load_script(driver_script);

    bool all_ok = true;
    for (const Case& c : kCases) {
        loom::LoomLuaBridge::Value result = bridge.call("main", {{"tokens", c.prompt}});
        const auto got = static_cast<int32_t>(std::get<double>(result));
        std::fprintf(stderr, "'%s' prompt of %zu tokens: expected top-1 %d, got %d\n", gguf_path.c_str(),
                     c.prompt.size(), c.expected_top1, got);
        LOOM_CHECK(got == c.expected_top1);
        all_ok = all_ok && (got == c.expected_top1);
    }
    return all_ok;
}

} // namespace

int main() {
    const char* mono_env = std::getenv("LOOM_LFM2_MONOLITHIC_GGUF");
    const std::string mono_path = mono_env != nullptr ? mono_env : "lfm2_350m_monolithic.gguf";

    if (!path_exists(mono_path)) {
        std::fprintf(stderr,
                      "skipping: '%s' not found (set LOOM_LFM2_MONOLITHIC_GGUF, or run "
                      "export_lfm2_monolithic.py from the repo root to produce it)\n",
                      mono_path.c_str());
        return kSkipReturnCode;
    }

    run_gguf_case(mono_path);

    LOOM_TEST_REPORT_AND_RETURN();
}
