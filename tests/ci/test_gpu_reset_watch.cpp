// The GPU reset watch (loom/core/gpu_reset_watch.h, ADR-140, loom.cpp#124), with the kernel's counter
// replaced by one this test moves: a graph during which the count moves must fail, through the same
// driver call a model makes, and one during which it does not must run as before. The kernel half (the
// v3d ioctl, the log fallback) is the Pi 4 gate in ADR-140, with a planted hang.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <atomic>
#include <string>
#include <unordered_map>

namespace {

const char* kStage = R"JSON({
  "version": 1,
  "inputs": [{"name":"tokens","dtype":"i32","shape":["n_tokens"]}],
  "output": "cur",
  "nodes": [{"op": "GET_ROWS", "inputs": ["token_embd.weight", "tokens"], "outputs": ["cur"]}]
})JSON";

const char* kScript = R"lua(
    function infer(inputs)
        return loom.run_subgraph('stage', {n_tokens = #inputs.tokens, n_past = 0}, {tokens = inputs.tokens})
    end
)lua";

bool call_fails_with(loom::LoomLuaBridge& bridge, const std::string& needle) {
    try {
        bridge.call("infer", {{"tokens", std::vector<double>{1, 2, 3}}});
    } catch (const loom::Error& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

}  // namespace

int main() {
    namespace watch = loom::gpu_reset_watch;
    ggml_backend_ptr backend(loom_test::cpu_backend());
    auto model = loom::GgufModel::load(std::string(LOOM_TEST_FIXTURE_DIR) + "/builder_test.gguf", backend.get());
    loom::LoomLuaBridge bridge(backend.get());
    bridge.register_module("stage", *model, loom::GraphTopology::parse(kStage));
    bridge.load_script(kScript);
    const std::unordered_map<std::string, loom::LoomLuaBridge::Value> inputs{{"tokens", std::vector<double>{1, 2, 3}}};

    // Unarmed: the default for every device but a V3D, and nothing is read.
    LOOM_CHECK(!watch::armed());
    LOOM_CHECK(watch::count() == -1);
    bridge.call("infer", inputs);

    // A device that is not a V3D does not arm it.
    watch::arm("NVIDIA GeForce RTX 5090");
    watch::arm("AMD Radeon Vega 3 Graphics (RADV RAVEN)");
    LOOM_CHECK(!watch::armed());

    // Armed and still: graphs run as before.
    std::atomic<int64_t> resets{4};
    watch::arm_for_test([&] { return resets.load(); });
    LOOM_CHECK(watch::armed());
    bridge.call("infer", inputs);
    bridge.call("infer", inputs);

    // A reset during the graph (the count moves between the two reads): the call fails, naming it.
    watch::arm_for_test([&] { return resets.fetch_add(1); });
    LOOM_CHECK(call_fails_with(bridge, "the GPU was reset while a graph on CPU ran"));

    // The same check, without the bridge in between: a DeviceLostError, the type a host can catch.
    bool typed = false;
    try {
        const watch::Check check;
        check.verify("this graph");
    } catch (const loom::DeviceLostError&) {
        typed = true;
    }
    LOOM_CHECK(typed);

    // A source that cannot be read (-1) is not a reset.
    watch::arm_for_test([] { return int64_t{-1}; });
    bridge.call("infer", inputs);

    // Disarmed again, and the next call runs.
    watch::arm_for_test({});
    LOOM_CHECK(!watch::armed());
    bridge.call("infer", inputs);

    // The kernel-log fallback (kernels before 6.18), on the lines the Pi 4 logged for the planted hang.
    // klogctl prefixes each line with its priority, as `<3>`.
    const std::string quiet = "<6>[   16.099007] brcmfmac: brcmf_cfg80211_set_power_mgmt: power save enabled\n";
    const std::string reset =
        "<3>[185515.569297] v3d fec00000.v3d: [drm:v3d_reset [v3d]] *ERROR* Resetting GPU for hang.\n"
        "<3>[185515.569392] v3d fec00000.v3d: [drm:v3d_reset [v3d]] *ERROR* V3D_ERR_STAT: 0x00001000\n";
    const std::string again =
        "<3>[185731.002001] v3d fec00000.v3d: [drm:v3d_reset [v3d]] *ERROR* Resetting GPU for hang.\n";
    LOOM_CHECK(watch::log_token(quiet) == 0);
    LOOM_CHECK(watch::log_token(quiet + reset) == 185515569297 + 1);
    LOOM_CHECK(watch::log_token(quiet + reset + quiet) == watch::log_token(quiet + reset));  // no new reset
    LOOM_CHECK(watch::log_token(quiet + reset + again) == 185731002001 + 1);               // a second one
    // Without printk times: the count of reset lines.
    LOOM_CHECK(watch::log_token("v3d: Resetting GPU for hang.\nv3d: Resetting GPU for hang.\n") == 2);

    LOOM_TEST_REPORT_AND_RETURN();
}
