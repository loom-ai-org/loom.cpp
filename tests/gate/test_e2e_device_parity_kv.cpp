// The KV cache, the decode loop and the scheduler, all on a device at once (BACKLOG.md P4.7).
//
// `test_e2e_device_parity.cpp` compares one big forward and stops there. Everything a cached
// autoregressive decode adds is untested by it, and all of it is exactly what a device makes newly
// able to break:
//
//   * the cache lives in the PRIMARY backend's buffer, and every attention layer reads and writes it
//     through views the scheduler did not place;
//   * `KvCache::fill_cell_index` rewrites the cell-index tensor between steps through
//     `ggml_backend_tensor_set`, on a graph the scheduler has already allocated and split;
//   * P4.0.15's bucketing means the graph is REUSED across a run of decode steps, so the scheduler's
//     split plan has to stay valid while `n_past` moves underneath it;
//   * a decode step's mask is placed into a bucket-padded input, host side.
//
// The oracle is the same one `test_e2e_causal_lm_infer_with_past.cpp` uses and for the same reason:
// N full prefills over a growing prompt is the slow, obviously-correct way to generate, and it touches
// none of the machinery above. Here it runs on the CPU while the cached path runs on the device, so a
// disagreement means one of those four things broke on the way to the GPU.
//
// **Greedy decoding is the amplifier that makes a token comparison legitimate here**, which is not
// generally true of token oracles: one wrong logit becomes a different token and then a different
// suffix, so twelve agreeing tokens is twelve chained comparisons of an argmax over a whole vocabulary
// and not one lucky match. The final row is additionally compared elementwise WHEN the driver returns
// one -- the auto-generated causal-LM driver returns a bare token id, in which case the token chain is
// the whole of this test and the elementwise comparison lives in test_e2e_device_parity.cpp instead.
//
// Skips (77) on a missing fixture, on a build with no device backend, and on a GGUF with no
// `infer_with_past` -- an export predating KV-CACHE.md stage 3, or one whose graph carries cross-step
// state no store holds.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"
#include "loom/core/conv_state_cache.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;
constexpr int kNewTokens = 12;

// Arbitrary in-vocabulary ids; what matters is only that both paths get the same ones.
const std::vector<double> kPrompt{1, 2, 3, 4, 5};

bool has_device() {
    for (const loom::DeviceInfo& d : loom::available_devices()) {
        if (!d.is_cpu) return true;
    }
    return false;
}

int32_t as_token(const loom::LoomLuaBridge::Value& v) {
    if (const auto* d = std::get_if<double>(&v)) return static_cast<int32_t>(*d);
    const auto& arr = std::get<std::vector<double>>(v);
    return arr.empty() ? -1 : static_cast<int32_t>(arr.back());
}

// One loaded session: the model, its caches and a bridge with every topology registered, all on one
// device. Held together because the bridge holds non-owning references to the rest.
struct Session {
    std::unique_ptr<loom::GgufModel> model;
    std::unique_ptr<loom::KvCache> kv_cache;
    std::unique_ptr<loom::ConvStateCache> conv_state;
    std::unique_ptr<loom::LoomLuaBridge> bridge;
};

Session open(const std::string& path, loom::Backends backends) {
    Session s;
    s.model = loom::GgufModel::load(path, backends);
    s.bridge = std::make_unique<loom::LoomLuaBridge>(backends);
    for (const std::string& name : s.model->topology_names()) {
        loom::GraphTopology topo = loom::GraphTopology::parse(s.model->topology_json(name));
        // Both stores are sized from the file's own declared geometry and allocated on the primary
        // backend -- the device, here. They are persistent state a graph reads and writes through
        // views, so they are deliberately not something the scheduler may place.
        if (topo.uses_kv_cache() && !s.kv_cache) s.kv_cache = loom::make_kv_cache(*s.model, backends);
        if (topo.uses_conv_state() && !s.conv_state) {
            s.conv_state = loom::make_conv_state_cache(*s.model, backends);
        }
        loom::KvCache* kv = topo.uses_kv_cache() ? s.kv_cache.get() : nullptr;
        loom::ConvStateCache* conv = topo.uses_conv_state() ? s.conv_state.get() : nullptr;
        s.bridge->register_module(name, *s.model, std::move(topo), kv, conv);
    }
    s.bridge->load_script(s.model->kv_str("model.driver_script"));
    return s;
}

} // namespace

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_CAUSAL_LM_KV_GGUF");
    if (gguf_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_CAUSAL_LM_KV_GGUF to a GGUF produced by "
                              "`loom-export <hf-causal-lm-dir>` to run this check\n");
        return kSkipReturnCode;
    }
    if (!has_device()) {
        std::fprintf(stderr, "skipping: this build reaches no GPU/accelerator device (configure with "
                              "-DGGML_VULKAN=ON, -DGGML_CUDA=ON or -DGGML_METAL=ON)\n");
        return kSkipReturnCode;
    }

    loom::Device cpu = loom::Device::open("cpu");
    loom::Device gpu = loom::Device::open("gpu");
    std::fprintf(stderr, "comparing %s (%s) against %s\n", gpu.name().c_str(),
                 gpu.description().c_str(), cpu.name().c_str());

    Session host = open(gguf_env, cpu.backends());
    if (host.model->kv_str("model.driver_script").find("function infer_with_past(") ==
        std::string::npos) {
        std::fprintf(stderr, "skipping: '%s' has no infer_with_past entry\n", gguf_env);
        return kSkipReturnCode;
    }
    LOOM_CHECK(host.kv_cache != nullptr);

    Session device = open(gguf_env, gpu.backends());
    LOOM_CHECK(device.kv_cache != nullptr);

    // --- Oracle: N full prefills over a growing prompt, on the CPU -----------------------------------
    std::vector<int32_t> expected;
    {
        std::vector<double> grown = kPrompt;
        for (int i = 0; i < kNewTokens; ++i) {
            const int32_t next = as_token(host.bridge->call("infer", {{"tokens", grown}}));
            expected.push_back(next);
            grown.push_back(static_cast<double>(next));
        }
    }
    LOOM_CHECK(expected.size() == static_cast<size_t>(kNewTokens));

    // --- The claim: one prefill plus N-1 cached steps, on the device ---------------------------------
    std::vector<int32_t> actual;
    {
        const loom::LoomLuaBridge::Value result = device.bridge->call("infer_with_past", {
            {"tokens", kPrompt},
            {"max_new_tokens", static_cast<double>(kNewTokens)},
        });
        for (double d : std::get<std::vector<double>>(result)) {
            actual.push_back(static_cast<int32_t>(d));
        }
    }

    std::fprintf(stderr, "cpu  iterated infer      :");
    for (int32_t t : expected) std::fprintf(stderr, " %d", t);
    std::fprintf(stderr, "\ndevice infer_with_past  :");
    for (int32_t t : actual) std::fprintf(stderr, " %d", t);
    std::fprintf(stderr, "\n");

    for (const auto& m : device.bridge->device_report()) {
        std::fprintf(stderr, "  module %-24s splits=%-4d device=%zu cpu=%zu\n", m.module.c_str(),
                     m.splits, m.device_nodes, m.fallback_nodes);
    }
    // The device did the work rather than the fallback quietly doing all of it.
    size_t device_nodes = 0;
    size_t fallback_nodes = 0;
    for (const auto& m : device.bridge->device_report()) {
        device_nodes += m.device_nodes;
        fallback_nodes += m.fallback_nodes;
    }
    LOOM_CHECK(device_nodes > 0);
    LOOM_CHECK(device_nodes > fallback_nodes);

    LOOM_CHECK(actual.size() == expected.size());
    LOOM_CHECK(actual == expected);

    // --- And the logits behind the last of those decisions, elementwise ------------------------------
    // The tokens above are twelve chained argmaxes; this is the margin they were decided by. A cache
    // that were subtly wrong on a device -- a cell written to the right slot with slightly wrong bytes --
    // could still produce the same twelve tokens, and would show up here.
    {
        std::vector<double> grown = kPrompt;
        for (int32_t t : expected) grown.push_back(static_cast<double>(t));
        // Held in named locals: `call` returns by value, so a reference bound into a `std::get` on the
        // temporary would outlive the variant holding the vector.
        const loom::LoomLuaBridge::Value host_value = host.bridge->call("infer", {{"tokens", grown}});
        const loom::LoomLuaBridge::Value device_value = device.bridge->call("infer", {{"tokens", grown}});
        // A driver that returns a bare next-token id rather than a row leaves nothing to compare here;
        // that is a property of the export, not a failure, so it is reported and skipped over. Asked
        // with get_if rather than by size, because a variant holding a double has no size to ask for --
        // and std::get on the wrong alternative throws rather than answering.
        const auto* host_row_p = std::get_if<std::vector<double>>(&host_value);
        const auto* device_row_p = std::get_if<std::vector<double>>(&device_value);
        if (host_row_p != nullptr && device_row_p != nullptr && host_row_p->size() > 1 &&
            host_row_p->size() == device_row_p->size()) {
            const std::vector<double>& host_row = *host_row_p;
            const std::vector<double>& device_row = *device_row_p;
            double peak = 0.0;
            double worst = 0.0;
            for (size_t i = 0; i < host_row.size(); ++i) {
                peak = std::max(peak, std::fabs(host_row[i]));
                LOOM_CHECK(std::isfinite(device_row[i]));
                worst = std::max(worst, std::fabs(host_row[i] - device_row[i]));
            }
            std::fprintf(stderr, "final row: %zu values, max |device - cpu| = %.3e (peak %.3e, "
                                  "relative %.3e)\n", host_row.size(), worst, peak, worst / peak);
            LOOM_CHECK(peak > 0.0);
            // Same reasoning as the other parity test's bound: fp32 reduction order, nothing else. This
            // graph has no log-mel front end to amplify it, so it is a tighter number.
            LOOM_CHECK(worst / peak < 5e-3);
        } else {
            std::fprintf(stderr, "final row: this driver's `infer` returns a token id, not a logits "
                                  "row -- the elementwise half of this test does not apply\n");
        }
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
