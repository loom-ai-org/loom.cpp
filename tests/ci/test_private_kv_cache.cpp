// `kv_cache_scope: "private"`: a module that gets a KV cache of its own instead of the session's.
//
// **The failure this exists for is silent.** Classifier-free guidance runs one decoder twice per step
// -- once on the conditional input, once on the unconditional one -- over two histories that must not
// see each other. Registered against one shared cache, step t of the second stream writes over the
// cell the first just wrote and then attends to a mixture of the two. Nothing raises: the shapes are
// right, the graph is right, and what comes out is a plausible token sequence that is not the model's.
// So the check here is not "does it run" but "does interleaving change the answer", and the arm below
// that shares a cache is in the test to prove that it would.
//
// Whether a module needs a cache is DERIVED from its graph (`uses_kv_cache()`); whether it needs its
// own cannot be, and this is why -- two modules running one topology are two streams when a driver
// runs them side by side and two phases when it runs them in sequence, and only the driver knows
// which. Hence a declaration, and hence a test that the declaration is what routes the cache.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

const std::vector<double> kStreamA = {1, 3, 5, 2, 7, 4};
const std::vector<double> kStreamB = {6, 6, 0, 1, 3, 3};

using Args = std::unordered_map<std::string, loom::LoomLuaBridge::Value>;

// `interleaved` returns stream A's whole run followed by stream B's, both of the same length.
std::vector<double> half(const std::vector<double>& v, size_t which) {
    LOOM_CHECK(v.size() % 2 == 0);
    const size_t n = v.size() / 2;
    return std::vector<double>(v.begin() + static_cast<long>(which * n),
                                v.begin() + static_cast<long>((which + 1) * n));
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    LOOM_CHECK(a.size() == b.size());
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::abs(a[i] - b[i]));
    return worst;
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/cfg_streams.gguf";
    auto model = loom::GgufModel::load(path, backend.get());

    // The declaration reaches the parsed topology at all -- checked before anything is run, because
    // every assertion below is meaningless if this silently defaulted.
    LOOM_CHECK(!loom::GraphTopology::parse(model->topology_json("cond")).private_kv_cache);
    LOOM_CHECK(loom::GraphTopology::parse(model->topology_json("uncond")).private_kv_cache);

    // --- 1. Each stream on its own, which is the oracle. ---
    std::vector<double> alone_a, alone_b;
    {
        loom::Session session(*model, backend.get());
        alone_a = std::get<std::vector<double>>(
            session.bridge().call("alone", {{"a", kStreamA}, {"uncond", 0.0}}));
    }
    {
        loom::Session session(*model, backend.get());
        alone_b = std::get<std::vector<double>>(
            session.bridge().call("alone", {{"a", kStreamB}, {"uncond", 1.0}}));
    }
    LOOM_CHECK(alone_a.size() == kStreamA.size() * 8);   // 8 classes per step
    LOOM_CHECK(alone_b.size() == kStreamB.size() * 8);
    // Two different prompts through the same weights: if these agreed, "the streams did not interfere"
    // would be satisfied by a cache that ignored its input entirely.
    LOOM_CHECK(alone_a != alone_b);

    // --- 2. Interleaved through a Session, which honours the declaration. Each stream must be exactly
    //        what it was alone -- a stream whose cache is its own cannot tell that another ran. ---
    {
        loom::Session session(*model, backend.get());
        const auto both = std::get<std::vector<double>>(
            session.bridge().call("interleaved", {{"a", kStreamA}, {"b", kStreamB}}));
        LOOM_CHECK(both.size() == alone_a.size() + alone_b.size());
        const auto got_a = half(both, 0);
        const auto got_b = half(both, 1);
        std::fprintf(stderr, "private caches: max |diff| A %.3e, B %.3e\n",
                     max_abs_diff(got_a, alone_a), max_abs_diff(got_b, alone_b));
        // BIT-identical, not close: the same graph over the same cells is the same arithmetic, and a
        // tolerance here would be room for exactly the interference this is checking for.
        LOOM_CHECK(got_a == alone_a);
        LOOM_CHECK(got_b == alone_b);
    }

    // --- 3. THE SAME RUN WITH ONE SHARED CACHE, so the check above is known to be able to fail.
    //        This is the arrangement `Session` would build if it ignored the declaration, wired here by
    //        hand: both modules registered against one cache. It must produce a different answer --
    //        and if it ever stops doing so, check 2 has become a test of nothing. ---
    {
        loom::LoomLuaBridge bridge(backend.get());
        auto shared = loom::make_kv_cache(*model, backend.get());
        for (const std::string& name : model->topology_names()) {
            bridge.register_module(name, *model,
                                   loom::GraphTopology::parse(model->topology_json(name)),
                                   shared.get(), nullptr);
        }
        bridge.load_script(model->kv_str("model.driver_script"));
        const auto both = std::get<std::vector<double>>(
            bridge.call("interleaved", {{"a", kStreamA}, {"b", kStreamB}}));
        const auto shared_a = half(both, 0);
        const auto shared_b = half(both, 1);
        const double drift_a = max_abs_diff(shared_a, alone_a);
        const double drift_b = max_abs_diff(shared_b, alone_b);
        std::fprintf(stderr, "one shared cache: max |diff| A %.3e, B %.3e\n", drift_a, drift_b);
        // A real divergence, not a rounding one -- the number is printed so that a fixture change
        // which quietly shrank it is visible rather than merely still passing.
        LOOM_CHECK(std::max(drift_a, drift_b) > 1e-2);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
