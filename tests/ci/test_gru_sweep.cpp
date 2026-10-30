// The engine's recurrent sweep stepping a GRU cell: `h_prev` in, `h_new` out, no cell state.
//
// `cell_sweep` carried h AND c for every cell until AuroraTTS-100K's BiGRU; a GRU's whole state is h,
// so its cell declares neither `c_prev` nor `c_new`, and the sweep tells the two shapes apart by that.
// Checked here against a float64 numpy GRU (tests/fixtures/make_gru_sweep_gguf.py) in both directions
// and through the interleaving bidirectional binding, plus the shape the sweep must refuse: two outputs
// with no `c_prev`, which would silently drop a cell state.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

double worst(const std::vector<double>& got, const std::vector<double>& want) {
    if (got.size() != want.size()) return INFINITY;
    double w = 0.0;
    for (size_t i = 0; i < got.size(); ++i) w = std::max(w, std::abs(got[i] - want[i]));
    return w;
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/gru_sweep.gguf";
    auto model = loom::GgufModel::load(path, backend.get());
    loom::Session session(*model, backend.get());

    const auto json = [&](const char* key) {
        return nlohmann::json::parse(model->kv_str(key)).get<std::vector<double>>();
    };
    const std::unordered_map<std::string, loom::LoomLuaBridge::Value> args = {{"sequence", json("test.sequence")}};

    const struct { const char* fn; const char* ref; } cases[] = {
        {"forward", "test.forward"}, {"backward", "test.backward"}, {"bidirectional", "test.bidirectional"},
    };
    for (const auto& c : cases) {
        const auto got = std::get<std::vector<double>>(session.bridge().call(c.fn, args));
        const double w = worst(got, json(c.ref));
        std::fprintf(stderr, "%s: %zu values, worst |d| = %.3g\n", c.fn, got.size(), w);
        LOOM_CHECK(w < 1e-5);
    }

    bool refused = false;
    try {
        session.bridge().call("two_outputs_no_c", args);
    } catch (const std::exception& e) {
        refused = std::string(e.what()).find("c_prev") != std::string::npos;
    }
    LOOM_CHECK(refused);
    std::fprintf(stderr, "test_gru_sweep: OK\n");
    return 0;
}
