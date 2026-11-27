// `loom.run_bi_lstm_and_retain`: a two-layer BiLSTM as one graph per layer, against a float64 numpy
// BiLSTM (tests/fixtures/make_bilstm_fused_gguf.py) in both retained layouts. Layer 0 has loom-exporter's
// gate order and reads a Lua array; layer 1 has torch.nn.LSTM's gate order and reads layer 0 by
// reference. The fused result must also match the per-timestep sweep, and a cell that is not an LSTM
// is refused.

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
    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/bilstm_fused.gguf";
    auto model = loom::GgufModel::load(path, backend.get());
    loom::Session session(*model, backend.get());

    const auto json = [&](const char* key) {
        return nlohmann::json::parse(model->kv_str(key)).get<std::vector<double>>();
    };
    const std::unordered_map<std::string, loom::LoomLuaBridge::Value> args = {{"sequence", json("test.sequence")}};
    const auto call = [&](const char* fn) { return std::get<std::vector<double>>(session.bridge().call(fn, args)); };

    const struct { const char* fused; const char* stepped; const char* ref; } cases[] = {
        {"fused", "stepped", "test.rows"},
        {"fused_layout_a", "stepped_layout_a", "test.layout_a"},
    };
    for (const auto& c : cases) {
        const auto got = call(c.fused);
        const double w_ref = worst(got, json(c.ref));
        const double w_step = worst(got, call(c.stepped));
        std::fprintf(stderr, "%s: %zu values, worst |d| = %.3g vs numpy, %.3g vs %s\n", c.fused, got.size(), w_ref,
                     w_step, c.stepped);
        LOOM_CHECK(w_ref < 1e-5);
        LOOM_CHECK(w_step < 1e-5);
    }

    bool refused = false;
    try {
        session.bridge().call("refuses_gru", args);
    } catch (const loom::Error& e) {
        refused = std::string(e.what()).find("not the exporter's LSTM cell") != std::string::npos;
    }
    LOOM_CHECK(refused);
    return 0;
}
