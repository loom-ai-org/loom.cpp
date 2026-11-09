// The engine's recurrent sweep stepping a GRU cell: `h_prev` in, `h_new` out, no cell state.
//
// `cell_sweep` carried h AND c for every cell until AuroraTTS-100K's BiGRU; a GRU's whole state is h,
// so its cell declares neither `c_prev` nor `c_new`, and the sweep tells the two shapes apart by that.
// Checked here against a float64 numpy GRU (tests/fixtures/make_gru_sweep_gguf.py) in both directions
// and through the interleaving bidirectional binding, plus the shape the sweep must refuse: two outputs
// with no `c_prev`, which would silently drop a cell state.
//
// And `carry` (ADR-104): an LSTM cell against a float64 numpy LSTM, then each cell swept in two calls that
// share one carry table -- the one call over the whole sequence, with the final h in the table. Dropping
// the LSTM's `c` between the calls must change the answer, and a carry of the wrong width is refused.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// make_gru_sweep_gguf.py's HIDDEN: the carry holds one state this wide.
constexpr size_t kHidden = 4;

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

    {
        const auto got = std::get<std::vector<double>>(session.bridge().call("lstm_forward", args));
        const double w = worst(got, json("test.lstm_forward"));
        std::fprintf(stderr, "lstm_forward: %zu values, worst |d| = %.3g\n", got.size(), w);
        LOOM_CHECK(w < 1e-5);
    }
    for (const auto& [fn, whole_fn] : {std::pair<const char*, const char*>{"lstm_split", "lstm_forward"},
                                       {"gru_split", "forward"}}) {
        const auto whole = std::get<std::vector<double>>(session.bridge().call(whole_fn, args));
        const auto got = std::get<std::vector<double>>(session.bridge().call(fn, args));
        LOOM_CHECK(got.size() == whole.size() + kHidden);
        if (got.size() != whole.size() + kHidden) continue;
        const std::vector<double> rows(got.begin(), got.begin() + whole.size());
        const std::vector<double> carried(got.begin() + whole.size(), got.end());
        const std::vector<double> last(whole.end() - kHidden, whole.end());
        std::fprintf(stderr, "%s: two calls against one %s, carry against the last row %s\n", fn,
                     rows == whole ? "identical" : "DIFFER", carried == last ? "identical" : "DIFFERS");
        LOOM_CHECK(rows == whole);
        LOOM_CHECK(carried == last);
    }
    {
        const auto whole = std::get<std::vector<double>>(session.bridge().call("lstm_forward", args));
        const auto dropped = std::get<std::vector<double>>(session.bridge().call("lstm_split_c", args));
        const std::vector<double> tail(whole.begin() + 2 * kHidden, whole.end());
        std::fprintf(stderr, "lstm_split_c: c dropped, worst |d| = %.3g\n", worst(dropped, tail));
        LOOM_CHECK(worst(dropped, tail) > 1e-3);
    }
    {
        bool bad = false;
        try {
            session.bridge().call("bad_carry", args);
        } catch (const std::exception& e) {
            bad = std::string(e.what()).find("carry.h") != std::string::npos;
        }
        LOOM_CHECK(bad);
    }

    bool refused = false;
    try {
        session.bridge().call("two_outputs_no_c", args);
    } catch (const std::exception& e) {
        refused = std::string(e.what()).find("c_prev") != std::string::npos;
    }
    LOOM_CHECK(refused);
    LOOM_TEST_REPORT_AND_RETURN();
}
