// Validates LoomLuaBridge::l_run_recurrent (EXPORT-IMPROVEMENT-BACKLOG.md item 4): a REAL bidirectional
// torch.nn.LSTM, traced through the standard ct.convert(...) pipeline and reduced to per-timestep cell
// topologies via tools/loom_mil_compiler/recurrent.py's build_lstm_cell_topologies, driven here through
// loom.run_recurrent -- the generic, topology-name-driven generalization of BiLstmStepper's own
// per-timestep loop (src/core/bilstm_stepper.cpp), which today can only be constructed directly in C++ by
// hand-written drivers (styletts2_driver.cpp/kokoro_driver.cpp), not reached from a GGUF's own
// driver_script the way loom.run_subgraph reaches an ordinary topology.
//
// The fixture GGUF carries its own PyTorch-computed reference output as KV metadata (this test binary has
// no PyTorch available to it) -- see export_lstm_test_fixture.py's own top comment.
//
// Not generated at ctest time (needs coremltools/torch) -- skips cleanly if the fixture isn't present,
// same convention as test_e2e_lfm2_mil_export.cpp. To (re)generate:
// `~/.venvs/piper/bin/python3 tools/loom_mil_compiler/export_lstm_test_fixture.py` from the repo root
// (writes lstm_recurrent_test.gguf there), or point LOOM_LSTM_RECURRENT_GGUF at an existing copy.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
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

// A single forward pass (h_fwd/c_fwd) plus, for the reverse direction, a second pass (h_bwd/c_bwd) --
// concatenated per-timestep along the hidden axis, matching torch.nn.LSTM(bidirectional=True)'s own
// output convention (and MIL's own lstm op docstring: "[b, :H] and [b, H:] represent forward and
// reverse direction values respectively").
const char* kDriverScript = R"lua(
function infer(inputs)
    local out_fwd = loom.run_recurrent('fwd', inputs.sequence, inputs.seq_len, inputs.input_dim,
                                        inputs.hidden_dim, false)
    local out_bwd = loom.run_recurrent('bwd', inputs.sequence, inputs.seq_len, inputs.input_dim,
                                        inputs.hidden_dim, true)
    local result = {}
    local idx = 1
    for t = 0, inputs.seq_len - 1 do
        for k = 1, inputs.hidden_dim do
            result[idx] = out_fwd[t * inputs.hidden_dim + k]
            idx = idx + 1
        end
        for k = 1, inputs.hidden_dim do
            result[idx] = out_bwd[t * inputs.hidden_dim + k]
            idx = idx + 1
        end
    end
    return result
end
)lua";

} // namespace

int main() {
    const char* env = loom_test::fixture_env("LOOM_LSTM_RECURRENT_GGUF");
    const std::string gguf_path = env != nullptr ? env : "lstm_recurrent_test.gguf";

    if (!path_exists(gguf_path)) {
        std::fprintf(stderr,
                      "skipping: '%s' not found (set LOOM_LSTM_RECURRENT_GGUF, or run "
                      "tools/loom_mil_compiler/export_lstm_test_fixture.py from the repo root to produce "
                      "it)\n",
                      gguf_path.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_path, backend.get());

    const auto seq_len = static_cast<uint32_t>(model->kv_i32("test.seq_len", 0));
    const auto input_dim = static_cast<uint32_t>(model->kv_i32("test.input_dim", 0));
    const auto hidden_dim = static_cast<uint32_t>(model->kv_i32("test.hidden_dim", 0));
    LOOM_CHECK(seq_len > 0 && input_dim > 0 && hidden_dim > 0);

    const nlohmann::json sequence_json = nlohmann::json::parse(model->kv_str("test.input_sequence"));
    const nlohmann::json reference_json = nlohmann::json::parse(model->kv_str("test.reference_output"));
    LOOM_CHECK(sequence_json.size() == static_cast<size_t>(seq_len) * input_dim);
    LOOM_CHECK(reference_json.size() == static_cast<size_t>(seq_len) * 2 * hidden_dim);

    std::vector<double> sequence = sequence_json.get<std::vector<double>>();
    std::vector<double> reference = reference_json.get<std::vector<double>>();

    loom::LoomLuaBridge bridge(backend.get());
    for (const std::string& mod_name : model->topology_names()) {
        bridge.register_module(mod_name, *model, loom::GraphTopology::parse(model->topology_json(mod_name)), nullptr);
    }
    bridge.load_script(kDriverScript);

    loom::LoomLuaBridge::Value result = bridge.call("infer", {
        {"sequence", sequence},
        {"seq_len", static_cast<double>(seq_len)},
        {"input_dim", static_cast<double>(input_dim)},
        {"hidden_dim", static_cast<double>(hidden_dim)},
    });
    const auto got = std::get<std::vector<double>>(result);
    LOOM_CHECK(got.size() == reference.size());

    double max_diff = 0.0;
    for (size_t i = 0; i < got.size() && i < reference.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(got[i] - reference[i]));
    }
    std::fprintf(stderr, "loom.run_recurrent vs real torch.nn.LSTM(bidirectional=True): max diff %.8f over %zu values\n",
                 max_diff, got.size());
    LOOM_CHECK(max_diff < 1e-3);

    LOOM_TEST_REPORT_AND_RETURN();
}
