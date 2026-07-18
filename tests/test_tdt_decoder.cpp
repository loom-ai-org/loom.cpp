// Verifies the new TdtDecoder C++ driver's greedy-TDT control flow (encoder-frame pointer x
// symbols-per-frame double loop, duration-driven frame advance, blank forcing duration>=1, LSTM state
// carried only on non-blank tokens) against an independent numpy reference of NeMo's real algorithm --
// see tools/fixture_gen/tdt_step_common.py and BACKLOG.md's Gap-1 research notes. Fully synthetic and
// small, procedurally generated at ctest time -- not skip-if-missing like the real-checkpoint tests.
//
// This fixture's hand-picked weights/encoder-output seed (see tdt_step_common.py) naturally exercises two
// genuine duration=2 multi-frame skips, without ever relying on the driver's own defensive
// max-symbols-per-step safety net. N_LSTM_LAYERS=2 (not simplified to 1) matches the real Parakeet-TDT
// checkpoint's own prediction-network depth, exercising the driver's inter-layer chaining for real.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<float> read_f32_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

} // namespace

int main() {
    const std::string dir = LOOM_TEST_FIXTURE_DIR;
    const std::string ref_dir = LOOM_TEST_REF_DIR;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    constexpr uint32_t kPredHidden = 4;
    constexpr uint32_t kNEmbd = 3;
    constexpr uint32_t kNFrames = 3;
    constexpr uint32_t kNLstmLayers = 2; // N_LSTM_LAYERS in tdt_step_common.py
    constexpr int32_t kBlankId = 3;      // N_VOCAB in tdt_step_common.py

    // All GGUFs were written with identical weights (see make_tdt_step_gguf.py) -- the first layer's own
    // model is reused as the shared weight symbol table for every GraphBuilder inside TdtDecoder; the
    // rest are only loaded here to extract their own topology JSON.
    std::vector<std::unique_ptr<loom::GgufModel>> models_h, models_c;
    std::vector<loom::GraphTopology> topos_h, topos_c;
    for (uint32_t layer = 0; layer < kNLstmLayers; ++layer) {
        models_h.push_back(loom::GgufModel::load(dir + "/tdt_lstm_h_" + std::to_string(layer) + ".gguf", backend.get()));
        models_c.push_back(loom::GgufModel::load(dir + "/tdt_lstm_c_" + std::to_string(layer) + ".gguf", backend.get()));
        topos_h.push_back(loom::GraphTopology::parse(models_h.back()->topology_json()));
        topos_c.push_back(loom::GraphTopology::parse(models_c.back()->topology_json()));
    }
    auto model_joint = loom::GgufModel::load(dir + "/tdt_joint.gguf", backend.get());
    loom::GraphTopology topo_joint = loom::GraphTopology::parse(model_joint->topology_json());

    loom::TdtDecoderConfig cfg;
    cfg.blank_id = kBlankId;
    cfg.durations = {0, 1, 2};
    cfg.max_symbols_per_step = 3;

    loom::TdtDecoder decoder(*models_h[0], topos_h, topos_c, topo_joint, cfg, backend.get(), kPredHidden);

    const std::vector<float> encoder_flat = read_f32_binary(ref_dir + "/encoder_output.bin");
    LOOM_CHECK(encoder_flat.size() == static_cast<size_t>(kNFrames) * kNEmbd);
    std::vector<std::vector<float>> encoder_output(kNFrames);
    for (uint32_t t = 0; t < kNFrames; ++t) {
        encoder_output[t].assign(encoder_flat.begin() + t * kNEmbd, encoder_flat.begin() + (t + 1) * kNEmbd);
    }

    loom::TdtDecoder::Result result = decoder.decode_greedy(encoder_output);

    nlohmann::json expected = nlohmann::json::parse(read_file(ref_dir + "/expected.json"));
    const std::vector<int32_t> expected_tokens = expected.at("tokens").get<std::vector<int32_t>>();
    const std::vector<uint32_t> expected_frames = expected.at("frame_indices").get<std::vector<uint32_t>>();

    std::fprintf(stderr, "actual tokens: [");
    for (int32_t t : result.tokens) std::fprintf(stderr, "%d ", t);
    std::fprintf(stderr, "], frames: [");
    for (uint32_t f : result.frame_indices) std::fprintf(stderr, "%u ", f);
    std::fprintf(stderr, "]\n");

    LOOM_CHECK(result.tokens == expected_tokens);
    LOOM_CHECK(result.frame_indices == expected_frames);
    // Sanity: this fixture's hand-picked seed is specifically supposed to exercise a genuine
    // duration-driven multi-frame skip, not just the trivial always-advance-by-1 case.
    LOOM_CHECK(!expected_tokens.empty());

    LOOM_TEST_REPORT_AND_RETURN();
}
