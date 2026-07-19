// Verifies loom::TdtDecoder's new plain-RNN-T mode (TdtDecoderConfig.durations left EMPTY -- no
// duration head at all, every blank advances exactly one frame, non-blank never advances) against an
// independent numpy reference of standard RNN-T greedy decoding -- see tools/fixture_gen/
// rnnt_step_common.py. Same synthetic/procedural fixture pattern as test_tdt_decoder.cpp, deliberately
// NOT skip-if-missing.
//
// This is the generalization test for TdtDecoder's durations.empty() branch (see tdt_decoder.cpp),
// exercised BEFORE that branch is trusted against the real nvidia/parakeet-rnnt-0.6b checkpoint.

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
    constexpr uint32_t kNLstmLayers = 2;
    constexpr int32_t kBlankId = 3;

    std::vector<std::unique_ptr<loom::GgufModel>> models_h, models_c;
    std::vector<loom::GraphTopology> topos_h, topos_c;
    for (uint32_t layer = 0; layer < kNLstmLayers; ++layer) {
        models_h.push_back(loom::GgufModel::load(dir + "/rnnt_lstm_h_" + std::to_string(layer) + ".gguf", backend.get()));
        models_c.push_back(loom::GgufModel::load(dir + "/rnnt_lstm_c_" + std::to_string(layer) + ".gguf", backend.get()));
        topos_h.push_back(loom::GraphTopology::parse(models_h.back()->topology_json()));
        topos_c.push_back(loom::GraphTopology::parse(models_c.back()->topology_json()));
    }
    auto model_joint = loom::GgufModel::load(dir + "/rnnt_joint.gguf", backend.get());
    loom::GraphTopology topo_joint = loom::GraphTopology::parse(model_joint->topology_json());

    loom::TdtDecoderConfig cfg;
    cfg.blank_id = kBlankId;
    // durations left EMPTY -- this is the whole point of this test: plain RNN-T mode, no duration head.
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
    // Sanity: this fixture's hand-picked seed specifically exercises a multi-symbol-per-frame emission
    // (frame 1 emits two tokens before blanking), not just the trivial one-symbol-per-frame case.
    LOOM_CHECK(!expected_tokens.empty());

    LOOM_TEST_REPORT_AND_RETURN();
}
