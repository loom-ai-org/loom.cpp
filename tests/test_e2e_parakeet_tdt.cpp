// Real-model end-to-end test: loads the real nvidia/parakeet-tdt-0.6b-v3 checkpoint (converted by
// tools/convert_nemo/convert_parakeet_tdt.py) and runs (1) the real FastConformer encoder -- a
// non-autoregressive GraphBuilder::build() call, no different in kind from Conformer-CTC's, but exercising
// the real dw_striding subsampling (new CONV_2D_DW primitive) and no-bias/no-xscale differences -- then
// (2) the new TdtDecoder driver's greedy TDT decode loop over that real encoder output, comparing both the
// encoder tensor and the final decoded token/frame-index sequence against
// tools/convert_nemo/reference_forward_parakeet_tdt.py's independent hand-rolled PyTorch computation
// (NeMo's own toolkit is broken in this venv -- a huggingface_hub/transformers version conflict --
// confirmed before choosing to hand-roll the reference, not assumed).
//
// Same "not generated at ctest time, skip cleanly if the real checkpoint isn't prepared" pattern as every
// other real-model test in this suite -- this one needs a real ~2.5GB download.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <sys/stat.h>

namespace {

constexpr int kSkipReturnCode = 77;

bool path_exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

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
    // LOOM_PARAKEET_TDT_DIR must contain gguf/{parakeet_encoder.gguf,parakeet_lstm_h_0.gguf,
    // parakeet_lstm_c_0.gguf,parakeet_lstm_h_1.gguf,parakeet_lstm_c_1.gguf,parakeet_joint.gguf} + ref/
    // {waveform,pos_emb_raw,expected_encoder_output}.bin + expected_decode.json, e.g. produced via:
    //   python3 tools/convert_nemo/convert_parakeet_tdt.py parakeet-tdt-0.6b-v3.nemo $DIR/gguf
    //   python3 tools/convert_nemo/reference_forward_parakeet_tdt.py parakeet-tdt-0.6b-v3.nemo $DIR/ref
    const char* dir_env = std::getenv("LOOM_PARAKEET_TDT_DIR");
    const std::string dir = dir_env != nullptr ? dir_env : "/tmp/parakeet_tdt_model";
    const std::string gguf_dir = dir + "/gguf";
    const std::string ref_dir = dir + "/ref";

    if (!path_exists(gguf_dir + "/parakeet_encoder.gguf") || !path_exists(ref_dir)) {
        std::fprintf(stderr,
                      "skipping: real Parakeet-TDT fixture not found under '%s' (set LOOM_PARAKEET_TDT_DIR "
                      "or see tools/convert_nemo/ to produce one)\n",
                      dir.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    // --- Encoder forward pass ---
    auto enc_model = loom::GgufModel::load(gguf_dir + "/parakeet_encoder.gguf", backend.get());
    LOOM_CHECK(enc_model->hparam_u32("n_head") == 8);
    LOOM_CHECK(enc_model->hparam_u32("n_layers") == 24);
    loom::GraphTopology enc_topo = loom::GraphTopology::parse(enc_model->topology_json());

    constexpr uint32_t kNSamples = 16000;
    constexpr uint32_t kNSubsampled = 13;
    constexpr uint32_t kNPos = 2 * kNSubsampled - 1;
    constexpr uint32_t kNEmbd = 1024;

    loom::GraphBuilder enc_builder(enc_topo, *enc_model, backend.get(), /*kv_cache=*/nullptr);
    const loom::GraphBuilder::BuildResult& enc_result = enc_builder.build({{"n_tokens", kNSamples}, {"n_past", /*n_past=*/0}});

    ggml_tensor* waveform_t = enc_result.input_tensors.at("waveform");
    ggml_tensor* pos_emb_raw_t = enc_result.input_tensors.at("pos_emb_raw");
    ggml_tensor* kq_mask_t = enc_result.input_tensors.at("kq_mask");

    const std::vector<float> waveform = read_f32_binary(ref_dir + "/waveform.bin");
    const std::vector<float> pos_emb = read_f32_binary(ref_dir + "/pos_emb_raw.bin");
    LOOM_CHECK(waveform.size() == kNSamples);
    LOOM_CHECK(pos_emb.size() == static_cast<size_t>(kNEmbd) * kNPos);

    ggml_backend_tensor_set(waveform_t, waveform.data(), 0, waveform.size() * sizeof(float));
    ggml_backend_tensor_set(pos_emb_raw_t, pos_emb.data(), 0, pos_emb.size() * sizeof(float));
    const std::vector<float> zero_mask(static_cast<size_t>(kNSubsampled) * kNSubsampled, 0.0f);
    ggml_backend_tensor_set(kq_mask_t, zero_mask.data(), 0, zero_mask.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), enc_result.graph);

    LOOM_CHECK(static_cast<uint32_t>(enc_result.output->ne[0]) == kNEmbd);
    LOOM_CHECK(static_cast<uint32_t>(enc_result.output->ne[1]) == kNSubsampled);

    std::vector<float> encoder_out_flat(static_cast<size_t>(kNEmbd) * kNSubsampled);
    ggml_backend_tensor_get(enc_result.output, encoder_out_flat.data(), 0, encoder_out_flat.size() * sizeof(float));

    const std::vector<float> expected_encoder_out = read_f32_binary(ref_dir + "/expected_encoder_output.bin");
    LOOM_CHECK(expected_encoder_out.size() == encoder_out_flat.size());
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < encoder_out_flat.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, std::fabs(encoder_out_flat[i] - expected_encoder_out[i]));
    }
    std::fprintf(stderr, "encoder max abs diff = %f\n", static_cast<double>(max_abs_diff));
    // Real 24-layer accumulation, looser than the smaller Conformer-CTC-small's 1e-3 -- same reasoning as
    // test_e2e_qwen3.cpp's own 5e-2 (a real, deep model accumulates more floating-point summation-order
    // drift between numpy/BLAS and ggml's own kernels than a small toy fixture).
    LOOM_CHECK(max_abs_diff <= 5e-2f);

    // --- TdtDecoder greedy decode over the real encoder output ---
    constexpr uint32_t kPredHidden = 640;
    constexpr uint32_t kNLstmLayers = 2;
    constexpr int32_t kBlankId = 8192;

    std::vector<std::unique_ptr<loom::GgufModel>> lstm_models_h, lstm_models_c;
    std::vector<loom::GraphTopology> lstm_topos_h, lstm_topos_c;
    for (uint32_t layer = 0; layer < kNLstmLayers; ++layer) {
        lstm_models_h.push_back(loom::GgufModel::load(
            gguf_dir + "/parakeet_lstm_h_" + std::to_string(layer) + ".gguf", backend.get()));
        lstm_models_c.push_back(loom::GgufModel::load(
            gguf_dir + "/parakeet_lstm_c_" + std::to_string(layer) + ".gguf", backend.get()));
        lstm_topos_h.push_back(loom::GraphTopology::parse(lstm_models_h.back()->topology_json()));
        lstm_topos_c.push_back(loom::GraphTopology::parse(lstm_models_c.back()->topology_json()));
    }
    auto joint_model = loom::GgufModel::load(gguf_dir + "/parakeet_joint.gguf", backend.get());
    loom::GraphTopology joint_topo = loom::GraphTopology::parse(joint_model->topology_json());

    loom::TdtDecoderConfig cfg;
    cfg.blank_id = kBlankId;
    cfg.durations = {0, 1, 2, 3, 4};
    cfg.max_symbols_per_step = 10;

    loom::TdtDecoder decoder(*lstm_models_h[0], lstm_topos_h, lstm_topos_c, joint_topo, cfg, backend.get(), kPredHidden);

    std::vector<std::vector<float>> encoder_output(kNSubsampled);
    for (uint32_t t = 0; t < kNSubsampled; ++t) {
        encoder_output[t].assign(encoder_out_flat.begin() + t * kNEmbd, encoder_out_flat.begin() + (t + 1) * kNEmbd);
    }

    loom::TdtDecoder::Result result = decoder.decode_greedy(encoder_output);

    nlohmann::json expected = nlohmann::json::parse(read_file(ref_dir + "/expected_decode.json"));
    const std::vector<int32_t> expected_tokens = expected.at("tokens").get<std::vector<int32_t>>();
    const std::vector<uint32_t> expected_frames = expected.at("frame_indices").get<std::vector<uint32_t>>();

    std::fprintf(stderr, "actual tokens: [");
    for (int32_t t : result.tokens) std::fprintf(stderr, "%d ", t);
    std::fprintf(stderr, "], frames: [");
    for (uint32_t f : result.frame_indices) std::fprintf(stderr, "%u ", f);
    std::fprintf(stderr, "]\nexpected tokens: [");
    for (int32_t t : expected_tokens) std::fprintf(stderr, "%d ", t);
    std::fprintf(stderr, "], frames: [");
    for (uint32_t f : expected_frames) std::fprintf(stderr, "%u ", f);
    std::fprintf(stderr, "]\n");

    LOOM_CHECK(result.tokens == expected_tokens);
    LOOM_CHECK(result.frame_indices == expected_frames);

    // Plumbing smoke check: detokenize the real decoded ids end to end through the new SentencePiece
    // BPE Vocab path (now embedded in the encoder GGUF -- see tools/convert_nemo/tokenizer_common.py).
    // Not a word-accuracy check (the input waveform is synthetic noise, not real speech) -- just confirms
    // Vocab::decode runs against the real 8192-piece vocab without crashing and produces well-formed text.
    auto vocab = loom::Vocab::load(*enc_model);
    LOOM_CHECK(vocab != nullptr);
    // 8192 real pieces -- the tokenizer.model itself has no "blank" entry at all; blank_id=8192 is purely
    // an artifact of the RNNT/TDT decoder's own embedding table (8193 rows: 8192 pieces + 1 blank row),
    // never a real vocab piece, and TdtDecoder's own result.tokens never contains it regardless.
    LOOM_CHECK(vocab->size() == 8192);
    const std::string text = vocab->decode(result.tokens);
    std::fprintf(stderr, "decoded (%zu tokens) -> '%s'\n", result.tokens.size(), text.c_str());
    for (int32_t id : result.tokens) {
        LOOM_CHECK(id >= 0 && static_cast<size_t>(id) < vocab->size());
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
