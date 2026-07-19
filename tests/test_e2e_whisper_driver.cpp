// End-to-end check for loom::WhisperDriver::transcribe against a real greedy-decoded reference sequence
// (tools/convert_whisper/reference_forward_whisper_driver.py, a plain argmax loop built directly from
// Whisper's own encoder/decoder forward passes -- NOT model.decode(), which applies extra logic this
// driver doesn't implement). Both are deterministic (argmax, no sampling), so this is a plain exact
// generated-token-sequence comparison, one level up from test_e2e_whisper_{encoder,decoder}_reference.cpp
// (which check each topology in isolation) -- exercises the real two-phase driver loop itself: one
// encoder pass, then prefill + incremental decode with a persistent KvCache and a fixed cross-attention
// `xa` fed unchanged every step. Skips cleanly if the GGUF/reference files aren't present.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<float> read_npy_f32(const std::string& path, std::vector<int64_t>& shape_out) {
    std::ifstream f(path, std::ios::binary);
    LOOM_CHECK(static_cast<bool>(f));
    char magic[6];
    f.read(magic, 6);
    f.ignore(2);
    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);
    const size_t shape_pos = header.find("'shape':");
    const size_t paren_open = header.find('(', shape_pos);
    const size_t paren_close = header.find(')', paren_open);
    std::string shape_str = header.substr(paren_open + 1, paren_close - paren_open - 1);
    shape_out.clear();
    std::stringstream ss(shape_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        std::string trimmed;
        for (char c : tok) if (c != ' ') trimmed += c;
        if (!trimmed.empty()) shape_out.push_back(std::stoll(trimmed));
    }
    int64_t total = 1;
    for (int64_t d : shape_out) total *= d;
    std::vector<float> data(static_cast<size_t>(total));
    f.read(reinterpret_cast<char*>(data.data()), total * static_cast<int64_t>(sizeof(float)));
    return data;
}

std::vector<int32_t> read_npy_i32(const std::string& path, std::vector<int64_t>& shape_out) {
    std::ifstream f(path, std::ios::binary);
    LOOM_CHECK(static_cast<bool>(f));
    char magic[6];
    f.read(magic, 6);
    f.ignore(2);
    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);
    const size_t shape_pos = header.find("'shape':");
    const size_t paren_open = header.find('(', shape_pos);
    const size_t paren_close = header.find(')', paren_open);
    std::string shape_str = header.substr(paren_open + 1, paren_close - paren_open - 1);
    shape_out.clear();
    std::stringstream ss(shape_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        std::string trimmed;
        for (char c : tok) if (c != ' ') trimmed += c;
        if (!trimmed.empty()) shape_out.push_back(std::stoll(trimmed));
    }
    int64_t total = 1;
    for (int64_t d : shape_out) total *= d;
    std::vector<int32_t> data(static_cast<size_t>(total));
    f.read(reinterpret_cast<char*>(data.data()), total * static_cast<int64_t>(sizeof(int32_t)));
    return data;
}

} // namespace

int main() {
    const char* dir_env = std::getenv("LOOM_WHISPER_DIR");
    const char* ref_dir_env = std::getenv("LOOM_WHISPER_DRIVER_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_WHISPER_DIR (whisper_encoder.gguf/whisper_decoder.gguf) "
                              "and LOOM_WHISPER_DRIVER_REF_DIR (ref_driver_*.npy, produced by "
                              "reference_forward_whisper_driver.py) to run this check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    std::vector<int64_t> wav_shape, prompt_shape, gen_shape;
    std::vector<float> waveform = read_npy_f32(ref_dir + "/ref_driver_waveform_padded.npy", wav_shape);
    std::vector<int32_t> prompt = read_npy_i32(ref_dir + "/ref_driver_prompt.npy", prompt_shape);
    std::vector<int32_t> ref_generated = read_npy_i32(ref_dir + "/ref_driver_generated.npy", gen_shape);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto encoder_model = loom::GgufModel::load(dir + "/whisper_encoder.gguf", backend.get());
    auto decoder_model = loom::GgufModel::load(dir + "/whisper_decoder.gguf", backend.get());
    LOOM_CHECK(encoder_model != nullptr);
    LOOM_CHECK(decoder_model != nullptr);
    loom::GraphTopology encoder_topo = loom::GraphTopology::parse(encoder_model->topology_json());
    loom::GraphTopology decoder_topo = loom::GraphTopology::parse(decoder_model->topology_json());

    loom::WhisperConfig cfg;
    cfg.n_audio_state = 384;
    cfg.n_audio_ctx = 1500;
    cfg.n_text_state = 384;
    cfg.n_text_head = 6;
    cfg.n_text_layer = 4;
    cfg.n_text_ctx = 448;
    cfg.eot_token = 50256;

    loom::WhisperDriver driver(*encoder_model, std::move(encoder_topo), *decoder_model, std::move(decoder_topo),
                                cfg, backend.get());
    std::vector<int32_t> generated = driver.transcribe(waveform, prompt, /*max_new_tokens=*/16);

    std::fprintf(stderr, "generated %zu tokens: ", generated.size());
    for (int32_t t : generated) std::fprintf(stderr, "%d ", t);
    std::fprintf(stderr, "\n");

    LOOM_CHECK(generated.size() == ref_generated.size());
    for (size_t i = 0; i < generated.size(); ++i) {
        LOOM_CHECK(generated[i] == ref_generated[i]);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
