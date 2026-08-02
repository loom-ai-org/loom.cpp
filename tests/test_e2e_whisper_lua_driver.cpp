// Validates the procedural-generalization architecture (LOOM_PROCEDURAL_GENERALIZATION.md /
// LOOM_MIL_CONVERSION.md): runs the SAME real Whisper checkpoint's transcribe() through TWO independent
// paths -- the existing hand-written loom::WhisperDriver (C++ control flow) and a LoomLuaBridge running
// the hand-ported tools/convert_whisper/whisper_driver.lua (embedded in whisper_decoder.gguf's own
// "model.driver_script" KV, read generically via GgufModel::kv_str) -- and asserts they produce the
// EXACT SAME generated-token sequence. Both are deterministic greedy argmax decoding, so this is a
// stronger, simpler check than a floating-point tolerance comparison: no tolerance question for an
// integer token sequence. Reuses the same real-checkpoint env vars and reference waveform/prompt as
// test_e2e_whisper_driver.cpp (does NOT need the reference's own `ref_driver_generated.npy` -- the C++
// WhisperDriver run in this SAME binary is the oracle). Skips cleanly if the GGUF/reference files aren't
// present.

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
        std::fprintf(stderr, "skipping: set LOOM_WHISPER_DIR (whisper.gguf, produced by "
                              "convert_whisper_all.py -- embeds encoder+decoder topologies AND "
                              "model.driver_script in one file) and LOOM_WHISPER_DRIVER_REF_DIR "
                              "(ref_driver_waveform_padded.npy/ref_driver_prompt.npy) to run this check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    std::vector<int64_t> wav_shape, prompt_shape;
    std::vector<float> waveform = read_npy_f32(ref_dir + "/ref_driver_waveform_padded.npy", wav_shape);
    std::vector<int32_t> prompt = read_npy_i32(ref_dir + "/ref_driver_prompt.npy", prompt_shape);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    // Encoder + decoder topologies AND the driver script all live in ONE GGUF file now (see
    // LOOM_PROCEDURAL_GENERALIZATION.md / BACKLOG.md's dated entry).
    auto model = loom::GgufModel::load(dir + "/whisper.gguf", backend.get());
    LOOM_CHECK(model != nullptr);

    const std::string driver_script = model->kv_str("model.driver_script");
    LOOM_CHECK(!driver_script.empty());

    loom::WhisperConfig cfg;
    cfg.n_audio_state = 384;
    cfg.n_audio_ctx = 1500;
    cfg.n_text_state = 384;
    cfg.n_text_head = 6;
    cfg.n_text_layer = 4;
    cfg.n_text_ctx = 448;
    cfg.eot_token = 50256;
    constexpr uint32_t kMaxNewTokens = 16;

    // --- Oracle: the existing hand-written C++ driver ---
    std::vector<int32_t> ref_generated;
    {
        loom::GraphTopology encoder_topo = loom::GraphTopology::parse(model->topology_json("encoder"));
        loom::GraphTopology decoder_topo = loom::GraphTopology::parse(model->topology_json("decoder"));
        loom::WhisperDriver driver(*model, std::move(encoder_topo), *model, std::move(decoder_topo), cfg,
                                    backend.get());
        ref_generated = driver.transcribe(waveform, prompt, kMaxNewTokens);
    }

    // --- New path: LoomLuaBridge running the hand-ported whisper_driver.lua ---
    std::vector<int32_t> lua_generated;
    {
        loom::GraphTopology encoder_topo = loom::GraphTopology::parse(model->topology_json("encoder"));
        loom::GraphTopology decoder_topo = loom::GraphTopology::parse(model->topology_json("decoder"));
        // Same construction as WhisperDriver's own internal KvCache (src/core/whisper_driver.cpp).
        loom::KvCache kv_cache(cfg.n_text_layer, cfg.n_text_state, cfg.n_text_state, cfg.n_text_ctx, backend.get());

        loom::LoomLuaBridge bridge(backend.get());
        bridge.register_module("encoder", *model, std::move(encoder_topo), /*kv_cache=*/nullptr);
        bridge.register_module("decoder", *model, std::move(decoder_topo), &kv_cache);
        bridge.load_script(driver_script);

        const std::vector<double> waveform_d(waveform.begin(), waveform.end());
        const std::vector<double> prompt_d(prompt.begin(), prompt.end());
        loom::LoomLuaBridge::Value result = bridge.call("infer", {
            {"waveform", waveform_d},
            {"prompt_tokens", prompt_d},
            {"n_audio_ctx", static_cast<double>(cfg.n_audio_ctx)},
            {"max_new_tokens", static_cast<double>(kMaxNewTokens)},
            {"eot_token", static_cast<double>(cfg.eot_token)},
        });
        const auto& generated_d = std::get<std::vector<double>>(result);
        lua_generated.reserve(generated_d.size());
        for (double v : generated_d) lua_generated.push_back(static_cast<int32_t>(v));
    }

    std::fprintf(stderr, "C++ driver generated %zu tokens: ", ref_generated.size());
    for (int32_t t : ref_generated) std::fprintf(stderr, "%d ", t);
    std::fprintf(stderr, "\nLua driver generated  %zu tokens: ", lua_generated.size());
    for (int32_t t : lua_generated) std::fprintf(stderr, "%d ", t);
    std::fprintf(stderr, "\n");

    LOOM_CHECK(lua_generated.size() == ref_generated.size());
    for (size_t i = 0; i < lua_generated.size(); ++i) {
        LOOM_CHECK(lua_generated[i] == ref_generated[i]);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
