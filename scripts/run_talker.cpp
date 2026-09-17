// run_talker: drive Qwen3-TTS's talker -- text and a reference voice in, codec frames out.
//
//   ./run_talker <model.gguf> <device> <tokens.txt> <ref24k.f32> <out.txt>
//
// Env: GREEDY=1 sets temperature 0 on BOTH draws (the talker's and the code predictor's), which is
// what reproduces `transformers` exactly; XVEC=<file> supplies a 1024-float x-vector and skips the
// speaker encoder; MAXNEW=<n> caps the frame count -- use it, a bounded run answers "do the codes
// match" in a minute where a full utterance takes ~15.
//
// Fixtures for both live at /home/flavio/.claude/tmp/qwen3tts/ -- see [[loom-qwen3-tts-shipped]].
//
// Build (from the repo root):
//   g++ -O2 -std=c++17 -I include -I build/_deps/ggml-src/include \
//       -I build/_deps/nlohmann_json-src/include scripts/run_talker.cpp \
//       -L build -lloom_engine -L build/_deps/ggml-build/src -lggml -lggml-base -lpthread \
//       -o run_talker
//   LD_LIBRARY_PATH=$PWD/build:$PWD/build/_deps/ggml-build/src ./run_talker ...
//
// **Why these exist rather than `loom_cli`.** The CLI has a transcription mode and a
// token-generation mode, so it cannot run a model whose answer is CODES at all, and when it does
// write audio it writes PCM16 -- whose 1.5e-05 quantisation floor hid this codec's real accuracy
// (5.0e-05 through the wav, 4.2e-06 on the engine's own floats). `scripts/bench_driver.cpp` calls
// any driver but prints only a checksum, because it exists to time one rather than to read it.
//
#include "loom/loom.h"
#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

static std::vector<double> read_numbers(const char* path) {
    std::vector<double> v; double x;
    std::ifstream f(path);
    while (f >> x) v.push_back(x);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: %s <model.gguf> <device> <tokens.txt> <wav.f32> <out.txt>\n", argv[0]); return 2; }
    auto tokens = read_numbers(argv[3]);
    std::vector<double> wav;
    { std::ifstream f(argv[4], std::ios::binary); float s;
      while (f.read(reinterpret_cast<char*>(&s), sizeof(float))) wav.push_back(s); }
    std::fprintf(stderr, "%zu tokens, %zu samples\n", tokens.size(), wav.size());

    loom::Device device = loom::Device::open(argv[2]);
    auto model = loom::GgufModel::load(argv[1], device.backends().primary);
    loom::Session session(*model, device.backends());
    std::unordered_map<std::string, loom::LoomLuaBridge::Value> inputs;
    inputs["tokens"] = tokens;
    if (getenv("XVEC") != nullptr) { inputs["x_vector"] = read_numbers(getenv("XVEC")); }
    else { inputs["waveform"] = wav; }
    // ICL: the reference clip's transcript, and either its codes or the clip itself. With REFAUDIO
    // the driver runs the codec's encoder (four phases) to draw the codes here; with REFCODES it is
    // handed them, which is how the encode half and the prompt half are graded separately.
    if (getenv("REFTOKENS") != nullptr) inputs["ref_tokens"] = read_numbers(getenv("REFTOKENS"));
    if (getenv("REFCODES") != nullptr) inputs["ref_code"] = read_numbers(getenv("REFCODES"));
    else if (getenv("REFAUDIO") != nullptr) {
        std::vector<double> ref; std::ifstream f(getenv("REFAUDIO"), std::ios::binary); float s;
        while (f.read(reinterpret_cast<char*>(&s), sizeof(float))) ref.push_back(s);
        std::fprintf(stderr, "ref_audio: %zu samples\n", ref.size());
        inputs["ref_audio"] = ref;
    }
    if (getenv("GREEDY") != nullptr) {
        inputs["temperature"] = 0.0;            // the engine's spelling of greedy
        inputs["subtalker_temperature"] = 0.0;
    }
    if (getenv("MAXNEW") != nullptr) inputs["max_new_tokens"] = atof(getenv("MAXNEW"));
    // PENALTY=1 turns the repetition penalty off, which is how "does the reference penalise
    // the same history we do" gets asked rather than argued.
    if (getenv("PENALTY") != nullptr) inputs["repetition_penalty"] = atof(getenv("PENALTY"));
    const auto out = session.bridge().call("infer", inputs);
    const auto* arr = std::get_if<std::vector<double>>(&out);
    if (arr == nullptr) { std::fprintf(stderr, "driver did not return an array\n"); return 1; }
    std::fprintf(stderr, "%zu codes = %zu frames\n", arr->size(), arr->size() / 16);
    FILE* g = std::fopen(argv[5], "w");
    for (double d : *arr) std::fprintf(g, "%lld\n", static_cast<long long>(d));
    std::fclose(g);
    return 0;
}
