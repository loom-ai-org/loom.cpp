// Zipformer streaming (`pkufool/zipformer-*-streaming`) decodes chunk by chunk through its own embedded Lua
// driver, carrying the encoder's 74 state tensors from call to call as retained outputs, and agrees with
// upstream's released jit file run as `zipformer inference --streaming 1` runs it (P5).
//
// The risk is the carry: a state fed back wrong still decodes the first chunk correctly and drifts after.
// Hence the TENSOR oracle first -- every chunk's encoder rows, through the driver's own loop
// (`return_encoder`) -- then tokens, then text.
//
// Fixture: `loom-exporter/fixture_gen/reference_forward_zipformer_streaming.py`. Set
// LOOM_ZIPFORMER_STREAMING_GGUF and LOOM_ZIPFORMER_STREAMING_REF_DIR; the test skips cleanly without them.

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;

bool path_exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

std::vector<float> read_f32(const std::string& path) {
    std::vector<int64_t> shape;
    return loom_test::read_npy_f32(path, shape);
}

std::string read_text(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<std::string> read_lines(const std::string& path) {
    std::ifstream f(path);
    std::vector<std::string> lines;
    for (std::string line; std::getline(f, line);) {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0.0f;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

float abs_max(const std::vector<float>& a) {
    float m = 0.0f;
    for (float v : a) m = std::max(m, std::fabs(v));
    return m;
}

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_ZIPFORMER_STREAMING_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_ZIPFORMER_STREAMING_REF_DIR");
    const std::string ref = ref_env != nullptr ? ref_env : "";
    if (gguf == nullptr || !path_exists(gguf) || ref.empty() || !path_exists(ref + "/ref_names.txt")) {
        std::fprintf(stderr, "skipping: need LOOM_ZIPFORMER_STREAMING_GGUF and LOOM_ZIPFORMER_STREAMING_REF_DIR "
                             "(fixture_gen/reference_forward_zipformer_streaming.py)\n");
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf, backend.get());
    LOOM_CHECK(model != nullptr);

    const std::vector<std::string> names = read_lines(ref + "/ref_names.txt");
    LOOM_CHECK(!names.empty());

    loom::LoomLuaBridge bridge(backend.get());
    for (const std::string& name : model->topology_names()) {
        bridge.register_module(name, *model, loom::GraphTopology::parse(model->topology_json(name)));
    }
    bridge.load_script(model->kv_str("model.driver_script"));

    for (size_t i = 0; i < names.size(); ++i) {
        const std::string index = std::to_string(i);
        const std::vector<float> waveform = read_f32(ref + "/ref_waveform_" + index + ".npy");
        LOOM_CHECK(!waveform.empty());

        // ---- 1. every chunk's encoder rows, through the driver's own loop: a tensor oracle ----------
        const std::vector<double> waveform_d(waveform.begin(), waveform.end());
        const std::vector<float> want_encoder = read_f32(ref + "/ref_encoder_" + index + ".npy");
        std::unordered_map<std::string, loom::LoomLuaBridge::Value> encode{{"waveform", waveform_d},
                                                                           {"return_encoder", 1.0}};
        const auto got_d = std::get<std::vector<double>>(bridge.call("infer", encode));
        const std::vector<float> got_encoder(got_d.begin(), got_d.end());
        LOOM_CHECK(got_encoder.size() == want_encoder.size());
        const float d = max_abs_diff(got_encoder, want_encoder);
        std::fprintf(stderr, "%s: encoder, %zu values, max |d| %.3e of absmax %.3f\n", names[i].c_str(),
                     got_encoder.size(), d, abs_max(want_encoder));
        LOOM_CHECK(d < 1e-4f * abs_max(want_encoder));

        // ---- 2. the driver, token for token ---------------------------------------------------------
        std::unordered_map<std::string, loom::LoomLuaBridge::Value> inputs{{"waveform", waveform_d}};
        const auto got = std::get<std::vector<double>>(bridge.call("infer", inputs));
        const std::vector<float> want = read_f32(ref + "/ref_ids_" + index + ".npy");
        std::fprintf(stderr, "%s: driver, %zu token(s), reference %zu\n", names[i].c_str(), got.size(), want.size());
        LOOM_CHECK(got.size() == want.size());
        for (size_t t = 0; t < std::min(got.size(), want.size()); ++t) {
            LOOM_CHECK(static_cast<int32_t>(got[t]) == static_cast<int32_t>(want[t]));
        }

        // ---- 3. the text door -----------------------------------------------------------------------
        const loom::audio::Transcription text = loom::audio::transcribe(bridge, *model, waveform, {});
        std::fprintf(stderr, "%s: text \"%s\"\n", names[i].c_str(), text.text.c_str());
        LOOM_CHECK(text.text == read_text(ref + "/ref_text_" + index + ".txt"));
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
