// Nemotron-Labs-Unified-ASR (a `.nemo` FastConformer RNN-T with a language-ID prompt) decodes through its
// own embedded Lua driver, and agrees with NeMo's `EncDecRNNTBPEModelWithPrompt` (P5, rc20).
//
// The decode loop is `transducer_export.py`'s, shared with Parakeet, GigaAM and Nemotron 3.5. What this
// test is for is the LANGUAGE PROMPT, folded by the export from NeMo's scattered one-hot + concat into
// an embedding row added to every frame. On jfk.wav every prompt decodes to the same 38 tokens, so a
// prompt that never arrived would pass a token oracle. Hence the TENSOR oracle first: the encoder
// phase's output under each prompt against NeMo's `forward(..., prompt_indices=...)`, plus a check that
// two prompts' references differ by far more than the tolerance -- so the comparison can tell them apart.
//
// Fixture: `loom-exporter/fixture_gen/reference_forward_nemotron_unified.py` (NeMo 3.0, the nemo3 venv),
// prompts en / de / auto on jfk.wav. Set LOOM_NEMOTRON_UNIFIED_ASR_GGUF and
// LOOM_NEMOTRON_UNIFIED_ASR_REF_DIR; the test skips cleanly without them.

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

// The encoder phase alone: `(waveform, length, prompt) -> [d, T]`.
std::vector<float> run_encoder(loom::GgufModel& model, ggml_backend_t backend,
                               const std::vector<float>& waveform, int32_t prompt, size_t& frames) {
    loom::GraphTopology topo = loom::GraphTopology::parse(model.topology_json("encoder"));
    loom::GraphBuilder builder(topo, model, backend, /*kv_cache=*/nullptr);
    const loom::GraphBuilder::BuildResult& result =
        builder.build({{"n_samples", static_cast<uint32_t>(waveform.size())}, {"n_past", 0}});
    ggml_backend_tensor_set(result.input_tensors.at("waveform"), waveform.data(), 0,
                            waveform.size() * sizeof(float));
    // Both int inputs are written in the type the graph declared them as, which is the exporter's call.
    const auto set_scalar = [&](const char* name, double value) {
        ggml_tensor* t = result.input_tensors.at(name);
        if (t->type == GGML_TYPE_I32) {
            const auto v = static_cast<int32_t>(value);
            ggml_backend_tensor_set(t, &v, 0, sizeof(v));
        } else {
            const auto v = static_cast<float>(value);
            ggml_backend_tensor_set(t, &v, 0, sizeof(v));
        }
    };
    set_scalar("length", static_cast<double>(waveform.size()));
    set_scalar("prompt", prompt);
    ggml_backend_graph_compute(backend, result.graph);
    frames = static_cast<size_t>(result.output->ne[1]);
    std::vector<float> out(static_cast<size_t>(ggml_nelements(result.output)));
    ggml_backend_tensor_get(result.output, out.data(), 0, out.size() * sizeof(float));
    return out;
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
    const char* gguf = loom_test::fixture_env("LOOM_NEMOTRON_UNIFIED_ASR_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_NEMOTRON_UNIFIED_ASR_REF_DIR");
    const std::string ref = ref_env != nullptr ? ref_env : "";
    if (gguf == nullptr || !path_exists(gguf) || ref.empty() || !path_exists(ref + "/ref_prompts.npy")) {
        std::fprintf(stderr, "skipping: need LOOM_NEMOTRON_UNIFIED_ASR_GGUF and LOOM_NEMOTRON_UNIFIED_ASR_REF_DIR "
                             "(fixture_gen/reference_forward_nemotron_unified.py)\n");
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf, backend.get());
    LOOM_CHECK(model != nullptr);

    const std::vector<float> waveform = read_f32(ref + "/ref_waveform.npy");
    const std::vector<float> prompts = read_f32(ref + "/ref_prompts.npy");
    const std::vector<std::string> names = read_lines(ref + "/ref_prompt_names.txt");
    LOOM_CHECK(!waveform.empty() && prompts.size() >= 2 && names.size() == prompts.size());

    // ---- 1. the encoder phase under each prompt: a tensor oracle -----------------------------------
    std::vector<std::vector<float>> refs;
    for (size_t i = 0; i < prompts.size(); ++i) {
        refs.push_back(read_f32(ref + "/ref_encoder_" + std::to_string(i) + ".npy"));
        size_t frames = 0;
        const std::vector<float> got =
            run_encoder(*model, backend.get(), waveform, static_cast<int32_t>(prompts[i]), frames);
        LOOM_CHECK(got.size() == refs[i].size());
        const float d = max_abs_diff(got, refs[i]);
        std::fprintf(stderr, "encoder, %s (prompt %d): %zu frames, max |d| %.3e of absmax %.3f\n",
                     names[i].c_str(), static_cast<int>(prompts[i]), frames, d, abs_max(refs[i]));
        LOOM_CHECK(d < 1e-3f * abs_max(refs[i]));
    }
    // The comparison above can only catch a lost prompt if the prompts' own references are farther
    // apart than its tolerance.
    const float between = max_abs_diff(refs[0], refs[1]);
    std::fprintf(stderr, "encoder references, %s vs %s: max |d| %.3e\n", names[0].c_str(), names[1].c_str(),
                 between);
    LOOM_CHECK(between > 1e-2f * abs_max(refs[0]));

    // ---- 2. the driver, token for token, under each prompt -----------------------------------------
    loom::LoomLuaBridge bridge(backend.get());
    for (const std::string& name : model->topology_names()) {
        bridge.register_module(name, *model, loom::GraphTopology::parse(model->topology_json(name)));
    }
    bridge.load_script(model->kv_str("model.driver_script"));
    const std::vector<double> waveform_d(waveform.begin(), waveform.end());
    for (size_t i = 0; i < prompts.size(); ++i) {
        const std::vector<float> want = read_f32(ref + "/ref_ids_" + std::to_string(i) + ".npy");
        std::unordered_map<std::string, loom::LoomLuaBridge::Value> inputs{
            {"waveform", waveform_d},
            {"length", std::vector<double>{static_cast<double>(waveform.size())}},
            {"language", static_cast<double>(prompts[i])}};
        const auto got = std::get<std::vector<double>>(bridge.call("infer", inputs));
        std::fprintf(stderr, "driver, %s: %zu token(s), NeMo %zu\n", names[i].c_str(), got.size(), want.size());
        LOOM_CHECK(got.size() == want.size());
        for (size_t t = 0; t < std::min(got.size(), want.size()); ++t) {
            LOOM_CHECK(static_cast<int32_t>(got[t]) == static_cast<int32_t>(want[t]));
        }
    }

    // ---- 3. the text door: the name resolved through the contract's language table ----------------
    for (size_t i = 0; i < names.size(); ++i) {
        loom::audio::TranscribeOptions options;
        options.language = names[i];
        const loom::audio::Transcription text = loom::audio::transcribe(bridge, *model, waveform, options);
        const std::string want = read_text(ref + "/ref_text_" + std::to_string(i) + ".txt");
        std::fprintf(stderr, "text, language=%s: \"%s\"\n", names[i].c_str(), text.text.c_str());
        LOOM_CHECK(text.text == want);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
