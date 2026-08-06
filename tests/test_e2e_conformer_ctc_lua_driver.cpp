// Conformer-CTC's embedded Lua driver decodes the same transcript the C++ path does (BACKLOG.md
// P4.0.17).
//
// Until this, the NeMo ASR encoders were the one family with no working `infer`: the MIL exporter gave
// them the causal-LM epilogue, which argmaxes row `n_tokens - 1` -- and for these topologies `n_tokens`
// is the *sample* count while the output has one row per subsampled frame, so the call raised rather
// than returning anything. `test_e2e_conformer_ctc_mil_export.cpp` says as much in its own header, and
// works around it by driving `GraphBuilder` directly.
//
// **The claim is an equivalence, so it is written against the oracle it replaces.** `loom::
// ctc_greedy_decode` is the C++ implementation the driver's Lua now does instead: per-frame argmax,
// collapse consecutive duplicates, drop the blank. Both run here over the SAME model and the SAME real
// reference waveform, and must produce the same token ids. If the Lua were merely plausible -- an
// off-by-one in the blank id, a collapse that drops a legitimate repeat, a frame loop that misses the
// last row -- it would still return a sequence, and only a comparison catches that.
//
// **What the fixtures can and cannot show, stated up front.** There is no speech recording in this
// tree, and a trained CTC model decodes synthetic audio to blank almost everywhere: the reference
// waveform (`reference_forward_conformer.py`'s Gaussian noise) yields 0 tokens, and the best synthetic
// signal found -- a chirp -- yields 1. Both are still worth running and neither is trivial. An empty
// transcript is a real check of the blank id: get it wrong and every frame is KEPT, so the driver
// returns n_frames tokens where the oracle returns none. The chirp then shows a real token surviving
// the collapse. What no fixture here can exercise is the DEDUPLICATION rule, which needs a token
// spanning consecutive frames; that is pinned instead by asserting the emitted Lua verbatim in
// `test_driver_components.py`, the way every other component is, and the gap is recorded in
// BACKLOG.md P4.0.17.
//
// Set LOOM_CONFORMER_CTC_MIL_GGUF and LOOM_CONFORMER_CTC_DIR (whose `ref/` subdir holds waveform.bin);
// skips cleanly if either is absent, same convention as its siblings.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;

bool path_exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
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
    const char* gguf_env = std::getenv("LOOM_CONFORMER_CTC_MIL_GGUF");
    const std::string gguf_path = gguf_env != nullptr ? gguf_env
                                                       : "conformer_ctc_small_mil_monolithic.gguf";
    const char* dir_env = std::getenv("LOOM_CONFORMER_CTC_DIR");
    const std::string ref_dir = std::string(dir_env != nullptr ? dir_env : "/tmp/nemo_model") + "/ref";

    if (!path_exists(gguf_path) || !path_exists(ref_dir + "/waveform.bin")) {
        std::fprintf(stderr, "skipping: MIL-exported Conformer-CTC GGUF ('%s') or ref waveform ('%s') "
                              "not found\n", gguf_path.c_str(), (ref_dir + "/waveform.bin").c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_path, backend.get());

    // One bridge for every case: the driver is the model's own, read out of the GGUF, and reaching it
    // is the thing that did not work before (BACKLOG.md P4.0.17).
    loom::LoomLuaBridge bridge(backend.get());
    for (const std::string& mod_name : model->topology_names()) {
        bridge.register_module(mod_name, *model,
                                loom::GraphTopology::parse(model->topology_json(mod_name)));
    }
    bridge.load_script(model->kv_str("model.driver_script"));

    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("main_topology"));
    loom::GraphBuilder builder(topo, *model, backend.get(), /*kv_cache=*/nullptr);

    // The oracle: the same graph, decoded by the C++ implementation the driver's Lua replaces --
    // exactly what `loom_cli --wav` does today.
    auto cpp_decode = [&](const std::vector<float>& waveform) {
        const auto n_samples = static_cast<uint32_t>(waveform.size());
        const loom::GraphBuilder::BuildResult& result =
            builder.build({{"n_samples", n_samples}, {"n_past", 0}});
        ggml_backend_tensor_set(result.input_tensors.at("waveform"), waveform.data(), 0,
                                 waveform.size() * sizeof(float));
        const auto length_val = static_cast<int32_t>(n_samples);
        ggml_backend_tensor_set(result.input_tensors.at("length"), &length_val, 0, sizeof(int32_t));
        ggml_backend_graph_compute(backend.get(), result.graph);

        const int64_t n_classes = result.output->ne[0];
        const int64_t n_frames = result.output->ne[1];
        std::vector<float> logits(static_cast<size_t>(n_classes) * static_cast<size_t>(n_frames));
        ggml_backend_tensor_get(result.output, logits.data(), 0, logits.size() * sizeof(float));
        return loom::ctc_greedy_decode(logits.data(), n_frames, n_classes,
                                        /*blank_id=*/static_cast<int32_t>(n_classes) - 1);
    };

    // The driver: `length` is a declared topology input of shape [1], not a scalar knob -- the driver
    // forwards it to `run_subgraph` verbatim, so the host passes the same one-element tensor the C++
    // path writes into `input_tensors.at("length")`.
    auto lua_decode = [&](const std::vector<float>& waveform) {
        const std::vector<double> waveform_d(waveform.begin(), waveform.end());
        const std::vector<double> length_d{static_cast<double>(waveform.size())};
        return std::get<std::vector<double>>(bridge.call(
            "infer", {{"waveform", waveform_d}, {"length", length_d}}));
    };

    auto check_agrees = [&](const char* what, const std::vector<float>& waveform, bool expect_tokens) {
        const std::vector<int32_t> expected = cpp_decode(waveform);
        const std::vector<double> got = lua_decode(waveform);
        std::fprintf(stderr, "%s (%zu samples): C++ -> %zu token(s), Lua -> %zu token(s)\n",
                      what, waveform.size(), expected.size(), got.size());
        LOOM_CHECK(got.size() == expected.size());
        for (size_t i = 0; i < got.size() && i < expected.size(); ++i) {
            if (static_cast<int32_t>(got[i]) != expected[i]) {
                std::fprintf(stderr, "  token %zu: Lua %d, C++ %d\n", i, static_cast<int32_t>(got[i]),
                              expected[i]);
            }
            LOOM_CHECK(static_cast<int32_t>(got[i]) == expected[i]);
        }
        if (expect_tokens) LOOM_CHECK(!expected.empty());
    };

    // --- 1. The reference waveform. Decodes to nothing, which is the blank id being right: a wrong
    // blank would keep every frame and return one token per frame instead. ---
    const std::vector<float> reference = read_f32_binary(ref_dir + "/waveform.bin");
    LOOM_CHECK(!reference.empty());
    check_agrees("reference waveform", reference, /*expect_tokens=*/false);

    // --- 2. A chirp, the one synthetic signal found that this checkpoint emits a real token for, so
    // that at least one id travels the whole path rather than every frame being dropped. ---
    std::vector<float> chirp(32000); // 2 s at the checkpoint's own 16 kHz
    for (size_t i = 0; i < chirp.size(); ++i) {
        const double t = static_cast<double>(i) / 16000.0;
        chirp[i] = static_cast<float>(0.4 * std::sin(2.0 * M_PI * (200.0 + 800.0 * t) * t));
    }
    check_agrees("chirp", chirp, /*expect_tokens=*/true);

    LOOM_TEST_REPORT_AND_RETURN();
}
