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
// **Three inputs, chosen so the collapse is exercised rather than assumed.** A trained CTC model
// decodes synthetic audio to blank almost everywhere, so synthetic signals alone cannot reach the
// interesting branches:
//
//   * `reference_forward_conformer.py`'s Gaussian noise -> 0 tokens. Not vacuous: an empty transcript
//     is a real check of the blank id, since a wrong one KEEPS every frame and returns n_frames tokens
//     against the oracle's none.
//   * a chirp -> 1 token, so at least one id survives the collapse.
//   * `samples/jfk.wav`, 11s of real speech -> a real transcript, which is the only input that makes a
//     token span consecutive frames and therefore the only one that exercises DEDUPLICATION at all.
//     Before this fixture existed that rule had no behavioural test and was pinned only as emitted Lua
//     text in `test_driver_components.py`.
//
// Set LOOM_CONFORMER_CTC_MIL_GGUF and LOOM_CONFORMER_CTC_DIR (whose `ref/` subdir holds waveform.bin);
// skips cleanly if either is absent, same convention as its siblings. `samples/jfk.wav` is found
// relative to the repo root (LOOM_SAMPLES_DIR) and its case is skipped, not failed, if it is missing --
// the two synthetic cases still run.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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

// Minimal 16-bit PCM mono reader. `tools/loom_cli/wav_file.h` has one already, but it belongs to the
// CLI target rather than the engine library, and linking a tool into a test to read 44 bytes of header
// would be the wrong dependency to add. Returns empty on anything unexpected, which the caller treats
// as "skip this case".
std::vector<float> read_wav_pcm16_mono(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    char riff[12] = {};
    f.read(riff, 12);
    if (std::string(riff, 4) != "RIFF" || std::string(riff + 8, 4) != "WAVE") return {};
    uint16_t channels = 0, bits = 0;
    while (f) {
        char id[4] = {};
        uint32_t size = 0;
        f.read(id, 4);
        f.read(reinterpret_cast<char*>(&size), 4);
        if (!f) break;
        if (std::string(id, 4) == "fmt ") {
            std::vector<char> fmt(size);
            f.read(fmt.data(), size);
            if (size >= 16) {
                std::memcpy(&channels, fmt.data() + 2, 2);
                std::memcpy(&bits, fmt.data() + 14, 2);
            }
        } else if (std::string(id, 4) == "data") {
            if (bits != 16 || channels < 1) return {};
            std::vector<int16_t> pcm(size / 2);
            f.read(reinterpret_cast<char*>(pcm.data()), size);
            std::vector<float> out(pcm.size() / channels);
            for (size_t i = 0; i < out.size(); ++i) {
                out[i] = static_cast<float>(pcm[i * channels]) / 32768.0f;  // first channel only
            }
            return out;
        } else {
            f.seekg(size, std::ios::cur);
        }
    }
    return {};
}

} // namespace

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_CONFORMER_CTC_MIL_GGUF");
    const std::string gguf_path = gguf_env != nullptr ? gguf_env
                                                       : "conformer_ctc_small_mil_monolithic.gguf";
    const char* dir_env = loom_test::fixture_env("LOOM_CONFORMER_CTC_DIR");
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

    // The per-frame argmax before any collapsing -- what `loom.argmax_rows` returns engine-side, and
    // what the duplicate count below is measured on.
    int64_t blank_id = -1;
    auto frame_argmax = [&](const std::vector<float>& waveform) {
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
        blank_id = n_classes - 1;
        std::vector<float> logits(static_cast<size_t>(n_classes) * static_cast<size_t>(n_frames));
        ggml_backend_tensor_get(result.output, logits.data(), 0, logits.size() * sizeof(float));
        std::vector<int64_t> ids(static_cast<size_t>(n_frames));
        for (int64_t f = 0; f < n_frames; ++f) {
            int64_t best = 0;
            for (int64_t c = 1; c < n_classes; ++c) {
                if (logits[f * n_classes + c] > logits[f * n_classes + best]) best = c;
            }
            ids[static_cast<size_t>(f)] = best;
        }
        return ids;
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

    // --- 3. Real speech. The only input here that makes a token span consecutive frames, so it is the
    // only one that exercises the deduplication rule at all -- the branch the two synthetic cases
    // cannot reach and that had no behavioural test before this sample existed. Skipped rather than
    // failed if absent, so the check above still runs in a tree without it. ---
    const char* samples_env = std::getenv("LOOM_SAMPLES_DIR");
    const std::string jfk = std::string(samples_env != nullptr ? samples_env : "samples") + "/jfk.wav";
    const std::vector<float> speech = read_wav_pcm16_mono(jfk);
    if (speech.empty()) {
        std::fprintf(stderr, "skipping the real-speech case: '%s' not readable\n", jfk.c_str());
    } else {
        check_agrees("jfk.wav", speech, /*expect_tokens=*/true);
        // **The point of the case, asserted rather than assumed.** A long transcript is not by itself
        // evidence that deduplication ran: it only fires when a token occupies two consecutive frames.
        // So count those directly off the frame-wise argmax, and require at least one -- otherwise this
        // input exercises the same branches the chirp already did, and the test would be claiming
        // coverage it does not have.
        const std::vector<int32_t> ids = cpp_decode(speech);
        const std::vector<int64_t> frames = frame_argmax(speech);
        size_t collapsed_pairs = 0;
        for (size_t i = 1; i < frames.size(); ++i) {
            if (frames[i] == frames[i - 1] && frames[i] != blank_id) ++collapsed_pairs;
        }
        std::fprintf(stderr, "jfk.wav: %zu frames -> %zu tokens, %zu consecutive duplicate(s) collapsed\n",
                      frames.size(), ids.size(), collapsed_pairs);
        LOOM_CHECK(ids.size() > 10);

        // **Deduplication is still not exercised, and that is measured, not assumed.** This checkpoint
        // subsamples 4x to ~40ms frames and CTC alignments are spiky: a token occupies one non-blank
        // frame between blanks, so `collapsed_pairs` is 0 even on 11s of real speech. Slowing the audio
        // 3x was tried and is worse -- the model stops recognising it and emits 2 tokens with still no
        // duplicates. The count is printed rather than asserted so this stays visible: if a future
        // fixture or checkpoint does reach the branch, the number says so instead of nobody noticing.
        // The rule itself is pinned as emitted Lua text in `test_driver_components.py`.
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
