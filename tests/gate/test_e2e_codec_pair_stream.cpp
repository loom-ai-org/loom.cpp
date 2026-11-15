// A streamed codec pair (ADR-102) on real files: a text-to-codes model's `SpeechStream` hands back its frames
// as they are drawn, and a `CodecStream` decodes them as they arrive.
//
//   1. The LM's stream is its one-pass `infer`, bit for bit: the driver's `infer` is the stream run in one
//      call, and a seeded stream draws its seed's sequence.
//   2. The pair -- each frame pushed to the codec as it is drawn, in pushes that grow 1, 2, 4, 8 as the
//      hosts push them -- is the codec's one-pass decode of the same codes, to within f32 summation order.
//   3. Two LM streams pulled in alternation, with a one-pass `infer` of OTHER text between their pulls,
//      under both cache modes: each draws exactly what it drew alone. The one-pass call overwrites every
//      module's retained outputs, the prompt's text schedule included, which is why a stream keeps its own.
//
// LOOM_STREAM_LM_CODES_GGUF: a text-to-codes model exported with the stream functions (qwen3-tts-12hz-0.6b,
//                            audio8-tts-realistic-0.6b).
// LOOM_STREAM_CODEC_GGUF:    its codec, exported with them.
// LOOM_STREAM_REF_NPY:       a reference clip at the LM's rate, F32, for a model that clones and has no voice
//                            of its own (Qwen3-TTS Base); optional otherwise.

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;
constexpr double kOnePassTolerance = 1e-3;
constexpr double kFrames = 12;

using Inputs = loom::audio::SpeechStream::Inputs;

std::vector<int32_t> pull_codes(loom::audio::SpeechStream& s) {
    std::vector<int32_t> out;
    while (!s.finished()) {
        const std::vector<float> piece = s.next();
        out.insert(out.end(), piece.begin(), piece.end());
    }
    return out;
}

} // namespace

int main() {
    const char* lm_path = loom_test::fixture_env("LOOM_STREAM_LM_CODES_GGUF");
    const char* codec_path = loom_test::fixture_env("LOOM_STREAM_CODEC_GGUF");
    const char* ref_path = loom_test::fixture_env("LOOM_STREAM_REF_NPY");
    if (lm_path == nullptr || codec_path == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_STREAM_LM_CODES_GGUF and LOOM_STREAM_CODEC_GGUF "
                              "(and LOOM_STREAM_REF_NPY for a model that clones)\n");
        return kSkipReturnCode;
    }
    std::vector<int64_t> shape;
    const std::vector<float> ref = ref_path ? loom_test::read_npy_f32(ref_path, shape) : std::vector<float>{};

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto lm = loom::GgufModel::load(lm_path, backend.get());
    auto codec = loom::GgufModel::load(codec_path, backend.get());
    LOOM_CHECK(lm != nullptr && codec != nullptr);
    const uint32_t width = lm->hparam_u32("codec.n_codebooks");
    LOOM_CHECK(width == codec->hparam_u32("codec.n_codebooks"));
    auto vocab = loom::BpeVocab::load(*lm);
    LOOM_CHECK(vocab != nullptr);
    loom::Session lm_session(*lm, backend.get());
    loom::Session codec_session(*codec, backend.get());
    loom::LoomLuaBridge& bridge = lm_session.bridge();
    LOOM_CHECK(loom::audio::SpeechStream::supported(bridge) && loom::audio::CodecStream::supported(codec_session.bridge()));

    const auto inputs_for = [&](const std::string& text, double seed) {
        const std::vector<int32_t> ids = vocab->encode(text);
        Inputs in{{"tokens", std::vector<double>(ids.begin(), ids.end())}, {"max_new_tokens", kFrames},
                  {"seed", seed}};
        if (!ref.empty()) in["waveform"] = std::vector<double>(ref.begin(), ref.end());
        return in;
    };
    const Inputs a = inputs_for("The history of the city of Rome begins on the Palatine hill.", 5.0);
    const Inputs b = inputs_for("Bread needs flour, water and salt.", 6.0);
    const auto one_pass = [&](const Inputs& in) {
        const auto got = std::get<std::vector<double>>(bridge.call("infer", in));
        return std::vector<int32_t>(got.begin(), got.end());
    };
    const std::vector<int32_t> want_a = one_pass(a), want_b = one_pass(b);
    LOOM_CHECK(!want_a.empty() && want_a.size() % width == 0);

    // 1. The LM's stream.
    {
        loom::audio::SpeechStream s(bridge, a);
        const std::vector<int32_t> got = pull_codes(s);
        std::fprintf(stderr, "LM stream: %zu frames, one pass %zu: %s\n", got.size() / width, want_a.size() / width,
                     got == want_a ? "identical" : "DIFFERS");
        LOOM_CHECK(got == want_a);
    }

    // 2. The pair.
    {
        const auto decoded = std::get<std::vector<double>>(
            codec_session.bridge().call("infer", {{"codes", std::vector<double>(want_a.begin(), want_a.end())}}));
        loom::audio::SpeechStream codes(bridge, a);
        loom::audio::CodecStream audio(codec_session.bridge());
        std::vector<float> got;
        std::vector<int32_t> pending;
        size_t pushes = 0;
        while (!codes.finished()) {
            const std::vector<float> piece = codes.next();
            pending.insert(pending.end(), piece.begin(), piece.end());
            if (!pending.empty() && pending.size() / width >= std::min<size_t>(8, size_t{1} << std::min<size_t>(pushes, 3))) {
                const std::vector<float> w = audio.push(pending);
                got.insert(got.end(), w.begin(), w.end());
                pending.clear();
                ++pushes;
            }
        }
        if (!pending.empty()) {
            const std::vector<float> w = audio.push(pending);
            got.insert(got.end(), w.begin(), w.end());
        }
        const std::vector<float> tail = audio.finish();
        got.insert(got.end(), tail.begin(), tail.end());
        double d = 0.0;
        for (size_t i = 0; i < std::min(got.size(), decoded.size()); ++i) d = std::max(d, std::abs(got[i] - decoded[i]));
        std::fprintf(stderr, "pair: %zu pushes, %zu samples, one pass %zu, max|d| %.3g\n", pushes + !pending.empty(),
                     got.size(), decoded.size(), d);
        LOOM_CHECK(got.size() == decoded.size());
        LOOM_CHECK(d <= kOnePassTolerance);
    }

    // 3. Two LM streams in alternation, a one-pass call of other text between their pulls.
    const Inputs c = inputs_for("Three colours are primary.", 7.0);
    const std::vector<int32_t> want_c = one_pass(c);
    for (loom::StreamCaches mode : {loom::StreamCaches::Park, loom::StreamCaches::PerStream}) {
        bridge.set_stream_caches(mode);
        loom::audio::SpeechStream sa(bridge, a), sb(bridge, b);
        std::vector<int32_t> got_a, got_b;
        bool one_pass_ok = true;
        size_t turn = 0;
        while (!sa.finished() || !sb.finished()) {
            const std::vector<float> pa = sa.next();
            got_a.insert(got_a.end(), pa.begin(), pa.end());
            if (turn++ == 2) one_pass_ok = one_pass(c) == want_c;
            const std::vector<float> pb = sb.next();
            got_b.insert(got_b.end(), pb.begin(), pb.end());
        }
        std::fprintf(stderr, "%s: a %s, b %s, one pass between %s\n",
                     mode == loom::StreamCaches::Park ? "park" : "per-stream", got_a == want_a ? "ok" : "DIFFERS",
                     got_b == want_b ? "ok" : "DIFFERS", one_pass_ok ? "ok" : "DIFFERS");
        LOOM_CHECK(got_a == want_a);
        LOOM_CHECK(got_b == want_b);
        LOOM_CHECK(one_pass_ok);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
