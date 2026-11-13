// The audio-out live door (ADR-100, ADR-101) on a real TTS: `SpeechStream` must hand back, piece by piece,
// the waveform `infer` returns, and streams on one model must not see each other.
//
//   1. One stream, seeded: its pieces joined are `infer`'s samples -- the same count, and the same values to
//      within f32 summation order. A driver that cuts where its one pass already cuts (a sentence), or
//      recomputes a window holding the decoder's whole receptive field (Soprano's rows, ADR-114), gives them
//      bit for bit; one that decodes a frame at a time through a state-carrying step (Pocket-TTS) sums the
//      same terms in another order, ~1e-6 apart. There are at least two pieces, and the first is
//      ready before the whole is.
//   2. `reset` gives the same pieces again, bit for bit.
//   3. Two streams pulled in alternation, with a one-pass `infer` between their pulls, under both cache
//      modes: each gives exactly what it gave alone. Their draws come from generators of their own and
//      their decoder state is their own, so a piece of one stream between two of the other's moves
//      neither.
//
// LOOM_STREAM_TTS_GGUF: a TTS exported with the stream functions (pocket-tts, soprano, voxcpm2,
// lfm2.5-audio's tts, voxtral-4b-tts). A file
// exported before them is refused by the door, which is checked too when LOOM_STREAM_TTS_OLD_GGUF names one.
// LOOM_STREAM_TTS_TEXT_A / _B replace the two texts: VoxCPM2 takes ~20 s per second of audio on the 2-core
// dev box, and the gate synthesises each text about ten times.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;
// A stream against one pass. Pocket-TTS's frame step measured 1.0e-6 against its one-call decoder on a
// waveform peaking near 1; a frame decoded from the wrong state is off by the signal itself.
constexpr double kOnePassTolerance = 1e-4;

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) d = std::max(d, std::abs(double(a[i]) - b[i]));
    return d;
}

// Three sentences long enough that Pocket-TTS's vocabulary cannot regroup them into one chunk.
const char* const kTextA =
    "The history of the city of Rome begins on the Palatine hill, where shepherds built their first huts "
    "beside the river. A good recipe for country bread needs only flour, water, salt and a great deal of "
    "patience in the morning. The three primary colours are red, yellow and blue, and every other colour "
    "can be mixed from them.";
const char* const kTextB =
    "Trains leave the northern station every twenty minutes during the week, and every hour on Sundays. "
    "Bring a warm coat, because the platform is open to the wind from the sea.";

std::vector<double> encode(const loom::GgufModel& model, const std::string& text) {
    const std::string tag = model.has_kv("tokenizer.ggml.model") ? model.kv_str("tokenizer.ggml.model") : "";
    std::vector<int32_t> ids;
    if (tag == "pocket_tts") {
        ids = loom::PocketTtsVocab::load(model)->encode(text);
    } else if (tag == "soprano") {
        ids = loom::SopranoVocab::load(model)->encode(text);
    } else if (tag == "voxcpm2") {
        ids = loom::VoxCpmVocab::load(model)->encode(text);
    } else if (tag == "gpt2") {
        // A plain byte-level BPE whose driver builds the rest of the prompt (LFM2.5-Audio's TTS).
        ids = loom::BpeVocab::load(model)->encode(text);
    } else {
        throw std::runtime_error("test_e2e_speech_stream: no text front end here for tokenizer '" + tag + "'");
    }
    return std::vector<double>(ids.begin(), ids.end());
}

std::vector<float> one_pass(loom::LoomLuaBridge& bridge, const loom::audio::SpeechStream::Inputs& inputs) {
    const auto got = std::get<std::vector<double>>(bridge.call("infer", inputs));
    return std::vector<float>(got.begin(), got.end());
}

struct Pulled {
    std::vector<float> samples;
    size_t pieces = 0;
    size_t first_piece = 0;
};

Pulled pull_all(loom::audio::SpeechStream& stream) {
    Pulled out;
    while (!stream.finished()) {
        const std::vector<float> piece = stream.next();
        if (piece.empty()) continue;
        if (out.pieces++ == 0) out.first_piece = piece.size();
        out.samples.insert(out.samples.end(), piece.begin(), piece.end());
    }
    return out;
}

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_STREAM_TTS_GGUF");
    if (gguf == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_STREAM_TTS_GGUF to a TTS exported with the stream functions\n");
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, backend.get());
    loom::LoomLuaBridge& bridge = session.bridge();
    LOOM_CHECK(loom::audio::SpeechStream::supported(bridge));

    const char* text_a = std::getenv("LOOM_STREAM_TTS_TEXT_A");
    const char* text_b = std::getenv("LOOM_STREAM_TTS_TEXT_B");
    const loom::audio::SpeechStream::Inputs a{{"tokens", encode(*model, text_a ? text_a : kTextA)},
                                              {"seed", 1234.0}};
    const loom::audio::SpeechStream::Inputs b{{"tokens", encode(*model, text_b ? text_b : kTextB)},
                                              {"seed", 99.0}};

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    const std::vector<float> want_a = one_pass(bridge, a);
    const double one_pass_s = std::chrono::duration<double>(clock::now() - t0).count();
    const std::vector<float> want_b = one_pass(bridge, b);
    LOOM_CHECK(!want_a.empty() && !want_b.empty());

    // 1. One stream.
    std::vector<float> alone_a, alone_b;
    {
        loom::audio::SpeechStream stream(bridge, a);
        t0 = clock::now();
        LOOM_CHECK(!stream.next().empty());
        const double first_s = std::chrono::duration<double>(clock::now() - t0).count();
        stream.reset();
        const Pulled got = pull_all(stream);
        const double d = max_abs_diff(got.samples, want_a);
        std::fprintf(stderr, "stream a: %zu pieces (driver's total %zu, first %zu samples), %zu samples; one "
                             "pass %zu, max|d| %.3g; first piece after %.2f s, one pass took %.2f s\n",
                     got.pieces, stream.pieces_total(), got.first_piece, got.samples.size(), want_a.size(), d,
                     first_s, one_pass_s);
        LOOM_CHECK(got.samples.size() == want_a.size());
        LOOM_CHECK(d <= kOnePassTolerance);
        LOOM_CHECK(got.pieces >= 2);
        LOOM_CHECK(stream.pieces_total() == 0 || stream.pieces_total() == stream.pieces_done());
        LOOM_CHECK(got.first_piece < want_a.size());
        LOOM_CHECK(stream.samples() == want_a.size());
        LOOM_CHECK(stream.next().empty());
        alone_a = got.samples;

        // 2. Reset.
        stream.reset();
        LOOM_CHECK(!stream.finished() && stream.samples() == 0);
        LOOM_CHECK(pull_all(stream).samples == alone_a);
    }
    {
        loom::audio::SpeechStream stream(bridge, b);
        alone_b = pull_all(stream).samples;
        LOOM_CHECK(alone_b.size() == want_b.size() && max_abs_diff(alone_b, want_b) <= kOnePassTolerance);
    }

    // 3. Interleaved, under both cache modes.
    for (loom::StreamCaches mode : {loom::StreamCaches::Park, loom::StreamCaches::PerStream}) {
        const bool park = mode == loom::StreamCaches::Park;
        bridge.set_stream_caches(mode);
        loom::audio::SpeechStream sa(bridge, a);
        loom::audio::SpeechStream sb(bridge, b);
        std::vector<float> got_a, got_b;
        bool one_pass_ok = true;
        size_t turn = 0;
        while (!sa.finished() || !sb.finished()) {
            const std::vector<float> pa = sa.next();
            got_a.insert(got_a.end(), pa.begin(), pa.end());
            if (turn++ == 0) one_pass_ok = one_pass_ok && one_pass(bridge, b) == want_b;
            const std::vector<float> pb = sb.next();
            got_b.insert(got_b.end(), pb.begin(), pb.end());
        }
        std::fprintf(stderr, "%s: a %s, b %s, one pass between %s\n", park ? "park" : "per-stream",
                     got_a == alone_a ? "ok" : "DIFFERS", got_b == alone_b ? "ok" : "DIFFERS",
                     one_pass_ok ? "ok" : "DIFFERS");
        LOOM_CHECK(got_a == alone_a);
        LOOM_CHECK(got_b == alone_b);
        LOOM_CHECK(one_pass_ok);
    }

    // An earlier export: the one-pass loop only, refused by the door.
    if (const char* old = loom_test::fixture_env("LOOM_STREAM_TTS_OLD_GGUF")) {
        auto old_model = loom::GgufModel::load(old, backend.get());
        loom::Session old_session(*old_model, backend.get());
        LOOM_CHECK(!loom::audio::SpeechStream::supported(old_session.bridge()));
        LOOM_CHECK_THROWS(loom::audio::SpeechStream(old_session.bridge(), a), loom::LoadError);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
