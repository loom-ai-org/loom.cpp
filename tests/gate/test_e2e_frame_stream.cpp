// The audio-in rows-out door (ADR-106) on a real model: `FrameStream` answers each frame as its audio
// arrives with what one `classify` / `embed` call over the whole clip answers, and streams on one model do
// not see each other.
//
//   1. One stream, the audio pushed in pieces of 1, 160 (10 ms), 512, 1000 and 16000 samples, a ragged
//      cycle and the whole clip at once: the same rows as one call, within f32 summation order of it (a
//      push's graphs run over fewer frames than the clip's), and each update's `first_row` is the count
//      of rows before it.
//   2. `reset` answers the same rows again, bit for bit.
//   3. Two streams pushed in alternation, with a one-pass call between their pushes: each answers exactly
//      what it answered alone.
//   4. With LOOM_FRAME_STREAM_OLD_GGUF, the file exported before the stream functions: its one-pass answer
//      is this file's, bit for bit (`infer` did not move), and the door refuses it.
//   5. With LOOM_FRAME_STREAM_LOOKAHEAD=N, a model whose frames read audio after them (ADR-120): a stream
//      opened with `lookahead_frames = N` answers its first frame once N + 1 frames of audio are in, where
//      the exact stream has answered nothing yet; it answers as many rows as one call; and a look-ahead
//      past the exact one is refused. Its values are approximate by design and only reported.
//
// LOOM_FRAME_STREAM_GGUF: a frame classifier or frame embedder exported with the stream functions
// (silero-vad: the cell's h and c carried across pushes, ADR-106; wakehubert and marblenet-vad: the one-pass
// graph over windows holding each frame's reach, ADR-120).
// LOOM_FRAME_STREAM_WAV_NPY: mono F32 audio at the file's rate -- real speech, since a VAD on noise sits
// near one probability and hides a wrong carry; one that does not end on a frame covers `stream_finish`.

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;
// A push's frontend and head run over fewer frames than one call's -- a frame at a time is ggml's
// one-column mul_mat path, not its tiled one -- so the sums inside a matmul are split differently; nothing
// else differs. Measured on silero-vad (jfk): 1.9e-06 for every piece size up to 1000 samples, 3.0e-07 for
// the whole clip in one push. The reference's own f32-against-f64 floor is 1.0e-05. A window over the one-pass
// graph (ADR-120) splits them the same way: wakehubert 4.8e-06 on features of norm up to ~40 (0 for the
// whole clip, one window), marblenet-vad 2.4e-07.
constexpr double kTolerance = 1e-5;

std::vector<float> one_pass(loom::LoomLuaBridge& bridge, const loom::GgufModel& model,
                            const std::vector<float>& wav) {
    const loom::ModelContract contract = loom::ModelContract::read(model);
    if (contract.output_kind == loom::modality::CLASS) return loom::audio::classify(bridge, model, wav).probabilities;
    return loom::audio::embed(bridge, model, wav).values;
}

std::vector<float> pushed(loom::audio::FrameStream& s, const std::vector<float>& wav,
                          const std::vector<size_t>& pieces) {
    std::vector<float> out;
    size_t at = 0, i = 0;
    while (at < wav.size()) {
        const size_t n = std::min(pieces[i++ % pieces.size()], wav.size() - at);
        const loom::audio::FrameUpdate u = s.push(wav.data() + at, n);
        LOOM_CHECK(u.first_row * s.width() == out.size());
        out.insert(out.end(), u.values.begin(), u.values.end());
        at += n;
    }
    const loom::audio::FrameUpdate tail = s.finish();
    LOOM_CHECK(tail.first_row * s.width() == out.size());
    out.insert(out.end(), tail.values.begin(), tail.values.end());
    return out;
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) d = std::max(d, std::fabs(double(a[i]) - b[i]));
    return d;
}

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_FRAME_STREAM_GGUF");
    const char* npy = loom_test::fixture_env("LOOM_FRAME_STREAM_WAV_NPY");
    if (!gguf || !npy) {
        std::fprintf(stderr, "SKIP: set LOOM_FRAME_STREAM_GGUF and LOOM_FRAME_STREAM_WAV_NPY\n");
        return kSkipReturnCode;
    }
    ggml_backend_ptr backend(loom_test::cpu_backend());
    std::vector<int64_t> shape;
    const std::vector<float> wav = loom_test::read_npy_f32(npy, shape);
    LOOM_CHECK(!wav.empty());
    auto model = loom::GgufModel::load(gguf, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, backend.get());
    loom::LoomLuaBridge& bridge = session.bridge();

    const std::vector<float> want = one_pass(bridge, *model, wav);
    loom::audio::FrameStream s(bridge, *model);
    LOOM_CHECK(want.size() % s.width() == 0);
    std::printf("%zu samples, %zu rows of %u\n", wav.size(), want.size() / s.width(), s.width());

    // 1. Every cut of the audio answers one call's rows.
    std::vector<float> first;
    for (const std::vector<size_t>& pieces : std::vector<std::vector<size_t>>{
             {1}, {160}, {512}, {1000}, {16000}, {1, 37, 511, 513, 2048, 0}, {wav.size()}}) {
        s.reset();
        const std::vector<float> got = pushed(s, wav, pieces);
        const std::vector<float>& ref = want;
        const double d = max_diff(got, ref);
        std::printf("pieces of %zu%s: %zu rows, max|d| %.3g\n", pieces[0], pieces.size() > 1 ? "..." : "",
                    got.size() / s.width(), d);
        LOOM_CHECK(got.size() == ref.size());
        LOOM_CHECK(d <= kTolerance);
        if (pieces[0] == 512) first = got;
    }

    // 2. A reset stream is the stream it was.
    s.reset();
    LOOM_CHECK(pushed(s, wav, {512}) == first);

    // 3. Two streams in alternation, a one-pass call between their pushes.
    {
        loom::audio::FrameStream a(bridge, *model), b(bridge, *model);
        std::vector<float> got_a, got_b;
        for (size_t at = 0; at < wav.size(); at += 512) {
            const size_t n = std::min<size_t>(512, wav.size() - at);
            const loom::audio::FrameUpdate ua = a.push(wav.data() + at, n);
            got_a.insert(got_a.end(), ua.values.begin(), ua.values.end());
            if (at == 512 * 40) one_pass(bridge, *model, std::vector<float>(wav.rbegin(), wav.rbegin() + 8000));
            // `b` hears the clip backwards, so the two carries differ.
            std::vector<float> rev(wav.rbegin() + at, wav.rbegin() + at + n);
            const loom::audio::FrameUpdate ub = b.push(rev);
            got_b.insert(got_b.end(), ub.values.begin(), ub.values.end());
        }
        const loom::audio::FrameUpdate ta = a.finish(), tb = b.finish();
        got_a.insert(got_a.end(), ta.values.begin(), ta.values.end());
        got_b.insert(got_b.end(), tb.values.begin(), tb.values.end());
        LOOM_CHECK(got_a == first);
        loom::audio::FrameStream alone(bridge, *model);
        LOOM_CHECK(got_b == pushed(alone, std::vector<float>(wav.rbegin(), wav.rend()), {512}));
        LOOM_CHECK(got_a != got_b);
    }

    // 4. The file exported before the door: the same one pass, and no stream.
    if (const char* old_gguf = loom_test::fixture_env("LOOM_FRAME_STREAM_OLD_GGUF")) {
        auto old_model = loom::GgufModel::load(old_gguf, backend.get());
        LOOM_CHECK(old_model != nullptr);
        loom::Session old_session(*old_model, backend.get());
        LOOM_CHECK(one_pass(old_session.bridge(), *old_model, wav) == want);
        LOOM_CHECK(!loom::audio::FrameStream::supported(old_session.bridge()));
        LOOM_CHECK_THROWS(loom::audio::FrameStream(old_session.bridge(), *old_model), loom::LoadError);
    }

    // 5. A shorter look-ahead.
    if (const char* la = std::getenv("LOOM_FRAME_STREAM_LOOKAHEAD")) {
        const int lookahead = std::atoi(la);
        const loom::ModelContract contract = loom::ModelContract::read(*model);
        const size_t hop = static_cast<size_t>(std::lround(contract.sample_rate / s.frame_rate()));
        loom::audio::FrameStream exact(bridge, *model);
        loom::audio::FrameStream early(bridge, *model, {{"lookahead_frames", static_cast<double>(lookahead)}});
        std::vector<float> got;
        uint32_t exact_rows = 0;
        for (size_t at = 0; at < wav.size(); at += hop) {
            const size_t n = std::min(hop, wav.size() - at);
            const loom::audio::FrameUpdate u = early.push(wav.data() + at, n);
            got.insert(got.end(), u.values.begin(), u.values.end());
            exact_rows += exact.push(wav.data() + at, n).n_rows;
            if (at / hop == static_cast<size_t>(lookahead)) {
                std::printf("look-ahead %d: after %zu samples, %zu row(s) answered; exact: %u\n", lookahead,
                            at + n, got.size() / s.width(), exact_rows);
                LOOM_CHECK(got.size() / s.width() >= 1);
                LOOM_CHECK(exact_rows == 0);
            }
        }
        const loom::audio::FrameUpdate tail = early.finish();
        got.insert(got.end(), tail.values.begin(), tail.values.end());
        LOOM_CHECK(got.size() == want.size());
        double sum = 0.0;
        for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) sum += std::fabs(double(got[i]) - want[i]);
        std::printf("look-ahead %d: max|d| %.3g, mean|d| %.3g from one call\n", lookahead, max_diff(got, want),
                    sum / std::max<size_t>(1, want.size()));
        LOOM_CHECK_THROWS(loom::audio::FrameStream(bridge, *model, {{"lookahead_frames", 1e6}}), std::exception);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
