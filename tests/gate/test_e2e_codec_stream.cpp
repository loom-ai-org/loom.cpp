// The codec half of a streamed pair (ADR-102) on a real codec: `CodecStream` decodes codes pushed a few
// frames at a time to what one call over all of them decodes, and streams on one codec do not see each other.
//
//   1. One stream, the codes pushed in uneven pieces: the same sample count as `infer`, and within f32
//      summation order of it while the clip is one of `infer`'s chunks (Qwen3-TTS's 12 Hz codec: 300
//      frames). Past that, `infer` is the reference's `chunked_decode`, whose 25-frame left context cuts a
//      receptive field the stream keeps -- the difference is printed, not bounded.
//   2. `reset` decodes the same samples again, bit for bit.
//   3. Two streams pushed in alternation, with a one-pass `infer` between their pushes: each gives exactly
//      what it gave alone.
//   4. With LOOM_STREAM_CODEC_OLD_GGUF, a file exported before the step: its one-pass decode is this
//      file's, bit for bit (the decode graph did not move), and the stream door refuses it.
//
// LOOM_STREAM_CODEC_GGUF: a codec exported with the stream functions (qwen3-tts-tokenizer-12hz,
// audio8-tts-codec and moss-audio-tokenizer-v2 -- whose `infer` is one call at any length, so their streams
// are checked to the end -- and dac-44khz, whose windows match one call to f32 rounding).
// LOOM_STREAM_CODES_NPY: `[frames, width]` codes, F32, from the codec's own LM -- random ids decode to noise
// that clips at +-1, where a difference can hide.

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
// The step against one call: 2.3e-05 on random codes at export (`codec_stream_export.check_step`).
constexpr double kOnePassTolerance = 1e-3;
constexpr size_t kOneChunk = 300;

std::vector<float> one_pass(loom::LoomLuaBridge& bridge, const std::vector<int32_t>& codes) {
    const auto got = std::get<std::vector<double>>(
        bridge.call("infer", {{"codes", std::vector<double>(codes.begin(), codes.end())}}));
    return std::vector<float>(got.begin(), got.end());
}

std::vector<float> pushed(loom::audio::CodecStream& s, const std::vector<int32_t>& codes, size_t width,
                          const std::vector<size_t>& pieces) {
    std::vector<float> out;
    size_t at = 0, i = 0;
    const size_t frames = codes.size() / width;
    while (at < frames) {
        const size_t n = std::min(pieces[i++ % pieces.size()], frames - at);
        const std::vector<int32_t> piece(codes.begin() + at * width, codes.begin() + (at + n) * width);
        const std::vector<float> wav = s.push(piece);
        out.insert(out.end(), wav.begin(), wav.end());
        at += n;
    }
    const std::vector<float> tail = s.finish();
    out.insert(out.end(), tail.begin(), tail.end());
    return out;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) d = std::max(d, std::abs(double(a[i]) - b[i]));
    return d;
}

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_STREAM_CODEC_GGUF");
    const char* npy = loom_test::fixture_env("LOOM_STREAM_CODES_NPY");
    if (gguf == nullptr || npy == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_STREAM_CODEC_GGUF and LOOM_STREAM_CODES_NPY\n");
        return kSkipReturnCode;
    }
    std::vector<int64_t> shape;
    const std::vector<float> raw = loom_test::read_npy_f32(npy, shape);
    LOOM_CHECK(shape.size() == 2 && shape[0] > 0);
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf, backend.get());
    LOOM_CHECK(model != nullptr);
    // Codes narrower than the codec (MOSS-TTS emits 12 of its codec's 32) are widened with the codec's
    // absent id, as every host widens them (ADR-050).
    const size_t width = model->hparam_u32("codec.n_codebooks");
    const size_t given = static_cast<size_t>(shape[1]);
    LOOM_CHECK(given == width || (given < width && model->has_kv("loom.codec.absent_code")));
    std::vector<int32_t> codes;
    for (size_t f = 0; f < raw.size() / given; ++f) {
        for (size_t c = 0; c < width; ++c) {
            codes.push_back(c < given ? static_cast<int32_t>(raw[f * given + c])
                                      : static_cast<int32_t>(model->hparam_u32("codec.absent_code")));
        }
    }
    const size_t frames = codes.size() / width;
    loom::Session session(*model, backend.get());
    loom::LoomLuaBridge& bridge = session.bridge();
    LOOM_CHECK(loom::audio::CodecStream::supported(bridge));

    const std::vector<float> want = one_pass(bridge, codes);
    LOOM_CHECK(!want.empty());

    // 1. One stream, uneven pieces.
    std::vector<float> alone;
    {
        loom::audio::CodecStream s(bridge);
        alone = pushed(s, codes, width, {1, 2, 3, 5, 8, 13});
        const double d = max_abs_diff(alone, want);
        std::fprintf(stderr, "%zu frames: stream %zu samples, one pass %zu, max|d| %.3g\n", frames, alone.size(),
                     want.size(), d);
        LOOM_CHECK(alone.size() == want.size());
        if (frames <= kOneChunk) {
            LOOM_CHECK(d <= kOnePassTolerance);
        } else {
            // Inside the first chunk the two are still the same decode.
            const size_t first = kOneChunk * (want.size() / frames);
            const std::vector<float> a(alone.begin(), alone.begin() + first), b(want.begin(), want.begin() + first);
            std::fprintf(stderr, "first %zu frames max|d| %.3g; past them the reference's chunks differ\n",
                         kOneChunk, max_abs_diff(a, b));
            LOOM_CHECK(max_abs_diff(a, b) <= kOnePassTolerance);
        }
        LOOM_CHECK(s.samples() == alone.size());
        LOOM_CHECK_THROWS(s.push({}), loom::Error);

        // 2. Reset: the same pushes again, the same samples.
        s.reset();
        LOOM_CHECK(pushed(s, codes, width, {1, 2, 3, 5, 8, 13}) == alone);
    }

    // 3. Two streams in alternation, a one-pass decode between their pushes. Stream b decodes the codes
    // in another grouping, so it is checked against itself alone, not against a.
    for (loom::StreamCaches mode : {loom::StreamCaches::Park, loom::StreamCaches::PerStream}) {
        bridge.set_stream_caches(mode);
        loom::audio::CodecStream b_alone_stream(bridge);
        const std::vector<float> b_alone = pushed(b_alone_stream, codes, width, {7});
        loom::audio::CodecStream a(bridge), b(bridge);
        std::vector<float> got_a, got_b;
        bool one_pass_ok = true;
        size_t at_a = 0, at_b = 0, turn = 0;
        const std::vector<size_t> pieces_a = {1, 2, 3, 5, 8, 13};
        while (at_a < frames || at_b < frames) {
            if (at_a < frames) {
                const size_t n = std::min(pieces_a[turn % pieces_a.size()], frames - at_a);
                const std::vector<float> w = a.push({codes.begin() + at_a * width, codes.begin() + (at_a + n) * width});
                got_a.insert(got_a.end(), w.begin(), w.end());
                at_a += n;
            }
            if (turn++ == 1) one_pass_ok = one_pass(bridge, codes) == want;
            if (at_b < frames) {
                const size_t n = std::min<size_t>(7, frames - at_b);
                const std::vector<float> w = b.push({codes.begin() + at_b * width, codes.begin() + (at_b + n) * width});
                got_b.insert(got_b.end(), w.begin(), w.end());
                at_b += n;
            }
        }
        const std::vector<float> tail_a = a.finish(), tail_b = b.finish();
        got_a.insert(got_a.end(), tail_a.begin(), tail_a.end());
        got_b.insert(got_b.end(), tail_b.begin(), tail_b.end());
        std::fprintf(stderr, "%s: a %s, b %s, one pass between %s\n",
                     mode == loom::StreamCaches::Park ? "park" : "per-stream", got_a == alone ? "ok" : "DIFFERS",
                     got_b == b_alone ? "ok" : "DIFFERS", one_pass_ok ? "ok" : "DIFFERS");
        LOOM_CHECK(got_a == alone);
        LOOM_CHECK(got_b == b_alone);
        LOOM_CHECK(one_pass_ok);
    }

    // 4. A file exported before the step.
    if (const char* old = loom_test::fixture_env("LOOM_STREAM_CODEC_OLD_GGUF")) {
        auto old_model = loom::GgufModel::load(old, backend.get());
        loom::Session old_session(*old_model, backend.get());
        LOOM_CHECK(!loom::audio::CodecStream::supported(old_session.bridge()));
        LOOM_CHECK_THROWS(loom::audio::CodecStream(old_session.bridge()), loom::LoadError);
        const bool same = one_pass(old_session.bridge(), codes) == want;
        std::fprintf(stderr, "one pass against the file before the step: %s\n", same ? "identical" : "DIFFERS");
        LOOM_CHECK(same);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
