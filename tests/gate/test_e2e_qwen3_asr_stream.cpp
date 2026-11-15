// Qwen3-ASR live (ADR-121): upstream's streaming mode through `TranscriptionStream` -- every 2 s all the audio
// so far decoded again, from the third update on with the previous answer minus its last 5 ids as the start of
// the assistant turn. The stream ends on upstream's streamed transcript, not `infer`'s. Upstream re-feeds
// everything; the driver re-encodes only the 8 s encoder windows that changed and keeps the decoder's keys
// over the rest, so the 60 s clip (7 windows) is what tells the two apart if a kept window goes stale.
//
// The oracle is qwen-asr 0.0.6's `streaming_transcribe` loop run on transformers (greedy, f32, eager) in place
// of vLLM: `scripts/qwen3_asr_stream_reference.py` writes `<clip>/samples.npy` (the audio each update decoded),
// `hyp_<k>.npy` (the transcript ids after update k) and `text_<k>.txt` (upstream's text after it).
//
//   1. Every update's transcript is upstream's, in 100 ms pieces.
//      A `<clip>@<seconds>/` reference ran at another cadence (0.5 s): many short, unsteady hypotheses. Most
//      clips cannot tell one prefix from another, so the set is chosen by sabotage, and each sabotage is caught
//      by one case only: a stream that never prefixes, by sv_ja at 2 s (its prefix keeps 弁当生, where a fresh
//      decode reads 弁当制); one that drops 3 ids instead of 5, by para_zh at 0.5 s (5 of 27 updates differ).
//   2. The transcript after `finish` is the same at every cut tried (10 ms and the whole clip on the first clip,
//      2 s and the whole clip on the rest): where pushes break never matters, only the cadence.
//   3. Text arrives before the end. A re-decoding stream may revise it (`revised`); the last 5 ids are re-decided
//      at every update.
//   4. A cadence longer than the clip decodes once, at `finish`, with no prefix: `transcribe`'s own text.
//   5. A stream that outgrows the decoder's cache is refused by name, never under 30 s, and every update before
//      the refusal is still upstream's (oruk-loop's cache holds one 30 s piece; Qwen3-ASR's takes the 60 s
//      clip whole). `update_seconds <= 0` is refused too.
//   6. Two streams pushed in alternation, with a one-pass `transcribe` between their pushes, each end where they
//      end alone.
//   7. With LOOM_QWEN3_ASR_STREAM_OLD_GGUF, the file exported before the stream functions: the same text in one
//      call, and the live door refuses it.
//   8. A TENSOR check, per clip: after `finish`, the decoder's K and V over the prompt and every audio row are
//      the ones a one-pass `transcribe` of the same clip writes, which encodes the whole clip at once. A kept
//      window gone stale often flips no token: keeping window 0 before the 200 samples past its end arrived
//      (its last frame read the reflection) and never re-encoding a kept window when the floor moved both
//      passed sections 1-7 on every clip. The prompt's length and the rows' count are the driver's own
//      (`#S_AUDIO_PREFIX`, `S_audio_rows` per window), read by loading its script with a probe appended.
//
// LOOM_QWEN3_ASR_STREAM_GGUF: qwen3-asr-0.6b exported with the stream functions.
// LOOM_QWEN3_ASR_STREAM_CLIPS_DIR: `<clip>.npy`, mono F32 at 16 kHz. LOOM_QWEN3_ASR_STREAM_REF_DIR: the reference.

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <variant>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;
constexpr size_t kRate = 16000;
constexpr size_t kUpdate = 2 * kRate;
constexpr size_t kMaxSamples = 30 * kRate;
// Section 8's bound on max|stream - one pass| / max|one pass| over the cached K and V. Intact, the worst clip is
// 8.3e-06 (long60, 7 windows; a single window is exactly 0); the two stale-window sabotages it exists for read
// 2.9e-04 (window 0 kept before its last frame's samples, jfk) and 1.8e-03 (the floor rule off, long60, the
// one clip whose floor moves under a kept window). Six times from either.
constexpr double kKeysTolerance = 5e-5;

std::vector<float> read_f32(const std::string& path) {
    std::vector<int64_t> shape;
    return loom_test::read_npy_f32(path, shape);
}

std::vector<int32_t> as_ids(const std::vector<float>& v) { return std::vector<int32_t>(v.begin(), v.end()); }

std::string read_text(const std::string& path) {
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// Every layer's K then V over positions [0, n), flattened. Views over the cache read without a graph
// (test_kv_cache.cpp says why).
std::vector<float> cached(const loom::KvCache& kv, uint32_t n) {
    ggml_context_ptr ctx(ggml_init(ggml_init_params{256 * 1024, nullptr, /*no_alloc=*/true}));
    std::vector<float> out;
    for (uint32_t layer = 0; layer < kv.n_layer(); ++layer) {
        for (bool value : {false, true}) {
            ggml_tensor* view = value ? kv.read_v(ctx.get(), layer, n) : kv.read_k(ctx.get(), layer, n);
            const size_t at = out.size();
            out.resize(at + ggml_nelements(view));
            ggml_backend_tensor_get(view, out.data() + at, 0, ggml_nbytes(view));
        }
    }
    return out;
}

// max|a - b| / max|b|.
double relative(const std::vector<float>& a, const std::vector<float>& b) {
    double diff = 0.0, peak = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        diff = std::max(diff, static_cast<double>(std::fabs(a[i] - b[i])));
        peak = std::max(peak, static_cast<double>(std::fabs(b[i])));
    }
    return peak > 0.0 ? diff / peak : diff;
}

std::string trimmed(const std::string& s) {
    const char* space = " \t\n\r\f\v";
    const size_t a = s.find_first_not_of(space);
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(space) - a + 1);
}

struct Clip {
    std::string name;                         // `<clip>` or `<clip>@<update seconds>`
    double seconds = 2.0;                     // the cadence the reference ran at
    size_t update = kUpdate;                  // ... in samples
    std::vector<float> wav;
    std::vector<float> samples;               // audio per update
    std::vector<std::vector<int32_t>> hyp;    // transcript ids after each update
    std::vector<std::string> text;            // upstream's text after each
};

struct Run {
    std::vector<int32_t> ids;
    std::string text;
    size_t updates_before_finish = 0;
    size_t updates_matched = 0;
    bool revised = false;
    double seconds = 0.0;
};

// Pushes `wav` in `piece`s; with `clip`, checks the transcript after each update against upstream's.
// `matched`, when given, follows `updates_matched` as it goes, so a stream refused midway still says how far it
// agreed.
Run streamed(loom::audio::TranscriptionStream& s, const std::vector<float>& wav, size_t piece,
             const Clip* clip = nullptr, size_t* matched = nullptr) {
    Run run;
    const auto t0 = std::chrono::steady_clock::now();
    size_t heard = 0;
    for (size_t at = 0; at < wav.size(); at += piece) {
        const size_t n = std::min(piece, wav.size() - at);
        const loom::audio::TranscriptUpdate u = s.push(wav.data() + at, n);
        if (!u.delta.empty()) ++run.updates_before_finish;
        run.revised = run.revised || u.revised;
        const size_t every = clip ? clip->update : kUpdate;
        const size_t before = heard / every;
        heard += n;
        if (clip && heard / every > before && heard / every <= clip->hyp.size()) {
            const size_t k = heard / every - 1;
            const bool same = u.tokens == clip->hyp[k];
            LOOM_CHECK(same);
            if (!same) {
                std::printf("  %s update %zu: got \"%s\"\n  upstream          \"%s\"\n", clip->name.c_str(), k,
                            u.text.c_str(), clip->text[k].c_str());
            }
            run.updates_matched += same;
            if (matched) *matched = run.updates_matched;
        }
    }
    const loom::audio::TranscriptUpdate last = s.finish();
    run.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    run.revised = run.revised || last.revised;
    run.ids = s.tokens();
    run.text = s.text();
    if (clip && clip->samples.back() != static_cast<float>(clip->hyp.size() * clip->update)) {
        run.updates_matched += run.ids == clip->hyp.back();  // finish's own update
    }
    return run;
}

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_QWEN3_ASR_STREAM_GGUF");
    const char* clips_dir = loom_test::fixture_env("LOOM_QWEN3_ASR_STREAM_CLIPS_DIR");
    const char* ref_dir = loom_test::fixture_env("LOOM_QWEN3_ASR_STREAM_REF_DIR");
    if (!gguf || !clips_dir || !ref_dir) {
        std::fprintf(stderr, "SKIP: set LOOM_QWEN3_ASR_STREAM_GGUF, _CLIPS_DIR and _REF_DIR\n");
        return kSkipReturnCode;
    }
    std::vector<Clip> clips;
    for (const auto& e : std::filesystem::directory_iterator(ref_dir)) {
        if (!e.is_directory()) continue;
        Clip c;
        c.name = e.path().filename().string();
        const size_t at = c.name.find('@');
        if (at != std::string::npos) {
            c.seconds = std::stod(c.name.substr(at + 1));
            c.update = static_cast<size_t>(std::lround(c.seconds * kRate));
        }
        c.wav = read_f32(std::string(clips_dir) + "/" + c.name.substr(0, at) + ".npy");
        c.samples = read_f32(e.path().string() + "/samples.npy");
        for (size_t k = 0; k < c.samples.size(); ++k) {
            c.hyp.push_back(as_ids(read_f32(e.path().string() + "/hyp_" + std::to_string(k) + ".npy")));
            c.text.push_back(read_text(e.path().string() + "/text_" + std::to_string(k) + ".txt"));
        }
        clips.push_back(std::move(c));
    }
    std::sort(clips.begin(), clips.end(), [](const Clip& a, const Clip& b) {
        return a.wav.size() == b.wav.size() ? a.name < b.name : a.wav.size() < b.wav.size();
    });
    LOOM_CHECK(!clips.empty());

    ggml_backend_ptr backend(loom_test::cpu_backend());
    auto model = loom::GgufModel::load(gguf, backend.get());
    LOOM_CHECK(model != nullptr);
    // Not a `Session`: section 8 reads the decoder's cache, which a Session keeps to itself. The caches are
    // declared first because the bridge holds them by pointer until it is gone (session.h).
    loom::ModuleCaches caches;
    loom::LoomLuaBridge bridge(backend.get());
    caches = loom::register_topologies(*model, backend.get(), bridge);
    const std::string driver = model->kv_str("model.driver_script");
    bridge.load_script(driver);
    LOOM_CHECK(loom::audio::TranscriptionStream::supported(bridge));
    LOOM_CHECK(caches.kv != nullptr);
    // Leased, so a stream's rows are in `caches.kv` once it has run (Auto picks this on the CPU anyway).
    bridge.set_stream_caches(loom::StreamCaches::Park);
    // The driver's prompt length and row count, from its own locals: its script again, a probe appended.
    loom::LoomLuaBridge probe(backend.get());
    probe.load_script(driver + R"(
function gate_prompt_rows(inputs)
    local rows, first = 0, 0
    while first < inputs.n_samples do
        rows = rows + S_audio_rows(math.min(S_WINDOW_SAMPLES, inputs.n_samples - first))
        first = first + S_WINDOW_SAMPLES
    end
    return {#S_AUDIO_PREFIX + rows}
end
)");
    auto prompt_rows = [&](size_t n_samples) {
        const auto v = std::get<std::vector<double>>(
            probe.call("gate_prompt_rows", {{"n_samples", static_cast<double>(n_samples)}}));
        return static_cast<uint32_t>(v.at(0));
    };

    double audio_s = 0.0, stream_s = 0.0;
    std::vector<const Clip*> short_clips;
    std::vector<Run> alone;
    bool refused_long = false;
    for (const Clip& c : clips) {
        loom::audio::TranscriptionStream stream(bridge, *model, {{"update_seconds", c.seconds}});
        // 1. Every update -- or, 5., every update until the stream outgrows the decoder's cache, which may
        // happen only past 30 s (oruk-loop's cache holds one 30 s piece; Qwen3-ASR's about five minutes).
        Run base;
        size_t matched = 0;
        try {
            base = streamed(stream, c.wav, kRate / 10, &c, &matched);
        } catch (const loom::Error& e) {
            const bool outgrown = std::string(e.what()).find("outgrown the decoder's cache") != std::string::npos;
            LOOM_CHECK(outgrown && c.wav.size() > kMaxSamples);
            refused_long = refused_long || outgrown;
            std::printf("%-26s %6.2f s  %2zu updates upstream's, then refused at the cache: %s\n", c.name.c_str(),
                        c.wav.size() / double(kRate), matched, outgrown ? "yes" : e.what());
            continue;
        }
        stream_s += base.seconds;
        audio_s += static_cast<double>(c.wav.size()) / kRate;
        LOOM_CHECK(base.ids == c.hyp.back());
        LOOM_CHECK(base.updates_matched == c.hyp.size());
        // 8. The keys it kept are the one pass's.
        const uint32_t n_prompt = prompt_rows(c.wav.size());
        const std::vector<float> kept_kv = cached(*caches.kv, n_prompt);
        loom::audio::transcribe(bridge, *model, c.wav, {});
        const double keys = relative(kept_kv, cached(*caches.kv, n_prompt));
        LOOM_CHECK(keys < kKeysTolerance);
        // 2. Other cuts.
        std::vector<size_t> cuts = {c.wav.size(), short_clips.empty() ? kRate / 100 : c.update};
        size_t same = base.ids == c.hyp.back();
        for (size_t piece : cuts) {
            stream.reset();
            const Run r = streamed(stream, c.wav, piece);
            same += r.ids == c.hyp.back();
            LOOM_CHECK(r.ids == c.hyp.back());
        }
        // 3. Text before the end.
        if (c.wav.size() > c.update + kRate) LOOM_CHECK(base.updates_before_finish > 0);
        if (c.update == kUpdate) {
            short_clips.push_back(&c);
            alone.push_back(base);
        }
        std::printf("%-26s %6.2f s  %2zu updates, %2zu upstream's  %zu/%zu cuts same  %zu with text before "
                    "finish%s  keys %.1e over %u positions  %.2fx real time\n  \"%s\"\n",
                    c.name.c_str(), c.wav.size() / double(kRate), c.hyp.size(), base.updates_matched, same,
                    cuts.size() + 1, base.updates_before_finish, base.revised ? " (revised)" : "", keys, n_prompt,
                    c.wav.size() / double(kRate) / base.seconds, base.text.c_str());
    }
    std::printf("streamed %.1f s of audio in 100 ms pieces in %.1f s: %.2fx real time, RTF %.3f\n", audio_s,
                stream_s, audio_s / stream_s, stream_s / audio_s);
    LOOM_CHECK(!short_clips.empty());

    const Clip& first = *short_clips.front();
    // 4. One update, at finish, no prefix: the one pass.
    {
        loom::audio::TranscriptionStream once(bridge, *model, {{"update_seconds", 1000.0}});
        const Run r = streamed(once, first.wav, kRate / 10);
        const std::string one_pass = loom::audio::transcribe(bridge, *model, first.wav, {}).text;
        LOOM_CHECK(trimmed(r.text) == trimmed(one_pass));
        LOOM_CHECK(r.updates_before_finish == 0);
        std::printf("one update at finish: \"%s\"\n             one pass: \"%s\"\n", r.text.c_str(),
                    one_pass.c_str());
    }
    // 5. A cadence of nothing.
    LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(bridge, *model, {{"update_seconds", 0.0}}), loom::Error);

    // 6. Two streams and a one-pass call between them.
    {
        const Clip& last = *short_clips.back();
        loom::audio::TranscriptionStream a(bridge, *model);
        loom::audio::TranscriptionStream b(bridge, *model);
        const size_t n = std::max(first.wav.size(), last.wav.size()), piece = kRate / 10;
        for (size_t at = 0; at < n; at += piece) {
            if (at < first.wav.size()) a.push(first.wav.data() + at, std::min(piece, first.wav.size() - at));
            if (at == piece * 25) loom::audio::transcribe(bridge, *model, first.wav, {});
            if (at < last.wav.size()) b.push(last.wav.data() + at, std::min(piece, last.wav.size() - at));
        }
        a.finish();
        b.finish();
        LOOM_CHECK(a.tokens() == alone.front().ids && a.text() == alone.front().text);
        LOOM_CHECK(b.tokens() == alone.back().ids && b.text() == alone.back().text);
    }

    // 7. The file exported before the door.
    if (const char* old_gguf = loom_test::fixture_env("LOOM_QWEN3_ASR_STREAM_OLD_GGUF")) {
        auto old_model = loom::GgufModel::load(old_gguf, backend.get());
        LOOM_CHECK(old_model != nullptr);
        loom::Session old_session(*old_model, backend.get());
        LOOM_CHECK(loom::audio::transcribe(old_session.bridge(), *old_model, first.wav, {}).text ==
                   loom::audio::transcribe(bridge, *model, first.wav, {}).text);
        LOOM_CHECK(!loom::audio::TranscriptionStream::supported(old_session.bridge()));
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(old_session.bridge(), *old_model), loom::LoadError);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
