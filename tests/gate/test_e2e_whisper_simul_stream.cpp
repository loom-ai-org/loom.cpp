// Whisper live (ADR-128): ufal SimulStreaming's AlignAtt through `TranscriptionStream`. Every `chunk_seconds` of
// audio is a segment of a 30 s buffer; each update re-reads the committed tokens and decodes on until the end
// of text, a rewind, or a token whose most-attended frame (the alignment heads' attention, z-scored, median
// filtered, averaged) is within `frame_threshold` frames of the audio's end.
//
// The oracle is SimulStreaming itself (`scripts/whisper_simulstreaming_reference.py`: its `PaddedAlignAttWhisper`
// over the HF checkpoint renamed into OpenAI's layout, the clip in `chunk` pieces), one directory per clip,
// language and chunk, `<clip>.<lang>.<chunk>/`.
//
//   1. Every update, at pushes of one chunk: the tokens it started from (context + start of transcript +
//      committed blocks), the tokens it added, and the most-attended frame after each token decoded (to
//      `kFrameSlack`: SimulStreaming z-scores in f32, the driver in double).
//   2. The ids after `finish` are SimulStreaming's at one-chunk pushes, 100 ms pushes and the whole clip; and
//      at 10 ms and 0.5 s on the first case.
//   3. The text is SimulStreaming's, and arrives before the end without being revised.
//   4. Two streams pushed in alternation, with a one-pass `transcribe` between their pushes, each end where they
//      end alone. A chunk out of range is refused.
//   5. With LOOM_WHISPER_STREAM_OLD_GGUF, the file exported before the stream functions: the same one-pass text,
//      and the live door refuses it.
//
// LOOM_WHISPER_STREAM_GGUF, LOOM_WHISPER_STREAM_CLIPS_DIR (`<clip>.npy`, mono F32 at 16 kHz),
// LOOM_WHISPER_STREAM_REF_DIR.

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
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;
constexpr size_t kRate = 16000;
constexpr double kFrameSlack = 2.0;

struct Step {
    size_t pushed = 0;
    bool last = false;
    std::vector<int32_t> ctx, added;
    std::vector<double> frames;
};

struct Case {
    std::string dir, clip, lang;
    double chunk = 0.0;
    double rewind = 0.0;  // `rewind.txt`: a reference made at another `rewind_threshold`
    std::vector<Step> steps;
    std::string text;
};

template <typename T>
std::vector<T> numbers(const std::string& row) {
    std::istringstream in(row);
    std::string head;
    in >> head;
    std::vector<T> out;
    for (T v; in >> v;) out.push_back(v);
    return out;
}

Case read_case(const std::filesystem::path& dir) {
    Case c;
    c.dir = dir.string();
    const std::string name = dir.filename().string();
    const size_t a = name.find('.'), b = name.find('.', a + 1);
    c.clip = name.substr(0, a);
    c.lang = name.substr(a + 1, b - a - 1);
    c.chunk = std::stod(name.substr(b + 1));
    std::ifstream steps(dir / "steps.txt");
    for (std::string row; std::getline(steps, row);) {
        if (row.rfind("step ", 0) == 0) {
            const auto v = numbers<double>(row);
            Step s;
            s.pushed = static_cast<size_t>(v[1]);
            s.last = v[2] != 0.0;
            c.steps.push_back(s);
        } else if (row.rfind("ctx", 0) == 0) {
            c.steps.back().ctx = numbers<int32_t>(row);
        } else if (row.rfind("new", 0) == 0) {
            c.steps.back().added = numbers<int32_t>(row);
        } else if (row.rfind("frames", 0) == 0) {
            c.steps.back().frames = numbers<double>(row);
        }
    }
    if (std::ifstream rewind(dir / "rewind.txt"); rewind) rewind >> c.rewind;
    std::ifstream text(dir / "text.txt");
    c.text = std::string((std::istreambuf_iterator<char>(text)), std::istreambuf_iterator<char>());
    return c;
}

std::vector<float> read_f32(const std::string& path) {
    std::vector<int64_t> shape;
    return loom_test::read_npy_f32(path, shape);
}

std::string trimmed(const std::string& s) {
    const char* space = " \t\n\r\f\v";
    const size_t a = s.find_first_not_of(space);
    if (a == std::string::npos) return {};
    return s.substr(a, s.find_last_not_of(space) - a + 1);
}

std::vector<double> state_array(const loom::audio::TranscriptionStream& s, const char* key) {
    const loom::StreamState::Value* v = s.state().value(key);
    return v ? v->data : std::vector<double>{};
}

struct Run {
    std::vector<int32_t> ids;
    std::string text;
    size_t updates_before_finish = 0;
    bool revised = false;
    double seconds = 0.0;
};

Run streamed(loom::audio::TranscriptionStream& s, const std::vector<float>& wav, size_t piece) {
    Run run;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t at = 0; at < wav.size(); at += piece) {
        const loom::audio::TranscriptUpdate u = s.push(wav.data() + at, std::min(piece, wav.size() - at));
        if (!u.delta.empty()) ++run.updates_before_finish;
        run.revised = run.revised || u.revised;
    }
    const loom::audio::TranscriptUpdate last = s.finish();
    run.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    run.revised = run.revised || last.revised;
    run.ids = s.tokens();
    run.text = s.text();
    return run;
}

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_WHISPER_STREAM_GGUF");
    const char* clips_dir = loom_test::fixture_env("LOOM_WHISPER_STREAM_CLIPS_DIR");
    const char* ref_dir = loom_test::fixture_env("LOOM_WHISPER_STREAM_REF_DIR");
    if (!gguf || !clips_dir || !ref_dir) {
        std::fprintf(stderr, "SKIP: set LOOM_WHISPER_STREAM_GGUF, _CLIPS_DIR and _REF_DIR\n");
        return kSkipReturnCode;
    }
    std::vector<Case> cases;
    for (const auto& e : std::filesystem::directory_iterator(ref_dir))
        if (e.is_directory()) cases.push_back(read_case(e.path()));
    std::sort(cases.begin(), cases.end(), [](const Case& a, const Case& b) { return a.dir < b.dir; });
    LOOM_CHECK(!cases.empty());

    loom::Device device = loom::Device::open(loom_test::test_device_name());
    std::fprintf(stderr, "device: %s\n", device.name().c_str());
    auto model = loom::GgufModel::load(gguf, device.backends().primary);
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, device.backends());
    loom::LoomLuaBridge& bridge = session.bridge();
    LOOM_CHECK(loom::audio::TranscriptionStream::supported(bridge));

    double audio_s = 0.0, stream_s = 0.0;
    std::vector<Run> alone;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const Case& c = cases[ci];
        const std::vector<float> wav = read_f32(std::string(clips_dir) + "/" + c.clip + ".npy");
        loom::audio::TranscriptionStream::Inputs inputs = {
            {"chunk_seconds", c.chunk},
            {"language", loom::audio::TranscriptionStream::language_input(*model, c.lang)}};
        if (c.rewind > 0) inputs["rewind_threshold"] = c.rewind;
        loom::audio::TranscriptionStream stream(bridge, *model, inputs);
        const size_t chunk = static_cast<size_t>(c.chunk * kRate);

        // 1. Every update at one-chunk pushes: one update per push, two at the end when a short segment is
        //    left (the last of them `is_last`).
        std::vector<int32_t> want;
        for (const Step& s : c.steps) want.insert(want.end(), s.added.begin(), s.added.end());
        size_t step = 0, ctx_same = 0, added_same = 0, frames_counted = 0;
        double frame_worst = 0.0;
        const auto compare = [&](const Step& s) {
            const auto start = state_array(stream, "start_tokens");
            const auto frames = state_array(stream, "frames");
            ctx_same += std::vector<int32_t>(start.begin(), start.end()) == s.ctx;
            LOOM_CHECK(std::vector<int32_t>(start.begin(), start.end()) == s.ctx);
            LOOM_CHECK(frames.size() == s.frames.size());
            for (size_t i = 0; i < std::min(frames.size(), s.frames.size()); ++i) {
                frame_worst = std::max(frame_worst, std::fabs(frames[i] - s.frames[i]));
                ++frames_counted;
            }
        };
        for (size_t at = 0; at < wav.size(); at += chunk) {
            const size_t n = std::min(chunk, wav.size() - at);
            const loom::audio::TranscriptUpdate u = stream.push(wav.data() + at, n);
            if (n < chunk) continue;  // the short last segment runs at finish
            LOOM_CHECK(step < c.steps.size() && !c.steps[step].last);
            if (step >= c.steps.size()) break;
            const Step& s = c.steps[step++];
            LOOM_CHECK(u.samples == s.pushed);
            compare(s);
            added_same += u.tokens == s.added || (s.added.empty() && u.delta.empty());
        }
        stream.finish();
        // `finish`'s last update is the reference's last step.
        LOOM_CHECK(!c.steps.empty() && c.steps.back().last);
        if (!c.steps.empty()) compare(c.steps.back());
        LOOM_CHECK(stream.tokens() == want);
        LOOM_CHECK(frame_worst <= kFrameSlack);

        // 2. The ids at every cut tried.
        stream.reset();
        const Run base = streamed(stream, wav, kRate / 10);
        stream_s += base.seconds;
        audio_s += static_cast<double>(wav.size()) / kRate;
        LOOM_CHECK(base.ids == want);
        std::vector<size_t> cuts = {wav.size()};
        if (ci == 0) cuts.insert(cuts.end(), {kRate / 100, kRate / 2});
        size_t same = base.ids == want;
        for (size_t piece : cuts) {
            stream.reset();
            const Run r = streamed(stream, wav, piece);
            same += r.ids == want;
            LOOM_CHECK(r.ids == want);
        }

        // 3. The text, early and never revised.
        LOOM_CHECK(trimmed(base.text) == trimmed(c.text));
        if (c.steps.size() > 3) LOOM_CHECK(base.updates_before_finish > 0);
        LOOM_CHECK(!base.revised);
        alone.push_back(base);
        std::printf("%-28s %6.2f s  %3zu updates  ctx %zu/%zu same  frames max|d| %.0f over %zu  ids %zu (%s)  "
                    "%zu/%zu cuts same  %.2fx real time\n  \"%s\"\n",
                    std::filesystem::path(c.dir).filename().string().c_str(), static_cast<double>(wav.size()) / kRate,
                    c.steps.size(), ctx_same, step + 1, frame_worst, frames_counted, want.size(),
                    base.ids == want ? "SimulStreaming's" : "DIFFERENT", same, cuts.size() + 1,
                    static_cast<double>(wav.size()) / kRate / base.seconds, base.text.c_str());
        (void)added_same;
    }
    std::printf("streamed %.1f s of audio in 100 ms pieces in %.1f s: %.2fx real time, RTF %.3f\n", audio_s, stream_s,
                audio_s / stream_s, stream_s / audio_s);

    const Case& first = cases.front();
    const std::vector<float> first_wav = read_f32(std::string(clips_dir) + "/" + first.clip + ".npy");
    const auto inputs_of = [&](const Case& c) {
        loom::audio::TranscriptionStream::Inputs in{
            {"chunk_seconds", c.chunk}, {"language", loom::audio::TranscriptionStream::language_input(*model, c.lang)}};
        if (c.rewind > 0) in["rewind_threshold"] = c.rewind;
        return in;
    };

    // 4. Two streams and a one-pass call between them; sizes out of range.
    {
        const Case& last = cases.back();
        const std::vector<float> b_wav = read_f32(std::string(clips_dir) + "/" + last.clip + ".npy");
        loom::audio::TranscriptionStream a(bridge, *model, inputs_of(first));
        loom::audio::TranscriptionStream b(bridge, *model, inputs_of(last));
        const size_t n = std::max(first_wav.size(), b_wav.size()), piece = kRate / 10;
        for (size_t at = 0; at < n; at += piece) {
            if (at < first_wav.size()) a.push(first_wav.data() + at, std::min(piece, first_wav.size() - at));
            if (at == piece * 20) loom::audio::transcribe(bridge, *model, first_wav, {});
            if (at < b_wav.size()) b.push(b_wav.data() + at, std::min(piece, b_wav.size() - at));
        }
        a.finish();
        b.finish();
        LOOM_CHECK(a.tokens() == alone.front().ids);
        LOOM_CHECK(b.tokens() == alone.back().ids);
        using Inputs = loom::audio::TranscriptionStream::Inputs;
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(bridge, *model, Inputs{{"chunk_seconds", 0.0}}), loom::Error);
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(bridge, *model, Inputs{{"chunk_seconds", 30.0}}), loom::Error);
    }

    // 5. The file exported before the door.
    if (const char* old_gguf = loom_test::fixture_env("LOOM_WHISPER_STREAM_OLD_GGUF")) {
        auto old_model = loom::GgufModel::load(old_gguf, device.backends().primary);
        LOOM_CHECK(old_model != nullptr);
        loom::Session old_session(*old_model, device.backends());
        LOOM_CHECK(loom::audio::transcribe(old_session.bridge(), *old_model, first_wav, {}).text ==
                   loom::audio::transcribe(bridge, *model, first_wav, {}).text);
        LOOM_CHECK(!loom::audio::TranscriptionStream::supported(old_session.bridge()));
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(old_session.bridge(), *old_model), loom::LoadError);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
