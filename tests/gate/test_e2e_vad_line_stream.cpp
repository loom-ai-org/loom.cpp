// VAD lines (ADR-127): Moonshine's streaming loop for a non-streaming recogniser through `LineStream`, Silero VAD
// cutting the audio into lines and the recogniser re-run over each line a push touched.
//
// The oracle is Moonshine's loop (`scripts/vad_lines_reference.py`, a line-for-line port of moonshine-ai/moonshine
// `core/transcriber.cpp` + `core/voice-activity-detector.cpp` over the Silero ONNX Moonshine pins), one directory
// per clip and push size, `<clip>.<push>/`.
//
//   1. The VAD: Silero's probability per hop against onnxruntime's to 1e-4, and what is compared to the threshold
//      on the same side of it at every hop.
//   2. Every push that runs, at the reference's push size: the lines it touched -- which, where they start, how
//      long, whether closed -- and each one's text. With an `--asr` reference the text is the reference
//      recogniser's (transformers, Moonshine's budget); without, it is this recogniser's one pass over the
//      reference's line.
//   3. The final lines, and the stream's text; a closed line's text never changes after it closes.
//   4. Two streams pushed in alternation, with a one-pass `transcribe` between their pushes, each end where they
//      end alone.
//
// LOOM_VAD_LINES_ASR_GGUF: the recogniser. LOOM_VAD_LINES_VAD_GGUF: Silero VAD with its live door.
// LOOM_VAD_LINES_CLIPS_DIR: `<clip>.npy`, mono F32 at 16 kHz. LOOM_VAD_LINES_REF_DIR: the reference.
// LOOM_VAD_LINES_TOKENS_PER_SECOND: Moonshine's per-line budget (6.5), for a reference made with
// `--budget moonshine`.

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;
constexpr size_t kRate = 16000;

struct RefLine {
    size_t k = 0, start = 0, samples = 0;
    bool complete = false;
    std::string text;
};

struct RefCall {
    size_t pushed = 0;
    std::vector<RefLine> lines;
};

struct Case {
    std::string dir, clip;
    size_t push = 0;
    bool has_text = false;  // made with an `--asr`: the texts are the reference recogniser's
    std::vector<RefCall> calls;
    std::vector<RefLine> lines;
};

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

RefLine parse_line(const std::string& head, const std::string& text) {
    std::istringstream in(head);
    RefLine l;
    int complete = 0;
    in >> l.k >> l.start >> l.samples >> complete;
    l.complete = complete != 0;
    l.text = trimmed(text);
    return l;
}

Case read_case(const std::filesystem::path& dir) {
    Case c;
    c.dir = dir.string();
    const std::string name = dir.filename().string();
    const size_t dot = name.rfind('.');
    c.clip = name.substr(0, dot);
    c.push = std::stoul(name.substr(dot + 1));
    c.has_text = std::filesystem::is_directory(dir / "ids") && !std::filesystem::is_empty(dir / "ids");
    std::ifstream updates(dir / "updates.txt");
    for (std::string row; std::getline(updates, row);) {
        if (row.rfind("call ", 0) == 0) {
            std::istringstream in(row.substr(5));
            size_t i = 0;
            RefCall call;
            in >> i >> call.pushed;
            c.calls.push_back(call);
        } else if (row.rfind("line ", 0) == 0) {
            const size_t tab = row.find('\t');
            c.calls.back().lines.push_back(parse_line(row.substr(5, tab - 5), row.substr(tab + 1)));
        }
    }
    std::ifstream lines(dir / "lines.txt");
    for (std::string row; std::getline(lines, row);) {
        const size_t tab = row.find('\t');
        c.lines.push_back(parse_line(row.substr(0, tab), row.substr(tab + 1)));
    }
    return c;
}

} // namespace

int main() {
    const char* asr_gguf = loom_test::fixture_env("LOOM_VAD_LINES_ASR_GGUF");
    const char* vad_gguf = loom_test::fixture_env("LOOM_VAD_LINES_VAD_GGUF");
    const char* clips_dir = loom_test::fixture_env("LOOM_VAD_LINES_CLIPS_DIR");
    const char* ref_dir = loom_test::fixture_env("LOOM_VAD_LINES_REF_DIR");
    if (!asr_gguf || !vad_gguf || !clips_dir || !ref_dir) {
        std::fprintf(stderr, "SKIP: set LOOM_VAD_LINES_ASR_GGUF, _VAD_GGUF, _CLIPS_DIR and _REF_DIR\n");
        return kSkipReturnCode;
    }
    loom::audio::LineOptions options;
    if (const char* tps = std::getenv("LOOM_VAD_LINES_TOKENS_PER_SECOND")) options.max_tokens_per_second = std::stof(tps);

    std::vector<Case> cases;
    for (const auto& e : std::filesystem::directory_iterator(ref_dir))
        if (e.is_directory()) cases.push_back(read_case(e.path()));
    std::sort(cases.begin(), cases.end(), [](const Case& a, const Case& b) { return a.dir < b.dir; });
    LOOM_CHECK(!cases.empty());

    loom::Device device = loom::Device::open(loom_test::test_device_name());
    std::fprintf(stderr, "device: %s\n", device.name().c_str());
    auto asr = loom::GgufModel::load(asr_gguf, device.backends().primary);
    auto vad = loom::GgufModel::load(vad_gguf, device.backends().primary);
    LOOM_CHECK(asr != nullptr && vad != nullptr);
    loom::Session asr_session(*asr, device.backends());
    loom::Session vad_session(*vad, device.backends());
    loom::LoomLuaBridge& asr_bridge = asr_session.bridge();
    loom::LoomLuaBridge& vad_bridge = vad_session.bridge();

    double audio_s = 0.0, stream_s = 0.0;
    std::map<std::string, std::string> alone;
    for (const Case& c : cases) {
        const std::vector<float> wav = read_f32(std::string(clips_dir) + "/" + c.clip + ".npy");
        loom::audio::LineStream stream(asr_bridge, *asr, vad_bridge, *vad, options);

        // 2. Each push that runs, at the reference's push size.
        size_t call = 0, line_checks = 0, text_same = 0;
        std::map<size_t, std::string> closed;  // 3: a closed line's text, when it closed
        bool closed_moved = false;
        const auto check = [&](const loom::audio::LineUpdate& u) {
            if (!u.ran) return;
            LOOM_CHECK(call < c.calls.size());
            if (call >= c.calls.size()) return;
            const RefCall& want = c.calls[call++];
            LOOM_CHECK(u.samples == want.pushed);
            LOOM_CHECK(u.touched.size() == want.lines.size());
            for (size_t i = 0; i < std::min(u.touched.size(), want.lines.size()); ++i) {
                const loom::audio::Line& got = stream.lines()[u.touched[i]];
                const RefLine& w = want.lines[i];
                LOOM_CHECK(u.touched[i] == w.k && got.start == w.start && got.samples == w.samples &&
                           got.complete == w.complete);
                // Without a reference text, this recogniser's one pass over the reference's line.
                std::string expect = w.text;
                if (!c.has_text && w.start + w.samples <= wav.size()) {
                    const std::vector<float> audio(wav.begin() + static_cast<long>(w.start),
                                                   wav.begin() + static_cast<long>(w.start + w.samples));
                    loom::audio::TranscribeOptions opts = options.transcribe;
                    expect = trimmed(loom::audio::transcribe(asr_bridge, *asr, audio, opts).text);
                }
                ++line_checks;
                text_same += trimmed(got.text) == expect;
                LOOM_CHECK(trimmed(got.text) == expect);
                if (got.complete) {
                    auto it = closed.find(w.k);
                    if (it != closed.end() && it->second != got.text) closed_moved = true;
                    closed[w.k] = got.text;
                }
            }
        };
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t at = 0; at < wav.size(); at += c.push)
            check(stream.push(wav.data() + at, std::min(c.push, wav.size() - at)));
        check(stream.finish());
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        LOOM_CHECK(call == c.calls.size());
        LOOM_CHECK(!closed_moved);

        // 1. The VAD.
        const std::vector<float> probs = read_f32(c.dir + "/probs.npy");
        const std::vector<float> smoothed = read_f32(c.dir + "/smoothed.npy");
        LOOM_CHECK(stream.probabilities().size() == probs.size() && stream.smoothed().size() == smoothed.size());
        double prob_worst = 0.0;
        size_t sides_same = 0;
        for (size_t h = 0; h < std::min(probs.size(), stream.probabilities().size()); ++h) {
            prob_worst = std::max(prob_worst, static_cast<double>(std::fabs(stream.probabilities()[h] - probs[h])));
            sides_same += (stream.smoothed()[h] > options.threshold) == (smoothed[h] > options.threshold);
        }
        LOOM_CHECK(prob_worst <= 1e-4);
        LOOM_CHECK(sides_same == smoothed.size());

        // 3. The final lines.
        LOOM_CHECK(stream.lines().size() == c.lines.size());
        for (size_t k = 0; k < std::min(stream.lines().size(), c.lines.size()); ++k) {
            const loom::audio::Line& got = stream.lines()[k];
            LOOM_CHECK(got.start == c.lines[k].start && got.samples == c.lines[k].samples && got.complete);
        }
        alone[c.dir] = stream.text();
        audio_s += static_cast<double>(wav.size()) / kRate;
        stream_s += seconds;
        std::printf("%-26s %6.2f s  push %5zu  %2zu lines  %3zu calls  prob max|d| %.2e  %zu/%zu hops' side same  "
                    "%zu/%zu line texts same  %.2fx real time\n",
                    std::filesystem::path(c.dir).filename().string().c_str(), static_cast<double>(wav.size()) / kRate,
                    c.push, stream.lines().size(), call, prob_worst, sides_same, smoothed.size(), text_same,
                    line_checks, static_cast<double>(wav.size()) / kRate / seconds);
        for (const auto& line : stream.lines())
            std::printf("  [%6.2f +%5.2f] %s\n", static_cast<double>(line.start) / kRate,
                        static_cast<double>(line.samples) / kRate, line.text.c_str());
    }
    std::printf("streamed %.1f s of audio in %.1f s: %.2fx real time, RTF %.3f\n", audio_s, stream_s,
                audio_s / stream_s, stream_s / audio_s);

    // 4. Two streams and a one-pass call between them: the first case and the last.
    {
        const Case& a_case = cases.front();
        const Case& b_case = cases.back();
        const std::vector<float> a_wav = read_f32(std::string(clips_dir) + "/" + a_case.clip + ".npy");
        const std::vector<float> b_wav = read_f32(std::string(clips_dir) + "/" + b_case.clip + ".npy");
        loom::audio::LineStream a(asr_bridge, *asr, vad_bridge, *vad, options);
        loom::audio::LineStream b(asr_bridge, *asr, vad_bridge, *vad, options);
        size_t ai = 0, bi = 0, turn = 0;
        while (ai < a_wav.size() || bi < b_wav.size()) {
            if (ai < a_wav.size()) {
                a.push(a_wav.data() + ai, std::min(a_case.push, a_wav.size() - ai));
                ai += a_case.push;
            }
            if (++turn == 7) loom::audio::transcribe(asr_bridge, *asr, a_wav, {});
            if (bi < b_wav.size()) {
                b.push(b_wav.data() + bi, std::min(b_case.push, b_wav.size() - bi));
                bi += b_case.push;
            }
        }
        a.finish();
        b.finish();
        LOOM_CHECK(a.text() == alone[a_case.dir]);
        LOOM_CHECK(b.text() == alone[b_case.dir]);
    }

    // A VAD without a live door, or a recogniser at another rate, is refused at construction (by FrameStream
    // and by the rate check); an option out of range too.
    {
        loom::audio::LineOptions bad = options;
        bad.window_seconds = 0.0f;
        LOOM_CHECK_THROWS(loom::audio::LineStream(asr_bridge, *asr, vad_bridge, *vad, bad), loom::Error);
        bad = options;
        bad.hop = 256;
        LOOM_CHECK_THROWS(loom::audio::LineStream(asr_bridge, *asr, vad_bridge, *vad, bad), loom::Error);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
