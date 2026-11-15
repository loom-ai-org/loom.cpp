// pyannote segmentation streamed as diart's voice activity detection streams it (ADR-129): `FrameStream` over
// the driver's port of diart's `VoiceActivityDetection`, graded against diart's own loop
// (`scripts/diart_vad_reference.py`, which runs diart's code with its RxPY plumbing replaced by a loop).
//
// Per reference case `<clip>.<duration>-<step>-<latency>/` (the stream opened with those settings):
//   1. The clip pushed in a ragged cycle of pieces: each pipeline call's rows are diart's aggregated speech
//      scores, row for row (the call's segmentation is the one-pass model's argmax, 0 or 1, so a score is a
//      ratio of Hamming weights: any frame whose class differs from diart's moves it far past kTolerance);
//      each row's start is diart's, the timestamp shift for audio padded in front included; and the speech
//      turns diart's `Binarize` (tau 0.6) and `PredictionAccumulator` (0.05 s patch) make of those rows are
//      diart's turns.
//   2. The same clip pushed 20 ms at a time on the default case: the same rows bit for bit -- where the
//      pieces break never moves a call.
//   3. `reset` answers the same rows again; two streams with different settings pushed in alternation, a
//      one-pass `classify` between their pushes, each answer what they answered alone.
//   4. The door's own contract: the rows are `loom.stream.labels` ({"speech"}), not `infer`'s seven classes;
//      a latency outside [step, duration] is refused at open.
//   5. With LOOM_DIART_VAD_STREAM_OLD_GGUF (the published file): its one pass is this file's, bit for bit
//      (`infer` did not move), and the door refuses it.
//
// LOOM_DIART_VAD_STREAM_GGUF: pyannote-segmentation-3.0 exported with the stream functions.
// LOOM_DIART_VAD_STREAM_CLIPS_DIR: `<clip>.npy`, mono f32 at 16 kHz.
// LOOM_DIART_VAD_STREAM_REF_DIR: the reference script's output.

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;
// Both sides compute a score as the same float64 sums over the same zeros and ones, and the door hands
// rows back as float32 (`FrameUpdate::values`), so a row is diart's score rounded to float32 -- compared
// as such; the slack is numpy's cosine in `np.hamming`, which may differ from libm's by an ulp. A frame
// classified differently moves a score by at least ~1e-3.
constexpr double kTolerance = 1e-9;
constexpr double kTau = 0.6;
constexpr double kCollar = 0.05;

struct Step {
    double region_start = 0.0, region_end = 0.0;
    uint32_t rows = 0;
};

struct Case {
    std::string name, clip;
    double duration = 0.0, step = 0.0, latency = 0.0, shift = 0.0;
    std::vector<Step> steps;
    std::vector<double> scores;
    std::vector<std::pair<double, double>> turns;
};

Case read_case(const std::filesystem::path& dir) {
    Case c;
    c.name = dir.filename().string();
    const size_t dot = c.name.find('.');
    c.clip = c.name.substr(0, dot);
    std::sscanf(c.name.c_str() + dot + 1, "%lf-%lf-%lf", &c.duration, &c.step, &c.latency);
    std::ifstream meta(dir / "meta.txt");
    for (std::string key; meta >> key;) {
        if (key == "shift") meta >> c.shift;
        else meta.ignore(1 << 20, '\n');
    }
    std::ifstream steps(dir / "steps.txt");
    for (std::string line; std::getline(steps, line);) {
        std::istringstream in(line);
        std::string word;
        int i = 0;
        double chunk_start = 0.0;
        Step s;
        in >> word >> i >> chunk_start >> s.region_start >> s.region_end >> s.rows;
        c.steps.push_back(s);
    }
    std::ifstream scores(dir / "scores.txt");
    for (std::string line; std::getline(scores, line);) {
        c.scores.push_back(static_cast<float>(std::strtod(line.c_str(), nullptr)));
    }
    std::ifstream turns(dir / "segments.txt");
    for (double a, b; turns >> a >> b;) c.turns.emplace_back(a, b);
    return c;
}

struct Answer {
    std::vector<double> rows, start, duration;
};

loom::audio::FrameStream::Inputs settings(const Case& c) {
    return {{"duration_seconds", c.duration}, {"step_seconds", c.step}, {"latency_seconds", c.latency}};
}

void absorb(Answer& a, const loom::audio::FrameUpdate& u) {
    LOOM_CHECK(u.first_row == a.rows.size());
    LOOM_CHECK(u.row_start.size() == u.n_rows && u.row_duration.size() == u.n_rows);
    a.rows.insert(a.rows.end(), u.values.begin(), u.values.end());
    a.start.insert(a.start.end(), u.row_start.begin(), u.row_start.end());
    a.duration.insert(a.duration.end(), u.row_duration.begin(), u.row_duration.end());
}

Answer pushed(loom::audio::FrameStream& s, const std::vector<float>& wav, const std::vector<size_t>& pieces) {
    Answer a;
    size_t at = 0, i = 0;
    while (at < wav.size()) {
        const size_t n = std::min(pieces[i++ % pieces.size()], wav.size() - at);
        absorb(a, s.push(wav.data() + at, n));
        at += n;
    }
    absorb(a, s.finish());
    return a;
}

// diart's `Binarize` on each call's rows, the turns of every call joined as `PredictionAccumulator` joins
// them (Annotation.update, then support(0.05) and support()): gaps of at most 1e-6 s (pyannote's segment
// precision) or under the collar are filled.
std::vector<std::pair<double, double>> turns(const Answer& a, const std::vector<Step>& steps) {
    std::vector<std::pair<double, double>> all;
    size_t row = 0;
    for (const Step& s : steps) {
        bool active = false;
        double start = 0.0;
        for (uint32_t t = 0; t <= s.rows; ++t) {
            const bool next = t < s.rows && a.rows[row + t] > kTau;
            const size_t r = row + std::min<uint32_t>(t, s.rows - 1);
            // The middle of frame t; past the last row, the frame after it on the same grid.
            const double frame_start = t < s.rows ? a.start[r] : a.start[r] + a.duration[r];
            const double middle = 0.5 * (frame_start + (frame_start + a.duration[r]));
            if (t == 0) start = middle;
            if (!active && next) start = middle;
            if (active && !next) all.emplace_back(start, middle);
            active = next;
        }
        row += s.rows;
    }
    std::sort(all.begin(), all.end());
    std::vector<std::pair<double, double>> merged;
    for (const auto& seg : all) {
        if (!merged.empty()) {
            const double gap = seg.first - merged.back().second;
            if (gap <= 1e-6 || gap < kCollar) {
                merged.back().second = std::max(merged.back().second, seg.second);
                continue;
            }
        }
        merged.push_back(seg);
    }
    return merged;
}

double max_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double d = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

// Each reference row's start: its call's region on its own grid, moved by the shift.
std::vector<double> reference_starts(const Case& c) {
    std::vector<double> out;
    for (const Step& s : c.steps) {
        const double res = (s.region_end - s.region_start) / s.rows;
        for (uint32_t k = 0; k < s.rows; ++k) out.push_back(s.region_start + k * res + c.shift);
    }
    return out;
}

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_DIART_VAD_STREAM_GGUF");
    const char* clips = loom_test::fixture_env("LOOM_DIART_VAD_STREAM_CLIPS_DIR");
    const char* refs = loom_test::fixture_env("LOOM_DIART_VAD_STREAM_REF_DIR");
    if (!gguf || !clips || !refs) {
        std::fprintf(stderr, "SKIP: set LOOM_DIART_VAD_STREAM_GGUF, _CLIPS_DIR and _REF_DIR\n");
        return kSkipReturnCode;
    }
    ggml_backend_ptr backend(loom_test::cpu_backend());
    auto model = loom::GgufModel::load(gguf, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, backend.get());
    loom::LoomLuaBridge& bridge = session.bridge();

    std::vector<std::filesystem::path> dirs;
    for (const auto& e : std::filesystem::directory_iterator(refs)) {
        if (e.is_directory()) dirs.push_back(e.path());
    }
    std::sort(dirs.begin(), dirs.end());
    LOOM_CHECK(!dirs.empty());

    // 4. The rows are speech scores.
    {
        loom::audio::FrameStream s(bridge, *model);
        LOOM_CHECK(s.labels() == std::vector<std::string>{"speech"});
        LOOM_CHECK(s.width() == 1);
        bool refused = false;
        try {
            loom::audio::FrameStream bad(bridge, *model, {{"latency_seconds", 6.0}});
        } catch (const std::exception& e) {
            refused = std::string(e.what()).find("latency") != std::string::npos;
        }
        LOOM_CHECK(refused);
    }

    Answer default_jfk;
    std::vector<float> jfk;
    for (const auto& dir : dirs) {
        const Case c = read_case(dir);
        std::vector<int64_t> shape;
        const std::vector<float> wav = loom_test::read_npy_f32(std::string(clips) + "/" + c.clip + ".npy", shape);
        LOOM_CHECK(!wav.empty());

        // 1. A ragged cycle of pieces against diart.
        loom::audio::FrameStream s(bridge, *model, settings(c));
        const Answer got = pushed(s, wav, {5923, 1, 16000, 800, 7999, 0, 31});
        LOOM_CHECK(got.rows.size() == c.scores.size());
        if (got.rows.size() != c.scores.size()) {
            // The rows cannot be lined up with diart's calls, and the turns read rows by call.
            std::printf("%-26s %zu rows, diart %zu\n", c.name.c_str(), got.rows.size(), c.scores.size());
            continue;
        }
        const double d = max_diff(got.rows, c.scores);
        size_t far = 0;
        for (size_t i = 0; i < std::min(got.rows.size(), c.scores.size()); ++i) far += std::fabs(got.rows[i] - c.scores[i]) > kTolerance;
        const double dt = max_diff(got.start, reference_starts(c));
        const auto mine = turns(got, c.steps);
        double dturn = mine.size() == c.turns.size() ? 0.0 : INFINITY;
        for (size_t i = 0; i < std::min(mine.size(), c.turns.size()); ++i) {
            dturn = std::max({dturn, std::fabs(mine[i].first - c.turns[i].first), std::fabs(mine[i].second - c.turns[i].second)});
        }
        std::printf("%-26s %3zu calls %5zu rows: max|d| %.3g (%zu rows off), starts %.3g, %zu turns (diart %zu) %.3g\n",
                    c.name.c_str(), c.steps.size(), got.rows.size(), d, far, dt, mine.size(), c.turns.size(), dturn);
        LOOM_CHECK(d <= kTolerance);
        LOOM_CHECK(dt <= 1e-9);
        LOOM_CHECK(dturn <= 1e-9);

        // 2. 20 ms pushes on the default case.
        if (c.name == "jfk.5-0.5-0.5") {
            s.reset();
            const Answer small = pushed(s, wav, {320});
            LOOM_CHECK(small.rows == got.rows && small.start == got.start);
            default_jfk = got;
            jfk = wav;
        }
    }

    // 3. reset, and two streams in alternation with a one-pass call between their pushes.
    if (!jfk.empty()) {
        Case def;
        def.duration = 5.0, def.step = 0.5, def.latency = 0.5;
        Case other = def;
        other.latency = 2.0;
        loom::audio::FrameStream a(bridge, *model, settings(def)), b(bridge, *model, settings(other));
        const Answer b_alone = pushed(b, jfk, {4000});
        b.reset();
        Answer got_a, got_b;
        for (size_t at = 0; at < jfk.size(); at += 4000) {
            const size_t n = std::min<size_t>(4000, jfk.size() - at);
            absorb(got_a, a.push(jfk.data() + at, n));
            if (at == 4000 * 7) loom::audio::classify(bridge, *model, std::vector<float>(jfk.begin(), jfk.begin() + 32000));
            absorb(got_b, b.push(jfk.data() + at, n));
        }
        absorb(got_a, a.finish());
        absorb(got_b, b.finish());
        LOOM_CHECK(got_a.rows == default_jfk.rows);
        LOOM_CHECK(got_b.rows == b_alone.rows && got_b.start == b_alone.start);
        std::printf("interleaved: %zu + %zu rows as alone\n", got_a.rows.size(), got_b.rows.size());
    }

    // 5. The published file: the same one pass, and no stream.
    if (const char* old_path = loom_test::fixture_env("LOOM_DIART_VAD_STREAM_OLD_GGUF")) {
        auto old = loom::GgufModel::load(old_path, backend.get());
        LOOM_CHECK(old != nullptr);
        loom::Session old_session(*old, backend.get());
        const std::vector<float> clip(jfk.begin(), jfk.end());
        LOOM_CHECK(loom::audio::classify(old_session.bridge(), *old, clip).probabilities ==
                   loom::audio::classify(bridge, *model, clip).probabilities);
        bool refused = false;
        try {
            loom::audio::FrameStream s(old_session.bridge(), *old);
        } catch (const loom::LoadError&) {
            refused = true;
        }
        LOOM_CHECK(refused);
        std::printf("published file: one pass identical, stream refused\n");
    }
    LOOM_TEST_REPORT_AND_RETURN();
}
