// The Chatterbox live stream (ADR-117) against the same loop run in PyTorch.
//
// S3Gen keeps CosyVoice2's `finalize=False` hook but ships no loop around it, so the stream is CosyVoice2's:
// each hop re-runs the flow over every token so far (the last 3 as look-ahead context) and keeps the new
// mel, and HiFT re-runs over all of it, handing out what no later frame can change. Chatterbox's flow
// attends over the whole sequence, so a hop's mel is not the one pass's: the stream is held to the
// PyTorch loop, and its distance from the one pass to the spread of another noise draw.
//
//   1. One pass (`infer`, guided greedy, every draw pinned, `return_mel`): the mel against the reference's.
//   2. The stream's mel, hop by hop: as many frames as the one pass, in more than one hop, against the
//      PyTorch loop; and nearer the one pass than the one pass under another noise table is.
//   3. The stream's waveform: as many samples as the one pass, in more than one piece; `infer` with
//      `piece_tokens = -1` (the same cut in one call) bit for bit; and -- the vocoder's streaming, exactly
//      -- the one-call vocoder over the stream's own mel, run alone (a phase probe).
//   4. `reset`, and two streams interleaved with a one-pass call between, under both cache modes: each
//      bit-identical to itself alone.
//   5. A seed alone, nothing pinned (a shorter text): the stream draws the one pass's tokens, so its mel has
//      the one pass's frame count, and its waveform the one pass's length.
//
// LOOM_CHATTERBOX_STREAM_GGUF       a Chatterbox exported with the stream functions
// LOOM_CHATTERBOX_STREAM_REF_DIR    text_ids, noise, nsf_phase, nsf_noise, mel_one, mel_stream_loop,
//                                   mel_other_draw .npy (`scripts/chatterbox_stream_reference.py`)
// LOOM_CHATTERBOX_STREAM_OLD_GGUF   (optional) an export from before, which the door must refuse

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"
#include "cpu_backend.h"

#include "loom/loom.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using Inputs = std::unordered_map<std::string, loom::LoomLuaBridge::Value>;

// The engine's f32 flow against PyTorch's f32 flow, one pass or one hop: the export's one-pass gate holds
// the waveform to 2e-3, and the mel's own f32/f64 spread is ~1e-5 (`chatterbox_stream_reference.py --f64`).
constexpr double kMelTolerance = 2e-3;
// The vocoder streamed (re-run over a growing mel, the settled samples kept) against one call over the
// whole: ADR-112's bound for CosyVoice3's, whose sine source is the same cumsum.
constexpr double kVocoderTolerance = 1e-4;
constexpr int kMel = 80;

std::vector<double> as_doubles(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }

template <class A, class B>
double max_abs_diff(const A& a, const B& b) {
    double d = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) d = std::max(d, std::abs(double(a[i]) - double(b[i])));
    return d;
}

template <class A, class B>
double mean_abs_diff(const A& a, const B& b) {
    const size_t n = std::min(a.size(), b.size());
    double d = 0.0;
    for (size_t i = 0; i < n; ++i) d += std::abs(double(a[i]) - double(b[i]));
    return n ? d / double(n) : 0.0;
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

std::vector<double> wave_of(const loom::LoomLuaBridge::Value& v) { return std::get<std::vector<double>>(v); }

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_CHATTERBOX_STREAM_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_CHATTERBOX_STREAM_REF_DIR");
    if (gguf == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_CHATTERBOX_STREAM_GGUF and LOOM_CHATTERBOX_STREAM_REF_DIR\n");
        return 77;
    }
    const std::string dir = ref_env;
    std::vector<int64_t> shape;
    const auto text_ids = loom_test::read_npy_f32(dir + "/text_ids.npy", shape);
    const auto noise = loom_test::read_npy_f32(dir + "/noise.npy", shape);
    const auto nsf_phase = loom_test::read_npy_f32(dir + "/nsf_phase.npy", shape);
    const auto nsf = loom_test::read_npy_f32(dir + "/nsf_noise.npy", shape);
    const auto ref_mel = loom_test::read_npy_f32(dir + "/mel_one.npy", shape);              // (frames, 80)
    const auto ref_loop_mel = loom_test::read_npy_f32(dir + "/mel_stream_loop.npy", shape);
    const auto other_draw_mel = loom_test::read_npy_f32(dir + "/mel_other_draw.npy", shape);
    LOOM_CHECK(!text_ids.empty() && nsf_phase.size() == 9 && !ref_mel.empty());

    loom::Device device = loom::Device::open(loom_test::test_device_name());
    std::fprintf(stderr, "device: %s\n", device.name().c_str());
    auto model = loom::GgufModel::load(gguf, device.backends().primary);
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, device.backends());
    loom::LoomLuaBridge& bridge = session.bridge();
    LOOM_CHECK(loom::audio::SpeechStream::supported(bridge));

    // Guided greedy, as the reference decodes, and every S3Gen draw pinned to the oracle's.
    const Inputs pinned{{"tokens", as_doubles(text_ids)}, {"temperature", 0.0}, {"noise", as_doubles(noise)},
                        {"nsf_phase", as_doubles(nsf_phase)}, {"nsf_noise", as_doubles(nsf)}};
    auto with = [&](Inputs in, std::initializer_list<std::pair<const char*, double>> extra) {
        for (const auto& [k, v] : extra) in[k] = v;
        return in;
    };

    using clock = std::chrono::steady_clock;
    // --- 1. One pass. ---
    const auto one_mel = wave_of(bridge.call("infer", with(pinned, {{"return_mel", 1.0}})));
    auto t0 = clock::now();
    const auto one_wave = wave_of(bridge.call("infer", pinned));
    const double one_pass_s = std::chrono::duration<double>(clock::now() - t0).count();
    const double d_ref = max_abs_diff(one_mel, ref_mel);
    std::fprintf(stderr, "one pass: %zu mel frames (reference %zu), max|d| %.3g; %zu samples in %.1f s\n",
                 one_mel.size() / kMel, ref_mel.size() / kMel, d_ref, one_wave.size(), one_pass_s);
    LOOM_CHECK(one_mel.size() == ref_mel.size());
    LOOM_CHECK(d_ref <= kMelTolerance);

    // --- 2. The stream's mel. ---
    std::vector<float> stream_mel;
    size_t mel_pieces = 0;
    {
        loom::audio::SpeechStream s(bridge, with(pinned, {{"return_mel", 1.0}}));
        while (!s.finished()) {
            const auto piece = s.next();
            if (piece.empty()) continue;
            ++mel_pieces;
            stream_mel.insert(stream_mel.end(), piece.begin(), piece.end());
        }
    }
    const double d_loop = max_abs_diff(stream_mel, ref_loop_mel);
    const double mean_one = mean_abs_diff(stream_mel, one_mel), mean_other = mean_abs_diff(other_draw_mel, ref_mel);
    std::fprintf(stderr, "stream mel: %zu hops, %zu frames; vs the PyTorch loop %.3g; mean |d| from the one pass %.3g "
                         "(max %.3g), another noise draw's %.3g\n", mel_pieces, stream_mel.size() / kMel, d_loop,
                 mean_one, max_abs_diff(stream_mel, one_mel), mean_other);
    LOOM_CHECK(stream_mel.size() == one_mel.size());
    LOOM_CHECK(mel_pieces >= 2);
    LOOM_CHECK(d_loop <= kMelTolerance);
    LOOM_CHECK(mean_one < mean_other);

    // --- 3. The stream's waveform, and the vocoder's streaming. ---
    Pulled alone;
    {
        loom::audio::SpeechStream s(bridge, pinned);
        t0 = clock::now();
        LOOM_CHECK(!s.next().empty());
        const double first_s = std::chrono::duration<double>(clock::now() - t0).count();
        s.reset();
        t0 = clock::now();
        alone = pull_all(s);
        const double stream_s = std::chrono::duration<double>(clock::now() - t0).count();
        std::fprintf(stderr, "stream wave: %zu pieces (first %zu samples), %zu samples (one pass %zu); first piece "
                             "after %.1f s, whole stream %.1f s, one pass %.1f s\n", alone.pieces, alone.first_piece,
                     alone.samples.size(), one_wave.size(), first_s, stream_s, one_pass_s);
        LOOM_CHECK(alone.samples.size() == one_wave.size());
        LOOM_CHECK(alone.pieces >= 2 && alone.first_piece < alone.samples.size());
        // 4a. Reset gives the same pieces again.
        s.reset();
        LOOM_CHECK(pull_all(s).samples == alone.samples);
    }
    {
        const auto in_one_call = wave_of(bridge.call("infer", with(pinned, {{"piece_tokens", -1.0}})));
        const bool same = in_one_call.size() == alone.samples.size() &&
                          std::equal(in_one_call.begin(), in_one_call.end(), alone.samples.begin(),
                                     [](double a, float b) { return float(a) == b; });
        std::fprintf(stderr, "infer with piece_tokens = -1: %s\n", same ? "the stream, bit for bit" : "DIFFERS");
        LOOM_CHECK(same);
    }
    {
        // The vocoder alone, once, over the stream's own mel with the same sine draws.
        const size_t m = stream_mel.size() / kMel, n = m * 480, total = nsf.size() / 9;
        std::vector<double> nsf_m(9 * n);
        for (size_t h = 0; h < 9; ++h) {
            for (size_t i = 0; i < n; ++i) nsf_m[h * n + i] = nsf[h * total + i];
        }
        bridge.load_script("function probe_vocoder(inputs)\n"
                           "  return loom.run_subgraph('vocoder', {n_enc_frames = #inputs.mel / 80, n_past = 0},\n"
                           "      {mel = inputs.mel, nsf_phase = inputs.nsf_phase, nsf_noise = inputs.nsf_noise})\n"
                           "end\n");
        const auto whole = wave_of(bridge.call("probe_vocoder", {{"mel", as_doubles(stream_mel)},
                                                                 {"nsf_phase", as_doubles(nsf_phase)},
                                                                 {"nsf_noise", nsf_m}}));
        const double d_voc = max_abs_diff(whole, alone.samples);
        std::fprintf(stderr, "vocoder streamed vs one call over the same mel: %.3g (%zu / %zu samples)\n", d_voc,
                     alone.samples.size(), whole.size());
        LOOM_CHECK(whole.size() == alone.samples.size());
        LOOM_CHECK(d_voc <= kVocoderTolerance);
    }

    // --- 4b. Interleaved, under both cache modes. The other stream is a short text, sampled. ---
    auto vocab = loom::ChatterboxVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    const auto short_ids = vocab->encode("Good morning, how are you today?");
    const Inputs other{{"tokens", std::vector<double>(short_ids.begin(), short_ids.end())}, {"seed", 99.0}};
    std::vector<float> other_alone;
    {
        loom::audio::SpeechStream s(bridge, other);
        other_alone = pull_all(s).samples;
    }
    const auto other_one = wave_of(bridge.call("infer", other));
    for (loom::StreamCaches mode : {loom::StreamCaches::Park, loom::StreamCaches::PerStream}) {
        bridge.set_stream_caches(mode);
        loom::audio::SpeechStream sa(bridge, pinned);
        loom::audio::SpeechStream sb(bridge, other);
        std::vector<float> got_a, got_b;
        bool one_pass_ok = true;
        size_t turn = 0;
        while (!sa.finished() || !sb.finished()) {
            const auto pa = sa.next();
            got_a.insert(got_a.end(), pa.begin(), pa.end());
            if (turn++ == 0) one_pass_ok = wave_of(bridge.call("infer", other)) == other_one;
            const auto pb = sb.next();
            got_b.insert(got_b.end(), pb.begin(), pb.end());
        }
        std::fprintf(stderr, "%s: a %s, b %s, one pass between %s\n",
                     mode == loom::StreamCaches::Park ? "park" : "per-stream", got_a == alone.samples ? "ok" : "DIFFERS",
                     got_b == other_alone ? "ok" : "DIFFERS", one_pass_ok ? "ok" : "DIFFERS");
        LOOM_CHECK(got_a == alone.samples);
        LOOM_CHECK(got_b == other_alone);
        LOOM_CHECK(one_pass_ok);
    }

    // --- 5. A seed alone, nothing pinned: the same tokens both ways. ---
    {
        const auto mid_ids = vocab->encode("The river ran quietly past the old mill, under a sky the colour of slate.");
        const Inputs seeded{{"tokens", std::vector<double>(mid_ids.begin(), mid_ids.end())}, {"seed", 7.0}};
        const auto one = wave_of(bridge.call("infer", with(seeded, {{"return_mel", 1.0}})));
        loom::audio::SpeechStream s(bridge, with(seeded, {{"return_mel", 1.0}}));
        size_t hops = 0;
        std::vector<float> streamed;
        while (!s.finished()) {
            const auto piece = s.next();
            if (piece.empty()) continue;
            ++hops;
            streamed.insert(streamed.end(), piece.begin(), piece.end());
        }
        std::fprintf(stderr, "seed only: stream %zu mel frames in %zu hops, one pass %zu; mean |d| %.3g\n",
                     streamed.size() / kMel, hops, one.size() / kMel, mean_abs_diff(one, streamed));
        LOOM_CHECK(streamed.size() == one.size());
        LOOM_CHECK(hops >= 2);
        const auto one_wave_seeded = wave_of(bridge.call("infer", seeded));
        loom::audio::SpeechStream w(bridge, seeded);
        const auto wave = pull_all(w).samples;
        std::fprintf(stderr, "seed only: stream %zu samples, one pass %zu\n", wave.size(), one_wave_seeded.size());
        LOOM_CHECK(wave.size() == one_wave_seeded.size());
    }

    if (const char* old = loom_test::fixture_env("LOOM_CHATTERBOX_STREAM_OLD_GGUF")) {
        auto old_model = loom::GgufModel::load(old, device.backends().primary);
        loom::Session old_session(*old_model, device.backends());
        LOOM_CHECK(!loom::audio::SpeechStream::supported(old_session.bridge()));
        LOOM_CHECK_THROWS(loom::audio::SpeechStream(old_session.bridge(), pinned), loom::LoadError);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
