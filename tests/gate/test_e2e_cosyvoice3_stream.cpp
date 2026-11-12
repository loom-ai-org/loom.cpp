// The CosyVoice3 live stream (ADR-112) against upstream's own streaming mode.
//
// Upstream's `tts(stream=True)` re-runs the flow over every token so far in the DiT's STREAMING mode (a
// block-causal chunk mask) and CausalHiFT over every frame so far, handing out what it had not. At f64
// that loop IS one pass in streaming mode (`flow.inference(streaming=True, finalize=True)` + HiFT), mel for
// mel; at f32 the ten guided Euler steps amplify the re-runs' different rounding to ~5e-3 on the mel, and
// CausalHiFT's sine phase (a cumsum of F0) turns that into O(1) sample differences. So the stream is held
// to its MEL, and its vocoder to the vocoder's own one call over that mel:
//
//   1. One pass in streaming mode (`infer` with `streaming = 1`, `return_mel`), every draw pinned to the
//      reference's: the mel against upstream's streaming one pass.
//   2. The stream's mel (`return_mel`), hop by hop: the same frame count, against the one pass and against
//      upstream's own stream loop, within the f32 re-run spread both show.
//   3. The stream's waveform: as many samples as the one pass, in more than one piece, and -- the vocoder's
//      streaming, exactly -- the one-call vocoder over the stream's own mel, run alone (a phase probe).
//   4. `reset`, and two streams interleaved with a one-pass call between, under both cache modes: each
//      bit-identical to itself alone.
//   5. Not `infer`: the full-attention one pass's mel is far from the stream's (it is another mode).
//
// LOOM_COSYVOICE3_STREAM_GGUF       a CosyVoice3 exported with the stream functions
// LOOM_COSYVOICE3_STREAM_REF_DIR    text_ids, draws, noise, nsf_noise, mel_streaming, mel_stream_loop,
//                                   mel_full .npy (`scripts/cosyvoice3_stream_reference.py`)
// LOOM_COSYVOICE3_STREAM_OLD_GGUF   (optional) an export from before, which the door must refuse

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using Inputs = std::unordered_map<std::string, loom::LoomLuaBridge::Value>;

// Measured on the dev box (2026-11-12): upstream's own f32 stream loop is 4.7e-03 from its f32 one pass,
// whose f64 counterpart it equals exactly. The engine's stream and its one pass re-run in the same way.
constexpr double kMelReRunTolerance = 2e-2;
// The engine's one pass against the reference's, both f32, both streaming mode: the export's gate holds
// the full-attention mel to its f32/f64 floor; this is the same arithmetic under another mask.
constexpr double kMelReferenceTolerance = 2e-2;
// The vocoder streamed (re-run over a growing mel, the settled samples kept) against one call over the
// whole: upstream's measured 1.07e-05.
constexpr double kVocoderTolerance = 1e-4;
constexpr int kMel = 80;

std::vector<double> as_doubles(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }

double max_abs_diff(const std::vector<double>& a, const std::vector<float>& b) {
    double d = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) d = std::max(d, std::abs(a[i] - double(b[i])));
    return d;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) d = std::max(d, std::abs(double(a[i]) - b[i]));
    return d;
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
    const char* gguf = loom_test::fixture_env("LOOM_COSYVOICE3_STREAM_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_COSYVOICE3_STREAM_REF_DIR");
    if (gguf == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_COSYVOICE3_STREAM_GGUF and LOOM_COSYVOICE3_STREAM_REF_DIR\n");
        return 77;
    }
    const std::string dir = ref_env;
    std::vector<int64_t> shape;
    const auto text_ids = loom_test::read_npy_f32(dir + "/text_ids.npy", shape);
    const auto draws = loom_test::read_npy_f32(dir + "/draws.npy", shape);
    const auto noise = loom_test::read_npy_f32(dir + "/noise.npy", shape);
    const auto nsf = loom_test::read_npy_f32(dir + "/nsf_noise.npy", shape);
    const auto ref_mel = loom_test::read_npy_f32(dir + "/mel_streaming.npy", shape);       // (frames, 80)
    const auto ref_loop_mel = loom_test::read_npy_f32(dir + "/mel_stream_loop.npy", shape);
    const auto ref_full_mel = loom_test::read_npy_f32(dir + "/mel_full.npy", shape);
    LOOM_CHECK(!text_ids.empty() && !draws.empty() && !ref_mel.empty());

    loom::Device device = loom::Device::open("cpu");
    auto model = loom::GgufModel::load(gguf, device.backends().primary);
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, device.backends());
    loom::LoomLuaBridge& bridge = session.bridge();
    LOOM_CHECK(loom::audio::SpeechStream::supported(bridge));

    // Every draw the reference made, pinned: the LM's uniforms, the flow's noise table, the sine table.
    const Inputs pinned{{"tokens", as_doubles(text_ids)}, {"draws", as_doubles(draws)},
                        {"noise", as_doubles(noise)}, {"nsf_noise", as_doubles(nsf)}};
    auto with = [&](Inputs in, std::initializer_list<std::pair<const char*, double>> extra) {
        for (const auto& [k, v] : extra) in[k] = v;
        return in;
    };

    using clock = std::chrono::steady_clock;
    // --- 1. One pass, streaming mode. ---
    const auto one_mel = std::get<std::vector<double>>(bridge.call("infer", with(pinned, {{"streaming", 1.0},
                                                                                          {"return_mel", 1.0}})));
    auto t0 = clock::now();
    const auto one_wave = std::get<std::vector<double>>(bridge.call("infer", with(pinned, {{"streaming", 1.0}})));
    const double one_pass_s = std::chrono::duration<double>(clock::now() - t0).count();
    const double d_ref = max_abs_diff(one_mel, ref_mel);
    std::fprintf(stderr, "one pass (streaming mode): %zu mel frames (reference %zu), max|d| %.3g; %zu samples in "
                         "%.1f s\n", one_mel.size() / kMel, ref_mel.size() / kMel, d_ref, one_wave.size(), one_pass_s);
    LOOM_CHECK(one_mel.size() == ref_mel.size());
    LOOM_CHECK(d_ref <= kMelReferenceTolerance);

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
    std::vector<float> one_mel_f(one_mel.begin(), one_mel.end());
    const double d_one = max_abs_diff(stream_mel, one_mel_f);
    const double d_loop = max_abs_diff(stream_mel, ref_loop_mel);
    std::fprintf(stderr, "stream mel: %zu hops, %zu frames; vs one pass %.3g, vs upstream's stream loop %.3g "
                         "(upstream's loop vs its one pass %.3g)\n", mel_pieces, stream_mel.size() / kMel, d_one,
                 d_loop, max_abs_diff(ref_loop_mel, ref_mel));
    LOOM_CHECK(stream_mel.size() == one_mel.size());
    LOOM_CHECK(mel_pieces >= 2);
    LOOM_CHECK(d_one <= kMelReRunTolerance);
    LOOM_CHECK(d_loop <= kMelReRunTolerance);

    // --- 3. The stream's waveform, and the vocoder's streaming. ---
    Pulled alone;
    {
        loom::audio::SpeechStream s(bridge, pinned);
        t0 = clock::now();
        LOOM_CHECK(!s.next().empty());
        const double first_s = std::chrono::duration<double>(clock::now() - t0).count();
        s.reset();
        alone = pull_all(s);
        std::fprintf(stderr, "stream wave: %zu pieces (first %zu samples), %zu samples (one pass %zu); first piece "
                             "after %.1f s, one pass took %.1f s\n", alone.pieces, alone.first_piece,
                     alone.samples.size(), one_wave.size(), first_s, one_pass_s);
        LOOM_CHECK(alone.samples.size() == one_wave.size());
        LOOM_CHECK(alone.pieces >= 2 && alone.first_piece < alone.samples.size());
        // 4a. Reset gives the same pieces again.
        s.reset();
        LOOM_CHECK(pull_all(s).samples == alone.samples);
    }
    {
        // The vocoder alone, once, over the stream's own mel (the hops' mel is the same whether or not the
        // stream vocodes it: the same calls in the same order) with the same sine table.
        const size_t m = stream_mel.size() / kMel, n = m * 480, total = nsf.size() / 9;
        std::vector<double> nsf_m(9 * n);
        for (size_t h = 0; h < 9; ++h) {
            for (size_t i = 0; i < n; ++i) nsf_m[h * n + i] = nsf[h * total + i];
        }
        bridge.load_script("function probe_vocoder(inputs)\n"
                           "  return loom.run_subgraph('vocoder', {n_enc_frames = #inputs.mel / 80, n_past = 0},\n"
                           "      {mel = inputs.mel, nsf_noise = inputs.nsf_noise})\n"
                           "end\n");
        const auto whole = std::get<std::vector<double>>(bridge.call("probe_vocoder",
            {{"mel", as_doubles(stream_mel)}, {"nsf_noise", nsf_m}}));
        const double d_voc = max_abs_diff(whole, alone.samples);
        std::fprintf(stderr, "vocoder streamed vs one call over the same mel: %.3g (%zu / %zu samples)\n", d_voc,
                     alone.samples.size(), whole.size());
        LOOM_CHECK(whole.size() == alone.samples.size());
        LOOM_CHECK(d_voc <= kVocoderTolerance);
    }

    // --- 4b. Interleaved, under both cache modes. ---
    const Inputs other{{"tokens", as_doubles(text_ids)}, {"seed", 99.0}};
    std::vector<float> other_alone;
    {
        loom::audio::SpeechStream s(bridge, other);
        other_alone = pull_all(s).samples;
    }
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
            if (turn++ == 0) {
                one_pass_ok = std::get<std::vector<double>>(
                                  bridge.call("infer", with(pinned, {{"streaming", 1.0}}))) == one_wave;
            }
            const auto pb = sb.next();
            got_b.insert(got_b.end(), pb.begin(), pb.end());
        }
        std::fprintf(stderr, "%s: a %s, b %s, one pass between %s\n",
                     mode == loom::StreamCaches::Park ? "park" : "per-stream",
                     got_a == alone.samples ? "ok" : "DIFFERS", got_b == other_alone ? "ok" : "DIFFERS",
                     one_pass_ok ? "ok" : "DIFFERS");
        LOOM_CHECK(got_a == alone.samples);
        LOOM_CHECK(got_b == other_alone);
        LOOM_CHECK(one_pass_ok);
    }

    // --- 5. Not `infer`'s mode. ---
    {
        const auto full = std::get<std::vector<double>>(bridge.call("infer", with(pinned, {{"return_mel", 1.0}})));
        const double d_full = max_abs_diff(full, ref_full_mel);
        std::fprintf(stderr, "full-attention one pass vs reference %.3g; vs the stream %.3g\n", d_full,
                     max_abs_diff(full, stream_mel.empty() ? std::vector<float>{} : stream_mel));
        LOOM_CHECK(d_full <= kMelReferenceTolerance);
        LOOM_CHECK(max_abs_diff(full, stream_mel) > 1.0);
    }

    if (const char* old = loom_test::fixture_env("LOOM_COSYVOICE3_STREAM_OLD_GGUF")) {
        auto old_model = loom::GgufModel::load(old, device.backends().primary);
        loom::Session old_session(*old_model, device.backends());
        LOOM_CHECK(!loom::audio::SpeechStream::supported(old_session.bridge()));
        LOOM_CHECK_THROWS(loom::audio::SpeechStream(old_session.bridge(), pinned), loom::LoadError);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
