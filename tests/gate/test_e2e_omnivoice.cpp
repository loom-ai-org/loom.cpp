// OmniVoice (ADR-131) against its own `generate`, every position-noise draw pinned to the oracle's.
//
// OmniVoice is a masked-diffusion TTS: a bidirectional Qwen3 unmasks eight codebooks over 32 steps, then
// HiggsAudio-v2's decoder makes 24 kHz and the reference post-processes the waveform through pydub (an
// int16 round trip, silences trimmed, the level set, the edges faded and padded). Three cases, each a
// directory `scripts/omnivoice_reference.py` writes:
//
//   hello    the model's own voice, one chunk
//   clone    a reference clip and its transcript: the clip goes through `ref_encode` in the engine
//   chunks   a voice-design instruct, a language, a non-verbal tag, an abbreviation, and the chunk
//            threshold lowered so the text is cut into four chunks (chunk 0 prompts the other three)
//
// For each:
//   1. The plan (`return_prompt`): every chunk's prompt prefix and frame count, id for id.
//   2. The codes, through the STREAM (`return_codes`, one piece per chunk): every code of every chunk.
//      hello and chunks are pulled interleaved, so this is also the two-streams-at-once check.
//   3. The one pass's waveform against `generate`'s: same length, within kWaveTolerance.
//   4. The audio stream: one piece per chunk; with one chunk, the one pass's waveform bit for bit.
//   5. A seed alone (nothing pinned): every cell unmasked, every code a real one.
//
// LOOM_OMNIVOICE_GGUF      the export
// LOOM_OMNIVOICE_REF_DIR   the case directories (hello/, clone/, chunks/)

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

using Inputs = std::unordered_map<std::string, loom::LoomLuaBridge::Value>;

// The codes are exact; the waveform is the reference's int16 grid, and the engine's f32 codec decode
// lands a sample on the neighbouring step now and then (1/32768 = 3.05e-5, scaled by the level). Two
// steps, measured at one (hello 1.8e-5 over 401 samples, chunks 1.5e-5, clone 3.05e-5).
constexpr double kWaveTolerance = 6.2e-5;
constexpr int kCodebooks = 8;
constexpr int kMaskId = 1024;

bool exists(const std::string& path) { return std::ifstream(path).good(); }

std::vector<double> npy(const std::string& path) {
    std::vector<int64_t> shape;
    const auto v = loom_test::read_npy_f32(path, shape);
    return std::vector<double>(v.begin(), v.end());
}

std::vector<double> npy_shape(const std::string& path, std::vector<int64_t>& shape) {
    const auto v = loom_test::read_npy_f32(path, shape);
    return std::vector<double>(v.begin(), v.end());
}

std::vector<double> wave_of(const loom::LoomLuaBridge::Value& v) { return std::get<std::vector<double>>(v); }

struct Case {
    std::string name;
    Inputs inputs;
    std::vector<std::vector<double>> prompts;   // each [8, L_i] row-major
    std::vector<std::vector<double>> codes;     // each [8, T_i] row-major
    std::vector<int64_t> frames;
    std::vector<double> audio;
};

Case load_case(const std::string& dir, const std::string& name) {
    Case c;
    c.name = name;
    const std::string d = dir + "/" + name;
    c.inputs["tokens"] = npy(d + "/tokens.npy");
    c.inputs["draws"] = npy(d + "/draws.npy");
    for (const char* opt : {"instruct", "ref_text", "waveform"}) {
        if (exists(d + "/" + opt + ".npy")) c.inputs[opt] = npy(d + "/" + opt + ".npy");
    }
    if (exists(d + "/language.npy")) c.inputs["language"] = npy(d + "/language.npy").at(0);
    // The chunk options the oracle lowered, spelled in the case's own json; read them by name.
    std::ifstream json(d + "/case.json");
    const std::string text((std::istreambuf_iterator<char>(json)), std::istreambuf_iterator<char>());
    for (const char* key : {"chunk_threshold", "chunk_duration"}) {
        const size_t at = text.find(std::string("\"") + key + "\": ");
        if (at == std::string::npos) continue;
        const std::string rest = text.substr(at + std::string(key).size() + 4);
        if (rest.rfind("null", 0) == 0) continue;
        c.inputs[key] = std::stod(rest);
    }
    for (int i = 0; exists(d + "/tokens_" + std::to_string(i) + ".npy"); ++i) {
        std::vector<int64_t> shape;
        c.codes.push_back(npy_shape(d + "/tokens_" + std::to_string(i) + ".npy", shape));
        c.frames.push_back(shape.at(1));
        c.prompts.push_back(npy(d + "/input_ids_" + std::to_string(i) + ".npy"));
    }
    c.audio = npy(d + "/audio.npy");
    return c;
}

Inputs with(Inputs in, const char* key, double value) {
    in[key] = value;
    return in;
}

void check_plan(loom::LoomLuaBridge& bridge, const Case& c) {
    const auto plan = wave_of(bridge.call("infer", with(c.inputs, "return_prompt", 1.0)));
    const size_t n = static_cast<size_t>(plan.at(0));
    LOOM_CHECK(n == c.frames.size());
    for (size_t i = 0; i < n; ++i) LOOM_CHECK(static_cast<int64_t>(plan.at(1 + i)) == c.frames[i]);
    const size_t base = 1 + n;
    for (size_t i = 0; i < n; ++i) {
        const auto lo = static_cast<size_t>(plan.at(base + i)), hi = static_cast<size_t>(plan.at(base + i + 1));
        const size_t width = c.prompts[i].size() / kCodebooks;
        // The prompt's row 0: the prefix (style + text) first, then the reference codes and the target,
        // which the plan does not hold. Every prefix id, and the prefix's end where the audio starts.
        for (size_t k = lo; k < hi; ++k) {
            LOOM_CHECK(static_cast<int64_t>(plan.at(base + n + 1 + k)) == static_cast<int64_t>(c.prompts[i][k - lo]));
        }
        LOOM_CHECK(hi - lo + static_cast<size_t>(c.frames[i]) <= width);
    }
    std::fprintf(stderr, "  %s: plan of %zu chunk(s) == the reference's prompts\n", c.name.c_str(), n);
}

void check_codes(const Case& c, const std::vector<std::vector<double>>& pieces) {
    LOOM_CHECK(pieces.size() == c.codes.size());
    size_t total = 0;
    for (size_t i = 0; i < pieces.size(); ++i) {
        LOOM_CHECK(pieces[i].size() == c.codes[i].size());
        for (size_t k = 0; k < pieces[i].size(); ++k) LOOM_CHECK(pieces[i][k] == c.codes[i][k]);
        total += pieces[i].size();
    }
    std::fprintf(stderr, "  %s: %zu codes in %zu piece(s), every one the reference's\n", c.name.c_str(), total,
                 pieces.size());
}

std::vector<std::vector<double>> to_double_pieces(const std::vector<std::vector<float>>& in) {
    std::vector<std::vector<double>> out;
    for (const auto& p : in) out.emplace_back(p.begin(), p.end());
    return out;
}

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_OMNIVOICE_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_OMNIVOICE_REF_DIR");
    if (gguf == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_OMNIVOICE_GGUF and LOOM_OMNIVOICE_REF_DIR\n");
        return 77;
    }
    std::vector<Case> cases;
    for (const char* name : {"hello", "clone", "chunks"}) cases.push_back(load_case(ref_env, name));
    LOOM_CHECK(cases[2].codes.size() > 1);   // the chunked case must actually chunk

    loom::Device device = loom::Device::open(loom_test::test_device_name());
    std::fprintf(stderr, "device: %s\n", device.name().c_str());
    auto model = loom::GgufModel::load(gguf, device.backends().primary);
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, device.backends());
    loom::LoomLuaBridge& bridge = session.bridge();
    LOOM_CHECK(loom::audio::SpeechStream::supported(bridge));

    // 1. The plans.
    for (const Case& c : cases) check_plan(bridge, c);

    // 2. The codes through the stream; hello and chunks interleaved, a piece of each in turn.
    {
        loom::audio::SpeechStream a(bridge, with(cases[0].inputs, "return_codes", 1.0));
        loom::audio::SpeechStream b(bridge, with(cases[2].inputs, "return_codes", 1.0));
        std::vector<std::vector<float>> pa, pb;
        while (!a.finished() || !b.finished()) {
            if (!a.finished()) { auto p = a.next(); if (!p.empty()) pa.push_back(p); }
            if (!b.finished()) { auto p = b.next(); if (!p.empty()) pb.push_back(p); }
        }
        check_codes(cases[0], to_double_pieces(pa));
        check_codes(cases[2], to_double_pieces(pb));
        loom::audio::SpeechStream s(bridge, with(cases[1].inputs, "return_codes", 1.0));
        std::vector<std::vector<float>> pc;
        while (!s.finished()) { auto p = s.next(); if (!p.empty()) pc.push_back(p); }
        check_codes(cases[1], to_double_pieces(pc));
    }

    // 3. The one pass's waveform.
    std::vector<std::vector<double>> one_pass;
    for (const Case& c : cases) {
        const auto wave = wave_of(bridge.call("infer", c.inputs));
        LOOM_CHECK(wave.size() == c.audio.size());
        double worst = 0.0;
        size_t differing = 0;
        for (size_t i = 0; i < wave.size(); ++i) {
            const double d = std::abs(wave[i] - c.audio[i]);
            worst = std::max(worst, d);
            differing += d > 0.0;
        }
        std::fprintf(stderr, "  %s: %zu samples, max|d| %.3g (%zu differ)\n", c.name.c_str(), wave.size(), worst,
                     differing);
        LOOM_CHECK(worst <= kWaveTolerance);
        one_pass.push_back(wave);
    }

    // 4. The audio stream: a piece per chunk; one chunk is the one pass exactly.
    for (size_t k : {size_t(0), size_t(2)}) {
        loom::audio::SpeechStream s(bridge, cases[k].inputs);
        std::vector<float> joined;
        size_t pieces = 0, first = 0;
        while (!s.finished()) {
            const auto p = s.next();
            if (p.empty()) continue;
            if (pieces++ == 0) first = p.size();
            joined.insert(joined.end(), p.begin(), p.end());
        }
        LOOM_CHECK(pieces == cases[k].codes.size());
        if (pieces == 1) {
            LOOM_CHECK(joined.size() == one_pass[k].size());
            for (size_t i = 0; i < joined.size(); ++i) LOOM_CHECK(double(joined[i]) == double(float(one_pass[k][i])));
        } else {
            LOOM_CHECK(first < joined.size());
        }
        std::fprintf(stderr, "  %s: stream of %zu piece(s), first %zu of %zu samples\n", cases[k].name.c_str(),
                     pieces, first, joined.size());
    }

    // 5. A seed alone: the engine's own draws, every cell unmasked into a real code.
    {
        Inputs seeded = cases[0].inputs;
        seeded.erase("draws");
        seeded["seed"] = 7.0;
        const auto codes = wave_of(bridge.call("infer", with(seeded, "return_codes", 1.0)));
        LOOM_CHECK(codes.size() == cases[0].codes[0].size());
        for (double v : codes) LOOM_CHECK(v >= 0 && v < kMaskId);
        std::fprintf(stderr, "  seed alone: %zu codes, none masked\n", codes.size());
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
