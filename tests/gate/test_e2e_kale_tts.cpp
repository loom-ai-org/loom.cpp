// kale-tts-1-tiny (loom-exporter kale_tts_export.py, ADR-142) against upstream's own code.
//
// The reference directory is what `scripts/kale_tts_reference.py` writes:
//
//   frontend.jsonl   texts through upstream's JavaScript front end (`KaleTTS.inputs`): the packed records, the
//                    sentences and the IPA. The file's tokenizer must give the same, id for id.
//   <case>/          one text through upstream's front end and modules (ModernBERT, K20Text, the teacher's
//                    predictor and text encoder, the release's clamp, the decoder through `HeadX`):
//     1. the one pass, its noise pinned: the reference's length (so every duration) and its samples within
//        kMinSnrDb;
//     2. the seed alone: the driver's `aten_randn` is ATen's own stream, so the pinned run's samples;
//     3. the stream: a piece per sentence, each the reference's sample count.
//     4. the text door: the tokenizer's records for the case's text are the reference's.
//
// LOOM_KALE_TTS_GGUF      the export
// LOOM_KALE_TTS_REF_DIR   frontend.jsonl and the case directories

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"
#include "loom/core/kale_tts_vocab.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using Inputs = std::unordered_map<std::string, loom::LoomLuaBridge::Value>;

// Measured 2026-11-22 on four cases (Kenzie, Amy, Magnus, Toby), engine against upstream's f32: 56 to 74 dB, as
// Paradee's head gives (its float32 STFTs, phase interpolation and harmonic sum, each rounded).
constexpr double kMinSnrDb = 50.0;

std::vector<double> npy(const std::string& path) {
    std::vector<int64_t> shape;
    const auto v = loom_test::read_npy_f32(path, shape);
    return std::vector<double>(v.begin(), v.end());
}

std::vector<double> wave_of(const loom::LoomLuaBridge::Value& v) { return std::get<std::vector<double>>(v); }

double snr_db(const std::vector<double>& want, const std::vector<double>& got) {
    double signal = 0.0, error = 0.0;
    for (size_t i = 0; i < std::min(want.size(), got.size()); ++i) {
        signal += want[i] * want[i];
        error += (got[i] - want[i]) * (got[i] - want[i]);
    }
    return 10.0 * std::log10(signal / std::max(error, 1e-30));
}

std::vector<double> as_doubles(const std::vector<int32_t>& ids) { return std::vector<double>(ids.begin(), ids.end()); }

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_KALE_TTS_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_KALE_TTS_REF_DIR");
    if (gguf == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_KALE_TTS_GGUF and LOOM_KALE_TTS_REF_DIR\n");
        return 77;
    }
    const std::string ref(ref_env);
    loom::Device device = loom::Device::open(loom_test::test_device_name());
    auto model = loom::GgufModel::load(gguf, device.backends().primary);
    LOOM_CHECK(model != nullptr);
    auto vocab = loom::KaleTtsVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);

    // The front end, id for id.
    {
        std::ifstream in(ref + "/frontend.jsonl");
        std::string line;
        size_t rows = 0, same = 0;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            const auto r = nlohmann::json::parse(line);
            const std::string text = r["text"], lang = r["lang"];
            const bool ok = vocab->encode(text, lang) == r["tokens"].get<std::vector<int32_t>>() &&
                            vocab->sentences(text, lang) == r["sentences"].get<std::vector<std::string>>();
            ++rows;
            if (ok) ++same;
            else if (rows - same <= 10) std::fprintf(stderr, "  front end differs (%s): %s\n", lang.c_str(), text.c_str());
        }
        LOOM_CHECK(rows > 0);
        LOOM_CHECK(same == rows);
        std::fprintf(stderr, "  front end: %zu/%zu texts identical to upstream's (records and sentences)\n", same, rows);
    }

    loom::Session session(*model, device.backends());
    loom::LoomLuaBridge& bridge = session.bridge();
    LOOM_CHECK(loom::audio::SpeechStream::supported(bridge));
    std::vector<std::string> dirs;
    for (const auto& e : std::filesystem::directory_iterator(ref)) {
        if (e.is_directory()) dirs.push_back(e.path().string());
    }
    std::sort(dirs.begin(), dirs.end());
    LOOM_CHECK(!dirs.empty());

    for (const std::string& d : dirs) {
        const auto c = nlohmann::json::parse(std::ifstream(d + "/case.json"));
        const std::string name = std::filesystem::path(d).filename().string();
        const auto tokens = npy(d + "/tokens.npy");
        const std::string lang = c["voice"] == "amy" || c["voice"] == "jack" ? "en-GB" : "";
        LOOM_CHECK(as_doubles(vocab->encode(c["text"].get<std::string>(), lang)) == tokens);
        Inputs in{{"tokens", tokens},
                  {"voice", static_cast<double>(c["voice_index"].get<int>())},
                  {"speed", c["speed"].get<double>()},
                  {"seed", static_cast<double>(c["seed"].get<int>())}};
        const auto want = npy(d + "/audio.npy");

        Inputs pinned = in;
        pinned["noise"] = npy(d + "/noise.npy");
        const auto wave = wave_of(bridge.call("infer", pinned));
        LOOM_CHECK(wave.size() == want.size());
        const double snr = snr_db(want, wave);
        LOOM_CHECK(snr >= kMinSnrDb);
        std::fprintf(stderr, "  %s: %zu samples (%zu sentences), %.1f dB from upstream\n", name.c_str(), wave.size(),
                     c["samples"].size(), snr);

        const auto seeded = wave_of(bridge.call("infer", in));
        double drift = 0.0;
        LOOM_CHECK(seeded.size() == wave.size());
        for (size_t i = 0; i < std::min(seeded.size(), wave.size()); ++i) drift = std::max(drift, std::abs(seeded[i] - wave[i]));
        LOOM_CHECK(drift <= 1e-6);

        loom::audio::SpeechStream s(bridge, pinned);
        std::vector<size_t> sizes;
        while (!s.finished()) {
            const auto p = s.next();
            if (!p.empty()) sizes.push_back(p.size());
        }
        const auto samples = c["samples"].get<std::vector<int>>();
        LOOM_CHECK(sizes.size() == samples.size());
        for (size_t i = 0; i < std::min(sizes.size(), samples.size()); ++i) LOOM_CHECK(sizes[i] == static_cast<size_t>(samples[i]));
        std::fprintf(stderr, "  %s: seed alone %.2g from pinned; stream of %zu piece(s)\n", name.c_str(), drift, sizes.size());
    }
    LOOM_TEST_REPORT_AND_RETURN();
}
