// Paradee (loom-exporter paradee_export.py, ADR-135) against upstream's own training code, sentence by sentence.
//
// Each case is a directory `scripts/paradee_reference.py` writes: the sentences' ids (each wrapped in 0s),
// the voice's row, the options, every sentence's noise draw and the joined waveform.
//
//   1. The one pass, its noise pinned: the reference's length (so every duration) and its samples within
//      kMinSnrDb -- Kokoro's modules and the source-filter head in f32, against upstream's torch.stft head.
//   2. The seed alone: the driver's `aten_randn` is ATen's own stream, so the pinned run's samples.
//   3. The stream: a piece per sentence, each the reference's sample count.
//
// LOOM_PARADEE_GGUF      the export
// LOOM_PARADEE_REF_DIR   the case directories

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

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

// Measured on 2026-11-18 over the eight voices (four languages), engine against upstream's f32: 55.9 to
// 66.0 dB. That is upstream's own float32 noise, not a difference in the graph: the export's synth graph
// run eagerly in torch at f32 is 63-69 dB from upstream on the same inputs, and at f64 the two agree to
// 111 dB (the per-sample phase interpolation, six 600-tap STFTs and a 64-harmonic sum, each rounded).
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

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_PARADEE_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_PARADEE_REF_DIR");
    if (gguf == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_PARADEE_GGUF and LOOM_PARADEE_REF_DIR\n");
        return 77;
    }
    loom::Device device = loom::Device::open(loom_test::test_device_name());
    auto model = loom::GgufModel::load(gguf, device.backends().primary);
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, device.backends());
    loom::LoomLuaBridge& bridge = session.bridge();
    LOOM_CHECK(loom::audio::SpeechStream::supported(bridge));

    std::vector<std::string> dirs;
    for (const auto& e : std::filesystem::directory_iterator(ref_env)) {
        if (e.is_directory()) dirs.push_back(e.path().string());
    }
    std::sort(dirs.begin(), dirs.end());
    LOOM_CHECK(!dirs.empty());

    for (const std::string& d : dirs) {
        const auto c = nlohmann::json::parse(std::ifstream(d + "/case.json"));
        const std::string name = std::filesystem::path(d).filename().string();
        Inputs in{{"tokens", npy(d + "/tokens.npy")},
                  {"voice", static_cast<double>(c["voice_index"].get<int>())},
                  {"speed", c["speed"].get<double>()},
                  {"seed", static_cast<double>(c["seed"].get<int>())}};
        const auto want = npy(d + "/audio.npy");

        // 1. Pinned.
        Inputs pinned = in;
        pinned["noise"] = npy(d + "/noise.npy");
        const auto wave = wave_of(bridge.call("infer", pinned));
        LOOM_CHECK(wave.size() == want.size());
        const double snr = snr_db(want, wave);
        LOOM_CHECK(snr >= kMinSnrDb);
        std::fprintf(stderr, "  %s (%s): %zu samples (%zu sentences), %.1f dB from upstream\n", name.c_str(),
                     c["voice"].get<std::string>().c_str(), wave.size(), c["samples"].size(), snr);

        // 2. The seed alone.
        const auto seeded = wave_of(bridge.call("infer", in));
        double drift = 0.0;
        LOOM_CHECK(seeded.size() == wave.size());
        for (size_t i = 0; i < std::min(seeded.size(), wave.size()); ++i) drift = std::max(drift, std::abs(seeded[i] - wave[i]));
        LOOM_CHECK(drift <= 1e-6);

        // 3. A piece per sentence, each its samples.
        loom::audio::SpeechStream s(bridge, pinned);
        std::vector<size_t> sizes;
        while (!s.finished()) {
            const auto p = s.next();
            if (!p.empty()) sizes.push_back(p.size());
        }
        const auto samples = c["samples"].get<std::vector<int>>();
        LOOM_CHECK(sizes.size() == samples.size());
        for (size_t i = 0; i < std::min(sizes.size(), samples.size()); ++i) {
            LOOM_CHECK(sizes[i] == static_cast<size_t>(samples[i]));
        }
        std::fprintf(stderr, "  %s: seed alone %.2g from pinned; stream of %zu piece(s)\n", name.c_str(), drift,
                     sizes.size());
    }
    LOOM_TEST_REPORT_AND_RETURN();
}
