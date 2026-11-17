// tiny-voca (ADR-133) against its own `speak_text`, sentence by sentence.
//
// Each case is a directory `scripts/tiny_voca_reference.py` writes: the sentences' phonemes (dizione's,
// stress as the table's offset ids), the options, every sentence's noise draw and the joined waveform.
//
//   1. The one pass, its noise pinned: the reference's length (so every duration), and its samples within
//      kWaveTolerance -- the source-filter head in f32, against upstream's FFTs (the same graph is 5.8e-8
//      from upstream at f64; tiny_voca_export.py).
//   2. The seed alone: the driver's `aten_randn` is ATen's own stream, so the pinned run's samples.
//   3. The stream: a piece per sentence, each the reference's frame count times the hop.
//
// LOOM_TINY_VOCA_GGUF      the export
// LOOM_TINY_VOCA_REF_DIR   the case directories

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

// Measured against upstream run at f64 on `hello` (peak 0.8): upstream's own f32 is 5.7e-4 away, the
// engine 2.1e-3 -- its STFTs are 512-tap convolutions summed in f32 where torch runs FFTs (the same graph
// at f64 is 5.8e-8 from upstream, so the difference is precision, not structure). Against upstream's
// f32 the cases measure 0.6e-3 to 2.6e-3, the same maxima with the pulse train in the graph (ADR-134:
// it moves the waveform by 6e-5, under the STFTs' error).
constexpr double kWaveTolerance = 4e-3;
constexpr int kVocab = 76;
constexpr int kHop = 256;

std::vector<double> npy(const std::string& path) {
    std::vector<int64_t> shape;
    const auto v = loom_test::read_npy_f32(path, shape);
    return std::vector<double>(v.begin(), v.end());
}

std::vector<double> wave_of(const loom::LoomLuaBridge::Value& v) { return std::get<std::vector<double>>(v); }

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_TINY_VOCA_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_TINY_VOCA_REF_DIR");
    if (gguf == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_TINY_VOCA_GGUF and LOOM_TINY_VOCA_REF_DIR\n");
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
        const auto tokens = npy(d + "/tokens.npy");
        const auto stress = npy(d + "/stress.npy");
        std::vector<double> ids(tokens.size());
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = tokens[i] + kVocab * stress[i];
        Inputs in{{"tokens", ids},
                  {"speaker", c["voice"].get<std::string>() == "M2" ? 1.0 : 0.0},
                  {"speed", c["speed"].get<double>()},
                  {"pitch_shift", c["pitch_shift"].get<double>()},
                  {"pitch_range", c["pitch_range"].get<double>()},
                  {"seed", static_cast<double>(c["seed"].get<int>())}};
        const auto want = npy(d + "/audio.npy");

        // 1. Pinned.
        Inputs pinned = in;
        pinned["noise"] = npy(d + "/noise.npy");
        const auto wave = wave_of(bridge.call("infer", pinned));
        LOOM_CHECK(wave.size() == want.size());
        double worst = 0.0;
        for (size_t i = 0; i < std::min(wave.size(), want.size()); ++i) worst = std::max(worst, std::abs(wave[i] - want[i]));
        LOOM_CHECK(worst <= kWaveTolerance);
        std::fprintf(stderr, "  %s: %zu samples (%zu sentences), max|d| %.3g\n", name.c_str(), wave.size(),
                     c["frames"].size(), worst);

        // 2. The seed alone.
        const auto seeded = wave_of(bridge.call("infer", in));
        double drift = 0.0;
        LOOM_CHECK(seeded.size() == wave.size());
        for (size_t i = 0; i < std::min(seeded.size(), wave.size()); ++i) drift = std::max(drift, std::abs(seeded[i] - wave[i]));
        LOOM_CHECK(drift <= 1e-6);

        // 3. A piece per sentence, each its frames.
        loom::audio::SpeechStream s(bridge, pinned);
        std::vector<size_t> sizes;
        while (!s.finished()) {
            const auto p = s.next();
            if (!p.empty()) sizes.push_back(p.size());
        }
        const auto frames = c["frames"].get<std::vector<int>>();
        LOOM_CHECK(sizes.size() == frames.size());
        for (size_t i = 0; i < std::min(sizes.size(), frames.size()); ++i) {
            LOOM_CHECK(sizes[i] == static_cast<size_t>(frames[i]) * kHop);
        }
        std::fprintf(stderr, "  %s: seed alone %.2g from pinned; stream of %zu piece(s)\n", name.c_str(), drift,
                     sizes.size());
    }
    LOOM_TEST_REPORT_AND_RETURN();
}
