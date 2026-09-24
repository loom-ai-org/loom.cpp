// `loom::load_voice`: a voice file's tensors become driver inputs by name, and only for the weights
// they were made with (ADR-045).
//
// The fixture (tests/fixtures/make_seed_kv_gguf.py) writes voice files for the seed_kv toy: one that
// fits, holding the toy's seed with every V shifted by 2000 under the input name the driver reads
// (`kv`), and three that must be refused. The one that fits is run all the way through `loom.seed_kv`,
// so a file that loads but carries the wrong numbers fails here too.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cmath>
#include <string>
#include <vector>

namespace {

const std::string kDir = LOOM_TEST_FIXTURE_DIR;

template <typename E>
bool throws(const loom::GgufModel& model, const std::string& path) {
    try {
        loom::load_voice(model, path);
    } catch (const E&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(kDir + "/seed_kv.gguf", backend.get());

    // 1. The file that fits: its metadata, and its one tensor as the input `kv`.
    const loom::VoiceFile voice = loom::load_voice(*model, kDir + "/seed_kv_voice.gguf");
    LOOM_CHECK(voice.name == "shifted" && voice.license == "CC0-1.0");
    LOOM_CHECK(voice.inputs.size() == 1 && voice.inputs.count("kv") == 1);
    LOOM_CHECK(voice.inputs.at("kv").size() == 2 * 2 * 3 * 4);

    // 2. ...and through the driver: each head picks the V it picked from the built-in seed, 2000 up.
    loom::Session session(*model, backend.get());
    const auto out = std::get<std::vector<double>>(session.bridge().call("seeded", {{"kv", voice.inputs.at("kv")}}));
    const std::vector<double> expected = {2011, 2012, 2023, 2024, 2101, 2102, 2113, 2114};
    LOOM_CHECK(out.size() == expected.size());
    for (size_t i = 0; i < std::min(out.size(), expected.size()); ++i) LOOM_CHECK(std::abs(out[i] - expected[i]) < 1e-3);

    // 3. Refused by name: other weights, another architecture (both SchemaError), a converted tensor
    //    (LoadError), and a path that is not a GGUF at all.
    LOOM_CHECK(throws<loom::SchemaError>(*model, kDir + "/seed_kv_voice_other_weights.gguf"));
    LOOM_CHECK(throws<loom::SchemaError>(*model, kDir + "/seed_kv_voice_other_arch.gguf"));
    LOOM_CHECK(throws<loom::LoadError>(*model, kDir + "/seed_kv_voice_f16.gguf"));
    LOOM_CHECK(throws<loom::LoadError>(*model, kDir + "/no_such_voice.gguf"));

    // 4. A model that declares no voice fingerprint takes no voice files, whatever the file says.
    auto other = loom::GgufModel::load(kDir + "/cfg_streams.gguf", backend.get());
    LOOM_CHECK(throws<loom::SchemaError>(*other, kDir + "/seed_kv_voice.gguf"));

    LOOM_TEST_REPORT_AND_RETURN();
}
