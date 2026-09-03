// Validates the MIL-compiler-exported Dia-1.6B GGUF (loom-exporter's `dia_export.py`, family 10)
// end to end: bytes in, realigned codec tokens out, against what `transformers` produces for the same
// sentence under the same decoding algorithm.
//
// **This is the whole family in one call**, which is why it is an end-to-end check and not a
// per-topology one. A Dia generation runs the byte encoder once, projects 36 cross-attention K/V once,
// then loops a KV-cached 18-layer decoder that emits NINE tokens a step -- and the driver on top of
// that applies a delay scaffold on the way in and undoes the delay on the way out. Every one of those
// pieces is load-bearing for the integers below, and each fails differently: a wrong encoder shifts
// every code, a wrong cross-attention axis binding produces garbage after the first frame, and a
// wrong delay realignment produces codes that are individually plausible and collectively silence.
//
// **The comparison is on exact integer equality over every emitted code, and on the frame COUNT.** The
// count is not a formality -- it is family 11's own lesson one family over (a codec decoder that
// silently returned one frame for every input, Retro-013's sibling): the number of frames is decided
// by where channel 0 says EOS, which is decided by 33 decoder steps' worth of correct cross-attention,
// so a length that matches is a strong statement on its own.
//
// The reference was captured from `DiaForConditionalGeneration.generate(do_sample=False,
// guidance_scale=None)` -- greedy, classifier-free guidance OFF, because that is the algorithm the
// driver implements today and a comparison against a different one would grade the sampler rather than
// the export. See loom-exporter's `dia_export.py` for what CFG would add.
//
// Not generated at ctest time (needs the real Dia-1.6B checkpoint + coremltools) -- skips cleanly when
// the fixture is absent. To (re)generate the FIXTURE: `loom-export /path/to/dia-1.6b -o dia_mil.gguf`,
// or point LOOM_DIA_MIL_GGUF at an existing copy. To regenerate the NUMBERS below:
//
//     ~/.venvs/piper/bin/python scripts/dia_reference_codes.py --model ~/Dev/models/dia-1.6b --frames 32
//
// which prints them as the declarations that follow, ready to paste. It exists because the reference
// for the family-2 gate did not: `test_e2e_lfm2_mil_export.cpp` says "see the conversation/PR
// description for the capture script", which is a reference nobody can re-derive. Whenever the driver
// changes decoding algorithm -- the sampler, classifier-free guidance -- that script has to change with
// it on the same commit, or this test quietly starts measuring two different samplers against
// each other rather than the export against `transformers`.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cstdio>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;

// "[S1] Hello world." through this checkpoint's own byte vocabulary: `[S1]` is one added token (id 1)
// and every byte after it is its own ordinal, which is the Dia parameterisation `loom::ByteVocab`
// grew a `byte_offset` KV for. Written out rather than encoded here so the test still isolates the
// GRAPH when the tokenizer is what broke -- the tokenizer has its own check below.
const std::vector<double> kPromptIds = {1, 32, 72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 46};

constexpr int kExpectedFrames = 32;
constexpr int kChannels = 9;

// Every code `transformers` produced for that sentence, frame-major -- 32 frames x 9 channels.
const int32_t kExpectedCodes[] = {
    568, 778, 338, 524, 967, 56, 728, 550, 90, 568, 778, 10,
    649, 364, 21, 741, 378, 258, 698, 697, 737, 674, 364, 320,
    983, 507, 930, 776, 754, 612, 832, 95, 21, 759, 771, 302,
    776, 740, 953, 1011, 217, 599, 530, 120, 541, 776, 402, 385,
    762, 900, 434, 129, 147, 53, 776, 402, 826, 135, 347, 510,
    754, 125, 769, 776, 402, 826, 769, 347, 790, 610, 147, 33,
    776, 402, 826, 135, 347, 20, 754, 676, 92, 776, 402, 826,
    135, 347, 20, 714, 113, 603, 776, 402, 826, 135, 347, 20,
    714, 147, 603, 776, 402, 826, 135, 347, 20, 714, 147, 603,
    776, 402, 826, 135, 347, 20, 714, 147, 603, 776, 402, 826,
    135, 347, 20, 714, 147, 603, 776, 402, 826, 135, 347, 20,
    714, 147, 603, 776, 402, 826, 135, 347, 20, 714, 147, 603,
    776, 402, 826, 135, 347, 20, 714, 147, 603, 776, 402, 826,
    135, 347, 20, 714, 147, 603, 776, 402, 826, 135, 347, 20,
    714, 147, 603, 776, 402, 826, 135, 347, 20, 714, 147, 603,
    776, 402, 826, 135, 347, 20, 714, 147, 603, 776, 402, 826,
    135, 347, 20, 714, 147, 603, 776, 402, 826, 135, 347, 20,
    714, 147, 603, 776, 402, 826, 135, 347, 20, 714, 147, 603,
    776, 402, 826, 135, 347, 20, 714, 147, 603, 776, 402, 826,
    135, 347, 20, 714, 147, 603, 776, 402, 826, 135, 347, 20,
    714, 147, 603, 776, 402, 826, 135, 347, 20, 714, 147, 603,
    776, 402, 826, 135, 347, 20, 714, 147, 603, 776, 402, 826,
    135, 347, 20, 714, 147, 68, 776, 402, 826, 135, 347, 20,
    714, 147, 68, 776, 606, 571, 135, 347, 757, 107, 120, 415,
};

} // namespace

int main() {
    const char* env = loom_test::fixture_env("LOOM_DIA_MIL_GGUF");
    const std::string gguf_path = env != nullptr ? env : "dia_mil.gguf";
    if (!loom_test::path_exists(gguf_path)) {
        std::fprintf(stderr, "skipping: '%s' not found (set LOOM_DIA_MIL_GGUF, or run "
                              "`loom-export <dia-1.6b> -o dia_mil.gguf`)\n", gguf_path.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_path, backend.get());
    const std::string driver_script = model->kv_str("model.driver_script");
    LOOM_CHECK(!driver_script.empty());

    // The tokenizer half, checked separately from the graph half: this checkpoint is the first
    // `byte_offset != 3` file, and `[S1]` is an ADDED token sitting at an id that is also a byte value
    // under that offset. Both properties are new, and a wrong answer here would otherwise show up as a
    // wrong code 33 steps later.
    auto vocab = loom::ByteVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    LOOM_CHECK(vocab->byte_offset() == 0);
    LOOM_CHECK(!vocab->adds_eos());
    const std::vector<int32_t> encoded = vocab->encode("[S1] Hello world.");
    LOOM_CHECK(encoded.size() == kPromptIds.size());
    for (size_t i = 0; i < encoded.size(); ++i) {
        LOOM_CHECK(encoded[i] == static_cast<int32_t>(kPromptIds[i]));
    }
    LOOM_CHECK(vocab->id_to_piece(1) == "[S1]");

    loom::LoomLuaBridge bridge(backend.get());
    std::unique_ptr<loom::KvCache> kv_cache;
    for (const std::string& mod_name : model->topology_names()) {
        loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json(mod_name));
        if (topo.uses_kv_cache() && kv_cache == nullptr) {
            kv_cache = loom::make_kv_cache(*model, backend.get());
        }
        loom::KvCache* cache_for_module = topo.uses_kv_cache() ? kv_cache.get() : nullptr;
        bridge.register_module(mod_name, *model, std::move(topo), cache_for_module, nullptr);
    }
    bridge.load_script(driver_script);

    // **Both sides stop because they were told to, at the same row, and that is deliberate.** This
    // sentence does not make the model emit EOS on its own inside any budget a gate test can afford --
    // it ran 63 rows without one. So the reference was captured under `max_new_tokens=48`, which
    // `transformers` turns into a FORCED eos at row 33 (`DiaEOSDelayPatternLogitsProcessor`'s
    // `reached_max_len` clause, at `max_length - max(delay) - 1`), yielding 32 audio frames.
    //
    // This driver's `max_new_tokens` counts AUDIO FRAMES rather than rows -- a better host contract,
    // since rows are an artefact of the delay pattern -- so the same stop is spelled 32 here. The
    // ceiling forcing an eos rather than breaking is what makes the two comparable at all: a driver
    // that truncated would return short final frames, because channel k's contribution to the last
    // frame lives MAX_DELAY rows further on.
    loom::LoomLuaBridge::Value result = bridge.call(
        "infer", {{"tokens", kPromptIds}, {"max_new_tokens", 32.0}});
    const auto& codes = std::get<std::vector<double>>(result);

    const size_t expected = static_cast<size_t>(kExpectedFrames) * kChannels;
    std::fprintf(stderr, "dia: %zu codes (%zu frames), expected %zu (%d frames)\n",
                 codes.size(), codes.size() / kChannels, expected, kExpectedFrames);
    LOOM_CHECK(codes.size() == expected);

    size_t mismatches = 0;
    for (size_t i = 0; i < expected; ++i) {
        const auto got = static_cast<int32_t>(codes[i]);
        if (got != kExpectedCodes[i]) {
            if (mismatches < 8) {
                std::fprintf(stderr, "  frame %zu channel %zu: expected %d, got %d\n",
                             i / kChannels, i % kChannels, kExpectedCodes[i], got);
            }
            ++mismatches;
        }
    }
    std::fprintf(stderr, "dia: %zu/%zu codes differ from the transformers reference\n",
                 mismatches, expected);
    LOOM_CHECK(mismatches == 0);

    LOOM_TEST_REPORT_AND_RETURN();
}
