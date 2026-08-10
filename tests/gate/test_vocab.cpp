// Vocab correctness test: loads the real converted Conformer-CTC model's real "tokenizer.ggml.*" KVs
// (a genuine SentencePiece unigram vocab) and checks Vocab::encode/decode against the actual
// `sentencepiece` Python library loaded against the SAME `.model` file -- there's no meaningful "toy
// charsmap" to substitute here, since correctness is entirely about faithfully walking this specific
// model's real XCDA-encoded precompiled_charsmap and Viterbi-segmenting with its real piece scores.
//
// Expected id sequences below were generated via:
//   python3 -c "import sentencepiece as spm; sp = spm.SentencePieceProcessor(); \
//       sp.load('<tokenizer.model>'); print(sp.encode('<text>', out_type=int))"
// against /tmp/nemo_model/extracted/*_tokenizer.model, the same checkpoint convert_conformer_ctc.py
// converts. Decode is checked only on id sequences with no <unk> (id 0) present -- real sentencepiece
// renders a decoded <unk> as the cosmetic glyph " \xE2\x81\x87 " ("⁇"), which Vocab::decode
// intentionally does not replicate (it emits the piece's literal text, "<unk>", instead); this is a
// documented, deliberate simplification (see BACKLOG.md), not a correctness gap in the normalizer or
// the Viterbi segmentation itself -- both of which this test verifies exactly, including on inputs
// that DO produce <unk> tokens (checked via encode()'s id sequence, not decode()'s rendering).

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cstdlib>
#include <sys/stat.h>

namespace {

constexpr int kSkipReturnCode = 77;

bool path_exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

} // namespace

int main() {
    // The MIL-exported artifact, which embeds the checkpoint's own SentencePiece vocab since the
    // bespoke converter retired (BACKLOG.md P4.0.17 step 3) -- the same `tokenizer.ggml.*` KVs written
    // by the same writer, which now lives at `loom_mil_compiler/spm_tokenizer_export.py`.
    const char* gguf_env = loom_test::fixture_env("LOOM_CONFORMER_CTC_MIL_GGUF");
    const std::string gguf_path = gguf_env != nullptr ? gguf_env : "conformer_ctc_mil.gguf";
    if (!path_exists(gguf_path)) {
        std::fprintf(stderr,
                      "skipping: MIL-exported Conformer-CTC GGUF not found at '%s' (set "
                      "LOOM_CONFORMER_CTC_MIL_GGUF, or run `loom-export <checkpoint> --task "
                      "automatic-speech-recognition --model conformer-ctc`)\n",
                      gguf_path.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_path, backend.get());
    auto vocab = loom::Vocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    LOOM_CHECK(vocab->size() == 1024);
    LOOM_CHECK(vocab->unk_id() == 0);

    // -- encode(): exact id-sequence match against real sentencepiece, including a case that DOES hit
    //    <unk> (punctuation/accented characters not in this ASR vocab) and one that exercises
    //    remove_extra_whitespaces (collapsing runs of spaces). --
    {
        const auto ids = vocab->encode("the cat sat");
        LOOM_CHECK((ids == std::vector<int32_t>{2, 239, 4, 230, 4}));
    }
    {
        // Also exercises add_space_prefix's lowercasing-via-charsmap-normalization: "Hello World!" ->
        // normalized to "hello world" before segmentation, and "!" has no vocab piece -> <unk>.
        const auto ids = vocab->encode("Hello World!");
        LOOM_CHECK((ids == std::vector<int32_t>{25, 42, 35, 519, 0}));
    }
    {
        // Accented character (non-ASCII) genuinely exercises the XCDA charsmap walk, not just the
        // identity path -- "é" has no vocab piece either, so it falls back to <unk> same as above.
        const auto ids = vocab->encode("café");
        LOOM_CHECK((ids == std::vector<int32_t>{239, 110, 0}));
    }
    {
        const auto ids = vocab->encode("  multiple   spaces  ");
        LOOM_CHECK((ids == std::vector<int32_t>{622, 66, 4, 704, 56, 226, 405, 67}));
    }
    {
        const auto ids = vocab->encode("a");
        LOOM_CHECK((ids == std::vector<int32_t>{3}));
    }

    // -- decode(): exact string match, restricted to id sequences with no <unk> present (see file
    //    header for why). --
    LOOM_CHECK(vocab->decode({2, 239, 4, 230, 4}) == "the cat sat");
    LOOM_CHECK(vocab->decode({3}) == "a");
    LOOM_CHECK(vocab->decode(vocab->encode("multiple spaces")) == "multiple spaces");

    // -- round-trip: decode(encode(x)) == x for inputs entirely covered by the vocab (already
    //    lowercase, no punctuation) -- matches real sentencepiece's own round-trip on the same input. --
    LOOM_CHECK(vocab->decode(vocab->encode("the cat sat")) == "the cat sat");

    LOOM_TEST_REPORT_AND_RETURN();
}
