// Tests loom::Vocab's bos_token_id/eos_token_id/add_bos_token/add_eos_token support (src/core/vocab.cpp)
// against a tiny synthetic SentencePiece-Unigram fixture (tests/fixtures/make_vocab_ugm_bos_eos_gguf.py)
// -- closes the gap that otherwise blocks ALBERT/XLNet-style Unigram models, which wrap sequences via
// SentencePiece's own BOS/EOS convention (see EXPORT-BACKLOG.md item 4). test_vocab.cpp's real
// Conformer-CTC fixture (no bos/eos KVs at all) is the regression guard that the default (both flags
// false, absent-KV) behavior is unaffected by this addition.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

int main() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/vocab_ugm_bos_eos_test.gguf";
    auto model = loom::GgufModel::load(path, backend.get());
    LOOM_CHECK(model != nullptr);

    auto vocab = loom::Vocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    LOOM_CHECK(vocab->size() == 6);
    LOOM_CHECK(vocab->unk_id() == 0);
    LOOM_CHECK(vocab->bos_id() == 1);
    LOOM_CHECK(vocab->eos_id() == 2);

    // "hi" -> normalize() prepends "▁" (add_space_prefix defaults true) -> "▁hi", which the Viterbi
    // search picks as a single token (score -1.0, versus -6.0 for "▁h"+"i") -- then bos_id_/eos_id_ wrap
    // it since both add_bos_token/add_eos_token are true in this fixture.
    {
        const auto ids = vocab->encode("hi");
        LOOM_CHECK((ids == std::vector<int32_t>{1, 3, 2}));
    }
    // Empty input still gets bos/eos-wrapped (no real tokens in between).
    {
        const auto ids = vocab->encode("");
        LOOM_CHECK((ids == std::vector<int32_t>{1, 2}));
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
