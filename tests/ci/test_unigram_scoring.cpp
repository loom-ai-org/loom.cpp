// Which arithmetic decides an exact Unigram tie (ADR-060). SentencePiece stores path scores as float
// and compares piece candidates in double; `tokenizers` uses doubles throughout. On the fixture's text the
// two pick different splits of `ggg`, so the same pieces and scores give `gg g` when the file says its
// vocabulary came from a SentencePiece `.model` and `g gg` when it does not. The expected ids are the
// real `sentencepiece` library's (0.2.1) for the first file.

#include "test_util.h"
#include "cpu_backend.h"

#include "loom/loom.h"
#include "loom/core/vocab.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

std::vector<int32_t> encode(const std::string& file, const std::string& text) {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    auto model = loom::GgufModel::load(std::string(LOOM_TEST_FIXTURE_DIR) + "/" + file, backend.get());
    auto vocab = loom::Vocab::load(*model);
    auto ids = vocab->encode(text);
    std::fprintf(stderr, "%s: ", file.c_str());
    for (int32_t id : ids) std::fprintf(stderr, "%s ", vocab->id_to_piece(id).c_str());
    std::fprintf(stderr, "\n");
    return ids;
}

} // namespace

int main() {
    // ids: <pad>=0 </s>=1 <unk>=2 ▁=3 x=4 y=5 g=6 gg=7
    const std::vector<int32_t> sentencepiece = {3, 5, 5, 4, 5, 4, 7, 6};
    const std::vector<int32_t> doubles = {3, 5, 5, 4, 5, 4, 6, 7};
    LOOM_CHECK(encode("unigram_tie_sentencepiece.gguf", "yyxyxggg") == sentencepiece);
    LOOM_CHECK(encode("unigram_tie_doubles.gguf", "yyxyxggg") == doubles);
    // Away from a tie the two agree.
    LOOM_CHECK(encode("unigram_tie_sentencepiece.gguf", "xyg") == encode("unigram_tie_doubles.gguf", "xyg"));
    LOOM_TEST_REPORT_AND_RETURN();
}
