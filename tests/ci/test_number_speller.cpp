// loom::NumberSpeller (ADR-059): transformers' `EnglishNumberNormalizer`, the one SpeechT5Tokenizer
// runs with `normalize=True`, with its words and character tables as file data. The expected strings
// are the reference's own output with loom's two documented fixes applied (an all-zero integer is
// "zero"; every thousands separator goes) -- produced by the reference, not written by hand, and
// covering each branch: currency before and after a sign, percent, the two-digit decimal cap and its
// backtracking, the `\w` lookarounds, Unicode digits, and the space collapse. The development-time
// differential against the reference is 40,046/40,046 strings identical, spelled text and ids.
//
// Also here: a Unigram vocabulary FUSES a run of unknown characters into one `<unk>`, as SentencePiece
// does, which this engine did not until the same differential found it.

#include "test_util.h"
#include "cpu_backend.h"

#include "loom/loom.h"
#include "loom/core/number_speller.h"
#include "loom/core/vocab.h"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

int main() {
    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/number_speller_test.gguf";
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(path, backend.get());
    LOOM_CHECK(model != nullptr);
    auto speller = loom::NumberSpeller::load(*model);
    LOOM_CHECK(speller != nullptr);

    const std::vector<std::pair<std::string, std::string>> cases = {
        {"It costs $15,000.50 in 2026.", "It costs fifteen thousand point five zero dollars in two thousand twenty six."},
        {"0", "zero"},
        {"0.5", "zero point five"},
        {"-0.05%", "minus zero point zero five percent"},
        {"1,000,000 people", "one million people"},
        {"$1,234,567.89", "one million two hundred and thirty four thousand five hundred and sixty seven point eight nine dollars"},
        {"5%a", "five%a"},
        {"12.345", "twelve.three hundred and forty five"},
        {"call 911", "call nine hundred and eleven"},
        {"10:00", "ten:zero"},
        {"1,2,3", "one hundred and twenty three"},
        {"-$5", "minus five dollars"},
        {"₹99.9", "ninety nine point nine indian rupees"},
        {"a1 1a _1 1_", "a1 1a _1 1_"},
        {"٣٤ apples", "thirty four apples"},
        {"1. .5", "one. .five"},
        {"100.00", "one hundred"},
        {"15 000", "fifteen zero"},
        {"1st", "1st"},
        {"two  spaces  3", "two spaces three"},
        {"€-5", "€minus five"},
        {"-€5", "minus five euros"},
        {"3.14159", "three.fourteen thousand one hundred and fifty nine"},
        {"1000000000000", "one trillion"},
        {"x=-7", "x=minus seven"},
        {"¥0", "zero japanese yen"},
        // Where the reference RAISES (two currency symbols; past the last scale word), the text stays.
        {"$€5", "$€5"},
        {std::string(37, '9'), std::string(37, '9')},
    };
    for (const auto& [in, want] : cases) {
        const std::string got = speller->apply(in);
        if (got != want) std::fprintf(stderr, "apply(\"%s\") = \"%s\", want \"%s\"\n", in.c_str(), got.c_str(), want.c_str());
        LOOM_CHECK(got == want);
    }

    // Through the vocabulary: the speller runs first, so a digit never reaches the table (which has
    // none), and the unknown run that is left fuses into one `<unk>` (id 3).
    auto vocab = loom::Vocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    const auto ids = vocab->encode("a 12 ☃☃☃ b");
    std::string decoded = vocab->decode(ids);
    std::fprintf(stderr, "encode(\"a 12 ☃☃☃ b\") -> \"%s\"\n", decoded.c_str());
    LOOM_CHECK(decoded == "a twelve <unk> b");
    int unknowns = 0;
    for (int32_t id : ids) unknowns += id == vocab->unk_id();
    LOOM_CHECK(unknowns == 1);

    LOOM_TEST_REPORT_AND_RETURN();
}
