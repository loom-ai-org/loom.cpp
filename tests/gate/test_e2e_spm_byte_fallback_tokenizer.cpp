// The SentencePiece-style byte-fallback BPE family (BpeShape::kSpmByteFallback), against the real
// HuggingFace tokenizer's own output.
//
// This family is structurally unlike every other shape BpeVocab implements -- no regex pretokenizer, no
// GPT-2 byte-level mapping, a space->U+2581 normalizer, and `<0xNN>` byte fallback -- which is why it
// was a named `NotImplementedError` in tokenizer_detect.py rather than a missing table row. Gemma 3
// shares llama.cpp's `granite-embed-multi-311m` chkhsh, which is computed over real tokenizer output,
// so the two tokenize identically.
//
// Every expectation below is `AutoTokenizer.from_pretrained(...).encode(text)` verbatim. The cases are
// chosen to hit each structural difference at least once: a bare first word (no dummy prefix), a
// multi-space run, combining characters, CJK, an emoji outside the BMP, literal tab/newline, digits
// (one token each, because the vocab has no multi-digit merges rather than because anything splits
// them), and the empty string, which must still emit BOS alone.
//
// Set LOOM_SPM_TOKENIZER_GGUF to a GGUF exported from /home/flavio/Dev/models/gemma-3-270m-it.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
constexpr int kSkipReturnCode = 77;
}

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_SPM_TOKENIZER_GGUF");
    if (gguf_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SPM_TOKENIZER_GGUF to a GGUF exported from a "
                              "SentencePiece-style BPE checkpoint (gemma-3-270m-it) to run this check\n");
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_env, backend.get());
    LOOM_CHECK(model != nullptr);

    // Auto-detected, not passed in: the point is that the exporter now recognizes this family.
    LOOM_CHECK(model->kv_str("tokenizer.ggml.pre") == "granite-embed-multi-311m");

    auto vocab = loom::BpeVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);

    struct Case {
        std::string text;
        std::vector<int32_t> ids;
    };
    const std::vector<Case> cases = {
        // "Hello" is bare and "world" carries the marker -- no dummy prefix on the first word.
        {"Hello world", {2, 9259, 1902}},
        {"The quick brown fox.", {2, 818, 3823, 8864, 37423, 236761}},
        // A two-space run merges into one piece, and the word after it is again bare.
        {"  leading spaces", {2, 138, 26016, 9952}},
        {"na\xc3\xafve caf\xc3\xa9", {2, 1789, 238527, 560, 33443}},
        {"\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88", {2, 94951, 88733}},
        {"emoji \xf0\x9f\x8e\x89 here", {2, 67906, 204906, 1590}},
        {"tab\tand\nnewline", {2, 4823, 255968, 624, 107, 73481}},
        {"1234567890",
         {2, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819, 236771}},
        {"", {2}},
    };

    for (const Case& c : cases) {
        const std::vector<int32_t> got = vocab->encode(c.text);
        if (got != c.ids) {
            std::fprintf(stderr, "encode(%s) mismatch\n  want:", c.text.c_str());
            for (int32_t i : c.ids) std::fprintf(stderr, " %d", i);
            std::fprintf(stderr, "\n  got :");
            for (int32_t i : got) std::fprintf(stderr, " %d", i);
            std::fprintf(stderr, "\n");
        }
        LOOM_CHECK(got == c.ids);

        // Round-trip: decode has to undo the U+2581 substitution and any byte fallback. The BOS piece
        // decodes literally, so compare against the text with it prepended rather than stripping ids.
        const std::string round_trip = vocab->decode(got);
        LOOM_CHECK(round_trip == "<bos>" + c.text);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
