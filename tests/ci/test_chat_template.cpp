// P4.23: an instruction-tuned checkpoint's chat turn, from both ends -- the tokenizer that has to be
// able to EMIT a marker, and the template that says where the markers go.
//
// The two are one feature and are tested as one. A chat template is worthless if its markers tokenize
// as seven literal ids apiece (which is what `encode` did before the added-token pre-pass), and the
// pre-pass has nothing to prove without a template that uses it.
//
// Hermetic: `make_chat_template_gguf.py` builds a 263-token byte-level vocab whose ids are
// hand-traceable, with three added tokens and a ChatML decomposition. The real-checkpoint counterpart,
// where every expectation is `AutoTokenizer.encode` verbatim, is
// tests/gate/test_e2e_spm_byte_fallback_tokenizer.cpp.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

// The fixture's own ids, from make_chat_template_gguf.py's construction: base vocab is the 256
// byte-level entries, then ll/he/hell/hello, then the three added tokens.
constexpr int32_t kHello = 259;
constexpr int32_t kImStart = 260;
constexpr int32_t kImEnd = 261;
constexpr int32_t kDoubleNewline = 262;

void expect_ids(const loom::BpeVocab& vocab, const std::string& text,
                const std::vector<int32_t>& want) {
    const std::vector<int32_t> got = vocab.encode(text);
    if (got != want) {
        std::fprintf(stderr, "encode(%s) mismatch\n  want:", text.c_str());
        for (int32_t i : want) std::fprintf(stderr, " %d", i);
        std::fprintf(stderr, "\n  got :");
        for (int32_t i : got) std::fprintf(stderr, " %d", i);
        std::fprintf(stderr, "\n");
    }
    LOOM_CHECK(got == want);
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    const std::string dir = std::string(LOOM_TEST_FIXTURE_DIR);
    auto model = loom::GgufModel::load(dir + "/chat_template_test.gguf", backend.get());
    auto vocab = loom::BpeVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);

    // --- 1. A marker is ONE id, which is the whole item. ---
    expect_ids(*vocab, "<|im_start|>", {kImStart});
    expect_ids(*vocab, "<|im_end|>", {kImEnd});

    // --- 2. Text either side of a marker still goes through BPE, and the marker does not merge with
    //        it. The longest-match scan is what makes the second half true: "hello<|im_end|>" must not
    //        let the marker's leading '<' join the word.
    expect_ids(*vocab, "hello<|im_end|>", {kHello, kImEnd});
    expect_ids(*vocab, "<|im_start|>hello<|im_end|>", {kImStart, kHello, kImEnd});

    // --- 3. A NON-special added token splits too. Gemma 3 declares 6408 of these (whitespace runs),
    //        and a pre-pass that only knew about markers would disagree with the reference tokenizer on
    //        ordinary prose rather than only on chat.
    const std::vector<int32_t> two_newlines = vocab->encode("\n\n");
    LOOM_CHECK(two_newlines == std::vector<int32_t>{kDoubleNewline});
    const std::vector<int32_t> around = vocab->encode("hello\n\nhello");
    LOOM_CHECK((around == std::vector<int32_t>{kHello, kDoubleNewline, kHello}));

    // --- 4. Adjacent markers leave an EMPTY segment between them, which must contribute nothing. ---
    expect_ids(*vocab, "<|im_start|><|im_end|>", {kImStart, kImEnd});

    // --- 5. Round-trip, including the added token whose content the byte decoder has no key for. ---
    LOOM_CHECK(vocab->decode({kImStart, kHello, kImEnd}) == "<|im_start|>hello<|im_end|>");
    LOOM_CHECK(vocab->decode({kHello, kDoubleNewline, kHello}) == "hello\n\nhello");
    for (const char* text : {"hello", "<|im_start|>hello", "hello\n\nhello", "a<|im_end|>b"}) {
        LOOM_CHECK(vocab->decode(vocab->encode(text)) == std::string(text));
    }

    // --- 6. CONTROL is what `is_control` reports, and USER_DEFINED is not it: an added token that is
    //        not special is text, and a host stripping "control tokens" from an answer must not eat it.
    LOOM_CHECK(vocab->is_control(kImStart));
    LOOM_CHECK(vocab->is_control(kImEnd));
    LOOM_CHECK(!vocab->is_control(kDoubleNewline));
    LOOM_CHECK(!vocab->is_control(kHello));

    // --- 7. The checkpoint's FULL stop set, not the one scalar KV. ---
    const std::vector<int32_t> stops = loom::text::eos_token_ids(*model);
    LOOM_CHECK((stops == std::vector<int32_t>{0, kImEnd}));

    // --- 8. The template assembles, and what it assembles TOKENIZES to the markers' own ids. ---
    auto tmpl = loom::ChatTemplate::load(*model);
    LOOM_CHECK(tmpl != nullptr);
    LOOM_CHECK(tmpl->has_role("user") && tmpl->has_role("assistant") && tmpl->has_role("system"));
    LOOM_CHECK(!tmpl->has_role("tool"));

    const std::string prompt = tmpl->apply({{"user", "hello"}}, /*add_generation_prompt=*/true);
    LOOM_CHECK(prompt == "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n");
    const std::vector<int32_t> prompt_ids = vocab->encode(prompt);
    // Only the markers' ids are asserted by name; what matters is that each appears ONCE rather than as
    // a run of literal characters, and that the turn opens and closes where the template said.
    LOOM_CHECK(prompt_ids.front() == kImStart);
    LOOM_CHECK(prompt_ids.back() != kImEnd);
    size_t marker_count = 0;
    for (int32_t id : prompt_ids) {
        if (id == kImStart || id == kImEnd) ++marker_count;
    }
    LOOM_CHECK(marker_count == 3); // open user, close user, open assistant
    LOOM_CHECK(vocab->decode(prompt_ids) == prompt);

    // A multi-turn transcript, and the generation prompt as the only difference between two renders.
    const std::vector<loom::ChatMessage> conversation = {
        {"system", "be terse"}, {"user", "hello"}, {"assistant", "hello"}, {"user", "hello"}};
    LOOM_CHECK(tmpl->apply(conversation, true) ==
               tmpl->apply(conversation, false) + "<|im_start|>assistant\n");

    // --- 9. An undeclared role is an ERROR, not a dropped message. Gemma 3 is the live case: its
    //        template has no system block at all, so a caller passing one must hear about it.
    bool threw = false;
    try {
        tmpl->apply({{"tool", "{}"}}, true);
    } catch (const loom::Error& e) {
        threw = true;
        LOOM_CHECK(std::string(e.what()).find("'tool'") != std::string::npos);
    }
    LOOM_CHECK(threw);

    threw = false;
    try {
        tmpl->apply({}, true);
    } catch (const loom::Error&) {
        threw = true;
    }
    LOOM_CHECK(threw);

    // --- 10. BACKWARD COMPATIBILITY, stated as a test rather than as a comment: the identical vocab
    //         WITHOUT `tokenizer.ggml.token_type` must tokenize a marker exactly as it did before
    //         P4.23 -- as its literal characters -- because the file does not say which ids are added
    //         and guessing would make `encode` disagree with the reference tokenizer on ordinary text
    //         that happens to look like a marker.
    auto legacy_model = loom::GgufModel::load(dir + "/chat_template_legacy_test.gguf", backend.get());
    auto legacy = loom::BpeVocab::load(*legacy_model);
    LOOM_CHECK(legacy != nullptr);
    const std::vector<int32_t> legacy_ids = legacy->encode("<|im_start|>");
    LOOM_CHECK(legacy_ids.size() > 1);
    LOOM_CHECK(legacy_ids != vocab->encode("<|im_start|>"));
    LOOM_CHECK(!legacy->is_control(kImStart));
    // ... and the ordinary text either side of it is unaffected by the pre-pass existing at all.
    LOOM_CHECK(legacy->encode("hello") == vocab->encode("hello"));

    LOOM_TEST_REPORT_AND_RETURN();
}
