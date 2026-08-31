// P4.23's acceptance, end to end on a real instruction-tuned checkpoint: templated, tokenized, decoded,
// and STOPPED where the checkpoint says a turn stops.
//
// This is the whole item in one test, because the item's three defects only produce the reported
// symptom together:
//
//   1. `tokenize` could not emit a marker, so a template's `<start_of_turn>` went in as eight literal
//      ids and the model never saw a turn boundary;
//   2. the export read the tokenizer config's single `eos_token`, so the loop did not stop on the id an
//      IT turn actually ends on -- gemma-3-270m-it declares `[1, 106]` and only 1 reached the file;
//   3. and greedy decoding turned the resulting malformed prompt into a repetition LOOP rather than a
//      merely wrong answer (P4.24).
//
// The observable consequence of all three is that the model kept OPENING NEW TURNS -- Gemma 3 emitted
// `<end_of_turn>model\n<start_of_turn>artist\n...` to the ceiling -- so the assertion this test is
// built around is that the answer contains none of the template's own markers. A model answering inside
// a turn it was given produces text; one that was never given a turn produces structure.
//
// It also checks that generation STOPS on its own, which is the other face of the same defect. The
// ceiling is 200 rather than 80 because a 350M model can legitimately write a long paragraph: LFM2's
// marker-free, on-topic answer to this question is 117 tokens, and failing it at 80 would be measuring
// verbosity rather than the bug.
//
// Generic over any GGUF carrying a chat template rather than pinned to one checkpoint, because what
// makes a model eligible is a property of the FILE (`tokenizer.chat_template.*`, written by
// loom-exporter's chat_template_export.py only for a template that verified against
// `apply_chat_template`). Gemma 3 270M IT and SmolLM2 360M Instruct both qualify; a base model does not,
// and the test says so and skips rather than pretending.
//
// Set LOOM_CHAT_LM_GGUF to a GGUF exported from an instruction-tuned causal LM.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"
#include "loom/core/conv_state_cache.h"

#include "cpu_backend.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {
constexpr int kSkipReturnCode = 77;

// Larger than any answer these models give -- Gemma 3 stops at 13 tokens, SmolLM2 at 29, LFM2 at 117 --
// so "it stopped" is a real observation rather than the ceiling being generous. The pre-P4.23 artifacts
// hit whatever this number was, on every prompt.
constexpr uint32_t kCeiling = 200;
} // namespace

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_CHAT_LM_GGUF");
    if (gguf_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_CHAT_LM_GGUF to a GGUF exported from an "
                              "instruction-tuned causal LM (gemma-3-270m-it, smollm2-360m-it)\n");
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_env, backend.get());
    LOOM_CHECK(model != nullptr);

    auto tmpl = loom::ChatTemplate::load(*model);
    if (tmpl == nullptr) {
        std::fprintf(stderr, "skipping: %s carries no chat template. A base model has none, and one "
                              "whose template did not decompose exports without it -- see the export's "
                              "own \"no chat template\" line.\n", gguf_env);
        return kSkipReturnCode;
    }
    auto vocab = loom::BpeVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);

    // --- 1. The template's own markers are single ids. Without this nothing below means anything: a
    //        turn boundary the model cannot see is not a turn boundary.
    LOOM_CHECK(!tmpl->roles().empty());
    const std::string opener = tmpl->apply({{"user", "x"}}, /*add_generation_prompt=*/false);
    const std::vector<int32_t> opener_ids = vocab->encode(opener);
    // Every checkpoint here brackets a turn with CONTROL tokens; if the pre-pass were off they would be
    // spelled out as ordinary text and none would be control.
    size_t control_ids = 0;
    for (int32_t id : opener_ids) {
        if (vocab->is_control(id)) ++control_ids;
    }
    std::fprintf(stderr, "templated user turn: %zu ids, %zu of them control\n",
                 opener_ids.size(), control_ids);
    LOOM_CHECK(control_ids >= 2);
    LOOM_CHECK(vocab->decode(opener_ids).find(opener) != std::string::npos);

    // --- 2. The checkpoint's stop set reached the file. gemma-3-270m-it declares two; a model with one
    //        is fine, and this only asserts the set is not empty, because "how many" is the
    //        checkpoint's business and "any at all" is the engine's.
    const std::vector<int32_t> stops = loom::text::eos_token_ids(*model);
    LOOM_CHECK(!stops.empty());
    std::fprintf(stderr, "stop ids:");
    for (int32_t id : stops) std::fprintf(stderr, " %d (%s)", id, vocab->id_to_piece(id).c_str());
    std::fprintf(stderr, "\n");

    // --- 3. The item's actual acceptance: a real question, correctly templated, STOPS. ---
    loom::Session session(*model, backend.get());
    const std::string prompt = tmpl->apply({{"user", "Who discovered Brazil?"}}, true);
    const std::vector<int32_t> prompt_tokens = vocab->encode(prompt);
    LOOM_CHECK(!prompt_tokens.empty());

    loom::text::GenerateOptions gen;
    gen.max_new_tokens = kCeiling;
    // Greedy, so the check is about the template and the stop set rather than about a lucky draw.
    // P4.24's own invariants are tested separately (tests/ci/test_sample_row.cpp).
    gen.temperature = 0.0f;
    const std::vector<int32_t> generated =
        loom::text::generate(session.bridge(), *model, prompt_tokens, gen);
    const std::string answer = vocab->decode(generated);
    std::fprintf(stderr, "answer (%zu tokens): %s\n", generated.size(), answer.c_str());

    // It said something. SmolLM2's failure mode was the opposite of Gemma's -- its first token was its
    // own `<|im_end|>`, `strip_eos` dropped it, and `generate` returned the empty string.
    LOOM_CHECK(!generated.empty());
    LOOM_CHECK(!answer.empty());

    // **The answer stays inside its turn**, and the check is on the IDS rather than on the text: not one
    // generated id is a CONTROL token. This is the reported symptom stated exactly -- without the
    // markers being encodable the model never saw a turn boundary, and what it emitted was turn after
    // turn (`<end_of_turn>model\n<start_of_turn>artist\n...`), which is a run of control ids. A trailing
    // stop was already removed by `strip_eos`, so anything left here is the model opening structure.
    for (int32_t id : generated) {
        if (vocab->is_control(id)) {
            std::fprintf(stderr, "generated a control token: %d (%s) -- the model is emitting turn "
                                  "structure rather than answering inside one\n",
                         id, vocab->id_to_piece(id).c_str());
        }
        LOOM_CHECK(!vocab->is_control(id));
    }

    // ... and it stopped on its own, which is the number the bug report was actually about.
    LOOM_CHECK(generated.size() < kCeiling);
    // No stop token survives into the text. Stripping only the SCALAR eos left `<end_of_turn>`'s literal
    // spelling on the end of every chat answer, which is P4.23's second defect showing through the
    // detokenizer rather than through the loop.
    for (int32_t id : stops) {
        LOOM_CHECK(answer.find(vocab->id_to_piece(id)) == std::string::npos);
    }
    // The prompt is not in the answer: `generate` returns the generated ids alone, and a model given a
    // real turn no longer re-emits its own prompt as a continuation either.
    LOOM_CHECK(answer.find("Who discovered Brazil?") == std::string::npos);

    // --- 4. A role the checkpoint does not declare is an error rather than a dropped message. Gemma 3
    //        is the live case; a ChatML model declares `system` and this is skipped for it.
    if (!tmpl->has_role("system")) {
        bool threw = false;
        try {
            tmpl->apply({{"system", "be terse"}, {"user", "hello"}}, true);
        } catch (const loom::Error&) {
            threw = true;
        }
        LOOM_CHECK(threw);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
