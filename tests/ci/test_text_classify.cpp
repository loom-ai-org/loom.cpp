// The token-classification door: what comes back, what is dropped, and what is refused.
//
// The three assertions that matter are all about DECLARED facts rather than about a graph. `classify`
// exists so a punctuation model, a truecaser and a NER tagger reach one function with nothing to
// distinguish them but their own metadata (loom/core/text_classify.h), so the fixture carries labels
// and framing ids and no vocabulary at all -- if any of this were keyed on a token's spelling, it
// could not pass.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <string>

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/classify_driver.gguf";
    auto model = loom::GgufModel::load(path, backend.get());
    LOOM_CHECK(model != nullptr);

    // The contract a host dispatches on: the first non-audio pair any family declares.
    const loom::ModelContract contract = loom::ModelContract::read(*model);
    LOOM_CHECK(contract.declared());
    LOOM_CHECK(contract.task == loom::task_names::TOKEN_CLASSIFICATION);
    LOOM_CHECK(contract.interface_name() == "text2class");
    LOOM_CHECK(contract.labels.size() == 3 && contract.labels[1] == "B-PER");

    loom::Session session(*model, backend.get());

    // ---- The framing tokens the encode added do not come back -------------------------------------
    {
        // [CLS] one two [SEP], as a WordPiece encode produces it. The driver labels by POSITION, so the
        // classes are 0,1,2,0,1 across all five rows and the two that survive are rows 1 and 2.
        const std::vector<int32_t> tokens = {101, 7, 8, 9, 102};
        const auto labelled = loom::text::classify(session.bridge(), *model, tokens);
        LOOM_CHECK(labelled.size() == 3);
        LOOM_CHECK(labelled[0].token == 7 && labelled[0].label_id == 1 && labelled[0].label == "B-PER");
        LOOM_CHECK(labelled[1].token == 8 && labelled[1].label_id == 2 && labelled[1].label == "I-PER");
        // Row 3's class is 0, and its NAME is the file's -- the point being that the id alone is not
        // the answer a host can show anyone.
        LOOM_CHECK(labelled[2].token == 9 && labelled[2].label_id == 0 && labelled[2].label == "O");
    }

    // ---- ... unless the caller wants the raw alignment --------------------------------------------
    {
        loom::text::ClassifyOptions options;
        options.strip_special = false;
        const std::vector<int32_t> tokens = {101, 7, 8, 9, 102};
        const auto labelled = loom::text::classify(session.bridge(), *model, tokens, options);
        LOOM_CHECK(labelled.size() == 5);
        LOOM_CHECK(labelled[0].token == 101 && labelled[0].label_id == 0);
        LOOM_CHECK(labelled[4].token == 102 && labelled[4].label_id == 1);
    }

    // ---- The framing is PER-TOKENIZER, so BOS/EOS is stripped where CLS/SEP was ------------------
    {
        // Family 12's third checkpoint is XLM-R, whose encode wraps in `<s> ... </s>` rather than in
        // CLS/SEP (P5). Reading only BOS/SEP/PAD left the trailing `</s>` in the answer wearing the
        // head's own label for it -- one extra entry, at the END, which is where a caller comparing
        // lengths against its own word count would find it and where a NER expectation would not look.
        const std::string spm_path =
            std::string(LOOM_TEST_FIXTURE_DIR) + "/classify_driver_spm.gguf";
        auto spm = loom::GgufModel::load(spm_path, backend.get());
        LOOM_CHECK(spm != nullptr);
        loom::Session spm_session(*spm, backend.get());

        // `<s> one two three </s>` at XLM-R's own ids. The driver labels by POSITION, so all five rows
        // get 0,1,2,0,1 and the three that survive are rows 1..3.
        const std::vector<int32_t> tokens = {0, 7, 8, 9, 2};
        const auto labelled = loom::text::classify(spm_session.bridge(), *spm, tokens);
        LOOM_CHECK(labelled.size() == 3);
        LOOM_CHECK(labelled[0].token == 7 && labelled[0].label_id == 1);
        LOOM_CHECK(labelled[2].token == 9 && labelled[2].label_id == 0);

        // This fixture names no separator at all, so a file naming only some of the four framing roles
        // strips only those -- and `strip_special = false` still hands back every row.
        loom::text::ClassifyOptions options;
        options.strip_special = false;
        LOOM_CHECK(loom::text::classify(spm_session.bridge(), *spm, tokens, options).size() == 5);
    }

    // ---- A model whose output is not one row per token is refused, not zipped ---------------------
    {
        const std::string pooled_path =
            std::string(LOOM_TEST_FIXTURE_DIR) + "/pooled_classifier.gguf";
        auto pooled = loom::GgufModel::load(pooled_path, backend.get());
        LOOM_CHECK(pooled != nullptr);
        loom::Session pooled_session(*pooled, backend.get());
        bool threw = false;
        try {
            loom::text::classify(pooled_session.bridge(), *pooled, {7, 8, 9});
        } catch (const loom::Error&) {
            threw = true;
        }
        // Silently pairing one class with the first token is the failure this exists to prevent: it
        // produces a plausible answer for two of the three tokens and no signal at all.
        LOOM_CHECK(threw);
    }

    // ---- An empty sentence has no labels, and says so ---------------------------------------------
    {
        bool threw = false;
        try {
            loom::text::classify(session.bridge(), *model, {});
        } catch (const loom::Error&) {
            threw = true;
        }
        LOOM_CHECK(threw);
    }

    std::printf("test_text_classify: OK\n");
    // NOT `return 0`. `LOOM_CHECK` counts a failure and keeps going, so a hand-written main that
    // returns 0 unconditionally reports every failure to stderr and exits green -- this file's checks
    // could not fail a ctest run, and were found not to while a real one was being sabotage-tested
    // (P5). Every other test here already ends on this macro.
    LOOM_TEST_REPORT_AND_RETURN();
}
