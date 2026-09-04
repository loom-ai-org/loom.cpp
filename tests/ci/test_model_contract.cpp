// What a file says about itself, and what a file that says nothing gets instead.
//
// The second half is the half that matters. `ModelContract` and `AsrDecodeTable` exist so a host can
// dispatch without recognising an architecture, but every GGUF on disk today predates them -- so a
// change that broke the fallback would break every model in the wild while a declared-only test stayed
// green. Both fixtures are loaded here for that reason.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <string>

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    // ---- A file that declares its contract --------------------------------------------------------
    {
        const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/contract_declared.gguf";
        auto model = loom::GgufModel::load(path, backend.get());
        LOOM_CHECK(model != nullptr);

        const loom::ModelContract c = loom::ModelContract::read(*model);
        LOOM_CHECK(c.declared());
        LOOM_CHECK(c.task == loom::task_names::ASR);
        LOOM_CHECK(c.input_kind == loom::modality::AUDIO);
        LOOM_CHECK(c.output_kind == loom::modality::TOKEN_IDS);
        LOOM_CHECK(c.sample_rate == 16000);
        LOOM_CHECK(c.clip_samples == 480000);
        LOOM_CHECK(c.text_frontend == "vocab");
        LOOM_CHECK(c.languages.size() == 2 && c.languages[0] == "en");
        LOOM_CHECK(c.entry_points.size() == 1 && c.entry_points[0] == "infer");

        // The interface a host exposes IS the modality pair, which is why no lookup table maps task
        // names to doors. `token_ids` out folds onto "text": supplying or receiving ids rather than a
        // string is how you talk to a model that cannot encode, not a different contract.
        LOOM_CHECK(c.interface_name() == "speech2text");

        // The decode table, read WITHOUT a vocabulary -- deliberately null here, because a file that
        // declares its table needs none, and passing one would hide a regression where the declared
        // values are ignored in favour of a spelled lookup.
        const loom::audio::AsrDecodeTable t =
            loom::audio::AsrDecodeTable::read(*model, nullptr, c);
        LOOM_CHECK(t.timestamped());
        LOOM_CHECK(t.timestamp_first_id == 50364);
        LOOM_CHECK(t.timestamp_step_sec > 0.019 && t.timestamp_step_sec < 0.021);
        LOOM_CHECK(t.prev_context == 448);
        LOOM_CHECK(t.control_ids.size() == 2);
        LOOM_CHECK(t.language("en") == 50259);
        LOOM_CHECK(t.language("de") == 50261);
        LOOM_CHECK(t.task("translate") == 50358);
        // An unknown name is -1 rather than a plausible id: `transcribe` turns that into an error
        // naming what the model does have, which is the behaviour a caller who asserted a language
        // wants over a silent fallback to detection.
        LOOM_CHECK(t.language("qq") == -1);
        LOOM_CHECK(t.task("summarise") == -1);
        // Nothing to fall back to, and nothing that needs to.
        LOOM_CHECK(!t.legacy_spelling);
    }

    // ---- Codec tokens are their own modality, and the fold is asymmetric on purpose --------------
    //
    // ADR-020. This block and the `speech2text` check above are the two halves of one claim: an id
    // modality folds onto "text" when text is what it encodes (`token_ids`, `phoneme_ids`) and does
    // NOT when it encodes audio (`audio_codes`). Making the fold uniform in either direction breaks
    // one of these two, which is the whole reason both are here.
    {
        const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/contract_codec.gguf";
        auto model = loom::GgufModel::load(path, backend.get());
        LOOM_CHECK(model != nullptr);

        const loom::ModelContract c = loom::ModelContract::read(*model);
        LOOM_CHECK(c.declared());
        LOOM_CHECK(c.task == loom::task_names::AUDIO_CODEC);
        LOOM_CHECK(c.input_kind == loom::modality::AUDIO_CODES);
        LOOM_CHECK(c.output_kind == loom::modality::AUDIO);
        // NOT "text2speech", which is what `token_ids` here would have produced -- and which would
        // have had every host offer a text door on a model with no vocabulary.
        LOOM_CHECK(c.interface_name() == "codes2speech");
        LOOM_CHECK(c.sample_rate == 44100);
        // The three numbers a caller cannot build the input without.
        LOOM_CHECK(model->hparam_u32("codec.n_codebooks") == 9);
        LOOM_CHECK(model->hparam_u32("codec.codebook_size") == 1024);
    }

    // ---- A file exported before any of it existed -------------------------------------------------
    {
        const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/contract_legacy.gguf";
        auto model = loom::GgufModel::load(path, backend.get());
        LOOM_CHECK(model != nullptr);

        const loom::ModelContract c = loom::ModelContract::read(*model);
        // Not declared -- and saying so is the point. A host must not offer a task-shaped door on the
        // strength of a guess; this is what it checks before it does.
        LOOM_CHECK(!c.declared());
        LOOM_CHECK(c.task.empty());
        LOOM_CHECK(c.interface_name().empty());
        // The hparams that always existed are still read, under their own names. Renaming them would
        // have cost a re-export of every model to buy nothing.
        LOOM_CHECK(c.sample_rate == 16000);
        LOOM_CHECK(c.clip_samples == 480000);
        // No vocabulary in this file, so no front end can be claimed for it.
        LOOM_CHECK(c.text_frontend.empty());

        const loom::audio::AsrDecodeTable t =
            loom::audio::AsrDecodeTable::read(*model, nullptr, c);
        // With no declared table and no vocabulary to spell against, there is nothing to report but
        // the absence -- which `transcribe` reads as "this model has no timestamps" and degrades to
        // fixed window cuts, exactly as it did before the table existed.
        LOOM_CHECK(!t.timestamped());
        LOOM_CHECK(t.timestamp_first_id < 0);
        // `loom.n_text_ctx` is where the prev_tokens cap lived before `loom.asr.prev_context`.
        LOOM_CHECK(t.prev_context == 448);
        // No vocab was supplied, so the legacy path could not even be attempted -- the flag stays
        // false rather than promising a fallback that is not available.
        LOOM_CHECK(!t.legacy_spelling);
    }

    // NOT `return 0`. `LOOM_CHECK` only counts failures; this macro is what turns the count into an
    // exit code, and a test that ends any other way passes unconditionally -- which this one did, until
    // deliberately breaking an expectation failed to turn it red.
    LOOM_TEST_REPORT_AND_RETURN();
}
