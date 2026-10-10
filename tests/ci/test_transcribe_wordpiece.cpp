// `transcribe` on a CTC recognizer whose vocabulary is WordPiece ("bert"), which is what NeMo's oldest
// checkpoints carry (`tokenizer.type: wpe`; Citrinet-256-ls). Before this, the dispatch tried BPE, CTC,
// FunASR and then SentencePiece, and the SentencePiece loader threw on a "bert" file.
//
// Two claims. The pieces compose as WordPiece does -- a word-initial piece opens a word, a continuation
// glues on -- and the specials are dropped, as NeMo's `ids_to_text` drops them: Citrinet-256's CTC head
// frames every utterance in `[CLS] ... [SEP]`, so without that every transcript would carry both.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <string>
#include <vector>

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/wordpiece_asr.gguf";
    auto model = loom::GgufModel::load(path, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, backend.get());

    const std::vector<float> waveform(16000, 0.25f);
    loom::audio::TranscribeOptions options;
    const loom::audio::Transcription r = loom::audio::transcribe(session.bridge(), *model, waveform, options);
    // The driver returned [CLS] ▁hello ▁wor ld [UNK] [SEP].
    LOOM_CHECK(r.text == "hello world");
    return 0;
}
