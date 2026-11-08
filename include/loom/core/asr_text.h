#pragma once
// An ASR file's token ids to text: whichever of the five vocabulary schemas the file carries, and the
// control ids a transcript must not print.
//
// Shared by `transcribe` and `TranscriptionStream` because the choice of vocabulary and the set of
// dropped ids decide what a transcript says, and the one-pass and live doors must not drift on either.
#include "loom/core/bpe_vocab.h"
#include "loom/core/ctc_vocab.h"
#include "loom/core/funasr_vocab.h"
#include "loom/core/gguf_model.h"
#include "loom/core/vocab.h"
#include "loom/core/wordpiece_vocab.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace loom {
namespace audio {

class AsrText {
public:
    // Throws loom::LoadError when the file carries no vocabulary this can read.
    explicit AsrText(const GgufModel& model);

    // The byte-level BPE vocabulary, or null: the only one that resolves a token by TEXT, which the
    // decode table's legacy Whisper spellings need.
    const BpeVocab* bpe() const { return bpe_.get(); }

    // Adds ids the decode table declares as control tokens (`AsrDecodeTable::control_ids`).
    void add_control_ids(const std::vector<int32_t>& ids);
    bool is_control(int32_t id) const;
    // `ids` with every control id removed.
    std::vector<int32_t> text_ids(const std::vector<int32_t>& ids) const;
    // The text of `ids`, control ids dropped.
    std::string detokenize(const std::vector<int32_t>& ids) const;

private:
    std::unique_ptr<BpeVocab> bpe_;
    std::unique_ptr<CtcVocab> ctc_;
    std::unique_ptr<FunasrVocab> funasr_;
    std::unique_ptr<WordPieceVocab> wp_;
    std::unique_ptr<Vocab> spm_;
    std::vector<int32_t> control_ids_;
};

} // namespace audio
} // namespace loom
