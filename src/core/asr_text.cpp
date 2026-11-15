#include "loom/core/asr_text.h"

#include "loom/core/audio_window.h"
#include "loom/loom_errors.h"

#include <algorithm>

namespace loom {
namespace audio {

AsrText::AsrText(const GgufModel& model) {
    // Five vocab schemas reach this path, so all are tried: NeMo's checkpoints carry SentencePiece
    // ("llama"/"t5" -> loom::Vocab), Whisper's carries GPT-2 byte-level BPE ("gpt2" -> BpeVocab),
    // family 4's carry the CTC character table ("ctc" -> CtcVocab), family 5's FunASR's table, and
    // NeMo's older WordPiece checkpoints (Citrinet-256-ls, `tokenizer.type: wpe`) a BERT vocabulary
    // ("bert" -> WordPieceVocab), written already in the phantom-space shape its decode reads.
    // Only the BPE one can resolve a token by TEXT, which is what timestamps need.
    //
    // **`Vocab::load` LAST, and the order is load-bearing**: the other two return nullptr for a schema
    // that is not their own, while `Vocab::load` THROWS on one -- so asking the SentencePiece loader
    // about a gpt2 or ctc file kills the run before the right loader is ever reached.
    bpe_ = BpeVocab::load(model);
    if (!bpe_) ctc_ = CtcVocab::load(model);
    // Family 5's, whose table is flat like family 4's but whose pieces compose differently -- `@@`
    // marks a continuation, CJK eats the space before it, and letter runs collapse. See FunasrVocab.
    if (!bpe_ && !ctc_) funasr_ = FunasrVocab::load(model);
    if (!bpe_ && !ctc_ && !funasr_) wp_ = WordPieceVocab::load(model);
    if (!bpe_ && !ctc_ && !funasr_ && !wp_) spm_ = Vocab::load(model);
    if (!spm_ && !bpe_ && !ctc_ && !funasr_ && !wp_) {
        throw LoadError("transcribe: model has no tokenizer vocab (tokenizer.ggml.model KV missing)");
    }

    // Control tokens a transcript must not contain, dropped before detokenizing because each is a real
    // vocabulary piece that otherwise prints as its literal spelling: the end-of-sequence token, which
    // the driver returns deliberately, and `<|notimestamps|>`, which is the model stating something
    // about the decode rather than a word that was spoken. Timestamp markers are NOT dropped.
    //
    // EOS is added here rather than read from the table because every vocabulary family names it the
    // same way, in `tokenizer.ggml.eos_token_id` -- it needs no per-task declaration to be found.
    control_ids_.push_back(default_eos_token(model));
    // A WordPiece vocabulary's specials are real CTC classes and NeMo's `ids_to_text` drops them
    // (`remove_special_tokens=True`), so a stray `[UNK]` never reaches the transcript there either.
    if (wp_) {
        for (int32_t id : {wp_->pad_id(), wp_->unk_id(), wp_->cls_id(), wp_->sep_id(), wp_->mask_id()}) {
            if (id >= 0) control_ids_.push_back(id);
        }
    }
}

void AsrText::add_control_ids(const std::vector<int32_t>& ids) {
    control_ids_.insert(control_ids_.end(), ids.begin(), ids.end());
}

bool AsrText::is_control(int32_t id) const {
    return std::find(control_ids_.begin(), control_ids_.end(), id) != control_ids_.end();
}

std::vector<int32_t> AsrText::text_ids(const std::vector<int32_t>& ids) const {
    std::vector<int32_t> kept;
    kept.reserve(ids.size());
    for (int32_t id : ids) {
        if (!is_control(id)) kept.push_back(id);
    }
    return kept;
}

std::string AsrText::detokenize(const std::vector<int32_t>& ids) const {
    const std::vector<int32_t> kept = text_ids(ids);
    if (ctc_) return ctc_->decode(kept);
    if (funasr_) return funasr_->decode(kept);
    if (wp_) return wp_->decode(kept);
    return spm_ ? spm_->decode(kept) : bpe_->decode(kept);
}

} // namespace audio
} // namespace loom
