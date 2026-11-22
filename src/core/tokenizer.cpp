#include "loom/core/tokenizer.h"

#include "loom/core/bpe_vocab.h"
#include "loom/core/byte_vocab.h"
#include "loom/core/chatterbox_vocab.h"
#include "loom/core/cosyvoice3_vocab.h"
#include "loom/core/ctc_vocab.h"
#include "loom/core/f5_vocab.h"
#include "loom/core/kale_tts_vocab.h"
#include "loom/core/phoneme_vocab.h"
#include "loom/core/plamo_vocab.h"
#include "loom/core/pocket_tts_vocab.h"
#include "loom/core/soprano_vocab.h"
#include "loom/core/supertonic_text_vectorizer.h"
#include "loom/core/vocab.h"
#include "loom/core/voxcpm_vocab.h"
#include "loom/core/wordpiece_vocab.h"
#include "loom/loom_errors.h"

namespace loom {

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;

std::unique_ptr<Tokenizer> Tokenizer::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.ggml.model")) return nullptr;
    std::unique_ptr<Tokenizer> t(new Tokenizer());
    t->kind_ = model.kv_str("tokenizer.ggml.model");
    const std::string& k = t->kind_;
    // "gpt2" by name before the SentencePiece branch: `Vocab::load` throws on a gpt2 file, where
    // `BpeVocab::load` reads it (the ordering `loom_cli` learned first).
    if (k == "bert") t->wordpiece_ = WordPieceVocab::load(model);
    else if (k == "byt5") t->byte_ = ByteVocab::load(model);
    else if (k == "gpt2") t->bpe_ = BpeVocab::load(model);
    // The phoneme-input TTS families: IPA in, model-ready ids out, assembly included.
    else if (k == "phonemes") t->phoneme_ = PhonemeVocab::load(model);
    else if (k == "ctc") t->ctc_ = CtcVocab::load(model);
    else if (k == "supertonic") t->supertonic_ = SupertonicTextVectorizer::load(model);
    // A character table: it raises on CJK by name rather than mapping a sentence to the unknown id.
    else if (k == "f5") t->f5_ = F5Vocab::load(model);
    // Each of these `encode`s is its reference's whole text path (normalisation, sentence chunks).
    else if (k == "chatterbox") t->chatterbox_ = ChatterboxVocab::load(model);
    else if (k == "pocket_tts") t->pocket_tts_ = PocketTtsVocab::load(model);
    else if (k == "voxcpm2") t->voxcpm_ = VoxCpmVocab::load(model);
    else if (k == "cosyvoice3") t->cosyvoice3_ = CosyVoice3Vocab::load(model);
    // kale-tts-1-tiny: its JavaScript front end, the packed records its driver reads (ADR-142).
    else if (k == "kale_tts") t->kale_tts_ = KaleTtsVocab::load(model);
    else if (k == "soprano") t->soprano_ = SopranoVocab::load(model);
    else if (k == "plamo") t->plamo_ = PlamoVocab::load(model);
    else if (k == "llama" || k == "t5") t->spm_ = Vocab::load(model);
    if (!t->valid()) return nullptr;
    return t;
}

bool Tokenizer::valid() const {
    return spm_ || bpe_ || wordpiece_ || byte_ || supertonic_ || phoneme_ || ctc_ || f5_ || chatterbox_ ||
           pocket_tts_ || voxcpm_ || cosyvoice3_ || soprano_ || plamo_ || kale_tts_;
}

size_t Tokenizer::size() const {
    if (spm_) return spm_->size();
    if (bpe_) return bpe_->size();
    if (wordpiece_) return wordpiece_->size();
    if (byte_) return byte_->size();
    if (phoneme_) return phoneme_->size();
    if (ctc_) return ctc_->size();
    if (f5_) return f5_->size();
    if (chatterbox_) return chatterbox_->size();
    if (pocket_tts_) return pocket_tts_->size();
    if (voxcpm_) return voxcpm_->size();
    if (cosyvoice3_) return cosyvoice3_->size();
    if (kale_tts_) return kale_tts_->size();
    if (soprano_) return soprano_->size();
    if (plamo_) return plamo_->size();
    // n_tokens(), not vocab_size(): the latter is the 65536-entry BMP lookup table.
    return supertonic_->n_tokens();
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, const std::string& lang) const {
    if (kale_tts_) return kale_tts_->encode(text, lang);
    if (!lang.empty() && !supertonic_) {
        throw SchemaError("tokenizer kind '" + kind_ + "' takes no language argument; only a vocabulary "
                          "that tags its input by language (supertonic) does");
    }
    if (ctc_) {
        throw SchemaError("tokenizer kind 'ctc' decodes only: a CTC model takes audio, so its vocabulary is "
                          "the table its head's classes are named by and there is no segmentation rule to "
                          "invert. Use decode().");
    }
    if (spm_) return spm_->encode(text);
    if (bpe_) return bpe_->encode(text);
    if (wordpiece_) return wordpiece_->encode(text);
    if (byte_) return byte_->encode(text);
    if (phoneme_) return phoneme_->encode(text);
    // TABLE ids: the driver adds the filler-token offset the embedding reserves row 0 for.
    if (f5_) return f5_->encode(text);
    // Without the start/stop ids, which the driver adds.
    if (chatterbox_) return chatterbox_->encode(text);
    if (pocket_tts_) return pocket_tts_->encode(text);
    if (voxcpm_) return voxcpm_->encode(text);
    if (cosyvoice3_) return cosyvoice3_->encode(text);
    if (soprano_) return soprano_->encode(text);
    if (plamo_) return plamo_->encode(text);
    return supertonic_->tokenize(text, lang);  // empty lang: the file's own default
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    if (spm_) return spm_->decode(ids);
    if (bpe_) return bpe_->decode(ids);
    if (wordpiece_) return wordpiece_->decode(ids);
    if (byte_) return byte_->decode(ids);
    if (phoneme_) return phoneme_->decode(ids);
    if (ctc_) return ctc_->decode(ids);
    if (f5_) return f5_->decode(ids);
    if (chatterbox_) return chatterbox_->decode(ids);
    if (pocket_tts_) return pocket_tts_->decode(ids);
    if (voxcpm_) return voxcpm_->decode(ids);
    if (cosyvoice3_) return cosyvoice3_->decode(ids);
    if (kale_tts_) return kale_tts_->decode(ids);
    if (soprano_) return soprano_->decode(ids);
    if (plamo_) return plamo_->decode(ids);
    return supertonic_->detokenize(ids);
}

std::string Tokenizer::default_lang() const { return supertonic_ ? supertonic_->default_lang() : std::string{}; }

} // namespace loom
