// The model's own tokenizer, whichever vocabulary family its file carries (ADR-139).
//
// A GGUF names its family in `tokenizer.ggml.model`, and each family is its own class with no common
// base. Every host needs the same dispatch over them, and two hosts already had their own:
// loom-py's `binding.cpp` (where this class was moved from, unchanged) and `loom_cli`. The web binding
// would have been a third copy, so the dispatch now lives here, where a host includes it.
//
// **Every family is dispatched by NAME, and an unrecognised tag yields no tokenizer rather than an
// error.** A GGUF from a newer exporter, carrying a vocabulary this engine has not been taught, still
// loads: it is a missing feature, not a broken file. A tag it DOES know with the data missing still
// throws from the family's own `load`, on purpose: that file is malformed.
//
// **A model may carry none, and that is not a defect.** `load` returns null, and a host says so.
//
// **"ctc" only decodes.** A CTC model takes audio, so its table names the classes its head emits and
// there is no segmentation rule to invert. `encode` on one throws, saying so.
#pragma once

#include "loom/core/gguf_model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace loom {

class Vocab;
class BpeVocab;
class WordPieceVocab;
class ByteVocab;
class SupertonicTextVectorizer;
class PlamoVocab;
class PhonemeVocab;
class CtcVocab;
class F5Vocab;
class ChatterboxVocab;
class PocketTtsVocab;
class VoxCpmVocab;
class CosyVoice3Vocab;
class KaleTtsVocab;
class SopranoVocab;

class Tokenizer {
public:
    // Null when the file carries no vocabulary, or one whose tag this engine does not know.
    static std::unique_ptr<Tokenizer> load(const GgufModel& model);
    ~Tokenizer();

    // The file's `tokenizer.ggml.model` tag.
    const std::string& kind() const { return kind_; }
    size_t size() const;
    // `lang` is honoured only by a vocabulary that tags its input by language (supertonic), and any
    // other throws rather than dropping it: a caller asking for Korean must not quietly get English.
    std::vector<int32_t> encode(const std::string& text, const std::string& lang = std::string{}) const;
    // The bytes as the vocabulary spells them: a byte-level BPE token may end inside a UTF-8 character.
    std::string decode(const std::vector<int32_t>& ids) const;
    // The language `encode` uses when none is named; empty for every family but supertonic.
    std::string default_lang() const;

private:
    Tokenizer();
    bool valid() const;

    std::string kind_;
    std::unique_ptr<Vocab> spm_;
    std::unique_ptr<BpeVocab> bpe_;
    std::unique_ptr<WordPieceVocab> wordpiece_;
    std::unique_ptr<ByteVocab> byte_;
    std::unique_ptr<SupertonicTextVectorizer> supertonic_;
    std::unique_ptr<PlamoVocab> plamo_;
    std::unique_ptr<PhonemeVocab> phoneme_;
    std::unique_ptr<CtcVocab> ctc_;
    std::unique_ptr<F5Vocab> f5_;
    std::unique_ptr<ChatterboxVocab> chatterbox_;
    std::unique_ptr<PocketTtsVocab> pocket_tts_;
    std::unique_ptr<VoxCpmVocab> voxcpm_;
    std::unique_ptr<CosyVoice3Vocab> cosyvoice3_;
    std::unique_ptr<KaleTtsVocab> kale_tts_;
    std::unique_ptr<SopranoVocab> soprano_;
};

} // namespace loom
