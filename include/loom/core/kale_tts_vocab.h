#pragma once

// `tokenizer.ggml.model == "kale_tts"` (ADR-138): kale-tts-1-tiny's own text front end. `encode(text, lang)` is
// upstream's whole text path (`KaleTTS.inputs`, frontend/ + js/kale-tts.mjs) -- the terms table, dizione's text
// rules, sentences, the G2P (lexicon, overrides, homographs, the GRU guesser; the British files and rules for
// `en-GB`), Kokoro's symbols -- and packs every sentence's model inputs into one id stream the file's driver
// reads (kale_tts_driver/00_kale.lua):
//
//   n_records, then per record: n, ids[n], widx[n], n_words, n_sub, sub[n_sub], sub_word[n_sub]
//
// `ids` are the phoneme ids without Kokoro's 0 pads, `widx` each one's word in the sentence, `sub` the Ettin
// sub-tokens of the previous sentence and this one (`[CLS] ... [SEP]`) and `sub_word` each sub-token's word, -1
// for the previous sentence and the two specials. A sentence past 500 phoneme characters is cut at a space into
// several records, each with the whole sentence's Ettin inputs, as upstream's `_parts` cuts it.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace loom {

class GgufModel;
class BpeVocab;

class KaleTtsVocab {
public:
    // nullptr unless `tokenizer.ggml.model` is "kale_tts"; throws `LoadError` when it is and a key is missing.
    static std::unique_ptr<KaleTtsVocab> load(const GgufModel& model);
    ~KaleTtsVocab();

    // `lang`: "en-GB" (any case, `_` or `-`) reads the British front end; anything else, or empty, American.
    std::vector<int32_t> encode(const std::string& text, const std::string& lang = std::string{}) const;

    // The front end's stages, for hosts and tests: upstream's `sentences(text)` and `ipa(sentence)`.
    std::vector<std::string> sentences(const std::string& text, const std::string& lang = std::string{}) const;
    std::string ipa(const std::string& sentence, const std::string& lang = std::string{}) const;

    size_t size() const;
    std::string decode(const std::vector<int32_t>& ids) const;

    KaleTtsVocab(const KaleTtsVocab&) = delete;
    KaleTtsVocab& operator=(const KaleTtsVocab&) = delete;

private:
    KaleTtsVocab();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace loom
