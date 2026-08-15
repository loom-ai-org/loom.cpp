#pragma once

// The phoneme vocabulary a TTS checkpoint carries: symbol -> id, plus the assembly around the lookup.
//
// WHY THIS IS A VOCABULARY AND NOT A PHONEMIZER. Four of the five TTS families consume phoneme ids, and
// their GGUFs carried no way to produce one -- `model.tokenizer` was None and `infer` took raw integers
// no caller could obtain. That reads like the missing piece is grapheme-to-phoneme conversion, and it
// is not: the symbol TABLE was sitting in the checkpoint the whole time (Piper's `phoneme_id_map` is
// 159 entries) and simply was not exported. Exporting it is what splits BACKLOG.md Task #79 in two --
// this half needs no phonemizer, no new dependency and raises no licence question, because a lookup
// table is data.
//
// What stays outside: grapheme -> phoneme, which is a property of the LANGUAGE rather than of any
// checkpoint, and therefore of no GGUF.
//
// THE ASSEMBLY IS PART OF THE CONVERSION AND NOT PART OF THE TABLE. Piper builds
// `[BOS, p1, blank, p2, blank, ..., pn, blank, EOS]` -- a blank between every phoneme, none right after
// BOS. A host that only looked symbols up would produce ids the model was not trained on, so the
// convention is declared by the export (`tokenizer.ggml.phoneme.*`) and applied here. That is the same
// arrangement `SupertonicTextVectorizer` has for its `<lang>` wrap, one modality over, and it is why
// both are vocabularies rather than lookup tables with instructions attached.

#include "loom/core/gguf_model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {

class PhonemeVocab {
public:
    // Null when this file carries no phoneme vocabulary, which is every model but the phoneme-input TTS
    // families. Absence is the answer for them, not an error -- same contract as every other
    // `Vocab::load` in this tree.
    static std::unique_ptr<PhonemeVocab> load(const GgufModel& model);

    // Phoneme symbols to model-ready ids: longest-match over the table, then the declared assembly.
    //
    // LONGEST MATCH, because IPA symbols are not one codepoint each -- `t͡ʃ` and `aɪ` are single entries
    // in a real table, and a per-codepoint walk would split them into pieces the model has never seen.
    // Scanning longest-first is what makes a table containing both `a` and `aɪ` unambiguous.
    //
    // An unknown symbol is SKIPPED rather than raising, and that is the one place this differs from the
    // text vocabularies. A phonemizer producing a symbol outside the checkpoint's inventory is expected
    // rather than exceptional -- the rule-based engines emit a superset of what any one model was
    // trained on -- so refusing the whole utterance over one diacritic would make every long sentence
    // fail. `unknown` reports how many were dropped, for a caller that wants to notice.
    std::vector<int32_t> encode(const std::string& phonemes, size_t* unknown = nullptr) const;

    // Ids back to the symbols they name, with the assembly ids omitted -- what a caller wants when
    // inspecting what a model was actually handed.
    std::string decode(const std::vector<int32_t>& ids) const;

    size_t size() const { return tokens_.size(); }
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t blank_id() const { return blank_id_; }
    bool interleave_blank() const { return interleave_blank_; }

    // The symbol for one id, or "" for an id outside the table.
    const std::string& piece(int32_t id) const;

private:
    PhonemeVocab() = default;

    std::vector<std::string> tokens_;              // indexed by id
    std::unordered_map<std::string, int32_t> ids_; // symbol -> id, for the longest-match scan
    size_t longest_ = 1;                           // bytes in the longest symbol, the scan's upper bound
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t blank_id_ = -1;
    bool interleave_blank_ = false;
};

} // namespace loom
