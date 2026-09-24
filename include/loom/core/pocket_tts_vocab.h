#pragma once

#include "loom/core/vocab.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace loom {

class GgufModel;

// Pocket-TTS's text front end, "tokenizer.ggml.model"=="pocket_tts" (family 9's fifth leaf).
//
// **The vocabulary is an ordinary SentencePiece unigram** (with byte fallback), held as a `Vocab`.
// What makes this its own tag (ADR-033: a tag answers "which scheme") is the reference's text path
// around it, which `TTSModel.generate_audio_stream` runs before any id reaches the model:
//
//   1. `prepare_text_prompt` on the whole text: strip, a replacement table applied pair by pair
//      (newlines to spaces, one pass of "  " -> " "), the first character upper-cased unless it
//      already is (`str.upper()`, from the file's table), and terminal punctuation ensured -- a text
//      ending in a weak mark (",", ":", a dash...) has it replaced by a full stop, closing quotes and
//      brackets kept after it; one ending in neither gets a full stop appended.
//   2. `split_into_best_sentences`: tokenize, cut after every run of sentence-ending ids (except a
//      period between two digits, which the reference detects on DECODED text), sub-split any
//      sentence over the chunk budget on clause marks, and regroup sentences greedily into chunks of
//      at most that many tokens.
//   3. Each chunk through step 1 again, then tokenized.
//
// `encode` returns every chunk's ids, each chunk OPENED by a header id: `chunk_header_short` when the
// reference's tail guess for it is the short-text one (`prepare_text_prompt`: at most four words ->
// 3 frames after EOS, else 1), `chunk_header_long` otherwise. The driver generates each chunk from a
// fresh copy of the voice, as the reference does, so this is the one place the chunking can be exact:
// steps 2 and 3 need decoded text, and so does the word count (`len(text.split())` BEFORE the terminal
// punctuation is fixed), and the driver never sees text (ADR-044).
//
// **Every constant is the file's** (`tokenizer.ggml.pocket_tts.*`), written by the exporter from the
// reference's own module and config -- ADR-041's rule: the shape is code, the rules are data. That
// includes the case table (Python's `c.upper()` for every codepoint whose `c.isupper()` is false and
// whose upper case differs) and Python's `str.isdigit()` set for the decimal rule.
class PocketTtsVocab {
public:
    // Returns nullptr if `model` has no "tokenizer.ggml.model" KV, or it is present but not
    // "pocket_tts". Throws `LoadError` if the tag is present and a required key is not.
    static std::unique_ptr<PocketTtsVocab> load(const GgufModel& model);

    // Steps 1-3 above: every chunk's ids, each opened by its header. Throws `Error` on a text that is
    // empty after stripping, as the reference raises.
    std::vector<int32_t> encode(const std::string& text) const;

    // Step 2's output as text, exposed so a host (and the tests) can see what each generation will be
    // asked to say.
    std::vector<std::string> chunks(const std::string& text) const;

    // Step 1 alone (`prepare_text_prompt`'s text). `*n_words` (when non-null) receives its word count,
    // `len(text.split())` after the replacements and before capitalisation and terminal punctuation --
    // the number the reference's tail guess is taken from.
    std::string prepare(const std::string& text, size_t* n_words = nullptr) const;

    // SentencePiece's decode, headers dropped.
    std::string decode(const std::vector<int32_t>& ids) const;

    const std::string& id_to_piece(int32_t id) const { return vocab_->id_to_piece(id); }
    size_t size() const { return vocab_->size(); }
    // The header opening a chunk of `n_words` words, and whether an id is either header.
    int32_t chunk_header(size_t n_words) const {
        return n_words <= short_chunk_max_words_ ? header_short_ : header_long_;
    }
    bool is_chunk_header(int32_t id) const { return id == header_short_ || id == header_long_; }

    PocketTtsVocab(const PocketTtsVocab&) = delete;
    PocketTtsVocab& operator=(const PocketTtsVocab&) = delete;

private:
    PocketTtsVocab() = default;

    // `_find_boundary_indices`: [0, cut..., len(ids)].
    std::vector<size_t> boundaries(const std::vector<int32_t>& ids, const std::unordered_set<int32_t>& marks,
                                   bool skip_decimal_periods) const;
    bool decimal_period_at(const std::vector<int32_t>& ids, size_t start) const;
    std::string terminate(const std::string& text) const;

    std::unique_ptr<Vocab> vocab_;
    std::vector<std::string> replace_from_;
    std::vector<std::string> replace_to_;
    std::unordered_map<char32_t, std::string> upper_;
    std::unordered_set<char32_t> terminal_;
    std::unordered_set<char32_t> weak_;
    std::unordered_set<char32_t> closers_;
    std::unordered_set<char32_t> digits_;
    std::string full_stop_;
    std::unordered_set<int32_t> sentence_end_ids_;
    std::unordered_set<int32_t> clause_end_ids_;
    size_t max_tokens_per_chunk_ = 0;
    int32_t header_short_ = -1;
    int32_t header_long_ = -1;
    size_t short_chunk_max_words_ = 0;
    bool capitalize_first_letter_ = true;
    bool append_terminal_punctuation_ = true;
};

} // namespace loom
