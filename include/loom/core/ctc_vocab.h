#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace loom {

class GgufModel;

// A CTC character vocabulary, "tokenizer.ggml.model"=="ctc" -- HF's `Wav2Vec2CTCTokenizer` scheme, which
// wav2vec 2.0, HuBERT and data2vec-audio all share (EXPORT-ROADMAP.md's family 4).
//
// **IT IS DECODE-ONLY, AND THAT IS THE SCHEME RATHER THAN A FIRST PASS.** Every other vocabulary class
// here exists because some text had to become ids; this one exists because ids have to become text. The
// ids come out of a CTC head's per-frame argmax, and the model has no text input at all -- so there is
// nothing for an `encode` to be the inverse of, and adding one would mean inventing a segmentation
// rule that no file in this family states. `Vocab`, `BpeVocab`, `WordPieceVocab` and `ByteVocab` each
// carry a real algorithm read off the checkpoint; this carries a table.
//
// Decoding is `Wav2Vec2CTCTokenizer.convert_tokens_to_string` minus the parts the DRIVER has already
// done: concatenate the pieces, writing a space wherever the word-delimiter piece appears. The CTC
// collapse -- drop consecutive duplicates, drop the blank -- happens in the exported driver's own
// epilogue before any of these ids exist (`driver_components.CtcGreedyEpilogue`), which is why it is
// not repeated here.
//
// **The word delimiter is an ID, not a spelling.** The two English checkpoints spell it "|" and the
// multilingual one spells it as a literal space, and both are ordinary rows of the same table -- so the
// file names the row and this reads the spelling off `tokens_` like everything else. A file that names
// none decodes by plain concatenation, which is what a vocabulary with no word boundary means.
//
// **The blank is `tokenizer.ggml.padding_token_id` and it is NOT the last class.** NeMo's CTC
// convention (family 1) puts the blank at `num_classes - 1`; HF's puts it at the tokenizer's `pad_token`,
// which is row 0 in every checkpoint here. Nothing downstream of this class reads it -- the driver holds
// its own copy, baked at export time -- so it is exposed for the same reason `ByteVocab::pad_id()` is:
// a host inspecting the file should be able to ask.
class CtcVocab {
public:
    // Returns nullptr if `model` has no "tokenizer.ggml.model" KV, or it is present but not "ctc" (i.e.
    // this model uses one of the other schemas -- callers should try those too).
    static std::unique_ptr<CtcVocab> load(const GgufModel& model);

    // Concatenates each id's piece, substituting a single space for the word-delimiter id. Ids outside
    // the vocabulary throw, matching every other class here: an id a CTC head cannot have produced is a
    // caller error, not something to render as an empty string.
    std::string decode(const std::vector<int32_t>& ids) const;

    const std::string& id_to_piece(int32_t id) const;
    size_t size() const { return tokens_.size(); }
    int32_t blank_id() const { return blank_id_; }
    int32_t unk_id() const { return unk_id_; }
    int32_t word_delimiter_id() const { return word_delimiter_id_; }

    CtcVocab(const CtcVocab&) = delete;
    CtcVocab& operator=(const CtcVocab&) = delete;

private:
    CtcVocab() = default;

    // One array indexed by id, the convention every vocab class in this schema shares. Its length is
    // also the CTC head's own row count, which is what makes an out-of-range id checkable at all.
    std::vector<std::string> tokens_;
    int32_t blank_id_ = 0;
    int32_t unk_id_ = -1;
    // -1 for a file that declares none: "this vocabulary has no word boundary", which decodes by plain
    // concatenation. A default of 0 would silently turn the blank into a space.
    int32_t word_delimiter_id_ = -1;
};

} // namespace loom
