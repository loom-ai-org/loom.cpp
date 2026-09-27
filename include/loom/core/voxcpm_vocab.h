#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {

class GgufModel;

// VoxCPM2's text front end, "tokenizer.ggml.model"=="voxcpm2" (EXPORT-ROADMAP.md's family 9).
//
// **A character-level BPE merged by RANK, with byte fallback -- none of the engine's other schemes.**
// `tokenizer.json` is a `tokenizers` BPE with no pre-tokenizer (the whole text is one word), a
// normalizer that prepends `▁` and spells every space `▁`, and `<0xNN>` pieces for characters the table
// lacks. "llama" merges by SCORE and refuses byte fallback; "gpt2" is byte-level; "chatterbox" has a
// pre-tokenizer and `[UNK]`. So this is its own tag (ADR-033: a tag answers "which scheme").
//
// `encode` is the reference's whole text path (`VoxCPM._generate` -> `mask_multichar_chinese_tokens`),
// in its order:
//   1. `text.replace("\n", " ")` and `re.sub(r"\s+", " ", text)`: every run of whitespace becomes one
//      space. Nothing is stripped.
//   2. `tokenizers`' encode: split on the ADDED tokens (leftmost-longest, their literal spelling), and
//      for each remaining run: prepend `▁`, replace every space with `▁`, split into characters, map a
//      character the table lacks to its UTF-8 bytes' `<0xNN>` pieces, and merge -- lowest rank first,
//      leftmost first among equals -- until no listed merge applies.
//   3. `CharTokenizerWrapper`: every piece whose `▁`-stripped text is a multi-character Chinese piece is
//      replaced by its characters' ids (the `▁` itself is dropped, as the reference drops it).
//
// **The normalizer's strings, the added tokens and the split table are the FILE's**
// (`tokenizer.ggml.voxcpm2.*`), written by the exporter from the reference. No BOS is added: the
// reference calls `tokenize`, not `encode`, and the driver appends `<|audio_start|>` itself.
class VoxCpmVocab {
public:
    // Returns nullptr if `model` has no "tokenizer.ggml.model" KV, or it is present but not "voxcpm2".
    // Throws `LoadError` if the tag is present and a required key is not.
    static std::unique_ptr<VoxCpmVocab> load(const GgufModel& model);

    // Steps 1-3 above. `*fallback` (when non-null) receives how many characters fell back to bytes --
    // exact, and still worth a host's notice: the model was trained on few of them.
    std::vector<int32_t> encode(const std::string& text, size_t* fallback = nullptr) const;

    // Step 1 alone, so a host (and the tests) can see the text the model is asked to say.
    std::string normalize(const std::string& text) const;

    // `tokenizers`' decode: `▁` back to a space, byte pieces re-assembled into UTF-8, the leading space
    // the normalizer added stripped once.
    std::string decode(const std::vector<int32_t>& ids) const;

    const std::string& id_to_piece(int32_t id) const;
    size_t size() const { return tokens_.size(); }
    int32_t unk_id() const { return unk_id_; }

    VoxCpmVocab(const VoxCpmVocab&) = delete;
    VoxCpmVocab& operator=(const VoxCpmVocab&) = delete;

private:
    VoxCpmVocab() = default;

    void encode_segment(const std::string& segment, std::vector<int32_t>& ids, size_t* fallback) const;
    int32_t added_token_at(const std::string& text, size_t pos, size_t* len) const;

    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int32_t> piece_to_id_;
    // "left right" -> rank, the order `merges` lists them in (lower merges first).
    std::unordered_map<std::string, int32_t> merge_rank_;
    std::unordered_map<std::string, int32_t> added_;
    size_t longest_added_ = 0;
    // `<0xNN>`'s id per byte.
    std::vector<int32_t> byte_ids_;
    // id -> the ids it splits into (step 3).
    std::unordered_map<int32_t, std::vector<int32_t>> split_;
    std::string prepend_;
    std::string space_;
    int32_t unk_id_ = 0;
};

} // namespace loom
