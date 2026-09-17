#include "loom/core/ctc_vocab.h"
#include "loom/core/gguf_model.h"
#include "loom/loom_errors.h"

#include <string>

namespace loom {

std::unique_ptr<CtcVocab> CtcVocab::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.ggml.model")) {
        return nullptr;
    }
    if (model.kv_str("tokenizer.ggml.model") != "ctc") {
        return nullptr; // not this vocab type -- caller should try BpeVocab/Vocab/... instead
    }

    auto vocab = std::unique_ptr<CtcVocab>(new CtcVocab());
    vocab->tokens_ = model.kv_arr_str("tokenizer.ggml.tokens");
    if (vocab->tokens_.empty()) {
        throw LoadError("CtcVocab::load: tokenizer.ggml.model is 'ctc' but tokenizer.ggml.tokens is "
                        "missing or empty; the table IS the vocabulary here");
    }
    vocab->blank_id_ = model.kv_i32("tokenizer.ggml.padding_token_id", 0);
    vocab->unk_id_ = model.kv_i32("tokenizer.ggml.unknown_token_id", -1);
    vocab->word_delimiter_id_ = model.kv_i32("tokenizer.ggml.word_delimiter_id", -1);

    // Both are row numbers of the same table the head emits, so an out-of-range one is a file that
    // cannot be decoded correctly rather than one that decodes oddly. Checked here because it is
    // answerable here and because the alternative is a throw from `decode` on some later transcript.
    const auto rows = static_cast<int32_t>(vocab->tokens_.size());
    if (vocab->blank_id_ < 0 || vocab->blank_id_ >= rows) {
        throw LoadError("CtcVocab::load: tokenizer.ggml.padding_token_id is " +
                        std::to_string(vocab->blank_id_) + ", which is not a row of the " +
                        std::to_string(rows) + "-piece vocabulary");
    }
    if (vocab->word_delimiter_id_ >= rows) {
        throw LoadError("CtcVocab::load: tokenizer.ggml.word_delimiter_id is " +
                        std::to_string(vocab->word_delimiter_id_) + ", which is not a row of the " +
                        std::to_string(rows) + "-piece vocabulary");
    }
    return vocab;
}

std::string CtcVocab::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        // The delimiter is checked before the lookup, not rewritten in `tokens_` at load: `id_to_piece`
        // must keep answering with what the checkpoint's own vocab.json says, which is the whole reason
        // this is a vocabulary tag of its own rather than a "t5" file with the piece rewritten to "▁".
        if (id == word_delimiter_id_) {
            out += ' ';
            continue;
        }
        out += id_to_piece(id);
    }
    return out;
}

const std::string& CtcVocab::id_to_piece(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) {
        throw LoadError("CtcVocab::id_to_piece: id " + std::to_string(id) + " out of range (" +
                        std::to_string(tokens_.size()) + " pieces)");
    }
    return tokens_[static_cast<size_t>(id)];
}

} // namespace loom
