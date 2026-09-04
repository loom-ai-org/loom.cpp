#include "loom/core/byte_vocab.h"
#include "loom/core/gguf_model.h"
#include "loom/loom_errors.h"

#include <algorithm>

namespace loom {

namespace {
// gguf's own `TokenType` values, the same two `BpeVocab::load` treats as added (3 CONTROL, a special
// marker; 4 USER_DEFINED, an added token that is not special).
constexpr int32_t kTokenTypeControl = 3;
constexpr int32_t kTokenTypeUserDefined = 4;
} // namespace

std::unique_ptr<ByteVocab> ByteVocab::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.ggml.model")) {
        return nullptr;
    }
    if (model.kv_str("tokenizer.ggml.model") != "byt5") {
        return nullptr; // not this vocab type -- caller should try BpeVocab/WordPieceVocab/Vocab instead
    }

    auto vocab = std::unique_ptr<ByteVocab>(new ByteVocab());
    vocab->tokens_ = model.kv_arr_str("tokenizer.ggml.tokens");
    vocab->pad_id_ = model.kv_i32("tokenizer.ggml.padding_token_id", 0);
    vocab->eos_id_ = model.kv_i32("tokenizer.ggml.eos_token_id", 1);
    vocab->unk_id_ = model.kv_i32("tokenizer.ggml.unknown_token_id", 2);
    // Both default to ByT5's own behaviour, so a file written before these KVs existed loads and
    // tokenizes byte-for-byte as it did. See the header for the two parameterisations covered.
    vocab->byte_offset_ = model.kv_i32("tokenizer.ggml.byte_offset", kDefaultByteOffset);
    vocab->add_eos_ = model.kv_bool("tokenizer.ggml.add_eos_token", true);
    if (vocab->byte_offset_ < 0) {
        throw LoadError("ByteVocab::load: tokenizer.ggml.byte_offset is " +
                        std::to_string(vocab->byte_offset_) + "; byte 0 cannot live at a negative id");
    }
    if (static_cast<size_t>(vocab->byte_offset_) + kByteRangeSize > vocab->tokens_.size()) {
        // The byte range has to FIT, or `encode` returns ids past the end of the vocabulary -- ids a
        // consumer will happily index a row with. Checked at load because it is answerable here and
        // because the failure it prevents is silent.
        throw LoadError("ByteVocab::load: byte_offset " + std::to_string(vocab->byte_offset_) +
                        " + 256 byte ids exceeds the " + std::to_string(vocab->tokens_.size()) +
                        "-token vocabulary; the byte range does not fit");
    }

    // The added set, exactly as `BpeVocab::load` builds it and from the same KV. Absent, this stays
    // empty and `encode` skips the scan entirely.
    if (model.has_kv("tokenizer.ggml.token_type")) {
        const std::vector<int32_t> token_type = model.kv_arr_i32("tokenizer.ggml.token_type");
        if (token_type.size() != vocab->tokens_.size()) {
            throw LoadError("ByteVocab::load: tokenizer.ggml.token_type has " +
                            std::to_string(token_type.size()) + " entries but there are " +
                            std::to_string(vocab->tokens_.size()) + " tokens -- the two arrays are "
                            "parallel by definition");
        }
        for (size_t i = 0; i < token_type.size(); ++i) {
            const int32_t type = token_type[i];
            if (type != kTokenTypeControl && type != kTokenTypeUserDefined) continue;
            const std::string& piece = vocab->tokens_[i];
            if (piece.empty()) continue;
            const auto id = static_cast<int32_t>(i);
            vocab->added_to_id_.emplace(piece, id);
            vocab->max_added_len_ = std::max(vocab->max_added_len_, piece.size());
            vocab->added_first_byte_[static_cast<unsigned char>(piece[0])] = true;
            // Only where it overlaps the byte range does the stored text have to win over the
            // arithmetic; outside it, `id_to_piece` already reads `tokens_`. Recording the overlap
            // rather than every added token keeps the two paths from disagreeing about ids the byte
            // range does not contain.
            if (id >= vocab->byte_offset_ && id < vocab->byte_offset_ + kByteRangeSize) {
                vocab->added_piece_.emplace(id, piece);
            }
        }
    }
    return vocab;
}

int32_t ByteVocab::added_token_at(const std::string& text, size_t pos, size_t* len) const {
    if (added_to_id_.empty() || !added_first_byte_[static_cast<unsigned char>(text[pos])]) return -1;
    // LONGEST match, the same rule `BpeVocab::added_token_at` applies and HF's own AddedVocabulary
    // resolves overlaps by. Bounded by `max_added_len_`, so a long document costs the same per
    // position as a short one.
    const size_t limit = std::min(max_added_len_, text.size() - pos);
    for (size_t n = limit; n >= 1; --n) {
        const auto it = added_to_id_.find(text.substr(pos, n));
        if (it != added_to_id_.end()) {
            *len = n;
            return it->second;
        }
    }
    return -1;
}

std::vector<int32_t> ByteVocab::encode(const std::string& text) const {
    std::vector<int32_t> ids;
    ids.reserve(text.size() + 1);
    for (size_t i = 0; i < text.size();) {
        size_t match_len = 0;
        const int32_t added = added_token_at(text, i, &match_len);
        if (added >= 0) {
            ids.push_back(added);
            i += match_len;
            continue;
        }
        ids.push_back(static_cast<int32_t>(static_cast<unsigned char>(text[i])) + byte_offset_);
        ++i;
    }
    if (add_eos_) {
        ids.push_back(eos_id_);
    }
    return ids;
}

std::string ByteVocab::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        out += id_to_piece(id);
    }
    return out;
}

std::string ByteVocab::id_to_piece(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) {
        throw LoadError("ByteVocab::id_to_piece: id " + std::to_string(id) + " out of range");
    }
    // An added token that falls inside the byte range wins: Dia's `[S1]` is id 1, which under its own
    // offset of 0 is also byte 0x01. The tokenizer it mirrors resolves that the same way -- the tag is
    // in `added_tokens_decoder`, and a literal 0x01 byte is simply not reachable through it.
    const auto added = added_piece_.find(id);
    if (added != added_piece_.end()) {
        return added->second;
    }
    if (id >= byte_offset_ && id < byte_offset_ + kByteRangeSize) {
        return std::string(1, static_cast<char>(id - byte_offset_));
    }
    return tokens_[static_cast<size_t>(id)];
}

} // namespace loom
