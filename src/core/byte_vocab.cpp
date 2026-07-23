#include "loom/core/byte_vocab.h"
#include "loom/core/gguf_model.h"
#include "loom/loom_errors.h"

namespace loom {

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
    return vocab;
}

std::vector<int32_t> ByteVocab::encode(const std::string& text) const {
    std::vector<int32_t> ids;
    ids.reserve(text.size() + 1);
    for (unsigned char b : text) {
        ids.push_back(static_cast<int32_t>(b) + kByteOffset);
    }
    ids.push_back(eos_id_);
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
    if (id >= kByteOffset && id < kByteOffset + kByteRangeSize) {
        return std::string(1, static_cast<char>(id - kByteOffset));
    }
    return tokens_[static_cast<size_t>(id)];
}

} // namespace loom
