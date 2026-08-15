#include "loom/core/phoneme_vocab.h"

#include "loom/loom_errors.h"

namespace loom {
namespace {
const std::string kEmpty;
} // namespace

std::unique_ptr<PhonemeVocab> PhonemeVocab::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.ggml.model") || model.kv_str("tokenizer.ggml.model") != "phonemes") {
        return nullptr;
    }
    if (!model.has_kv("tokenizer.ggml.tokens")) {
        // The tag is present and the data is not: that file is malformed rather than merely new, which
        // is the same distinction `Tokenizer::load` draws for every other family.
        throw LoadError("tokenizer.ggml.model is 'phonemes' but tokenizer.ggml.tokens is missing");
    }

    std::unique_ptr<PhonemeVocab> vocab(new PhonemeVocab());
    vocab->tokens_ = model.kv_arr_str("tokenizer.ggml.tokens");
    for (size_t id = 0; id < vocab->tokens_.size(); ++id) {
        const std::string& symbol = vocab->tokens_[id];
        if (symbol.empty()) continue;
        // First id wins for a repeated symbol. The export writes one id per symbol, so a duplicate
        // means the table was built from a map that was not injective -- keeping the lower id makes the
        // behaviour deterministic rather than dependent on array order.
        vocab->ids_.emplace(symbol, static_cast<int32_t>(id));
        vocab->longest_ = std::max(vocab->longest_, symbol.size());
    }

    vocab->bos_id_ = model.kv_i32("tokenizer.ggml.phoneme.bos_id", -1);
    vocab->eos_id_ = model.kv_i32("tokenizer.ggml.phoneme.eos_id", -1);
    vocab->blank_id_ = model.kv_i32("tokenizer.ggml.phoneme.blank_id", -1);
    vocab->interleave_blank_ = model.kv_bool("tokenizer.ggml.phoneme.interleave_blank", false);
    return vocab;
}

const std::string& PhonemeVocab::piece(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) return kEmpty;
    return tokens_[static_cast<size_t>(id)];
}

std::vector<int32_t> PhonemeVocab::encode(const std::string& phonemes, size_t* unknown) const {
    std::vector<int32_t> body;
    size_t dropped = 0;

    for (size_t pos = 0; pos < phonemes.size();) {
        // Longest match first: a table holding both `a` and `aɪ` must consume the two-codepoint symbol
        // when it is there, and a shortest-first scan would split every diphthong.
        size_t width = std::min(longest_, phonemes.size() - pos);
        bool matched = false;
        for (; width > 0; --width) {
            auto found = ids_.find(phonemes.substr(pos, width));
            if (found != ids_.end()) {
                body.push_back(found->second);
                pos += width;
                matched = true;
                break;
            }
        }
        if (matched) continue;
        // Advance by a whole UTF-8 sequence rather than one byte, so an unknown multi-byte symbol is
        // dropped once instead of being counted as two or three separate failures -- and so the next
        // match attempt starts on a character boundary rather than mid-sequence, where nothing in the
        // table could ever match.
        size_t step = 1;
        const auto lead = static_cast<unsigned char>(phonemes[pos]);
        if ((lead & 0xE0) == 0xC0) step = 2;
        else if ((lead & 0xF0) == 0xE0) step = 3;
        else if ((lead & 0xF8) == 0xF0) step = 4;
        pos += std::min(step, phonemes.size() - pos);
        ++dropped;
    }

    if (unknown != nullptr) *unknown = dropped;

    // The assembly the checkpoint declared. Piper: [BOS, p1, blank, p2, blank, ..., pn, blank, EOS],
    // with no blank right after BOS -- which is why the blank follows each phoneme rather than
    // preceding it.
    std::vector<int32_t> out;
    out.reserve(body.size() * 2 + 2);
    if (bos_id_ >= 0) out.push_back(bos_id_);
    for (int32_t id : body) {
        out.push_back(id);
        if (interleave_blank_ && blank_id_ >= 0) out.push_back(blank_id_);
    }
    if (eos_id_ >= 0) out.push_back(eos_id_);
    return out;
}

std::string PhonemeVocab::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        // The assembly ids are structure rather than sound; printing them would put a literal blank
        // symbol between every phoneme of anything a caller round-trips.
        if (id == bos_id_ || id == eos_id_ || (interleave_blank_ && id == blank_id_)) continue;
        out += piece(id);
    }
    return out;
}

} // namespace loom
