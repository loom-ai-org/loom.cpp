#include "loom/core/f5_vocab.h"

#include "loom/core/gguf_model.h"
#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

namespace loom {
namespace {

const std::string kEmpty;

// The reference's own `custom_trans` (`f5_tts.model.utils.convert_char_to_pinyin`), which exists there
// "to address oov": five characters that have no row of their own, mapped onto ones that do. Applied
// before any lookup, exactly as the reference applies it before segmentation.
bool try_custom_trans(char32_t cp, char32_t& out) {
    switch (cp) {
        case U';':   out = U','; return true;
        case 0x201C: out = U'"'; return true;   // "
        case 0x201D: out = U'"'; return true;   // "
        case 0x2018: out = U'\''; return true;  // '
        case 0x2019: out = U'\''; return true;  // '
        default: return false;
    }
}

// The reference's own `is_chinese`, character for character: `"㄀" <= c <= "鿿"`. Deliberately
// that range and not a tighter one -- it is what decides whether the reference reaches for pinyin, so
// it is what decides whether this table can answer.
bool needs_pinyin_cp(char32_t cp) {
    return cp >= 0x3100 && cp <= 0x9FFF;
}

} // namespace

std::unique_ptr<F5Vocab> F5Vocab::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.ggml.model") || model.kv_str("tokenizer.ggml.model") != "f5") {
        return nullptr;
    }
    if (!model.has_kv("tokenizer.ggml.tokens")) {
        // The tag is present and the data is not: malformed rather than merely new, the same
        // distinction every other family here draws.
        throw LoadError("F5Vocab::load: tokenizer.ggml.model is 'f5' but tokenizer.ggml.tokens is "
                         "missing");
    }

    std::unique_ptr<F5Vocab> vocab(new F5Vocab());
    vocab->tokens_ = model.kv_arr_str("tokenizer.ggml.tokens");
    vocab->unk_id_ = model.kv_i32("tokenizer.ggml.unknown_token_id", 0);
    vocab->filler_offset_ = model.has_kv("tokenizer.ggml.f5.filler_offset")
                                ? static_cast<uint32_t>(
                                      model.kv_i32("tokenizer.ggml.f5.filler_offset", 1))
                                : 1u;

    for (size_t id = 0; id < vocab->tokens_.size(); ++id) {
        const std::string& piece = vocab->tokens_[id];
        if (piece.empty()) continue;
        // One CODEPOINT, not one byte: the table's single-character rows include accented Latin and
        // punctuation outside ASCII, and a byte-length test would index only half of them.
        if (utf8_decode(piece).size() != 1) continue;
        // First id wins for a repeated character. The reference builds a dict and the LAST id wins
        // there -- a difference that is unreachable in this release (its table has no duplicate
        // single-character row) and is recorded rather than matched, because matching it would mean
        // preferring the later row for a table where the earlier one is the trained id.
        vocab->single_.emplace(piece, static_cast<int32_t>(id));
    }
    return vocab;
}

const std::string& F5Vocab::id_to_piece(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) return kEmpty;
    return tokens_[static_cast<size_t>(id)];
}

bool F5Vocab::needs_pinyin(const std::string& text, std::string* first) const {
    for (char32_t cp : utf8_decode(text)) {
        if (needs_pinyin_cp(cp)) {
            if (first != nullptr) *first = utf8_encode({cp});
            return true;
        }
    }
    return false;
}

std::vector<int32_t> F5Vocab::encode(const std::string& text, size_t* unknown) const {
    std::string offender;
    if (needs_pinyin(text, &offender)) {
        throw Error("F5Vocab::encode: the text contains '" + offender + "', which F5-TTS's own front "
                     "end converts to pinyin before it reaches this table -- that conversion is "
                     "`rjieba` + `pypinyin` and is not in the file. Run the reference's "
                     "convert_char_to_pinyin and pass ids instead.");
    }

    std::vector<int32_t> ids;
    size_t dropped = 0;
    for (char32_t cp : utf8_decode(text)) {
        char32_t replaced;
        if (try_custom_trans(cp, replaced)) cp = replaced;
        const auto found = single_.find(utf8_encode({cp}));
        if (found == single_.end()) {
            // The reference's `vocab_char_map.get(c, 0)`: an unmapped character is the unknown id, not
            // a skip and not an error. Counting it is what lets a host say the sentence changed.
            ids.push_back(unk_id_);
            ++dropped;
            continue;
        }
        ids.push_back(found->second);
    }
    if (unknown != nullptr) *unknown = dropped;
    return ids;
}

std::string F5Vocab::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) {
            throw Error("F5Vocab::decode: id " + std::to_string(id) + " is outside a " +
                         std::to_string(tokens_.size()) + "-row table. If these are ids the GRAPH "
                         "saw, subtract filler_offset() first.");
        }
        out += tokens_[static_cast<size_t>(id)];
    }
    return out;
}

} // namespace loom
