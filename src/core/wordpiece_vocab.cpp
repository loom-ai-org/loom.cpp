#include "loom/core/wordpiece_vocab.h"
#include "loom/core/gguf_model.h"
#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

#include <algorithm>

namespace loom {
namespace {

// SentencePiece's word-boundary marker, U+2581 ("▁"), UTF-8-encoded -- same convention loom::Vocab uses.
const std::string kEscapedSpace = "\xE2\x96\x81";

bool is_ws(char32_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }

// Cc (control) + a handful of Cf (format, e.g. zero-width/bidi marks a real BERT vocab.txt input can
// contain) codepoints WPM's own preprocess() strips -- approximated via fixed C0/C1 control ranges plus
// the common zero-width/bidi-format block, rather than a full generated Cc/Cf category table (this
// engine's existing convention elsewhere, e.g. is_ws, prefers a small fixed-range approximation over a new
// full Unicode category table when the affected code path is rare in real tokenizer input).
bool is_control_char(char32_t cp) {
    return (cp <= 0x1F) || (cp >= 0x7F && cp <= 0x9F) ||
           (cp >= 0x200B && cp <= 0x200F) || (cp >= 0x202A && cp <= 0x202E);
}

// The exact sub-128 "symbol" set llama.cpp's own unicode.cpp hardcodes for WPM's `cpt < 0x7F &&
// flags.is_symbol` check (k_ucat_map's SYMBOL entry: "$+<=>^`|") -- not general \p{S}, just these 8.
bool is_ascii_symbol(char32_t cp) {
    switch (cp) {
        case U'$': case U'+': case U'<': case U'=': case U'>': case U'^': case U'`': case U'|':
            return true;
        default:
            return false;
    }
}

} // namespace

std::unique_ptr<WordPieceVocab> WordPieceVocab::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.ggml.model")) {
        return nullptr;
    }
    if (model.kv_str("tokenizer.ggml.model") != "bert") {
        return nullptr; // not this vocab type -- caller should try BpeVocab/Vocab instead
    }

    auto vocab = std::unique_ptr<WordPieceVocab>(new WordPieceVocab());
    vocab->tokens_ = model.kv_arr_str("tokenizer.ggml.tokens");
    for (size_t i = 0; i < vocab->tokens_.size(); ++i) {
        vocab->trie_insert(vocab->tokens_[i], static_cast<int32_t>(i));
        vocab->max_token_len_ = std::max(vocab->max_token_len_, vocab->tokens_[i].size());
    }

    vocab->unk_id_ = model.kv_i32("tokenizer.ggml.unknown_token_id", 0);
    vocab->cls_id_ = model.kv_i32("tokenizer.ggml.bos_token_id", -1); // CLS reuses the generic BOS KV
    vocab->sep_id_ = model.kv_i32("tokenizer.ggml.seperator_token_id", -1); // llama.cpp's own (misspelled) KV name, kept verbatim
    vocab->pad_id_ = model.kv_i32("tokenizer.ggml.padding_token_id", -1);
    vocab->mask_id_ = model.kv_i32("tokenizer.ggml.mask_token_id", -1);
    vocab->add_bos_token_ = model.kv_bool("tokenizer.ggml.add_bos_token", false);
    vocab->add_sep_token_ = model.kv_bool("tokenizer.ggml.add_sep_token", false);
    vocab->lowercase_ = model.kv_bool("tokenizer.ggml.normalizer.lowercase", true);
    vocab->strip_accents_ = model.kv_bool("tokenizer.ggml.normalizer.strip_accents", true);
    return vocab;
}

void WordPieceVocab::trie_insert(const std::string& key, int32_t value) {
    TrieNode* node = &token_trie_;
    for (char c : key) {
        node = &node->children[c];
    }
    node->has_value = true;
    node->value = value;
}

bool WordPieceVocab::trie_find(const std::string& key, int32_t* id) const {
    const TrieNode* node = &token_trie_;
    for (char c : key) {
        auto it = node->children.find(c);
        if (it == node->children.end()) return false;
        node = &it->second;
    }
    if (!node->has_value) return false;
    *id = node->value;
    return true;
}

bool WordPieceVocab::is_chinese_char(char32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) ||
           (cp >= 0x2B920 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0x2F800 && cp <= 0x2FA1F);
}

// Ported natively from llm_tokenizer_wpm_session::preprocess (src/llama-vocab.cpp): NFD-decompose (if
// strip_accents_) then walk codepoints, splitting on whitespace and isolating punctuation/ASCII-symbols/
// CJK characters into their own single-character "words".
std::vector<std::string> WordPieceVocab::preprocess(const std::string& text) const {
    std::vector<char32_t> cps = utf8_decode(text);
    if (strip_accents_) {
        cps = utf8_decode(nfd_normalize(utf8_encode(cps)));
    }

    std::vector<std::string> words(1, "");
    for (char32_t cp : cps) {
        if (is_ws(cp)) {
            if (!words.back().empty()) words.emplace_back();
            continue;
        }
        if (cp == 0 || cp == 0xFFFD || is_control_char(cp)) {
            continue;
        }
        if (strip_accents_ && is_mark(cp)) {
            continue;
        }

        const std::string s = utf8_encode({lowercase_ ? to_lower(cp) : cp});
        if (is_punctuation(cp) || (cp < 0x7F && is_ascii_symbol(cp)) || is_chinese_char(cp)) {
            if (!words.back().empty()) words.emplace_back();
            words.back() = s; // single-char word
            words.emplace_back(); // start a new word
        } else {
            words.back() += s;
        }
    }
    if (words.back().empty()) words.pop_back();
    return words;
}

// Ported natively from llm_tokenizer_wpm_session::tokenize's inner loop: prepend the phantom-space
// marker, then greedy longest-match-first substring lookup from each position. Returns empty on total
// failure (caller falls back to a single [UNK] for the whole word, matching the reference exactly --
// NOT partial pieces + UNK).
std::vector<int32_t> WordPieceVocab::tokenize_word(const std::string& word) const {
    const std::string word1 = kEscapedSpace + word;
    const size_t n = word1.size();
    std::vector<int32_t> out;
    size_t i = 0;
    while (i < n) {
        bool matched = false;
        for (size_t j = std::min(n, i + max_token_len_ + 1); j > i; --j) {
            int32_t id = 0;
            if (trie_find(word1.substr(i, j - i), &id)) {
                out.push_back(id);
                i = j;
                matched = true;
                break;
            }
        }
        if (!matched) return {};
    }
    return out;
}

std::vector<int32_t> WordPieceVocab::encode(const std::string& text) const {
    std::vector<int32_t> ids;
    if (add_bos_token_ && cls_id_ >= 0) {
        ids.push_back(cls_id_);
    }
    for (const std::string& word : preprocess(text)) {
        if (word.empty()) continue;
        const std::vector<int32_t> word_ids = tokenize_word(word);
        if (word_ids.empty()) {
            ids.push_back(unk_id_);
        } else {
            ids.insert(ids.end(), word_ids.begin(), word_ids.end());
        }
    }
    if (add_sep_token_ && sep_id_ >= 0) {
        ids.push_back(sep_id_);
    }
    return ids;
}

std::string WordPieceVocab::decode(const std::vector<int32_t>& ids) const {
    std::string joined;
    for (int32_t id : ids) {
        joined += id_to_piece(id);
    }

    std::string out;
    out.reserve(joined.size());
    size_t i = 0;
    while (i < joined.size()) {
        if (joined.compare(i, kEscapedSpace.size(), kEscapedSpace) == 0) {
            out += ' ';
            i += kEscapedSpace.size();
        } else {
            out += joined[i];
            ++i;
        }
    }
    if (!out.empty() && out.front() == ' ') {
        out.erase(0, 1);
    }
    return out;
}

const std::string& WordPieceVocab::id_to_piece(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) {
        throw LoadError("WordPieceVocab::id_to_piece: id " + std::to_string(id) + " out of range");
    }
    return tokens_[static_cast<size_t>(id)];
}

} // namespace loom
