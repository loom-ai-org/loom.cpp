#include "loom/core/vocab.h"
#include "loom/core/gguf_model.h"
#include "loom/loom_errors.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace loom {

namespace {

// Standard UTF-8 leading-byte decode: number of bytes in the codepoint starting with `lead`, or 0 if
// `lead` is not a valid leading byte (a continuation byte or otherwise malformed).
size_t utf8_codepoint_len(unsigned char lead) {
    if ((lead & 0x80) == 0x00) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 0;
}

constexpr int32_t kTokenTypeNormal = 1;
constexpr int32_t kTokenTypeUserDefined = 4;
constexpr int32_t kTokenTypeUnused = 5;

// SentencePiece's word-boundary marker, U+2581 ("▁"), UTF-8-encoded.
const std::string kEscapedSpace = "\xE2\x96\x81";

} // namespace

std::unique_ptr<Vocab> Vocab::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.ggml.model")) {
        return nullptr;
    }
    const std::string model_type = model.kv_str("tokenizer.ggml.model");
    if (model_type != "t5" && model_type != "llama") {
        throw LoadError("Vocab::load: unsupported tokenizer.ggml.model '" + model_type +
                         "' -- only \"t5\" (SentencePiece unigram/UGM) and \"llama\" (SentencePiece BPE) "
                         "are implemented");
    }

    auto vocab = std::unique_ptr<Vocab>(new Vocab());
    vocab->is_bpe_ = (model_type == "llama");
    vocab->tokens_ = model.kv_arr_str("tokenizer.ggml.tokens");
    vocab->scores_ = model.kv_arr_f32("tokenizer.ggml.scores");
    vocab->token_type_ = model.kv_arr_i32("tokenizer.ggml.token_type");
    if (vocab->scores_.size() != vocab->tokens_.size() || vocab->token_type_.size() != vocab->tokens_.size()) {
        throw LoadError("Vocab::load: tokenizer.ggml.{scores,token_type} size mismatch with .tokens");
    }
    vocab->unk_id_ = model.kv_i32("tokenizer.ggml.unknown_token_id", 0);
    vocab->bos_id_ = model.kv_i32("tokenizer.ggml.bos_token_id", -1);
    vocab->eos_id_ = model.kv_i32("tokenizer.ggml.eos_token_id", -1);
    vocab->add_bos_token_ = model.kv_bool("tokenizer.ggml.add_bos_token", false);
    vocab->add_eos_token_ = model.kv_bool("tokenizer.ggml.add_eos_token", false);
    vocab->add_space_prefix_ = model.kv_bool("tokenizer.ggml.add_space_prefix", true);
    vocab->remove_extra_whitespaces_ = model.kv_bool("tokenizer.ggml.remove_extra_whitespaces", true);

    if (model.has_kv("tokenizer.ggml.precompiled_charsmap")) {
        vocab->charsmap_ = model.kv_arr_u8("tokenizer.ggml.precompiled_charsmap");
        // Layout (mirrors llama.cpp's llm_tokenizer_ugm constructor exactly): first 4 bytes = byte
        // length of an XCDA blob of uint32_t entries; the remainder is a NUL-terminated
        // prefix-replacement string table indexed by byte offset.
        if (vocab->charsmap_.size() >= sizeof(uint32_t)) {
            uint32_t xcda_blob_size = 0;
            std::memcpy(&xcda_blob_size, vocab->charsmap_.data(), sizeof(uint32_t));
            size_t offset = sizeof(uint32_t);
            vocab->xcda_array_ = reinterpret_cast<const uint32_t*>(vocab->charsmap_.data() + offset);
            vocab->xcda_array_size_ = xcda_blob_size / sizeof(uint32_t);
            offset += xcda_blob_size;
            vocab->prefix_replacements_ = reinterpret_cast<const char*>(vocab->charsmap_.data() + offset);
            vocab->prefix_replacements_size_ = vocab->charsmap_.size() - offset;
        }
    }

    // Same piece-type filter as llama.cpp's token_matcher construction: NORMAL/USER_DEFINED/UNUSED
    // pieces are matchable substrings; UNKNOWN/CONTROL/BYTE are not.
    for (size_t id = 0; id < vocab->tokens_.size(); ++id) {
        const int32_t type = vocab->token_type_[id];
        if (type == kTokenTypeNormal || type == kTokenTypeUserDefined || type == kTokenTypeUnused) {
            vocab->trie_insert(vocab->tokens_[id], static_cast<int32_t>(id));
        }
    }
    return vocab;
}

void Vocab::trie_insert(const std::string& key, int32_t value) {
    TrieNode* node = &token_trie_;
    for (char c : key) {
        node = &node->children[c];
    }
    node->has_value = true;
    node->value = value;
}

bool Vocab::trie_find(const std::string& key, int32_t* id) const {
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

uint32_t Vocab::xcda_node(size_t index) const {
    if (index >= xcda_array_size_) {
        throw LoadError("Vocab: index out of bounds in XCDA array");
    }
    return xcda_array_[index];
}

// Bit-unpacking below mirrors llama.cpp's xcda_array_view::get_base/get_lcheck/get_leaf/get_value
// exactly (a single packed uint32_t node: [31: value/lcheck][9: leaf flag][8: base-shift flag][...]).
uint32_t Vocab::xcda_base(size_t index) const {
    const uint32_t packed = xcda_node(index);
    return (packed >> 10) << ((packed & (1U << 9)) >> 6);
}

uint32_t Vocab::xcda_lcheck(size_t index) const {
    const uint32_t packed = xcda_node(index);
    return packed & ((1U << 31) | 0xffU);
}

bool Vocab::xcda_leaf(size_t index) const {
    const uint32_t packed = xcda_node(index);
    return (packed >> 8) & 1U;
}

uint32_t Vocab::xcda_value(size_t index) const {
    const uint32_t packed = xcda_node(index);
    return packed & ((1U << 31) - 1);
}

Vocab::PrefixMatch Vocab::normalize_prefix(const std::string& input, size_t offset) const {
    if (offset == input.size()) {
        return {"", 0};
    }

    size_t longest_prefix_length = 0;
    size_t longest_prefix_offset = 0;
    if (xcda_array_size_ > 0) {
        uint32_t node_index = xcda_base(0);
        for (size_t i = offset; i < input.size(); ++i) {
            const unsigned char c = static_cast<unsigned char>(input[i]);
            if (c == 0) break;
            node_index ^= c;
            if (xcda_lcheck(node_index) != c) break;
            const bool is_leaf = xcda_leaf(node_index);
            node_index ^= xcda_base(node_index);
            if (is_leaf) {
                longest_prefix_length = i - offset + 1;
                longest_prefix_offset = xcda_value(node_index);
            }
        }
    }

    if (longest_prefix_length > 0) {
        if (longest_prefix_offset >= prefix_replacements_size_) {
            throw LoadError("Vocab: index out of bounds in charsmap replacement table");
        }
        const char* repl = prefix_replacements_ + longest_prefix_offset;
        const size_t max_len = prefix_replacements_size_ - longest_prefix_offset;
        size_t repl_len = 0;
        while (repl_len < max_len && repl[repl_len] != '\0') ++repl_len;
        if (repl_len == max_len) {
            throw LoadError("Vocab: unterminated string in charsmap replacement table");
        }
        return {std::string(repl, repl_len), longest_prefix_length};
    }

    // No charsmap match: pass the next UTF-8 codepoint through unchanged, or the UTF-8 replacement
    // character (U+FFFD) for an invalid leading byte, consuming exactly 1 byte in that case.
    const size_t cp_len = utf8_codepoint_len(static_cast<unsigned char>(input[offset]));
    if (cp_len == 0) {
        return {"\xEF\xBF\xBD", 1};
    }
    const size_t consumed = std::min(cp_len, input.size() - offset);
    return {input.substr(offset, consumed), consumed};
}

std::string Vocab::normalize(const std::string& text) const {
    // escape_whitespaces is always true for every UGM model this engine has seen (space -> ▁); no KV
    // exists to disable it, matching llama.cpp's own default.
    const std::string& space = kEscapedSpace;
    const bool shall_prepend_space = add_space_prefix_;
    const bool shall_merge_spaces = remove_extra_whitespaces_;

    std::string normalized;
    normalized.reserve(text.size() * 3);
    bool processing_non_ws = false;
    bool is_space_prepended = false;

    size_t offset = 0;
    while (offset < text.size()) {
        const PrefixMatch m = normalize_prefix(text, offset);
        for (char c : m.replacement) {
            if (c != ' ') {
                if (!processing_non_ws) {
                    processing_non_ws = true;
                    if ((shall_prepend_space && !is_space_prepended) || shall_merge_spaces) {
                        normalized += space;
                        is_space_prepended = true;
                    }
                }
                normalized += c;
            } else {
                if (processing_non_ws) {
                    processing_non_ws = false;
                }
                if (!shall_merge_spaces) {
                    normalized += space;
                }
            }
        }
        offset += m.consumed;
    }
    return normalized;
}

std::vector<int32_t> Vocab::encode_bpe(const std::string& normalized) const {
    // Split into per-UTF-8-codepoint initial symbols (character-level BPE, not GPT2's byte-level --
    // that convention lives in loom::BpeVocab instead).
    std::vector<std::string> symbols;
    size_t offset = 0;
    while (offset < normalized.size()) {
        const size_t cp_len = utf8_codepoint_len(static_cast<unsigned char>(normalized[offset]));
        const size_t consumed = std::min(cp_len > 0 ? cp_len : size_t{1}, normalized.size() - offset);
        symbols.push_back(normalized.substr(offset, consumed));
        offset += consumed;
    }

    // Repeatedly merge the single highest-scoring adjacent pair whose concatenation is a real vocab
    // piece, leftmost wins ties (only replace best_j on strictly-greater score) -- until no pair merges.
    while (symbols.size() > 1) {
        int32_t best_j = -1;
        float best_score = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j + 1 < symbols.size(); ++j) {
            int32_t id = 0;
            if (trie_find(symbols[j] + symbols[j + 1], &id) && scores_[static_cast<size_t>(id)] > best_score) {
                best_score = scores_[static_cast<size_t>(id)];
                best_j = static_cast<int32_t>(j);
            }
        }
        if (best_j < 0) break;
        symbols[static_cast<size_t>(best_j)] += symbols[static_cast<size_t>(best_j) + 1];
        symbols.erase(symbols.begin() + best_j + 1);
    }

    std::vector<int32_t> ids;
    ids.reserve(symbols.size());
    for (const std::string& s : symbols) {
        int32_t id = 0;
        ids.push_back(trie_find(s, &id) ? id : unk_id_);
    }
    return ids;
}

std::vector<int32_t> Vocab::encode(const std::string& text) const {
    std::vector<int32_t> ids = encode_impl(text);
    if (add_bos_token_ && bos_id_ >= 0) {
        ids.insert(ids.begin(), bos_id_);
    }
    if (add_eos_token_ && eos_id_ >= 0) {
        ids.push_back(eos_id_);
    }
    return ids;
}

std::vector<int32_t> Vocab::encode_impl(const std::string& text) const {
    const std::string normalized = normalize(text);
    if (is_bpe_) {
        return encode_bpe(normalized);
    }
    const size_t n = normalized.size();

    struct BestTok {
        int32_t token_id = -1;
        size_t start = 0;
        double score_sum = -std::numeric_limits<double>::infinity();
    };
    std::vector<BestTok> best(n + 1);
    best[0] = {-1, 0, 0.0};

    double min_score = 0.0;
    for (float s : scores_) min_score = std::min(min_score, static_cast<double>(s));
    const double unknown_token_score = min_score - 10.0;

    // Mirrors llama.cpp's llm_tokenizer_ugm_session::tokenize almost line-for-line: move through the
    // normalized string one UTF-8 codepoint at a time, at each position walk the vocab trie to relax
    // every reachable Viterbi DP entry, and fall back to unk_id() only if NO vocab piece (not even a
    // single codepoint) matched starting here.
    for (size_t offset = 0; offset < n;) {
        const size_t cp_len = utf8_codepoint_len(static_cast<unsigned char>(normalized[offset]));
        const size_t n_code_units = std::min(cp_len > 0 ? cp_len : size_t{1}, n - offset);

        bool single_codepoint_found = false;
        const BestTok& current_best = best[offset];
        const TrieNode* node = &token_trie_;
        size_t prefix_offset = offset;
        while (prefix_offset < n) {
            auto it = node->children.find(normalized[prefix_offset]);
            if (it == node->children.end()) break;
            node = &it->second;
            ++prefix_offset;
            if (node->has_value) {
                if (prefix_offset - offset == n_code_units) single_codepoint_found = true;
                const double piece_score = token_type_[static_cast<size_t>(node->value)] == kTokenTypeUserDefined
                                                ? 0.0
                                                : static_cast<double>(scores_[static_cast<size_t>(node->value)]);
                const double challenger = current_best.score_sum + piece_score;
                if (challenger > best[prefix_offset].score_sum) {
                    best[prefix_offset] = {node->value, offset, challenger};
                }
            }
        }

        if (!single_codepoint_found) {
            const size_t target = offset + n_code_units;
            const double challenger = current_best.score_sum + unknown_token_score;
            if (challenger > best[target].score_sum) {
                best[target] = {unk_id_, offset, challenger};
            }
        }

        offset += n_code_units;
    }

    std::vector<int32_t> ids;
    size_t pos = n;
    while (pos > 0) {
        const BestTok& bt = best[pos];
        ids.push_back(bt.token_id);
        pos = bt.start;
    }
    std::reverse(ids.begin(), ids.end());
    return ids;
}

std::string Vocab::decode(const std::vector<int32_t>& ids) const {
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

    if (add_space_prefix_ && !out.empty() && out.front() == ' ') {
        out.erase(0, 1);
    }
    return out;
}

const std::string& Vocab::id_to_piece(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) {
        throw LoadError("Vocab::id_to_piece: id " + std::to_string(id) + " out of range");
    }
    return tokens_[static_cast<size_t>(id)];
}

} // namespace loom
