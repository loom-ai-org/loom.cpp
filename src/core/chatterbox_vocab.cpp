#include "loom/core/chatterbox_vocab.h"

#include "loom/core/gguf_model.h"
#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

#include <limits>

namespace loom {
namespace {

const std::string kEmpty;
const std::string kPrefix = "tokenizer.ggml.chatterbox.";

// Python's `str.isspace()`, which is what `text.split()` splits on: the Unicode White_Space set plus
// the four information separators U+001C..U+001F. Written out rather than approximated with a
// category test, because `" ".join(text.split())` is a step whose output is compared byte for byte.
bool is_python_space(char32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

void replace_all(std::string& text, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    std::string out;
    size_t pos = 0;
    while (true) {
        const size_t hit = text.find(from, pos);
        if (hit == std::string::npos) break;
        out.append(text, pos, hit - pos);
        out += to;
        pos = hit + from.size();
    }
    out.append(text, pos, std::string::npos);
    text.swap(out);
}

bool ends_with(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> required_strings(const GgufModel& model, const std::string& key) {
    if (!model.has_kv(key)) {
        throw LoadError("ChatterboxVocab::load: tokenizer.ggml.model is 'chatterbox' but " + key +
                         " is missing");
    }
    return model.kv_arr_str(key);
}

std::string required_string(const GgufModel& model, const std::string& key) {
    if (!model.has_kv(key)) {
        throw LoadError("ChatterboxVocab::load: tokenizer.ggml.model is 'chatterbox' but " + key +
                         " is missing");
    }
    return model.kv_str(key);
}

} // namespace

std::unique_ptr<ChatterboxVocab> ChatterboxVocab::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.ggml.model") || model.kv_str("tokenizer.ggml.model") != "chatterbox") {
        return nullptr;
    }
    std::unique_ptr<ChatterboxVocab> vocab(new ChatterboxVocab());
    vocab->tokens_ = required_strings(model, "tokenizer.ggml.tokens");
    const auto merges = required_strings(model, "tokenizer.ggml.merges");
    const auto added = required_strings(model, kPrefix + "added_tokens");
    const auto word_chars = required_strings(model, kPrefix + "word_chars");
    vocab->replace_from_ = required_strings(model, kPrefix + "replace_from");
    vocab->replace_to_ = required_strings(model, kPrefix + "replace_to");
    vocab->enders_ = required_strings(model, kPrefix + "sentence_enders");
    vocab->dropped_on_decode_ = required_strings(model, kPrefix + "decode_drop");
    vocab->space_token_ = required_string(model, kPrefix + "space_token");
    vocab->empty_text_ = required_string(model, kPrefix + "empty_text");
    vocab->terminal_ = required_string(model, kPrefix + "terminal");
    vocab->unk_id_ = model.kv_i32("tokenizer.ggml.unknown_token_id", 1);
    if (vocab->replace_from_.size() != vocab->replace_to_.size()) {
        throw LoadError("ChatterboxVocab::load: replace_from has " +
                         std::to_string(vocab->replace_from_.size()) + " entries and replace_to has " +
                         std::to_string(vocab->replace_to_.size()) + " -- they are one table of pairs");
    }

    for (size_t id = 0; id < vocab->tokens_.size(); ++id) {
        // First id wins for a repeated piece, which `tokenizers` cannot produce (its vocab is a map).
        vocab->piece_to_id_.emplace(vocab->tokens_[id], static_cast<int32_t>(id));
    }
    for (const std::string& spelling : added) {
        const auto it = vocab->piece_to_id_.find(spelling);
        if (it == vocab->piece_to_id_.end()) {
            throw LoadError("ChatterboxVocab::load: added token '" + spelling + "' has no row");
        }
        vocab->added_.emplace(spelling, it->second);
        vocab->longest_added_ = std::max(vocab->longest_added_, spelling.size());
    }
    if (vocab->added_.count(vocab->space_token_) == 0) {
        // `[SPACE]` reaching the pre-tokenizer instead would split into `[`, `SPACE`, `]`: every word
        // boundary in every sentence would be three wrong ids.
        throw LoadError("ChatterboxVocab::load: the space token '" + vocab->space_token_ +
                         "' is not an added token");
    }
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const std::string& m = merges[rank];
        const size_t sp = m.find(' ');
        if (sp == std::string::npos || m.find(' ', sp + 1) != std::string::npos) {
            throw LoadError("ChatterboxVocab::load: merge '" + m + "' is not 'left right'");
        }
        if (vocab->piece_to_id_.count(m.substr(0, sp) + m.substr(sp + 1)) == 0) {
            throw LoadError("ChatterboxVocab::load: merge '" + m + "' produces a piece with no row");
        }
        // First rank wins for a repeated merge, as `tokenizers` builds its map.
        vocab->merge_rank_.emplace(m, static_cast<int32_t>(rank));
    }
    const auto upper_from = required_strings(model, kPrefix + "upper_from");
    const auto upper_to = required_strings(model, kPrefix + "upper_to");
    if (upper_from.size() != upper_to.size()) {
        throw LoadError("ChatterboxVocab::load: upper_from and upper_to differ in length");
    }
    for (size_t i = 0; i < upper_from.size(); ++i) {
        const auto cps = utf8_decode(upper_from[i]);
        if (cps.size() != 1) {
            throw LoadError("ChatterboxVocab::load: upper_from entry '" + upper_from[i] +
                             "' is not one codepoint");
        }
        vocab->upper_.emplace(cps[0], upper_to[i]);
    }
    for (const std::string& c : word_chars) {
        const auto cps = utf8_decode(c);
        if (cps.size() != 1) {
            throw LoadError("ChatterboxVocab::load: word_chars entry '" + c + "' is not one codepoint");
        }
        vocab->word_chars_.insert(cps[0]);
    }
    return vocab;
}

const std::string& ChatterboxVocab::id_to_piece(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) return kEmpty;
    return tokens_[static_cast<size_t>(id)];
}

std::string ChatterboxVocab::normalize(const std::string& input) const {
    if (input.empty()) return empty_text_;
    std::vector<char32_t> cps = utf8_decode(input);
    // `if text[0].islower(): text = text[0].upper() + text[1:]` -- the table holds exactly the
    // codepoints for which `islower()` is true, and the uppercase may be more than one codepoint.
    const auto up = upper_.find(cps[0]);
    if (up != upper_.end()) {
        const auto upper = utf8_decode(up->second);
        cps.erase(cps.begin());
        cps.insert(cps.begin(), upper.begin(), upper.end());
    }
    // `" ".join(text.split())`
    std::vector<char32_t> collapsed;
    collapsed.reserve(cps.size());
    bool pending_space = false;
    for (char32_t cp : cps) {
        if (is_python_space(cp)) {
            pending_space = !collapsed.empty();
            continue;
        }
        if (pending_space) collapsed.push_back(U' ');
        pending_space = false;
        collapsed.push_back(cp);
    }
    std::string text = utf8_encode(collapsed);
    for (size_t i = 0; i < replace_from_.size(); ++i) replace_all(text, replace_from_[i], replace_to_[i]);
    // `text.rstrip(" ")`
    while (!text.empty() && text.back() == ' ') text.pop_back();
    bool ended = false;
    for (const std::string& e : enders_) ended = ended || ends_with(text, e);
    if (!ended) text += terminal_;
    return text;
}

int32_t ChatterboxVocab::added_token_at(const std::string& text, size_t pos, size_t* len) const {
    const size_t max_len = std::min(longest_added_, text.size() - pos);
    for (size_t n = max_len; n > 0; --n) {
        const auto it = added_.find(text.substr(pos, n));
        if (it != added_.end()) {
            *len = n;
            return it->second;
        }
    }
    return -1;
}

void ChatterboxVocab::encode_pretoken(const std::u32string& word, std::vector<int32_t>& ids,
                                      size_t* unknown) const {
    // Symbols start as single codepoints. An unknown one is `[UNK]` and is never merged: no merge
    // names it, so leaving it out of every pair is exactly what `tokenizers` does.
    struct Sym { std::string piece; bool known; };
    std::vector<Sym> syms;
    syms.reserve(word.size());
    for (char32_t cp : word) {
        std::string piece = utf8_encode({cp});
        const bool known = piece_to_id_.count(piece) != 0;
        syms.push_back({std::move(piece), known});
    }
    // Lowest rank first, leftmost first among equals -- `tokenizers`' priority queue orders the same way.
    while (syms.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best = 0;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            if (!syms[i].known || !syms[i + 1].known) continue;
            const auto it = merge_rank_.find(syms[i].piece + " " + syms[i + 1].piece);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best = i;
            }
        }
        if (best_rank == std::numeric_limits<int32_t>::max()) break;
        syms[best].piece += syms[best + 1].piece;
        syms.erase(syms.begin() + static_cast<std::ptrdiff_t>(best) + 1);
    }
    for (const Sym& s : syms) {
        if (s.known) {
            ids.push_back(piece_to_id_.at(s.piece));
        } else {
            ids.push_back(unk_id_);
            if (unknown != nullptr) ++*unknown;
        }
    }
}

void ChatterboxVocab::encode_run(const std::u32string& run, std::vector<int32_t>& ids,
                                 size_t* unknown) const {
    // `Whitespace`: `\w+|[^\w\s]+`. No whitespace survives `normalize`, so every codepoint is in one
    // class or the other and a pre-token is a maximal run of one class.
    size_t start = 0;
    while (start < run.size()) {
        const bool word = word_chars_.count(run[start]) != 0;
        size_t end = start + 1;
        while (end < run.size() && (word_chars_.count(run[end]) != 0) == word) ++end;
        encode_pretoken(run.substr(start, end - start), ids, unknown);
        start = end;
    }
}

std::vector<int32_t> ChatterboxVocab::encode(const std::string& input, size_t* unknown) const {
    if (unknown != nullptr) *unknown = 0;
    std::string text = normalize(input);
    replace_all(text, " ", space_token_);
    std::vector<int32_t> ids;
    std::string run;
    const auto flush = [&] {
        if (run.empty()) return;
        const auto cps = utf8_decode(run);
        encode_run(std::u32string(cps.begin(), cps.end()), ids, unknown);
        run.clear();
    };
    size_t pos = 0;
    while (pos < text.size()) {
        size_t len = 0;
        const int32_t id = added_token_at(text, pos, &len);
        if (id >= 0) {
            flush();
            ids.push_back(id);
            pos += len;
        } else {
            run += text[pos];
            ++pos;
        }
    }
    flush();
    return ids;
}

std::string ChatterboxVocab::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        const std::string& piece = id_to_piece(id);
        bool drop = false;
        for (const std::string& d : dropped_on_decode_) drop = drop || piece == d;
        if (drop) continue;
        out += piece == space_token_ ? std::string(" ") : piece;
    }
    return out;
}

} // namespace loom
