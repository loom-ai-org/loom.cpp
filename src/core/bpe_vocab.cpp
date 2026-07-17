#include "loom/core/bpe_vocab.h"
#include "loom/core/gguf_model.h"
#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

#include <array>
#include <climits>

namespace loom {
namespace {

// GPT2's standard byte<->unicode-codepoint mapping (Radford et al.'s "byte-level BPE" trick, identical
// across every GPT2/RoBERTa/GPT-NeoX/Qwen tokenizer -- not model-specific, a fixed universal convention):
// printable Latin-1 bytes map to themselves; every other byte value (control chars, space, DEL, etc.)
// maps to a codepoint >= 256 so every byte has SOME visible, round-trippable representation as text.
std::array<char32_t, 256> compute_byte_encoder() {
    std::vector<int> bs;
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::array<bool, 256> in_bs{};
    for (int b : bs) in_bs[static_cast<size_t>(b)] = true;
    std::vector<char32_t> cs(bs.begin(), bs.end());
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (!in_bs[static_cast<size_t>(b)]) {
            bs.push_back(b);
            cs.push_back(static_cast<char32_t>(256 + n));
            ++n;
        }
    }
    std::array<char32_t, 256> table{};
    for (size_t i = 0; i < bs.size(); ++i) table[static_cast<size_t>(bs[i])] = cs[i];
    return table;
}

const std::array<char32_t, 256>& byte_encoder() {
    static const std::array<char32_t, 256> table = compute_byte_encoder();
    return table;
}

const std::unordered_map<char32_t, uint8_t>& byte_decoder() {
    static const std::unordered_map<char32_t, uint8_t> table = [] {
        std::unordered_map<char32_t, uint8_t> m;
        const auto& enc = byte_encoder();
        for (size_t b = 0; b < 256; ++b) m.emplace(enc[b], static_cast<uint8_t>(b));
        return m;
    }();
    return table;
}

bool is_ws(char32_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }

// [^\s\p{L}\p{N}] -- "punctuation-ish": not whitespace, not a letter, not a number.
bool is_punct(char32_t c) { return !is_ws(c) && !is_letter(c) && !is_number(c); }

std::string slice_utf8(const std::vector<char32_t>& cps, size_t begin, size_t end) {
    return utf8_encode(std::vector<char32_t>(cps.begin() + static_cast<long>(begin),
                                              cps.begin() + static_cast<long>(end)));
}

// (?i:'s|'t|'re|'ve|'m|'ll|'d) -- the fixed set of English contraction suffixes the real Qwen2/Qwen3
// tokenizer.json pretokenizer regex special-cases ahead of the general letter-run alternative.
bool match_contraction(const std::vector<char32_t>& cps, size_t pos, size_t& end) {
    if (cps[pos] != U'\'') return false;
    static const char* const kSuffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    for (const char* suf : kSuffixes) {
        const size_t len = std::char_traits<char>::length(suf);
        if (pos + 1 + len > cps.size()) continue;
        bool match = true;
        for (size_t k = 0; k < len; ++k) {
            char32_t c = cps[pos + 1 + k];
            if (c >= U'A' && c <= U'Z') c = c - U'A' + U'a'; // ASCII-only lowercasing, matches (?i:) scope
            if (c != static_cast<char32_t>(suf[k])) { match = false; break; }
        }
        if (match) { end = pos + 1 + len; return true; }
    }
    return false;
}

// [^\r\n\p{L}\p{N}]?\p{L}+
bool match_letter_run(const std::vector<char32_t>& cps, size_t pos, size_t& end) {
    const size_t n = cps.size();
    size_t letters_begin = pos;
    if (!(cps[pos] == U'\r' || cps[pos] == U'\n' || is_letter(cps[pos]) || is_number(cps[pos]))) {
        if (pos + 1 < n && is_letter(cps[pos + 1])) letters_begin = pos + 1;
        else return false; // lone prefix char with no following letter -- not this alternative's match
    } else if (!is_letter(cps[pos])) {
        return false;
    }
    size_t p = letters_begin;
    while (p < n && is_letter(cps[p])) ++p;
    end = p;
    return true;
}

// ` ?[^\s\p{L}\p{N}]+[\r\n]*`
bool match_punct_run(const std::vector<char32_t>& cps, size_t pos, size_t& end) {
    const size_t n = cps.size();
    size_t punct_begin;
    if (cps[pos] == U' ' && pos + 1 < n && is_punct(cps[pos + 1])) {
        punct_begin = pos + 1;
    } else if (is_punct(cps[pos])) {
        punct_begin = pos;
    } else {
        return false;
    }
    size_t p = punct_begin;
    while (p < n && is_punct(cps[p])) ++p;
    while (p < n && (cps[p] == U'\r' || cps[p] == U'\n')) ++p;
    end = p;
    return true;
}

// `\s*[\r\n]+` -- see bpe_vocab.h's module-level design note (unicode.h doc comments) / BACKLOG.md for
// the backtracking trace justifying this closed-form reduction: succeeds iff the maximal whitespace run
// starting at `pos` contains at least one \r or \n, consuming up to and including the LAST one.
bool match_ws_then_newline(const std::vector<char32_t>& cps, size_t pos, size_t& end) {
    if (!is_ws(cps[pos])) return false;
    const size_t n = cps.size();
    size_t run_end = pos;
    while (run_end < n && is_ws(cps[run_end])) ++run_end;
    for (size_t k = run_end; k > pos; --k) {
        if (cps[k - 1] == U'\r' || cps[k - 1] == U'\n') {
            end = k;
            return true;
        }
    }
    return false;
}

// `\s+(?!\S)` -- only reached once match_ws_then_newline has already failed, so the run at `pos` is
// guaranteed newline-free. Matches the whole run if it reaches end-of-string; otherwise the lookahead
// forces giving back exactly the run's last character (still whitespace, so `(?!\S)` is satisfied).
bool match_ws_not_followed_by_nonspace(const std::vector<char32_t>& cps, size_t pos, size_t& end) {
    if (!is_ws(cps[pos])) return false;
    const size_t n = cps.size();
    size_t run_end = pos;
    while (run_end < n && is_ws(cps[run_end])) ++run_end;
    if (run_end == n) { end = run_end; return true; }
    if (run_end - pos < 2) return false; // single char, followed by non-whitespace -- can't satisfy \s+ AND the lookahead
    end = run_end - 1;
    return true;
}

// `\s+` -- unconditional final fallback.
bool match_ws_fallback(const std::vector<char32_t>& cps, size_t pos, size_t& end) {
    if (!is_ws(cps[pos])) return false;
    const size_t n = cps.size();
    size_t run_end = pos;
    while (run_end < n && is_ws(cps[run_end])) ++run_end;
    end = run_end;
    return true;
}

} // namespace

std::unique_ptr<BpeVocab> BpeVocab::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.ggml.model")) {
        return nullptr;
    }
    const std::string model_type = model.kv_str("tokenizer.ggml.model");
    if (model_type != "gpt2") {
        return nullptr; // not this vocab type -- caller should try loom::Vocab (SentencePiece) instead
    }

    auto vocab = std::unique_ptr<BpeVocab>(new BpeVocab());
    vocab->tokens_ = model.kv_arr_str("tokenizer.ggml.tokens");
    vocab->token_to_id_.reserve(vocab->tokens_.size());
    for (size_t i = 0; i < vocab->tokens_.size(); ++i) {
        vocab->token_to_id_.emplace(vocab->tokens_[i], static_cast<int32_t>(i));
    }

    const std::vector<std::string> merges = model.kv_arr_str("tokenizer.ggml.merges");
    vocab->merge_rank_.reserve(merges.size());
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const std::string& pair = merges[rank];
        const size_t sep = pair.find(' ');
        if (sep == std::string::npos) {
            throw LoadError("BpeVocab::load: malformed tokenizer.ggml.merges entry '" + pair + "' (expected \"a b\")");
        }
        vocab->merge_rank_.emplace(pair.substr(0, sep) + '\x01' + pair.substr(sep + 1), static_cast<int32_t>(rank));
    }

    vocab->bos_id_ = model.kv_i32("tokenizer.ggml.bos_token_id", -1);
    vocab->eos_id_ = model.kv_i32("tokenizer.ggml.eos_token_id", -1);
    return vocab;
}

std::vector<std::string> BpeVocab::pretokenize(const std::string& nfc_text) const {
    const std::vector<char32_t> cps = utf8_decode(nfc_text);
    std::vector<std::string> chunks;
    size_t pos = 0;
    const size_t n = cps.size();
    while (pos < n) {
        size_t end;
        if (match_contraction(cps, pos, end) || match_letter_run(cps, pos, end) ||
            (is_number(cps[pos]) && (end = pos + 1, true)) || match_punct_run(cps, pos, end) ||
            match_ws_then_newline(cps, pos, end) || match_ws_not_followed_by_nonspace(cps, pos, end) ||
            match_ws_fallback(cps, pos, end)) {
            chunks.push_back(slice_utf8(cps, pos, end));
            pos = end;
        } else {
            // Every codepoint is whitespace, a letter, a number, or "other" (punct) -- the 7 alternatives
            // above are exhaustive over those classes, so this should be unreachable; fail closed rather
            // than infinite-loop if some future Unicode edge case proves otherwise.
            throw LoadError("BpeVocab::pretokenize: no pretokenizer alternative matched at codepoint U+" +
                             std::to_string(static_cast<uint32_t>(cps[pos])));
        }
    }
    return chunks;
}

void BpeVocab::bpe_merge(std::vector<std::string>& pieces) const {
    while (pieces.size() >= 2) {
        int best_rank = INT_MAX;
        size_t best_idx = SIZE_MAX;
        for (size_t i = 0; i + 1 < pieces.size(); ++i) {
            const auto it = merge_rank_.find(pieces[i] + '\x01' + pieces[i + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        if (best_idx == SIZE_MAX) break;
        pieces[best_idx] += pieces[best_idx + 1];
        pieces.erase(pieces.begin() + static_cast<long>(best_idx) + 1);
    }
}

std::vector<int32_t> BpeVocab::encode(const std::string& text) const {
    const std::string normalized = nfc_normalize(text);
    const std::vector<std::string> chunks = pretokenize(normalized);
    const auto& enc = byte_encoder();

    std::vector<int32_t> ids;
    for (const std::string& chunk : chunks) {
        std::vector<std::string> pieces;
        pieces.reserve(chunk.size());
        for (unsigned char b : chunk) pieces.push_back(utf8_encode({enc[b]}));

        bpe_merge(pieces);

        for (const std::string& p : pieces) {
            const auto it = token_to_id_.find(p);
            if (it == token_to_id_.end()) {
                throw LoadError("BpeVocab::encode: merged piece '" + p + "' is not in the vocabulary "
                                 "(every single byte-mapped character should be a base vocab entry)");
            }
            ids.push_back(it->second);
        }
    }
    return ids;
}

std::string BpeVocab::decode(const std::vector<int32_t>& ids) const {
    const auto& dec = byte_decoder();
    std::string raw;
    for (int32_t id : ids) {
        const std::string& piece = id_to_piece(id);
        for (char32_t cp : utf8_decode(piece)) {
            const auto it = dec.find(cp);
            if (it != dec.end()) raw.push_back(static_cast<char>(it->second));
        }
    }
    return raw;
}

const std::string& BpeVocab::id_to_piece(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) {
        throw LoadError("BpeVocab::id_to_piece: id " + std::to_string(id) + " out of range");
    }
    return tokens_[static_cast<size_t>(id)];
}

} // namespace loom
