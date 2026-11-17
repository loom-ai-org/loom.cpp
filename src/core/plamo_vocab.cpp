#include "loom/core/plamo_vocab.h"

#include "loom/core/gguf_model.h"
#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

#include <cstdio>
#include <limits>

namespace loom {
namespace {

constexpr uint8_t kNormal = 0, kControl = 1, kByte = 2, kUnknown = 3;
// The reference's UNKNOWN_SCORE: what a character no piece covers costs, before its bytes are emitted.
constexpr int64_t kUnknownCost = 10000000;
const std::string kPrefix = "tokenizer.ggml.plamo.";

// `<|plamo:` + up to 64 characters that are neither `|` nor whitespace (Python's `\s`) + `|>`, at
// `text[i]`: the match's length, or 0. `re.sub` scans left to right without overlap, as the caller does.
size_t tag_at(const std::vector<char32_t>& t, size_t i) {
    static const char32_t open[] = U"<|plamo:";
    const size_t n = t.size();
    for (size_t k = 0; k < 8; ++k) {
        if (i + k >= n || t[i + k] != open[k]) return 0;
    }
    // `{,64}` is greedy with backtracking, but its class excludes `|`, so the only way to end is the
    // first `|` -- which must then be followed by `>`.
    for (size_t k = 0; k <= 64; ++k) {
        const size_t at = i + 8 + k;
        if (at >= n) return 0;
        if (t[at] == U'|') return (at + 1 < n && t[at + 1] == U'>') ? 8 + k + 2 : 0;
        if (is_python_space(t[at])) return 0;
    }
    return 0;
}

} // namespace

PlamoVocab::PlamoVocab(std::vector<std::string> pieces, std::vector<int32_t> scores, std::vector<uint8_t> types,
                       int32_t break_spaces, int32_t break_repeats)
    : pieces_(std::move(pieces)), types_(std::move(types)), break_spaces_(break_spaces),
      break_repeats_(break_repeats) {
    if (scores.size() != pieces_.size() || types_.size() != pieces_.size()) {
        throw LoadError("PlamoVocab: pieces, scores and types differ in length");
    }
    for (int& b : byte_ids_) b = -1;
    for (size_t id = 0; id < pieces_.size(); ++id) {
        const std::string& p = pieces_[id];
        by_piece_.emplace(p, static_cast<int32_t>(id));
        if (types_[id] == kByte) {
            unsigned value = 0;
            if (p.size() != 6 || std::sscanf(p.c_str(), "<0x%2X>", &value) != 1 || value > 255) {
                throw LoadError("PlamoVocab: a BYTE piece is not spelled <0xNN>: '" + p + "'");
            }
            byte_ids_[value] = static_cast<int32_t>(id);
            continue;
        }
        // The reference's `token_to_token_id` keeps the LAST id of a duplicated spelling.
        lookup_[p] = Piece{static_cast<int32_t>(id), scores[id]};
        max_len_ = std::max(max_len_, utf8_decode(p).size());
        if (p == "<|plamo:bos|>") bos_id_ = static_cast<int32_t>(id);
        if (p == "<|plamo:eos|>") eos_id_ = static_cast<int32_t>(id);
    }
    for (int32_t b : byte_ids_) {
        if (b < 0) throw LoadError("PlamoVocab: not every byte has a <0xNN> piece");
    }
}

std::unique_ptr<PlamoVocab> PlamoVocab::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.ggml.model") || model.kv_str("tokenizer.ggml.model") != "plamo") return nullptr;
    for (const char* key : {"scores", "types"}) {
        if (!model.has_kv(kPrefix + key)) {
            throw LoadError("PlamoVocab::load: tokenizer.ggml.model is 'plamo' but " + kPrefix + key +
                            " is missing; re-export with a current loom-exporter");
        }
    }
    if (!model.has_kv("tokenizer.ggml.tokens")) throw LoadError("PlamoVocab::load: tokenizer.ggml.tokens is missing");
    return std::make_unique<PlamoVocab>(model.kv_arr_str("tokenizer.ggml.tokens"), model.kv_arr_i32(kPrefix + "scores"),
                                        model.kv_arr_u8(kPrefix + "types"), model.kv_i32(kPrefix + "break_spaces", 0),
                                        model.kv_i32(kPrefix + "break_repeats", 0));
}

int32_t PlamoVocab::piece_to_id(const std::string& piece) const {
    const auto it = by_piece_.find(piece);
    return it == by_piece_.end() ? -1 : it->second;
}

bool PlamoVocab::is_special(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= types_.size()) return false;
    return types_[static_cast<size_t>(id)] == kControl || types_[static_cast<size_t>(id)] == kUnknown;
}

void PlamoVocab::encode_segment(const std::vector<char32_t>& t, size_t begin, size_t end,
                                std::vector<int32_t>& out) const {
    const size_t n = end - begin;
    if (n == 0) return;
    // UTF-8 offsets of each codepoint, so a candidate piece is one substring and one hash lookup.
    std::vector<std::string> chars(n);
    for (size_t i = 0; i < n; ++i) chars[i] = utf8_encode({t[begin + i]});
    std::vector<int64_t> cost(n + 1, std::numeric_limits<int64_t>::max());
    std::vector<int32_t> length(n + 1, 0), token(n + 1, -1);
    cost[n] = 0;
    std::string piece;
    for (size_t i = n; i-- > 0;) {
        piece.clear();
        // Longest first, and only a strictly cheaper split replaces the best so far: the reference's tie
        // rule (its table lists a suffix's pieces in decreasing length).
        std::vector<std::pair<size_t, const Piece*>> found;
        for (size_t len = 1; len <= max_len_ && i + len <= n; ++len) {
            piece += chars[i + len - 1];
            const auto it = lookup_.find(piece);
            if (it != lookup_.end()) found.emplace_back(len, &it->second);
        }
        for (size_t f = found.size(); f-- > 0;) {
            const size_t len = found[f].first;
            const int64_t c = cost[i + len] - found[f].second->score;
            if (c < cost[i]) {
                cost[i] = c;
                length[i] = static_cast<int32_t>(len);
                token[i] = found[f].second->id;
            }
        }
        // The sentinel: one character, unknown, to be spelled in bytes.
        const int64_t c = cost[i + 1] + kUnknownCost;
        if (c < cost[i]) {
            cost[i] = c;
            length[i] = 1;
            token[i] = -1;
        }
    }
    for (size_t i = 0; i < n; i += static_cast<size_t>(length[i])) {
        if (token[i] >= 0) {
            out.push_back(token[i]);
        } else {
            for (unsigned char b : chars[i]) out.push_back(byte_ids_[b]);
        }
    }
}

std::vector<int32_t> PlamoVocab::encode(const std::string& text) const {
    const std::vector<char32_t> t = utf8_decode(text);
    const size_t n = t.size();
    // Boundaries, as cut points: after the tag pass, then the run pass over what the tags left.
    std::vector<bool> cut(n + 1, false);
    std::vector<bool> in_tag(n, false);
    for (size_t i = 0; i < n;) {
        const size_t m = tag_at(t, i);
        if (m == 0) { ++i; continue; }
        cut[i] = cut[i + m] = true;
        for (size_t k = i; k < i + m; ++k) in_tag[k] = true;
        i += m;
    }
    // `((.)\2{k-1,}| {s,})`, leftmost, non-overlapping, on the text with the tags already bracketed by
    // boundary characters. A boundary character is just another character to `.`, but it never repeats
    // four times, so runs never reach across a tag; a run INSIDE a tag only adds cuts the tag's own
    // piece then cannot cross -- which the reference does too.
    for (size_t i = 0; i < n;) {
        size_t run = 1;
        while (i + run < n && t[i + run] == t[i]) ++run;
        const bool repeats = break_repeats_ > 0 && t[i] != U'\n' && run >= static_cast<size_t>(break_repeats_);
        const bool spaces = break_spaces_ > 0 && t[i] == U' ' && run >= static_cast<size_t>(break_spaces_);
        if (repeats || spaces) {
            cut[i] = cut[i + run] = true;
        }
        i += run;
    }
    std::vector<int32_t> out;
    size_t start = 0;
    for (size_t i = 1; i <= n; ++i) {
        if (cut[i] || i == n) {
            encode_segment(t, start, i, out);
            start = i;
        }
    }
    return out;
}

std::string PlamoVocab::decode(const std::vector<int32_t>& ids, bool skip_special) const {
    std::string bytes;
    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= pieces_.size()) continue;
        const size_t k = static_cast<size_t>(id);
        if (skip_special && is_special(id)) continue;
        if (types_[k] == kByte) {
            unsigned value = 0;
            std::sscanf(pieces_[k].c_str(), "<0x%2X>", &value);
            bytes.push_back(static_cast<char>(value));
        } else {
            bytes += pieces_[k];
        }
    }
    // `.decode("utf-8", errors="replace")`: every maximal invalid subsequence becomes U+FFFD.
    std::string out;
    for (size_t i = 0; i < bytes.size();) {
        const unsigned char c = static_cast<unsigned char>(bytes[i]);
        size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
        bool ok = len > 0 && i + len <= bytes.size();
        for (size_t k = 1; ok && k < len; ++k) ok = (static_cast<unsigned char>(bytes[i + k]) & 0xC0) == 0x80;
        if (ok) {
            out.append(bytes, i, len);
            i += len;
        } else {
            out += "\xEF\xBF\xBD";
            ++i;
        }
    }
    return out;
}

} // namespace loom
