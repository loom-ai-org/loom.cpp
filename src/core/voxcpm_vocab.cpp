#include "loom/core/voxcpm_vocab.h"

#include "loom/core/gguf_model.h"
#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

#include <cstdio>
#include <limits>

namespace loom {
namespace {

const std::string kEmpty;
const std::string kPrefix = "tokenizer.ggml.voxcpm2.";

std::string required_string(const GgufModel& model, const std::string& key) {
    if (!model.has_kv(key)) {
        throw LoadError("VoxCpmVocab::load: tokenizer.ggml.model is 'voxcpm2' but " + key + " is missing");
    }
    return model.kv_str(key);
}

std::vector<std::string> required_strings(const GgufModel& model, const std::string& key) {
    if (!model.has_kv(key)) {
        throw LoadError("VoxCpmVocab::load: tokenizer.ggml.model is 'voxcpm2' but " + key + " is missing");
    }
    return model.kv_arr_str(key);
}

std::vector<int32_t> required_ids(const GgufModel& model, const std::string& key) {
    if (!model.has_kv(key)) {
        throw LoadError("VoxCpmVocab::load: tokenizer.ggml.model is 'voxcpm2' but " + key + " is missing");
    }
    return model.kv_arr_i32(key);
}

void replace_all(std::string& text, const std::string& from, const std::string& to) {
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

std::string byte_piece(unsigned char b) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned>(b));
    return buf;
}

} // namespace

std::unique_ptr<VoxCpmVocab> VoxCpmVocab::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.ggml.model") || model.kv_str("tokenizer.ggml.model") != "voxcpm2") {
        return nullptr;
    }
    std::unique_ptr<VoxCpmVocab> vocab(new VoxCpmVocab());
    vocab->tokens_ = required_strings(model, "tokenizer.ggml.tokens");
    const auto merges = required_strings(model, "tokenizer.ggml.merges");
    const auto added = required_strings(model, kPrefix + "added_tokens");
    vocab->prepend_ = required_string(model, kPrefix + "prepend");
    vocab->space_ = required_string(model, kPrefix + "space");
    vocab->unk_id_ = model.kv_i32("tokenizer.ggml.unknown_token_id", 0);

    for (size_t id = 0; id < vocab->tokens_.size(); ++id) {
        vocab->piece_to_id_.emplace(vocab->tokens_[id], static_cast<int32_t>(id));
    }
    for (const std::string& spelling : added) {
        const auto it = vocab->piece_to_id_.find(spelling);
        if (it == vocab->piece_to_id_.end()) {
            throw LoadError("VoxCpmVocab::load: added token '" + spelling + "' has no row");
        }
        vocab->added_.emplace(spelling, it->second);
        vocab->longest_added_ = std::max(vocab->longest_added_, spelling.size());
    }
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const std::string& m = merges[rank];
        const size_t sp = m.find(' ');
        if (sp == std::string::npos || m.find(' ', sp + 1) != std::string::npos) {
            throw LoadError("VoxCpmVocab::load: merge '" + m + "' is not 'left right'");
        }
        if (vocab->piece_to_id_.count(m.substr(0, sp) + m.substr(sp + 1)) == 0) {
            throw LoadError("VoxCpmVocab::load: merge '" + m + "' produces a piece with no row");
        }
        // First rank wins for a repeated merge, as `tokenizers` builds its map.
        vocab->merge_rank_.emplace(m, static_cast<int32_t>(rank));
    }
    vocab->byte_ids_.assign(256, -1);
    for (int b = 0; b < 256; ++b) {
        const auto it = vocab->piece_to_id_.find(byte_piece(static_cast<unsigned char>(b)));
        if (it == vocab->piece_to_id_.end()) {
            throw LoadError("VoxCpmVocab::load: byte fallback needs a <0xNN> piece for byte " + std::to_string(b));
        }
        vocab->byte_ids_[static_cast<size_t>(b)] = it->second;
    }

    const auto from = required_ids(model, kPrefix + "split_from");
    const auto offsets = required_ids(model, kPrefix + "split_offsets");
    const auto to = required_ids(model, kPrefix + "split_to");
    if (offsets.size() != from.size() + 1 || offsets.front() != 0 ||
        static_cast<size_t>(offsets.back()) != to.size()) {
        throw LoadError("VoxCpmVocab::load: split_offsets does not delimit split_to into one run per split_from id");
    }
    const auto n_tokens = static_cast<int32_t>(vocab->tokens_.size());
    for (size_t i = 0; i < from.size(); ++i) {
        if (offsets[i] > offsets[i + 1]) throw LoadError("VoxCpmVocab::load: split_offsets decreases");
        std::vector<int32_t> run(to.begin() + offsets[i], to.begin() + offsets[i + 1]);
        for (int32_t id : run) {
            if (id < 0 || id >= n_tokens) throw LoadError("VoxCpmVocab::load: split_to names an id with no row");
        }
        vocab->split_.emplace(from[i], std::move(run));
    }
    return vocab;
}

const std::string& VoxCpmVocab::id_to_piece(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) return kEmpty;
    return tokens_[static_cast<size_t>(id)];
}

std::string VoxCpmVocab::normalize(const std::string& input) const {
    // `re.sub(r"\s+", " ", text.replace("\n", " "))`: `\s` is `str.isspace()` for a str pattern, and a
    // newline is one of those, so the replace is subsumed.
    std::vector<char32_t> out;
    bool in_space = false;
    for (char32_t cp : utf8_decode(input)) {
        if (is_python_space(cp)) {
            if (!in_space) out.push_back(U' ');
            in_space = true;
            continue;
        }
        in_space = false;
        out.push_back(cp);
    }
    return utf8_encode(out);
}

int32_t VoxCpmVocab::added_token_at(const std::string& text, size_t pos, size_t* len) const {
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

void VoxCpmVocab::encode_segment(const std::string& segment, std::vector<int32_t>& ids, size_t* fallback) const {
    // The normalizer, per segment the added tokens leave (`Prepend` touches only a non-empty one).
    std::string text = prepend_ + segment;
    replace_all(text, " ", space_);

    // A character the table lacks becomes its bytes' pieces BEFORE merging, as `tokenizers` builds its
    // word; no merge names a byte piece, so they sit out every pair.
    std::vector<std::string> syms;
    std::vector<bool> known;
    for (char32_t cp : utf8_decode(text)) {
        std::string piece = utf8_encode({cp});
        if (piece_to_id_.count(piece) != 0) {
            syms.push_back(std::move(piece));
            known.push_back(true);
            continue;
        }
        if (fallback != nullptr) ++*fallback;
        for (unsigned char b : piece) {
            syms.push_back(byte_piece(b));
            known.push_back(false);
        }
    }
    // Lowest rank first, leftmost first among equals -- `tokenizers`' priority queue orders the same way.
    while (syms.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best = 0;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            if (!known[i] || !known[i + 1]) continue;
            const auto it = merge_rank_.find(syms[i] + " " + syms[i + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best = i;
            }
        }
        if (best_rank == std::numeric_limits<int32_t>::max()) break;
        syms[best] += syms[best + 1];
        syms.erase(syms.begin() + static_cast<std::ptrdiff_t>(best) + 1);
        known.erase(known.begin() + static_cast<std::ptrdiff_t>(best) + 1);
    }
    for (const std::string& s : syms) {
        const auto it = piece_to_id_.find(s);
        const int32_t id = it == piece_to_id_.end() ? unk_id_ : it->second;
        const auto split = split_.find(id);
        if (split == split_.end()) {
            ids.push_back(id);
        } else {
            ids.insert(ids.end(), split->second.begin(), split->second.end());
        }
    }
}

std::vector<int32_t> VoxCpmVocab::encode(const std::string& input, size_t* fallback) const {
    if (fallback != nullptr) *fallback = 0;
    const std::string text = normalize(input);
    std::vector<int32_t> ids;
    std::string run;
    const auto flush = [&] {
        if (!run.empty()) encode_segment(run, ids, fallback);
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

std::string VoxCpmVocab::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        const std::string& piece = id_to_piece(id);
        if (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 && piece[5] == '>') {
            out += static_cast<char>(std::stoul(piece.substr(3, 2), nullptr, 16));
            continue;
        }
        out += piece;
    }
    replace_all(out, space_, " ");
    if (!out.empty() && out.front() == ' ') out.erase(out.begin());
    return out;
}

} // namespace loom
