#include "loom/core/cosyvoice3_vocab.h"

#include "loom/core/gguf_model.h"
#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

#include <algorithm>

namespace loom {
namespace {

const std::string kPrefix = "tokenizer.ggml.cosyvoice3.";

void require(const GgufModel& model, const std::string& key) {
    if (!model.has_kv(key)) {
        throw LoadError("CosyVoice3Vocab::load: tokenizer.ggml.model is 'cosyvoice3' but " + key + " is missing");
    }
}

std::vector<std::string> strings(const GgufModel& model, const std::string& key) {
    require(model, kPrefix + key);
    return model.kv_arr_str(kPrefix + key);
}

std::string string(const GgufModel& model, const std::string& key) {
    require(model, kPrefix + key);
    return model.kv_str(kPrefix + key);
}

std::vector<int32_t> ints(const GgufModel& model, const std::string& key) {
    require(model, kPrefix + key);
    return model.kv_arr_i32(kPrefix + key);
}

size_t positive(const GgufModel& model, const std::string& key) {
    const int32_t value = model.kv_i32(kPrefix + key, -1);
    if (value <= 0) throw LoadError("CosyVoice3Vocab::load: " + kPrefix + key + " is missing or not positive");
    return static_cast<size_t>(value);
}

char32_t one_char(const std::string& entry, const std::string& key) {
    const std::vector<char32_t> cps = utf8_decode(entry);
    if (cps.size() != 1) {
        throw LoadError("CosyVoice3Vocab::load: " + key + " entry '" + entry + "' is not one character");
    }
    return cps[0];
}

std::unordered_set<char32_t> char_set(const GgufModel& model, const std::string& key) {
    std::unordered_set<char32_t> out;
    for (const std::string& entry : strings(model, key)) out.insert(one_char(entry, key));
    return out;
}

// Inclusive [lo, hi] pairs, ascending and disjoint, so membership is one binary search.
std::vector<int32_t> ranges(const GgufModel& model, const std::string& key) {
    std::vector<int32_t> r = ints(model, key);
    if (r.size() % 2 != 0) throw LoadError("CosyVoice3Vocab::load: " + kPrefix + key + " is not [lo, hi] pairs");
    for (size_t i = 0; i < r.size(); i += 2) {
        if (r[i] > r[i + 1] || (i > 0 && r[i] <= r[i - 1])) {
            throw LoadError("CosyVoice3Vocab::load: " + kPrefix + key + " is not ascending disjoint ranges");
        }
    }
    return r;
}

bool in_ranges(const std::vector<int32_t>& r, char32_t cp) {
    // The first pair whose `hi` is >= cp; cp is in it when its `lo` is <= cp.
    size_t lo = 0, hi = r.size() / 2;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (static_cast<char32_t>(r[2 * mid + 1]) < cp) lo = mid + 1;
        else hi = mid;
    }
    return lo < r.size() / 2 && static_cast<char32_t>(r[2 * lo]) <= cp;
}

// Python's `str.replace(from, to)`: every non-overlapping occurrence, left to right, in one pass. On
// UTF-8 bytes this is the codepoint operation, since no character's encoding starts inside another's.
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

std::u32string to_u32(const std::string& text) {
    const std::vector<char32_t> cps = utf8_decode(text);
    return {cps.begin(), cps.end()};
}

std::string to_utf8(const std::u32string& text) { return utf8_encode({text.begin(), text.end()}); }

} // namespace

std::unique_ptr<CosyVoice3Vocab> CosyVoice3Vocab::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.ggml.model") || model.kv_str("tokenizer.ggml.model") != "cosyvoice3") {
        return nullptr;
    }
    std::unique_ptr<CosyVoice3Vocab> v(new CosyVoice3Vocab());
    v->bpe_ = BpeVocab::load_bpe(model);
    v->header_ = model.kv_i32(kPrefix + "chunk_header", -1);
    if (v->header_ < 0 || static_cast<size_t>(v->header_) >= v->bpe_->size()) {
        throw LoadError("CosyVoice3Vocab::load: " + kPrefix + "chunk_header is missing or not an id");
    }
    v->markup_open_ = string(model, "markup_open");
    v->markup_close_ = string(model, "markup_close");
    v->zh_ranges_ = ranges(model, "zh_ranges");
    v->punct_ranges_ = ranges(model, "punct_ranges");
    v->zh_pre_from_ = strings(model, "zh_pre_from");
    v->zh_pre_to_ = strings(model, "zh_pre_to");
    v->zh_replace_from_ = strings(model, "zh_replace_from");
    v->zh_replace_to_ = strings(model, "zh_replace_to");
    if (v->zh_pre_from_.size() != v->zh_pre_to_.size() || v->zh_replace_from_.size() != v->zh_replace_to_.size()) {
        throw LoadError("CosyVoice3Vocab::load: a replacement table's from and to differ in length");
    }
    v->zh_trailing_ = char_set(model, "zh_trailing");
    v->zh_trailing_to_ = string(model, "zh_trailing_to");
    v->zh_enders_ = char_set(model, "zh_enders");
    v->en_enders_ = char_set(model, "en_enders");
    v->closers_ = char_set(model, "closers");
    v->zh_terminal_ = to_u32(string(model, "zh_terminal"));
    v->en_terminal_ = to_u32(string(model, "en_terminal"));
    v->token_max_n_ = positive(model, "token_max_n");
    v->token_min_n_ = positive(model, "token_min_n");
    v->merge_len_ = positive(model, "merge_len");
    v->digits_ = char_set(model, "digits");
    const std::vector<std::string> decimal_from = strings(model, "decimal_from");
    const std::vector<int32_t> decimal_value = ints(model, "decimal_value");
    if (decimal_from.size() != decimal_value.size()) {
        throw LoadError("CosyVoice3Vocab::load: decimal_from and decimal_value differ in length");
    }
    for (size_t i = 0; i < decimal_from.size(); ++i) {
        if (decimal_value[i] < 0 || decimal_value[i] > 9) {
            throw LoadError("CosyVoice3Vocab::load: decimal_value holds a value that is not a digit");
        }
        v->decimal_.emplace(one_char(decimal_from[i], "decimal_from"), decimal_value[i]);
    }
    v->units_ = strings(model, "num_units");
    v->teens_ = strings(model, "num_teens");
    v->tens_ = strings(model, "num_tens");
    v->scales_ = strings(model, "num_scales");
    if (v->units_.size() != 10 || v->teens_.size() != 10 || v->tens_.size() != 10 || v->scales_.empty()) {
        throw LoadError("CosyVoice3Vocab::load: the number-word tables must hold 10 units, 10 teens, 10 tens "
                         "and at least one scale");
    }
    v->hundred_ = string(model, "num_hundred");
    v->and_word_ = string(model, "num_and");
    v->zero_word_ = string(model, "num_zero");
    v->one_word_ = string(model, "num_one");
    return v;
}

const std::string& CosyVoice3Vocab::scale(size_t index) const {
    if (index >= scales_.size()) {
        throw Error("CosyVoice3Vocab: a number with more digits than inflect has scale words for (the "
                    "reference raises NumOutOfRangeError); write it out, or split it");
    }
    return scales_[index];
}

std::string CosyVoice3Vocab::number_to_words(const std::u32string& run) const {
    // `_handle_chunk`: `NON_DIGIT.sub("", chunk)` keeps only `\d` digits, and an emptied chunk is "0".
    struct Digit {
        int value;
        bool ascii_zero;
    };
    std::vector<Digit> digits;
    for (char32_t cp : run) {
        const auto it = decimal_.find(cp);
        if (it != decimal_.end()) digits.push_back({it->second, cp == U'0'});
    }
    if (digits.empty()) digits.push_back({0, true});

    // `enword(chunk, 0)`: `int(num) == 0` / `== 1` first.
    size_t first_nonzero = 0;
    while (first_nonzero < digits.size() && digits[first_nonzero].value == 0) ++first_nonzero;
    std::string raw;
    if (first_nonzero == digits.size()) {
        raw = zero_word_;
    } else if (first_nonzero == digits.size() - 1 && digits.back().value == 1) {
        raw = one_word_;
    } else {
        // `num.lstrip("0")`: ASCII zeros only -- another script's zero stays, and is spoken as a group.
        size_t begin = 0;
        while (begin < digits.size() && digits[begin].ascii_zero) ++begin;
        auto tenfn = [&](int tens, int units, size_t mindex) -> std::string {
            if (tens != 1) {
                return tens_[tens] + (tens && units ? "-" : "") + units_[units] + scale(mindex);
            }
            return teens_[units] + scale(mindex);
        };
        // The regexes replace from the RIGHT: `THREE_DIGITS_WORD` is the last three digits, then the next
        // three, and what is left (one or two) goes through `TWO_`/`ONE_DIGIT_WORD` at the next scale.
        std::string tail;
        size_t end = digits.size();
        size_t mill = 0;
        while (end - begin >= 3) {
            const int h = digits[end - 3].value, t = digits[end - 2].value, u = digits[end - 1].value;
            std::string group;
            if (h) {
                const std::string andword = (t || u) ? " " + and_word_ + " " : "";
                group = units_[h] + hundred_ + andword + tenfn(t, u, 0) + scale(mill) + ", ";
            } else if (t || u) {
                group = tenfn(t, u, 0) + scale(mill) + ", ";
            }
            tail = group + tail;
            ++mill;
            end -= 3;
        }
        if (end - begin == 2) {
            tail = tenfn(digits[begin].value, digits[begin + 1].value, mill) + ", " + tail;
        } else if (end - begin == 1) {
            tail = units_[digits[begin].value] + scale(mill) + ", " + tail;
        }
        raw = tail;
    }

    // `_handle_chunk`'s clean-up, in order. Every whitespace character here is an ASCII space: the
    // strings are the tables' words and the separators above.
    if (raw.size() >= 2 && raw.compare(raw.size() - 2, 2, ", ") == 0) raw.resize(raw.size() - 2);
    // WHITESPACES_COMMA, `\s+,` -> ",".
    std::string s;
    for (size_t i = 0; i < raw.size();) {
        if (raw[i] == ' ') {
            size_t j = i;
            while (j < raw.size() && raw[j] == ' ') ++j;
            if (j < raw.size() && raw[j] == ',') {
                i = j;
                continue;
            }
            s.append(raw, i, j - i);
            i = j;
        } else {
            s.push_back(raw[i++]);
        }
    }
    // COMMA_WORD, `, (\S+)\s+\Z` -> " and \1": the last word, when the text ENDS in whitespace and a
    // ", " comes right before that word.
    if (!s.empty() && s.back() == ' ') {
        size_t trimmed = s.size();
        while (trimmed > 0 && s[trimmed - 1] == ' ') --trimmed;
        const size_t word = s.rfind(' ', trimmed == 0 ? 0 : trimmed - 1);
        const size_t word_begin = word == std::string::npos ? 0 : word + 1;
        if (trimmed > word_begin && word_begin >= 2 && s[word_begin - 1] == ' ' && s[word_begin - 2] == ',') {
            s = s.substr(0, word_begin - 2) + " " + and_word_ + " " + s.substr(word_begin, trimmed - word_begin);
        }
    }
    // WHITESPACES, `\s+` -> " ", then strip.
    std::string out;
    for (char c : s) {
        if (c == ' ' && !out.empty() && out.back() == ' ') continue;
        out.push_back(c);
    }
    const size_t b = out.find_first_not_of(' ');
    if (b == std::string::npos) return {};
    return out.substr(b, out.find_last_not_of(' ') - b + 1);
}

std::string CosyVoice3Vocab::spell_out_number(const std::string& text) const {
    const std::u32string cps = to_u32(text);
    std::string out;
    size_t i = 0;
    while (i < cps.size()) {
        if (digits_.count(cps[i])) {
            size_t j = i;
            while (j < cps.size() && digits_.count(cps[j])) ++j;
            out += number_to_words(cps.substr(i, j - i));
            i = j;
        } else {
            out += to_utf8(cps.substr(i, 1));
            ++i;
        }
    }
    return out;
}

bool CosyVoice3Vocab::contains_chinese(const std::u32string& text) const {
    for (char32_t cp : text) {
        if (in_ranges(zh_ranges_, cp)) return true;
    }
    return false;
}

bool CosyVoice3Vocab::is_only_punctuation(const std::u32string& text) const {
    for (char32_t cp : text) {
        if (!in_ranges(punct_ranges_, cp)) return false;
    }
    return true;
}

size_t CosyVoice3Vocab::length(const std::u32string& text, bool zh) const {
    return zh ? text.size() : bpe_->encode(to_utf8(text)).size();
}

std::vector<std::u32string> CosyVoice3Vocab::split_paragraph(std::u32string text, bool zh) const {
    const auto& enders = zh ? zh_enders_ : en_enders_;
    if (text.empty()) throw Error("CosyVoice3Vocab: nothing to say after normalisation");
    if (!enders.count(text.back())) text += zh ? zh_terminal_ : en_terminal_;

    std::vector<std::u32string> utts;
    size_t st = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        if (!enders.count(text[i])) continue;
        if (i > st) utts.push_back(text.substr(st, i - st) + text[i]);
        if (i + 1 < text.size() && closers_.count(text[i + 1])) {
            // The reference pops the LAST utterance, which is the previous sentence when this ender
            // closed nothing -- and an IndexError when there is none.
            if (utts.empty()) {
                throw Error("CosyVoice3Vocab: a closing quote after a sentence ender that ends no sentence "
                            "(the reference's split_paragraph raises IndexError on this text)");
            }
            utts.back() += text[i + 1];
            st = i + 2;
        } else {
            st = i + 1;
        }
    }

    std::vector<std::u32string> out;
    std::u32string cur;
    for (const std::u32string& utt : utts) {
        if (length(cur + utt, zh) > token_max_n_ && length(cur, zh) > token_min_n_) {
            out.push_back(cur);
            cur.clear();
        }
        cur += utt;
    }
    if (!cur.empty()) {
        if (length(cur, zh) < merge_len_ && !out.empty()) out.back() += cur;
        else out.push_back(cur);
    }
    return out;
}

std::vector<std::string> CosyVoice3Vocab::normalize(const std::string& text, bool* markup) const {
    *markup = text.find(markup_open_) != std::string::npos && text.find(markup_close_) != std::string::npos;
    if (*markup) return {text};
    if (text.empty()) throw Error("CosyVoice3Vocab: the text is empty");

    std::u32string cps = to_u32(text);
    size_t b = 0, e = cps.size();
    while (b < e && is_python_space(cps[b])) ++b;
    while (e > b && is_python_space(cps[e - 1])) --e;
    cps = cps.substr(b, e - b);
    if (cps.empty()) throw Error("CosyVoice3Vocab: the text is only whitespace");

    std::vector<std::u32string> pieces;
    if (contains_chinese(cps)) {
        std::string s = to_utf8(cps);
        for (size_t i = 0; i < zh_pre_from_.size(); ++i) replace_all(s, zh_pre_from_[i], zh_pre_to_[i]);
        // `replace_blank`: a space stays only between two ASCII non-spaces. `text[i - 1]` at i == 0 is
        // Python's last character, and `text[i + 1]` past the end raises.
        const std::u32string u = to_u32(s);
        std::u32string kept;
        for (size_t i = 0; i < u.size(); ++i) {
            if (u[i] != U' ') {
                kept += u[i];
                continue;
            }
            if (i + 1 >= u.size()) {
                throw Error("CosyVoice3Vocab: a trailing space after the newlines were dropped (the reference's "
                            "replace_blank raises IndexError on this text)");
            }
            const char32_t next = u[i + 1];
            const char32_t prev = i == 0 ? u.back() : u[i - 1];
            if (next < 0x80 && next != U' ' && prev < 0x80 && prev != U' ') kept += u[i];
        }
        s = to_utf8(kept);
        for (size_t i = 0; i < zh_replace_from_.size(); ++i) replace_all(s, zh_replace_from_[i], zh_replace_to_[i]);
        // `re.sub(r'[，,、]+$', '。', text)`.
        std::u32string z = to_u32(s);
        size_t end = z.size();
        while (end > 0 && zh_trailing_.count(z[end - 1])) --end;
        if (end < z.size()) z = z.substr(0, end) + to_u32(zh_trailing_to_);
        pieces = split_paragraph(z, /*zh=*/true);
    } else {
        pieces = split_paragraph(to_u32(spell_out_number(to_utf8(cps))), /*zh=*/false);
    }

    std::vector<std::string> out;
    for (const std::u32string& piece : pieces) {
        if (!is_only_punctuation(piece)) out.push_back(to_utf8(piece));
    }
    return out;
}

std::vector<std::string> CosyVoice3Vocab::chunks(const std::string& text) const {
    bool markup = false;
    return normalize(text, &markup);
}

std::vector<int32_t> CosyVoice3Vocab::encode(const std::string& text) const {
    bool markup = false;
    const std::vector<std::string> pieces = normalize(text, &markup);
    if (pieces.empty()) {
        throw Error("CosyVoice3Vocab: the text is only punctuation, so there is nothing to say (the reference "
                    "synthesises no audio for it)");
    }
    std::vector<int32_t> ids;
    for (const std::string& piece : pieces) {
        const std::vector<int32_t> piece_ids = bpe_->encode(piece);
        if (markup && std::find(piece_ids.begin(), piece_ids.end(), header_) != piece_ids.end()) {
            throw Error("CosyVoice3Vocab: the text spells '" + bpe_->id_to_piece(header_) +
                        "', the id that separates chunks; remove it");
        }
        ids.push_back(header_);
        ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
    }
    return ids;
}

std::string CosyVoice3Vocab::decode(const std::vector<int32_t>& ids) const {
    std::vector<int32_t> kept;
    for (int32_t id : ids) {
        if (id != header_) kept.push_back(id);
    }
    return bpe_->decode(kept);
}

} // namespace loom
