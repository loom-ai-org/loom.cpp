#include "loom/core/number_speller.h"

#include "loom/core/gguf_model.h"
#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

#include <algorithm>

namespace loom {

namespace {

const std::string kPrefix = "tokenizer.ggml.numbers.";

std::vector<char32_t> single_codepoints(const std::vector<std::string>& items, const std::string& key) {
    std::vector<char32_t> out;
    for (const auto& s : items) {
        const auto cps = utf8_decode(s);
        if (cps.size() != 1) throw LoadError("NumberSpeller: " + key + " holds '" + s + "', not one codepoint");
        out.push_back(cps[0]);
    }
    return out;
}

bool starts_with(const std::u32string& s, size_t at, char32_t c) { return at < s.size() && s[at] == c; }

} // namespace

std::unique_ptr<NumberSpeller> NumberSpeller::load(const GgufModel& model) {
    if (!model.has_kv(kPrefix + "scheme")) return nullptr;
    const std::string scheme = model.kv_str(kPrefix + "scheme");
    if (scheme != "english_number_normalizer") {
        throw LoadError("NumberSpeller: unknown scheme '" + scheme + "'; this engine spells numbers only "
                        "the way 'english_number_normalizer' does, and a newer file needs a newer engine");
    }
    std::unique_ptr<NumberSpeller> s(new NumberSpeller());
    s->ones_ = model.kv_arr_str(kPrefix + "ones");
    s->teens_ = model.kv_arr_str(kPrefix + "teens");
    s->tens_ = model.kv_arr_str(kPrefix + "tens");
    s->scales_ = model.kv_arr_str(kPrefix + "scales");
    if (s->ones_.size() != 10 || s->teens_.size() != 10 || s->tens_.size() != 10 || s->scales_.empty()) {
        throw LoadError("NumberSpeller: ones/teens/tens must hold 10 words each and scales at least one");
    }
    s->chain_ = single_codepoints(model.kv_arr_str(kPrefix + "symbol_chain"), "symbol_chain");
    s->currency_symbols_ = single_codepoints(model.kv_arr_str(kPrefix + "currency_symbols"), "currency_symbols");
    s->currency_names_ = model.kv_arr_str(kPrefix + "currency_names");
    if (s->currency_names_.size() != s->currency_symbols_.size()) {
        throw LoadError("NumberSpeller: currency_symbols and currency_names differ in length");
    }
    for (int32_t z : model.kv_arr_i32(kPrefix + "digit_zeros")) s->digit_zeros_.push_back(static_cast<char32_t>(z));
    std::sort(s->digit_zeros_.begin(), s->digit_zeros_.end());
    const auto ranges = model.kv_arr_i32(kPrefix + "word_ranges");
    if (ranges.size() % 2 != 0) throw LoadError("NumberSpeller: word_ranges must hold [lo, hi] pairs");
    for (size_t i = 0; i < ranges.size(); i += 2) {
        s->word_ranges_.emplace_back(static_cast<char32_t>(ranges[i]), static_cast<char32_t>(ranges[i + 1]));
    }
    std::sort(s->word_ranges_.begin(), s->word_ranges_.end());
    return s;
}

// `re`'s `\w` for a str pattern: `c.isalnum() or c == '_'`, as the exporter tabulated it.
bool NumberSpeller::is_word(char32_t c) const {
    auto it = std::upper_bound(word_ranges_.begin(), word_ranges_.end(), std::make_pair(c, char32_t(0x10FFFF)));
    if (it == word_ranges_.begin()) return false;
    --it;
    return c >= it->first && c <= it->second;
}

// `re`'s `\d` (category Nd), valued as `int()` values it. Every Nd run is ten consecutive codepoints.
int NumberSpeller::digit_value(char32_t c) const {
    auto it = std::upper_bound(digit_zeros_.begin(), digit_zeros_.end(), c);
    if (it == digit_zeros_.begin()) return -1;
    --it;
    return c - *it < 10 ? static_cast<int>(c - *it) : -1;
}

// The pattern `(?<!\w)(<chain>\d+(?:\.\d{1,2})?%?)(?!\w)` at position `i`, with `re`'s backtracking
// order spelled out. The chain's symbols are each optional and consumed greedily; a skipped symbol
// can never help, since the digits would then have to start on it. A shorter digit run can never
// help either: the character after it is a digit, which is `\w`. So the only choices are the
// decimal group (two digits, one, none) and the percent sign (with, without), in that order.
size_t NumberSpeller::match_at(const std::u32string& t, size_t i) const {
    const size_t n = t.size();
    if (i > 0 && is_word(t[i - 1])) return 0;
    size_t j = i;
    for (char32_t sym : chain_) {
        if (starts_with(t, j, sym)) ++j;
    }
    size_t k = j;
    while (k < n && digit_value(t[k]) >= 0) ++k;
    if (k == j) return 0;
    std::vector<size_t> ends;
    if (starts_with(t, k, U'.') && k + 1 < n && digit_value(t[k + 1]) >= 0) {
        if (k + 2 < n && digit_value(t[k + 2]) >= 0) ends.push_back(k + 3);
        ends.push_back(k + 2);
    }
    ends.push_back(k);
    for (size_t e : ends) {
        for (int with_percent = 1; with_percent >= 0; --with_percent) {
            size_t p = e;
            if (with_percent) {
                if (!starts_with(t, e, U'%')) continue;
                ++p;
            }
            if (p >= n || !is_word(t[p])) return p - i;
        }
    }
    return 0;
}

// `EnglishNumberNormalizer.spell_number` for 0..999.
std::string NumberSpeller::spell(int num) const {
    if (num == 0) return "zero";
    std::string part;
    const int hundreds = num / 100, tens_units = num % 100;
    if (hundreds > 0) {
        part += ones_[hundreds] + " hundred";
        if (tens_units > 0) part += " and ";
    }
    if (tens_units > 10 && tens_units < 20) {
        part += teens_[tens_units - 10];
    } else {
        const std::string& t = tens_[tens_units / 10];
        const std::string& o = ones_[tens_units % 10];
        part += t;
        if (!o.empty()) {
            if (!t.empty()) part += " ";
            part += o;
        }
    }
    return part;
}

// `EnglishNumberNormalizer.convert`. False where the reference raises; the caller keeps the text.
bool NumberSpeller::convert(const std::u32string& number, std::string& out) const {
    std::u32string integer = number, decimal = U"00";
    if (const size_t dot = number.find(U'.'); dot != std::u32string::npos) {
        integer = number.substr(0, dot);
        decimal = number.substr(dot + 1);
    }
    std::string currency;
    for (size_t s = 0; s < currency_symbols_.size(); ++s) {
        const char32_t sym = currency_symbols_[s];
        if (starts_with(integer, 0, sym)) {
            currency = currency_names_[s];
            integer = integer.substr(1);
            break;
        }
        if (starts_with(integer, 0, U'-') && starts_with(integer, 1, sym)) {
            currency = currency_names_[s];
            integer = U"-" + integer.substr(2);
            break;
        }
    }
    std::string minus;
    if (starts_with(integer, 0, U'-')) {
        minus = "minus ";
        integer = integer.substr(1);
    }
    std::string percent;
    if (integer.find(U'%') != std::u32string::npos || decimal.find(U'%') != std::u32string::npos) {
        percent = " percent";
        integer.erase(std::remove(integer.begin(), integer.end(), U'%'), integer.end());
        decimal.erase(std::remove(decimal.begin(), decimal.end(), U'%'), decimal.end());
    }
    std::vector<int> digits;
    for (char32_t c : integer) {
        const int d = digit_value(c);
        if (d < 0) return false;                 // a second currency symbol: `int()` raises upstream
        digits.push_back(d);
    }
    if (digits.empty()) return false;
    const size_t groups = (digits.size() + 2) / 3;
    if (groups > scales_.size()) return false;   // past the last scale word: an IndexError upstream
    digits.insert(digits.begin(), groups * 3 - digits.size(), 0);
    std::string spelled;
    for (size_t g = 0; g < groups; ++g) {
        const int chunk = digits[3 * g] * 100 + digits[3 * g + 1] * 10 + digits[3 * g + 2];
        if (chunk == 0) continue;
        std::string part = spell(chunk);
        const std::string& unit = scales_[groups - g - 1];
        if (!unit.empty()) part += " " + unit;
        if (!spelled.empty()) spelled += " ";
        spelled += part;
    }
    if (spelled.empty()) spelled = "zero";       // the one deviation: upstream spells 0 as nothing
    if (decimal == U"00") {
        out = minus + spelled + percent + currency;
        return true;
    }
    std::string spelled_decimal;
    for (char32_t c : decimal) {
        const int d = digit_value(c);
        if (d < 0) return false;
        if (!spelled_decimal.empty()) spelled_decimal += " ";
        spelled_decimal += spell(d);
    }
    out = minus + spelled + " point " + spelled_decimal + percent + currency;
    return true;
}

std::string NumberSpeller::apply(const std::string& text) const {
    const auto cps = utf8_decode(text);
    std::u32string src(cps.begin(), cps.end());

    // 1. Thousands separators: drop every ',' with a digit on both sides. Upstream's single
    //    `re.sub(r"(\d+,\d+)", ...)` never rescans a replacement, so "1,000,000" came out "1000,000" and
    //    was spoken "one thousand" and then nothing; this is that substitution run to its fixed point.
    std::u32string t;
    t.reserve(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        if (src[i] == U',' && i > 0 && i + 1 < src.size() && digit_value(src[i - 1]) >= 0 &&
            digit_value(src[i + 1]) >= 0) {
            continue;
        }
        t += src[i];
    }

    // 2. The number pattern, each match replaced by `convert`, scanning the text after step 1.
    std::string out;
    std::vector<char32_t> pending;
    auto flush = [&]() {
        out += utf8_encode(pending);
        pending.clear();
    };
    for (size_t i = 0; i < t.size();) {
        const size_t len = match_at(t, i);
        std::string spelled;
        if (len > 0 && convert(t.substr(i, len), spelled)) {
            flush();
            out += spelled;
            i += len;
        } else if (len > 0) {
            pending.insert(pending.end(), t.begin() + i, t.begin() + i + len);
            i += len;
        } else {
            pending.push_back(t[i++]);
        }
    }
    flush();

    // 3. `re.sub(" +", " ", ...)` over the whole result.
    std::string collapsed;
    collapsed.reserve(out.size());
    for (char c : out) {
        if (c == ' ' && !collapsed.empty() && collapsed.back() == ' ') continue;
        collapsed += c;
    }
    return collapsed;
}

} // namespace loom
