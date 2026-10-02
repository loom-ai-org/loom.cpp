#include "loom/core/soprano_vocab.h"

#include "loom/core/gguf_model.h"
#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

#include <algorithm>
#include <limits>

namespace loom {
namespace {

const std::string kEmpty;
const std::string kPrefix = "tokenizer.ggml.soprano.";

std::u32string u32(const std::string& s) {
    const auto cps = utf8_decode(s);
    return std::u32string(cps.begin(), cps.end());
}
std::string utf8(const std::u32string& s) { return utf8_encode(std::vector<char32_t>(s.begin(), s.end())); }

// `str.strip()`.
std::u32string strip(const std::u32string& s) {
    size_t b = 0, e = s.size();
    while (b < e && is_python_space(s[b])) ++b;
    while (e > b && is_python_space(s[e - 1])) --e;
    return s.substr(b, e - b);
}

// `str.split(sep)` with a non-empty separator.
std::vector<std::u32string> split(const std::u32string& s, const std::u32string& sep) {
    std::vector<std::u32string> out;
    size_t pos = 0;
    while (true) {
        const size_t hit = s.find(sep, pos);
        if (hit == std::u32string::npos) break;
        out.push_back(s.substr(pos, hit - pos));
        pos = hit + sep.size();
    }
    out.push_back(s.substr(pos));
    return out;
}

std::u32string join(const std::vector<std::u32string>& parts, const std::u32string& sep) {
    std::u32string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) out += sep;
        out += parts[i];
    }
    return out;
}

bool all_digits(const std::u32string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](char32_t c) { return c >= U'0' && c <= U'9'; });
}

// `int(s)` for a string of ASCII digits, kept as its decimal spelling (Python's ints have no width):
// leading zeros dropped, "0" for zero. Anything else is the `ValueError` the reference raises.
std::u32string py_int(const std::u32string& s, const char* where) {
    if (!all_digits(s)) {
        throw Error(std::string("SopranoVocab: ") + where + " is '" + utf8(s) +
                    "', which the reference's int() raises ValueError on");
    }
    size_t b = 0;
    while (b + 1 < s.size() && s[b] == U'0') ++b;
    return s.substr(b);
}

// A small int's value (an hour, a year, a count) -- only called on spellings of at most a few digits.
long small(const std::u32string& digits) {
    long v = 0;
    for (char32_t c : digits) v = v * 10 + static_cast<long>(c - U'0');
    return v;
}

std::vector<std::string> strings(const GgufModel& model, const std::string& key) {
    if (!model.has_kv(kPrefix + key)) {
        throw LoadError("SopranoVocab::load: tokenizer.ggml.model is 'soprano' but " + kPrefix + key +
                         " is missing");
    }
    return model.kv_arr_str(kPrefix + key);
}
std::vector<int32_t> ints(const GgufModel& model, const std::string& key) {
    if (!model.has_kv(kPrefix + key)) {
        throw LoadError("SopranoVocab::load: tokenizer.ggml.model is 'soprano' but " + kPrefix + key +
                         " is missing");
    }
    return model.kv_arr_i32(kPrefix + key);
}
int32_t integer(const GgufModel& model, const std::string& key) {
    if (!model.has_kv(kPrefix + key)) {
        throw LoadError("SopranoVocab::load: tokenizer.ggml.model is 'soprano' but " + kPrefix + key +
                         " is missing");
    }
    return model.kv_i32(kPrefix + key, 0);
}

std::vector<std::pair<PyRegex, std::u32string>> table(const GgufModel& model, const std::string& key) {
    const auto patterns = strings(model, key + "_patterns");
    const auto icase = ints(model, key + "_icase");
    const auto repl = strings(model, key + "_repl");
    if (patterns.size() != icase.size() || patterns.size() != repl.size()) {
        throw LoadError("SopranoVocab::load: " + key + "'s patterns, flags and replacements differ in length");
    }
    std::vector<std::pair<PyRegex, std::u32string>> out;
    for (size_t i = 0; i < patterns.size(); ++i) {
        PyRegex re(patterns[i], icase[i] != 0);
        const std::u32string r = u32(repl[i]);
        re.check_template(r);
        out.emplace_back(std::move(re), r);
    }
    return out;
}

// The rules `clean_text` and the splitter look up by name; a file missing one is refused at load.
const char* const kRequiredRules[] = {
    "num_prefix_re", "num_suffix_re", "num_letter_split_re", "comma_number_re", "date_re",
    "phone_number_re", "time_re", "pounds_re", "dollars_re", "decimal_number_re", "multiply_re",
    "divide_re", "add_re", "subtract_re", "fraction_re", "ordinal_re", "number_re", "link_header_re",
    "dash_re", "dot_re", "parentheses_re", "camelcase_re", "date_split", "phone_non_digit", "paren_open",
    "paren_close_inner", "paren_close", "camel_part", "unknown_chars", "unknown_symbols", "whitespace",
    "space_before_punct", "ellipsis", "commas", "periods", "exclamations", "questions", "ellipsis_back",
    "triple_letters", "split_newlines", "split_whitespace", "split_quotes", "split_empty",
    "inflect_non_digit", "inflect_whitespaces_comma", "inflect_comma_word", "inflect_whitespaces",
};
const char* const kRequiredTemplates[] = {
    "phone_non_digit", "paren_open", "paren_close_inner", "paren_close", "unknown_chars",
    "unknown_symbols", "whitespace", "ellipsis", "commas", "periods", "exclamations", "questions",
    "ellipsis_back", "split_newlines", "split_whitespace", "split_quotes", "pounds_re",
};

} // namespace

SopranoVocab::~SopranoVocab() = default;

std::unique_ptr<SopranoVocab> SopranoVocab::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.ggml.model") || model.kv_str("tokenizer.ggml.model") != "soprano") {
        return nullptr;
    }
    std::unique_ptr<SopranoVocab> v(new SopranoVocab());
    if (!model.has_kv("tokenizer.ggml.tokens") || !model.has_kv("tokenizer.ggml.merges")) {
        throw LoadError("SopranoVocab::load: tokenizer.ggml.tokens or tokenizer.ggml.merges is missing");
    }
    v->tokens_ = model.kv_arr_str("tokenizer.ggml.tokens");
    for (size_t id = 0; id < v->tokens_.size(); ++id) {
        v->piece_to_id_.emplace(v->tokens_[id], static_cast<int32_t>(id));
    }
    const auto merges = model.kv_arr_str("tokenizer.ggml.merges");
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const std::string& m = merges[rank];
        const size_t sp = m.find(' ');
        if (sp == std::string::npos || m.find(' ', sp + 1) != std::string::npos) {
            throw LoadError("SopranoVocab::load: merge '" + m + "' is not 'left right'");
        }
        if (v->piece_to_id_.count(m.substr(0, sp) + m.substr(sp + 1)) == 0) {
            throw LoadError("SopranoVocab::load: merge '" + m + "' produces a piece with no row");
        }
        v->merge_rank_.emplace(m, static_cast<int32_t>(rank));
    }
    v->unk_id_ = model.kv_i32("tokenizer.ggml.unknown_token_id", 0);
    v->stop_id_ = integer(model, "stop_id");
    v->text_id_ = integer(model, "text_id");
    v->start_id_ = integer(model, "start_id");
    for (int32_t id : {v->unk_id_, v->stop_id_, v->text_id_, v->start_id_}) {
        if (id < 0 || static_cast<size_t>(id) >= v->tokens_.size()) {
            throw LoadError("SopranoVocab::load: a special id is outside the vocabulary");
        }
    }

    const auto names = strings(model, "rule_names");
    const auto patterns = strings(model, "rule_patterns");
    const auto icase = ints(model, "rule_icase");
    if (names.size() != patterns.size() || names.size() != icase.size()) {
        throw LoadError("SopranoVocab::load: rule_names, rule_patterns and rule_icase differ in length");
    }
    for (size_t i = 0; i < names.size(); ++i) v->rules_.emplace(names[i], PyRegex(patterns[i], icase[i] != 0));
    for (const char* name : kRequiredRules) {
        if (v->rules_.count(name) == 0) throw LoadError(std::string("SopranoVocab::load: no rule '") + name + "'");
    }
    const auto tnames = strings(model, "template_names");
    const auto tvalues = strings(model, "template_values");
    if (tnames.size() != tvalues.size()) {
        throw LoadError("SopranoVocab::load: template_names and template_values differ in length");
    }
    for (size_t i = 0; i < tnames.size(); ++i) {
        const auto it = v->rules_.find(tnames[i]);
        if (it == v->rules_.end()) {
            throw LoadError("SopranoVocab::load: template '" + tnames[i] + "' names no rule");
        }
        const std::u32string t = u32(tvalues[i]);
        it->second.check_template(t);
        v->templates_.emplace(tnames[i], t);
    }
    for (const char* name : kRequiredTemplates) {
        if (v->templates_.count(name) == 0) {
            throw LoadError(std::string("SopranoVocab::load: no template for rule '") + name + "'");
        }
    }
    for (auto& [re, repl] : table(model, "preunicode")) v->preunicode_.push_back({std::move(re), repl});
    for (auto& [re, repl] : table(model, "abbreviations")) v->abbreviations_.push_back({std::move(re), repl});
    for (auto& [re, repl] : table(model, "special")) v->special_.push_back({std::move(re), repl});

    v->unit_ = strings(model, "inflect_unit");
    v->teen_ = strings(model, "inflect_teen");
    v->ten_ = strings(model, "inflect_ten");
    v->mill_ = strings(model, "inflect_mill");
    v->nth_suffixes_ = strings(model, "inflect_nth_suffixes");
    if (v->unit_.size() != 10 || v->teen_.size() != 10 || v->ten_.size() != 10 || v->mill_.empty()) {
        throw LoadError("SopranoVocab::load: inflect's tables must hold 10 units, 10 teens, 10 tens and a "
                         "scale");
    }
    const auto ord_from = strings(model, "inflect_ordinal_from");
    const auto ord_to = strings(model, "inflect_ordinal_to");
    if (ord_from.size() != ord_to.size()) {
        throw LoadError("SopranoVocab::load: inflect_ordinal_from and _to differ in length");
    }
    for (size_t i = 0; i < ord_from.size(); ++i) v->ordinal_.emplace_back(ord_from[i], ord_to[i]);

    v->desired_length_ = static_cast<size_t>(integer(model, "desired_length"));
    v->max_length_ = static_cast<size_t>(integer(model, "max_length"));
    v->min_length_ = static_cast<size_t>(integer(model, "min_length"));

    const auto cps = ints(model, "unidecode_cps");
    v->unidecode_offsets_ = ints(model, "unidecode_offsets");
    v->unidecode_text_ = u32(model.kv_str(kPrefix + "unidecode_text"));
    if (v->unidecode_offsets_.size() != cps.size() + 1 ||
        static_cast<size_t>(v->unidecode_offsets_.back()) != v->unidecode_text_.size()) {
        throw LoadError("SopranoVocab::load: the unidecode table's offsets do not fit its codepoints and text");
    }
    v->unidecode_cps_.assign(cps.begin(), cps.end());
    if (!std::is_sorted(v->unidecode_cps_.begin(), v->unidecode_cps_.end())) {
        throw LoadError("SopranoVocab::load: the unidecode table's codepoints are not sorted");
    }
    return v;
}

const PyRegex& SopranoVocab::rule(const char* name) const { return rules_.at(name); }
const std::u32string& SopranoVocab::templ(const char* name) const { return templates_.at(name); }

std::u32string SopranoVocab::apply(const std::vector<Rule>& table, std::u32string text) const {
    for (const Rule& r : table) text = r.re.sub(text, r.repl);
    return text;
}

const std::string& SopranoVocab::id_to_piece(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) return kEmpty;
    return tokens_[static_cast<size_t>(id)];
}

// ----------------------------------------------------------------------------------------- unidecode --

std::u32string SopranoVocab::unidecode(const std::u32string& text) const {
    std::u32string out;
    for (char32_t c : text) {
        if (c < 0x80) {
            out += c;
            continue;
        }
        const auto it = std::lower_bound(unidecode_cps_.begin(), unidecode_cps_.end(), c);
        if (it == unidecode_cps_.end() || *it != c) continue;    // `errors='ignore'`: an unmapped one is ''
        const size_t i = static_cast<size_t>(it - unidecode_cps_.begin());
        out.append(unidecode_text_, static_cast<size_t>(unidecode_offsets_[i]),
                   static_cast<size_t>(unidecode_offsets_[i + 1] - unidecode_offsets_[i]));
    }
    return out;
}

// ------------------------------------------------------------------------------------------- inflect --

const std::string& SopranoVocab::mill(size_t index) const {
    if (index >= mill_.size()) {
        throw Error("SopranoVocab: a number with more digits than inflect has scale words for (the reference "
                    "raises NumOutOfRangeError)");
    }
    return mill_[index];
}

// `tenfn`: the tens and units of one group, then the group's scale word (`mill[0]` is a single space).
std::string SopranoVocab::tenfn(int tens, int units, size_t mindex) const {
    if (tens != 1) {
        return ten_[static_cast<size_t>(tens)] + (tens && units ? "-" : "") + unit_[static_cast<size_t>(units)] +
               mill(mindex);
    }
    return teen_[static_cast<size_t>(units)] + mill(mindex);
}

std::string SopranoVocab::sub_ord(const std::string& value) const {
    // `ordinal_suff.sub(...)`: `(ty|one|two|...)\Z` -- at most one of them can end a word.
    for (const auto& [from, to] : ordinal_) {
        if (value.size() >= from.size() && value.compare(value.size() - from.size(), from.size(), from) == 0) {
            return value.substr(0, value.size() - from.size()) + to;
        }
    }
    return value + "th";
}

// `number_to_words(digits, group=group, andword=andword, zero=zero)` for a string of ASCII digits (the
// ordinal suffix already removed): `enword`, `_handle_chunk`'s clean-up, the split on ", " and the
// render. Every whitespace character here is an ASCII space from the tables.
std::string SopranoVocab::words(const std::string& digits, int group, const std::string& andword,
                                const std::string& zero, bool ordinal) const {
    std::string chunk = digits.empty() ? "0" : digits;
    std::string raw;
    auto d = [&](size_t i) { return static_cast<int>(chunk[i] - '0'); };
    if (group == 2) {
        // `TWO_DIGITS.sub(group2sub, num)`, then `DIGIT_GROUP.sub(group1bsub, num, 1)` on what is left.
        size_t i = 0;
        for (; i + 1 < chunk.size(); i += 2) {
            const int t = d(i), u = d(i + 1);
            if (t) raw += tenfn(t, u, 0) + ", ";
            else if (u) raw += " " + zero + " " + unit_[static_cast<size_t>(u)] + ", ";
            else raw += " " + zero + " " + zero + ", ";
        }
        if (i < chunk.size()) raw += d(i) ? unit_[static_cast<size_t>(d(i))] + ", " : " " + zero + ", ";
    } else {
        size_t first = 0;
        while (first < chunk.size() && chunk[first] == '0') ++first;
        if (first == chunk.size()) {
            raw = zero;
        } else if (first + 1 == chunk.size() && chunk.back() == '1') {
            raw = unit_[1];     // `self._number_args["one"]`, whose default is "one"
        } else {
            // `THREE_DIGITS_WORD` takes the LAST three digits of the leading run each time, the scale
            // counting up; then `TWO_DIGITS_WORD` and `ONE_DIGIT_WORD` take what is left at the next one.
            const std::string num = chunk.substr(first);
            std::string tail;
            size_t end = num.size(), mill_count = 0;
            auto at = [&](size_t i) { return static_cast<int>(num[i] - '0'); };
            while (end >= 3) {
                const int h = at(end - 3), t = at(end - 2), u = at(end - 1);
                std::string g;
                if (h) {
                    const std::string and_part = (t || u) ? " " + andword + " " : "";
                    g = unit_[static_cast<size_t>(h)] + " hundred" + and_part + tenfn(t, u, 0) + mill(mill_count) + ", ";
                } else if (t || u) {
                    g = tenfn(t, u, 0) + mill(mill_count) + ", ";
                }
                tail = g + tail;
                ++mill_count;
                end -= 3;
            }
            if (end == 2) tail = tenfn(at(0), at(1), mill_count) + ", " + tail;
            else if (end == 1) tail = unit_[static_cast<size_t>(at(0))] + mill(mill_count) + ", " + tail;
            raw = tail;
        }
    }

    std::u32string c = u32(raw);
    if (c.size() >= 2 && c.compare(c.size() - 2, 2, U", ") == 0) c.resize(c.size() - 2);
    c = rule("inflect_whitespaces_comma").sub(c, U",");
    if (group == 0) c = rule("inflect_comma_word").sub(c, u32(" " + andword + " \\1"));
    c = rule("inflect_whitespaces").sub(c, U" ");
    c = strip(c);

    std::vector<std::u32string> chunks = split(c, U", ");
    if (ordinal && !chunks.empty()) chunks.back() = u32(sub_ord(utf8(chunks.back())));
    // `_render` (group 0) and `', '.join` (group 2) agree when no chunk is the decimal word.
    return utf8(join(chunks, U", "));
}

std::string SopranoVocab::number_to_words(const std::string& digits, bool ordinal) const {
    return words(digits, 0, "", "zero", ordinal);
}

// `_expand_number`, on the digits of `int(m.group(0))`.
std::string SopranoVocab::expand_number(const std::string& digits_in) const {
    const std::string digits = utf8(py_int(u32(digits_in), "a number"));
    if (digits.size() == 4) {
        const long num = small(u32(digits));
        if (num > 1000 && num < 3000) {
            if (num == 2000) return "two thousand";
            // `_inflect.number_to_words(...)` with the DEFAULT `andword="and"` in these two arms, which
            // cannot show: 1..9 and 11..29 have no hundreds.
            if (num > 2000 && num < 2010) return "two thousand " + words(std::to_string(num % 100), 0, "and", "zero", false);
            if (num % 100 == 0) return words(std::to_string(num / 100), 0, "and", "zero", false) + " hundred";
            std::u32string w = u32(words(digits, 2, "", "oh", false));
            return utf8(join(split(w, U", "), U" "));
        }
    }
    return words(digits, 0, "", "zero", false);
}

// ------------------------------------------------------------------------------------- clean_text --

std::u32string SopranoVocab::normalize_numbers(std::u32string text) const {
    using M = PyRegex::Match;
    text = rule("num_prefix_re").sub(text, [](const std::u32string& s, const M& m) {
        return U"number " + std::u32string(1, s[m.begin + 1]);
    });
    text = rule("num_suffix_re").sub(text, [](const std::u32string& s, const M& m) {
        const std::u32string match = s.substr(m.begin, m.end - m.begin);
        const std::u32string head = match.substr(0, match.size() - 1);
        switch (match.back()) {
            case U'K': case U'k': return head + U" thousand";
            case U'M': case U'm': return head + U" million";
            case U'B': case U'b': return head + U" billion";
            case U'T': case U't': return head + U" trillion";
            default: return match;
        }
    });
    text = rule("comma_number_re").sub(text, [](const std::u32string& s, const M& m) {
        std::u32string out;
        for (char32_t c : m.group(s, 1)) {
            if (c != U',') out += c;
        }
        return out;
    });
    text = rule("date_re").sub(text, [this](const std::u32string& s, const M& m) {
        // `re.split('[./-]', match)`: the pieces between single-character separators, empties kept.
        const std::u32string date = m.group(s, 2);
        std::vector<std::u32string> parts;
        size_t pos = 0;
        PyRegex::Match sep;
        while (rule("date_split").search(date, pos, sep)) {
            parts.push_back(date.substr(pos, sep.begin - pos));
            pos = sep.end;
        }
        parts.push_back(date.substr(pos));
        return m.group(s, 1) + join(parts, U" dash ") + m.group(s, 3);
    });
    text = rule("phone_number_re").sub(text, [this](const std::u32string& s, const M& m) {
        const std::u32string d = rule("phone_non_digit").sub(m.group(s, 1), templ("phone_non_digit"));
        if (d.size() != 10) throw Error("SopranoVocab: a phone number without ten digits (the reference asserts)");
        auto spaced = [&](size_t b, size_t e) {
            std::u32string out;
            for (size_t i = b; i < e; ++i) {
                if (i > b) out += U' ';
                out += d[i];
            }
            return out;
        };
        return spaced(0, 3) + U", " + spaced(3, 6) + U", " + spaced(6, 10);
    });
    text = rule("time_re").sub(text, [](const std::u32string& s, const M& m) {
        const std::vector<std::u32string> p = split(m.group(s, 1), U":");
        auto oh = [](const std::u32string& x) { return x[0] == U'0'; };
        if (p.size() == 2) {
            const std::u32string& hours = p[0];
            std::u32string minutes = p[1];
            if (minutes == U"00") {
                if (small(hours) == 0) return std::u32string(U"0");
                if (small(hours) > 12) return hours + U" minutes";
                return hours + U" o'clock";
            }
            if (oh(minutes)) minutes = U"oh " + minutes.substr(1);
            return hours + U" " + minutes;
        }
        const std::u32string &hours = p[0], &minutes = p[1], &seconds = p[2];
        if (small(hours) != 0) {
            // `... else {minutes}` inside the f-string is a SET literal: Python renders `{'15'}`.
            const std::u32string mm = minutes == U"00" ? U"oh oh" : oh(minutes) ? U"oh " + minutes
                                                                                : U"{'" + minutes + U"'}";
            const std::u32string ss = seconds == U"00" ? U"" : oh(seconds) ? U"oh " + seconds : seconds;
            return hours + U" " + mm + U" " + ss;
        }
        if (minutes != U"00") {
            return minutes + U" " + (seconds == U"00" ? U"oh oh" : oh(seconds) ? U"oh " + seconds : seconds);
        }
        return seconds;
    });
    text = rule("pounds_re").sub(text, templ("pounds_re"));
    text = rule("dollars_re").sub(text, [](const std::u32string& s, const M& m) {
        const std::u32string match = m.group(s, 1);
        const std::vector<std::u32string> parts = split(match, U".");
        if (parts.size() > 2) return match + U" dollars";
        const std::u32string dollars = parts[0].empty() ? U"0" : py_int(parts[0], "a dollar amount");
        const std::u32string cents = parts.size() > 1 && !parts[1].empty() ? py_int(parts[1], "a cent amount")
                                                                           : U"0";
        const bool has_d = dollars != U"0", has_c = cents != U"0";
        const std::u32string d_unit = dollars == U"1" ? U"dollar" : U"dollars";
        const std::u32string c_unit = cents == U"1" ? U"cent" : U"cents";
        if (has_d && has_c) return dollars + U" " + d_unit + U", " + cents + U" " + c_unit;
        if (has_d) return dollars + U" " + d_unit;
        if (has_c) return cents + U" " + c_unit;
        return std::u32string(U"zero dollars");
    });
    text = rule("decimal_number_re").sub(text, [](const std::u32string& s, const M& m) {
        const std::vector<std::u32string> parts = split(m.group(s, 1), U".");
        std::vector<std::u32string> spelled;
        for (size_t i = 1; i < parts.size(); ++i) {
            std::u32string digits;
            for (size_t j = 0; j < parts[i].size(); ++j) {
                if (j) digits += U' ';
                digits += parts[i][j];
            }
            spelled.push_back(digits);
        }
        return parts[0] + U" point " + join(spelled, U" point ");
    });
    auto infix = [](const std::u32string& op, const std::u32string& word) {
        return [op, word](const std::u32string& s, const M& m) { return join(split(m.group(s, 1), op), word); };
    };
    text = rule("multiply_re").sub(text, infix(U"*", U" times "));
    text = rule("divide_re").sub(text, infix(U"/", U" over "));
    text = rule("add_re").sub(text, infix(U"+", U" plus "));
    text = rule("subtract_re").sub(text, infix(U"-", U" minus "));
    text = rule("fraction_re").sub(text, [](const std::u32string& s, const M& m) {
        const std::vector<std::u32string> parts = split(m.group(s, 1), U"/");
        return join(parts, parts.size() == 2 ? U" over " : U" slash ");
    });
    text = rule("ordinal_re").sub(text, [this](const std::u32string& s, const M& m) {
        // `number_to_words("21st")`: the two-letter suffix is dropped and the last word made ordinal.
        const std::u32string match = s.substr(m.begin, m.end - m.begin);
        return u32(number_to_words(utf8(match.substr(0, match.size() - 2)), true));
    });
    for (int pass = 0; pass < 2; ++pass) {
        text = rule("num_letter_split_re").sub(text, [](const std::u32string& s, const M& m) {
            const std::u32string g = m.group(s, 1);
            return std::u32string(1, g[0]) + U" " + std::u32string(1, g[1]);
        });
    }
    text = rule("number_re").sub(text, [this](const std::u32string& s, const M& m) {
        return u32(expand_number(utf8(s.substr(m.begin, m.end - m.begin))));
    });
    return text;
}

std::u32string SopranoVocab::normalize_special(std::u32string text) const {
    using M = PyRegex::Match;
    text = rule("link_header_re").sub(text, [](const std::u32string&, const M&) {
        return std::u32string(U"h t t p s colon slash slash ");
    });
    text = rule("dash_re").sub(text, [](const std::u32string& s, const M& m) {
        return std::u32string(1, s[m.begin]) + U", " + std::u32string(1, s[m.begin + 4]);
    });
    text = rule("dot_re").sub(text, [](const std::u32string& s, const M& m) {
        return std::u32string(1, s[m.begin]) + U" dot " + std::u32string(1, s[m.begin + 2]);
    });
    text = rule("parentheses_re").sub(text, [this](const std::u32string& s, const M& m) {
        std::u32string match = s.substr(m.begin, m.end - m.begin);
        match = rule("paren_open").sub(match, templ("paren_open"));
        match = rule("paren_close_inner").sub(match, templ("paren_close_inner"));
        return rule("paren_close").sub(match, templ("paren_close"));
    });
    return text;
}

std::u32string SopranoVocab::normalize_mixedcase(const std::u32string& text) const {
    return rule("camelcase_re").sub(text, [this](const std::u32string& s, const PyRegex::Match& m) {
        const std::u32string match = s.substr(m.begin, m.end - m.begin);
        const std::vector<std::u32string> parts = rule("camel_part").findall(match);
        if (parts.size() == 1) return match;                        // a single capitalised word
        if (parts.size() == match.size()) return match;             // all upper case
        if (parts.size() + 1 == match.size() && match.back() == U's') {
            return match.substr(0, match.size() - 1) + U"'s";       // a plural upper-case word
        }
        return join(parts, U" ");
    });
}

std::u32string SopranoVocab::clean(const std::u32string& input) const {
    std::u32string text = apply(preunicode_, input);
    text = unidecode(text);
    // `normalize_newlines`: per line, stripped; a non-empty one ends in [.!?] or gets a full stop; the
    // lines joined by single spaces, empty ones included.
    {
        std::vector<std::u32string> lines = split(text, U"\n");
        for (auto& line : lines) {
            line = strip(line);
            if (line.empty()) continue;
            const char32_t last = line.back();
            if (last != U'.' && last != U'!' && last != U'?') line += U'.';
        }
        text = join(lines, U" ");
    }
    text = normalize_numbers(std::move(text));
    text = normalize_special(std::move(text));
    text = apply(abbreviations_, std::move(text));
    text = normalize_mixedcase(text);
    text = apply(special_, std::move(text));
    for (char32_t& c : text) c = to_lower(c);
    text = rule("unknown_chars").sub(text, templ("unknown_chars"));
    text = rule("unknown_symbols").sub(text, templ("unknown_symbols"));
    text = rule("whitespace").sub(text, templ("whitespace"));
    text = rule("space_before_punct").sub(text, [](const std::u32string& s, const PyRegex::Match& m) {
        return std::u32string(1, s[m.begin + 1]);
    });
    text = strip(text);
    for (const char* name : {"ellipsis", "commas", "periods", "exclamations", "questions", "ellipsis_back"}) {
        text = rule(name).sub(text, templ(name));
    }
    text = rule("triple_letters").sub(text, [](const std::u32string& s, const PyRegex::Match& m) {
        return s.substr(m.begin, 2);
    });
    return text;
}

std::string SopranoVocab::clean_text(const std::string& text) const { return utf8(clean(u32(text))); }

// ------------------------------------------------------------------------------------------ splitter --

std::vector<std::u32string> SopranoVocab::split_and_recombine(std::u32string text) const {
    // tortoise-tts's `split_and_recombine_text`, line for line. `pos` starts at -1; `peek` never sees
    // the last character (`p < end_pos`), and returns "" there -- which `in '...'` then counts as a hit.
    text = rule("split_newlines").sub(text, templ("split_newlines"));
    text = rule("split_whitespace").sub(text, templ("split_whitespace"));
    text = rule("split_quotes").sub(text, templ("split_quotes"));

    std::vector<std::u32string> rv;
    bool in_quote = false;
    std::u32string current;
    std::vector<long> split_pos;
    long pos = -1;
    const long end_pos = static_cast<long>(text.size()) - 1;

    auto seek = [&](long delta) {
        const bool negative = delta < 0;
        for (long i = 0; i < (negative ? -delta : delta); ++i) {
            if (negative) {
                --pos;
                if (!current.empty()) current.pop_back();
            } else {
                ++pos;
                current += text[static_cast<size_t>(pos)];
            }
            if (text[static_cast<size_t>(pos)] == U'"') in_quote = !in_quote;
        }
        return text[static_cast<size_t>(pos)];
    };
    // `peek(delta)` as a codepoint, 0 standing for "".
    auto peek = [&](long delta) -> char32_t {
        const long p = pos + delta;
        return (p < end_pos && p >= 0) ? text[static_cast<size_t>(p)] : 0;
    };
    // `c in chars`, where "" (0) is in every string.
    auto in = [](char32_t c, const std::u32string& chars) { return c == 0 || chars.find(c) != std::u32string::npos; };
    auto commit = [&] {
        rv.push_back(current);
        current.clear();
        split_pos.clear();
    };

    while (pos < end_pos) {
        char32_t c = seek(1);
        if (current.size() >= max_length_) {
            if (!split_pos.empty() && static_cast<double>(current.size()) > desired_length_ / 2.0) {
                seek(-(pos - split_pos.back()));
            } else {
                while (!in(c, U"!?.\n ") && pos > 0 && current.size() > desired_length_) c = seek(-1);
            }
            commit();
        } else if (!in_quote && (in(c, U"!?\n") || (c == U'.' && in(peek(1), U"\n ")))) {
            while (pos < static_cast<long>(text.size()) - 1 && current.size() < max_length_ &&
                   in(peek(1), U"!?.")) {
                c = seek(1);
            }
            split_pos.push_back(pos);
            if (current.size() >= desired_length_) commit();
        } else if (in_quote && peek(1) == U'"' && in(peek(2), U"\n ")) {
            seek(2);
            split_pos.push_back(pos);
        }
    }
    rv.push_back(current);

    std::vector<std::u32string> out;
    PyRegex::Match m;
    for (const auto& s : rv) {
        const std::u32string t = strip(s);
        if (t.empty() || rule("split_empty").match_at(t, 0, m)) continue;
        out.push_back(t);
    }
    return out;
}

std::vector<std::string> SopranoVocab::sentences(const std::string& text) const {
    const std::vector<std::u32string> pieces = split_and_recombine(clean(strip(u32(text))));
    // `_preprocess_text`'s merge: a piece shorter than `min_length` joins the one before it, or, when it
    // is the first, the one after; a lone short piece stays.
    std::vector<std::u32string> processed = pieces, merged;
    if (min_length_ > 0 && processed.size() > 1) {
        for (size_t i = 0; i < processed.size(); ++i) {
            const std::u32string& cur = processed[i];
            if (cur.size() < min_length_) {
                if (!merged.empty()) {
                    merged.back() = strip(merged.back() + U" " + cur);
                } else if (i + 1 < processed.size()) {
                    processed[i + 1] = strip(cur + U" " + processed[i + 1]);
                } else {
                    merged.push_back(cur);
                }
            } else {
                merged.push_back(cur);
            }
        }
        processed = merged;
    }
    std::vector<std::string> out;
    for (const auto& p : processed) out.push_back(utf8(p));
    return out;
}

// ------------------------------------------------------------------------------------------ tokenizer --

void SopranoVocab::bpe(const std::u32string& word, std::vector<int32_t>& ids) const {
    // Symbols start as single codepoints; an unknown one is `[UNK]` and takes part in no merge (no merge
    // names it), which is what `tokenizers` does with `fuse_unk` off.
    struct Sym { std::string piece; bool known; };
    std::vector<Sym> syms;
    for (char32_t cp : word) {
        std::string piece = utf8_encode({cp});
        const bool known = piece_to_id_.count(piece) != 0;
        syms.push_back({std::move(piece), known});
    }
    // Lowest rank first, leftmost among equals -- `tokenizers`' priority queue.
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
    for (const Sym& s : syms) ids.push_back(s.known ? piece_to_id_.at(s.piece) : unk_id_);
}

std::vector<int32_t> SopranoVocab::tokenize(const std::string& text) const {
    // Normalizer: `Lowercase`, then `\s+` -> " " (Rust's `\s`: ASCII's six and the Unicode spaces, NOT
    // Python's `\x1c`-`\x1f`). Pre-tokenizer: every digit alone, then `\s+|\w+|[^\w\s]+` runs.
    std::u32string t;
    bool in_space = false;
    for (char32_t c : u32(text)) {
        const bool space = is_python_space(c) && !(c >= 0x1c && c <= 0x1f);
        if (space) {
            if (!in_space) t += U' ';
            in_space = true;
            continue;
        }
        in_space = false;
        t += to_lower(c);
    }
    auto cls = [](char32_t c) {
        if (is_python_space(c) && !(c >= 0x1c && c <= 0x1f)) return 0;
        if ((c < 0x80 && ((c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z') || (c >= U'0' && c <= U'9') ||
                          c == U'_')) || (c >= 0x80 && is_letter_or_number(c))) {
            return 1;
        }
        return 2;
    };
    std::vector<int32_t> ids;
    size_t start = 0;
    while (start < t.size()) {
        const bool digit = t[start] >= U'0' && t[start] <= U'9';
        size_t end = start + 1;
        if (!digit) {
            const int k = cls(t[start]);
            while (end < t.size() && cls(t[end]) == k && !(t[end] >= U'0' && t[end] <= U'9')) ++end;
        }
        bpe(t.substr(start, end - start), ids);
        start = end;
    }
    return ids;
}

std::vector<int32_t> SopranoVocab::encode(const std::string& text) const {
    const std::vector<std::string> pieces = sentences(text);
    if (pieces.empty()) {
        throw Error("SopranoVocab: the text has nothing left to say after normalisation (the reference "
                    "fails on an empty batch)");
    }
    std::vector<int32_t> ids;
    for (const std::string& p : pieces) {
        ids.push_back(stop_id_);
        ids.push_back(text_id_);
        const std::vector<int32_t> body = tokenize(p);
        ids.insert(ids.end(), body.begin(), body.end());
        ids.push_back(start_id_);
    }
    return ids;
}

std::string SopranoVocab::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        if (id == stop_id_ || id == text_id_ || id == start_id_ || id == unk_id_) continue;
        out += id_to_piece(id);
    }
    return out;
}

} // namespace loom
