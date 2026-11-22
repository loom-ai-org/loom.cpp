// dizione's text rules as kale-tts-1 runs them (frontend/dizione/spoken.js, a port of dizione's normalize.py; and
// frontend/words/kaleTts.js's `normalizer` wrapper), method for method, in upstream's order. The patterns are the
// file's (`rule("...")`, ADR-041); what is here is what each match turns into.

#include "kale_tts_frontend.h"

#include "loom/core/unicode.h"

#include <algorithm>
#include <cmath>

namespace loom {
namespace kale {
namespace {

const char* const kOnes[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                             "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                             "nineteen"};
const char* const kTens[] = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
const char* const kScales[] = {"thousand", "million", "billion", "trillion", "quadrillion"};
const char* const kMonths[] = {"January", "February", "March", "April", "May", "June", "July", "August",
                               "September", "October", "November", "December"};
constexpr char32_t kSpell = 0xE000, kEnd = 0xE001;

bool is_digit(char32_t c) { return c >= U'0' && c <= U'9'; }
bool is_ascii_alpha(char32_t c) { return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z'); }

bool all_digits(const u32& s) {
    if (s.empty()) return false;
    for (char32_t c : s) if (!is_digit(c)) return false;
    return true;
}
bool has_digit(const u32& s) {
    for (char32_t c : s) if (is_digit(c)) return true;
    return false;
}
bool all_letters(const u32& s) {   // /^\p{L}+$/u
    if (s.empty()) return false;
    for (char32_t c : s) if (!is_letter(c)) return false;
    return true;
}

// `Number.parseInt(s, 10)`: the leading digits (dizione's inputs are digit runs); NaN as -1.
long long to_int(const u32& s) {
    size_t i = 0;
    while (i < s.size() && (s[i] == U' ' || s[i] == U'\t' || s[i] == U'\n')) ++i;
    bool neg = false;
    if (i < s.size() && (s[i] == U'-' || s[i] == U'+')) { neg = s[i] == U'-'; ++i; }
    long long v = 0;
    size_t n = 0;
    while (i < s.size() && is_digit(s[i]) && n < 18) { v = v * 10 + (s[i] - U'0'); ++i; ++n; }
    if (n == 0) return -1;
    return neg ? -v : v;
}

u32 num(long long v) { return U(std::to_string(v)); }

u32 join(const std::vector<u32>& parts, const u32& sep) {
    u32 out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) out += sep;
        out += parts[i];
    }
    return out;
}

std::vector<u32> split_on(const u32& s, char32_t c) {
    std::vector<u32> out;
    u32 cur;
    for (char32_t x : s) {
        if (x == c) { out.push_back(cur); cur.clear(); }
        else cur += x;
    }
    out.push_back(cur);
    return out;
}

u32 replace_all(const u32& text, const u32& from, const u32& to) {
    if (from.empty()) return text;
    u32 out;
    size_t pos = 0;
    while (true) {
        const size_t hit = text.find(from, pos);
        if (hit == u32::npos) break;
        out.append(text, pos, hit - pos);
        out += to;
        pos = hit + from.size();
    }
    out.append(text, pos, u32::npos);
    return out;
}

bool starts_with(const u32& s, const u32& p) { return s.size() >= p.size() && s.compare(0, p.size(), p) == 0; }
bool ends_with(const u32& s, const u32& p) { return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0; }

u32 trim(const u32& s) {   // String.prototype.trim (ECMAScript white space)
    auto sp = [](char32_t c) {
        return c == 0x09 || c == 0x0A || c == 0x0B || c == 0x0C || c == 0x0D || c == 0x20 || c == 0xA0 || c == 0x1680 ||
               (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000 ||
               c == 0xFEFF;
    };
    size_t a = 0, b = s.size();
    while (a < b && sp(s[a])) ++a;
    while (b > a && sp(s[b - 1])) --b;
    return s.substr(a, b - a);
}
u32 trim_start(const u32& s) {
    const u32 t = trim(s + U"x");
    return t.substr(0, t.size() - 1);
}

u32 cardinal(long long number);

u32 digits(const u32& text, const char* zero = "zero") {
    std::vector<u32> out;
    for (char32_t c : text) {
        if (!is_digit(c)) continue;
        out.push_back(c == U'0' ? U(zero) : U(kOnes[c - U'0']));
    }
    return join(out, U" ");
}

u32 cardinal(long long number) {
    if (number < 0) return U"minus " + cardinal(-number);
    if (number < 20) return U(kOnes[number]);
    if (number < 100) {
        const long long tens = number / 10, ones = number % 10;
        return U(kTens[tens - 2]) + (ones ? U" " + U(kOnes[ones]) : U"");
    }
    if (number < 1000) {
        const long long rest = number % 100;
        return U(kOnes[number / 100]) + U" hundred" + (rest ? U" " + cardinal(rest) : U"");
    }
    if (number >= 1000000000000000000LL) return digits(num(number));
    std::vector<u32> parts;
    long long unit = 1000000000000000LL;   // 1000 ** 5
    for (int power = 5; power > 0; --power) {
        const long long count = number / unit;
        number %= unit;
        if (count) parts.push_back(cardinal(count) + U" " + U(kScales[power - 1]));
        unit /= 1000;
    }
    if (number) parts.push_back(cardinal(number));
    return join(parts, U" ");
}

u32 ordinal(long long number) {
    std::vector<u32> list = split_on(cardinal(number), U' ');
    u32& last = list.back();
    static const std::map<u32, u32> kOrd = {{U"one", U"first"}, {U"two", U"second"}, {U"three", U"third"},
                                            {U"five", U"fifth"}, {U"eight", U"eighth"}, {U"nine", U"ninth"},
                                            {U"twelve", U"twelfth"}};
    const auto it = kOrd.find(last);
    if (it != kOrd.end()) last = it->second;
    else if (ends_with(last, U"y")) last = last.substr(0, last.size() - 1) + U"ieth";
    else last += U"th";
    return join(list, U" ");
}

u32 year(long long number) {
    if (!(number >= 1000 && number <= 2999)) return cardinal(number);
    const long long high = number / 100, low = number % 100;
    if (low == 0) return high % 10 == 0 ? cardinal(number) : cardinal(high) + U" hundred";
    if (high == 20 && low < 10) return U"two thousand " + U(kOnes[low]);
    return cardinal(high) + (low < 10 ? U" oh " + U(kOnes[low]) : U" " + cardinal(low));
}

u32 pairs(long long number) {
    const long long high = number / 100, low = number % 100;
    if (!(number >= 100 && number <= 9999) || (low == 0 && (high < 10 || high % 10 == 0))) return cardinal(number);
    if (low == 0) return cardinal(high) + U" hundred";
    return cardinal(high) + (low < 10 ? U" oh " + U(kOnes[low]) : U" " + cardinal(low));
}

u32 number_words(const u32& text) {
    const u32 plain = replace_all(text, U",", U"");
    const size_t dot = plain.find(U'.');
    const u32 whole = dot == u32::npos ? plain : plain.substr(0, dot);
    const u32 fraction = dot == u32::npos ? U"" : plain.substr(dot + 1);
    if (!fraction.empty()) return (whole.empty() ? U"" : cardinal(to_int(whole))) + U" point " + digits(fraction);
    if ((whole.size() > 1 && whole[0] == U'0') || whole.size() > 15) return digits(whole);
    return cardinal(to_int(whole));
}

u32 duration(long long hours, long long minutes, long long seconds) {
    std::vector<u32> parts;
    const std::pair<long long, const char*> list[] = {{hours, "hour"}, {minutes, "minute"}, {seconds, "second"}};
    for (const auto& p : list) {
        if (!p.first) continue;
        parts.push_back(cardinal(p.first) + U" " + (p.first == 1 ? U(p.second) : U(p.second) + U"s"));
    }
    if (parts.size() < 2) return parts.empty() ? U"zero seconds" : parts[0];
    std::vector<u32> head(parts.begin(), parts.end() - 1);
    return join(head, U", ") + U" and " + parts.back();
}

u32 plural_of(const u32& spoken) {
    const size_t at = spoken.rfind(U' ');
    const u32 head = at == u32::npos ? U"" : spoken.substr(0, at);
    const u32 last = at == u32::npos ? spoken : spoken.substr(at + 1);
    return (head.empty() ? U"" : head + U" ") + (ends_with(last, U"y") ? last.substr(0, last.size() - 1) + U"ies" : last + U"s");
}

u32 fraction_words(long long top, long long bottom, bool article = false) {
    u32 one, many;
    if (bottom == 2) { one = U"half"; many = U"halves"; }
    else if (bottom == 4) { one = U"quarter"; many = U"quarters"; }
    else { one = ordinal(bottom); many = ordinal(bottom) + U"s"; }
    if (top != 1) return cardinal(top) + U" " + many;
    if (!article) return U"one " + one;
    return (U"aeiou"[0] && u32(U"aeiou").find(one[0]) != u32::npos ? U"an " : U"a ") + one;
}

u32 version_part(const u32& part) { return part.size() > 2 ? digits(part) : number_words(part); }
u32 minor(const u32& part) { return part[0] == U'0' ? digits(part, "oh") : cardinal(to_int(part)); }
bool short_number(const u32& text) {
    if (text.empty() || text[0] == U'0') return false;
    if (text.size() <= 2) return true;
    for (size_t i = 1; i < text.size(); ++i) if (text[i] != U'0') return false;
    return true;
}
u32 spelled(const u32& letters) { return u32(1, kSpell) + letters + u32(1, kEnd); }

u32 address_digits(const u32& text) {
    if (text.size() <= 2 && !(text.size() == 2 && text[0] == U'0')) return cardinal(to_int(text));
    if (text.size() == 4 && to_int(text) >= 1900 && to_int(text) <= 2099) return year(to_int(text));
    return digits(text);
}

std::set<u32> words(const char* list) {
    std::set<u32> out;
    for (const u32& w : split_on(U(list), U' ')) out.insert(w);
    return out;
}

const std::set<u32>& kSentenceStarters() {
    static const auto s = words("the he she it i we they this that there but and in on at a an if when what why how who my our "
                                "you then so now after before however");
    return s;
}
const std::set<u32>& kYearContext() {
    static const auto s = words("in since by from until till before after during around circa of year early late mid summer "
                                "winter spring fall autumn class");
    return s;
}
const std::set<u32>& kPhoneWords() { static const auto s = words("call dial ring phone telephone tel mobile cell fax hotline text sms whatsapp"); return s; }
const std::set<u32>& kPhoneLinks() { static const auto s = words("to on at or us me him her them back"); return s; }
const std::set<u32>& kPinWords() { static const auto s = words("code pin extension ext passcode password zip postcode cvv cvc otp"); return s; }
const std::set<u32>& kStrongWords() { static const auto s = words("number id reference ref serial account tracking booking confirmation"); return s; }
const std::set<u32>& kWeakWords() {
    static const auto s = words("order ticket invoice case claim policy receipt transaction parcel shipment container payment "
                                "reservation membership customer batch lot item job issue bug request");
    return s;
}
const std::set<u32>& kLinkWords() { static const auto s = words("is was are were ends ending in with starts starting begins beginning"); return s; }
const std::set<u32>& kNotCounted() {
    static const auto s = words("is was has does goes ships arrives departs leaves shows needs blocks remains includes contains "
                                "says reads starts ends closes opens runs takes comes this his its us as yes plus minus less");
    return s;
}
const std::set<u32>& kNumbered() {
    static const auto s = words("ref reference serial account order invoice tracking part model phone tel fax id room flight case "
                                "ticket policy licence license booking customer vat tax reg registration");
    return s;
}
const std::set<u32>& kDateBefore() {
    static const auto s = words("on dated due until till by since from before after for shipped delivered arriving arrives arrive "
                                "arrived expires expired ends ending starts starting begins beginning monday tuesday wednesday "
                                "thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun");
    return s;
}
const std::set<long long> kClassicFractions = {2, 3, 4, 5, 6, 8, 10, 16, 32, 64, 100};
const std::set<u32>& kRatioWords() { static const auto s = words("ratio ratios scale odds aspect mix split proportion"); return s; }
const std::set<u32>& kDurationWords() { static const auto s = words("lasted took duration runtime recording video clip track lap pace length long"); return s; }
const std::set<u32>& kClockWords() {
    static const auto s = words("window slot open opens opening hours shift session served service available booked appointment "
                                "delivery arrive arrives arriving");
    return s;
}
const std::set<u32>& kCapitalWords() {
    static const auto s = words("IN ON AT TO OF BY OR IS AS IF NO SO WE HE MY ME DO GO UP AN AM PM BE IT US THE AND FOR BUT NOT ALL");
    return s;
}
const std::set<u32>& kShortWords() { static const auto s = words("my to in on of go up no do me we us it is at by be so or an as if he hi ok"); return s; }

const std::map<u32, std::vector<u32>>& kCurrency() {
    static const std::map<u32, std::vector<u32>> m = [] {
        std::map<u32, std::vector<u32>> c;
        c[U"$"] = {U"dollar", U"dollars", U"cent", U"cents"};
        c[U"£"] = {U"pound", U"pounds", U"penny", U"pence"};
        c[U"€"] = {U"euro", U"euros", U"cent", U"cents"};
        c[U"¥"] = {U"yen", U"yen", U"", U""};
        c[U"₹"] = {U"rupee", U"rupees", U"paisa", U"paise"};
        c[U"USD"] = c[U"$"]; c[U"EUR"] = c[U"€"]; c[U"GBP"] = c[U"£"]; c[U"JPY"] = c[U"¥"]; c[U"INR"] = c[U"₹"];
        c[U"CHF"] = {U"Swiss franc", U"Swiss francs", U"centime", U"centimes"};
        c[U"CAD"] = {U"Canadian dollar", U"Canadian dollars", U"cent", U"cents"};
        c[U"AUD"] = {U"Australian dollar", U"Australian dollars", U"cent", U"cents"};
        c[U"CNY"] = {U"yuan", U"yuan", U"", U""};
        return c;
    }();
    return m;
}

int month_number(const u32& name, const Shared& sh) {   // MONTH_NUMBER[name.toLowerCase().slice(0, 3)]
    const u32 key = sh.to_lower(name).substr(0, 3);
    for (int i = 0; i < 12; ++i) {
        if (sh.to_lower(U(kMonths[i])).substr(0, 3) == key) return i + 1;
    }
    return 0;
}

} // namespace

// A match with its text, as dizione's `sub` hands one to a method.
struct Normalizer::M {
    const u32& string;
    const JsRegex::Match& m;
    const JsRegex& re;
    size_t start() const { return m.begin; }
    size_t end() const { return m.end; }
    u32 all() const { return string.substr(m.begin, m.end - m.begin); }
    std::optional<u32> group(size_t i) const {
        if (!m.has(i)) return std::nullopt;
        return m.group(string, i);
    }
    std::optional<u32> named(const char* n) const { return group(re.group_index(n)); }
    u32 g(size_t i) const { return m.group(string, i); }   // undefined as ""
};

Normalizer::Normalizer(const Shared& sh, const Tables& t, std::function<bool(const u32&)> is_word, bool british)
    : sh_(sh), t_(t), is_word_(std::move(is_word)), british_(british) {}

static bool truthy(const std::optional<u32>& v) { return v.has_value() && !v->empty(); }

std::vector<u32> Normalizer::before(const u32& text, size_t position, size_t count) const {
    const size_t from = position > 80 ? position - 80 : 0;
    const std::vector<u32> parts = sh_.rule("before_split").split(text.substr(from, position - from));
    std::vector<u32> list;
    for (const u32& w : sh_.rule("letters").match_all(parts.back())) list.push_back(sh_.to_lower(w));
    if (list.size() > count) list.erase(list.begin(), list.end() - static_cast<long>(count));
    return list;
}

std::optional<std::string> Normalizer::identifier(const u32& text, size_t position) const {
    const auto list = before(text, position);
    if (list.empty()) return std::nullopt;
    const u32& last = list.back();
    if (kPhoneWords().count(last)) return std::string("phone");
    if (kPhoneLinks().count(last)) {
        const size_t lo = list.size() >= 3 ? list.size() - 3 : 0;
        for (size_t i = lo; i + 1 < list.size(); ++i) if (kPhoneWords().count(list[i])) return std::string("phone");
    }
    size_t index = list.size() - 1;
    while (index > 0 && kLinkWords().count(list[index])) --index;
    if (kPhoneWords().count(list[index])) return std::string("phone");
    if (kPinWords().count(list[index])) return std::string("pin");
    if (kStrongWords().count(list[index])) return std::string("strong");
    if (kWeakWords().count(list[index])) return std::string("weak");
    if (last == U"enter") return std::string("enter");
    return std::nullopt;
}

u32 Normalizer::amount(const u32& symbol, const u32& whole, const std::optional<u32>& fraction,
                       const std::optional<u32>& scale) const {
    const auto& c = kCurrency().at(symbol);
    const u32 &one = c[0], &many = c[1], &cent = c[2], &cents = c[3];
    if (truthy(scale)) {
        static const std::map<u32, u32> kScale = {{U"k", U"thousand"}, {U"m", U"million"}, {U"b", U"billion"},
                                                  {U"bn", U"billion"}, {U"t", U"trillion"}};
        const u32 low = sh_.to_lower(*scale);
        const auto it = kScale.find(low);
        const u32 said = it != kScale.end() ? it->second : low;
        return number_words(whole + (truthy(fraction) ? U"." + *fraction : U"")) + U" " + said + U" " + many;
    }
    if (truthy(fraction) && (fraction->size() > 2 || cent.empty())) return number_words(whole + U"." + *fraction) + U" " + many;
    const long long amt = to_int(whole);
    u32 f = truthy(fraction) ? *fraction : U"0";
    while (f.size() < 2) f += U'0';
    const long long change = to_int(f);
    std::vector<u32> parts;
    if (amt || !change) parts.push_back(cardinal(amt) + U" " + (amt == 1 ? one : many));
    if (change && !cent.empty()) parts.push_back(cardinal(change) + U" " + (change == 1 ? cent : cents));
    return join(parts, U" and ");
}

u32 Normalizer::money(const M& m) const {
    const u32 said = amount(*m.named("symbol"), replace_all(*m.named("whole"), U",", U""), m.named("fraction"), m.named("scale"));
    return (truthy(m.named("minus")) || truthy(m.named("inner")) ? U" minus " : U" ") + said + U" ";
}

u32 Normalizer::half_day(const M& m, const std::optional<u32>& dotted, const std::optional<u32>& plain) const {
    if (!(truthy(dotted) || truthy(plain))) return U"";
    const u32 following = trim_start(m.string.substr(m.end()));
    JsRegex::Match w;
    const bool first = sh_.rule("word_start").match_at(following, 0, w);
    const bool closes = truthy(dotted) && ends_with(m.all(), U".") &&
                        (following.empty() || (sh_.is_upper(following.substr(0, 1)) && first &&
                                               kSentenceStarters().count(sh_.to_lower(w.group(following, 0)))));
    const u32 letter = truthy(dotted) ? *dotted : *plain;
    return spelled(sh_.to_upper(letter) + U"M") + (closes ? U" ." : U"") + U" ";
}

u32 Normalizer::time(const M& m) const {
    const long long hour = to_int(m.g(1)), minute = to_int(m.g(2));
    const std::optional<u32> second = m.group(3);
    const bool half = truthy(m.group(4)) || truthy(m.group(5));
    const auto list = before(m.string, m.start(), 3);
    if (second && !half && !(!list.empty() && list.back() == U"at")) {
        return U" " + duration(hour, minute, to_int(*second)) + U" ";
    }
    if (hour > 24 && !second && !half) {
        JsRegex::Match a;
        const u32 rest = m.string.substr(m.end());
        const bool after = sh_.rule("next_word").match_at(rest, 0, a);
        bool dur = false;
        for (const u32& w : list) if (kDurationWords().count(w)) dur = true;
        if (dur || (after && kDurationWords().count(sh_.to_lower(a.group(rest, 1))))) return U" " + duration(0, hour, minute) + U" ";
        return U" " + cardinal(hour) + U" " + (minute < 10 ? U"oh " : U"") + cardinal(minute) + U" ";
    }
    if (hour > 24) return m.all();
    u32 said = cardinal(hour);
    if (minute == 0) {
        if (half) {}
        else if (hour == 0) said = U"midnight";
        else if (hour > 12) said += U" hundred";
        else said += U" o'clock";
    } else {
        said += (minute < 10 ? U" oh " : U" ") + cardinal(minute);
    }
    if (truthy(second)) {
        const long long s = to_int(*second);
        said += U" and " + cardinal(s) + (s == 1 ? U" second" : U" seconds");
    }
    return U" " + said + U" " + half_day(m, m.group(4), m.group(5));
}

u32 Normalizer::dot_times(const M& m) const {
    const u32 following = m.string.substr(m.end());
    bool clock = false;
    for (const u32& w : before(m.string, m.start())) if (kClockWords().count(w)) clock = true;
    if (to_int(m.g(1)) > 23 || to_int(m.g(4)) > 23 || !clock || sh_.rule("dot_times_unit").test(following)) return m.all();
    return m.g(1) + U":" + m.g(2) + m.g(3) + m.g(4) + U":" + m.g(5);
}

u32 Normalizer::ratio(const M& m) const {
    const u32 first = m.g(1), second = m.g(2);
    if (first.size() <= 2 && second.size() == 2 && u32(U"012345").find(second[0]) != u32::npos) {
        JsRegex::Match a;
        const u32 rest = m.string.substr(m.end());
        const bool after = sh_.rule("next_word").match_at(rest, 0, a);
        bool hit = false;
        for (const u32& w : before(m.string, m.start(), 3)) if (kRatioWords().count(w)) hit = true;
        if (!(hit || (after && kRatioWords().count(sh_.to_lower(a.group(rest, 1)))))) return m.all();
    }
    return U" " + number_words(first) + U" to " + number_words(second) + U" ";
}

std::optional<u32> Normalizer::date(long long month, long long day, const std::optional<u32>& year_text, bool the) const {
    if (!(month >= 1 && month <= 12) || !(day >= 1 && day <= 31)) return std::nullopt;
    u32 said = the ? U"the " + ordinal(day) + U" of " + U(kMonths[month - 1]) : U(kMonths[month - 1]) + U" " + ordinal(day);
    if (truthy(year_text)) {
        said += U" " + (year_text->size() == 4 ? year(to_int(*year_text))
                                                : (to_int(*year_text) < 10 ? U"oh " : U"") + cardinal(to_int(*year_text)));
    }
    return U" " + said + U" ";
}

// The two-argument call sites: dizione's default `the = false`, which enGb.js's BritishNormalizer turns on.
std::optional<u32> Normalizer::date(long long month, long long day, const std::optional<u32>& year_text) const {
    return date(month, day, year_text, british_);
}

u32 Normalizer::slash_date(const M& m) const {
    if (british_) day_first_ = true;
    long long first = to_int(m.g(1)), second = to_int(m.g(2));
    if ((first > 12 && 12 >= second) || (day_first_ && second <= 12)) std::swap(first, second);
    const auto d = date(first, second, m.group(3));
    return d ? *d : m.all();
}

u32 Normalizer::dot_date(const M& m) const {
    long long day = to_int(m.g(1)), month = to_int(m.g(2));
    if (month > 12 && 12 >= day) std::swap(day, month);
    const auto d = date(month, day, m.group(3));
    return d ? *d : m.all();
}

u32 Normalizer::day_month(const M& m) const {
    const u32 month = m.g(2);
    const auto spoken = date(month_number(month, sh_), to_int(m.g(1)), m.group(4), true);
    if (!spoken) return m.all();
    bool full = false;
    for (const char* n : kMonths) if (month == U(n)) full = true;
    return *spoken + (truthy(m.group(3)) && full ? U". " : U"");
}

u32 Normalizer::days_of_month(const M& m) const {
    const bool month_first = is_letter(m.g(1)[0]);
    const u32 month = month_first ? m.g(1) : m.g(3);
    const u32 a = month_first ? m.g(2) : m.g(1);
    const u32 b = month_first ? m.g(3) : m.g(2);
    const long long first = to_int(a), second = to_int(b);
    if (!(first >= 1 && first <= 31 && second >= 1 && second <= 31)) return m.all();
    const u32 name = U(kMonths[month_number(month, sh_) - 1]);
    const u32 said = month_first ? name + U" " + ordinal(first) + U" to " + ordinal(second)
                                 : U"the " + ordinal(first) + U" to the " + ordinal(second) + U" of " + name;
    return U" " + said + (truthy(m.group(4)) ? U" " + year(to_int(*m.group(4))) : U"") + U" ";
}

u32 Normalizer::day_range(const M& m) const {
    static const std::map<u32, u32> kDays = {{U"mon", U"Monday"}, {U"tue", U"Tuesday"}, {U"wed", U"Wednesday"},
                                             {U"thu", U"Thursday"}, {U"fri", U"Friday"}, {U"sat", U"Saturday"},
                                             {U"sun", U"Sunday"}};
    u32 names[2];
    const u32 src[2] = {m.g(1), m.g(3)};
    for (int i = 0; i < 2; ++i) {
        const u32 key = sh_.to_lower(src[i]);
        const auto ab = t_.abbreviations.find(key);
        if (ab != t_.abbreviations.end()) names[i] = ab->second.second;
        else {
            const auto d = kDays.find(key);
            names[i] = d != kDays.end() ? d->second : src[i];
        }
    }
    const u32 following = trim_start(m.string.substr(m.end()));
    JsRegex::Match w;
    const bool first = sh_.rule("word_start").match_at(following, 0, w);
    const bool ends = truthy(m.group(4)) &&
                      (following.empty() || (sh_.is_upper(following.substr(0, 1)) && first &&
                                             kSentenceStarters().count(sh_.to_lower(w.group(following, 0)))));
    return names[0] + U" to " + names[1] + (ends ? U" ." : U"");
}

std::optional<u32> Normalizer::short_date(const M& m) const {
    if (british_) day_first_ = true;
    const u32 first_text = m.g(1), second_text = m.g(2);
    if (first_text.size() > 2 || second_text.size() > 2) return std::nullopt;
    long long first = to_int(first_text), second = to_int(second_text);
    const auto list = before(m.string, m.start());
    if (first >= 1 && first <= 12 && second >= 20 && second <= 49 && std::find(list.begin(), list.end(), U"card") != list.end()) {
        return U" " + U(kMonths[first - 1]) + U" " + year(2000 + second) + U" ";
    }
    const u32 last = list.empty() ? U"" : list.back();
    const bool classic = first < second && kClassicFractions.count(second);
    const bool zero = (first_text.size() == 2 && first_text[0] == U'0') || (second_text.size() == 2 && second_text[0] == U'0');
    if (!(zero || (first > 12 && first <= 31 && second <= 12 && !(first == 24 && second == 7)) ||
          (kDateBefore().count(last) && (!classic || last == U"on")) || (dated_ && !classic && first != second))) {
        return std::nullopt;
    }
    if (first > 12 || (day_first_ && second <= 12)) std::swap(first, second);
    return date(first, second, std::nullopt);
}

u32 Normalizer::fraction(const M& m) const {
    const auto spoken = short_date(m);
    if (spoken && !spoken->empty()) return *spoken;
    const long long top = to_int(m.g(1)), bottom = to_int(m.g(2));
    if ((top == 24 && bottom == 7) || top == bottom) return U" " + cardinal(top) + U" " + cardinal(bottom) + U" ";
    if (!(top > 0 && top < 100) || !(bottom > 1 && bottom <= 100)) return U" " + cardinal(top) + U" over " + cardinal(bottom) + U" ";
    return U" " + fraction_words(top, bottom) + U" ";
}

u32 Normalizer::mixed_fraction(const M& m) const {
    const long long top = to_int(m.g(2)), bottom = to_int(m.g(3));
    if (!(top > 0 && top < bottom)) return m.all();
    return U" " + number_words(m.g(1)) + U" and " + fraction_words(top, bottom, true) + U" ";
}

u32 Normalizer::vulgar(const M& m) const {
    const u32 f = sh_.nfkc_normalize(m.g(2));
    const auto parts = split_on(f, U'⁄');
    const long long top = to_int(parts[0]), bottom = parts.size() > 1 ? to_int(parts[1]) : -1;
    if (!m.group(1)) return U" " + num(top) + U"/" + num(bottom) + U" ";
    return U" " + cardinal(to_int(*m.group(1))) + U" and " + fraction_words(top, bottom, true) + U" ";
}

u32 Normalizer::phone(const M& m) const {
    std::vector<u32> out;
    for (size_t i = 1; i <= 4; ++i) {
        const auto g = m.group(i);
        if (!truthy(g)) continue;
        out.push_back(*g == U"800" ? U"eight hundred" : digits(*g, "oh"));
    }
    return U" " + join(out, U", ") + U" ";
}

u32 Normalizer::measure(const M& m, size_t* extra) const {
    const u32 unit = m.g(3);
    const std::pair<u32, u32>* names = nullptr;
    const auto e = t_.exact_units.find(unit);
    if (e != t_.exact_units.end()) names = &e->second;
    else {
        const auto u = t_.units.find(sh_.to_lower(unit));
        if (u != t_.units.end()) names = &u->second;
    }
    const u32 next = m.string.substr(m.end(), 1);
    if (names == nullptr || (unit.size() == 1 && (next == U"-" || next == U"'" || next == U"/"))) return m.all();
    const u32 value = m.g(1);
    const auto upper = m.group(2);
    *extra = sh_.rule("stop_small").test(m.string.substr(m.end())) ? 1 : 0;
    if (truthy(upper)) return U" " + number_words(value) + U" to " + number_words(*upper) + U" " + names->second + U" ";
    return U" " + number_words(value) + U" " + (value == U"1" || value == U"1.0" ? names->first : names->second) + U" ";
}

u32 Normalizer::square(const M& m) const {
    const u32 unit = m.g(2);
    std::pair<u32, u32> names;
    if (unit == U"in") names = {U"inch", U"inches"};
    else {
        const auto e = t_.exact_units.find(unit);
        names = e != t_.exact_units.end() ? e->second : t_.units.at(unit);
    }
    const u32 mark = m.g(3);
    const u32 kind = (mark == U"²" || mark == U"2") ? U"square" : U"cubic";
    const u32 v = m.g(1);
    return U" " + number_words(v) + U" " + kind + U" " + (v == U"1" || v == U"1.0" ? names.first : names.second) + U" ";
}

bool Normalizer::counted(const u32& text, size_t end) const {
    JsRegex::Match a;
    const u32 rest = text.substr(end);
    if (!sh_.rule("next_word").match_at(rest, 0, a)) return false;
    const u32 word = sh_.to_lower(a.group(rest, 1));
    for (const char* w : {"people", "men", "women", "children", "feet", "miles"}) if (word == U(w)) return true;
    return word.size() > 3 && ends_with(word, U"s") && !ends_with(word, U"ss") && !kNotCounted().count(word) &&
           is_word_(word.substr(0, word.size() - 1));
}

u32 Normalizer::number(const M& m) const {
    const u32& text = m.string;
    const u32 value = m.all();
    const u32 plain = replace_all(value, U",", U"");
    const bool has_dot = plain.find(U'.') != u32::npos, has_comma = value.find(U',') != u32::npos;
    if (!has_dot && !has_comma && plain[0] != U'0' && plain.size() > 2) {
        const auto kind = identifier(text, m.start());
        size_t limit = 7;
        if (kind == std::string("phone") || kind == std::string("pin")) limit = 3;
        else if (kind == std::string("strong") || kind == std::string("enter")) limit = 4;
        else if (kind == std::string("weak")) limit = 5;
        const bool null_phone_weak = !kind || *kind == "phone" || *kind == "weak";
        if (plain.size() >= limit && !(!kind && ends_with(plain, U"000")) &&
            !(kind && *kind == "enter" && plain.size() == 4 && to_int(plain) >= 1900 && to_int(plain) <= 2099) &&
            !(null_phone_weak && counted(text, m.end()))) {
            return U" " + digits(plain, kind && *kind == "phone" ? "oh" : "zero") + U" ";
        }
    }
    if (!has_dot && !has_comma && plain.size() == 4 && plain[0] != U'0') {
        const long long n = to_int(plain);
        const u32 head = text.substr(0, m.start());
        const u32 tail24 = head.size() > 24 ? head.substr(head.size() - 24) : head;
        const auto before_words = sh_.rule("letters").match_all(tail24);
        JsRegex::Match a;
        const u32 rest = text.substr(m.end());
        const bool after = sh_.rule("next_word").match_at(rest, 0, a);
        const bool cnt = after && sh_.rule("counted_word").test(sh_.to_lower(a.group(rest, 1)));
        const u32 last_before = before_words.empty() ? U"" : sh_.to_lower(before_words.back());
        bool month_prefix = false;
        if (!last_before.empty()) {
            const u32 three = last_before.substr(0, 3);
            for (const char* mo : kMonths) if (sh_.to_lower(U(mo)).substr(0, 3) == three) month_prefix = true;
        }
        const bool dated = !last_before.empty() && (kYearContext().count(last_before) || month_prefix);
        JsRegex::Match md;
        const bool model = sh_.rule("model_before").search(head, 0, md);
        if (model && !kCapitalWords().count(md.group(head, 1)) && !cnt) return U" " + pairs(n) + U" ";
        if ((n >= 1100 && n <= 1999) || (n >= 2000 && n <= 2099 && (dated || !cnt))) return U" " + year(n) + U" ";
    }
    return U" " + number_words(value) + U" ";
}

bool Normalizer::reads_as_word(const u32& letters) const {
    const u32 lower = sh_.to_lower(letters);
    if (lower.size() < 3 || !is_word_(lower)) return false;
    if (lower.size() > 3) return true;
    for (const char* e : {"s", "es", "ed", "ing"}) if (is_word_(lower + U(e))) return true;
    return false;
}

u32 Normalizer::alnum(const u32& group, const char* zero) const {
    std::vector<u32> out;
    for (const u32& run : sh_.rule("alnum_runs").match_all(group)) out.push_back(all_digits(run) ? digits(run, zero) : spelled(run));
    return join(out, U" ");
}

u32 Normalizer::digit_groups(const std::vector<u32>& groups, const char* zero) const {
    bool small = true;
    for (const u32& g : groups) if (!(short_number(g) && g.size() <= 2)) small = false;
    std::vector<u32> out;
    for (const u32& g : groups) out.push_back(small ? cardinal(to_int(g)) : digits(g, zero));
    return join(out, U" , ");
}

std::optional<u32> Normalizer::code_words(const std::vector<u32>& groups, const M& m) const {
    const u32 token = join(groups, U"-");
    if (!has_digit(token)) return std::nullopt;
    std::vector<char> kinds;
    for (const u32& g : groups) kinds.push_back(all_digits(g) ? 'd' : has_digit(g) ? 'm' : sh_.is_upper(g) ? 'u' : 'l');
    const bool all_d = std::all_of(kinds.begin(), kinds.end(), [](char k) { return k == 'd'; });
    if (all_d) {
        if (groups.size() == 2) {
            bool years = true;
            for (const u32& g : groups) if (!(g.size() == 4 && to_int(g) >= 1000 && to_int(g) <= 2100)) years = false;
            bool lead_zero = false;
            size_t shortest = groups[0].size();
            for (const u32& g : groups) { if (g.size() > 1 && g[0] == U'0') lead_zero = true; shortest = std::min(shortest, g.size()); }
            const auto kind = identifier(m.string, m.start());
            if (!(truthy(m.group(1)) || lead_zero || (shortest >= 4 && !years) ||
                  (kind && (*kind == "strong" || *kind == "pin")))) {
                return std::nullopt;
            }
        }
        return digit_groups(groups);
    }
    if (groups.size() == 2 && kinds[1] == 'd' && (kinds[0] == 'u' || kinds[0] == 'l')) {
        const u32 &letters = groups[0], &number = groups[1];
        if ((kinds[0] == 'l' && letters.size() > 1) || reads_as_word(letters)) return std::nullopt;
        if (short_number(number)) return spelled(letters) + U" " + cardinal(to_int(number));
        return spelled(letters) + U" , " + digits(number);
    }
    if (groups.size() == 2 && kinds[0] == 'd' &&
        (kinds[1] == 'l' || groups[1].size() == 1 || sh_.rule("digits_letters").test(groups[1]))) {
        return std::nullopt;
    }
    for (size_t i = 0; i < groups.size(); ++i) {
        if ((kinds[i] == 'l' && groups[i].size() > 2) || (kinds[i] == 'm' && sh_.rule("digits_suffix").test(groups[i]))) {
            return std::nullopt;
        }
    }
    bool upper_letter = false;
    for (char32_t c : token) if (c >= U'A' && c <= U'Z') upper_letter = true;
    if (!upper_letter && (std::count(kinds.begin(), kinds.end(), 'l') || std::count(kinds.begin(), kinds.end(), 'm'))) {
        return std::nullopt;
    }
    std::vector<u32> out;
    for (size_t i = 0; i < groups.size(); ++i) {
        const u32& g = groups[i];
        if (kinds[i] == 'd') out.push_back(digits(g));
        else if (kinds[i] == 'm') out.push_back(alnum(g));
        else if (g.size() > 3 && is_word_(sh_.to_lower(g))) out.push_back(g);
        else out.push_back(g.size() <= 5 ? spelled(g) : g);
    }
    return join(out, U" , ");
}

u32 Normalizer::code(const M& m) const {
    const auto said = code_words(split_on(m.g(2), U'-'), m);
    if (!said) return m.all();
    return (truthy(m.group(1)) ? U" number " : U" ") + *said + U" ";
}

u32 Normalizer::grouped(const M& m) const {
    const u32& text = m.string;
    const auto plus = m.group(1);
    std::optional<u32> lead = m.group(2), trail = m.group(4);
    const std::vector<u32> core = split_on(m.group(3) ? *m.group(3) : m.g(5), U' ');
    u32 head, tail;
    if (truthy(lead) && kCapitalWords().count(*lead)) {
        head = *lead + U" ";
        tail = truthy(trail) ? U" " + *trail : U"";
        lead.reset();
        trail.reset();
    }
    const auto kind = identifier(text, m.start());
    const bool pure = std::all_of(core.begin(), core.end(), all_digits);
    if (pure && !truthy(plus) && !truthy(lead)) {
        const u32 following = text.substr(m.end());
        const bool amt = counted(text, m.end()) || sh_.rule("grouped_unit").test(following);
        bool rest3 = true;
        for (size_t i = 1; i < core.size(); ++i) if (core[i].size() != 3) rest3 = false;
        if (!kind && core[0][0] != U'0' && core[0].size() <= 3 && rest3 && (core.back() == U"000" || (amt && core.size() > 2))) {
            return head + join(core, U",") + tail;
        }
        bool long_part = false;
        for (const u32& p : core) if (p.size() > 2) long_part = true;
        if (!(kind || (core[0].size() > 1 && core[0][0] == U'0') || (core.size() > 2 && long_part))) return m.all();
    } else if (!pure && !truthy(lead) && !truthy(plus) && !kind) {
        return m.all();
    }
    size_t total = 0;
    for (const u32& p : core) total += p.size();
    const bool oh = truthy(plus) || (kind && *kind == "phone") || (pure && core[0][0] == U'0' && total >= 10);
    const char* zero = oh ? "oh" : "zero";
    u32 spoken;
    if (pure) spoken = digit_groups(core, zero);
    else {
        std::vector<u32> out;
        for (const u32& p : core) out.push_back(all_digits(p) ? digits(p, zero) : alnum(p, zero));
        spoken = join(out, U" , ");
    }
    if (truthy(lead)) spoken = spelled(*lead) + U" , " + spoken;
    if (truthy(trail)) spoken = spoken + U" , " + spelled(*trail);
    return head + (truthy(plus) ? U" plus " : U" ") + spoken + U" " + tail;
}

u32 Normalizer::glued(const M& m) const {
    std::vector<u32> out;
    for (const u32& run : sh_.rule("alnum_runs").match_all(m.all())) {
        if (all_digits(run)) out.push_back(short_number(run) ? cardinal(to_int(run)) : digits(run));
        else if (run.size() == 1 || ((sh_.is_upper(run) || run.size() <= 3) && !reads_as_word(run))) out.push_back(spelled(run));
        else out.push_back(run);
    }
    return U" " + join(out, U" ") + U" ";
}

u32 Normalizer::address_part(const u32& part) const {
    const u32 lower = sh_.to_lower(part);
    if (part.size() > 1 && is_word_(lower) && (part.size() > 2 || kShortWords().count(lower))) return part;
    return part.size() <= 3 ? spelled(part) : part;
}

u32 Normalizer::spoken_address(const u32& core) const {
    static const std::map<char32_t, u32> kSep = {{U'.', U"dot"}, {U'/', U"slash"}, {U'@', U"at"}, {U'-', U"dash"},
                                                 {U'_', U"underscore"}, {U':', U"colon"}, {U'=', U"equals"},
                                                 {U'&', U"and"}, {U'+', U"plus"}};
    const auto pieces = sh_.rule("address_pieces").match_all(core);
    const bool mail = core.find(U'@') != u32::npos;
    bool host = false;
    std::vector<u32> out;
    for (size_t index = 0; index < pieces.size(); ++index) {
        const u32& piece = pieces[index];
        const u32 following = index + 1 < pieces.size() ? pieces[index + 1] : U"";
        if (piece == U"@") host = true;
        if (all_digits(piece)) { out.push_back(address_digits(piece)); continue; }
        if (!all_letters(piece)) {
            const auto it = piece.size() == 1 ? kSep.find(piece[0]) : kSep.end();
            out.push_back(it != kSep.end() ? it->second : U"");
            continue;
        }
        const u32 low = sh_.to_lower(piece);
        const bool ending_known = t_.endings.count(low) != 0;
        if (index && pieces[index - 1] == U"." && (host || !mail) &&
            (following.empty() || following == U"/" || following == U":" || following == U"?" || following == U"#" ||
             (piece.size() <= 3 && (ending_known || !is_word_(low))))) {
            const auto e = t_.endings.find(low);
            out.push_back(e != t_.endings.end() && e->second == U"word" ? low : spelled(piece));
            continue;
        }
        u32 letters;
        for (const u32& part : sh_.rule("address_parts").match_all(piece)) {
            const u32 spoken = address_part(part);
            if (!spoken.empty() && spoken[0] == kSpell) { letters += part; continue; }
            if (!letters.empty()) { out.push_back(spelled(letters)); letters.clear(); }
            out.push_back(spoken);
        }
        if (!letters.empty()) out.push_back(spelled(letters));
    }
    std::vector<u32> kept;
    for (const u32& o : out) if (!o.empty()) kept.push_back(o);
    return join(kept, U" ");
}

u32 Normalizer::address(const M& m) const {
    const u32 whole = m.all();
    const u32 text = sh_.rule("address_trail").replace(whole, U"");
    const u32 tail = whole.substr(text.size());
    if (m.named("file")) {
        const u32 ending = *m.named("ending");
        const bool known = t_.endings.count(sh_.to_lower(ending)) != 0;
        const bool marked = truthy(m.named("folder")) || truthy(m.named("path")) || m.named("stem")->find(U'_') != u32::npos;
        if (!(known || (marked && ending.size() <= 5 && (ending == sh_.to_lower(ending) || sh_.is_upper(ending)) &&
                        !is_word_(sh_.to_lower(ending))))) {
            return whole;
        }
    }
    const u32 core = sh_.rule("address_slashes").replace(sh_.rule("address_scheme").replace(text, U""), U"");
    return U" " + spoken_address(core) + U" " + tail;
}

u32 Normalizer::chat_form(const M& m) const {
    const auto it = t_.chat.find(sh_.to_lower(m.all()));
    if (it == t_.chat.end()) return m.all();
    if (it->second == U"-") return U" ";
    std::vector<u32> out;
    for (const u32& part : split_on(it->second, U' ')) out.push_back(!part.empty() && part[0] == U'*' ? spelled(part.substr(1)) : part);
    return U" " + join(out, U" ") + U" ";
}

u32 Normalizer::dotted(const M& m) const {
    const u32 letters = replace_all(m.all(), U".", U"");
    const u32 key = sh_.to_lower(letters);
    const u32 following = trim_start(m.string.substr(m.end()));
    const u32 said = key == U"eg" ? U"for example" : key == U"ie" ? U"that is" : spelled(sh_.to_upper(letters));
    JsRegex::Match w;
    const bool first = sh_.rule("word_start").match_at(following, 0, w);
    const bool ends = following.empty() || (sh_.is_upper(following.substr(0, 1)) && first &&
                                            kSentenceStarters().count(sh_.to_lower(w.group(following, 0))));
    return U" " + said + (ends && ends_with(m.all(), U".") ? U" ." : U"") + U" ";
}

u32 Normalizer::abbreviation(const M& m) const {
    const u32& text = m.string;
    const u32 token = m.g(1), stop = m.g(2);
    const u32 key = sh_.to_lower(token);
    const u32 following = text.substr(m.end());
    JsRegex::Match nw;
    const bool next = sh_.rule("next_word_or_digit").match_at(following, 0, nw);
    const u32 next_word = next ? nw.group(following, 1) : U"";
    const bool capital_next = next && sh_.is_upper(next_word.substr(0, 1));
    const bool capital_before = sh_.rule("capital_before").test(text.substr(0, m.start()));
    const bool upper0 = sh_.is_upper(token.substr(0, 1));
    if ((key == U"st" || key == U"dr") && upper0) {
        if (capital_next && (!stop.empty() || key == U"dr")) return key == U"st" ? U" saint " : U" doctor ";
        if (capital_before && !stop.empty()) return u32(key == U"st" ? U" street" : U" drive") + (trim(following).empty() ? U" ." : U" ");
        return m.all();
    }
    const auto ab = t_.abbreviations.find(key);
    if (ab == t_.abbreviations.end()) return m.all();
    const u32 when = ab->second.first;
    u32 said = ab->second.second;
    const bool number_next = next && all_digits(next_word);
    if (when == U"title") {
        if (!(upper0 && (capital_next || !stop.empty()))) return m.all();
    } else if (when == U"number") {
        JsRegex::Match pm;
        const u32 head = text.substr(0, m.start());
        const bool prev = sh_.rule("previous_word").search(head, 0, pm);
        if (!(!stop.empty() && (number_next || (key == U"no" && prev && kNumbered().count(sh_.to_lower(pm.group(head, 1))))))) {
            return m.all();
        }
    } else if (stop.empty() || (when == U"small" && sh_.is_upper(token))) {
        return m.all();
    } else {
        static const std::set<u32> kMarked = {U"mar", U"sun", U"sat", U"jan", U"dec", U"max", U"min", U"est", U"no"};
        if (kMarked.count(key) && !upper0 && !number_next) {
            const bool mme = key == U"max" || key == U"min" || key == U"est";
            const bool lower_next = next && sh_.to_lower(next_word.substr(0, 1)) == next_word.substr(0, 1) &&
                                    sh_.to_upper(next_word.substr(0, 1)) != next_word.substr(0, 1);
            if (!(mme && lower_next)) return m.all();
        }
    }
    if (key == U"est" && !number_next) said = U"estimated";
    const bool ends = when != U"title" && (trim(following).empty() ||
                                          (capital_next && kSentenceStarters().count(sh_.to_lower(next_word))));
    return U" " + said + (ends ? U" ." : U"") + U" ";
}

// ------------------------------------------------------------------------------------------------ rewriting
u32 Normalizer::rewrite(u32 text) const {
    using R = JsRegex;
    auto with = [this](const char* name, const u32& t, std::function<u32(const M&, size_t*)> fn) {
        const R& re = sh_.rule(name);
        return re.replace(t, [&](const u32& s, const R::Match& mm, size_t* extra) { return fn(M{s, mm, re}, extra); });
    };
    text = with("vulgar", text, [this](const M& m, size_t*) { return vulgar(m); });
    text = with("area", text, [this](const M& m, size_t*) { return square(m); });
    text = replace_all(text, U"²", U" squared ");
    text = replace_all(text, U"³", U" cubed ");
    text = sh_.nfkc_normalize(text);
    text = replace_all(text, U"⁄", U"/");
    text = replace_all(text, U"−", U" minus ");
    text = sh_.rule("en_dash_digits").replace(text, U"-");
    text = with("day_range", text, [this](const M& m, size_t*) { return day_range(m); });
    const std::pair<const char32_t*, const char32_t*> quotes[] = {
        {U"‘", U"'"}, {U"’", U"'"}, {U"‚", U"'"}, {U"‛", U"'"}, {U"`", U"'"}, {U"´", U"'"}, {U"“", U"\""}, {U"”", U"\""},
        {U"„", U"\""}, {U"‟", U"\""}, {U"«", U"\""}, {U"»", U"\""}, {U"–", U"—"}, {U"―", U"—"}, {U"[", U"("}, {U"]", U")"},
        {U"{", U"("}, {U"}", U")"}};
    for (const auto& q : quotes) text = replace_all(text, q.first, q.second);
    text = with("address", text, [this](const M& m, size_t*) { return address(m); });
    text = replace_all(text, U"→", U" to ");
    text = sh_.rule("emoji").replace(text, U" ");
    text = sh_.rule("emoticon").replace(text, U" ");
    text = sh_.rule("dots3").replace(text, U"…");
    text = sh_.rule("minus_eq").replace(text, U" minus ");
    text = sh_.rule("dashes").replace(text, U"—");
    text = sh_.rule("without").replace(text, U" without ");
    text = sh_.rule("with").replace(text, U" with ");
    text = sh_.rule("not_applicable").replace(text, U" not applicable ");
    text = replace_all(text, U"'n'", U" 'n ");
    if (has_digit(text)) text = figures(text);
    return signs(text);
}

u32 Normalizer::figures(u32 text) const {
    using R = JsRegex;
    auto with = [this](const char* name, const u32& t, std::function<u32(const M&, size_t*)> fn) {
        const R& re = sh_.rule(name);
        return re.replace(t, [&](const u32& s, const R::Match& mm, size_t* extra) { return fn(M{s, mm, re}, extra); });
    };
    text = with("money", text, [this](const M& m, size_t*) { return money(m); });
    text = with("money_after", text, [this](const M& m, size_t*) { return money(m); });
    text = with("cents", text, [](const M& m, size_t*) {
        const u32 n = m.g(1);
        return U" " + number_words(n) + U" " + (n == U"1" ? U"cent" : U"cents") + U" ";
    });
    text = with("zone", text, [](const M& m, size_t*) { return m.g(1) + U" " + spelled(m.g(2)); });
    text = sh_.rule("dot_half").replace(text, U"$1:$2");
    text = with("dot_times", text, [this](const M& m, size_t*) { return dot_times(m); });
    text = sh_.rule("clock_range").replace(text, U"$1 to ");
    text = with("ratio", text, [this](const M& m, size_t*) { return ratio(m); });
    text = with("time", text, [this](const M& m, size_t*) { return time(m); });
    text = with("hour_half", text, [this](const M& m, size_t*) {
        return U" " + cardinal(to_int(m.g(1))) + U" " + half_day(m, m.group(2), m.group(3));
    });

    // Which way round a numeric date is, from the text's own slash dates.
    {
        const R& seen_re = sh_.rule("seen_dates");
        std::vector<std::pair<long long, long long>> seen;
        size_t pos = 0;
        R::Match sm;
        while (pos <= text.size() && seen_re.search(text, pos, sm)) {
            const long long a = to_int(sm.group(text, 1)), b = to_int(sm.group(text, 2));
            if (!(a == 24 && b == 7)) seen.emplace_back(a, b);
            pos = sm.end == sm.begin ? sm.begin + 1 : sm.end;
        }
        bool any_df = false, any_mf = false;
        for (const auto& p : seen) {
            if (p.first > 12 && 12 >= p.second) any_df = true;
            if (p.second > 12 && 12 >= p.first) any_mf = true;
        }
        day_first_ = any_df && !any_mf;
        dated_ = sh_.rule("dated").test(text);
    }
    auto ymd = [this](const M& m, size_t*) {
        const auto d = date(to_int(m.g(2)), to_int(m.g(3)), m.group(1));
        return d ? *d : m.all();
    };
    text = with("iso_date", text, ymd);
    text = with("ymd_slash", text, ymd);
    text = with("dot_date", text, [this](const M& m, size_t*) { return dot_date(m); });
    text = with("slash_date", text, [this](const M& m, size_t*) { return slash_date(m); });
    text = with("month_days", text, [this](const M& m, size_t*) { return days_of_month(m); });
    text = with("days_month", text, [this](const M& m, size_t*) { return days_of_month(m); });
    text = with("month_day", text, [this](const M& m, size_t*) {
        const auto d = date(month_number(m.g(1), sh_), to_int(m.g(2)), m.group(3));
        return d ? *d : m.all();
    });
    text = with("day_month", text, [this](const M& m, size_t*) { return day_month(m); });

    text = with("phone", text, [this](const M& m, size_t*) { return phone(m); });
    text = with("phone7", text, [this](const M& m, size_t*) { return phone(m); });
    text = with("version3", text, [](const M& m, size_t*) {
        std::vector<u32> parts;
        for (const u32& p : split_on(m.g(2), U'.')) parts.push_back(version_part(p));
        return U" " + (truthy(m.group(1)) ? spelled(m.g(1)) + U" " : U"") + join(parts, U" point ") +
               (truthy(m.group(3)) ? U" dash " + number_words(m.g(3)) : U"") + U" ";
    });
    text = with("version_v", text, [](const M& m, size_t*) {
        return U" " + spelled(m.g(1)) + U" " + cardinal(to_int(m.g(2))) + U" point " + minor(m.g(3)) + U" ";
    });
    text = with("versioned", text, [](const M& m, size_t*) {
        return m.g(1) + m.g(2) + cardinal(to_int(m.g(3))) + U" point " + minor(m.g(4)) +
               (truthy(m.group(5)) ? m.g(5) + cardinal(to_int(m.g(6))) + U" point " + minor(m.g(7)) : U"") + U" ";
    });
    text = with("code", text, [this](const M& m, size_t*) { return code(m); });
    text = with("slashed", text, [this](const M& m, size_t*) { return U" " + digit_groups(split_on(m.g(1), U'/')) + U" "; });
    text = with("grouped", text, [this](const M& m, size_t*) { return grouped(m); });

    text = sh_.rule("negative").replace(text, U" minus ");
    text = sh_.rule("by").replace(text, U" by ");
    text = sh_.rule("times").replace(text, U" times ");
    text = with("chat", text, [this](const M& m, size_t*) { return chat_form(m); });
    text = with("decades", text, [](const M& m, size_t*) {
        const u32 n = m.g(1);
        return U" " + plural_of(n.size() == 4 ? year(to_int(n)) : cardinal(to_int(n))) + U" ";
    });
    text = with("ordinal", text, [](const M& m, size_t*) { return U" " + ordinal(to_int(m.g(1))) + U" "; });
    text = with("resolution", text, [](const M& m, size_t*) { return U" " + pairs(to_int(m.g(1))) + U" " + spelled(m.g(2)) + U" "; });
    text = with("percent", text, [](const M& m, size_t*) { return U" " + number_words(m.g(1)) + U" percent "; });
    text = with("area_plain", text, [this](const M& m, size_t*) { return square(m); });
    text = with("unit", text, [this](const M& m, size_t* extra) { return measure(m, extra); });
    text = with("mixed", text, [this](const M& m, size_t*) { return mixed_fraction(m); });
    text = with("fraction", text, [this](const M& m, size_t*) { return fraction(m); });
    text = with("year_range", text, [](const M& m, size_t*) {
        return U" " + year(to_int(m.g(1))) + U" to " + year(to_int(m.g(2))) + U" ";
    });
    text = with("number_range", text, [](const M& m, size_t*) {
        return U" " + number_words(m.g(1)) + U" to " + number_words(m.g(2)) + U" ";
    });
    text = with("nx", text, [](const M& m, size_t*) { return U" " + number_words(m.g(1)) + U" times "; });
    text = sh_.rule("power").replace(text, U" to the power of ");
    return text;
}

u32 Normalizer::signs(u32 text) const {
    using R = JsRegex;
    auto with = [this](const char* name, const u32& t, std::function<u32(const M&, size_t*)> fn) {
        const R& re = sh_.rule(name);
        return re.replace(t, [&](const u32& s, const R::Match& mm, size_t* extra) { return fn(M{s, mm, re}, extra); });
    };
    text = with("sizes", text, [](const M& m, size_t*) { return U" " + spelled(m.g(1)) + U" to " + spelled(m.g(2)) + U" "; });
    text = sh_.rule("hash_number").replace(text, U" number ");
    text = sh_.rule("hash_code").replace(text, U" number ");
    text = sh_.rule("hashtag").replace(text, U" hashtag ");
    text = sh_.rule("about").replace(text, U" about ");
    text = with("compare", text, [](const M& m, size_t*) {
        const u32 s = m.g(1);
        const u32 said = s == U">=" ? U"greater than or equal to" : s == U"<=" ? U"less than or equal to" : U"not equal to";
        return U" " + said + U" ";
    });
    text = sh_.rule("ti").replace(text, U" " + spelled(U"TI"));
    text = sh_.rule("ref_no").replace(text, U"$1eference number");
    if (text.find(U'.') != u32::npos) text = with("dotted_letters", text, [this](const M& m, size_t*) { return dotted(m); });
    text = with("abbreviation", text, [this](const M& m, size_t*) { return abbreviation(m); });
    if (has_digit(text)) {
        text = with("glued", text, [this](const M& m, size_t*) { return glued(m); });
        text = with("number", text, [this](const M& m, size_t*) { return number(m); });
    }
    text = sh_.rule("slash_words").replace(text, U" ");
    text = with("less_greater", text, [](const M& m, size_t*) {
        return m.all().find(U'<') != u32::npos ? U" less than " : U" greater than ";
    });
    text = with("symbols", text, [this](const M& m, size_t*) { return U" " + t_.symbol_words.at(m.all()) + U" "; });
    return text;
}

u32 Normalizer::spell(const u32& letters) const {   // kaleTts.js normalizer's `spell`
    const u32 upper = sh_.to_upper(letters);
    if (upper.size() <= 5 && !(upper.size() >= 3 && is_word_(sh_.to_lower(letters)))) return upper;
    u32 out;
    for (size_t i = 0; i < upper.size(); ++i) {
        if (i) out += U' ';
        out += upper[i];
    }
    return out;
}

u32 Normalizer::spoken_text(const u32& text) const {
    using R = JsRegex;
    u32 t = rewrite(text);
    t = sh_.rule("spelled").replace(t, [this](const u32& s, const R::Match& m, size_t*) {
        return U" " + spell(replace_all(m.group(s, 1), U" ", U"")) + U" ";
    });
    t = sh_.rule("spaces").replace(t, U" ");
    t = sh_.rule("space_before_mark").replace(t, U"");
    t = sh_.rule("mark_before_mark").replace(t, [](const u32& s, const R::Match& m, size_t*) {
        return s[m.begin] == s[m.begin + 1] ? U"" : s.substr(m.begin, 1);
    });
    return trim(t);
}

u32 Normalizer::operator()(const u32& text) const {
    using R = JsRegex;
    // sizeSuffixes
    u32 t = sh_.rule("size_suffix").replace(text, [](const u32& s, const R::Match& m, size_t*) {
        const char32_t c = s[m.groups[2].first];
        const u32 w = c == U'M' ? U"million" : c == U'B' ? U"billion" : U"thousand";
        return m.group(s, 1) + U" " + w;
    });
    t = sh_.rule("size_times").replace(t, [](const u32& s, const R::Match& m, size_t*) { return m.group(s, 1) + U" times"; });
    t = sh_.rule("size_seconds").replace(t, [this](const u32& s, const R::Match& m, size_t*) {
        return sh_.rule("seconds_tail").replace(m.group(s, 1), U"") + U" sec";
    });
    t = sh_.rule("dr_end").replace(t, U"Doctor");
    t = spoken_text(t);
    // dottedNames
    return sh_.rule("dotted_names").replace(t, [](const u32& s, const R::Match& m, size_t*) {
        const u32 name = m.group(s, 0);
        const auto parts = split_on(name, U'.');
        if (std::all_of(parts.begin(), parts.end(), [](const u32& x) { return x.size() == 1; })) return name;
        return join(parts, U" dot ");
    });
}

} // namespace kale
} // namespace loom
