#include "loom/core/js_regex.h"

#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace loom {
namespace {

constexpr size_t kNone = std::numeric_limits<size_t>::max();
constexpr size_t kUnbounded = std::numeric_limits<size_t>::max();

bool is_digit(char32_t c) { return c >= U'0' && c <= U'9'; }
// ECMAScript's `\w`: ASCII letters, digits and `_`, with or without the `u` flag.
bool is_word(char32_t c) {
    return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z') || is_digit(c) || c == U'_';
}
// ECMAScript's WhiteSpace and LineTerminator (`\s`).
bool is_space(char32_t c) {
    switch (c) {
        case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20: case 0xA0: case 0x1680:
        case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000: case 0xFEFF:
            return true;
        default:
            return c >= 0x2000 && c <= 0x200A;
    }
}
bool is_line_terminator(char32_t c) { return c == 0x0A || c == 0x0D || c == 0x2028 || c == 0x2029; }
char32_t fold(char32_t c) { return to_lower(c); }

enum class Prop { None, L, N, Ll, Lu, Mn };

bool prop_test(Prop p, char32_t c) {
    switch (p) {
        case Prop::L: return is_letter(c);
        case Prop::N: return is_number(c);
        case Prop::Ll: return is_lowercase_letter(c);
        // Lu and Lt together: no rule here can meet a titlecase letter (they are a handful of digraphs).
        case Prop::Lu: return is_upper_or_titlecase(c) && is_letter(c);
        // M*: the rules use \p{Mn} after NFD on Latin text, where every mark is non-spacing.
        case Prop::Mn: return is_mark(c);
        default: return false;
    }
}

struct CharClass {
    std::vector<std::pair<char32_t, char32_t>> ranges;
    std::vector<std::pair<Prop, bool>> props;       // (property, negated)
    bool digit = false, not_digit = false, word = false, not_word = false, space = false, not_space = false;
    bool negated = false;

    bool raw(char32_t c) const {
        for (const auto& r : ranges) {
            if (c >= r.first && c <= r.second) return true;
        }
        for (const auto& p : props) {
            if (prop_test(p.first, c) != p.second) return true;
        }
        return (digit && is_digit(c)) || (not_digit && !is_digit(c)) || (word && is_word(c)) ||
               (not_word && !is_word(c)) || (space && is_space(c)) || (not_space && !is_space(c));
    }
    bool test(char32_t c, bool icase) const {
        bool hit = raw(c);
        if (!hit && icase) {
            // Both cases of a letter: the class may hold either spelling.
            const char32_t lo = to_lower(c);
            hit = lo != c && raw(lo);
            if (!hit && c >= U'a' && c <= U'z') hit = raw(c - 32);
            if (!hit) {
                for (const auto& r : ranges) {
                    for (char32_t x = r.first; x <= r.second && x - r.first < 512; ++x) {
                        if (x != c && to_lower(x) == lo) { hit = true; break; }
                    }
                    if (hit) break;
                }
            }
        }
        return hit != negated;
    }
};

enum class Kind {
    Char, Any, Class, Start, End, WordBoundary, NotWordBoundary, Group, BackRef,
    LookAhead, NegLookAhead, LookBehind, NegLookBehind
};

struct Node {
    Kind kind = Kind::Char;
    char32_t ch = 0;
    CharClass cls;
    int cap = -1;                                   // Group: capture index, -1 for (?:...)
    std::vector<std::vector<Node>> alts;            // Group and lookarounds: the alternatives
    int ref = 0;
    size_t min = 1, max = 1;
    bool lazy = false;
    // Lookbehind: the body's narrowest and widest match, so the assertion tries only the starts that can work.
    size_t lb_min = 0, lb_max = kUnbounded;

    bool single_char() const { return kind == Kind::Char || kind == Kind::Any || kind == Kind::Class; }
    bool assertion() const {
        return kind == Kind::Start || kind == Kind::End || kind == Kind::WordBoundary ||
               kind == Kind::NotWordBoundary || kind == Kind::LookAhead || kind == Kind::NegLookAhead ||
               kind == Kind::LookBehind || kind == Kind::NegLookBehind;
    }
};

class Parser {
public:
    Parser(const std::string& pattern, bool unicode) : src_(pattern), p_(utf8_decode(pattern)), unicode_(unicode) {}

    std::vector<std::vector<Node>> parse(int* n_groups, std::unordered_map<std::string, size_t>* names) {
        auto alts = parse_alts();
        if (i_ != p_.size()) fail("an unbalanced ')'");
        *n_groups = groups_;
        *names = names_;
        return alts;
    }

private:
    [[noreturn]] void fail(const std::string& what) const {
        throw LoadError("JsRegex: pattern '" + src_ + "': " + what + " at offset " + std::to_string(i_));
    }
    bool at_end() const { return i_ >= p_.size(); }
    char32_t peek() const { return p_[i_]; }

    std::vector<std::vector<Node>> parse_alts() {
        std::vector<std::vector<Node>> alts;
        alts.push_back(parse_seq());
        while (!at_end() && peek() == U'|') {
            ++i_;
            alts.push_back(parse_seq());
        }
        return alts;
    }

    std::vector<Node> parse_seq() {
        std::vector<Node> seq;
        while (!at_end() && peek() != U'|' && peek() != U')') {
            Node n = parse_atom();
            parse_quantifier(n);
            seq.push_back(std::move(n));
        }
        return seq;
    }

    bool class_escape(char32_t e, CharClass& cls) {
        switch (e) {
            case U'd': cls.digit = true; return true;
            case U'D': cls.not_digit = true; return true;
            case U'w': cls.word = true; return true;
            case U'W': cls.not_word = true; return true;
            case U's': cls.space = true; return true;
            case U'S': cls.not_space = true; return true;
            case U'p': case U'P': {
                if (at_end() || peek() != U'{') fail("a \\p without {...}");
                ++i_;
                std::string name;
                while (!at_end() && peek() != U'}') name += static_cast<char>(p_[i_++]);
                if (at_end()) fail("an unterminated \\p{");
                ++i_;
                Prop p = Prop::None;
                if (name == "L" || name == "Letter") p = Prop::L;
                else if (name == "N" || name == "Number") p = Prop::N;
                else if (name == "Ll" || name == "Lowercase_Letter") p = Prop::Ll;
                else if (name == "Lu" || name == "Uppercase_Letter") p = Prop::Lu;
                else if (name == "Mn" || name == "Nonspacing_Mark") p = Prop::Mn;
                else fail("the property \\p{" + name + "}, which this subset does not have");
                cls.props.emplace_back(p, e == U'P');
                return true;
            }
            default: return false;
        }
    }

    uint32_t hex(size_t n) {
        uint32_t v = 0;
        for (size_t k = 0; k < n; ++k) {
            if (at_end()) fail("a short hex escape");
            const char32_t c = p_[i_++];
            int d = (c >= U'0' && c <= U'9') ? int(c - U'0') : (c >= U'a' && c <= U'f') ? int(c - U'a' + 10)
                    : (c >= U'A' && c <= U'F') ? int(c - U'A' + 10) : -1;
            if (d < 0) fail("a hex escape with a non-hex digit");
            v = v * 16 + static_cast<uint32_t>(d);
        }
        return v;
    }

    char32_t literal_escape(char32_t e) {
        switch (e) {
            case U'n': return U'\n';
            case U't': return U'\t';
            case U'r': return U'\r';
            case U'f': return U'\f';
            case U'v': return U'\v';
            case U'0': return U'\0';
            case U'x': return hex(2);
            case U'u': {
                if (!at_end() && peek() == U'{') {
                    ++i_;
                    uint32_t v = 0;
                    size_t n = 0;
                    while (!at_end() && peek() != U'}') {
                        const char32_t c = p_[i_];
                        const int d = (c >= U'0' && c <= U'9') ? int(c - U'0') : (c >= U'a' && c <= U'f') ? int(c - U'a' + 10)
                                      : (c >= U'A' && c <= U'F') ? int(c - U'A' + 10) : -1;
                        if (d < 0) fail("a \\u{...} with a non-hex digit");
                        v = v * 16 + static_cast<uint32_t>(d);
                        ++i_;
                        ++n;
                    }
                    if (at_end() || n == 0) fail("an unterminated \\u{");
                    ++i_;
                    return v;
                }
                return hex(4);
            }
            default: break;
        }
        if (e < 0x80 && ((e >= U'a' && e <= U'z') || (e >= U'A' && e <= U'Z') || is_digit(e))) {
            fail(std::string("the escape \\") + static_cast<char>(e) + ", which this subset does not have");
        }
        return e;   // an escaped syntax character or punctuation is itself
    }

    Node parse_atom() {
        Node n;
        const char32_t c = p_[i_++];
        switch (c) {
            case U'.': n.kind = Kind::Any; return n;
            case U'^': n.kind = Kind::Start; return n;
            case U'$': n.kind = Kind::End; return n;
            case U'[': n.kind = Kind::Class; n.cls = parse_class(); return n;
            case U'(': {
                n.kind = Kind::Group;
                if (!at_end() && peek() == U'?') {
                    const char32_t a = i_ + 1 < p_.size() ? p_[i_ + 1] : 0;
                    const char32_t b = i_ + 2 < p_.size() ? p_[i_ + 2] : 0;
                    if (a == U':') { i_ += 2; }
                    else if (a == U'=') { i_ += 2; n.kind = Kind::LookAhead; }
                    else if (a == U'!') { i_ += 2; n.kind = Kind::NegLookAhead; }
                    else if (a == U'<' && b == U'=') { i_ += 3; n.kind = Kind::LookBehind; }
                    else if (a == U'<' && b == U'!') { i_ += 3; n.kind = Kind::NegLookBehind; }
                    else if (a == U'<') {
                        i_ += 2;
                        std::string name;
                        while (!at_end() && peek() != U'>') name += static_cast<char>(p_[i_++]);
                        if (at_end() || name.empty()) fail("an unterminated group name");
                        ++i_;
                        n.cap = ++groups_;
                        if (!names_.emplace(name, static_cast<size_t>(n.cap)).second) fail("a repeated group name '" + name + "'");
                    } else {
                        fail("a '(?' extension other than (?:) (?=) (?!) (?<=) (?<!) (?<name>)");
                    }
                } else {
                    n.cap = ++groups_;
                }
                n.alts = parse_alts();
                if (at_end() || peek() != U')') fail("a missing ')'");
                ++i_;
                return n;
            }
            case U'*': case U'+': case U'?': fail("a quantifier with nothing to repeat");
            case U'\\': {
                if (at_end()) fail("a trailing backslash");
                const char32_t e = p_[i_++];
                if (e == U'b') { n.kind = Kind::WordBoundary; return n; }
                if (e == U'B') { n.kind = Kind::NotWordBoundary; return n; }
                if (e >= U'1' && e <= U'9') {
                    n.kind = Kind::BackRef;
                    n.ref = static_cast<int>(e - U'0');
                    return n;
                }
                if (class_escape(e, n.cls)) { n.kind = Kind::Class; return n; }
                n.kind = Kind::Char;
                n.ch = literal_escape(e);
                return n;
            }
            default:
                n.kind = Kind::Char;
                n.ch = c;
                return n;
        }
    }

    CharClass parse_class() {
        CharClass cls;
        if (!at_end() && peek() == U'^') {
            cls.negated = true;
            ++i_;
        }
        // ECMAScript: `[]` is an empty class and `]` first is the end, unlike Python.
        while (true) {
            if (at_end()) fail("an unterminated '['");
            const char32_t c = p_[i_];
            if (c == U']') { ++i_; break; }
            ++i_;
            char32_t lo;
            if (c == U'\\') {
                if (at_end()) fail("a trailing backslash");
                const char32_t e = p_[i_++];
                if (class_escape(e, cls)) continue;
                lo = e == U'b' ? U'\b' : e == U'-' ? U'-' : literal_escape(e);
            } else {
                lo = c;
            }
            if (i_ + 1 < p_.size() && p_[i_] == U'-' && p_[i_ + 1] != U']') {
                ++i_;
                char32_t hi = p_[i_++];
                if (hi == U'\\') {
                    if (at_end()) fail("a trailing backslash");
                    const char32_t e = p_[i_++];
                    CharClass probe;
                    if (e == U'd' || e == U'D' || e == U'w' || e == U'W' || e == U's' || e == U'S' || e == U'p' || e == U'P') {
                        fail("a range ending in a class escape");
                    }
                    hi = literal_escape(e);
                }
                if (hi < lo) fail("a reversed range");
                cls.ranges.emplace_back(lo, hi);
            } else {
                cls.ranges.emplace_back(lo, lo);
            }
        }
        return cls;
    }

    bool parse_number(size_t* out) {
        const size_t start = i_;
        size_t v = 0;
        while (!at_end() && is_digit(peek())) v = v * 10 + static_cast<size_t>(p_[i_++] - U'0');
        *out = v;
        return i_ > start;
    }

    void parse_quantifier(Node& n) {
        if (at_end()) return;
        const char32_t c = peek();
        size_t lo = 1, hi = 1;
        if (c == U'?') { lo = 0; hi = 1; ++i_; }
        else if (c == U'*') { lo = 0; hi = kUnbounded; ++i_; }
        else if (c == U'+') { lo = 1; hi = kUnbounded; ++i_; }
        else if (c == U'{') {
            const size_t save = i_;
            ++i_;
            size_t a = 0, b = 0;
            const bool has_a = parse_number(&a);
            if (!has_a) { i_ = save; return; }    // a literal '{' (Annex B)
            if (!at_end() && peek() == U'}') {
                ++i_;
                lo = hi = a;
            } else if (!at_end() && peek() == U',') {
                ++i_;
                const bool has_b = parse_number(&b);
                if (at_end() || peek() != U'}') { i_ = save; return; }
                ++i_;
                lo = a;
                hi = has_b ? b : kUnbounded;
                if (hi < lo) fail("a {min,max} with max below min");
            } else {
                i_ = save;
                return;
            }
        } else {
            return;
        }
        if (n.kind == Kind::LookBehind || n.kind == Kind::NegLookBehind || n.kind == Kind::Start ||
            n.kind == Kind::End || n.kind == Kind::WordBoundary || n.kind == Kind::NotWordBoundary) {
            fail("a quantified assertion");
        }
        if (!at_end() && peek() == U'?') { n.lazy = true; ++i_; }
        n.min = lo;
        n.max = hi;
    }

    std::string src_;
    std::vector<char32_t> p_;
    bool unicode_;
    size_t i_ = 0;
    int groups_ = 0;
    std::unordered_map<std::string, size_t> names_;
};

struct Ctx {
    const std::u32string& s;
    bool icase, dotall;
    std::vector<std::pair<size_t, size_t>> caps;
};

using Cont = std::function<bool(size_t)>;

bool single(const Node& n, const Ctx& c, size_t pos) {
    if (pos >= c.s.size()) return false;
    const char32_t ch = c.s[pos];
    switch (n.kind) {
        case Kind::Char: return ch == n.ch || (c.icase && fold(ch) == fold(n.ch));
        case Kind::Any: return c.dotall || !is_line_terminator(ch);
        case Kind::Class: return n.cls.test(ch, c.icase);
        default: return false;
    }
}

bool seq(const std::vector<Node>& nodes, size_t i, size_t pos, Ctx& c, const Cont& k);

size_t sat_add(size_t a, size_t b) { return (a == kUnbounded || b == kUnbounded) ? kUnbounded : a + b; }
size_t sat_mul(size_t a, size_t b) {
    if (a == 0 || b == 0) return 0;
    if (a == kUnbounded || b == kUnbounded) return kUnbounded;
    return a > kUnbounded / b ? kUnbounded : a * b;
}

std::pair<size_t, size_t> alts_width(const std::vector<std::vector<Node>>& alts);

// The narrowest and widest a node can match (a back reference is unbounded).
std::pair<size_t, size_t> node_width(const Node& n) {
    std::pair<size_t, size_t> one{0, 0};
    if (n.single_char()) one = {1, 1};
    else if (n.kind == Kind::Group) one = alts_width(n.alts);
    else if (n.kind == Kind::BackRef) one = {0, kUnbounded};
    else return {0, 0};   // assertions
    return {sat_mul(one.first, n.min), sat_mul(one.second, n.max)};
}

std::pair<size_t, size_t> alts_width(const std::vector<std::vector<Node>>& alts) {
    size_t lo = kUnbounded, hi = 0;
    for (const auto& alt : alts) {
        size_t a = 0, b = 0;
        for (const auto& n : alt) {
            const auto w = node_width(n);
            a = sat_add(a, w.first);
            b = sat_add(b, w.second);
        }
        lo = std::min(lo, a);
        hi = std::max(hi, b);
    }
    return {lo == kUnbounded ? 0 : lo, hi};
}

void annotate(std::vector<std::vector<Node>>& alts) {
    for (auto& alt : alts) {
        for (auto& n : alt) {
            if (!n.alts.empty()) annotate(n.alts);
            if (n.kind == Kind::LookBehind || n.kind == Kind::NegLookBehind) {
                const auto w = alts_width(n.alts);
                n.lb_min = w.first;
                n.lb_max = w.second;
            }
        }
    }
}

// The single-character nodes one of which every match must start with, or false when that cannot be said
// (an alternative that may start empty, or with a back reference). Assertions before the first one are skipped.
bool first_set(const std::vector<std::vector<Node>>& alts, std::vector<const Node*>& out) {
    for (const auto& alt : alts) {
        bool found = false;
        for (const auto& n : alt) {
            if (n.assertion()) continue;
            if (n.min == 0) return false;
            if (n.single_char()) { out.push_back(&n); found = true; break; }
            if (n.kind == Kind::Group) {
                if (!first_set(n.alts, out)) return false;
                found = true;
                break;
            }
            return false;
        }
        if (!found) return false;
    }
    return true;
}

bool alts_match(const std::vector<std::vector<Node>>& alts, size_t pos, Ctx& c, const Cont& k) {
    for (const auto& alt : alts) {
        if (seq(alt, 0, pos, c, k)) return true;
    }
    return false;
}

bool assertion(const Node& n, Ctx& c, size_t pos) {
    const size_t len = c.s.size();
    switch (n.kind) {
        case Kind::Start: return pos == 0;
        case Kind::End: return pos == len;
        case Kind::WordBoundary:
        case Kind::NotWordBoundary: {
            const bool before = pos > 0 && is_word(c.s[pos - 1]);
            const bool after = pos < len && is_word(c.s[pos]);
            return n.kind == Kind::WordBoundary ? before != after : before == after;
        }
        case Kind::LookAhead:
        case Kind::NegLookAhead: {
            const auto saved = c.caps;
            const bool hit = alts_match(n.alts, pos, c, [](size_t) { return true; });
            if (n.kind == Kind::NegLookAhead || !hit) c.caps = saved;
            return n.kind == Kind::LookAhead ? hit : !hit;
        }
        case Kind::LookBehind:
        case Kind::NegLookBehind: {
            const auto saved = c.caps;
            bool hit = false;
            if (pos >= n.lb_min) {
                const size_t lo = n.lb_max == kUnbounded || n.lb_max > pos ? 0 : pos - n.lb_max;
                for (size_t q = pos - n.lb_min + 1; q-- > lo && !hit;) {
                    hit = alts_match(n.alts, q, c, [pos](size_t p) { return p == pos; });
                }
            }
            if (n.kind == Kind::NegLookBehind || !hit) c.caps = saved;
            return n.kind == Kind::LookBehind ? hit : !hit;
        }
        default: return false;
    }
}

bool one(const Node& n, size_t pos, Ctx& c, const Cont& k) {
    if (n.kind == Kind::BackRef) {
        const size_t r = static_cast<size_t>(n.ref);
        if (r >= c.caps.size() || c.caps[r].first == kNone) return k(pos);   // ECMAScript: matches empty
        const auto g = c.caps[r];
        const size_t len = g.second - g.first;
        if (pos + len > c.s.size()) return false;
        for (size_t j = 0; j < len; ++j) {
            const char32_t a = c.s[g.first + j], b = c.s[pos + j];
            if (a != b && !(c.icase && fold(a) == fold(b))) return false;
        }
        return k(pos + len);
    }
    for (const auto& alt : n.alts) {
        const bool ok = seq(alt, 0, pos, c, [&](size_t p) {
            if (n.cap < 0) return k(p);
            const auto prev = c.caps[static_cast<size_t>(n.cap)];
            c.caps[static_cast<size_t>(n.cap)] = {pos, p};
            if (k(p)) return true;
            c.caps[static_cast<size_t>(n.cap)] = prev;
            return false;
        });
        if (ok) return true;
    }
    return false;
}

bool rep(const Node& n, size_t count, size_t pos, Ctx& c, const Cont& next) {
    const auto more = [&]() {
        return count < n.max && one(n, pos, c, [&](size_t p) {
            // An iteration that consumed nothing ends the loop once the minimum is met (ECMAScript's
            // empty-check), so `(x*)*` terminates.
            if (p == pos && count + 1 > n.min) return false;
            return rep(n, count + 1, p, c, next);
        });
    };
    if (n.lazy) {
        if (count >= n.min && next(pos)) return true;
        return more();
    }
    if (more()) return true;
    return count >= n.min && next(pos);
}

bool seq(const std::vector<Node>& nodes, size_t i, size_t pos, Ctx& c, const Cont& k) {
    if (i == nodes.size()) return k(pos);
    const Node& n = nodes[i];
    const Cont next = [&](size_t p) { return seq(nodes, i + 1, p, c, k); };
    if (n.assertion() && n.min == 1 && n.max == 1) return assertion(n, c, pos) && next(pos);
    if (n.single_char()) {
        size_t m = 0;
        while (m < n.max && single(n, c, pos + m)) ++m;
        if (m < n.min) return false;
        if (n.lazy) {
            for (size_t j = n.min; j <= m; ++j) {
                if (next(pos + j)) return true;
            }
            return false;
        }
        for (size_t j = m + 1; j-- > n.min;) {
            if (next(pos + j)) return true;
        }
        return false;
    }
    if (n.assertion()) {   // a quantified lookahead: {0,1} is optional, anything else is the assertion once
        if (n.min == 0) return next(pos);
        return assertion(n, c, pos) && next(pos);
    }
    return rep(n, 0, pos, c, next);
}

} // namespace

struct JsRegex::Impl {
    std::vector<std::vector<Node>> alts;
    int n_groups = 0;
    std::unordered_map<std::string, size_t> names;
    bool icase = false, dotall = false;
    std::vector<const Node*> first;   // empty: no prefilter
    bool has_first = false;

    bool may_start(const std::u32string& s, size_t pos) const {
        if (!has_first) return true;
        Ctx c{s, icase, dotall, {}};
        for (const Node* n : first) if (single(*n, c, pos)) return true;
        return false;
    }

    bool match_at(const std::u32string& s, size_t pos, Match& m) const {
        Ctx c{s, icase, dotall, std::vector<std::pair<size_t, size_t>>(static_cast<size_t>(n_groups) + 1, {kNone, kNone})};
        for (const auto& alt : alts) {
            bool done = false;
            const bool ok = seq(alt, 0, pos, c, [&](size_t p) {
                c.caps[0] = {pos, p};
                done = true;
                return true;
            });
            if (ok && done) {
                m.begin = pos;
                m.end = c.caps[0].second;
                m.groups = c.caps;
                return true;
            }
        }
        return false;
    }
};

bool JsRegex::Match::has(size_t i) const { return i < groups.size() && groups[i].first != kNone; }

std::u32string JsRegex::Match::group(const std::u32string& text, size_t i) const {
    if (!has(i)) return {};
    return text.substr(groups[i].first, groups[i].second - groups[i].first);
}

JsRegex::JsRegex(const std::string& pattern_utf8, const std::string& flags)
    : impl_(new Impl()), pattern_(pattern_utf8) {
    bool unicode = false;
    for (char f : flags) {
        if (f == 'i') impl_->icase = true;
        else if (f == 's') impl_->dotall = true;
        else if (f == 'u') unicode = true;
        else if (f == 'g' || f == 'y') continue;
        else throw LoadError(std::string("JsRegex: pattern '") + pattern_utf8 + "': the flag '" + f +
                             "', which this subset does not have");
    }
    Parser parser(pattern_utf8, unicode);
    impl_->alts = parser.parse(&impl_->n_groups, &impl_->names);
    annotate(impl_->alts);
    std::vector<const Node*> first;
    if (first_set(impl_->alts, first)) {
        impl_->first = std::move(first);
        impl_->has_first = true;
    }
}
JsRegex::~JsRegex() = default;
JsRegex::JsRegex(JsRegex&&) noexcept = default;
JsRegex& JsRegex::operator=(JsRegex&&) noexcept = default;

size_t JsRegex::n_groups() const { return static_cast<size_t>(impl_->n_groups); }

size_t JsRegex::group_index(const std::string& name) const {
    const auto it = impl_->names.find(name);
    return it == impl_->names.end() ? 0 : it->second;
}

bool JsRegex::match_at(const std::u32string& text, size_t pos, Match& m) const {
    return pos <= text.size() && impl_->match_at(text, pos, m);
}

bool JsRegex::search(const std::u32string& text, size_t pos, Match& m) const {
    for (size_t p = pos; p <= text.size(); ++p) {
        if (impl_->has_first && (p == text.size() || !impl_->may_start(text, p))) continue;
        if (impl_->match_at(text, p, m)) return true;
    }
    return false;
}

bool JsRegex::test(const std::u32string& text) const {
    Match m;
    return search(text, 0, m);
}

std::u32string JsRegex::replace(const std::u32string& text, const Replacer& fn) const {
    std::u32string out;
    size_t pos = 0;   // where the copied text resumes
    size_t from = 0;  // where the next search starts (`lastIndex`)
    Match m;
    while (from <= text.size() && search(text, from, m)) {
        size_t extra = 0;
        if (m.begin < pos) {   // inside text a previous replacement consumed: skipped, as dizione's `sub`
            from = m.end == m.begin ? m.begin + 1 : m.end;
            continue;
        }
        const std::u32string r = fn(text, m, &extra);
        out.append(text, pos, m.begin - pos);
        out += r;
        pos = std::min(text.size(), m.end + extra);
        from = m.end == m.begin ? m.end + 1 : m.end;
    }
    if (pos < text.size()) out.append(text, pos, std::u32string::npos);
    return out;
}

std::u32string JsRegex::replace(const std::u32string& text, const std::u32string& templ) const {
    return replace(text, [&](const std::u32string& s, const Match& m, size_t*) {
        std::u32string r;
        for (size_t i = 0; i < templ.size(); ++i) {
            if (templ[i] != U'$' || i + 1 >= templ.size()) { r += templ[i]; continue; }
            const char32_t n = templ[i + 1];
            if (n == U'$') { r += U'$'; ++i; continue; }
            if (n == U'&') { r += s.substr(m.begin, m.end - m.begin); ++i; continue; }
            if (n == U'<') {
                const size_t close = templ.find(U'>', i + 2);
                if (close == std::u32string::npos) { r += templ[i]; continue; }
                std::string name;
                for (size_t k = i + 2; k < close; ++k) name += static_cast<char>(templ[k]);
                r += m.group(s, group_index(name));
                i = close;
                continue;
            }
            if (n >= U'0' && n <= U'9') {
                size_t g = static_cast<size_t>(n - U'0');
                size_t used = 1;
                if (i + 2 < templ.size() && templ[i + 2] >= U'0' && templ[i + 2] <= U'9') {
                    const size_t two = g * 10 + static_cast<size_t>(templ[i + 2] - U'0');
                    if (two >= 1 && two <= n_groups()) { g = two; used = 2; }
                }
                if (g >= 1 && g <= n_groups()) {
                    r += m.group(s, g);
                    i += used;
                    continue;
                }
            }
            r += templ[i];
        }
        return r;
    });
}

std::vector<std::u32string> JsRegex::match_all(const std::u32string& text) const {
    std::vector<std::u32string> out;
    size_t pos = 0;
    Match m;
    while (pos <= text.size() && search(text, pos, m)) {
        out.push_back(text.substr(m.begin, m.end - m.begin));
        pos = m.end == m.begin ? m.begin + 1 : m.end;
    }
    return out;
}

std::vector<std::u32string> JsRegex::split(const std::u32string& text) const {
    // ECMAScript's @@split: a match may not be empty at the start nor end at the position the last one did.
    std::vector<std::u32string> out;
    if (text.empty()) {
        Match m;
        if (!match_at(text, 0, m)) out.push_back(text);
        return out;
    }
    size_t p = 0, q = 0;
    Match m;
    while (q < text.size()) {
        if (!match_at(text, q, m) || m.end == p) {
            ++q;
            continue;
        }
        out.push_back(text.substr(p, q - p));
        p = m.end;
        q = p;
    }
    out.push_back(text.substr(p));
    return out;
}

} // namespace loom
