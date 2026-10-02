#include "loom/core/py_regex.h"

#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

#include <limits>
#include <utility>

namespace loom {
namespace {

constexpr size_t kNone = std::numeric_limits<size_t>::max();
constexpr size_t kUnbounded = std::numeric_limits<size_t>::max();

bool is_digit(char32_t c) { return c >= U'0' && c <= U'9'; }
bool is_word(char32_t c) {
    if (c < 0x80) return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z') || is_digit(c) || c == U'_';
    // Python's is `str.isalnum()`; `\p{L}` and `\p{N}` are that up to a handful of codepoints.
    return is_letter_or_number(c);
}
// `str.isspace()`, which is what `\s` matches in a str pattern -- ASCII's six, the four information
// separators `\x1c`-`\x1f` (Python counts them, ECMAScript does not), and the Unicode spaces.
bool is_space(char32_t c) { return is_python_space(c); }

char32_t fold(char32_t c) { return (c >= U'A' && c <= U'Z') ? c + 32 : c; }
char32_t swap_case(char32_t c) {
    if (c >= U'A' && c <= U'Z') return c + 32;
    if (c >= U'a' && c <= U'z') return c - 32;
    return c;
}

struct CharClass {
    std::vector<std::pair<char32_t, char32_t>> ranges;
    bool digit = false, not_digit = false, word = false, not_word = false, space = false, not_space = false;
    bool negated = false;

    bool raw(char32_t c) const {
        for (const auto& r : ranges) {
            if (c >= r.first && c <= r.second) return true;
        }
        return (digit && is_digit(c)) || (not_digit && !is_digit(c)) || (word && is_word(c)) ||
               (not_word && !is_word(c)) || (space && is_space(c)) || (not_space && !is_space(c));
    }
    bool test(char32_t c, bool icase) const {
        const bool hit = raw(c) || (icase && raw(swap_case(c)));
        return hit != negated;
    }
};

enum class Kind { Char, Any, Class, Start, End, EndAbsolute, WordBoundary, NotWordBoundary, Group, BackRef };

struct Node {
    Kind kind = Kind::Char;
    char32_t ch = 0;
    CharClass cls;
    int cap = -1;                                   // Group: capture index, -1 for (?:...)
    std::vector<std::vector<Node>> alts;            // Group: its alternatives
    int ref = 0;                                    // BackRef: the group it names
    size_t min = 1, max = 1;

    bool single_char() const { return kind == Kind::Char || kind == Kind::Any || kind == Kind::Class; }
    bool assertion() const {
        return kind == Kind::Start || kind == Kind::End || kind == Kind::EndAbsolute ||
               kind == Kind::WordBoundary || kind == Kind::NotWordBoundary;
    }
};

class Parser {
public:
    Parser(const std::string& pattern) : src_(pattern), p_(utf8_decode(pattern)) {}

    std::vector<std::vector<Node>> parse(int* n_groups) {
        auto alts = parse_alts();
        if (i_ != p_.size()) fail("an unbalanced ')'");
        *n_groups = groups_;
        return alts;
    }

private:
    [[noreturn]] void fail(const std::string& what) const {
        throw LoadError("PyRegex: pattern '" + src_ + "': " + what + " at offset " + std::to_string(i_));
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

    // Adds an escape's class (`\d`...) to `cls`; false when the escape is not a class.
    bool class_escape(char32_t e, CharClass& cls) const {
        switch (e) {
            case U'd': cls.digit = true; return true;
            case U'D': cls.not_digit = true; return true;
            case U'w': cls.word = true; return true;
            case U'W': cls.not_word = true; return true;
            case U's': cls.space = true; return true;
            case U'S': cls.not_space = true; return true;
            default: return false;
        }
    }
    char32_t literal_escape(char32_t e) const {
        switch (e) {
            case U'n': return U'\n';
            case U't': return U'\t';
            case U'r': return U'\r';
            case U'f': return U'\f';
            case U'v': return U'\v';
            default: break;
        }
        if (e < 0x80 && ((e >= U'a' && e <= U'z') || (e >= U'A' && e <= U'Z') || is_digit(e))) {
            fail(std::string("the escape \\") + static_cast<char>(e) + ", which this subset does not have");
        }
        return e;   // an escaped metacharacter or punctuation is itself
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
                    if (i_ + 1 < p_.size() && p_[i_ + 1] == U':') {
                        i_ += 2;
                    } else {
                        fail("a '(?' extension other than (?:...)");
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
                // `\A` is `^` without MULTILINE, which this subset never has; `\Z` is the very end, where
                // `$` also accepts the position before a final newline.
                if (e == U'A') { n.kind = Kind::Start; return n; }
                if (e == U'Z') { n.kind = Kind::EndAbsolute; return n; }
                if (e == U'B') { n.kind = Kind::NotWordBoundary; return n; }
                if (e >= U'1' && e <= U'9') {
                    n.kind = Kind::BackRef;
                    n.ref = static_cast<int>(e - U'0');
                    if (n.ref > groups_) fail("a back reference to a group not yet opened");
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
        bool first = true;
        while (true) {
            if (at_end()) fail("an unterminated '['");
            char32_t c = p_[i_];
            if (c == U']' && !first) { ++i_; break; }
            first = false;
            ++i_;
            char32_t lo;
            if (c == U'\\') {
                if (at_end()) fail("a trailing backslash");
                const char32_t e = p_[i_++];
                if (class_escape(e, cls)) continue;
                lo = literal_escape(e);
            } else {
                lo = c;
            }
            // `a-b` is a range unless the '-' is last before ']' (then it is a literal, as is a leading one).
            if (i_ + 1 < p_.size() && p_[i_] == U'-' && p_[i_ + 1] != U']') {
                ++i_;
                char32_t hi = p_[i_++];
                if (hi == U'\\') {
                    if (at_end()) fail("a trailing backslash");
                    const char32_t e = p_[i_++];
                    CharClass probe;
                    if (class_escape(e, probe)) fail("a range ending in a class escape");
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
            // `{n}`, `{n,}`, `{n,m}`, and (Python 3.11) `{,m}` and `{,}` with the minimum 0; anything
            // else is a literal '{', as Python reads it.
            const size_t save = i_;
            ++i_;
            size_t a = 0, b = 0;
            const bool has_a = parse_number(&a);
            if (!at_end() && peek() == U'}' && has_a) {
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
        if (n.assertion()) fail("a quantified assertion");
        if (!at_end() && (peek() == U'?' || peek() == U'+')) fail("a lazy or possessive quantifier");
        n.min = lo;
        n.max = hi;
    }

    std::string src_;
    std::vector<char32_t> p_;
    size_t i_ = 0;
    int groups_ = 0;
};

struct Ctx {
    const std::u32string& s;
    bool icase;
    std::vector<std::pair<size_t, size_t>> caps;
};

using Cont = std::function<bool(size_t)>;

bool word_at(const std::u32string& s, size_t i) { return i < s.size() && is_word(s[i]); }

bool single(const Node& n, const Ctx& c, size_t pos) {
    if (pos >= c.s.size()) return false;
    const char32_t ch = c.s[pos];
    switch (n.kind) {
        case Kind::Char: return ch == n.ch || (c.icase && fold(ch) == fold(n.ch));
        case Kind::Any: return ch != U'\n';
        case Kind::Class: return n.cls.test(ch, c.icase);
        default: return false;
    }
}

bool assertion(const Node& n, const Ctx& c, size_t pos) {
    const size_t len = c.s.size();
    switch (n.kind) {
        case Kind::Start: return pos == 0;
        case Kind::End: return pos == len || (pos + 1 == len && c.s[pos] == U'\n');
        case Kind::EndAbsolute: return pos == len;
        case Kind::WordBoundary:
        case Kind::NotWordBoundary: {
            const bool before = pos > 0 && is_word(c.s[pos - 1]);
            const bool boundary = before != word_at(c.s, pos);
            return n.kind == Kind::WordBoundary ? boundary : !boundary;
        }
        default: return false;
    }
}

bool seq(const std::vector<Node>& nodes, size_t i, size_t pos, Ctx& c, const Cont& k);

// One iteration of a Group or BackRef node, continuing with `k`.
bool one(const Node& n, size_t pos, Ctx& c, const Cont& k) {
    if (n.kind == Kind::BackRef) {
        const auto g = c.caps[static_cast<size_t>(n.ref)];
        if (g.first == kNone) return false;           // a reference to a group that did not take part
        const size_t len = g.second - g.first;
        if (pos + len > c.s.size()) return false;
        for (size_t j = 0; j < len; ++j) {
            const char32_t a = c.s[g.first + j], b = c.s[pos + j];
            if (a != b && !(c.icase && fold(a) == fold(b))) return false;
        }
        return k(pos + len);
    }
    // Group.
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

// `count` iterations of `n` done, at `pos`; greedy: one more first, then what follows.
bool rep(const Node& n, size_t count, size_t pos, Ctx& c, const Cont& next) {
    if (count < n.max) {
        const bool more = one(n, pos, c, [&](size_t p) {
            // An iteration that consumed nothing ends the loop (Python's guard against `(x*)*`), unless
            // the minimum still needs it.
            if (p == pos && count + 1 > n.min) return false;
            return rep(n, count + 1, p, c, next);
        });
        if (more) return true;
    }
    return count >= n.min && next(pos);
}

bool seq(const std::vector<Node>& nodes, size_t i, size_t pos, Ctx& c, const Cont& k) {
    if (i == nodes.size()) return k(pos);
    const Node& n = nodes[i];
    const Cont next = [&](size_t p) { return seq(nodes, i + 1, p, c, k); };
    if (n.assertion()) return assertion(n, c, pos) && next(pos);
    if (n.single_char()) {
        // Iterative, not recursive: the longest run first, then shorter ones, so that `\s+` over a long
        // run of spaces or `.*` over a whole text costs no stack per character.
        size_t m = 0;
        while (m < n.max && single(n, c, pos + m)) ++m;
        if (m < n.min) return false;
        for (size_t j = m + 1; j-- > n.min;) {
            if (next(pos + j)) return true;
        }
        return false;
    }
    return rep(n, 0, pos, c, next);
}

} // namespace

struct PyRegex::Impl {
    std::vector<std::vector<Node>> alts;
    int n_groups = 0;
    bool icase = false;

    bool match_at(const std::u32string& s, size_t pos, Match& m) const {
        Ctx c{s, icase, std::vector<std::pair<size_t, size_t>>(static_cast<size_t>(n_groups) + 1,
                                                                {kNone, kNone})};
        // The whole pattern is group 0, walked alternative by alternative.
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

std::u32string PyRegex::Match::group(const std::u32string& text, size_t i) const {
    if (i >= groups.size() || groups[i].first == kNone) return {};
    return text.substr(groups[i].first, groups[i].second - groups[i].first);
}

PyRegex::PyRegex(const std::string& pattern_utf8, bool ignore_case)
    : impl_(new Impl()), pattern_(pattern_utf8) {
    Parser parser(pattern_utf8);
    impl_->alts = parser.parse(&impl_->n_groups);
    impl_->icase = ignore_case;
}
PyRegex::~PyRegex() = default;
PyRegex::PyRegex(PyRegex&&) noexcept = default;
PyRegex& PyRegex::operator=(PyRegex&&) noexcept = default;

size_t PyRegex::n_groups() const { return static_cast<size_t>(impl_->n_groups); }

bool PyRegex::match_at(const std::u32string& text, size_t pos, Match& m) const {
    return pos <= text.size() && impl_->match_at(text, pos, m);
}

bool PyRegex::search(const std::u32string& text, size_t pos, Match& m) const {
    for (size_t p = pos; p <= text.size(); ++p) {
        if (impl_->match_at(text, p, m)) return true;
    }
    return false;
}

std::u32string PyRegex::sub(const std::u32string& text, const Replacer& repl, size_t count) const {
    std::u32string out;
    size_t pos = 0, n = 0;
    Match m;
    while (pos <= text.size() && (count == 0 || n < count) && search(text, pos, m)) {
        out.append(text, pos, m.begin - pos);
        out += repl(text, m);
        ++n;
        if (m.end == m.begin) {
            // An empty match: keep the character it sat before and move past it.
            if (m.begin < text.size()) out += text[m.begin];
            pos = m.begin + 1;
        } else {
            pos = m.end;
        }
    }
    if (pos < text.size()) out.append(text, pos, std::u32string::npos);
    return out;
}

void PyRegex::check_template(const std::u32string& templ) const {
    for (size_t i = 0; i < templ.size(); ++i) {
        if (templ[i] != U'\\') continue;
        const bool group = i + 1 < templ.size() && templ[i + 1] >= U'1' && templ[i + 1] <= U'9' &&
                           static_cast<size_t>(templ[i + 1] - U'0') <= n_groups();
        if (!group) {
            throw LoadError("PyRegex: pattern '" + pattern_ + "': its replacement holds a backslash that is "
                             "not a group reference \\1..\\9 to one of its groups");
        }
        ++i;
    }
}

std::u32string PyRegex::sub(const std::u32string& text, const std::u32string& templ, size_t count) const {
    check_template(templ);
    return sub(text, [&templ](const std::u32string& s, const Match& m) {
        std::u32string r;
        for (size_t i = 0; i < templ.size(); ++i) {
            if (templ[i] == U'\\') {
                r += m.group(s, static_cast<size_t>(templ[++i] - U'0'));
            } else {
                r += templ[i];
            }
        }
        return r;
    }, count);
}

std::vector<std::u32string> PyRegex::findall(const std::u32string& text) const {
    std::vector<std::u32string> out;
    size_t pos = 0;
    Match m;
    while (pos <= text.size() && search(text, pos, m)) {
        out.push_back(text.substr(m.begin, m.end - m.begin));
        pos = m.end == m.begin ? m.begin + 1 : m.end;
    }
    return out;
}

} // namespace loom
