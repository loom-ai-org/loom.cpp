#pragma once

// A backtracking matcher for the subset of Python's `re` that a text front end's rule tables are
// written in, so that the tables can ship as the reference's own pattern strings (ADR-041: the
// rules are data) instead of being re-derived by hand into C++, one bespoke scanner per pattern.
//
// It is Python's semantics, not ECMAScript's: leftmost-first alternation, greedy quantifiers that
// backtrack, captures that keep their last iteration, `re.sub`'s non-overlapping left-to-right scan.
// The subset, and nothing outside it compiles (a pattern using anything else is a `LoadError` naming
// the construct, so a table written for a newer reference fails at load rather than matching wrongly):
//
//   * literals, `.` (anything but `\n`), `^` and `\A` (start only, no MULTILINE), `$` (end, or before a
//     final `\n`), `\Z` (end), `\b` `\B`;
//   * `\d` `\w` `\s` and their negations, in and out of classes; `[...]` and `[^...]` with ranges and
//     escaped metacharacters;
//   * `(...)`, `(?:...)`, `|`, back references `\1`..`\9`;
//   * greedy `?` `*` `+` `{n}` `{n,}` `{n,m}` (no lazy forms);
//   * IGNORECASE, as ASCII case folding.
//
// The classes are Python's on ASCII: `\d` is 0-9, `\w` is `[A-Za-z0-9_]`, `\s` is `str.isspace()`
// (which includes `\x1c`-`\x1f`). Outside ASCII, `\s` is Python's Unicode spaces exactly, `\w` is
// `\p{L}` or `\p{N}` (Python's `str.isalnum()` up to a handful of codepoints) and `\d` matches nothing
// (Python's matches every Nd digit). Every caller today runs on text that is ASCII by its first class
// (Soprano's `clean_text` runs `unidecode` first), so neither gap is reached.

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace loom {

class PyRegex {
public:
    struct Match {
        size_t begin = 0, end = 0;
        // Group i (1-based; [0] is the whole match) as [begin, end), or npos when it did not take part.
        std::vector<std::pair<size_t, size_t>> groups;
        std::u32string group(const std::u32string& text, size_t i) const;
    };
    using Replacer = std::function<std::u32string(const std::u32string& text, const Match& m)>;

    // Throws `LoadError` for a malformed pattern or one outside the subset above.
    PyRegex(const std::string& pattern_utf8, bool ignore_case = false);
    ~PyRegex();
    PyRegex(PyRegex&&) noexcept;
    PyRegex& operator=(PyRegex&&) noexcept;

    // `re.match` at `pos`: anchored there, not searched forward.
    bool match_at(const std::u32string& text, size_t pos, Match& m) const;
    // `re.search` from `pos`.
    bool search(const std::u32string& text, size_t pos, Match& m) const;
    // `re.sub(pattern, repl, text, count)`, `count == 0` meaning every match.
    std::u32string sub(const std::u32string& text, const Replacer& repl, size_t count = 0) const;
    // `re.sub` with a template: `\1`..`\9` are groups (an unmatched group is empty, as Python 3.5+),
    // and the template may hold no other backslash -- checked by `check_template`.
    std::u32string sub(const std::u32string& text, const std::u32string& templ, size_t count = 0) const;
    // `re.findall` for a pattern with no groups: every whole match.
    std::vector<std::u32string> findall(const std::u32string& text) const;

    size_t n_groups() const;
    const std::string& pattern() const { return pattern_; }

    // Throws `LoadError` if `templ` holds a backslash other than `\1`..`\9` naming a group this pattern
    // has. Python would interpret `\n`, `\g<1>` and the rest; no table uses them, so none is accepted.
    void check_template(const std::u32string& templ) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string pattern_;
};

} // namespace loom
