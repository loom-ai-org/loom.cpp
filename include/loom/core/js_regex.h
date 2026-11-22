#pragma once

// A backtracking matcher for the subset of ECMAScript's `RegExp` that a JavaScript text front end's rules
// are written in (kale-tts-1's, a port of dizione's), so that the rules can ship as the reference's own
// pattern strings (ADR-041) rather than be re-derived into C++ scanners. `PyRegex` is the same idea for
// Python's `re`; the two dialects differ where it matters to these rules, so they are two classes:
//
//   * `\w` `\d` `\b` are ASCII (ECMAScript's, `u` flag or not), `\s` is ECMAScript's white space and line
//     terminators, `.` is anything but `\n` `\r` U+2028 U+2029 (anything with the `s` flag);
//   * lookahead `(?=...)` `(?!...)` and lookbehind `(?<=...)` `(?<!...)`, of any length (a lookbehind holds
//     when its body matches some span ending exactly here -- ECMAScript matches it right to left, which
//     gives the same answer for a body without captures, and no rule here captures inside one);
//   * named groups `(?<name>...)`; `\p{L}` `\p{N}` `\p{Ll}` `\p{Lu}` `\p{Mn}` and their `\P` negations;
//     `\uXXXX`, `\u{X...}`, `\xXX`; greedy and lazy quantifiers; back references `\1`..`\9`, which match
//     the empty string when their group did not take part (ECMAScript, where Python fails);
//   * flags `i` (simple case folding: the engine's lower-case map, both sides), `s`, `u`, `g` (taken and
//     ignored: every entry point here scans the whole string or one position, as the caller asks).
//
// Text is UTF-32 throughout, so the `u` flag's code-point semantics are the only ones: a pattern written
// without `u` treats an astral character as two code units in ECMAScript, which no rule here can see.
// Anything outside the subset is a `LoadError` naming the construct, so a table from a newer reference
// fails at load rather than matching wrongly.

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace loom {

class JsRegex {
public:
    struct Match {
        size_t begin = 0, end = 0;
        // Group i (1-based; [0] is the whole match) as [begin, end), npos when it did not take part.
        std::vector<std::pair<size_t, size_t>> groups;
        bool has(size_t i) const;
        std::u32string group(const std::u32string& text, size_t i) const;
    };
    // `fn(text, match)` returns the replacement, and may set `*extra` to consume that many characters
    // after the match as well (dizione's `sub` helper; 0 otherwise).
    using Replacer = std::function<std::u32string(const std::u32string& text, const Match& m, size_t* extra)>;

    JsRegex(const std::string& pattern_utf8, const std::string& flags = "");
    ~JsRegex();
    JsRegex(JsRegex&&) noexcept;
    JsRegex& operator=(JsRegex&&) noexcept;

    // Anchored at `pos`.
    bool match_at(const std::u32string& text, size_t pos, Match& m) const;
    // The first match at or after `pos` (`RegExp.prototype.exec` with `lastIndex = pos`).
    bool search(const std::u32string& text, size_t pos, Match& m) const;
    // `String.prototype.replace` with a global regex and a function (an empty match keeps the character
    // after it and moves past it, as `lastIndex` does).
    std::u32string replace(const std::u32string& text, const Replacer& fn) const;
    // ... with a replacement template: `$1`..`$99`, `$<name>`, `$&`, `$$`.
    std::u32string replace(const std::u32string& text, const std::u32string& templ) const;
    // `String.prototype.match` with a global regex: every whole match.
    std::vector<std::u32string> match_all(const std::u32string& text) const;
    // `String.prototype.split(regex)`, captures spliced in after each piece (an unmatched one as empty).
    std::vector<std::u32string> split(const std::u32string& text) const;
    // `RegExp.prototype.test`.
    bool test(const std::u32string& text) const;

    size_t n_groups() const;
    // The index of a named group, or 0 when there is none of that name.
    size_t group_index(const std::string& name) const;
    const std::string& pattern() const { return pattern_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string pattern_;
};

} // namespace loom
