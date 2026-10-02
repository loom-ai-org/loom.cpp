// Tests loom::PyRegex (src/core/py_regex.cpp), the matcher Soprano's text front end runs its rules with.
//
// **The measured claim is that it IS Python's `re` on its subset**: every one of the 115 patterns in
// Soprano's `text_normalizer`, `text_splitter` and inflect's clean-up, `re.sub`-ed over random ASCII-
// heavy texts with each match rendered as its groups -- 92,000/92,000 identical over two seeds. What
// belongs here is each semantic the subset commits to, one case apiece, its expectation taken from
// Python 3.11's `re`; the subset's refusals; and the subset's documented gaps.

#include "test_util.h"

#include "loom/core/py_regex.h"
#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

#include <string>

namespace {

std::u32string u32(const std::string& s) {
    const auto cps = loom::utf8_decode(s);
    return std::u32string(cps.begin(), cps.end());
}
std::string utf8(const std::u32string& s) { return loom::utf8_encode(std::vector<char32_t>(s.begin(), s.end())); }

// `re.sub(pattern, lambda m: "<" + "|".join(groups) + ">", text)`.
std::string marked(const std::string& pattern, const std::string& text, bool icase = false) {
    loom::PyRegex re(pattern, icase);
    return utf8(re.sub(u32(text), [&](const std::u32string& s, const loom::PyRegex::Match& m) {
        std::u32string out = U"<";
        for (size_t g = 0; g <= re.n_groups(); ++g) {
            if (g) out += U"|";
            out += m.group(s, g);
        }
        return out + U">";
    }));
}

bool refuses(const std::string& pattern) {
    try {
        loom::PyRegex re(pattern);
    } catch (const loom::LoadError&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    // Leftmost-FIRST alternation, not longest: Python's, where POSIX would take "ab".
    LOOM_CHECK(marked("a|ab", "ab abc") == "<a>b <a>bc");
    // Greedy, backtracking until the rest matches; captures keep what the successful path gave them.
    LOOM_CHECK(marked("(\\d+)(\\d)", "12345") == "<12345|1234|5>");
    // `\s` is `str.isspace()`, which counts the information separators U+001C-U+001F.
    LOOM_CHECK(marked("\\s+", "a \x1c\x1d b") == "a< \x1c\x1d >b");
    // A back reference, quantified.
    LOOM_CHECK(marked("(\\w)\\1{2,}", "aaab bbbb cc") == "<aaa|a>b <bbbb|b> cc");
    // IGNORECASE, and `\b`.
    LOOM_CHECK(marked("\\bmr\\.", "Mr. MR. mr.x amr.", true) == "<Mr.> <MR.> <mr.>x amr.");
    // The tortoise date pattern backtracks out of its optional `(?:\d\d)?` when a '/' follows, and the
    // third group then takes the digit after it.
    LOOM_CHECK(marked("(^|[^/])(\\d\\d?[/-]\\d\\d?[/-]\\d\\d(?:\\d\\d)?)($|[^/])", "1/1/2025/ and 1/2/3") ==
               "<1/1/202||1/1/20|2>5/ and 1/2/3");
    // `,-.` in a class is a RANGE (comma to full stop), a leading or trailing '-' a literal.
    LOOM_CHECK(marked("[^A-Za-z !\\$%&'\\*\\+,-./0-9]", "a-b,c.d#e") == "a-b,c.d<#>e");
    LOOM_CHECK(marked("[-.\\s]", "a-b.c d") == "a<->b<.>c< >d");
    // Counted repetition, and Python 3.11's `{,m}`; a brace that is no quantifier is a literal.
    LOOM_CHECK(marked("\\d{3}", "1234567") == "<123><456>7");
    LOOM_CHECK(marked("x{2,3}", "xxxxxxx") == "<xxx><xxx>x");
    LOOM_CHECK(marked("a{,2}", "aaa{,}") == "<aa><a><>{<>,<>}<>");
    LOOM_CHECK(marked("a{x}", "a{x}") == "<a{x}>");
    // A group that did not take part renders as empty, as Python 3.5+ substitutes it.
    LOOM_CHECK(marked("(a)|b", "ab") == "<a|a><b|>");
    // `.*` runs to the end and backtracks to the last closer.
    LOOM_CHECK(marked("[\\(\\[\\{].*[\\)\\]\\}](.|$)", "x (a) b (c)") == "x <(a) b (c)|>");
    // `\Z` is the very end; `$` also matches before a final newline, and empty matches are substituted
    // between the characters they sit before.
    LOOM_CHECK(marked(", (\\S+)\\s+\\Z", "one, two ") == "one<, two |two>");
    LOOM_CHECK(marked("$", "ab\n") == "ab<>\n<>");
    // A quantified group, whose capture is its LAST iteration.
    LOOM_CHECK(marked("\\b([A-Z][a-z]*)+\\b", "LMDeploy TPUs a1B") == "<LMDeploy|Deploy> <TPUs|Us> a1B");

    // `re.sub` with a template: `\1` is the group, and nothing else may follow a backslash.
    {
        loom::PyRegex pounds("£([\\d\\,]*\\d+)");
        LOOM_CHECK(utf8(pounds.sub(u32("£20 and £1,5"), U"\\1 pounds")) == "20 pounds and 1,5 pounds");
        LOOM_CHECK_THROWS(pounds.check_template(U"\\n"), loom::LoadError);
        LOOM_CHECK_THROWS(pounds.check_template(U"\\2"), loom::LoadError);
    }
    // `re.match` is anchored where it is asked; `re.search` scans; `re.findall` returns whole matches.
    {
        loom::PyRegex empty_line("^[\\s\\.,;:!?]*$");
        loom::PyRegex::Match m;
        LOOM_CHECK(empty_line.match_at(U" ..!", 0, m));
        LOOM_CHECK(!empty_line.match_at(U" .a", 0, m));
        loom::PyRegex part("[A-Z][a-z]*");
        LOOM_CHECK((part.findall(U"LMDeploy") == std::vector<std::u32string>{U"L", U"M", U"Deploy"}));
        LOOM_CHECK(part.search(U"abC", 0, m) && m.begin == 2 && m.end == 3);
    }
    // A long run costs no stack per character: one quantified single-character atom is iterative.
    {
        loom::PyRegex spaces("\\s+");
        LOOM_CHECK(spaces.sub(u32("a" + std::string(200000, ' ') + "b"), U" ") == U"a b");
    }
    // Outside ASCII, `\w` is `\p{L}`/`\p{N}` (Python's `isalnum`); `\d` stays ASCII, the documented gap.
    LOOM_CHECK(marked("\\w+", "caf\xc3\xa9!") == "<caf\xc3\xa9>!");
    LOOM_CHECK(marked("\\d", "\xd9\xa3") == "\xd9\xa3");

    // Outside the subset, refused at compile rather than matched wrongly.
    LOOM_CHECK(refuses("a*?"));          // lazy
    LOOM_CHECK(refuses("a++"));          // possessive
    LOOM_CHECK(refuses("(?=a)"));        // lookahead
    LOOM_CHECK(refuses("(?i)a"));        // inline flags
    LOOM_CHECK(refuses("\\p{L}"));       // an escape Python's `re` does not have either
    LOOM_CHECK(refuses("(a"));
    LOOM_CHECK(refuses("a)"));
    LOOM_CHECK(refuses("[a"));
    LOOM_CHECK(refuses("\\2(a)"));       // a reference to a group not yet opened
    LOOM_CHECK(refuses("^*"));           // a quantified assertion
    LOOM_CHECK(!refuses("[]a]"));        // a ']' first in a class is a literal

    LOOM_TEST_REPORT_AND_RETURN();
}
