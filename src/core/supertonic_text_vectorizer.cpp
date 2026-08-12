#include "loom/core/supertonic_text_vectorizer.h"
#include "loom/core/gguf_model.h"
#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

#include <algorithm>

namespace loom {
namespace {

bool is_emoji(char32_t cp) {
    // Ranges copied verbatim from the real TextVectorizer._preprocess_text's own emoji_pattern.
    return (cp >= 0x1F600 && cp <= 0x1F64F) || // emoticons
           (cp >= 0x1F300 && cp <= 0x1F5FF) || // symbols & pictographs
           (cp >= 0x1F680 && cp <= 0x1F6FF) || // transport & map symbols
           (cp >= 0x1F700 && cp <= 0x1F77F) ||
           (cp >= 0x1F780 && cp <= 0x1F7FF) ||
           (cp >= 0x1F800 && cp <= 0x1F8FF) ||
           (cp >= 0x1F900 && cp <= 0x1F9FF) ||
           (cp >= 0x1FA00 && cp <= 0x1FA6F) ||
           (cp >= 0x1FA70 && cp <= 0x1FAFF) ||
           (cp >= 0x2600 && cp <= 0x26FF) ||
           (cp >= 0x2700 && cp <= 0x27BF) ||
           (cp >= 0x1F1E6 && cp <= 0x1F1FF); // regional indicators (flags)
}

// The real "[♥☆♡©\\]" removal regex -- 5 specific codepoints, dropped entirely.
bool is_removed_symbol(char32_t cp) {
    switch (cp) {
        case 0x2665: // ♥
        case 0x2606: // ☆
        case 0x2661: // ♡
        case 0x00A9: // ©
        case 0x005C: // backslash
            return true;
        default:
            return false;
    }
}

// The real `replacements` dict (single codepoint -> single ASCII char, see class doc comment for why
// applying these as one left-to-right pass over codepoints is behaviorally identical to Python's
// sequential whole-string .replace() loop: every key is a distinct single character, and none of the 4
// distinct replacement values ever collides with a not-yet-processed key).
bool try_single_char_replacement(char32_t cp, char32_t& out) {
    switch (cp) {
        case 0x2013: // – en dash
        case 0x2011: // ‑ non-breaking hyphen
        case 0x2014: // — em dash
            out = U'-';
            return true;
        case U'_':
        case U'[':
        case U']':
        case U'|':
        case U'/':
        case U'#':
        case 0x2192: // →
        case 0x2190: // ←
            out = U' ';
            return true;
        case 0x201C: // left double quote "
        case 0x201D: // right double quote "
            out = U'"';
            return true;
        case 0x2018: // left single quote '
        case 0x2019: // right single quote '
        case 0x00B4: // ´
        case U'`':
            out = U'\'';
            return true;
        default:
            return false;
    }
}

// The real trailing-terminal-punctuation check's character class (`[.!?;:,'"')\]}…。」』】〉》›»]`).
bool is_terminal_punct(char32_t cp) {
    switch (cp) {
        case U'.': case U'!': case U'?': case U';': case U':': case U',':
        case U'\'': case U'"': case U')': case U']': case U'}':
        case 0x2026: case 0x3002: case 0x300D: case 0x300F: case 0x3011:
        case 0x3009: case 0x300B: case 0x203A: case 0x00BB:
            return true;
        default:
            return false;
    }
}

// Non-overlapping, left-to-right literal substring replacement -- same semantics as Python's
// `str.replace()`/`re.sub()` on a literal (non-regex-metacharacter) pattern.
std::string replace_all(const std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    std::string out;
    size_t pos = 0;
    while (true) {
        const size_t next = s.find(from, pos);
        if (next == std::string::npos) {
            out.append(s, pos, std::string::npos);
            break;
        }
        out.append(s, pos, next - pos);
        out += to;
        pos = next + from.size();
    }
    return out;
}

} // namespace

std::unique_ptr<SupertonicTextVectorizer> SupertonicTextVectorizer::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.ggml.model")) {
        return nullptr;
    }
    if (model.kv_str("tokenizer.ggml.model") != "supertonic") {
        return nullptr;
    }

    auto vec = std::unique_ptr<SupertonicTextVectorizer>(new SupertonicTextVectorizer());
    vec->table_ = model.kv_arr_i32("tokenizer.ggml.supertonic.codepoint_to_id");
    vec->default_lang_ = model.has_kv("tokenizer.ggml.supertonic.default_lang")
                             ? model.kv_str("tokenizer.ggml.supertonic.default_lang")
                             : std::string("en");

    // Build the id -> codepoint inverse once at load, sized to the highest mapped id. First writer wins
    // on a collision and `invertible_` records that it happened, so `detokenize` can refuse rather than
    // return whichever codepoint happened to come first (see its declaration).
    int32_t max_id = -1;
    for (int32_t id : vec->table_) max_id = std::max(max_id, id);
    vec->inverse_.assign(static_cast<size_t>(max_id + 1), -1);
    for (size_t cp = 0; cp < vec->table_.size(); ++cp) {
        const int32_t id = vec->table_[cp];
        if (id < 0) continue;
        if (vec->inverse_[static_cast<size_t>(id)] >= 0) {
            vec->invertible_ = false;
            continue;
        }
        vec->inverse_[static_cast<size_t>(id)] = static_cast<int32_t>(cp);
    }
    return vec;
}

std::string SupertonicTextVectorizer::preprocess(const std::string& text, const std::string& lang) const {
    // Step 1: NFKD-approximate normalize (see class doc comment), then in one pass over codepoints:
    // strip emojis, apply the single-codepoint replacement table, drop the 5 removed symbols.
    const std::vector<char32_t> decomposed = utf8_decode(nfd_normalize(text));
    std::vector<char32_t> filtered;
    filtered.reserve(decomposed.size());
    for (char32_t cp : decomposed) {
        if (is_emoji(cp) || is_removed_symbol(cp)) continue;
        char32_t repl;
        filtered.push_back(try_single_char_replacement(cp, repl) ? repl : cp);
    }
    std::string s = utf8_encode(filtered);

    // Step 2: expression replacements (multi-character substrings, unlike step 1's per-codepoint table).
    s = replace_all(s, "@", " at ");
    s = replace_all(s, "e.g.,", "for example, ");
    s = replace_all(s, "i.e.,", "that is, ");

    // Step 3: remove a single space immediately before these punctuation characters.
    for (const char* punct : {" ,", " .", " !", " ?", " ;", " :", " '"}) {
        s = replace_all(s, punct, std::string(1, punct[1]));
    }

    // Step 4: collapse repeated identical quote characters down to one.
    while (s.find("\"\"") != std::string::npos) s = replace_all(s, "\"\"", "\"");
    while (s.find("''") != std::string::npos) s = replace_all(s, "''", "'");
    while (s.find("``") != std::string::npos) s = replace_all(s, "``", "`");

    // Step 5: collapse any run of ASCII whitespace to a single space, then strip leading/trailing.
    {
        std::string collapsed;
        collapsed.reserve(s.size());
        bool in_ws = false;
        for (char c : s) {
            const bool is_ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f');
            if (is_ws) {
                in_ws = true;
            } else {
                if (in_ws && !collapsed.empty()) collapsed += ' ';
                in_ws = false;
                collapsed += c;
            }
        }
        s = collapsed;
    }

    // Step 6: if the text doesn't already end with a terminal punctuation/quote/closing-bracket
    // codepoint, append a period.
    {
        const std::vector<char32_t> cps = utf8_decode(s);
        if (cps.empty() || !is_terminal_punct(cps.back())) {
            s += '.';
        }
    }

    // Step 7: wrap in <lang>...</lang> (no validation against AVAILABLE_LANGS -- the real Python class
    // keeps strictness low here too, see its own comment).
    return "<" + lang + ">" + s + "</" + lang + ">";
}

std::vector<int32_t> SupertonicTextVectorizer::tokenize(const std::string& text, const std::string& lang) const {
    const std::string preprocessed = preprocess(text, lang.empty() ? default_lang_ : lang);
    const std::vector<char32_t> cps = utf8_decode(preprocessed);

    std::vector<int32_t> ids;
    ids.reserve(cps.size());
    for (char32_t cp : cps) {
        if (cp < table_.size() && table_[static_cast<size_t>(cp)] >= 0) {
            ids.push_back(table_[static_cast<size_t>(cp)]);
        }
    }
    return ids;
}

std::string SupertonicTextVectorizer::detokenize(const std::vector<int32_t>& ids) const {
    if (!invertible_) {
        throw SchemaError("SupertonicTextVectorizer::detokenize: this file's codepoint_to_id table sends "
                           "two codepoints to the same id, so ids cannot be mapped back to text "
                           "unambiguously (the real unicode_indexer.json is injective; this one is not)");
    }
    std::vector<char32_t> cps;
    cps.reserve(ids.size());
    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= inverse_.size()) continue;
        const int32_t cp = inverse_[static_cast<size_t>(id)];
        if (cp >= 0) cps.push_back(static_cast<char32_t>(cp));
    }
    return utf8_encode(cps);
}

} // namespace loom
