#include "loom/core/bpe_vocab.h"
#include "loom/core/gguf_model.h"
#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

#include <cstdio>

#include <array>
#include <climits>
#include <unordered_map>

namespace loom {
namespace {

// GPT2's standard byte<->unicode-codepoint mapping (Radford et al.'s "byte-level BPE" trick, identical
// across every GPT2/RoBERTa/GPT-NeoX/Qwen tokenizer -- not model-specific, a fixed universal convention):
// printable Latin-1 bytes map to themselves; every other byte value (control chars, space, DEL, etc.)
// maps to a codepoint >= 256 so every byte has SOME visible, round-trippable representation as text.
std::array<char32_t, 256> compute_byte_encoder() {
    std::vector<int> bs;
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::array<bool, 256> in_bs{};
    for (int b : bs) in_bs[static_cast<size_t>(b)] = true;
    std::vector<char32_t> cs(bs.begin(), bs.end());
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (!in_bs[static_cast<size_t>(b)]) {
            bs.push_back(b);
            cs.push_back(static_cast<char32_t>(256 + n));
            ++n;
        }
    }
    std::array<char32_t, 256> table{};
    for (size_t i = 0; i < bs.size(); ++i) table[static_cast<size_t>(bs[i])] = cs[i];
    return table;
}

const std::array<char32_t, 256>& byte_encoder() {
    static const std::array<char32_t, 256> table = compute_byte_encoder();
    return table;
}

const std::unordered_map<char32_t, uint8_t>& byte_decoder() {
    static const std::unordered_map<char32_t, uint8_t> table = [] {
        std::unordered_map<char32_t, uint8_t> m;
        const auto& enc = byte_encoder();
        for (size_t b = 0; b < 256; ++b) m.emplace(enc[b], static_cast<uint8_t>(b));
        return m;
    }();
    return table;
}

bool is_ws(char32_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }

// [^\s\p{L}\p{N}] -- "punctuation-ish": not whitespace, not a letter, not a number. `include_marks`
// additionally excludes \p{M} (qwen35's own regex moves marks into the letter-run alternative instead,
// see match_letter_run's own `include_marks` param).
bool is_punct(char32_t c, bool include_marks = false) {
    return !is_ws(c) && !is_letter(c) && !is_number(c) && !(include_marks && is_mark(c));
}

std::string slice_utf8(const std::vector<char32_t>& cps, size_t begin, size_t end) {
    return utf8_encode(std::vector<char32_t>(cps.begin() + static_cast<long>(begin),
                                              cps.begin() + static_cast<long>(end)));
}

// (?i:'s|'t|'re|'ve|'m|'ll|'d) -- the fixed set of English contraction suffixes the real Qwen2/Qwen3/
// llama3-family tokenizer.json pretokenizer regex special-cases ahead of the general letter-run
// alternative (kQwenLlama3 shape -- ASCII-only case-insensitive match).
bool match_contraction(const std::vector<char32_t>& cps, size_t pos, size_t& end) {
    if (cps[pos] != U'\'') return false;
    static const char* const kSuffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    for (const char* suf : kSuffixes) {
        const size_t len = std::char_traits<char>::length(suf);
        if (pos + 1 + len > cps.size()) continue;
        bool match = true;
        for (size_t k = 0; k < len; ++k) {
            char32_t c = cps[pos + 1 + k];
            if (c >= U'A' && c <= U'Z') c = c - U'A' + U'a'; // ASCII-only lowercasing, matches (?i:) scope
            if (c != static_cast<char32_t>(suf[k])) { match = false; break; }
        }
        if (match) { end = pos + 1 + len; return true; }
    }
    return false;
}

// 's|'t|'re|'ve|'m|'ll|'d -- kGpt2Classic shape's contraction alternative: unlike kQwenLlama3's, this is
// case-SENSITIVE literal text (no (?i:) in the real GPT-2/starcoder-family regex), so "'S"/"'T"/etc. do
// NOT match here and fall through to the punctuation-run alternative instead.
bool match_contraction_cs(const std::vector<char32_t>& cps, size_t pos, size_t& end) {
    if (cps[pos] != U'\'') return false;
    static const char* const kSuffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    for (const char* suf : kSuffixes) {
        const size_t len = std::char_traits<char>::length(suf);
        if (pos + 1 + len > cps.size()) continue;
        bool match = true;
        for (size_t k = 0; k < len; ++k) {
            if (cps[pos + 1 + k] != static_cast<char32_t>(suf[k])) { match = false; break; }
        }
        if (match) { end = pos + 1 + len; return true; }
    }
    return false;
}

// [^\r\n\p{L}\p{N}]?\p{L}+ (kQwenLlama3 shape) -- `include_marks` widens the run (but NOT the leading
// prefix-exclusion class) to `[\p{L}\p{M}]+`, qwen35's own variant (marks attach to the letter run rather
// than falling into the punctuation-run alternative).
bool match_letter_run(const std::vector<char32_t>& cps, size_t pos, size_t& end, bool include_marks = false) {
    const size_t n = cps.size();
    auto is_run_char = [&](char32_t c) { return is_letter(c) || (include_marks && is_mark(c)); };
    size_t letters_begin = pos;
    if (!(cps[pos] == U'\r' || cps[pos] == U'\n' || is_run_char(cps[pos]) || is_number(cps[pos]))) {
        if (pos + 1 < n && is_run_char(cps[pos + 1])) letters_begin = pos + 1;
        else return false; // lone prefix char with no following run char -- not this alternative's match
    } else if (!is_run_char(cps[pos])) {
        return false;
    }
    size_t p = letters_begin;
    while (p < n && is_run_char(cps[p])) ++p;
    end = p;
    return true;
}

// ` ?[^\s\p{L}\p{N}]+[\r\n]*` (kQwenLlama3 shape) -- `include_marks` also excludes `\p{M}` from the run
// (qwen35's own variant, see match_letter_run).
bool match_punct_run(const std::vector<char32_t>& cps, size_t pos, size_t& end, bool include_marks = false) {
    const size_t n = cps.size();
    size_t punct_begin;
    if (cps[pos] == U' ' && pos + 1 < n && is_punct(cps[pos + 1], include_marks)) {
        punct_begin = pos + 1;
    } else if (is_punct(cps[pos], include_marks)) {
        punct_begin = pos;
    } else {
        return false;
    }
    size_t p = punct_begin;
    while (p < n && is_punct(cps[p], include_marks)) ++p;
    while (p < n && (cps[p] == U'\r' || cps[p] == U'\n')) ++p;
    end = p;
    return true;
}

// `\p{N}` (max_run==1, e.g. Qwen2/Qwen3's own regex -- no quantifier, so digits never group; also used,
// with no leading space, for kGpt2Classic's "isolated single digit" families -- see
// match_number_run_gpt2_unbounded for that shape's OTHER digit-run variant) or `\p{N}{1,3}` (max_run==3,
// LFM2's own regex, shared with llama.cpp's "llama3" pretokenizer type) -- greedily consumes up to
// `max_run` consecutive digit codepoints starting at `pos`.
bool match_number_run(const std::vector<char32_t>& cps, size_t pos, size_t max_run, size_t& end) {
    if (!is_number(cps[pos])) return false;
    const size_t n = cps.size();
    size_t p = pos;
    while (p < n && (p - pos) < max_run && is_number(cps[p])) ++p;
    end = p;
    return true;
}

// ` ?\p{L}+` (kGpt2Classic shape) -- unlike kQwenLlama3's match_letter_run, the optional prefix is
// SPECIFICALLY a single space (not "any non-letter/non-number/non-newline character").
bool match_letter_run_gpt2(const std::vector<char32_t>& cps, size_t pos, size_t& end) {
    const size_t n = cps.size();
    size_t letters_begin;
    if (cps[pos] == U' ' && pos + 1 < n && is_letter(cps[pos + 1])) letters_begin = pos + 1;
    else if (is_letter(cps[pos])) letters_begin = pos;
    else return false;
    size_t p = letters_begin;
    while (p < n && is_letter(cps[p])) ++p;
    end = p;
    return true;
}

// ` ?\p{N}+` (kGpt2Classic shape's GPT-2/MPT/OLMO/JAIS/TRILLION/GRANITE_DOCLING variant -- inline,
// unbounded digit run with an optional leading space, distinct from the STARCODER-family variant, which
// isolates single digits via a separate leading pass with NO leading space -- already exactly
// match_number_run(pos, 1, end)).
bool match_number_run_gpt2_unbounded(const std::vector<char32_t>& cps, size_t pos, size_t& end) {
    const size_t n = cps.size();
    size_t digits_begin;
    if (cps[pos] == U' ' && pos + 1 < n && is_number(cps[pos + 1])) digits_begin = pos + 1;
    else if (is_number(cps[pos])) digits_begin = pos;
    else return false;
    size_t p = digits_begin;
    while (p < n && is_number(cps[p])) ++p;
    end = p;
    return true;
}

// ` ?[^\s\p{L}\p{N}]+` (kGpt2Classic shape) -- same punctuation-ish complement as match_punct_run, but
// no trailing `[\r\n]*` absorption.
bool match_punct_run_gpt2(const std::vector<char32_t>& cps, size_t pos, size_t& end) {
    const size_t n = cps.size();
    size_t punct_begin;
    if (cps[pos] == U' ' && pos + 1 < n && is_punct(cps[pos + 1])) punct_begin = pos + 1;
    else if (is_punct(cps[pos])) punct_begin = pos;
    else return false;
    size_t p = punct_begin;
    while (p < n && is_punct(cps[p])) ++p;
    end = p;
    return true;
}

// " ?[^(\s|.,!?…。，、।۔،)]+" (kWhitespacePunctExclude shape -- poro-chat/bloom/gpt3-finnish/viking).
// The parenthesized contents are LITERAL characters inside the `[...]` character class, not regex
// metacharacters -- this excludes whitespace plus a fixed, small punctuation set (ASCII + CJK/Devanagari/
// Arabic sentence punctuation), not general \p{P}. `exclude_digits` (viking only, whose own regex_exprs
// runs an extra bare `\p{N}` pass ahead of this pattern, isolating single digits everywhere -- including
// mid-run) additionally stops/excludes the run at digit codepoints, so they're picked up separately by
// match_number_run(pos, 1, end) at the outer dispatch's higher priority instead.
bool is_viking_excluded_punct(char32_t c) {
    switch (c) {
        case U'(': case U')': case U'|': case U'.': case U',': case U'!': case U'?':
        case 0x2026: // … HORIZONTAL ELLIPSIS
        case 0x3002: // 。 IDEOGRAPHIC FULL STOP
        case 0xFF0C: // ， FULLWIDTH COMMA
        case 0x3001: // 、 IDEOGRAPHIC COMMA
        case 0x0964: // । DEVANAGARI DANDA
        case 0x06D4: // ۔ ARABIC FULL STOP
        case 0x060C: // ، ARABIC COMMA
            return true;
        default:
            return false;
    }
}

bool match_ws_excl_punct(const std::vector<char32_t>& cps, size_t pos, size_t& end, bool exclude_digits) {
    const size_t n = cps.size();
    auto is_run_char = [&](char32_t c) {
        return !is_ws(c) && !is_viking_excluded_punct(c) && !(exclude_digits && is_number(c));
    };
    size_t begin;
    if (cps[pos] == U' ' && pos + 1 < n && is_run_char(cps[pos + 1])) begin = pos + 1;
    else if (is_run_char(cps[pos])) begin = pos;
    else return false;
    size_t p = begin;
    while (p < n && is_run_char(cps[p])) ++p;
    end = p;
    return true;
}

// The single alternative above is not regex-complete over whitespace/excluded-punctuation codepoints --
// llama.cpp's own real implementation runs this pattern through std::regex (no custom scanner exists for
// this shape), and std::regex_iterator's contract makes the ENTIRE unmatched span between two consecutive
// matches its own chunk (not one codepoint at a time -- confirmed directly against
// unicode_regex_split_stl's `match.position() - start_idx` gap-emission). E.g. "abc,def" splits into
// ["abc", ",", "def"], and "abc  def" (two spaces) into ["abc", "  ", "def"]. `exclude_digits` (viking)
// stops the gap at a digit too, since that's picked off by the separate, higher-priority
// match_number_run(pos, 1, end) check in the outer dispatch instead.
bool match_ws_excl_punct_gap(const std::vector<char32_t>& cps, size_t pos, size_t& end, bool exclude_digits) {
    const size_t n = cps.size();
    auto is_gap_char = [&](char32_t c) {
        if (exclude_digits && is_number(c)) return false;
        return is_ws(c) || is_viking_excluded_punct(c);
    };
    if (!is_gap_char(cps[pos])) return false;
    size_t p = pos;
    while (p < n && is_gap_char(cps[p])) ++p;
    end = p;
    return true;
}

// `\s*[\r\n]+` -- see bpe_vocab.h's module-level design note (unicode.h doc comments) / BACKLOG.md for
// the backtracking trace justifying this closed-form reduction: succeeds iff the maximal whitespace run
// starting at `pos` contains at least one \r or \n, consuming up to and including the LAST one.
bool match_ws_then_newline(const std::vector<char32_t>& cps, size_t pos, size_t& end) {
    if (!is_ws(cps[pos])) return false;
    const size_t n = cps.size();
    size_t run_end = pos;
    while (run_end < n && is_ws(cps[run_end])) ++run_end;
    for (size_t k = run_end; k > pos; --k) {
        if (cps[k - 1] == U'\r' || cps[k - 1] == U'\n') {
            end = k;
            return true;
        }
    }
    return false;
}

// `\s+(?!\S)` -- shared by kQwenLlama3 (reached once match_ws_then_newline has already failed, so the
// run at `pos` is guaranteed newline-free there) and kGpt2Classic (reached directly, no newline-splitting
// alternative exists in that shape). Matches the whole run if it reaches end-of-string; otherwise the
// lookahead forces giving back exactly the run's last character (still whitespace, so `(?!\S)` is
// satisfied) -- a lone whitespace char immediately followed by non-whitespace fails here and falls
// through to match_ws_fallback instead, same as llama.cpp's own custom scanners for both shapes do.
bool match_ws_not_followed_by_nonspace(const std::vector<char32_t>& cps, size_t pos, size_t& end) {
    if (!is_ws(cps[pos])) return false;
    const size_t n = cps.size();
    size_t run_end = pos;
    while (run_end < n && is_ws(cps[run_end])) ++run_end;
    if (run_end == n) { end = run_end; return true; }
    if (run_end - pos < 2) return false; // single char, followed by non-whitespace -- can't satisfy \s+ AND the lookahead
    end = run_end - 1;
    return true;
}

// `\s+` -- unconditional final fallback.
bool match_ws_fallback(const std::vector<char32_t>& cps, size_t pos, size_t& end) {
    if (!is_ws(cps[pos])) return false;
    const size_t n = cps.size();
    size_t run_end = pos;
    while (run_end < n && is_ws(cps[run_end])) ++run_end;
    end = run_end;
    return true;
}

struct PreSpec {
    BpeShape shape;
    size_t max_number_run; // meaning is per-shape -- see BpeShape's own doc comment
    bool include_marks = false;
};

// `tokenizer.ggml.pre` name -> {shape, params}, verified directly against llama.cpp's real
// `llm_tokenizer_bpe` constructor switch (src/llama-vocab.cpp) and its string->enum table
// (llama_vocab::impl::load) -- not guessed/recalled, every entry below was cross-checked against that
// source. Names present in llama.cpp but absent here are real pretokenizer families this project doesn't
// implement yet (CJK-script splitters, case-transition/camelCase shapes, cascading-whitespace shapes, and
// the "byte_encode=false" SPM-style-BPE family, which all need more than a new regex -- they need a
// different symbol-initialization step in BpeVocab::encode() itself) -- BpeVocab::load() throws a named
// LoadError for any of those rather than silently mis-tokenizing.
const std::unordered_map<std::string, PreSpec>& pre_spec_table() {
    static const std::unordered_map<std::string, PreSpec> table = {
        // kQwenLlama3, single-digit \p{N} (STABLELM2/QWEN2/HUNYUAN/SOLAR_OPEN/GROK_2 share one
        // regex_exprs array; deepseek-r1-qwen/kormo/f2llmv2/megrez are QWEN2 string aliases).
        // SentencePiece-style BPE with byte fallback -- Gemma 3 tokenizes identically to llama.cpp's
        // `granite-embed-multi-311m` (they share a chkhsh, which is computed over real tokenizer
        // output, so the collision means the same behaviour rather than a coincidence of names).
        {"granite-embed-multi-311m", {BpeShape::kSpmByteFallback, 0}},
        {"granite-embed-multi-97m", {BpeShape::kSpmByteFallback, 0}},
        {"qwen2", {BpeShape::kQwenLlama3, 1}},
        {"deepseek-r1-qwen", {BpeShape::kQwenLlama3, 1}},
        {"kormo", {BpeShape::kQwenLlama3, 1}},
        {"f2llmv2", {BpeShape::kQwenLlama3, 1}},
        {"megrez", {BpeShape::kQwenLlama3, 1}},
        {"stablelm2", {BpeShape::kQwenLlama3, 1}},
        {"hunyuan", {BpeShape::kQwenLlama3, 1}},
        {"solar-open", {BpeShape::kQwenLlama3, 1}},
        {"grok-2", {BpeShape::kQwenLlama3, 1}},
        // kQwenLlama3 + \p{M} attaches to the letter run instead of the punct run (QWEN35's own variant).
        {"qwen35", {BpeShape::kQwenLlama3, 1, /*include_marks=*/true}},
        // kQwenLlama3, grouped-digit \p{N}{1,3} (LLAMA3/DBRX/SMAUG/CHATGLM4 share one regex_exprs array;
        // llama-v3/llama-bpe/falcon3/falcon-h1/pixtral/midm-2.0/lfm2/jina-v5-nano are LLAMA3 string
        // aliases; chatglm-bpe is a CHATGLM4 string alias).
        {"llama3", {BpeShape::kQwenLlama3, 3}},
        {"llama-v3", {BpeShape::kQwenLlama3, 3}},
        {"llama-bpe", {BpeShape::kQwenLlama3, 3}},
        {"falcon3", {BpeShape::kQwenLlama3, 3}},
        {"falcon-h1", {BpeShape::kQwenLlama3, 3}},
        {"pixtral", {BpeShape::kQwenLlama3, 3}},
        {"midm-2.0", {BpeShape::kQwenLlama3, 3}},
        {"lfm2", {BpeShape::kQwenLlama3, 3}},
        {"jina-v5-nano", {BpeShape::kQwenLlama3, 3}},
        {"dbrx", {BpeShape::kQwenLlama3, 3}},
        {"smaug-bpe", {BpeShape::kQwenLlama3, 3}},
        {"glm4", {BpeShape::kQwenLlama3, 3}},
        {"chatglm-bpe", {BpeShape::kQwenLlama3, 3}},
        // kGpt2Classic, unbounded \p{N}+ (GPT2/MPT/OLMO/JAIS/TRILLION/GRANITE_DOCLING share one
        // regex_exprs array; phi-2/gigachat/jina-v2-es/jina-v2-de/a.x-4.0/mellum/modern-bert are GPT2
        // string aliases, as is exaone4; jina-v1-en/jina-v2-code/roberta-bpe are also GPT2 string aliases
        // that additionally default add_sep=true in llama.cpp -- loom reads add_sep_token from its own
        // GGUF KV instead of hardcoding per pre-type, same convention as add_bos_token).
        {"gpt-2", {BpeShape::kGpt2Classic, 0}},
        {"phi-2", {BpeShape::kGpt2Classic, 0}},
        {"jina-v2-es", {BpeShape::kGpt2Classic, 0}},
        {"jina-v2-de", {BpeShape::kGpt2Classic, 0}},
        {"gigachat", {BpeShape::kGpt2Classic, 0}},
        {"a.x-4.0", {BpeShape::kGpt2Classic, 0}},
        {"mellum", {BpeShape::kGpt2Classic, 0}},
        {"modern-bert", {BpeShape::kGpt2Classic, 0}},
        {"exaone4", {BpeShape::kGpt2Classic, 0}},
        {"mpt", {BpeShape::kGpt2Classic, 0}},
        {"olmo", {BpeShape::kGpt2Classic, 0}},
        {"jais", {BpeShape::kGpt2Classic, 0}},
        {"trillion", {BpeShape::kGpt2Classic, 0}},
        {"granite-docling", {BpeShape::kGpt2Classic, 0}},
        {"roberta-bpe", {BpeShape::kGpt2Classic, 0}},
        {"jina-v1-en", {BpeShape::kGpt2Classic, 0}},
        {"jina-v2-code", {BpeShape::kGpt2Classic, 0}},
        // kGpt2Classic, single-digit (STARCODER/REFACT/COMMAND_R/SMOLLM/CODESHELL/EXAONE/MINERVA/
        // MELLUM2 share one regex_exprs array: a separate bare `\p{N}` pass ahead of the main pattern,
        // which isolates every digit to length 1 -- equivalent to max_number_run=1).
        {"starcoder", {BpeShape::kGpt2Classic, 1}},
        {"refact", {BpeShape::kGpt2Classic, 1}},
        {"command-r", {BpeShape::kGpt2Classic, 1}},
        {"smollm", {BpeShape::kGpt2Classic, 1}},
        {"codeshell", {BpeShape::kGpt2Classic, 1}},
        {"exaone", {BpeShape::kGpt2Classic, 1}},
        {"minerva-7b", {BpeShape::kGpt2Classic, 1}},
        {"mellum2", {BpeShape::kGpt2Classic, 1}},
        // kWhitespacePunctExclude (PORO/BLOOM/GPT3_FINNISH share one single-alternative regex_exprs
        // array; VIKING's own array adds a second `\p{N}` pass isolating single digits).
        {"poro-chat", {BpeShape::kWhitespacePunctExclude, 0}},
        {"bloom", {BpeShape::kWhitespacePunctExclude, 0}},
        {"gpt3-finnish", {BpeShape::kWhitespacePunctExclude, 0}},
        {"viking", {BpeShape::kWhitespacePunctExclude, 1}},
    };
    return table;
}

} // namespace

std::unique_ptr<BpeVocab> BpeVocab::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.ggml.model")) {
        return nullptr;
    }
    const std::string model_type = model.kv_str("tokenizer.ggml.model");
    if (model_type != "gpt2") {
        return nullptr; // not this vocab type -- caller should try loom::Vocab (SentencePiece) instead
    }

    auto vocab = std::unique_ptr<BpeVocab>(new BpeVocab());
    const std::string pre_type = model.has_kv("tokenizer.ggml.pre") ? model.kv_str("tokenizer.ggml.pre") : "qwen2";
    const auto& table = pre_spec_table();
    const auto spec_it = table.find(pre_type);
    if (spec_it == table.end()) {
        throw LoadError("BpeVocab::load: unimplemented pretokenizer family '" + pre_type + "' -- either "
                         "pass a supported --tokenizer-pre value, or extend bpe_vocab.cpp's "
                         "pre_spec_table() for this family (see EXPORT-BACKLOG.md item 4)");
    }
    vocab->shape_ = spec_it->second.shape;
    vocab->max_number_run_ = spec_it->second.max_number_run;
    vocab->include_marks_ = spec_it->second.include_marks;
    vocab->add_bos_token_ = model.kv_bool("tokenizer.ggml.add_bos_token", false);
    vocab->add_sep_token_ = model.kv_bool("tokenizer.ggml.add_sep_token", false);
    vocab->tokens_ = model.kv_arr_str("tokenizer.ggml.tokens");
    vocab->token_to_id_.reserve(vocab->tokens_.size());
    for (size_t i = 0; i < vocab->tokens_.size(); ++i) {
        vocab->token_to_id_.emplace(vocab->tokens_[i], static_cast<int32_t>(i));
    }

    const std::vector<std::string> merges = model.kv_arr_str("tokenizer.ggml.merges");
    vocab->merge_rank_.reserve(merges.size());
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const std::string& pair = merges[rank];
        const size_t sep = pair.find(' ');
        if (sep == std::string::npos) {
            throw LoadError("BpeVocab::load: malformed tokenizer.ggml.merges entry '" + pair + "' (expected \"a b\")");
        }
        vocab->merge_rank_.emplace(pair.substr(0, sep) + '\x01' + pair.substr(sep + 1), static_cast<int32_t>(rank));
    }

    vocab->bos_id_ = model.kv_i32("tokenizer.ggml.bos_token_id", -1);
    vocab->eos_id_ = model.kv_i32("tokenizer.ggml.eos_token_id", -1);
    vocab->sep_id_ = model.kv_i32("tokenizer.ggml.seperator_token_id", -1); // llama.cpp's own (misspelled) KV name, kept verbatim
    return vocab;
}

namespace {

// The SPM family's whole normalizer: every space becomes U+2581. No dummy prefix is added -- HF gives
// `"Hello world"` -> `['Hello', '\u2581world']`, with the first word bare -- so this really is a
// character substitution and nothing more.
std::string spm_normalize(const std::string& text) {
    static const char* kMarker = "\xe2\x96\x81";
    std::string out;
    out.reserve(text.size());
    for (char c : text) {
        if (c == ' ') out += kMarker;
        else out.push_back(c);
    }
    return out;
}

} // namespace

std::vector<std::string> BpeVocab::pretokenize(const std::string& nfc_text) const {
    // The SPM-style family has no pretokenizer: merges run over the whole normalized string, so the
    // one "chunk" is all of it. Returning early keeps the scanner below about regex shapes only.
    if (shape_ == BpeShape::kSpmByteFallback) {
        return {nfc_text};
    }
    const std::vector<char32_t> cps = utf8_decode(nfc_text);
    std::vector<std::string> chunks;
    size_t pos = 0;
    const size_t n = cps.size();
    while (pos < n) {
        size_t end;
        bool matched;
        switch (shape_) {
            case BpeShape::kQwenLlama3:
                matched = match_contraction(cps, pos, end) ||
                          match_letter_run(cps, pos, end, include_marks_) ||
                          match_number_run(cps, pos, max_number_run_, end) ||
                          match_punct_run(cps, pos, end, include_marks_) ||
                          match_ws_then_newline(cps, pos, end) ||
                          match_ws_not_followed_by_nonspace(cps, pos, end) ||
                          match_ws_fallback(cps, pos, end);
                break;
            case BpeShape::kGpt2Classic:
                matched = match_contraction_cs(cps, pos, end) ||
                          match_letter_run_gpt2(cps, pos, end) ||
                          (max_number_run_ == 0 ? match_number_run_gpt2_unbounded(cps, pos, end)
                                                 : match_number_run(cps, pos, max_number_run_, end)) ||
                          match_punct_run_gpt2(cps, pos, end) ||
                          match_ws_not_followed_by_nonspace(cps, pos, end) ||
                          match_ws_fallback(cps, pos, end);
                break;
            case BpeShape::kWhitespacePunctExclude:
                matched = (max_number_run_ >= 1 && match_number_run(cps, pos, 1, end)) ||
                          match_ws_excl_punct(cps, pos, end, /*exclude_digits=*/max_number_run_ >= 1) ||
                          match_ws_excl_punct_gap(cps, pos, end, /*exclude_digits=*/max_number_run_ >= 1);
                break;
            default:
                matched = false;
                break;
        }
        if (matched) {
            chunks.push_back(slice_utf8(cps, pos, end));
            pos = end;
        } else {
            // Every codepoint is whitespace, a letter, a number, or "other" (punct) -- the alternatives
            // for this shape are exhaustive over those classes, so this should be unreachable; fail closed
            // rather than infinite-loop if some future Unicode edge case proves otherwise.
            throw LoadError("BpeVocab::pretokenize: no pretokenizer alternative matched at codepoint U+" +
                             std::to_string(static_cast<uint32_t>(cps[pos])));
        }
    }
    return chunks;
}

void BpeVocab::bpe_merge(std::vector<std::string>& pieces) const {
    while (pieces.size() >= 2) {
        int best_rank = INT_MAX;
        size_t best_idx = SIZE_MAX;
        for (size_t i = 0; i + 1 < pieces.size(); ++i) {
            const auto it = merge_rank_.find(pieces[i] + '\x01' + pieces[i + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        if (best_idx == SIZE_MAX) break;
        pieces[best_idx] += pieces[best_idx + 1];
        pieces.erase(pieces.begin() + static_cast<long>(best_idx) + 1);
    }
}

// U+2581 LOWER ONE EIGHTH BLOCK, SentencePiece's space marker.
static const char* kSpmSpace = "\xe2\x96\x81";

std::vector<int32_t> BpeVocab::encode(const std::string& text) const {
    // NFC is deliberately skipped for the SPM family: its HF normalizer is a bare
    // Replace(" ", "\u2581") and nothing else, so composing decomposed sequences here would tokenize
    // differently from the reference model on exactly the inputs where it matters.
    const std::string normalized = shape_ == BpeShape::kSpmByteFallback ? spm_normalize(text)
                                                                        : nfc_normalize(text);
    const std::vector<std::string> chunks = pretokenize(normalized);
    const auto& enc = byte_encoder();

    std::vector<int32_t> ids;
    if (add_bos_token_ && bos_id_ >= 0) {
        ids.push_back(bos_id_);
    }
    for (const std::string& chunk : chunks) {
        std::vector<std::string> pieces;
        if (shape_ == BpeShape::kSpmByteFallback) {
            // Initial symbols are CHARACTERS, not byte-mapped stand-ins -- this vocabulary stores
            // literal UTF-8.
            for (char32_t cp : utf8_decode(chunk)) pieces.push_back(utf8_encode({cp}));
        } else {
            pieces.reserve(chunk.size());
            for (unsigned char b : chunk) pieces.push_back(utf8_encode({enc[b]}));
        }

        bpe_merge(pieces);

        for (const std::string& p : pieces) {
            const auto it = token_to_id_.find(p);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
                continue;
            }
            if (shape_ == BpeShape::kSpmByteFallback) {
                // Byte fallback: a character the vocabulary has no entry for becomes its raw UTF-8
                // bytes as `<0xNN>` tokens. A byte-level vocab never needs this (every byte maps to a
                // base entry); this one does, and without it any unseen character would throw.
                for (unsigned char b : p) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
                    const auto byte_it = token_to_id_.find(buf);
                    if (byte_it == token_to_id_.end()) {
                        throw LoadError(std::string("BpeVocab::encode: piece '") + p + "' is not in the "
                                         "vocabulary and its byte fallback '" + buf + "' is missing too");
                    }
                    ids.push_back(byte_it->second);
                }
                continue;
            }
            throw LoadError("BpeVocab::encode: merged piece '" + p + "' is not in the vocabulary "
                             "(every single byte-mapped character should be a base vocab entry)");
        }
    }
    if (add_sep_token_ && sep_id_ >= 0) {
        ids.push_back(sep_id_);
    }
    return ids;
}

std::string BpeVocab::decode(const std::vector<int32_t>& ids) const {
    if (shape_ == BpeShape::kSpmByteFallback) {
        // The mirror of encode: `<0xNN>` back to a raw byte, U+2581 back to a space, everything else
        // verbatim -- the vocabulary already holds literal UTF-8, so there is no byte map to undo.
        std::string out;
        for (int32_t id : ids) {
            const std::string& piece = id_to_piece(id);
            unsigned int byte_value = 0;
            if (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 && piece[5] == '>' &&
                std::sscanf(piece.c_str(), "<0x%02X>", &byte_value) == 1) {
                out.push_back(static_cast<char>(byte_value));
                continue;
            }
            for (size_t i = 0; i < piece.size();) {
                if (piece.compare(i, 3, kSpmSpace) == 0) {
                    out.push_back(' ');
                    i += 3;
                } else {
                    out.push_back(piece[i]);
                    ++i;
                }
            }
        }
        return out;
    }
    const auto& dec = byte_decoder();
    std::string raw;
    for (int32_t id : ids) {
        const std::string& piece = id_to_piece(id);
        for (char32_t cp : utf8_decode(piece)) {
            const auto it = dec.find(cp);
            if (it != dec.end()) raw.push_back(static_cast<char>(it->second));
        }
    }
    return raw;
}

const std::string& BpeVocab::id_to_piece(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) {
        throw LoadError("BpeVocab::id_to_piece: id " + std::to_string(id) + " out of range");
    }
    return tokens_[static_cast<size_t>(id)];
}

int32_t BpeVocab::piece_to_id(const std::string& piece) const {
    // -1 rather than a throw: "does this vocabulary have this token" is a legitimate question with a
    // legitimate negative answer -- an English-only Whisper genuinely has no `<|de|>`, and the caller's
    // response to that is a message about the checkpoint, not an exception.
    const auto it = token_to_id_.find(piece);
    return it == token_to_id_.end() ? -1 : it->second;
}

} // namespace loom
