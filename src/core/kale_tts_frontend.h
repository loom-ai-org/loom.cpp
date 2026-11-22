#pragma once

// kale-tts-1-tiny's text front end (upstream's `frontend/`, JavaScript), the engine half of ADR-142: the shape of
// every step, with each step's tables and patterns read from the file (`tokenizer.ggml.kale.*`, written by
// loom-exporter's `kale_tts_tokenizer_export.py`). Internal to `KaleTtsVocab`; one class per upstream module:
//
//   Normalizer  dizione's `spoken.js` (`Normalizer.rewrite`, `spokenText`) and kaleTts.js's `normalizer` wrapper
//   G2p         kaleTts.js's `G2p` and `G2pModel`, dizione's `homographs.js`, enGb.js's `BritishG2p`/`toBritish`
//   FrontEnd    one accent's normalizer + G2p, `sentences` and the terms table (kale-tts.mjs `buildFrontEnd`)
//
// Strings are UTF-32. Upstream's are UTF-16, which differs only on astral characters, and every rule that could
// meet one either carries the `u` flag or only tests ASCII.

#include "loom/core/js_regex.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace loom {
class GgufModel;
namespace kale {

using u32 = std::u32string;

u32 U(const std::string& utf8);
std::string S(const u32& text);

// Text-level helpers every module shares: the rules by name, and the case and NFKC tables.
struct Shared {
    std::unordered_map<std::string, JsRegex> rules;
    std::unordered_map<char32_t, u32> upper, lower, nfkc;

    const JsRegex& rule(const char* name) const;
    u32 to_upper(const u32& s) const;
    u32 to_lower(const u32& s) const;
    u32 nfkc_normalize(const u32& s) const;   // per-codepoint NFKC, then the engine's canonical composition
    bool is_upper(const u32& s) const;        // dizione's isUpper: s != lower(s) && s == upper(s)
};

// dizione's tables (frontend/dizione/tables.js) and homograph data, parsed once.
struct Tables {
    std::map<u32, std::pair<u32, u32>> abbreviations;               // key -> (when, said)
    std::map<u32, std::pair<u32, u32>> units, exact_units;          // spelling -> (one, many)
    std::map<u32, u32> chat, endings, symbol_words;
    std::vector<std::tuple<u32, u32, u32>> homograph_rules;         // word, sense, conditions
    std::string homographs_json;                                    // homographs.data.js, parsed by G2p
    std::unordered_map<u32, int32_t> vocab;                         // the model's symbol -> id
};

class Normalizer {
public:
    Normalizer(const Shared& sh, const Tables& t, std::function<bool(const u32&)> is_word, bool british);
    // kaleTts.js `normalizer(lex, Rules)(text)`: size suffixes, a final "Dr.", dizione's rewrite with Kale's
    // letter spelling, dotted names.
    u32 operator()(const u32& text) const;

private:
    struct M;
    u32 rewrite(u32 text) const;
    u32 figures(u32 text) const;
    u32 signs(u32 text) const;
    u32 spoken_text(const u32& text) const;
    u32 spell(const u32& letters) const;
    // The methods, named as upstream's.
    u32 amount(const u32& symbol, const u32& whole, const std::optional<u32>& fraction, const std::optional<u32>& scale) const;
    u32 money(const M& m) const;
    u32 half_day(const M& m, const std::optional<u32>& dotted, const std::optional<u32>& plain) const;
    u32 time(const M& m) const;
    u32 dot_times(const M& m) const;
    u32 ratio(const M& m) const;
    std::optional<u32> date(long long month, long long day, const std::optional<u32>& year_text, bool the) const;
    std::optional<u32> date(long long month, long long day, const std::optional<u32>& year_text) const;
    u32 slash_date(const M& m) const;
    u32 dot_date(const M& m) const;
    u32 day_month(const M& m) const;
    u32 days_of_month(const M& m) const;
    u32 day_range(const M& m) const;
    std::optional<u32> short_date(const M& m) const;
    u32 fraction(const M& m) const;
    u32 mixed_fraction(const M& m) const;
    u32 vulgar(const M& m) const;
    u32 phone(const M& m) const;
    u32 measure(const M& m, size_t* extra) const;
    u32 square(const M& m) const;
    bool counted(const u32& text, size_t end) const;
    u32 number(const M& m) const;
    bool reads_as_word(const u32& letters) const;
    u32 alnum(const u32& group, const char* zero = "zero") const;
    u32 digit_groups(const std::vector<u32>& groups, const char* zero = "zero") const;
    std::optional<u32> code_words(const std::vector<u32>& groups, const M& m) const;
    u32 code(const M& m) const;
    u32 grouped(const M& m) const;
    u32 glued(const M& m) const;
    u32 address_part(const u32& part) const;
    u32 spoken_address(const u32& core) const;
    u32 address(const M& m) const;
    u32 chat_form(const M& m) const;
    u32 dotted(const M& m) const;
    u32 abbreviation(const M& m) const;
    std::optional<std::string> identifier(const u32& text, size_t position) const;
    std::vector<u32> before(const u32& text, size_t position, size_t count = 4) const;

    const Shared& sh_;
    const Tables& t_;
    std::function<bool(const u32&)> is_word_;
    bool british_;
    // Set by `figures` for the date rules after it, as upstream's instance fields.
    mutable bool day_first_ = false, dated_ = false;
};

// `G2pModel`: the guesser for a word the lexicon lacks, a GRU encoder-decoder with attention.
class G2pModel {
public:
    explicit G2pModel(const std::vector<uint8_t>& bytes);
    std::optional<u32> phonemes(const u32& word) const;

private:
    struct Gru { std::vector<double> w_ih, w_hh, b_ih, b_hh; };
    std::vector<double> gru_cell(const Gru& g, const std::vector<double>& x, const std::vector<double>& h, int size) const;
    int e_ = 0, h_ = 0, d_ = 0, max_out_ = 0;
    std::vector<int32_t> outputs_;
    std::unordered_map<char32_t, int> in_index_;
    std::vector<double> emb_in_, init_w_, init_b_, emb_out_, att_w_, comb_w_, comb_b_, out_w_, out_b_;
    Gru enc_fwd_, enc_bwd_, dec_;
};

class Homographs;

class G2p {
public:
    G2p(const Shared& sh, const Tables& t, std::unordered_map<u32, u32> lexicon, std::unique_ptr<G2pModel> model,
        bool british, const Homographs& homographs);
    u32 phonemes(const u32& sentence) const;
    bool has(const u32& word) const { return lex_.count(word) != 0; }

private:
    std::optional<u32> lex(const u32& key) const;
    u32 spell(const u32& word) const;
    std::optional<u32> homograph(const u32& token, const std::vector<u32>& tokens, size_t index) const;
    std::optional<u32> word(const u32& token, const std::optional<u32>& next) const;
    std::optional<u32> lookup(const u32& low, const std::optional<u32>& original) const;
    std::optional<u32> joined(const u32& low, const u32& said) const;
    std::optional<u32> fallback(const u32& low) const;
    u32 letters(const u32& low, const u32& ipa) const;
    u32 chunks(const u32& w) const;
    u32 british(const u32& ipa, const u32& key) const;   // enGb.js toBritish

    const Shared& sh_;
    const Tables& t_;
    std::unordered_map<u32, u32> lex_;
    std::unique_ptr<G2pModel> model_;
    bool british_;
    const Homographs& homographs_;
    mutable std::unordered_map<u32, std::optional<u32>> learnt_;
};

// dizione's homographs.js: hand rules, then a small linear classifier per word.
class Homographs {
public:
    Homographs(const Shared& sh, const Tables& t);
    ~Homographs();
    bool covers(const u32& word) const;
    struct Reading { u32 sense; std::vector<u32> senses; bool plural; };
    Reading pronounce(const u32& word, const std::vector<u32>& tokens, size_t index) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class FrontEnd {
public:
    FrontEnd(const Shared& sh, const Tables& t, const GgufModel& model, const std::string& accent, const Homographs& h,
             const std::vector<std::pair<u32, u32>>& terms);
    std::vector<u32> sentences(const u32& text) const;
    u32 ipa(const u32& sentence) const { return g2p_->phonemes(sentence); }
    bool british() const { return british_; }

private:
    u32 apply_terms(const u32& text) const;
    u32 normalize(const u32& text) const;

    const Shared& sh_;
    bool british_;
    std::unique_ptr<G2p> g2p_;
    std::unique_ptr<Normalizer> norm_;
    std::unordered_map<u32, u32> terms_;
};

} // namespace kale
} // namespace loom
