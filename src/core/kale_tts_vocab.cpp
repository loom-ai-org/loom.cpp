#include "loom/core/kale_tts_vocab.h"

#include "kale_tts_frontend.h"

#include "loom/core/bpe_vocab.h"
#include "loom/core/gguf_model.h"
#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

#include <nlohmann/json.hpp>

#include <algorithm>

namespace loom {
namespace kale {

const std::string kPrefix = "tokenizer.ggml.kale.";

u32 U(const std::string& utf8) {
    const auto cps = utf8_decode(utf8);
    return u32(cps.begin(), cps.end());
}
std::string S(const u32& text) { return utf8_encode(std::vector<char32_t>(text.begin(), text.end())); }

const JsRegex& Shared::rule(const char* name) const {
    const auto it = rules.find(name);
    if (it == rules.end()) {
        throw LoadError(std::string("KaleTtsVocab: the file has no rule '") + name + "' (tokenizer.ggml.kale.rule_names); "
                        "it was written by an older kale_tts_tokenizer_export");
    }
    return it->second;
}

static u32 map_each(const std::unordered_map<char32_t, u32>& table, const u32& s, char32_t (*fallback)(char32_t)) {
    u32 out;
    for (char32_t c : s) {
        const auto it = table.find(c);
        if (it != table.end()) out += it->second;
        else out += fallback ? fallback(c) : c;
    }
    return out;
}
static char32_t same(char32_t c) { return c; }

u32 Shared::to_upper(const u32& s) const { return map_each(upper, s, same); }
u32 Shared::to_lower(const u32& s) const { return map_each(lower, s, same); }
u32 Shared::nfkc_normalize(const u32& s) const { return U(nfc_normalize(S(map_each(nfkc, s, same)))); }
bool Shared::is_upper(const u32& s) const { return s != to_lower(s) && s == to_upper(s); }

// ------------------------------------------------------------------------------------------------ FrontEnd
namespace {

std::unordered_map<u32, u32> read_pairs(const GgufModel& model, const std::string& keys, const std::string& values) {
    const auto k = model.kv_arr_str(kPrefix + keys), v = model.kv_arr_str(kPrefix + values);
    if (k.size() != v.size()) throw LoadError("KaleTtsVocab: " + keys + " and " + values + " differ in length");
    std::unordered_map<u32, u32> out;
    out.reserve(k.size());
    for (size_t i = 0; i < k.size(); ++i) out[U(k[i])] = U(v[i]);
    return out;
}

} // namespace

FrontEnd::FrontEnd(const Shared& sh, const Tables& t, const GgufModel& model, const std::string& accent, const Homographs& h,
                   const std::vector<std::pair<u32, u32>>& terms)
    : sh_(sh), british_(accent == "gb") {
    auto lex = read_pairs(model, accent + ".lexicon_keys", accent + ".lexicon_values");
    auto guesser = std::make_unique<G2pModel>(model.kv_arr_u8(kPrefix + accent + ".g2p"));
    g2p_ = std::make_unique<G2p>(sh, t, std::move(lex), std::move(guesser), british_, h);
    // dizione's `isWord` is the voice's word list (`lex.has`), the G2p's own.
    const G2p* g2p = g2p_.get();
    norm_ = std::make_unique<Normalizer>(sh, t, [g2p](const u32& w) { return g2p->has(w); }, british_);
    for (const auto& p : terms) terms_[p.first] = p.second;
}

u32 FrontEnd::apply_terms(const u32& text) const {
    // kale-tts.mjs termsReplacer: look up, split camelCase, look up again.
    if (terms_.empty()) return text;
    const JsRegex& pattern = sh_.rule("terms");
    auto look_up = [&](const u32& t) {
        return pattern.replace(t, [&](const u32& whole, const JsRegex::Match& m, size_t*) {
            const u32 f = whole.substr(m.begin, m.end - m.begin);
            const auto it = terms_.find(f);
            const u32 spoken = it != terms_.end() ? it->second : f;
            const bool glued = !f.empty() && f[0] == U'.' && m.begin > 0 &&
                               (is_letter(whole[m.begin - 1]) || is_number(whole[m.begin - 1]));
            return glued ? U" " + spoken : spoken;
        });
    };
    const JsRegex& camel = sh_.rule("camel");
    const JsRegex& camel_split = sh_.rule("camel_split");
    const u32 split = camel.replace(look_up(text), [&](const u32& whole, const JsRegex::Match& m, size_t*) {
        return camel_split.replace(whole.substr(m.begin, m.end - m.begin), U" ");
    });
    return look_up(split);
}

u32 FrontEnd::normalize(const u32& text) const { return (*norm_)(apply_terms(text)); }

std::vector<u32> FrontEnd::sentences(const u32& text) const {
    // kaleTts.js `sentences`: split after a sentence mark (or at a newline), the list's last comma dropped, a
    // one-word sentence carried into the next, and a last word alone added to the sentence before it.
    auto trim = [](const u32& s) {
        size_t a = 0, b = s.size();
        auto sp = [](char32_t c) { return c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' || c == 0xA0 || c == 0x0B || c == 0x0C; };
        while (a < b && sp(s[a])) ++a;
        while (b > a && sp(s[b - 1])) --b;
        return s.substr(a, b - a);
    };
    auto drop_list_comma = [&](const u32& s) {
        return sh_.rule("list_comma").replace(s, [&](const u32& whole, const JsRegex::Match& m, size_t*) {
            const u32 full = whole.substr(m.begin, m.end - m.begin);
            const u32 tail = m.group(whole, 1);
            const auto clauses = sh_.rule("clause_split").split(whole.substr(0, m.begin));
            const u32 clause = clauses.back();
            std::vector<u32> parts;
            u32 cur;
            for (char32_t c : clause) {
                if (c == U',') { parts.push_back(cur); cur.clear(); }
                else cur += c;
            }
            parts.push_back(cur);
            if (parts.size() < 2) return full;
            for (size_t i = 1; i < parts.size(); ++i) {
                size_t n = 0;
                for (const u32& w : sh_.rule("space_split").split(trim(parts[i]))) if (!w.empty()) ++n;
                if (!(n >= 1 && n <= 2)) return full;
            }
            return tail;
        });
    };
    std::vector<u32> out;
    u32 carry;
    for (const u32& part : sh_.rule("sentence_split").split(normalize(text))) {
        const u32 s = drop_list_comma(trim(part));
        if (s.empty()) continue;
        const u32 joined = carry.empty() ? s : carry + U" " + s;
        if (sh_.rule("one_word").match_all(joined).size() == 1) { carry = joined; continue; }
        out.push_back(joined);
        carry.clear();
    }
    if (!carry.empty()) {
        if (!out.empty()) out.back() += U" " + carry;
        else out.push_back(carry);
    }
    return out;
}

} // namespace kale

// ------------------------------------------------------------------------------------------------ KaleTtsVocab
struct KaleTtsVocab::Impl {
    kale::Shared shared;
    kale::Tables tables;
    std::unique_ptr<kale::Homographs> homographs;
    std::unique_ptr<kale::FrontEnd> us, gb;
    std::unique_ptr<BpeVocab> ettin;
    int32_t cls = -1, sep = -1;
    bool strip_first = false, strip_last = false;

    const kale::FrontEnd& front(const std::string& lang) const {
        std::string l;
        for (char c : lang) l += c == '_' ? '-' : static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return (l == "en-gb" || l == "en-uk") ? *gb : *us;
    }

    std::vector<int32_t> plain_bpe(const std::string& text) const {
        std::vector<int32_t> ids = ettin->encode(text);
        if (strip_first && !ids.empty() && ids.front() == cls) ids.erase(ids.begin());
        if (strip_last && !ids.empty() && ids.back() == sep) ids.pop_back();
        return ids;
    }

    // js/kale-tts.mjs `_ids` with `knIPA` (the model reads Kokoro's own symbols): the IPA characters the table
    // knows, rewritten, then their ids.
    std::vector<int32_t> ids_of(const kale::u32& ipa, bool british) const {
        using kale::u32;
        std::vector<u32> chars;
        for (char32_t c : ipa) if (tables.vocab.count(u32(1, c))) chars.push_back(u32(1, c));
        std::vector<u32> out;
        for (size_t i = 0; i < chars.size();) {
            const u32 two = chars[i] + (i + 1 < chars.size() ? chars[i + 1] : u32());
            u32 pair;
            if (two == U"eɪ") pair = U"A"; else if (two == U"aɪ") pair = U"I"; else if (two == U"aʊ") pair = U"W";
            else if (two == U"ɔɪ") pair = U"Y"; else if (two == U"tʃ") pair = U"ʧ"; else if (two == U"dʒ") pair = U"ʤ";
            else if (british && two == U"əʊ") pair = U"Q";
            else if (!british && two == U"oʊ") pair = U"O";
            if (!pair.empty()) { out.push_back(pair); i += 2; continue; }
            const u32 c = chars[i++];
            if (british) { out.push_back(c); continue; }
            if (c == U"ː") continue;
            if (c == U"ɚ") { out.push_back(U"ə"); out.push_back(U"ɹ"); }
            else if (c == U"ɝ") { out.push_back(U"ɜ"); out.push_back(U"ɹ"); }
            else if (c == U"ɾ") out.push_back(U"T");
            else out.push_back(c);
        }
        static const u32 kVow = U"aeiouɑɐɒæəɘɚɛɜɝɞɪɨʊʉʌɔøœɤɯʏyAIWOYᵊᵻ";
        static const u32 kEnd = U" .,!?;:";
        for (size_t k = 1; k + 1 < out.size(); ++k) {
            const u32& nx = out[k + 1];
            const u32& pv = out[k - 1];
            const bool space_before = pv.size() == 1 && (pv[0] == U' ' || pv[0] == U'\t' || pv[0] == U'\n' || pv[0] == 0xA0);
            if (out[k] == U"ə" && nx.size() == 1 && u32(U"lnm").find(nx[0]) != u32::npos &&
                (k + 2 == out.size() || (out[k + 2].size() == 1 && kEnd.find(out[k + 2][0]) != u32::npos)) &&
                !(pv.size() == 1 && kVow.find(pv[0]) != u32::npos) && pv != U"ˈ" && pv != U"ˌ" && !space_before) {
                out[k] = U"ᵊ";
            }
        }
        std::vector<int32_t> ids;
        for (const u32& c : out) {
            const auto it = tables.vocab.find(c);
            if (it != tables.vocab.end()) ids.push_back(it->second);
        }
        return ids;
    }
};

KaleTtsVocab::KaleTtsVocab() : impl_(new Impl()) {}
KaleTtsVocab::~KaleTtsVocab() = default;

std::unique_ptr<KaleTtsVocab> KaleTtsVocab::load(const GgufModel& model) {
    using kale::kPrefix;
    using kale::U;
    if (!model.has_kv("tokenizer.ggml.model") || model.kv_str("tokenizer.ggml.model") != "kale_tts") return nullptr;
    for (const char* key : {"rule_names", "tables", "us.lexicon_keys", "gb.lexicon_keys", "us.g2p", "gb.g2p", "nfkc_from"}) {
        if (!model.has_kv(kPrefix + key)) {
            throw LoadError(std::string("KaleTtsVocab::load: tokenizer.ggml.model is 'kale_tts' but ") + kPrefix + key + " is missing");
        }
    }
    std::unique_ptr<KaleTtsVocab> v(new KaleTtsVocab());
    Impl& p = *v->impl_;
    const auto names = model.kv_arr_str(kPrefix + "rule_names");
    const auto patterns = model.kv_arr_str(kPrefix + "rule_patterns");
    const auto flags = model.kv_arr_str(kPrefix + "rule_flags");
    if (names.size() != patterns.size() || names.size() != flags.size()) {
        throw LoadError("KaleTtsVocab::load: rule_names, rule_patterns and rule_flags differ in length");
    }
    for (size_t i = 0; i < names.size(); ++i) p.shared.rules.emplace(names[i], JsRegex(patterns[i], flags[i]));
    for (const char* table : {"nfkc", "upper", "lower"}) {
        const auto from = model.kv_arr_i32(kPrefix + table + "_from");
        const auto to = model.kv_arr_str(kPrefix + table + "_to");
        if (from.size() != to.size()) throw LoadError(std::string("KaleTtsVocab::load: ") + table + "_from and _to differ in length");
        auto& m = std::string(table) == "nfkc" ? p.shared.nfkc : std::string(table) == "upper" ? p.shared.upper : p.shared.lower;
        for (size_t i = 0; i < from.size(); ++i) m[static_cast<char32_t>(from[i])] = U(to[i]);
    }

    const auto j = nlohmann::json::parse(model.kv_str(kPrefix + "tables"));
    for (const auto& [k, val] : j["ABBREVIATIONS"].items()) p.tables.abbreviations[U(k)] = {U(val[0]), U(val[1])};
    for (const auto& [k, val] : j["UNITS"].items()) p.tables.units[U(k)] = {U(val[0]), U(val[1])};
    for (const auto& [k, val] : j["EXACT_UNITS"].items()) p.tables.exact_units[U(k)] = {U(val[0]), U(val[1])};
    for (const auto& [k, val] : j["CHAT"].items()) p.tables.chat[U(k)] = U(val.get<std::string>());
    for (const auto& [k, val] : j["ENDINGS"].items()) p.tables.endings[U(k)] = U(val.get<std::string>());
    for (const auto& [k, val] : j["SYMBOL_WORDS"].items()) p.tables.symbol_words[U(k)] = U(val.get<std::string>());
    for (const auto& r : j["HOMOGRAPH_RULES"]) p.tables.homograph_rules.emplace_back(U(r[0]), U(r[1]), U(r[2]));
    for (const auto& [k, val] : j["VOCAB"].items()) p.tables.vocab[U(k)] = val.get<int32_t>();
    p.tables.homographs_json = j["HOMOGRAPHS"].dump();
    p.homographs = std::make_unique<kale::Homographs>(p.shared, p.tables);

    std::vector<std::pair<kale::u32, kale::u32>> terms;
    const auto tk = model.kv_arr_str(kPrefix + "terms_keys"), tv = model.kv_arr_str(kPrefix + "terms_values");
    for (size_t i = 0; i < tk.size() && i < tv.size(); ++i) if (!tk[i].empty()) terms.emplace_back(U(tk[i]), U(tv[i]));
    p.us = std::make_unique<kale::FrontEnd>(p.shared, p.tables, model, "us", *p.homographs, terms);
    p.gb = std::make_unique<kale::FrontEnd>(p.shared, p.tables, model, "gb", *p.homographs, terms);

    p.ettin = BpeVocab::load_bpe(model);
    p.cls = p.ettin->piece_to_id("[CLS]");
    p.sep = p.ettin->piece_to_id("[SEP]");
    if (p.cls < 0 || p.sep < 0) throw LoadError("KaleTtsVocab::load: the Ettin vocabulary has no [CLS] or [SEP]");
    const auto probe = p.ettin->encode("a");
    p.strip_first = !probe.empty() && probe.front() == p.cls;
    p.strip_last = !probe.empty() && probe.back() == p.sep;
    return v;
}

size_t KaleTtsVocab::size() const { return impl_->ettin->size(); }

std::string KaleTtsVocab::decode(const std::vector<int32_t>& ids) const { return impl_->ettin->decode(ids); }

std::vector<std::string> KaleTtsVocab::sentences(const std::string& text, const std::string& lang) const {
    std::vector<std::string> out;
    for (const auto& s : impl_->front(lang).sentences(kale::U(text))) out.push_back(kale::S(s));
    return out;
}

std::string KaleTtsVocab::ipa(const std::string& sentence, const std::string& lang) const {
    return kale::S(impl_->front(lang).ipa(kale::U(sentence)));
}

std::vector<int32_t> KaleTtsVocab::encode(const std::string& text, const std::string& lang) const {
    using kale::u32;
    const Impl& p = *impl_;
    const kale::FrontEnd& fe = p.front(lang);
    const bool british = fe.british();
    constexpr int32_t kSpace = 16;   // Kokoro's id for " " (wordIndex's spaceId)
    constexpr size_t kMaxChars = 500;  // MAX_IDS: one model call at most, cut at a space
    std::vector<int32_t> out = {0};
    int32_t records = 0;
    u32 prev;
    for (const u32& s : fe.sentences(kale::U(text))) {
        const u32 ipa = fe.ipa(s);
        // The chunks of the sentence's IPA (`_parts`).
        std::vector<u32> chunks;
        u32 rest = ipa;
        while (rest.size() > kMaxChars) {
            size_t cut = rest.rfind(U' ', kMaxChars);
            if (cut == u32::npos || cut == 0) cut = kMaxChars;
            chunks.push_back(rest.substr(0, cut));
            rest = rest.size() > cut ? rest.substr(cut + 1) : u32();
        }
        chunks.push_back(rest);

        // The sentence's words and Ettin inputs (ettin.js `ettinInputs`, `wordIndex`).
        std::vector<u32> words;
        {
            u32 cur;
            for (char32_t c : s) {
                if (c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' || c == 0xA0) { if (!cur.empty()) words.push_back(cur); cur.clear(); }
                else cur += c;
            }
            if (!cur.empty()) words.push_back(cur);
        }
        const std::vector<int32_t> all = p.ids_of(ipa, british);
        std::vector<int32_t> widx;
        {
            std::vector<std::vector<int32_t>> per;
            for (const u32& w : words) per.push_back(p.ids_of(fe.ipa(w), british));
            std::vector<int32_t> joined;
            for (size_t i = 0; i < per.size(); ++i) {
                joined.insert(joined.end(), per[i].begin(), per[i].end());
                if (i + 1 < per.size()) joined.push_back(kSpace);
            }
            if (joined == all) {
                for (size_t i = 0; i < per.size(); ++i) {
                    for (size_t k = 0; k < per[i].size(); ++k) widx.push_back(static_cast<int32_t>(i));
                    if (i + 1 < per.size()) widx.push_back(static_cast<int32_t>(i));
                }
            } else {
                const long n_words = static_cast<long>(words.size());
                long nw = 1;
                for (int32_t x : all) if (x == kSpace) ++nw;
                long k = 0;
                for (int32_t x : all) {
                    const long w = std::min<long>(std::max<long>(n_words, 1) - 1, (k * n_words) / nw);
                    widx.push_back(static_cast<int32_t>(w));
                    if (x == kSpace) ++k;
                }
            }
        }
        std::vector<int32_t> sub = {p.cls}, sub_word = {-1};
        {
            // `prev.trim() + " " + words.join(" ")`, tokenized piece by piece: GPT-2's pre-tokenizer joins a single
            // space to the word after it, so every word's sub-tokens are its own ` word`'s.
            u32 pre = prev;
            while (!pre.empty() && (pre.back() == U' ')) pre.pop_back();
            while (!pre.empty() && pre.front() == U' ') pre.erase(pre.begin());
            if (!pre.empty()) {
                for (int32_t id : p.plain_bpe(kale::S(pre))) { sub.push_back(id); sub_word.push_back(-1); }
            }
            for (size_t i = 0; i < words.size(); ++i) {
                const bool spaced = !(i == 0 && pre.empty());
                const u32 piece = (spaced ? u32(U" ") : u32()) + words[i];
                const std::vector<int32_t> ids = p.plain_bpe(kale::S(piece));
                for (size_t k = 0; k < ids.size(); ++k) {
                    // A sub-token that is the leading space alone ("Ġ" before ")))'") spans no character of the word,
                    // and upstream's overlap test gives it to none.
                    const bool space_only = spaced && k == 0 && p.ettin->id_to_piece(ids[k]) == "\xC4\xA0";
                    sub.push_back(ids[k]);
                    sub_word.push_back(space_only ? -1 : static_cast<int32_t>(i));
                }
            }
            sub.push_back(p.sep);
            sub_word.push_back(-1);
        }
        const int32_t n_words = static_cast<int32_t>(std::max<size_t>(words.size(), 1));

        size_t at = 0;
        for (const u32& c : chunks) {
            const std::vector<int32_t> ids = p.ids_of(c, british);
            if (!ids.empty()) {
                while (at < all.size() && all[at] != ids[0]) ++at;
            }
            if (!ids.empty()) {
                out.push_back(static_cast<int32_t>(ids.size()));
                out.insert(out.end(), ids.begin(), ids.end());
                for (size_t k = 0; k < ids.size(); ++k) {
                    // `widx.slice(at, at + n)`: past the sentence's end it is short, and the record takes the last.
                    const size_t q = std::min(at + k, widx.empty() ? 0 : widx.size() - 1);
                    out.push_back(widx.empty() ? 0 : widx[q]);
                }
                out.push_back(n_words);
                out.push_back(static_cast<int32_t>(sub.size()));
                out.insert(out.end(), sub.begin(), sub.end());
                out.insert(out.end(), sub_word.begin(), sub_word.end());
                ++records;
            }
            at += ids.size();
        }
        prev = s;
    }
    out[0] = records;
    return out;
}

} // namespace loom
