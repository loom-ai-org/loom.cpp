// kale-tts-1's words -> IPA: frontend/words/kaleTts.js's `G2p` and `G2pModel`, frontend/dizione/homographs.js, and
// frontend/words/enGb.js's `BritishG2p` / `toBritish`, method for method. The lexicons, the guesser's weights, the
// homograph data and the patterns are the file's (ADR-142).
//
// The guesser is evaluated as upstream's JavaScript evaluates it: every product and sum in double, one term at a
// time, in upstream's order (this file is compiled with -ffp-contract=off so no multiply-add is fused), fp16
// weights widened exactly.

#include "kale_tts_frontend.h"

#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace loom {
namespace kale {
namespace {

const u32 kVowels = U"aeiouɑɐɒæɔəɘɚɛɜɝɞɨɪʊʌɤøœɶɯɵʉʏᵻ";
bool is_vowel(char32_t c) { return kVowels.find(c) != u32::npos; }

u32 letter_sound(char32_t c) {
    switch (c) {
        case U'a': return U"æ"; case U'b': return U"b"; case U'c': return U"k"; case U'd': return U"d"; case U'e': return U"ɛ";
        case U'f': return U"f"; case U'g': return U"ɡ"; case U'h': return U"h"; case U'i': return U"ɪ"; case U'j': return U"dʒ";
        case U'k': return U"k"; case U'l': return U"l"; case U'm': return U"m"; case U'n': return U"n"; case U'o': return U"ɑː";
        case U'p': return U"p"; case U'q': return U"k"; case U'r': return U"ɹ"; case U's': return U"s"; case U't': return U"t";
        case U'u': return U"ʌ"; case U'v': return U"v"; case U'w': return U"w"; case U'x': return U"ks"; case U'y': return U"j";
        case U'z': return U"z";
        default: return U"";
    }
}
bool has_letter_sound(char32_t c) { return c >= U'a' && c <= U'z'; }

bool ends_with(const u32& s, const u32& p) { return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0; }
bool starts_at(const u32& s, size_t i, const u32& p) { return s.size() >= i + p.size() && s.compare(i, p.size(), p) == 0; }

u32 norm(const u32& t) {   // kaleTts.js `norm`: ’ -> '
    u32 out = t;
    for (char32_t& c : out) if (c == U'’') c = U'\'';
    return out;
}
u32 strip(const u32& ipa, const u32& chars) {
    u32 out;
    for (char32_t c : ipa) if (chars.find(c) == u32::npos) out += c;
    return out;
}
u32 unstress(const u32& ipa) { return strip(ipa, U"ˈˌ"); }
u32 secondary(const u32& ipa) {
    u32 out = ipa;
    for (char32_t& c : out) if (c == U'ˈ') c = U'ˌ';
    return out;
}
u32 bare(const u32& ipa) { return strip(ipa, U"ˈˌː"); }

std::optional<char32_t> first_sound(const u32& ipa) {
    for (char32_t c : ipa) if (c != U'ˈ' && c != U'ˌ') return c;
    return std::nullopt;
}

u32 stress_first_vowel(const u32& ipa) {
    const u32 plain = unstress(ipa);
    for (size_t at = 0; at < plain.size(); ++at) {
        if (is_vowel(plain[at])) return plain.substr(0, at) + U"ˈ" + plain.substr(at);
    }
    return plain;
}

u32 plural_suffix(const u32& ipa) {
    for (const char32_t* s : {U"s", U"z", U"ʃ", U"ʒ", U"tʃ", U"dʒ"}) if (ends_with(ipa, s)) return U"ᵻz";
    for (const char32_t* s : {U"p", U"t", U"k", U"f", U"θ"}) if (ends_with(ipa, s)) return U"s";
    return U"z";
}
u32 past_suffix(const u32& ipa) {
    if (ends_with(ipa, U"t") || ends_with(ipa, U"d")) return U"ᵻd";
    for (const char32_t* s : {U"p", U"t", U"k", U"f", U"θ", U"s", U"ʃ", U"tʃ"}) if (ends_with(ipa, s)) return U"t";
    return U"d";
}

size_t edit_distance(const u32& x, const u32& y) {
    std::vector<size_t> d(y.size() + 1);
    for (size_t j = 0; j <= y.size(); ++j) d[j] = j;
    for (size_t i = 1; i <= x.size(); ++i) {
        size_t prev = d[0];
        d[0] = i;
        for (size_t j = 1; j <= y.size(); ++j) {
            const size_t t = d[j];
            d[j] = std::min({d[j] + 1, d[j - 1] + 1, prev + (x[i - 1] != y[j - 1] ? 1 : 0)});
            prev = t;
        }
    }
    return d[y.size()];
}

// kaleTts.js `fold`: accents off (NFD, marks dropped), a handful of letters spelled out, NFC.
u32 fold(const u32& text) {
    const auto nfd = utf8_decode(nfd_normalize(S(text)));
    u32 out;
    for (char32_t ch : nfd) {
        if (is_mark(ch)) continue;
        switch (ch) {
            case U'æ': out += U"ae"; break; case U'Æ': out += U"Ae"; break; case U'œ': out += U"oe"; break;
            case U'Œ': out += U"Oe"; break; case U'ø': out += U"o"; break; case U'Ø': out += U"O"; break;
            case U'ß': out += U"ss"; break; case U'ł': out += U"l"; break; case U'Ł': out += U"L"; break;
            case U'đ': out += U"d"; break; case U'Đ': out += U"D"; break; case U'ð': out += U"d"; break;
            case U'Ð': out += U"D"; break; case U'þ': out += U"th"; break; case U'Þ': out += U"Th"; break;
            case U'ı': out += U"i"; break; case U'ʻ': break; case U'ʼ': break;
            case U'‘': out += U"'"; break; case U'`': out += U"'"; break;
            default: out += ch;
        }
    }
    return U(nfc_normalize(S(out)));
}

// kaleTts.js `PUNCT`: a mark token and what it appends to the word before it.
std::optional<u32> punct(const u32& tok) {
    if (tok.size() != 1) return std::nullopt;
    switch (tok[0]) {
        case U'.': return u32(U"."); case U',': return u32(U","); case U'!': return u32(U"!"); case U'?': return u32(U"?");
        case U';': return u32(U";"); case U':': return u32(U":"); case U'…': return u32(U"…");
        case U'—': return u32(U","); case U'–': return u32(U",");
        default: return std::nullopt;
    }
}

bool single_capital(const u32& t) { return t.size() == 1 && t[0] >= U'A' && t[0] <= U'Z'; }
bool capital_plural(const u32& t) {   // /^[A-Z]['’]s$/
    return t.size() == 3 && t[0] >= U'A' && t[0] <= U'Z' && (t[1] == U'\'' || t[1] == U'’') && t[2] == U's';
}
bool lower_ascii(const u32& t) {
    if (t.empty()) return false;
    for (char32_t c : t) if (!(c >= U'a' && c <= U'z')) return false;
    return true;
}

double half_to_double(uint16_t v) {
    const int sign = (v & 0x8000) ? -1 : 1, ex = (v >> 10) & 0x1f, m = v & 0x3ff;
    if (ex == 0) return static_cast<double>(static_cast<float>(sign * m * std::ldexp(1.0, -24)));
    if (ex == 0x1f) return m == 0 ? sign * INFINITY : NAN;
    return static_cast<double>(static_cast<float>(sign * (1.0 + m / 1024.0) * std::ldexp(1.0, ex - 15)));
}

double sigmoid(double x) { return 1 / (1 + std::exp(-x)); }
double row(const std::vector<double>& m, size_t k, const std::vector<double>& x, double bias) {
    double s = bias;
    const size_t base = k * x.size();
    for (size_t i = 0; i < x.size(); ++i) s += m[base + i] * x[i];
    return s;
}
std::vector<double> mat_vec(const std::vector<double>& m, const std::vector<double>& x, size_t rows, const std::vector<double>* b) {
    std::vector<double> o(rows);
    for (size_t k = 0; k < rows; ++k) o[k] = row(m, k, x, b ? (*b)[k] : 0);
    return o;
}
std::vector<double> cat(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> o(a);
    o.insert(o.end(), b.begin(), b.end());
    return o;
}

} // namespace

// ------------------------------------------------------------------------------------------------ G2pModel
G2pModel::G2pModel(const std::vector<uint8_t>& buf) {
    size_t o = 0;
    auto need = [&](size_t n) {
        if (o + n > buf.size()) throw LoadError("KaleTtsVocab: the g2p model ends early (it is the release's g2p.bin, KG2P v1)");
    };
    auto i32 = [&]() {
        need(4);
        int32_t v;
        std::memcpy(&v, buf.data() + o, 4);
        o += 4;
        return v;
    };
    if (buf.size() < 4 || std::memcmp(buf.data(), "KG2P", 4) != 0) throw LoadError("KaleTtsVocab: not a g2p model (no KG2P magic)");
    o = 4;
    if (i32() != 1) throw LoadError("KaleTtsVocab: a g2p model of another version than 1");
    e_ = i32(); h_ = i32(); d_ = i32();
    const int n_in = i32(), n_out = i32();
    max_out_ = i32();
    std::vector<int32_t> inputs(n_in);
    for (auto& v : inputs) v = i32();
    outputs_.resize(n_out);
    for (auto& v : outputs_) v = i32();
    auto take = [&](size_t n) {
        need(2 * n);
        std::vector<double> a(n);
        for (size_t k = 0; k < n; ++k) {
            uint16_t h;
            std::memcpy(&h, buf.data() + o, 2);
            o += 2;
            a[k] = half_to_double(h);
        }
        return a;
    };
    auto gru = [&](size_t cols, size_t size) {
        Gru g;
        g.w_ih = take(3 * size * cols);
        g.w_hh = take(3 * size * size);
        g.b_ih = take(3 * size);
        g.b_hh = take(3 * size);
        return g;
    };
    const size_t e = e_, h = h_, d = d_;
    emb_in_ = take(n_in * e);
    enc_fwd_ = gru(e, h);
    enc_bwd_ = gru(e, h);
    init_w_ = take(d * 2 * h);
    init_b_ = take(d);
    emb_out_ = take(n_out * e);
    dec_ = gru(e + 2 * h, d);
    att_w_ = take(2 * h * d);
    comb_w_ = take(d * (d + 2 * h));
    comb_b_ = take(d);
    out_w_ = take(n_out * d);
    out_b_ = take(n_out);
    if (o != buf.size()) throw LoadError("KaleTtsVocab: the g2p model has trailing bytes");
    for (int i = 1; i < n_in; ++i) in_index_[static_cast<char32_t>(inputs[i])] = i;
}

std::vector<double> G2pModel::gru_cell(const Gru& g, const std::vector<double>& x, const std::vector<double>& hp, int size) const {
    std::vector<double> out(size);
    for (int k = 0; k < size; ++k) {
        const double r = sigmoid(row(g.w_ih, k, x, g.b_ih[k]) + row(g.w_hh, k, hp, g.b_hh[k]));
        const double z = sigmoid(row(g.w_ih, size + k, x, g.b_ih[size + k]) + row(g.w_hh, size + k, hp, g.b_hh[size + k]));
        const double c = std::tanh(row(g.w_ih, 2 * size + k, x, g.b_ih[2 * size + k]) + r * row(g.w_hh, 2 * size + k, hp, g.b_hh[2 * size + k]));
        out[k] = (1 - z) * c + z * hp[k];
    }
    return out;
}

std::optional<u32> G2pModel::phonemes(const u32& word) const {
    if (word.empty() || word.size() > 24) return std::nullopt;
    std::vector<int> ids;
    for (char32_t ch : word) {
        const auto it = in_index_.find(ch);
        if (it == in_index_.end()) return std::nullopt;
        ids.push_back(it->second);
    }
    const size_t n = ids.size(), e = e_;
    auto emb = [&](const std::vector<double>& table, int id) {
        return std::vector<double>(table.begin() + id * e, table.begin() + id * e + e);
    };
    std::vector<std::vector<double>> fwd(n), bwd(n);
    std::vector<double> hf(h_, 0.0), hb(h_, 0.0);
    for (size_t t = 0; t < n; ++t) { hf = gru_cell(enc_fwd_, emb(emb_in_, ids[t]), hf, h_); fwd[t] = hf; }
    for (size_t t = n; t-- > 0;) { hb = gru_cell(enc_bwd_, emb(emb_in_, ids[t]), hb, h_); bwd[t] = hb; }
    std::vector<std::vector<double>> enc(n);
    for (size_t t = 0; t < n; ++t) enc[t] = cat(fwd[t], bwd[t]);
    std::vector<double> state = mat_vec(init_w_, cat(hf, hb), d_, &init_b_);
    for (double& v : state) v = std::tanh(v);
    std::vector<double> ctx(2 * h_, 0.0);
    int prev = 1;   // BOS
    u32 out;
    const size_t limit = std::min<size_t>(max_out_, 3 * n + 6);
    bool ended = false;
    for (size_t step = 0; step < limit; ++step) {
        state = gru_cell(dec_, cat(emb(emb_out_, prev), ctx), state, d_);
        const std::vector<double> q = mat_vec(att_w_, state, 2 * h_, nullptr);
        std::vector<double> scores(n);
        for (size_t t = 0; t < n; ++t) {
            double s = 0;
            for (size_t i = 0; i < q.size(); ++i) s += enc[t][i] * q[i];
            scores[t] = s;
        }
        double top = scores[0];
        for (double s : scores) top = std::max(top, s);
        double total = 0;
        for (size_t t = 0; t < n; ++t) { scores[t] = std::exp(scores[t] - top); total += scores[t]; }
        ctx.assign(2 * h_, 0.0);
        for (int k = 0; k < 2 * h_; ++k) {
            double s = 0;
            for (size_t t = 0; t < n; ++t) s += scores[t] * enc[t][k];
            ctx[k] = s / total;
        }
        std::vector<double> hid = mat_vec(comb_w_, cat(state, ctx), d_, &comb_b_);
        for (double& v : hid) v = std::tanh(v);
        const std::vector<double> logits = mat_vec(out_w_, hid, outputs_.size(), &out_b_);
        size_t best = 0;
        for (size_t i = 1; i < logits.size(); ++i) if (logits[i] > logits[best]) best = i;
        if (best == 2) { ended = true; break; }   // EOS
        if (best < 3) return std::nullopt;
        out += static_cast<char32_t>(outputs_[best]);
        prev = static_cast<int>(best);
    }
    if (!ended) return std::nullopt;
    for (char32_t c : out) if (is_vowel(c)) return out;
    return std::nullopt;
}

// ------------------------------------------------------------------------------------------------ homographs
struct Homographs::Impl {
    const Shared& sh;
    nlohmann::json data;
    std::unordered_map<u32, std::string> tag_of;
    std::set<u32> plural_bases;
    std::map<u32, std::vector<std::pair<u32, std::vector<std::pair<u32, std::set<u32>>>>>> rules;

    std::string tag(const u32& token) const {
        if (token == U"<s>" || token == U"</s>") return S(token);
        if (token.empty() || !is_letter(token[0])) return (!token.empty() && token[0] >= U'0' && token[0] <= U'9') ? "NUM" : "PUNCT";
        const u32 lower = sh.to_lower(token);
        const auto it = tag_of.find(lower);
        if (it != tag_of.end()) return it->second;
        const std::pair<const char32_t*, const char*> ends[] = {{U"'s", "POS"}, {U"ly", "LY"}, {U"ing", "ING"}, {U"ed", "ED"}};
        for (const auto& e : ends) {
            const u32 ending = e.first;
            if (ends_with(lower, ending) && lower.size() > ending.size() + 1) return e.second;
        }
        if (ends_with(lower, U"s") && !ends_with(lower, U"ss") && lower.size() > 3) return "S";
        return sh.is_upper(token.substr(0, 1)) ? "CAP" : "W";
    }

    std::pair<std::vector<u32>, size_t> sentence_of(const std::vector<u32>& tokens, size_t index) const {
        auto end_mark = [](const u32& t) { return t == U"." || t == U"?" || t == U"!"; };
        size_t start = index;
        while (start > 0 && !end_mark(tokens[start - 1])) --start;
        size_t end = index;
        while (end < tokens.size() && !end_mark(tokens[end])) ++end;
        const size_t stop = std::min(tokens.size(), end + 1);
        return {std::vector<u32>(tokens.begin() + start, tokens.begin() + stop), index - start};
    }

    std::vector<std::string> features(const std::vector<u32>& tokens, size_t index) const {
        auto at = [&](long offset) -> u32 {
            const long position = static_cast<long>(index) + offset;
            if (position >= 0 && position < static_cast<long>(tokens.size())) return tokens[position];
            return offset < 0 ? U"<s>" : U"</s>";
        };
        const u32 p2 = at(-2), p1 = at(-1), n1 = at(1), n2 = at(2);
        const std::string tp2 = tag(p2), tp1 = tag(p1), tn1 = tag(n1), tn2 = tag(n2);
        std::vector<std::string> out = {"tp1=" + tp1, "tp2=" + tp2, "tn1=" + tn1, "tn2=" + tn2, "tp2p1=" + tp2 + "+" + tp1,
                                        "tp1n1=" + tp1 + "+" + tn1, "tn1n2=" + tn1 + "+" + tn2};
        if (index == 0) out.push_back("first");
        else if (sh.is_upper(tokens[index].substr(0, 1))) out.push_back("cap");
        out.push_back("p1=" + S(sh.to_lower(p1)));
        out.push_back("p2=" + S(sh.to_lower(p2)));
        out.push_back("n1=" + S(sh.to_lower(n1)));
        out.push_back("n2=" + S(sh.to_lower(n2)));
        const u32 target = sh.to_lower(tokens[index]);
        std::vector<u32> nearby;
        const size_t lo = index >= 6 ? index - 6 : 0, hi = std::min(tokens.size(), index + 7);
        for (size_t i = lo; i < hi; ++i) {
            const u32& t = tokens[i];
            if (t.empty() || !is_letter(t[0])) continue;
            const u32 l = sh.to_lower(t);
            if (l == target || std::find(nearby.begin(), nearby.end(), l) != nearby.end()) continue;
            nearby.push_back(l);
        }
        for (const u32& w : nearby) out.push_back("w=" + S(w));
        return out;
    }

    u32 left(const std::vector<u32>& tokens, size_t index, bool aux) const {
        const long stop = std::max<long>(static_cast<long>(index) - 6, -1);
        for (long position = static_cast<long>(index) - 1; position > stop; --position) {
            const u32& token = tokens[position];
            const std::string t = tag(token);
            const bool skip = t == "ADV" || t == "NEG" || t == "LY" ||
                              (aux && (t == "SUBJ" || t == "OBJ" || t == "DET" || t == "POSS"));
            if (skip) continue;
            const u32 lower = sh.to_lower(token);
            return ends_with(lower, U"'ll") ? U"will" : ends_with(lower, U"'ve") ? U"have" : lower;
        }
        return U"<s>";
    }

    bool holds(const u32& condition, const std::set<u32>& words, const std::vector<u32>& tokens, size_t index) const {
        if (condition == U"first") return index == 0;
        if (condition == U"aux") return words.count(left(tokens, index, true)) != 0;
        if (condition == U"before") return words.count(left(tokens, index, false)) != 0;
        if (condition == U"next") return words.count(index + 1 < tokens.size() ? sh.to_lower(tokens[index + 1]) : U"</s>") != 0;
        const size_t lo = index >= 6 ? index - 6 : 0, hi = std::min(tokens.size(), index + 7);
        for (size_t i = lo; i < hi; ++i) if (words.count(sh.to_lower(tokens[i]))) return true;
        return false;
    }

    std::vector<double> scores(const std::string& word, const std::vector<u32>& tokens, size_t index) const {
        const auto& entry = data["words"][word];
        const auto [sentence, position] = sentence_of(tokens, index);
        const auto names = features(sentence, position);
        const auto& model = entry["model"];
        auto weight = [](const nlohmann::json& w, const std::string& name) {
            const auto it = w.find(name);
            return it == w.end() ? 0.0 : it->get<double>();
        };
        if (model["kind"] == "noun-verb") {
            double shared = 0;
            for (const auto& n : names) {
                if (n[0] == 't' || n == "first" || n == "cap") shared = shared + weight(data["pooled"]["weights"], n);
            }
            double verb = data["pooled"]["bias"].get<double>() + shared;
            double own = 0;
            for (const auto& n : names) own = own + weight(model["weights"], n);
            verb += model["bias"].get<double>() + own;
            std::vector<double> out;
            for (const auto& s : entry["senses"]) out.push_back(s["role"] == "v" ? verb : -verb);
            return out;
        }
        std::vector<double> out;
        for (size_t i = 0; i < model["bias"].size(); ++i) {
            double sum = 0;
            for (const auto& n : names) sum = sum + weight(model["weights"][i], n);
            out.push_back(model["bias"][i].get<double>() + sum);
        }
        return out;
    }

    const nlohmann::json& choose(const std::string& word, const u32& wword, const std::vector<u32>& tokens, size_t index) const {
        const auto& entry = data["words"][word];
        const auto [sentence, position] = sentence_of(tokens, index);
        const auto it = rules.find(wword);
        if (it != rules.end()) {
            for (const auto& [sense, conditions] : it->second) {
                bool all = true;
                for (const auto& [name, ws] : conditions) {
                    if (!holds(name, ws, sentence, position)) { all = false; break; }
                }
                if (all) {
                    for (const auto& s : entry["senses"]) if (s["id"] == S(sense)) return s;
                }
            }
        }
        const auto list = scores(word, tokens, index);
        size_t best = 0;
        for (size_t i = 1; i < list.size(); ++i) if (list[i] > list[best]) best = i;
        return entry["senses"][best];
    }
};

Homographs::Homographs(const Shared& sh, const Tables& t) : impl_(new Impl{sh, nlohmann::json::parse(t.homographs_json), {}, {}, {}}) {
    const std::pair<const char*, const char*> classes[] = {
        {"TO", "to"},
        {"DET", "the a an this that these those every each no some any another such what which whose"},
        {"POSS", "my your his her its our their"},
        {"SUBJ", "i you he she it we they who"},
        {"OBJ", "me him us them"},
        {"MODAL", "will would shall should can could may might must 'll 'd won't can't cannot couldn't shouldn't wouldn't let's please"},
        {"DO", "do does did don't doesn't didn't"},
        {"BE", "am is are was were be been being isn't aren't wasn't weren't i'm you're we're they're he's she's it's that's there's"},
        {"HAVE", "have has had having haven't hasn't hadn't i've you've we've they've"},
        {"NEG", "not never"},
        {"PREP", "of in on at by for from with into onto over under about after before between through during without within "
                 "against among around across behind beside near upon toward towards as than like"},
        {"CONJ", "and or but nor so yet if because although though while when where unless until since whether"},
        {"ADV", "very too quite rather more most less least also just only even still always often sometimes usually again "
                "now then here there soon already almost"},
    };
    for (const auto& c : classes) {
        u32 cur;
        for (char32_t ch : U(c.second)) {
            if (ch == U' ') { impl_->tag_of[cur] = c.first; cur.clear(); }
            else cur += ch;
        }
        impl_->tag_of[cur] = c.first;
    }
    for (const auto& b : impl_->data["plural_bases"]) impl_->plural_bases.insert(U(b.get<std::string>()));
    for (const auto& [word, sense, conditions] : t.homograph_rules) {
        std::vector<std::pair<u32, std::set<u32>>> parsed;
        u32 part;
        auto flush = [&](const u32& raw) {
            // `part.trim()`, then `name: words`.
            size_t a = 0, b = raw.size();
            while (a < b && (raw[a] == U' ' || raw[a] == U'\t')) ++a;
            while (b > a && (raw[b - 1] == U' ' || raw[b - 1] == U'\t')) --b;
            const u32 text = raw.substr(a, b - a);
            const size_t colon = text.find(U':');
            const u32 name = colon == u32::npos ? text : text.substr(0, colon);
            std::set<u32> ws;
            if (colon != u32::npos) {
                u32 w;
                for (char32_t ch : text.substr(colon + 1)) {
                    if (ch == U' ' || ch == U'\t') { if (!w.empty()) ws.insert(w); w.clear(); }
                    else w += ch;
                }
                if (!w.empty()) ws.insert(w);
            }
            parsed.emplace_back(name, ws);
        };
        for (char32_t ch : conditions) {
            if (ch == U';') { flush(part); part.clear(); }
            else part += ch;
        }
        flush(part);
        impl_->rules[word].emplace_back(sense, parsed);
    }
}
Homographs::~Homographs() = default;

bool Homographs::covers(const u32& word) const {
    if (impl_->data["words"].contains(S(word))) return true;
    return ends_with(word, U"s") && impl_->plural_bases.count(word.substr(0, word.size() - 1)) != 0;
}

Homographs::Reading Homographs::pronounce(const u32& word, const std::vector<u32>& tokens, size_t index) const {
    auto ipa = [](const nlohmann::json& sense) {
        u32 out;
        for (char32_t c : U(sense["phones"].get<std::string>())) if (c != U' ') out += c;
        return out;
    };
    const std::string w = S(word);
    if (impl_->data["words"].contains(w)) {
        Reading r{ipa(impl_->choose(w, word, tokens, index)), {}, false};
        for (const auto& s : impl_->data["words"][w]["senses"]) r.senses.push_back(ipa(s));
        return r;
    }
    const u32 base = word.substr(0, word.size() - 1);
    const std::string b = S(base);
    const auto& entry = impl_->data["words"][b];
    const nlohmann::json* adjective = nullptr;
    for (const auto& s : entry["senses"]) if (s["role"] == "n") { adjective = &s; break; }
    const nlohmann::json* chosen = nullptr;
    if (adjective && (*adjective)["id"] == "adj") {
        for (const auto& s : entry["senses"]) if (s["role"] == "v") { chosen = &s; break; }
    } else {
        chosen = &impl_->choose(b, base, tokens, index);
    }
    Reading r{chosen ? ipa(*chosen) : u32(), {}, true};
    for (const auto& s : entry["senses"]) r.senses.push_back(ipa(s));
    return r;
}

// ------------------------------------------------------------------------------------------------ G2p
G2p::G2p(const Shared& sh, const Tables& t, std::unordered_map<u32, u32> lexicon, std::unique_ptr<G2pModel> model, bool british,
         const Homographs& homographs)
    : sh_(sh), t_(t), lex_(std::move(lexicon)), model_(std::move(model)), british_(british), homographs_(homographs) {}

std::optional<u32> G2p::lex(const u32& key) const {
    const auto it = lex_.find(key);
    if (it == lex_.end()) return std::nullopt;
    return it->second;
}

u32 G2p::spell(const u32& word) const {
    std::vector<u32> parts;
    for (char32_t ch : word) {
        const u32 c = sh_.to_lower(u32(1, ch));
        u32 p;
        if (c == U"a") p = U"ˈeɪ";
        else if (c == U"i") p = U"ˈaɪ";
        else if (const auto l = lex(c)) p = *l;
        else if (c.size() == 1) p = letter_sound(c[0]);
        if (!p.empty()) parts.push_back(p);
    }
    if (parts.empty()) return U"";
    u32 out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) out += U' ';
        out += i + 1 < parts.size() ? secondary(parts[i]) : parts[i];
    }
    return out;
}

// kaleTts.js `senseRule`: "lead" and "read", ahead of dizione's picker.
static std::optional<u32> sense_rule(const u32& low, const std::vector<u32>& words, size_t i) {
    auto at = [&](long k) -> u32 { return k >= 0 && k < static_cast<long>(words.size()) ? words[k] : U""; };
    const long ii = static_cast<long>(i);
    if (low == U"lead") {
        static const std::set<u32> next_metal = {U"pipe", U"pipes", U"piping", U"paint", U"paints", U"poisoning", U"poison",
                                                 U"free", U"based", U"acid", U"oxide", U"ore", U"solder"};
        static const std::set<u32> prev_metal = {U"red", U"white"};
        static const std::set<u32> made = {U"made", U"built", U"cast", U"lined", U"pipes", U"weights"};
        const bool metal = next_metal.count(at(ii + 1)) || prev_metal.count(at(ii - 1)) ||
                           ((at(ii - 1) == U"of" || at(ii - 1) == U"from") && made.count(at(ii - 2)));
        return metal ? u32(U"lˈɛd") : u32(U"lˈiːd");
    }
    if (low == U"read") {
        static const std::set<u32> habit_words = {U"daily", U"often", U"nightly", U"regularly"};
        static const std::set<u32> periods = {U"day", U"night", U"morning", U"evening", U"week", U"weekend"};
        static const std::set<u32> past = {U"yesterday", U"ago", U"last", U"had", U"has", U"have", U"was", U"were", U"already"};
        const bool habit = habit_words.count(at(ii + 1)) ||
                           ((at(ii + 1) == U"every" || at(ii + 1) == U"each") && periods.count(at(ii + 2)));
        bool any_past = false;
        for (const u32& w : words) if (past.count(w)) any_past = true;
        if (habit && !any_past) return u32(U"ɹˈiːd");
    }
    return std::nullopt;
}

std::optional<u32> G2p::homograph(const u32& token, const std::vector<u32>& tokens, size_t index) const {
    const u32 w = norm(token), low = sh_.to_lower(w);
    if (w.size() > 1 && w == sh_.to_upper(w)) return std::nullopt;
    std::vector<u32> lowered, normed;
    for (const u32& t : tokens) { normed.push_back(norm(t)); lowered.push_back(sh_.to_lower(norm(t))); }
    std::optional<u32> r = sense_rule(low, lowered, index);
    if (!r) {
        if (!homographs_.covers(low)) return std::nullopt;
        const auto reading = homographs_.pronounce(low, normed, index);
        const auto own = lex(reading.plural ? low.substr(0, low.size() - 1) : low);
        u32 ipa = reading.sense;
        if (own) {
            auto distance = [](const u32& a, const u32& b) { return edit_distance(a, b); };
            u32 nearest = reading.senses[0];
            for (size_t k = 1; k < reading.senses.size(); ++k) {
                if (distance(reading.senses[k], *own) < distance(nearest, *own)) nearest = reading.senses[k];
            }
            if (nearest == reading.sense) ipa = *own;
        }
        r = reading.plural ? ipa + plural_suffix(ipa) : ipa;
    }
    return british_ ? british(*r, sh_.to_lower(token)) : *r;
}

std::optional<u32> G2p::word(const u32& token, const std::optional<u32>& next) const {
    const u32 w = norm(token), low = sh_.to_lower(w);
    if (w.size() > 1 && w == sh_.to_upper(w)) {
        if (const auto l = lex(w)) return *l;
        if (!(w.size() >= 3 && lex_.count(low)) && w.size() <= 5) return spell(w);
    }
    if (next) {
        const auto following = lookup(sh_.to_lower(norm(*next)), std::nullopt);
        const auto variant = lex(low + U"#v");
        if (following && variant) {
            const auto f = first_sound(*following);
            if (f && is_vowel(*f)) return *variant;
        }
    }
    return lookup(low, w);
}

std::optional<u32> G2p::lookup(const u32& low, const std::optional<u32>& original) const {
    if (const auto l = lex(low)) return letters(low, *l);
    if (!original) return std::nullopt;
    // `stems`, in upstream's order: each candidate stem and the ending its reading takes.
    using Suffix = u32 (*)(const u32&);
    std::vector<std::pair<std::vector<u32>, std::function<u32(const u32&)>>> stems;
    const u32& w = low;
    auto doubled = [](const u32& b) {
        std::vector<u32> out;
        if (b.size() > 2 && b[b.size() - 1] == b[b.size() - 2]) out.push_back(b.substr(0, b.size() - 1));
        return out;
    };
    auto cut = [&](size_t n) { return w.substr(0, w.size() - n); };
    if (ends_with(w, U"'s")) stems.push_back({{cut(2)}, static_cast<Suffix>(plural_suffix)});
    if (ends_with(w, U"s'")) stems.push_back({{cut(1)}, [](const u32&) { return u32(); }});
    if (ends_with(w, U"ies") && w.size() > 4) stems.push_back({{cut(3) + U"y"}, static_cast<Suffix>(plural_suffix)});
    if (ends_with(w, U"es") && w.size() > 3) stems.push_back({{cut(2)}, static_cast<Suffix>(plural_suffix)});
    if (ends_with(w, U"s") && !ends_with(w, U"ss") && w.size() > 3) stems.push_back({{cut(1)}, static_cast<Suffix>(plural_suffix)});
    if (ends_with(w, U"ied") && w.size() > 4) stems.push_back({{cut(3) + U"y"}, static_cast<Suffix>(past_suffix)});
    if (ends_with(w, U"ed") && w.size() > 4) {
        const u32 b = cut(2);
        std::vector<u32> c = {b, b + U"e"};
        for (const u32& d : doubled(b)) c.push_back(d);
        stems.push_back({c, static_cast<Suffix>(past_suffix)});
    }
    if (ends_with(w, U"ing") && w.size() > 5) {
        const u32 b = cut(3);
        std::vector<u32> c = {b, b + U"e"};
        for (const u32& d : doubled(b)) c.push_back(d);
        stems.push_back({c, [](const u32&) { return u32(U"ɪŋ"); }});
    }
    const std::pair<const char32_t*, const char32_t*> suffixes[] = {{U"ly", U"li"}, {U"ness", U"nəs"}, {U"ful", U"fəl"},
                                                                    {U"less", U"ləs"}, {U"er", U"ɚ"}, {U"est", U"ɪst"},
                                                                    {U"ment", U"mənt"}};
    for (const auto& [suf_c, sound_c] : suffixes) {
        const u32 suf = suf_c, sound = sound_c;
        if (ends_with(w, suf) && w.size() > suf.size() + 2) {
            const u32 b = cut(suf.size());
            std::vector<u32> c = {b};
            if (ends_with(b, U"i")) c.push_back(b.substr(0, b.size() - 1) + U"y");
            for (const u32& d : doubled(b)) c.push_back(d);
            stems.push_back({c, [sound](const u32&) { return sound; }});
        }
    }
    for (const auto& [cands, suffix] : stems) {
        for (const u32& stem : cands) {
            const auto base = lex(stem);
            if (base && !base->empty()) return *base + suffix(*base);
        }
    }
    if (low.size() == 1) {
        if (const auto l = lex(low)) return *l;
        return has_letter_sound(low[0]) ? letter_sound(low[0]) : u32();
    }
    if (const auto said = fallback(low)) {
        const auto j = joined(low, *said);
        return j ? *j : *said;
    }
    for (size_t i = 3; i + 2 < low.size(); ++i) {
        const auto a = lex(low.substr(0, i)), b = lex(low.substr(i));
        if (a && b) return *a + secondary(*b);
    }
    return chunks(low);
}

std::optional<u32> G2p::joined(const u32& low, const u32& said) const {
    if (low.size() < 6) return std::nullopt;
    const JsRegex& vowel = sh_.rule("vowel");
    for (size_t i = 3; i + 3 <= low.size(); ++i) {
        const auto a = lex(low.substr(0, i)), b = lex(low.substr(i));
        if (!a || !b) continue;
        const u32 two = *a + secondary(*b);
        const bool more = vowel.match_all(said).size() > vowel.match_all(two).size();
        if (more && edit_distance(bare(said), bare(two)) <= 1) return two;
        return std::nullopt;
    }
    return std::nullopt;
}

std::optional<u32> G2p::fallback(const u32& low) const {
    if (!model_) return std::nullopt;
    std::optional<u32> said;
    const auto it = learnt_.find(low);
    if (it != learnt_.end()) said = it->second;
    else {
        said = model_->phonemes(low);
        if (learnt_.size() >= 4096) learnt_.clear();
        learnt_[low] = said;
    }
    if (said && british_) return british(*said, low);
    return said;
}

u32 G2p::letters(const u32& low, const u32& ipa) const {
    if (low.size() < 2 || low.size() > 5 || low == U"ok" || !lower_ascii(low)) return ipa;
    u32 names;
    for (char32_t c : low) {
        if (c == U'a') names += U"eɪ";
        else {
            const auto l = lex(u32(1, c));
            names += bare(l ? *l : u32());
        }
    }
    return bare(ipa) == names ? spell(low) : ipa;
}

u32 G2p::chunks(const u32& w) const {
    u32 out;
    size_t i = 0;
    while (i < w.size()) {
        bool matched = false;
        for (size_t j = w.size(); j > i + 2; --j) {
            const auto piece = lex(w.substr(i, j - i));
            if (piece && !piece->empty()) {
                out += unstress(*piece);
                i = j;
                matched = true;
                break;
            }
        }
        if (!matched) {
            out += has_letter_sound(w[i]) ? letter_sound(w[i]) : u32();
            ++i;
        }
    }
    const u32 r = stress_first_vowel(out);
    return british_ ? british(r, w) : r;
}

u32 G2p::phonemes(const u32& sentence) const {
    const std::vector<u32> tokens = sh_.rule("token").match_all(fold(sentence));
    std::vector<u32> out;
    bool skip = false;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const u32& tok = tokens[i];
        if (const auto mark = punct(tok)) {
            if (!out.empty()) out.back() += *mark;
            continue;
        }
        if (skip) { skip = false; continue; }
        // `letterRun`: two or more one-letter capitals in a row (and an 's after them) are one spelled name.
        {
            u32 letters_run;
            size_t k = i;
            while (k < tokens.size() && single_capital(tokens[k])) letters_run += tokens[k++];
            const bool plural = k < tokens.size() && capital_plural(tokens[k]);
            if (plural) letters_run += tokens[k++][0];
            if (letters_run.size() >= 2) {
                const u32 ipa = spell(letters_run);
                out.push_back(plural ? ipa + plural_suffix(ipa) : ipa);
                i = k - 1;
                continue;
            }
        }
        std::optional<u32> next;
        if (i + 1 < tokens.size() && !punct(tokens[i + 1])) next = tokens[i + 1];
        if (next) {
            if (const auto pair = lex(sh_.to_lower(norm(tok)) + U" " + sh_.to_lower(norm(*next)))) {
                out.push_back(*pair);
                skip = true;
                continue;
            }
        }
        if (const auto reading = homograph(tok, tokens, i)) {
            out.push_back(*reading);
            continue;
        }
        const auto ipa = word(tok, next);
        if (ipa && !ipa->empty()) out.push_back(*ipa);
    }
    u32 joined_out;
    for (size_t i = 0; i < out.size(); ++i) {
        if (i) joined_out += U' ';
        joined_out += out[i];
    }
    return joined_out;
}

// ------------------------------------------------------------------------------------------------ toBritish
namespace {

struct Unit {
    u32 t;
    bool v = false;
    u32 s;
    std::optional<u32> g;
    std::optional<size_t> at;
    size_t len = 0;
};

const std::vector<u32>& kUnits() {
    static const std::vector<u32> u = {U"aɪə", U"aʊə", U"eɪ", U"aɪ", U"aʊ", U"ɔɪ", U"oʊ", U"əʊ", U"ɪə", U"ɛə", U"ʊə", U"ɑː",
                                       U"ɔː", U"ɜː", U"iː", U"uː", U"oː", U"tʃ", U"dʒ", U"n̩", U"l̩", U"m̩"};
    return u;
}
bool unit_vowel(const u32& t) {
    if (t == U"n̩" || t == U"l̩" || t == U"m̩") return true;
    static const u32 vowel_ch = U"aeiouɑɐɒæɔəɛɜɪʊʌᵻɚɝ";
    return !t.empty() && vowel_ch.find(t[0]) != u32::npos;
}

std::vector<Unit> units(const u32& ipa) {
    std::vector<Unit> out;
    u32 stress;
    for (size_t i = 0; i < ipa.size();) {
        const char32_t ch = ipa[i];
        if (ch == U'ˈ' || ch == U'ˌ') { stress = u32(1, ch); ++i; continue; }
        u32 t(1, ch);
        for (const u32& u : kUnits()) if (starts_at(ipa, i, u)) { t = u; break; }
        i += t.size();
        const bool v = unit_vowel(t);
        out.push_back({t, v, v ? stress : u32()});
        if (!v && !stress.empty()) out.back().s = stress;
        stress.clear();
    }
    return out;
}

u32 join_units(const std::vector<Unit>& us) {
    u32 out;
    for (const auto& u : us) out += u.s + u.t;
    return out;
}

bool in(const u32& x, std::initializer_list<const char32_t*> list) {
    for (const char32_t* l : list) if (x == l) return true;
    return false;
}

const std::set<u32>& kBath() {
    static const std::set<u32> s = [] {
        std::set<u32> out;
        const u32 list = U"after afternoon afterwards answer ask aunt banana basket bath bathroom blast branch brass can't cask cast "
                         U"castle caste chance chancellor chant clasp class classroom command commander craft crafty dance dancer demand disaster draft "
                         U"draught example fast fasten flask gasp ghastly giraffe glance glass graft grant grasp grass half calf laugh laughter last "
                         U"mask master nasty pass passport past pastor pastoral path plant plaster prance raft rafter rasp rather shaft slant staff "
                         U"task vast advance advantage disadvantage enhance photograph telegraph paragraph autograph graph sample command "
                         U"commando contrast forecast broadcast outcast downcast overcast aghast mast lance france bypass surpass trespass";
        u32 cur;
        for (char32_t c : list) {
            if (c == U' ') { out.insert(cur); cur.clear(); }
            else cur += c;
        }
        out.insert(cur);
        return out;
    }();
    return s;
}

bool is_bath(const u32& word) {
    if (kBath().count(word)) return true;
    for (const char32_t* e_c : {U"", U"s", U"es", U"ed", U"ing", U"er", U"ers", U"y", U"ier", U"iest", U"ly", U"ful", U"room",
                                U"rooms", U"hopper", U"hoppers", U"land", U"lands", U"work", U"time", U"word", U"words", U"ship"}) {
        const u32 e = e_c;
        if (!e.empty() && ends_with(word, e) && kBath().count(word.substr(0, word.size() - e.size()))) return true;
    }
    for (const char32_t* e_c : {U"ed", U"ing", U"er", U"es"}) {
        const u32 e = e_c;
        if (ends_with(word, e) && kBath().count(word.substr(0, word.size() - e.size()) + U"e")) return true;
    }
    return false;
}

} // namespace

u32 G2p::british(const u32& ipa, const u32& key) const {
    if (ipa.empty()) return ipa;
    const bool linking = ends_with(key, U"#v");
    u32 word = sh_.to_lower(linking ? key.substr(0, key.size() - 2) : key);
    for (char32_t& c : word) if (c == U'’') c = U'\'';
    u32 letters_s;
    for (char32_t c : word) if ((c >= U'a' && c <= U'z') || c == U'\'' || c == U' ') letters_s += c;
    std::vector<Unit> u = units(ipa);
    std::vector<size_t> vi;
    for (size_t i = 0; i < u.size(); ++i) if (u[i].v) vi.push_back(i);
    u32 plain;
    for (char32_t c : letters_s) if (c != U'\'') plain += c;
    // `spellings`: each vowel's letters when the word's vowel-letter groups line up one to one.
    {
        struct G { size_t at; u32 g; };
        std::vector<G> groups;
        const JsRegex& gr = sh_.rule("gb_groups");
        size_t pos = 0;
        JsRegex::Match m;
        while (pos <= plain.size() && gr.search(plain, pos, m)) {
            groups.push_back({m.begin, plain.substr(m.begin, m.end - m.begin)});
            pos = m.end == m.begin ? m.begin + 1 : m.end;
        }
        std::vector<std::vector<G>> cands = {groups};
        if (!groups.empty() && sh_.rule("gb_silent_e").test(plain) && groups.back().g == U"e") {
            cands.emplace_back(groups.begin(), groups.end() - 1);
        }
        const std::vector<G>* sp = nullptr;
        for (const auto& c : cands) if (c.size() == vi.size()) { sp = &c; break; }
        if (sp) {
            for (size_t k = 0; k < vi.size(); ++k) {
                Unit& x = u[vi[k]];
                const size_t after = (*sp)[k].at + (*sp)[k].g.size();
                x.g = (*sp)[k].g + (after < plain.size() && plain[after] == U'w' ? U"w" : U"");
                x.at = (*sp)[k].at;
                x.len = (*sp)[k].g.size();
            }
        }
    }
    const bool has_t = letters_s.find(U't') != u32::npos, has_d = letters_s.find(U'd') != u32::npos;
    auto vowel_before = [&](size_t i) -> const Unit* {
        for (size_t j = i; j-- > 0;) if (u[j].v) return &u[j];
        return nullptr;
    };
    auto vowel_after = [&](size_t i) -> const Unit* {
        for (size_t j = i + 1; j < u.size(); ++j) if (u[j].v) return &u[j];
        return nullptr;
    };
    for (size_t i = 0; i < u.size(); ++i) {
        Unit& x = u[i];
        const Unit* prev = i > 0 ? &u[i - 1] : nullptr;
        const Unit* next = i + 1 < u.size() ? &u[i + 1] : nullptr;
        const Unit* after = i + 2 < u.size() ? &u[i + 2] : nullptr;
        const std::optional<u32> g = x.g;
        const bool g_truthy = g && !g->empty();
        if (x.t == U"ɾ") {
            bool d = has_d && !has_t;
            const Unit* a = vowel_before(i);
            const Unit* b = vowel_after(i);
            if (a && b && a->at && b->at) {
                const size_t from = *a->at + a->len;
                const u32 mid = from < *b->at ? plain.substr(from, *b->at - from) : u32();
                d = mid.find(U't') == u32::npos && mid.find(U'd') != u32::npos;
            }
            x.t = d ? U"d" : U"t";
        } else if (x.t == U"ᵻ") x.t = U"ɪ";
        else if (x.t == U"oʊ") x.t = U"əʊ";
        else if (x.t == U"ɝ") x.t = U"ɜː";
        else if (x.t == U"ɔ") x.t = U"ɒ";
        else if (x.t == U"ɑː" || x.t == U"ɑ") {
            const bool r_cons = next && next->t == U"ɹ" && !(after && after->v);
            if (!r_cons) {
                const bool lot = g_truthy ? (g->find(U'o') != u32::npos || (*g == U"a" && prev && prev->t == U"w"))
                                          : (letters_s.find(U'o') != u32::npos && letters_s.find(U'a') == u32::npos);
                x.t = lot ? U"ɒ" : U"ɑː";
            }
        } else if (x.t == U"æ" && !x.s.empty() && is_bath(word) && next && u32(U"fsθnmlŋ").find(next->t) != u32::npos) {
            x.t = U"ɑː";
        } else if ((x.t == U"uː" || ((x.t == U"ʊ" || x.t == U"ʊə") && next && next->t == U"ɹ")) && !x.s.empty() && prev &&
                   in(prev->t, {U"t", U"d", U"n", U"θ"}) && g_truthy && in(*g, {U"u", U"ue", U"ew", U"eu", U"ui", U"ewe"}) &&
                   !(i >= 2 && u[i - 2].t == U"j")) {
            u.insert(u.begin() + static_cast<long>(i), Unit{U"j", false, u32()});
            ++i;
        }
    }
    if (sh_.rule("gb_ary").test(word)) {
        for (size_t i = u.size() >= 3 ? u.size() - 3 + 1 : 0; i-- > 0;) {
            if (u[i].v && u[i].s != U"ˈ" && in(u[i].t, {U"ɛ", U"ɔː", U"oː", U"eɪ"}) && i + 1 < u.size() && u[i + 1].t == U"ɹ" &&
                i + 2 < u.size() && in(u[i + 2].t, {U"i", U"iː", U"ɪ"})) {
                u[i] = Unit{U"ə", true, u32()};
                break;
            }
        }
    }
    if (sh_.rule("gb_ile").test(word) && !u.empty()) {
        size_t k = u.size() - 1;
        if (k > 0 && in(u[k].t, {U"z", U"d"})) --k;
        if (u[k].t == U"l̩" && u[k].s.empty()) {
            u.erase(u.begin() + static_cast<long>(k));
            u.insert(u.begin() + static_cast<long>(k), {Unit{U"aɪ", true, u32()}, Unit{U"l", false, u32()}});
        } else if (u[k].t == U"l" && k > 0 && in(u[k - 1].t, {U"ə", U"ɪ", U"ʌ"}) && u[k - 1].s.empty()) {
            u[k - 1].t = U"aɪ";
        }
    }
    if (sh_.rule("gb_isation").test(word)) {
        for (size_t i = 1; i + 2 < u.size(); ++i) {
            if (in(u[i].t, {U"ə", U"ɪ"}) && u[i].s.empty() && u[i + 1].t == U"z" && u[i + 2].t == U"eɪ") u[i].t = U"aɪ";
        }
    }
    std::vector<Unit> out;
    for (size_t i = 0; i < u.size(); ++i) {
        const Unit& x = u[i];
        const Unit* next = i + 1 < u.size() ? &u[i + 1] : nullptr;
        if (x.t == U"ɚ") {
            const bool keep = next && next->t == U"ɹ" && ((i + 2 < u.size() && u[i + 2].v) || (linking && i + 2 == u.size()));
            Unit schwa = x;
            schwa.t = U"ə";
            out.push_back(schwa);
            if (keep) { out.push_back(*next); ++i; }
            else if (next && next->t == U"ɹ" && !(i + 2 < u.size() && u[i + 2].v)) ++i;
            continue;
        }
        if (x.t == U"ɹ" && !out.empty() && out.back().v) {
            const Unit* nx = next;
            Unit& p = out.back();
            const bool before_vowel = (nx && nx->v) || (linking && !nx);
            if (before_vowel) {
                if (p.t == U"oː") p.t = U"ɔː";
                const bool rr = p.at && plain.substr(std::min(plain.size(), *p.at + p.len), 2) == U"rr";
                const bool pg = p.g.has_value();
                if (nx && p.t == U"ɛ" && pg && *p.g == U"a" && rr) p.t = U"æ";
                else if (nx && p.t == U"ɛ" && pg && in(*p.g, {U"ai", U"ae", U"ea", U"ei"})) p.t = U"ɛə";
                else if (nx && p.t == U"ɪ" && pg && in(*p.g, {U"e", U"ea", U"ee", U"ie", U"eo", U"ei"}) && !rr) p.t = U"ɪə";
                else if (nx && p.t == U"ʊ" && pg && in(*p.g, {U"u", U"ou", U"oo"}) && !rr) p.t = U"ʊə";
                if (p.t == U"ɜː" && sh_.rule("gb_strut_r").test(word) && !sh_.rule("gb_nurse_r").test(word)) p.t = U"ʌ";
                if (!nx) {
                    static const std::map<u32, u32> link = {{U"ɑː", U"ɑː"}, {U"ɔː", U"ɔː"}, {U"ɛ", U"ɛə"}, {U"ɪ", U"ɪə"},
                                                            {U"ʊ", U"ʊə"}, {U"iː", U"ɪə"}, {U"uː", U"ʊə"}, {U"eɪ", U"ɛə"},
                                                            {U"aɪ", U"aɪə"}, {U"aʊ", U"aʊə"}};
                    const auto it = link.find(p.t);
                    if (it != link.end()) p.t = it->second;
                }
                out.push_back(x);
                continue;
            }
            static const std::map<u32, u32> drop = {{U"ɑː", U"ɑː"}, {U"ɑ", U"ɑː"}, {U"ɔː", U"ɔː"}, {U"oː", U"ɔː"}, {U"ɒ", U"ɔː"},
                                                    {U"ɛ", U"ɛə"}, {U"eɪ", U"ɛə"}, {U"ɪ", U"ɪə"}, {U"iː", U"ɪə"}, {U"ʊ", U"ʊə"},
                                                    {U"uː", U"ʊə"}, {U"aɪ", U"aɪə"}, {U"aʊ", U"aʊə"}, {U"ɜː", U"ɜː"}, {U"ə", U"ə"},
                                                    {U"ʌ", U"ɜː"}, {U"ɐ", U"ə"}, {U"æ", U"ɑː"}};
            const auto it = drop.find(p.t);
            if (it != drop.end()) p.t = it->second;
            continue;
        }
        out.push_back(x);
    }
    return join_units(out);
}

} // namespace kale
} // namespace loom
