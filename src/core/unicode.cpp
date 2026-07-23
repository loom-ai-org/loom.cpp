#include "loom/core/unicode.h"
#include "loom/core/unicode_data.h"

#include <algorithm>
#include <unordered_map>

namespace loom {
namespace {

using namespace unicode_data;

// --- Hangul algorithmic decomposition/composition (UAX #15 section 16) -- deliberately not table-driven
// (see unicode_data.h's generation-time note: UnicodeData.txt omits these ~11172 mappings on purpose). ---
constexpr char32_t kSBase = 0xAC00;
constexpr char32_t kLBase = 0x1100;
constexpr char32_t kVBase = 0x1161;
constexpr char32_t kTBase = 0x11A7;
constexpr int kLCount = 19;
constexpr int kVCount = 21;
constexpr int kTCount = 28;
constexpr int kNCount = kVCount * kTCount;   // 588
constexpr int kSCount = kLCount * kNCount;   // 11172

bool is_hangul_syllable(char32_t cp) { return cp >= kSBase && cp < kSBase + kSCount; }

void hangul_decompose_append(char32_t cp, std::vector<char32_t>& out) {
    const int s_index = static_cast<int>(cp - kSBase);
    const char32_t l = kLBase + static_cast<char32_t>(s_index / kNCount);
    const char32_t v = kVBase + static_cast<char32_t>((s_index % kNCount) / kTCount);
    const char32_t t = kTBase + static_cast<char32_t>(s_index % kTCount);
    out.push_back(l);
    out.push_back(v);
    if (t != kTBase) out.push_back(t);
}

// Returns true and sets `out` if (a, b) is a valid Hangul L+V or LV+T composition pair.
bool hangul_compose_pair(char32_t a, char32_t b, char32_t& out) {
    if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount) {
        const int l_index = static_cast<int>(a - kLBase);
        const int v_index = static_cast<int>(b - kVBase);
        out = kSBase + static_cast<char32_t>((l_index * kVCount + v_index) * kTCount);
        return true;
    }
    if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 && b > kTBase && b < kTBase + kTCount) {
        out = a + (b - kTBase);
        return true;
    }
    return false;
}

// --- Generated-table lookups (binary search -- every table in unicode_data.h is sorted by codepoint) ---

uint8_t combining_class(char32_t cp) {
    const auto it = std::lower_bound(std::begin(kCombiningClass), std::end(kCombiningClass), cp,
                                      [](const CpEntry& e, char32_t c) { return e.cp < c; });
    if (it != std::end(kCombiningClass) && it->cp == cp) return it->value;
    return 0;
}

bool is_composition_excluded(char32_t cp) {
    return std::binary_search(std::begin(kCompositionExclusions), std::end(kCompositionExclusions), cp);
}

const CpDecompEntry* find_canonical_decomp(char32_t cp) {
    const auto it = std::lower_bound(std::begin(kCanonicalDecomp), std::end(kCanonicalDecomp), cp,
                                      [](const CpDecompEntry& e, char32_t c) { return e.cp < c; });
    if (it != std::end(kCanonicalDecomp) && it->cp == cp) return &*it;
    return nullptr;
}

// Recursively expands `cp` to its fully (maximally) canonically decomposed form, appending to `out`.
// Hangul syllables decompose algorithmically to jamo, which never further decompose; other codepoints
// consult the generated one-step mapping and recurse on its results (UnicodeData.txt lists only a single
// decomposition step per character -- full decomposition requires following the chain to a fixed point).
void decompose_recursive(char32_t cp, std::vector<char32_t>& out) {
    if (is_hangul_syllable(cp)) {
        hangul_decompose_append(cp, out);
        return;
    }
    const CpDecompEntry* entry = find_canonical_decomp(cp);
    if (!entry) {
        out.push_back(cp);
        return;
    }
    for (uint8_t i = 0; i < entry->len; ++i) {
        decompose_recursive(kCanonicalDecompSeq[entry->offset + i], out);
    }
}

// Canonical Ordering Algorithm (UAX #15): stably reorders runs of combining marks by non-decreasing
// combining class, never moving a mark across a starter (ccc==0) boundary. Bubble-sort restricted to
// adjacent pairs where the right element has non-zero class is exactly that invariant (a ccc==0 element
// on the right blocks any swap; a ccc==0 element on the left can never satisfy ccc(i) > ccc(i+1)).
void canonical_order(std::vector<char32_t>& cps) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i + 1 < cps.size(); ++i) {
            const uint8_t c1 = combining_class(cps[i]);
            const uint8_t c2 = combining_class(cps[i + 1]);
            if (c2 != 0 && c1 > c2) {
                std::swap(cps[i], cps[i + 1]);
                changed = true;
            }
        }
    }
}

// Reverse map of every one-step canonical decomposition of length exactly 2 (the only entries valid as
// primary composition pairs per UAX #15 -- singleton and length>2 decompositions never recompose
// directly). Canonical decomposition mappings are injective by Unicode design, so this reversal is safe.
const std::unordered_map<uint64_t, char32_t>& pairwise_composition_map() {
    static const std::unordered_map<uint64_t, char32_t> map = [] {
        std::unordered_map<uint64_t, char32_t> m;
        for (const CpDecompEntry& e : kCanonicalDecomp) {
            if (e.len != 2) continue;
            const char32_t a = kCanonicalDecompSeq[e.offset];
            const char32_t b = kCanonicalDecompSeq[e.offset + 1];
            const uint64_t key = (static_cast<uint64_t>(a) << 32) | b;
            m.emplace(key, e.cp);
        }
        return m;
    }();
    return map;
}

bool try_compose_pair(char32_t a, char32_t b, char32_t& out) {
    if (hangul_compose_pair(a, b, out)) return true;
    const auto& map = pairwise_composition_map();
    const uint64_t key = (static_cast<uint64_t>(a) << 32) | b;
    const auto it = map.find(key);
    if (it != map.end()) {
        out = it->second;
        return true;
    }
    return false;
}

// UAX #15's reference composition algorithm: greedily composes each character into the current
// "starter" unless blocked by an intervening character of combining class >= its own (tracked via
// `last_class`, reset to 0 whenever a new starter is emitted, left untouched when a mark composes away
// since it's removed from the sequence entirely).
std::vector<char32_t> canonical_compose(const std::vector<char32_t>& src) {
    if (src.empty()) return {};
    std::vector<char32_t> result;
    result.push_back(src[0]);
    size_t starter_idx = 0;
    int last_class = combining_class(src[0]);
    for (size_t i = 1; i < src.size(); ++i) {
        const char32_t c = src[i];
        const int c_class = combining_class(c);
        const bool blocked = last_class != 0 && last_class >= c_class;
        char32_t composite;
        if (!blocked && try_compose_pair(result[starter_idx], c, composite) && !is_composition_excluded(composite)) {
            result[starter_idx] = composite;
            continue; // c is absorbed; last_class (the gap before c) is unchanged
        }
        result.push_back(c);
        if (c_class == 0) {
            starter_idx = result.size() - 1;
            last_class = 0;
        } else {
            last_class = c_class;
        }
    }
    return result;
}

} // namespace

std::vector<char32_t> utf8_decode(const std::string& text) {
    std::vector<char32_t> out;
    out.reserve(text.size());
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        const unsigned char b0 = static_cast<unsigned char>(text[i]);
        int extra = 0;
        char32_t cp;
        if (b0 < 0x80) {
            cp = b0;
            extra = 0;
        } else if ((b0 & 0xE0) == 0xC0) {
            cp = b0 & 0x1F;
            extra = 1;
        } else if ((b0 & 0xF0) == 0xE0) {
            cp = b0 & 0x0F;
            extra = 2;
        } else if ((b0 & 0xF8) == 0xF0) {
            cp = b0 & 0x07;
            extra = 3;
        } else {
            out.push_back(b0); // invalid lead byte -- lenient passthrough
            ++i;
            continue;
        }
        bool valid = true;
        char32_t acc = cp;
        for (int k = 1; k <= extra; ++k) {
            if (i + static_cast<size_t>(k) >= n) { valid = false; break; }
            const unsigned char bk = static_cast<unsigned char>(text[i + static_cast<size_t>(k)]);
            if ((bk & 0xC0) != 0x80) { valid = false; break; }
            acc = (acc << 6) | (bk & 0x3F);
        }
        if (!valid) {
            out.push_back(b0);
            ++i;
            continue;
        }
        out.push_back(acc);
        i += static_cast<size_t>(extra) + 1;
    }
    return out;
}

std::string utf8_encode(const std::vector<char32_t>& codepoints) {
    std::string out;
    out.reserve(codepoints.size());
    for (char32_t cp : codepoints) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return out;
}

bool in_ranges(char32_t cp, const CpRange* ranges, size_t count) {
    const auto it = std::upper_bound(ranges, ranges + count, cp,
                                      [](char32_t c, const CpRange& r) { return c < r.lo; });
    if (it == ranges) return false;
    const auto& r = *(it - 1);
    return cp >= r.lo && cp <= r.hi;
}

bool is_letter(char32_t cp) { return in_ranges(cp, kLetterRanges, kLetterRangesCount); }
bool is_number(char32_t cp) { return in_ranges(cp, kNumberRanges, kNumberRangesCount); }
bool is_letter_or_number(char32_t cp) { return is_letter(cp) || is_number(cp); }
bool is_punctuation(char32_t cp) { return in_ranges(cp, kPunctuationRanges, kPunctuationRangesCount); }
bool is_mark(char32_t cp) { return in_ranges(cp, kMarkRanges, kMarkRangesCount); }

char32_t to_lower(char32_t cp) {
    const auto it = std::lower_bound(std::begin(kLowercaseMap), std::end(kLowercaseMap), cp,
                                      [](const CpMapEntry& e, char32_t c) { return e.cp < c; });
    if (it != std::end(kLowercaseMap) && it->cp == cp) return it->value;
    return cp;
}

std::string nfc_normalize(const std::string& text) {
    const std::vector<char32_t> input = utf8_decode(text);
    std::vector<char32_t> decomposed;
    decomposed.reserve(input.size() * 2);
    for (char32_t cp : input) decompose_recursive(cp, decomposed);
    canonical_order(decomposed);
    const std::vector<char32_t> composed = canonical_compose(decomposed);
    return utf8_encode(composed);
}

std::string nfd_normalize(const std::string& text) {
    const std::vector<char32_t> input = utf8_decode(text);
    std::vector<char32_t> decomposed;
    decomposed.reserve(input.size() * 2);
    for (char32_t cp : input) decompose_recursive(cp, decomposed);
    canonical_order(decomposed);
    return utf8_encode(decomposed);
}

} // namespace loom
