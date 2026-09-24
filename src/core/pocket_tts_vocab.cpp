#include "loom/core/pocket_tts_vocab.h"

#include "loom/core/gguf_model.h"
#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

namespace loom {
namespace {

const std::string kPrefix = "tokenizer.ggml.pocket_tts.";

std::vector<std::string> required_strings(const GgufModel& model, const std::string& key) {
    if (!model.has_kv(key)) {
        throw LoadError("PocketTtsVocab::load: tokenizer.ggml.model is 'pocket_tts' but " + key + " is missing");
    }
    return model.kv_arr_str(key);
}

// A table of single characters, one per entry, as a set of codepoints. Each entry must be exactly one
// codepoint: the reference tests membership with `in` on a string of them.
std::unordered_set<char32_t> required_chars(const GgufModel& model, const std::string& key) {
    std::unordered_set<char32_t> out;
    for (const std::string& entry : required_strings(model, key)) {
        const std::vector<char32_t> cps = utf8_decode(entry);
        if (cps.size() != 1) {
            throw LoadError("PocketTtsVocab::load: " + key + " entry '" + entry + "' is not one character");
        }
        out.insert(cps[0]);
    }
    return out;
}

// Python's `str.replace(from, to)`: every non-overlapping occurrence, left to right, in ONE pass -- so
// "   " -> " " leaves two spaces, which the reference relies on nothing to fix.
void replace_all(std::string& text, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    std::string out;
    size_t pos = 0;
    while (true) {
        const size_t hit = text.find(from, pos);
        if (hit == std::string::npos) break;
        out.append(text, pos, hit - pos);
        out += to;
        pos = hit + from.size();
    }
    out.append(text, pos, std::string::npos);
    text.swap(out);
}

// `str.strip()`: Python whitespace from both ends.
std::vector<char32_t> python_strip(const std::vector<char32_t>& cps) {
    size_t begin = 0;
    size_t end = cps.size();
    while (begin < end && is_python_space(cps[begin])) ++begin;
    while (end > begin && is_python_space(cps[end - 1])) --end;
    return {cps.begin() + static_cast<std::ptrdiff_t>(begin), cps.begin() + static_cast<std::ptrdiff_t>(end)};
}

std::string python_strip(const std::string& text) { return utf8_encode(python_strip(utf8_decode(text))); }

} // namespace

std::unique_ptr<PocketTtsVocab> PocketTtsVocab::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.ggml.model") || model.kv_str("tokenizer.ggml.model") != "pocket_tts") {
        return nullptr;
    }
    std::unique_ptr<PocketTtsVocab> v(new PocketTtsVocab());
    v->vocab_ = Vocab::load_sentencepiece(model, /*is_bpe=*/false);
    v->replace_from_ = required_strings(model, kPrefix + "replace_from");
    v->replace_to_ = required_strings(model, kPrefix + "replace_to");
    if (v->replace_from_.size() != v->replace_to_.size()) {
        throw LoadError("PocketTtsVocab::load: replace_from and replace_to differ in length -- they are one "
                         "table of pairs");
    }
    v->terminal_ = required_chars(model, kPrefix + "terminal");
    v->weak_ = required_chars(model, kPrefix + "weak");
    v->closers_ = required_chars(model, kPrefix + "closers");
    v->digits_ = required_chars(model, kPrefix + "digits");
    const auto full_stop = required_strings(model, kPrefix + "full_stop");
    if (full_stop.size() != 1) {
        throw LoadError("PocketTtsVocab::load: " + kPrefix + "full_stop must hold one string");
    }
    v->full_stop_ = full_stop[0];
    const auto upper_from = required_strings(model, kPrefix + "upper_from");
    const auto upper_to = required_strings(model, kPrefix + "upper_to");
    if (upper_from.size() != upper_to.size()) {
        throw LoadError("PocketTtsVocab::load: upper_from and upper_to differ in length");
    }
    for (size_t i = 0; i < upper_from.size(); ++i) {
        const std::vector<char32_t> cps = utf8_decode(upper_from[i]);
        if (cps.size() != 1) {
            throw LoadError("PocketTtsVocab::load: upper_from entry '" + upper_from[i] + "' is not one character");
        }
        v->upper_.emplace(cps[0], upper_to[i]);
    }
    for (int32_t id : model.kv_arr_i32(kPrefix + "sentence_end_ids")) v->sentence_end_ids_.insert(id);
    for (int32_t id : model.kv_arr_i32(kPrefix + "clause_end_ids")) v->clause_end_ids_.insert(id);
    const int32_t max_tokens = model.kv_i32(kPrefix + "max_tokens_per_chunk", 0);
    if (max_tokens <= 0) {
        throw LoadError("PocketTtsVocab::load: " + kPrefix + "max_tokens_per_chunk is missing or not positive");
    }
    v->max_tokens_per_chunk_ = static_cast<size_t>(max_tokens);
    v->separator_ = model.kv_i32(kPrefix + "chunk_separator", -1);
    if (v->separator_ < 0 || static_cast<size_t>(v->separator_) >= v->vocab_->size()) {
        throw LoadError("PocketTtsVocab::load: " + kPrefix + "chunk_separator is missing or not an id");
    }
    v->capitalize_first_letter_ = model.kv_bool(kPrefix + "capitalize_first_letter", true);
    v->append_terminal_punctuation_ = model.kv_bool(kPrefix + "append_terminal_punctuation", true);
    return v;
}

std::string PocketTtsVocab::terminate(const std::string& text) const {
    // `_ensure_terminal_punctuation`. The core is the text with trailing closers and ASCII spaces
    // stripped (`rstrip(_CLOSERS + " ")`: a character set, not a suffix); what was stripped is kept,
    // itself stripped of whitespace, and re-attached after any full stop added.
    const std::vector<char32_t> cps = utf8_decode(text);
    size_t core_end = cps.size();
    while (core_end > 0 && (closers_.count(cps[core_end - 1]) != 0 || cps[core_end - 1] == U' ')) --core_end;
    if (core_end == 0 || terminal_.count(cps[core_end - 1]) != 0) return text;
    const std::vector<char32_t> tail(cps.begin() + static_cast<std::ptrdiff_t>(core_end), cps.end());
    const std::string closers = utf8_encode(python_strip(tail));
    if (weak_.count(cps[core_end - 1]) != 0) {
        size_t end = core_end;
        while (end > 0 && (weak_.count(cps[end - 1]) != 0 || cps[end - 1] == U' ')) --end;
        const std::vector<char32_t> core(cps.begin(), cps.begin() + static_cast<std::ptrdiff_t>(end));
        return utf8_encode(core) + full_stop_ + closers;
    }
    return text + full_stop_;
}

std::string PocketTtsVocab::prepare(const std::string& text) const {
    std::string out = python_strip(text);
    if (out.empty()) {
        throw Error("PocketTtsVocab: the text is empty (the reference raises 'Text prompt cannot be empty')");
    }
    for (size_t i = 0; i < replace_from_.size(); ++i) replace_all(out, replace_from_[i], replace_to_[i]);
    if (capitalize_first_letter_) {
        // `text[0].upper() + text[1:]` unless `text[0].isupper()`; the table holds exactly the
        // codepoints for which that changes anything.
        const std::vector<char32_t> cps = utf8_decode(out);
        const auto it = cps.empty() ? upper_.end() : upper_.find(cps[0]);
        if (it != upper_.end()) {
            out = it->second + utf8_encode(std::vector<char32_t>(cps.begin() + 1, cps.end()));
        }
    }
    if (append_terminal_punctuation_) out = terminate(out);
    return out;
}

bool PocketTtsVocab::decimal_period_at(const std::vector<int32_t>& ids, size_t start) const {
    // `_is_decimal_period_boundary`: judged on DECODED text, `prefix[-2:] == digit + "."` and the
    // suffix opening with a digit -- which is why the chunking cannot live in the driver.
    const std::vector<char32_t> prefix =
        utf8_decode(vocab_->decode(std::vector<int32_t>(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(start))));
    const std::vector<char32_t> suffix =
        utf8_decode(vocab_->decode(std::vector<int32_t>(ids.begin() + static_cast<std::ptrdiff_t>(start), ids.end())));
    return prefix.size() >= 2 && prefix.back() == U'.' && digits_.count(prefix[prefix.size() - 2]) != 0 &&
           !suffix.empty() && digits_.count(suffix.front()) != 0;
}

std::vector<size_t> PocketTtsVocab::boundaries(const std::vector<int32_t>& ids, const std::unordered_set<int32_t>& marks,
                                               bool skip_decimal_periods) const {
    // `_find_boundary_indices`: a cut before the first non-mark id after a run of marks.
    std::vector<size_t> out{0};
    bool previous_was_boundary = false;
    for (size_t idx = 0; idx < ids.size(); ++idx) {
        if (marks.count(ids[idx]) != 0) {
            previous_was_boundary = true;
            continue;
        }
        if (previous_was_boundary && !(skip_decimal_periods && decimal_period_at(ids, idx))) out.push_back(idx);
        previous_was_boundary = false;
    }
    out.push_back(ids.size());
    return out;
}

std::vector<std::string> PocketTtsVocab::chunks(const std::string& text) const {
    // `split_into_best_sentences`.
    const std::vector<int32_t> ids = vocab_->encode(python_strip(prepare(text)));
    struct Segment {
        size_t n_tokens;
        std::string text;
    };
    const auto segments = [this](const std::vector<int32_t>& seq, const std::vector<size_t>& cuts) {
        std::vector<Segment> out;
        for (size_t i = 0; i + 1 < cuts.size(); ++i) {
            out.push_back({cuts[i + 1] - cuts[i],
                           vocab_->decode(std::vector<int32_t>(seq.begin() + static_cast<std::ptrdiff_t>(cuts[i]),
                                                               seq.begin() + static_cast<std::ptrdiff_t>(cuts[i + 1])))});
        }
        return out;
    };

    std::vector<Segment> refined;
    for (Segment& sentence : segments(ids, boundaries(ids, sentence_end_ids_, /*skip_decimal_periods=*/true))) {
        if (sentence.n_tokens <= max_tokens_per_chunk_) {
            refined.push_back(std::move(sentence));
            continue;
        }
        // An oversized sentence is re-tokenized on its own and cut on clause marks; kept whole (with its
        // ORIGINAL count) when that finds nothing to cut.
        const std::vector<int32_t> sub = vocab_->encode(python_strip(sentence.text));
        std::vector<Segment> parts = segments(sub, boundaries(sub, clause_end_ids_, /*skip_decimal_periods=*/false));
        if (parts.size() > 1) {
            for (Segment& part : parts) refined.push_back(std::move(part));
        } else {
            refined.push_back(std::move(sentence));
        }
    }

    std::vector<std::string> out;
    std::string current;
    size_t current_tokens = 0;
    for (const Segment& sentence : refined) {
        if (current.empty()) {
            current = sentence.text;
            current_tokens = sentence.n_tokens;
            continue;
        }
        if (current_tokens + sentence.n_tokens > max_tokens_per_chunk_) {
            out.push_back(python_strip(current));
            current = sentence.text;
            current_tokens = sentence.n_tokens;
        } else {
            current += " " + sentence.text;
            current_tokens += sentence.n_tokens;
        }
    }
    if (!current.empty()) out.push_back(python_strip(current));
    return out;
}

std::vector<int32_t> PocketTtsVocab::encode(const std::string& text) const {
    std::vector<int32_t> out;
    for (const std::string& chunk : chunks(text)) {
        if (!out.empty()) out.push_back(separator_);
        const std::vector<int32_t> ids = vocab_->encode(prepare(chunk));
        out.insert(out.end(), ids.begin(), ids.end());
    }
    return out;
}

std::string PocketTtsVocab::decode(const std::vector<int32_t>& ids) const {
    std::vector<int32_t> kept;
    kept.reserve(ids.size());
    for (int32_t id : ids) {
        if (id != separator_) kept.push_back(id);
    }
    return vocab_->decode(kept);
}

} // namespace loom
