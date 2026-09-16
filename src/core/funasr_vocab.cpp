#include "loom/core/funasr_vocab.h"
#include "loom/core/gguf_model.h"
#include "loom/loom_errors.h"

#include <cctype>
#include <string>

namespace loom {
namespace {

// llama.cpp's `llama_token_type`, which this schema shares: 1 NORMAL, 3 CONTROL. Only CONTROL matters
// here, and it is what the exporter marks `<blank>`/`<s>`/`</s>`/`<unk>`/`<OOV>` with.
constexpr int32_t TOKEN_TYPE_CONTROL = 3;

bool ascii_alpha(const std::string& s) {
    // `abbr_dispose`'s own test is `words[num].encode("utf-8").isalpha()` -- on BYTES, so it is ASCII
    // only however Unicode-aware the rest of the reference is. A single CJK character is three bytes
    // and never triggers it, which is why "中文BBC" keeps its Chinese and upper-cases only the run.
    return s.size() == 1 && (std::isalpha(static_cast<unsigned char>(s[0])) != 0);
}

// `abbr_dispose`: a run of single ASCII letters separated by spaces becomes one upper-cased word.
// "b b c news" -> "BBC news". Transcribed rather than reimplemented, including the way the reference's
// inner cursor advances independently of its loop variable.
std::vector<std::string> abbr_dispose(const std::vector<std::string>& words) {
    const int64_t n = static_cast<int64_t>(words.size());
    std::vector<int64_t> begins, ends;
    int64_t last = -1;
    for (int64_t start = 0; start < n; ++start) {
        if (start <= last) continue;
        if (!ascii_alpha(words[start])) continue;
        if (!(start + 1 < n && words[start + 1] == " " && start + 2 < n && ascii_alpha(words[start + 2]))) {
            continue;
        }
        int64_t cursor = start + 2;
        begins.push_back(start);
        ends.push_back(cursor);
        while (true) {
            ++cursor;
            if (cursor < n && words[cursor] == " ") {
                ++cursor;
                if (cursor < n && ascii_alpha(words[cursor])) {
                    ends.back() = cursor;
                    continue;
                }
                last = cursor;
                break;
            }
            last = cursor;
            break;
        }
    }

    std::vector<std::string> out;
    out.reserve(words.size());
    size_t next_run = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (next_run < begins.size() && i == begins[next_run]) {
            std::string joined;
            for (int64_t k = i; k <= ends[next_run]; ++k) {
                if (words[k] == " ") continue;
                for (char c : words[k]) {
                    joined.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
                }
            }
            out.push_back(joined);
            i = ends[next_run];
            ++next_run;
            continue;
        }
        out.push_back(words[i]);
    }
    return out;
}

} // namespace

std::unique_ptr<FunasrVocab> FunasrVocab::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.ggml.model")) {
        return nullptr;
    }
    if (model.kv_str("tokenizer.ggml.model") != "funasr") {
        return nullptr; // not this vocab type -- caller should try CtcVocab/BpeVocab/Vocab/...
    }

    auto vocab = std::unique_ptr<FunasrVocab>(new FunasrVocab());
    vocab->tokens_ = model.kv_arr_str("tokenizer.ggml.tokens");
    if (vocab->tokens_.empty()) {
        throw LoadError("FunasrVocab::load: tokenizer.ggml.model is 'funasr' but "
                        "tokenizer.ggml.tokens is missing or empty; the table IS the vocabulary here");
    }

    // Required rather than defaulted: without it every piece would assemble as OTHER, which produces a
    // readable-looking transcript with no spaces between English words and no `@@` merged. A silent
    // near-miss is exactly what this array exists to prevent, so its absence is a load error.
    const std::vector<int32_t> script = model.kv_arr_i32("tokenizer.ggml.piece_script");
    if (script.size() != vocab->tokens_.size()) {
        throw LoadError("FunasrVocab::load: tokenizer.ggml.piece_script has " +
                        std::to_string(script.size()) + " entries for a " +
                        std::to_string(vocab->tokens_.size()) +
                        "-piece vocabulary; it is one script class per piece (0 other, 1 CJK, 2 Latin)");
    }
    vocab->script_.reserve(script.size());
    for (int32_t s : script) {
        if (s < OTHER || s > LATIN) {
            throw LoadError("FunasrVocab::load: tokenizer.ggml.piece_script contains " +
                            std::to_string(s) + "; the classes are 0 (other), 1 (CJK), 2 (Latin)");
        }
        vocab->script_.push_back(static_cast<uint8_t>(s));
    }

    const std::vector<int32_t> types = model.kv_arr_i32("tokenizer.ggml.token_type");
    vocab->control_.assign(vocab->tokens_.size(), false);
    for (size_t i = 0; i < types.size() && i < vocab->control_.size(); ++i) {
        vocab->control_[i] = (types[i] == TOKEN_TYPE_CONTROL);
    }
    return vocab;
}

const std::string& FunasrVocab::id_to_piece(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) {
        throw SchemaError("FunasrVocab: id " + std::to_string(id) + " is outside the " +
                          std::to_string(tokens_.size()) + "-piece vocabulary");
    }
    return tokens_[static_cast<size_t>(id)];
}

std::string FunasrVocab::decode(const std::vector<int32_t>& ids) const {
    std::vector<std::string> words;
    std::string pending_word;   // the reference's `word_item`: subword pieces awaiting their last one
    bool space_pending = false; // its `alpha_blank`: a space was emitted that CJK may still eat

    for (int32_t id : ids) {
        const std::string& piece = id_to_piece(id);
        if (control_[static_cast<size_t>(id)]) continue;

        // FOUR BRANCHES IN THIS ORDER, and the order is the reference's rather than a tidy switch.
        // The `@@` test is ORTHOGONAL to the script and sits between the two: `f@@` is neither CJK (an
        // 'f' is not) nor Latin (an '@' is not alphabetic), so it classifies as OTHER and would be
        // emitted literally by a switch that only looked at the script. It is a continuation because it
        // carries the marker, not because of what it is made of -- and `9@@` is CJK by the test above,
        // so it never reaches here at all.
        const Script script = static_cast<Script>(script_[static_cast<size_t>(id)]);
        if (script == CJK) {
            // Eats the space a completed Latin word left behind, which is the whole of "hello你好".
            if (space_pending && !words.empty()) words.pop_back();
            words.push_back(piece);
            space_pending = false;
        } else if (piece.find("@@") != std::string::npos) {
            // A continuation: strip every marker and hold the body until a piece without one closes
            // the word.
            std::string body = piece;
            for (size_t at = body.find("@@"); at != std::string::npos; at = body.find("@@")) {
                body.erase(at, 2);
            }
            pending_word += body;
            space_pending = false;
        } else if (script == LATIN) {
            pending_word += piece;
            words.push_back(pending_word);
            words.push_back(" ");
            pending_word.clear();
            space_pending = true;
        } else {
            // Punctuation and anything unclassified: emitted as-is, spacing untouched. The reference
            // leaves `word_item` alone here too, so a subword run survives across it.
            words.push_back(piece);
        }
    }

    std::string out;
    for (const std::string& word : abbr_dispose(words)) out += word;
    const size_t begin = out.find_first_not_of(" \t\n\r");
    if (begin == std::string::npos) return {};
    return out.substr(begin, out.find_last_not_of(" \t\n\r") - begin + 1);
}

} // namespace loom
