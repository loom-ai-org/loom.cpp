#pragma once

// A rule-based phonemizer that a file carries as a script plus data (ADR-145). The engine runs the script
// in a Lua state of its own, opened without `io`, `os` or file loaders, through a fixed contract:
//
//   phonemize(codepoints, lang) -> ipa_codepoints[, dropped_codepoints]
//   set_lexicon(lang, {{word_codepoints, ipa_codepoints}, ...})          (optional)
//
// and one host function the script calls for its data: phonemizer_data(lang) -> string | nil, the value
// of `loom.phonemizer.data.<code>` for the language at 0-based index `lang` in `loom.text.languages`.
// IPA to ids is not done here: that is the voice's `PhonemeVocab`.

#include "loom/core/gguf_model.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

struct lua_State;

namespace loom {

class RuleBasedPhonemizer {
public:
    struct Result {
        std::string ipa;
        // What the script could not read, in text order with repeats, for the caller to warn about.
        std::string dropped;
    };

    // Null when the file declares no phonemizer (`loom.phonemizer.kind` absent). Throws LoadError for a
    // kind other than "lua", or for a script that fails to load or does not define `phonemize`.
    static std::unique_ptr<RuleBasedPhonemizer> load(const GgufModel& model);
    ~RuleBasedPhonemizer();
    RuleBasedPhonemizer(const RuleBasedPhonemizer&) = delete;
    RuleBasedPhonemizer& operator=(const RuleBasedPhonemizer&) = delete;

    // UTF-8 in and out. Throws Error for a `lang` outside `languages()` or an error in the script.
    Result phonemize(const std::string& text, int lang);

    // A run-time overlay of word -> IPA entries for one language. Throws Error when the script does not
    // define `set_lexicon`. The ruleset does not cover an overlay: a caller records it itself.
    void set_lexicon(int lang, const std::vector<std::pair<std::string, std::string>>& entries);

    const std::vector<std::string>& languages() const { return languages_; }
    const std::string& ruleset() const { return ruleset_; }

private:
    RuleBasedPhonemizer() = default;
    static int l_phonemizer_data(lua_State* L);
    void check_lang(int lang) const;

    lua_State* L_ = nullptr;
    std::vector<std::string> languages_;
    std::vector<std::string> data_;  // by language index; empty when the file has no key for it
    std::string ruleset_;
};

} // namespace loom
