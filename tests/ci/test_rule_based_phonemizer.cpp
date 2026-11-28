// Tests loom::RuleBasedPhonemizer (src/core/rule_based_phonemizer.cpp, ADR-145) against the toy script of
// tests/fixtures/make_rule_based_phonemizer_gguf.py: the contract, the data host function, the sandbox,
// the optional `dropped` result, the run-time lexicon, and the two files load must refuse.

#include "test_util.h"

#include "loom/loom.h"

int main() {
    auto load = [](const char* name) {
        return loom::GgufModel::load_metadata(std::string(LOOM_TEST_FIXTURE_DIR) + "/" + name);
    };

    // A file with no phonemizer gives null, not an error.
    LOOM_CHECK(loom::RuleBasedPhonemizer::load(*load("phoneme_vocab_test.gguf")) == nullptr);

    auto model = load("rule_phonemizer_test.gguf");
    auto ph = loom::RuleBasedPhonemizer::load(*model);
    LOOM_CHECK(ph != nullptr);
    LOOM_CHECK(ph->languages() == std::vector<std::string>({"x", "y"}));
    LOOM_CHECK(ph->ruleset() == "toy 1; x=sha256:00");

    // Through the data of language 0: a -> ɑ, b -> β. "!" would mean io, os or a file loader leaked.
    auto r = ph->phonemize("ab", 0);
    LOOM_CHECK(r.ipa == "ɑβ");
    LOOM_CHECK(r.dropped.empty());

    // What the script cannot read comes back in text order, with repeats.
    r = ph->phonemize("a2b2!", 0);
    LOOM_CHECK(r.ipa == "ɑβ");
    LOOM_CHECK(r.dropped == "22!");

    // Language 1 has no data key: phonemizer_data gives nil, and every codepoint is dropped.
    r = ph->phonemize("ab", 1);
    LOOM_CHECK(r.ipa.empty());
    LOOM_CHECK(r.dropped == "ab");

    // The overlay wins for its word, and only in its language.
    ph->set_lexicon(0, {{"ab", "æβ"}});
    LOOM_CHECK(ph->phonemize("ab", 0).ipa == "æβ");
    LOOM_CHECK(ph->phonemize("ab", 1).ipa.empty());

    LOOM_CHECK_THROWS(ph->phonemize("a", 2), loom::Error);
    LOOM_CHECK_THROWS(loom::RuleBasedPhonemizer::load(*load("rule_phonemizer_bad_kind.gguf")), loom::LoadError);
    LOOM_CHECK_THROWS(loom::RuleBasedPhonemizer::load(*load("rule_phonemizer_no_entry.gguf")), loom::LoadError);

    LOOM_TEST_REPORT_AND_RETURN();
}
