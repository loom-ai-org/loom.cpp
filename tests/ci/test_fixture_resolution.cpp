// The gate suite's fixture resolution (tests/support/fixtures.h).
//
// This is a ci test about the gate suite, which sounds like a category error and is not: 78 tests
// decide whether to run at all by asking `fixture_env` a question, and if that answer is wrong the
// whole gate suite skips silently and reports success. A rule that can make every real-model check
// disappear without saying so is exactly the rule worth pinning in the suite that always runs.
//
// The property that matters most is the LAST one: the per-test variable wins over the root. That is
// what made converting 78 files safe -- with the old variable set, resolution is what it always was.

#include "test_util.h"
#include "fixtures.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace {

std::string temp_root() {
    std::string dir = "/tmp/loom_fixture_resolution_test";
    if (const char* build_tmp = std::getenv("TMPDIR"); build_tmp != nullptr && *build_tmp != '\0') {
        dir = std::string(build_tmp) + "/loom_fixture_resolution_test";
    }
    return dir;
}

void write_file(const std::string& path) {
    std::ofstream out(path);
    out << "not a real gguf\n";
}

} // namespace

int main() {
    // --- the derivation rule, which `scripts/fixtures.py` implements a second time ---
    LOOM_CHECK(loom_test::fixture_relpath("LOOM_KOKORO_MIL_GGUF") == "kokoro_mil.gguf");
    LOOM_CHECK(loom_test::fixture_relpath("LOOM_GRANITE_SPEECH_MIL_GGUF") == "granite_speech_mil.gguf");
    // `_DIR` is dropped: the path already says it is a directory by not being a file.
    LOOM_CHECK(loom_test::fixture_relpath("LOOM_KOKORO_ALBERT_REF_DIR") == "kokoro_albert_ref");
    LOOM_CHECK(loom_test::fixture_relpath("LOOM_KOKORO_DIR") == "kokoro");
    // A variable that is neither keeps its whole name, lowercased.
    LOOM_CHECK(loom_test::fixture_relpath("LOOM_SUPERTONIC_VOICE_STYLE_JSON") ==
               "supertonic_voice_style_json");

    const std::string root = temp_root();
    std::string mkdir_cmd = "rm -rf '" + root + "' && mkdir -p '" + root + "'";
    LOOM_CHECK(std::system(mkdir_cmd.c_str()) == 0);
    write_file(root + "/kokoro_mil.gguf");

    ::unsetenv("LOOM_FIXTURES");
    ::unsetenv("LOOM_KOKORO_MIL_GGUF");

    // --- neither route configured: nullptr, which is what makes a gate test skip ---
    LOOM_CHECK(loom_test::fixture_env("LOOM_KOKORO_MIL_GGUF") == nullptr);

    // --- the root alone resolves a fixture that is present ---
    ::setenv("LOOM_FIXTURES", root.c_str(), 1);
    const char* via_root = loom_test::fixture_env("LOOM_KOKORO_MIL_GGUF");
    LOOM_CHECK(via_root != nullptr);
    LOOM_CHECK(std::string(via_root) == root + "/kokoro_mil.gguf");

    // --- ...and only one that is present. A half-populated root skips rather than fails, which is
    //     the difference between "I have not fetched that fixture" and "that model is broken". ---
    LOOM_CHECK(loom_test::fixture_env("LOOM_MATCHA_MIL_GGUF") == nullptr);

    // --- the per-test variable wins, even when the root would have resolved ---
    ::setenv("LOOM_KOKORO_MIL_GGUF", "/somewhere/else/kokoro_mil.gguf", 1);
    const char* via_explicit = loom_test::fixture_env("LOOM_KOKORO_MIL_GGUF");
    LOOM_CHECK(via_explicit != nullptr);
    LOOM_CHECK(std::string(via_explicit) == "/somewhere/else/kokoro_mil.gguf");

    // --- ...and is taken at its word: it names an artifact you just rebuilt, and a test that
    //     insisted on its existence here would be second-guessing the developer who set it. Existence
    //     is the test's own question, answered by its own skip. ---
    ::setenv("LOOM_KOKORO_MIL_GGUF", "/definitely/not/here.gguf", 1);
    LOOM_CHECK(std::string(loom_test::fixture_env("LOOM_KOKORO_MIL_GGUF")) == "/definitely/not/here.gguf");

    std::string cleanup = "rm -rf '" + root + "'";
    LOOM_CHECK(std::system(cleanup.c_str()) == 0);
    LOOM_TEST_REPORT_AND_RETURN();
}
