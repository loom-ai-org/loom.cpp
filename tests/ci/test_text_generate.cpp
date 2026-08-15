// The causal-LM decode loop: both driver shapes, the file's own stop token, and the ceiling.
//
// This coverage used to live in loom-py, as tests over a Python reimplementation of the same loop. It
// moved here with the loop itself (docs/HIGH-LEVEL-API.md §4) -- and it had to, because the reason the
// loop moved is that the two hosts' copies disagreed: `loom_cli` ran to the ceiling with no stop token,
// took the FIRST element of a list return where the new token is the last, and clamped ids to < 65536.
// Tests against either host's copy could only ever pin that host.
//
// The fixture driver returns `#tokens + 100`, so each token it produces records how long the prompt was
// when it ran. That is what makes prompt growth observable: a constant would pass whether the loop
// re-fed the grown prompt or handed the driver the original one every time.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <string>

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/generate_driver.gguf";
    auto model = loom::GgufModel::load(path, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, backend.get());

    // ---- A driver that generated internally is taken at its word --------------------------------
    {
        loom::text::GenerateOptions options;
        options.max_new_tokens = 64;
        options.extra_inputs["mode"] = 1.0;
        const std::vector<int32_t> ids =
            loom::text::generate(session.bridge(), *model, {1, 2}, options);
        // Its own stop condition already applied inside the driver; the host must not loop over it.
        LOOM_CHECK(ids.size() == 3);
        LOOM_CHECK(ids[0] == 7 && ids[1] == 8 && ids[2] == 9);
    }

    // ---- A single-token driver is looped, with the prompt growing --------------------------------
    {
        loom::text::GenerateOptions options;
        options.max_new_tokens = 3;
        options.eos_token = -1;  // no early stop, so the ceiling is what ends this
        options.extra_inputs["mode"] = 0.0;
        const std::vector<int32_t> ids =
            loom::text::generate(session.bridge(), *model, {1, 2}, options);
        LOOM_CHECK(ids.size() == 3);
        // 2 prompt tokens -> 102; then the prompt is 3 long -> 103; then 4 -> 104. Any of these being
        // 102 twice would mean the grown prompt never reached the driver.
        LOOM_CHECK(ids[0] == 102 && ids[1] == 103 && ids[2] == 104);
    }

    // ---- The stop token comes from the FILE when the caller names none ---------------------------
    {
        loom::text::GenerateOptions options;
        options.max_new_tokens = 10;  // deliberately past where the stop token appears
        options.extra_inputs["mode"] = 0.0;
        // eos_token left at kEosFromFile: the fixture declares 103, which this driver produces second.
        const std::vector<int32_t> ids =
            loom::text::generate(session.bridge(), *model, {1, 2}, options);
        // Stopped at the stop token, and did not emit it -- it is a control token, and a caller
        // detokenizing the result would otherwise get its literal spelling on the end.
        LOOM_CHECK(ids.size() == 1);
        LOOM_CHECK(ids[0] == 102);
    }

    // ---- ...and an explicitly disabled stop runs to the ceiling instead ---------------------------
    {
        loom::text::GenerateOptions options;
        options.max_new_tokens = 5;
        options.eos_token = -1;
        options.extra_inputs["mode"] = 0.0;
        const std::vector<int32_t> ids =
            loom::text::generate(session.bridge(), *model, {1, 2}, options);
        // Five tokens including 103, which the previous case stopped on. `-1` and "unspecified" are
        // genuinely different requests, which is why the option carries a separate sentinel for the
        // second rather than collapsing both onto -1.
        LOOM_CHECK(ids.size() == 5);
        LOOM_CHECK(ids[1] == 103);
    }

    // ---- An empty prompt is refused rather than silently producing something ----------------------
    {
        loom::text::GenerateOptions options;
        options.extra_inputs["mode"] = 0.0;
        LOOM_CHECK_THROWS(loom::text::generate(session.bridge(), *model, {}, options), loom::Error);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
