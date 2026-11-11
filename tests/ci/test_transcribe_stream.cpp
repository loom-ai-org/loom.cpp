// `TranscriptionStream`, the audio-in door (ADR-098), against toy drivers on a WordPiece fixture: the door's
// own rules, with no model.
//
//   * a driver that returns ids has them appended, and the text only grows;
//   * a re-decoding driver's `hypothesis` state is the whole transcript after every call (ADR-106): what it
//     returns is ignored, and an update that does not extend the old text is `revised`, its delta the whole
//     text;
//   * `stream_open` gets the host's inputs, again on `reset`.
#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <string>
#include <vector>

namespace {

// Ids 5, 6, 7 are "▁hello", "▁wor", "ld" in the fixture's table.
const char* const kAppending = R"(
function stream_open(inputs) loom.set_state('n', 0) return {} end
function stream_feed(inputs)
    local n = loom.state('n') + 1
    loom.set_state('n', n)
    if n == 1 then return {5} end
    return {6, 7}
end
function stream_finish(inputs) return {} end
)";

const char* const kRedecoding = R"(
function stream_open(inputs)
    loom.set_state('n', 0)
    loom.set_state('every', inputs.update_seconds or 0)
    loom.set_state('hypothesis', {})
    return {}
end
function stream_feed(inputs)
    local n = loom.state('n') + 1
    loom.set_state('n', n)
    if n == 1 then loom.set_state('hypothesis', loom.state('every') == 2 and {5, 6} or {6})
    else loom.set_state('hypothesis', {6, 7}) end
    return {7, 7, 7}  -- ignored: the hypothesis is the transcript
end
function stream_finish(inputs)
    loom.set_state('hypothesis', {5, 6, 7})
    return {}
end
)";

std::unique_ptr<loom::GgufModel> load(ggml_backend_t backend) {
    return loom::GgufModel::load(std::string(LOOM_TEST_FIXTURE_DIR) + "/wordpiece_asr.gguf", backend);
}

const std::vector<float> kPiece(160, 0.0f);

void test_ids_are_appended(ggml_backend_t backend) {
    auto model = load(backend);
    loom::Session session(*model, backend);
    session.bridge().load_script(kAppending);
    loom::audio::TranscriptionStream s(session.bridge(), *model);
    const auto a = s.push(kPiece);
    LOOM_CHECK(a.text == "hello" && a.delta == "hello" && !a.revised && a.tokens == std::vector<int32_t>{5});
    const auto b = s.push(kPiece);
    LOOM_CHECK(b.text == "hello world" && b.delta == " world" && !b.revised);
    LOOM_CHECK(s.finish().delta.empty() && s.tokens() == (std::vector<int32_t>{5, 6, 7}));
}

void test_a_hypothesis_replaces(ggml_backend_t backend) {
    auto model = load(backend);
    loom::Session session(*model, backend);
    session.bridge().load_script(kRedecoding);
    loom::audio::TranscriptionStream s(session.bridge(), *model, {{"update_seconds", 2.0}});
    // The inputs reached `stream_open`: the first hypothesis is the one it chose for them.
    const auto a = s.push(kPiece);
    LOOM_CHECK(a.text == "hello wor" && !a.revised && a.tokens == (std::vector<int32_t>{5, 6}));
    // "wor" -> "world" extends nothing it had: revised, and the delta is the whole text.
    const auto b = s.push(kPiece);
    LOOM_CHECK(b.text == "world" && b.revised && b.delta == "world");
    LOOM_CHECK(s.tokens() == (std::vector<int32_t>{6, 7}));
    const auto c = s.finish();
    LOOM_CHECK(c.text == "hello world" && c.revised && c.tokens == (std::vector<int32_t>{5, 6, 7}));
    // `reset` opens again with the same inputs.
    s.reset();
    LOOM_CHECK(s.push(kPiece).tokens == (std::vector<int32_t>{5, 6}));
    // Without them the driver's default.
    loom::audio::TranscriptionStream plain(session.bridge(), *model);
    LOOM_CHECK(plain.push(kPiece).tokens == std::vector<int32_t>{6});
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    test_ids_are_appended(backend.get());
    test_a_hypothesis_replaces(backend.get());
    LOOM_TEST_REPORT_AND_RETURN();
}
