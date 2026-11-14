// `TranscriptionStream`, the audio-in door (ADR-098), against toy drivers on a WordPiece fixture: the door's
// own rules, with no model.
//
//   * a driver that returns ids has them appended, and the text only grows;
//   * a re-decoding driver's `hypothesis` state is the whole transcript after every call (ADR-106): what it
//     returns is ignored, and an update that does not extend the old text is `revised`, its delta the whole
//     text;
//   * `stream_open` gets the host's inputs, again on `reset`;
//   * an array entry grows, is read and trimmed in place (`loom.append_state`, `state_slice`, `drop_state`,
//     `state_size`); a slice past its end and an append to a number are refused; a one-pass call does not
//     see a stream's entries.
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

// A sliding buffer the way the buffered drivers keep their audio: everything appended, the newest two read
// back by slice, the oldest dropped to keep three.
const char* const kBuffered = R"(
function stream_open(inputs) loom.set_state('pcm', {}) loom.set_state('n', 0) return {} end
function stream_feed(inputs)
    -- Sentinel pushes for the refusals: inside a stream, so each fails for its own reason.
    if inputs.waveform[1] == 99 then return loom.state_slice('pcm', 2, 5) end
    if inputs.waveform[1] == 98 then loom.set_state('k', 1) loom.append_state('k', {1}) end
    local n = loom.append_state('pcm', inputs.waveform)
    loom.set_state('newest', loom.state_slice('pcm', n - 2, 2))
    loom.set_state('size', loom.state_size('pcm'))
    if n > 3 then loom.drop_state('pcm', n - 3) end
    return {}
end
function stream_finish(inputs) return {} end
function outside_a_stream(inputs) return loom.state_size('pcm') end
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

void test_an_array_entry_slides(ggml_backend_t backend) {
    auto model = load(backend);
    loom::Session session(*model, backend);
    session.bridge().load_script(kBuffered);
    loom::audio::TranscriptionStream s(session.bridge(), *model);
    auto values = [&](const char* name) {
        const loom::StreamState::Value* v = s.state().value(name);
        return v ? v->data : std::vector<double>{-1.0};
    };
    s.push(std::vector<float>{1, 2});
    LOOM_CHECK(values("pcm") == (std::vector<double>{1, 2}) && values("newest") == (std::vector<double>{1, 2}));
    s.push(std::vector<float>{3, 4, 5});
    // Appended to 5, read 4 and 5, trimmed to the last three.
    LOOM_CHECK(values("size") == std::vector<double>{5} && values("newest") == (std::vector<double>{4, 5}));
    LOOM_CHECK(values("pcm") == (std::vector<double>{3, 4, 5}));
    // A push of nothing appends nothing.
    s.push(std::vector<float>{});
    LOOM_CHECK(values("pcm") == (std::vector<double>{3, 4, 5}));
    // Refused, not padded or clamped: a slice past the end, an append to a number, a call with no stream.
    LOOM_CHECK_THROWS(s.push(std::vector<float>{99}), loom::Error);
    LOOM_CHECK_THROWS(s.push(std::vector<float>{98}), loom::Error);
    // A one-pass call has a state of its own (ADR-098): the stream's buffer is not in it.
    const auto outside = session.bridge().call("outside_a_stream", {});
    LOOM_CHECK(std::holds_alternative<double>(outside) && std::get<double>(outside) == 0.0);
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    test_ids_are_appended(backend.get());
    test_a_hypothesis_replaces(backend.get());
    test_an_array_entry_slides(backend.get());
    LOOM_TEST_REPORT_AND_RETURN();
}
