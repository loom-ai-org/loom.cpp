// `SpeechStream`, the audio-out door (ADR-100), against toy drivers: the door's own rules, with no model.
//
//   * pieces come back in order, an empty piece is skipped rather than handed back, `stream_finish`'s
//     tail joins the last piece, and the counters are the driver's `piece` / `n_pieces`;
//   * a seeded stream draws its seed's sequence however another stream and a one-pass call interleave;
//   * a driver whose `stream_feed` makes nothing, moves nothing and never finishes is an error, not a
//     loop forever;
//   * a driver without the three functions is refused at construction.
#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <string>
#include <vector>

namespace {

// Three pieces, the second empty; each draws one normal so a stream's draws are visible in its audio.
const char* const kDriver = R"(
function stream_open(inputs)
    if inputs.seed then loom.seed_rng(inputs.seed) end
    loom.set_state('base', inputs.tokens[1])
    loom.set_state('piece', 0)
    loom.set_state('n_pieces', 3)
    loom.set_state('done', 0)
    return {}
end
function stream_feed(inputs)
    local c = loom.state('piece') + 1
    local draw = loom.gaussian_array(1)[1]
    loom.set_state('piece', c)
    if c == 3 then loom.set_state('done', 1) end
    if c == 2 then return {} end
    return {loom.state('base') + c, draw}
end
function stream_finish(inputs)
    return {-1}
end
function infer(inputs)
    stream_open(inputs)
    local out = {}
    while loom.state('done') == 0 do
        for _, v in ipairs(stream_feed({})) do out[#out + 1] = v end
    end
    for _, v in ipairs(stream_finish({})) do out[#out + 1] = v end
    return out
end
function draw(inputs)
    return loom.gaussian_array(4)
end
)";

std::vector<float> pull_all(loom::audio::SpeechStream& s) {
    std::vector<float> out;
    while (!s.finished()) {
        const std::vector<float> piece = s.next();
        out.insert(out.end(), piece.begin(), piece.end());
    }
    return out;
}

void test_pieces_in_order(loom::LoomLuaBridge& bridge) {
    loom::audio::SpeechStream s(bridge, {{"tokens", std::vector<double>{100}}, {"seed", 1.0}});
    LOOM_CHECK(s.pieces_total() == 0 && s.pieces_done() == 0); // nothing runs before the first `next`
    const std::vector<float> first = s.next();
    LOOM_CHECK(first.size() == 2 && first[0] == 101.0f);
    LOOM_CHECK(s.pieces_done() == 1 && s.pieces_total() == 3 && !s.finished());
    // Piece 2 is empty and skipped; piece 3 is the last, so the finish's tail joins it.
    const std::vector<float> last = s.next();
    LOOM_CHECK(last.size() == 3 && last[0] == 103.0f && last[2] == -1.0f);
    LOOM_CHECK(s.finished() && s.pieces_done() == 3 && s.samples() == 5);
    LOOM_CHECK(s.next().empty());
}

void test_seeded_streams_keep_their_draws(loom::LoomLuaBridge& bridge) {
    const loom::audio::SpeechStream::Inputs a_in{{"tokens", std::vector<double>{100}}, {"seed", 7.0}};
    const loom::audio::SpeechStream::Inputs b_in{{"tokens", std::vector<double>{200}}, {"seed", 8.0}};
    const auto one_pass = [&](const loom::audio::SpeechStream::Inputs& in) {
        const auto got = std::get<std::vector<double>>(bridge.call("infer", in));
        return std::vector<float>(got.begin(), got.end());
    };
    const std::vector<float> want_a = one_pass(a_in), want_b = one_pass(b_in);
    LOOM_CHECK(want_a != want_b);

    loom::audio::SpeechStream a(bridge, a_in), b(bridge, b_in);
    std::vector<float> got_a, got_b;
    while (!a.finished() || !b.finished()) {
        const std::vector<float> pa = a.next();
        got_a.insert(got_a.end(), pa.begin(), pa.end());
        bridge.call("draw", {}); // the bridge's own generator moves between the two streams' pieces
        const std::vector<float> pb = b.next();
        got_b.insert(got_b.end(), pb.begin(), pb.end());
    }
    LOOM_CHECK(got_a == want_a);
    LOOM_CHECK(got_b == want_b);

    a.reset();
    LOOM_CHECK(!a.finished() && a.samples() == 0);
    LOOM_CHECK(pull_all(a) == want_a);
}

void test_a_stalled_driver_is_an_error(ggml_backend_t backend) {
    loom::LoomLuaBridge bridge(loom::Backends{backend, nullptr});
    bridge.load_script("function stream_open(i) loom.set_state('piece', 0) return {} end\n"
                       "function stream_feed(i) return {} end\n"
                       "function stream_finish(i) return {} end\n");
    loom::audio::SpeechStream s(bridge, {});
    LOOM_CHECK_THROWS(s.next(), loom::Error);
}

void test_a_named_key_is_refused_not_dropped(ggml_backend_t backend) {
    // `{h = {...}}` would read as the empty array of its positional part; the next call would find its
    // state gone. An array and a number still go in, a named key does not.
    loom::LoomLuaBridge bridge(loom::Backends{backend, nullptr});
    bridge.load_script("function stream_open(i) loom.set_state('a', {1, 2}) loom.set_state('n', 3)\n"
                       "  loom.set_state('carry', {h = {1, 2}}) return {} end\n"
                       "function stream_feed(i) return {} end\n"
                       "function stream_finish(i) return {} end\n");
    bool refused = false;
    try {
        loom::audio::CodecStream s(bridge, {});
    } catch (const loom::Error& e) {
        refused = std::string(e.what()).find("named key") != std::string::npos;
    }
    LOOM_CHECK(refused);
}

void test_a_driver_without_the_functions_is_refused(ggml_backend_t backend) {
    loom::LoomLuaBridge bridge(loom::Backends{backend, nullptr});
    bridge.load_script("function infer(i) return {0.5} end\n");
    LOOM_CHECK(!loom::audio::SpeechStream::supported(bridge));
    LOOM_CHECK_THROWS(loom::audio::SpeechStream(bridge, {}), loom::LoadError);
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    loom::LoomLuaBridge bridge(loom::Backends{backend.get(), nullptr});
    bridge.load_script(kDriver);
    LOOM_CHECK(loom::audio::SpeechStream::supported(bridge));

    test_pieces_in_order(bridge);
    test_seeded_streams_keep_their_draws(bridge);
    test_a_stalled_driver_is_an_error(backend.get());
    test_a_driver_without_the_functions_is_refused(backend.get());
    test_a_named_key_is_refused_not_dropped(backend.get());

    LOOM_TEST_REPORT_AND_RETURN();
}
