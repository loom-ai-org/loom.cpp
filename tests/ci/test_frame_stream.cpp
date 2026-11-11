// `FrameStream`, the audio-in rows-out door (ADR-106), against toy drivers on the contract fixtures: the
// door's own rules, with no model.
//
//   * the rows of every push joined are `classify`'s rows, however the audio is cut, and each update
//     says which frame its first row is;
//   * two streams on one model, and a one-pass call between their pushes, leave each other alone;
//   * an answer that is not whole rows is an error naming the numbers;
//   * a clip answer, a contract that is not audio in, and a driver without the three functions are
//     refused at construction.
#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cmath>
#include <string>
#include <vector>

namespace {

// The fixtures' `infer` answers one row per four samples and drops the remainder; this is the same answer
// a push at a time: row r is [0.25, 0.75] for even r and the other way round for odd, as the fixture's.
const char* const kStream = R"(
function stream_open(inputs)
    loom.set_state('pending', 0)
    loom.set_state('row', 0)
    return {}
end
local function rows(n)
    local out, r = {}, loom.state('row')
    for i = 1, n do
        if r % 2 == 0 then out[#out + 1] = 0.25; out[#out + 1] = 0.75
        else out[#out + 1] = 0.75; out[#out + 1] = 0.25 end
        r = r + 1
    end
    loom.set_state('row', r)
    return out
end
function stream_feed(inputs)
    local have = loom.state('pending') + #(inputs.waveform or {})
    loom.set_state('pending', have % 4)
    return rows(math.floor(have / 4))
end
function stream_finish(inputs)
    return {}
end
)";

std::unique_ptr<loom::GgufModel> load(const char* name, ggml_backend_t backend) {
    return loom::GgufModel::load(std::string(LOOM_TEST_FIXTURE_DIR) + "/" + name, backend);
}

std::vector<float> pushed(loom::audio::FrameStream& s, const std::vector<size_t>& cuts) {
    std::vector<float> out;
    for (size_t n : cuts) {
        const loom::audio::FrameUpdate u = s.push(std::vector<float>(n, 0.5f));
        LOOM_CHECK(u.first_row * s.width() == out.size());
        LOOM_CHECK(u.values.size() == u.n_rows * s.width());
        out.insert(out.end(), u.values.begin(), u.values.end());
    }
    const loom::audio::FrameUpdate tail = s.finish();
    out.insert(out.end(), tail.values.begin(), tail.values.end());
    return out;
}

void test_rows_are_the_one_pass_rows(ggml_backend_t backend) {
    auto model = load("audio_frame_classifier.gguf", backend);
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, backend);
    session.bridge().load_script(kStream);
    const std::vector<float> clip(38, 0.5f);
    const std::vector<float> want = loom::audio::classify(session.bridge(), *model, clip).probabilities;
    LOOM_CHECK(want.size() == 9 * 2);

    loom::audio::FrameStream s(session.bridge(), *model);
    LOOM_CHECK(s.output_kind() == loom::modality::CLASS && s.width() == 2 && s.labels()[1] == "speech");
    // A sample at a time, ragged pieces, nothing at all, and the whole clip in one push.
    for (const std::vector<size_t>& cuts : std::vector<std::vector<size_t>>{
             std::vector<size_t>(38, 1), {3, 0, 5, 9, 1, 20}, {0, 38, 0}, {38}}) {
        s.reset();
        LOOM_CHECK(pushed(s, cuts) == want);
        LOOM_CHECK(s.rows() == 9 && s.finished());
    }
    // Frame 2 at 4 frames/s from 0.5 s.
    LOOM_CHECK(std::fabs(s.row_start(2) - 1.0) < 1e-9);
    LOOM_CHECK_THROWS(s.push(std::vector<float>(4, 0.5f)), loom::Error);
    LOOM_CHECK_THROWS(s.finish(), loom::Error);
}

void test_streams_leave_each_other_alone(ggml_backend_t backend) {
    auto model = load("audio_frame_classifier.gguf", backend);
    loom::Session session(*model, backend);
    session.bridge().load_script(kStream);
    loom::audio::FrameStream a(session.bridge(), *model), b(session.bridge(), *model);
    // Three samples into `a`, then a one-pass call and a whole row into `b`: `a`'s fourth sample is still
    // its first row's, and that row is row 0.
    LOOM_CHECK(a.push(std::vector<float>(3, 0.5f)).n_rows == 0);
    LOOM_CHECK(loom::audio::classify(session.bridge(), *model, std::vector<float>(10, 0.5f)).n_rows == 2);
    LOOM_CHECK(b.push(std::vector<float>(8, 0.5f)).n_rows == 2);
    const loom::audio::FrameUpdate u = a.push(std::vector<float>(1, 0.5f));
    LOOM_CHECK(u.n_rows == 1 && u.first_row == 0 && u.values[1] == 0.75f && u.samples == 4);
    LOOM_CHECK(b.rows() == 2 && a.rows() == 1);
}

void test_an_embedder_streams_rows_of_its_width(ggml_backend_t backend) {
    auto model = load("audio_frame_embedder.gguf", backend);
    loom::Session session(*model, backend);
    session.bridge().load_script("function stream_open(i) return {} end\n"
                                 "function stream_feed(i) return {1, 2, 3, 4, 5, 6} end\n"
                                 "function stream_finish(i) return {} end\n");
    loom::audio::FrameStream s(session.bridge(), *model);
    LOOM_CHECK(s.output_kind() == loom::modality::EMBEDDINGS && s.width() == 3 && s.labels().empty());
    const loom::audio::FrameUpdate u = s.push(std::vector<float>(1, 0.0f));
    LOOM_CHECK(u.n_rows == 2 && u.values[3] == 4.0f);
}

void test_a_skewed_answer_is_an_error(ggml_backend_t backend) {
    auto model = load("audio_frame_classifier.gguf", backend);
    loom::Session session(*model, backend);
    session.bridge().load_script("function stream_open(i) return {} end\n"
                                 "function stream_feed(i) return {0.1, 0.2, 0.3} end\n"
                                 "function stream_finish(i) return 0.5 end\n");
    loom::audio::FrameStream s(session.bridge(), *model);
    bool named = false;
    try {
        s.push(std::vector<float>(4, 0.5f));
    } catch (const loom::Error& e) {
        named = std::string(e.what()).find("returned 3 numbers") != std::string::npos;
    }
    LOOM_CHECK(named);
    LOOM_CHECK_THROWS(s.finish(), loom::Error); // a bare number is not rows
}

void test_what_cannot_stream_is_refused(ggml_backend_t backend) {
    // The fixture's own driver: `infer` alone.
    {
        auto model = load("audio_frame_classifier.gguf", backend);
        loom::Session session(*model, backend);
        LOOM_CHECK(!loom::audio::FrameStream::supported(session.bridge()));
        LOOM_CHECK_THROWS(loom::audio::FrameStream(session.bridge(), *model), loom::LoadError);
    }
    // A clip classifier has nothing to hand back before the end, whatever its driver exports.
    {
        auto model = load("audio_clip_classifier.gguf", backend);
        loom::Session session(*model, backend);
        session.bridge().load_script(kStream);
        LOOM_CHECK_THROWS(loom::audio::FrameStream(session.bridge(), *model), loom::Error);
    }
    // A frame embedder exported before `loom.output.embedding_dim`: no width, no cut.
    {
        auto model = load("audio_frame_embedder_undeclared.gguf", backend);
        loom::Session session(*model, backend);
        session.bridge().load_script(kStream);
        LOOM_CHECK_THROWS(loom::audio::FrameStream(session.bridge(), *model), loom::Error);
    }
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    test_rows_are_the_one_pass_rows(backend.get());
    test_streams_leave_each_other_alone(backend.get());
    test_an_embedder_streams_rows_of_its_width(backend.get());
    test_a_skewed_answer_is_an_error(backend.get());
    test_what_cannot_stream_is_refused(backend.get());

    LOOM_TEST_REPORT_AND_RETURN();
}
