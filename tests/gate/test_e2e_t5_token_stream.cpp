// The tokens-out door (ADR-099) on an encoder-decoder (ADR-130): flan-t5's `GenerationStream` encodes the
// source once in `stream_open`, keeps the cross-attention K/V and the source length in the stream's state,
// and draws a token per feed with the one pass's loop.
//
//   1. Per prompt of the reference (`scripts/t5_stream_reference.py`: transformers' greedy `generate`, the ids
//      a `TextIteratorStreamer` hands out): `generate` returns them, and the stream pulled 1, 4 and 48 at
//      a time returns them too, the stop -- `</s>` or the 48-token ceiling -- included.
//   2. `reset` draws the same ids again.
//   3. Two streams on different sources pulled in alternation, a one-pass `generate` on a third between
//      their pulls, under both cache modes: each gives its own ids. The K/V a stream keeps is its own; a
//      stream reading the model's retained `cross_kv` would read the last call's source instead.
//   4. With LOOM_T5_STREAM_OLD_GGUF (the published file): its one pass is this file's, and the door refuses
//      it -- its `infer` runs the whole loop, and the one-token path fed the answer back as the source.
//
// LOOM_T5_STREAM_GGUF: flan-t5-small exported with the stream functions.
// LOOM_T5_STREAM_REF: the reference script's output.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;
constexpr uint32_t kCeiling = 48;

struct Case {
    std::vector<int32_t> src, out;
};

std::vector<int32_t> ids_of(const std::string& line) {
    std::istringstream in(line);
    std::string tag;
    in >> tag;
    std::vector<int32_t> ids;
    for (int32_t id; in >> id;) ids.push_back(id);
    return ids;
}

std::vector<int32_t> pull_all(loom::text::GenerationStream& stream, uint32_t per_pull) {
    while (!stream.finished()) stream.next(per_pull);
    return stream.tokens();
}

void print_ids(const char* label, const std::vector<int32_t>& ids) {
    std::fprintf(stderr, "%s (%zu):", label, ids.size());
    for (int32_t id : ids) std::fprintf(stderr, " %d", id);
    std::fprintf(stderr, "\n");
}

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_T5_STREAM_GGUF");
    const char* ref = loom_test::fixture_env("LOOM_T5_STREAM_REF");
    if (gguf == nullptr || ref == nullptr) {
        std::fprintf(stderr, "SKIP: set LOOM_T5_STREAM_GGUF and LOOM_T5_STREAM_REF\n");
        return kSkipReturnCode;
    }
    std::vector<Case> cases;
    {
        std::ifstream in(ref);
        for (std::string src, out; std::getline(in, src) && std::getline(in, out);) {
            cases.push_back({ids_of(src), ids_of(out)});
        }
    }
    LOOM_CHECK(cases.size() >= 3);

    ggml_backend_ptr backend(loom_test::cpu_backend());
    auto model = loom::GgufModel::load(gguf, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, backend.get());
    loom::LoomLuaBridge& bridge = session.bridge();
    LOOM_CHECK(bridge.has_function("stream_open"));
    LOOM_CHECK(loom::text::GenerationStream::supported(bridge, *model));

    // The reference keeps the final `</s>`: so does this, by not stripping it.
    loom::text::GenerateOptions greedy;
    greedy.max_new_tokens = kCeiling;
    greedy.temperature = 0.0f;
    greedy.strip_eos = false;

    // 1 + 2. Each prompt, one pass and streamed.
    for (size_t i = 0; i < cases.size(); ++i) {
        const Case& c = cases[i];
        const std::vector<int32_t> one = loom::text::generate(bridge, *model, c.src, greedy);
        if (one != c.out) print_ids("generate", one), print_ids("transformers", c.out);
        LOOM_CHECK(one == c.out);
        int ok = 0;
        for (uint32_t per_pull : {1u, 4u, kCeiling}) {
            loom::text::GenerationStream stream(bridge, *model, c.src, greedy);
            const std::vector<int32_t> got = pull_all(stream, per_pull);
            if (got != c.out) print_ids("stream", got);
            LOOM_CHECK(got == c.out);
            LOOM_CHECK(stream.next(1).empty());
            ok += got == c.out;
            if (per_pull == 1) {
                stream.reset();
                LOOM_CHECK(pull_all(stream, 3) == c.out);
            }
        }
        std::printf("prompt %zu: %zu source ids, %zu out; one pass %s, streams %d/3\n", i, c.src.size(),
                    c.out.size(), one == c.out ? "ok" : "DIFFERS", ok);
    }

    // 3. Interleaved, under both cache modes.
    const Case& a_case = cases[3];   // the 48-token ceiling
    const Case& b_case = cases[2];
    const Case& c_case = cases[1];
    for (loom::StreamCaches mode : {loom::StreamCaches::Park, loom::StreamCaches::PerStream}) {
        const bool park = mode == loom::StreamCaches::Park;
        bridge.set_stream_caches(mode);
        loom::text::GenerationStream a(bridge, *model, a_case.src, greedy);
        loom::text::GenerationStream b(bridge, *model, b_case.src, greedy);
        size_t turn = 0;
        while (!a.finished() || !b.finished()) {
            a.next(3);
            if (turn++ % 2 == 0) LOOM_CHECK(loom::text::generate(bridge, *model, c_case.src, greedy) == c_case.out);
            b.next(2);
        }
        std::printf("%s: a %s, b %s\n", park ? "park" : "per-stream", a.tokens() == a_case.out ? "ok" : "DIFFERS",
                    b.tokens() == b_case.out ? "ok" : "DIFFERS");
        LOOM_CHECK(a.tokens() == a_case.out);
        LOOM_CHECK(b.tokens() == b_case.out);
    }

    // 4. The published file.
    if (const char* old = loom_test::fixture_env("LOOM_T5_STREAM_OLD_GGUF")) {
        auto old_model = loom::GgufModel::load(old, backend.get());
        LOOM_CHECK(old_model != nullptr);
        loom::Session old_session(*old_model, backend.get());
        for (const Case& c : cases) {
            LOOM_CHECK(loom::text::generate(old_session.bridge(), *old_model, c.src, greedy) == c.out);
        }
        LOOM_CHECK(!loom::text::GenerationStream::supported(old_session.bridge(), *old_model));
        LOOM_CHECK_THROWS(loom::text::GenerationStream(old_session.bridge(), *old_model, cases[0].src, greedy),
                          loom::LoadError);
        std::printf("published file: one pass identical, stream refused\n");
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
