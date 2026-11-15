// The tokens-out live door (ADR-099) on a real causal LM: `GenerationStream` must hand back exactly the
// ids `generate` returns, however it is pulled, and streams on one model must not see each other.
//
//   1. Greedy, pulled 1, 5 and 48 at a time: the ids and the stop are `generate`'s.
//   2. Seeded sampling: the same, because a seeded stream draws from a generator of its own.
//   3. Two streams pulled in alternation, with a one-pass `generate` between their pulls, under both
//      cache modes: each gives its own `generate` ids. In Park mode the stream that is not running has
//      its rows parked on the host; in PerStream mode it owns a cache and parks nothing. This is the
//      check that sees a leak: a lease that did not park would hand the second stream's K/V to the
//      first.
//
// LOOM_STREAM_LM_GGUF: an LM exported with the stream functions (qwen3-0.6b-base, smollm2-360m-it,
// gemma-3-270m-it, lfm2-350m-monolithic). A file exported before them is refused by the door, which is
// checked too when LOOM_STREAM_LM_OLD_GGUF names one.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;
constexpr uint32_t kCeiling = 48;

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
    const char* gguf = loom_test::fixture_env("LOOM_STREAM_LM_GGUF");
    if (gguf == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_STREAM_LM_GGUF to a causal LM exported with the "
                              "stream functions\n");
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf, backend.get());
    LOOM_CHECK(model != nullptr);
    auto vocab = loom::BpeVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    loom::Session session(*model, backend.get());
    loom::LoomLuaBridge& bridge = session.bridge();
    LOOM_CHECK(loom::text::GenerationStream::supported(bridge));
    LOOM_CHECK(bridge.has_function("stream_open"));

    const std::vector<int32_t> prompt_a = vocab->encode("The history of the city of Rome begins");
    const std::vector<int32_t> prompt_b = vocab->encode("A recipe for bread needs flour, water and");
    const std::vector<int32_t> prompt_c = vocab->encode("The three primary colours are");

    loom::text::GenerateOptions greedy;
    greedy.max_new_tokens = kCeiling;
    greedy.temperature = 0.0f;

    // 1. Greedy, at three pull sizes.
    const std::vector<int32_t> want_a = loom::text::generate(bridge, *model, prompt_a, greedy);
    print_ids("generate a", want_a);
    LOOM_CHECK(!want_a.empty());
    for (uint32_t per_pull : {1u, 5u, kCeiling}) {
        loom::text::GenerationStream stream(bridge, *model, prompt_a, greedy);
        const std::vector<int32_t> got = pull_all(stream, per_pull);
        if (got != want_a) print_ids("stream a", got);
        LOOM_CHECK(got == want_a);
        LOOM_CHECK(stream.next(1).empty());
    }

    // 2. Seeded sampling.
    loom::text::GenerateOptions sampled = greedy;
    sampled.temperature = 0.9f;
    sampled.top_k = 40;
    sampled.seed = 1234;
    const std::vector<int32_t> want_sampled = loom::text::generate(bridge, *model, prompt_b, sampled);
    print_ids("generate b sampled", want_sampled);
    {
        loom::text::GenerationStream stream(bridge, *model, prompt_b, sampled);
        LOOM_CHECK(pull_all(stream, 3) == want_sampled);
        stream.reset();
        LOOM_CHECK(pull_all(stream, 7) == want_sampled);
    }

    // 3. Interleaved, under both cache modes.
    const std::vector<int32_t> want_c = loom::text::generate(bridge, *model, prompt_c, greedy);
    for (loom::StreamCaches mode : {loom::StreamCaches::Park, loom::StreamCaches::PerStream}) {
        const bool park = mode == loom::StreamCaches::Park;
        bridge.set_stream_caches(mode);
        loom::text::GenerationStream a(bridge, *model, prompt_a, greedy);
        loom::text::GenerationStream b(bridge, *model, prompt_b, sampled);
        bool saw_parked = false;
        size_t one_pass = 0;
        while (!a.finished() || !b.finished()) {
            a.next(3);
            if (one_pass++ % 2 == 0) LOOM_CHECK(loom::text::generate(bridge, *model, prompt_c, greedy) == want_c);
            b.next(2);
            if (!a.finished()) {
                // b ran last, so a's rows are parked on the host -- or a owns a cache and parks nothing.
                saw_parked = saw_parked || a.state().parked_bytes() > 0;
                LOOM_CHECK(park ? a.state().cache_bytes() == 0 : a.state().parked_bytes() == 0);
            }
        }
        std::fprintf(stderr, "%s: a %s, b %s\n", park ? "park" : "per-stream",
                     a.tokens() == want_a ? "ok" : "DIFFERS", b.tokens() == want_sampled ? "ok" : "DIFFERS");
        LOOM_CHECK(a.tokens() == want_a);
        LOOM_CHECK(b.tokens() == want_sampled);
        LOOM_CHECK(saw_parked == park);
        if (!park) LOOM_CHECK(a.state().cache_bytes() > 0);
    }

    // An earlier export: the one-pass loop only, refused by the door.
    if (const char* old = loom_test::fixture_env("LOOM_STREAM_LM_OLD_GGUF")) {
        auto old_model = loom::GgufModel::load(old, backend.get());
        loom::Session old_session(*old_model, backend.get());
        LOOM_CHECK(!loom::text::GenerationStream::supported(old_session.bridge()));
        LOOM_CHECK_THROWS(loom::text::GenerationStream(old_session.bridge(), *old_model, prompt_a, greedy),
                          loom::LoadError);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
