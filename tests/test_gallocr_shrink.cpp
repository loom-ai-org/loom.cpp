// The compute buffer is given back when a build stops needing it (BACKLOG.md P4.0.16).
//
// gallocr grows and never shrinks -- `ggml_gallocr_reserve_n_impl` reallocates a chunk only when
// `new_chunk_size > cur_chunk_size` -- and since P4.0.13 the builder that ran the prefill is the builder
// that serves every decode step. So before this, a generation held its prefill-sized compute buffer for
// its whole length: measured at 513.2 MiB held where 1.0 MiB was needed, on gemma-3-270m-it at a
// 512-token prefill.
//
// This is the sibling of `test_gallocr_reserve_reuse.cpp`, and deliberately states the OPPOSITE
// contract, so the pair reads as the one decision it is: reserve() means "hold the worst case", and a
// builder that was never told a worst case gives memory back instead. The last case here is what keeps
// them from contradicting each other in practice.
//
// Uses the same attention-free toy fixture as its sibling: the numbers are tiny, but "did it shrink"
// and "how many times did it probe" are exactly as answerable at 16 tokens as at 512.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cstdio>

int main() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/builder_test.gguf";
    auto model = loom::GgufModel::load(path, backend.get());
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    // --- 1. A big build then a small one gives the buffer back. ---
    {
        loom::GraphBuilder builder(topo, *model, backend.get());
        builder.build({{"n_tokens", 64}, {"n_past", 0}});
        const size_t big = builder.buffer_size();
        LOOM_CHECK(big > 0);

        builder.build({{"n_tokens", 1}, {"n_past", 0}});
        std::fprintf(stderr, "prefill-shaped %zu B -> decode-shaped %zu B (%llu shrink(s))\n",
                      big, builder.buffer_size(), static_cast<unsigned long long>(builder.shrinks()));
        LOOM_CHECK(builder.buffer_size() < big);
        LOOM_CHECK(builder.shrinks() == 1);
    }

    // --- 2. It costs ONE probe, not one per step. The check is a second planning pass over the graph,
    // so running it per build would be a real per-step cost; arming it on a growth is what makes a whole
    // generation pay for it once. Counted rather than timed -- a wall-clock assertion would be a flaky
    // test of the machine, and this is the property the design actually rests on. ---
    {
        loom::GraphBuilder builder(topo, *model, backend.get());
        builder.build({{"n_tokens", 64}, {"n_past", 0}});
        for (uint32_t step = 0; step < 32; ++step) {
            builder.build({{"n_tokens", 1}, {"n_past", step}});
        }
        std::fprintf(stderr, "33 builds -> %llu shrink(s)\n",
                      static_cast<unsigned long long>(builder.shrinks()));
        LOOM_CHECK(builder.shrinks() == 1);
    }

    // --- 3. A loop at one shape never shrinks at all: nothing grew, so nothing is oversized. ---
    {
        loom::GraphBuilder builder(topo, *model, backend.get());
        for (int i = 0; i < 8; ++i) {
            builder.build({{"n_tokens", 8}, {"n_past", 0}});
        }
        LOOM_CHECK(builder.shrinks() == 0);
        // ...and every call after the first was served from the retained graph, so this is also the
        // P4.0.13 reuse path confirming the shrink did not disturb it.
        LOOM_CHECK(builder.builds() == 1);
        LOOM_CHECK(builder.reuses() == 7);
    }

    // --- 4. reserve() wins. The two policies are opposites over the same buffer, and a caller that
    // declared a worst case must keep it -- otherwise the first decode-shaped call would hand back
    // exactly what reserve() was called to hold, and test_gallocr_reserve_reuse's contract would be a
    // lie. ---
    {
        loom::GraphBuilder builder(topo, *model, backend.get());
        builder.reserve(64);
        const size_t reserved = builder.buffer_size();
        builder.build({{"n_tokens", 1}, {"n_past", 0}});
        LOOM_CHECK(builder.shrinks() == 0);
        LOOM_CHECK(builder.buffer_size() == reserved);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
