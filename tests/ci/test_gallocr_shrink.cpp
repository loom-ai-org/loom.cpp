// The compute buffer is given back when a WINDOW of calls stops needing it (BACKLOG.md P4.0.16,
// ADR-072).
//
// gallocr grows and never shrinks -- `ggml_gallocr_reserve_n_impl` reallocates a chunk only when
// `new_chunk_size > cur_chunk_size` -- and since P4.0.13 the builder that ran the prefill is the builder
// that serves every decode step. So before P4.0.16, a generation held its prefill-sized compute buffer
// for its whole length: measured at 513.2 MiB held where 1.0 MiB was needed, on gemma-3-270m-it at a
// 512-token prefill.
//
// P4.0.16 then shrank on the FIRST small call after a big one, which a TTS module serving sentences of
// mixed length turned into a free-and-refault on every switch (38 740 minor faults per long VITS call
// against 1 372). ADR-072 waits for kShrinkWindow under-used calls in a row instead, and cases 5 and 6
// are the two sides of that: an alternating workload never shrinks, a burst followed by quiet does.
//
// This is the sibling of `test_gallocr_reserve_reuse.cpp`, and deliberately states the OPPOSITE
// contract, so the pair reads as the one decision it is: reserve() means "hold the worst case", and a
// builder that was never told a worst case gives memory back instead. Case 4 is what keeps them from
// contradicting each other in practice; cases 7-9 are the host's own two controls.
//
// Uses the same attention-free toy fixture as its sibling: the numbers are tiny, but "did it shrink"
// and "after how many calls" are exactly as answerable at 16 tokens as at 512. Counted rather than
// timed throughout -- a wall-clock assertion would be a flaky test of the machine.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cstdio>
#include <cstring>
#include <variant>
#include <vector>

namespace {

constexpr uint32_t W = loom::GraphBuilder::kShrinkWindow;

// Runs the toy at `n_tokens` with deterministic token ids and returns its logits. Goes through build(),
// so it is a CALL for the shrink policy like any other.
std::vector<float> run(loom::GraphBuilder& builder, uint32_t n_tokens) {
    const auto& r = builder.build({{"n_tokens", n_tokens}, {"n_past", 0}});
    std::vector<int32_t> tokens(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) tokens[i] = static_cast<int32_t>((i * 5 + 1) % 6);
    ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens.data(), 0, tokens.size() * sizeof(int32_t));
    builder.compute();
    std::vector<float> out(ggml_nelements(r.output));
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

bool same_bytes(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/builder_test.gguf";
    auto model = loom::GgufModel::load(path, backend.get());
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    // The answers every rebuilt, shrunk or released graph below must still give, from a builder that
    // has done nothing but this one call.
    std::vector<float> want_small, want_big;
    {
        loom::GraphBuilder fresh(topo, *model, backend.get());
        want_small = run(fresh, 1);
    }
    {
        loom::GraphBuilder fresh(topo, *model, backend.get());
        want_big = run(fresh, 64);
    }

    // --- 1. A big call then small ones: the buffer is HELD for W-1 small calls and given back on the
    // W-th. The small calls after the first are all reuses, so this is also the case where the
    // shrink is triggered from the hit path -- and the graph that hit turns into must compute the same
    // bytes as a builder that never saw the big call. ---
    {
        loom::GraphBuilder builder(topo, *model, backend.get());
        run(builder, 64);
        const size_t big = builder.buffer_size();
        LOOM_CHECK(big > 0);

        for (uint32_t call = 1; call < W; ++call) run(builder, 1);
        LOOM_CHECK(builder.shrinks() == 0);
        LOOM_CHECK(builder.buffer_size() == big);

        const std::vector<float> got = run(builder, 1);
        std::fprintf(stderr, "prefill-shaped %zu B -> decode-shaped %zu B after %u calls (%llu shrink(s))\n",
                      big, builder.buffer_size(), W, static_cast<unsigned long long>(builder.shrinks()));
        LOOM_CHECK(builder.shrinks() == 1);
        LOOM_CHECK(builder.buffer_size() < big);
        LOOM_CHECK(same_bytes(got, want_small));
        // One build for the big call, one for the first small one, one for the shrink; the rest reused.
        LOOM_CHECK(builder.builds() == 3);
    }

    // --- 2. A prefill-then-decode generation shrinks exactly ONCE. This toy has no KV cache, so every
    // decode step's `n_past` is a new key and every step REBUILDS -- the probe path, not the hit path --
    // and the window must still close once and stay closed: after the shrink, the buffer is the size the
    // steps need, so no later step counts as under-used. ---
    {
        loom::GraphBuilder builder(topo, *model, backend.get());
        builder.build({{"n_tokens", 64}, {"n_past", 0}});
        for (uint32_t step = 0; step < 32; ++step) {
            builder.build({{"n_tokens", 1}, {"n_past", step}});
        }
        std::fprintf(stderr, "33 builds -> %llu shrink(s)\n",
                      static_cast<unsigned long long>(builder.shrinks()));
        LOOM_CHECK(builder.builds() == 33);
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
    // declared a worst case must keep it -- otherwise the decode-shaped calls would hand back exactly
    // what reserve() was called to hold, and test_gallocr_reserve_reuse's contract would be a lie. ---
    {
        loom::GraphBuilder builder(topo, *model, backend.get());
        builder.reserve(64);
        const size_t reserved = builder.buffer_size();
        for (uint32_t call = 0; call < 2 * W; ++call) builder.build({{"n_tokens", 1}, {"n_past", call}});
        LOOM_CHECK(builder.shrinks() == 0);
        LOOM_CHECK(builder.buffer_size() == reserved);
    }

    // --- 5. THE CASE ADR-072 EXISTS FOR: alternating a long and a short shape never shrinks, so the
    // long call never pays to fault its buffer back in. Under P4.0.16 this shrank on every switch. ---
    {
        loom::GraphBuilder builder(topo, *model, backend.get());
        run(builder, 64);
        const size_t big = builder.buffer_size();
        bool buffer_moved = false;
        for (int i = 0; i < 4 * static_cast<int>(W); ++i) {
            const std::vector<float> got = run(builder, i % 2 == 0 ? 1 : 64);
            LOOM_CHECK(same_bytes(got, i % 2 == 0 ? want_small : want_big));
            if (builder.buffer_size() != big) buffer_moved = true;
        }
        std::fprintf(stderr, "alternating 1/64 tokens x%u -> %llu shrink(s)\n", 4 * W,
                      static_cast<unsigned long long>(builder.shrinks()));
        LOOM_CHECK(builder.shrinks() == 0);
        LOOM_CHECK(!buffer_moved);
    }

    // --- 6. A burst then quiet traffic of VARYING small shapes (every call a rebuild, so every call
    // is measured by the probe): the shrink lands on exactly the W-th quiet call. A single call that
    // needs more than half the buffer resets the count -- the window is consecutive, not cumulative. ---
    {
        loom::GraphBuilder builder(topo, *model, backend.get());
        run(builder, 64);
        for (uint32_t call = 1; call < W; ++call) run(builder, 1 + call % 3);
        LOOM_CHECK(builder.shrinks() == 0);
        run(builder, 60);   // needs more than half of the 64-token buffer: the window starts over
        for (uint32_t call = 1; call < W; ++call) run(builder, 1 + call % 3);
        LOOM_CHECK(builder.shrinks() == 0);
        // A shape the loop above did not end on, so this call too is a rebuild rather than a reuse.
        LOOM_CHECK(same_bytes(run(builder, 1), want_small));
        LOOM_CHECK(builder.shrinks() == 1);
    }

    // --- 7. hold_compute_buffer(true): no automatic shrink however long the quiet lasts, and the buffer
    // is the one the largest call grew it to. Releasing the hold restarts the window from zero rather
    // than counting the calls made while held. ---
    {
        loom::GraphBuilder builder(topo, *model, backend.get());
        builder.hold_compute_buffer(true);
        run(builder, 64);
        const size_t big = builder.buffer_size();
        for (uint32_t call = 0; call < 3 * W; ++call) run(builder, 1);
        LOOM_CHECK(builder.shrinks() == 0);
        LOOM_CHECK(builder.buffer_size() == big);

        builder.hold_compute_buffer(false);
        for (uint32_t call = 1; call < W; ++call) run(builder, 1);
        LOOM_CHECK(builder.shrinks() == 0);
        run(builder, 1);
        LOOM_CHECK(builder.shrinks() == 1);
    }

    // --- 8. release_compute_buffer(): everything goes now, the next call rebuilds at its own size and
    // computes the same bytes, and the release is not counted as a policy shrink. Releasing a builder
    // that never built is a no-op, not a crash. ---
    {
        loom::GraphBuilder never_built(topo, *model, backend.get());
        never_built.release_compute_buffer();
        LOOM_CHECK(never_built.buffer_size() == 0);

        loom::GraphBuilder builder(topo, *model, backend.get());
        run(builder, 64);
        builder.release_compute_buffer();
        LOOM_CHECK(builder.buffer_size() == 0);
        LOOM_CHECK(builder.shrinks() == 0);
        LOOM_CHECK(same_bytes(run(builder, 1), want_small));
        LOOM_CHECK(builder.buffer_size() > 0);
        // The same shape again after a release is a rebuild, not a reuse of the graph it dropped.
        builder.release_compute_buffer();
        LOOM_CHECK(same_bytes(run(builder, 1), want_small));
        LOOM_CHECK(builder.builds() == 3);
        LOOM_CHECK(builder.reuses() == 0);
    }

    // --- 9. The bridge-wide controls, and the one thing they add over the builder's: a hold set before
    // a module has ever run must reach the builder that run creates. ---
    {
        const char* script = R"LUA(
            function infer(inputs)
                local n = #inputs.tokens
                return loom.run_subgraph('toy', {n_tokens = n, n_past = 0}, {tokens = inputs.tokens})
            end
        )LUA";
        auto call = [](loom::LoomLuaBridge& bridge, size_t n) {
            std::vector<double> tokens(n);
            for (size_t i = 0; i < n; ++i) tokens[i] = static_cast<double>((i * 5 + 1) % 6);
            return std::get<std::vector<double>>(bridge.call("infer", {{"tokens", tokens}}));
        };

        loom::LoomLuaBridge bridge(backend.get());
        bridge.register_module("toy", *model, loom::GraphTopology::parse(model->topology_json()));
        bridge.load_script(script);
        LOOM_CHECK(bridge.compute_buffer_bytes() == 0);

        bridge.hold_compute_buffers(true);   // before the module's builder exists
        call(bridge, 64);
        const size_t big = bridge.compute_buffer_bytes();
        LOOM_CHECK(big > 0);
        std::vector<double> small_held;
        for (uint32_t i = 0; i < 2 * W; ++i) small_held = call(bridge, 1);
        LOOM_CHECK(bridge.compute_buffer_bytes() == big);

        bridge.release_compute_buffers();
        LOOM_CHECK(bridge.compute_buffer_bytes() == 0);
        const std::vector<double> small_after = call(bridge, 1);
        LOOM_CHECK(small_after == small_held);
        LOOM_CHECK(bridge.compute_buffer_bytes() > 0);
        LOOM_CHECK(bridge.compute_buffer_bytes() < big);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
