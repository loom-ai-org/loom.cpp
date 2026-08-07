// The bit-identical-to-rebuild regression test BACKLOG.md P4.0.13 asks for: a GraphBuilder now RETAINS
// the last graph it built and hands it straight back when build() is called again with the same axes,
// so every result this file compares has to be indistinguishable from the one a from-scratch rebuild
// would have produced -- not "close", identical, since the two run the same kernels over the same
// bytes and any difference at all would mean the reused graph is not the graph.
//
// This is the *engine-side* half of the finding tests/test_graph_reuse_safety.cpp documents. That file
// pins down, in plain ggml, why reusing a graph used to be dangerous: ggml_gallocr may hand a computed
// tensor the same buffer as one of the graph's own declared INPUTS, so a second compute that doesn't
// rewrite every input silently reads a previous pass's intermediate. GraphBuilder no longer exposes
// that hazard, because it allocates the declared inputs in their own persistent context and backend
// buffer -- outside the gallocr pool, the same seam KvCache/ConvStateCache/OutputStore use -- and
// gallocr never places anything on top of a tensor whose data is already set. The last test below
// asserts exactly that, and is the one that fails if a future ggml changes it.
//
// Run against the toy LLM fixture rather than a synthetic topology on purpose: it has a KV cache, a
// repeat_for block, RoPE and an f32 mask input, so "the graph" here is a real one with real side
// effects rather than a single op whose reuse could not go wrong.
//
// **BACKLOG.md P4.0.15 inverts one of the assertions below.** This file used to state that a decode
// sequence rebuilds at every step, because `n_past` was baked into the KV write's byte offset. It is a
// cell-index tensor now, and `n_kv` is rounded up to a bucket, so consecutive decode steps ARE the same
// graph -- and the two tests at the end are what that item is gated on: the loop reuses, and the cache
// cells inside the bucket but past the end of the sequence contribute exactly nothing even when they
// hold another generation's numbers.

#include "ggml_test_helpers.h"
#include "test_util.h"

#include "loom/loom.h"

#include <ggml-alloc.h>
#include <ggml-cpu.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kNCtxMax = 32;

// Sized straight from the fixture's own hparams, exactly as Generator's constructor does. Not
// make_kv_cache(), which additionally wants a `loom.kv_cache_size` KV the toy fixture predates -- the
// capacity is this test's choice here, the way n_ctx_max is a GenerationConfig field there.
std::unique_ptr<loom::KvCache> make_cache(loom::GgufModel& model, ggml_backend_t backend,
                                           uint32_t kv_size = kNCtxMax) {
    return std::make_unique<loom::KvCache>(
        model.hparam_u32("n_layer"),
        model.hparam_u32("n_head_kv") * model.hparam_u32("n_embd_head_k"),
        model.hparam_u32("n_head_kv") * model.hparam_u32("n_embd_head_v"),
        kv_size, backend);
}

// One forward pass, written the way Generator::write_inputs writes one: every declared input, every
// call. See BACKLOG.md P4.0.13 -- with the inputs living outside gallocr this is no longer what stands
// between reuse and corruption, but it is still what a driver means by "run this shape".
std::vector<float> run_once(loom::GraphBuilder& builder, ggml_backend_t backend,
                            const std::vector<int32_t>& tokens, uint32_t n_past) {
    const auto n_tokens = static_cast<uint32_t>(tokens.size());

    const loom::GraphBuilder::BuildResult& r =
        builder.build({{"n_tokens", static_cast<double>(n_tokens)}, {"n_past", static_cast<double>(n_past)}});

    ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens.data(), 0, n_tokens * sizeof(int32_t));

    std::vector<int32_t> positions(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) positions[i] = static_cast<int32_t>(n_past + i);
    ggml_backend_tensor_set(r.input_tensors.at("positions"), positions.data(), 0, n_tokens * sizeof(int32_t));

    // The mask's width is the TENSOR's, not `n_past + n_tokens`: since P4.0.15 the builder rounds n_kv
    // up to a bucket, and the extra columns have to be masked off or the graph reads whatever the cache
    // happens to hold there. The `j <= n_past + i` rule already writes -inf past the sequence, so this
    // is the same loop over a wider row -- exactly what Generator::write_inputs does.
    ggml_tensor* mask_t = r.input_tensors.at("kq_mask");
    const auto n_kv = static_cast<uint32_t>(mask_t->ne[0]);
    std::vector<float> mask(static_cast<size_t>(n_kv) * n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        for (uint32_t j = 0; j < n_kv; ++j) {
            mask[static_cast<size_t>(i) * n_kv + j] =
                (j <= n_past + i) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }
    ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(float));

    ggml_backend_graph_compute(backend, r.graph);

    std::vector<float> logits(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, logits.data(), 0, logits.size() * sizeof(float));
    return logits;
}

bool bit_identical(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        // Bit comparison, not ==: two NaNs would compare unequal and two zeros of different sign
        // equal, and neither is what "the reused graph produced the same bytes" means.
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) return false;
    }
    return true;
}

// A fixed-shape loop -- an ODE sampler's steps, an LSTM's timesteps, one stage of a chained module --
// is the case reuse exists for: the axes never move, only the data does. Every iteration through the
// retained graph must equal the same iteration through a builder that has only ever built once.
void test_fixed_shape_loop_matches_rebuild(loom::GgufModel& model, const loom::GraphTopology& topo,
                                            ggml_backend_t backend) {
    const std::vector<std::vector<int32_t>> steps = {
        {1, 2, 3, 4}, {5, 6, 7, 8}, {2, 2, 2, 2}, {9, 0, 11, 3}, {1, 2, 3, 4},
    };

    auto reused_kv = make_cache(model, backend);
    loom::GraphBuilder reused(topo, model, backend, reused_kv.get());

    std::vector<std::vector<float>> reused_out, rebuilt_out;
    for (const auto& tokens : steps) {
        reused_out.push_back(run_once(reused, backend, tokens, /*n_past=*/0));
    }
    for (const auto& tokens : steps) {
        // The oracle: a builder that has never seen another shape, and a cache in the same state the
        // reused one's was in at this step (n_past=0 rewrites cells [0, 4) every time, so a fresh
        // cache per iteration is the same cache the reused run had).
        auto kv = make_cache(model, backend);
        loom::GraphBuilder fresh(topo, model, backend, kv.get());
        rebuilt_out.push_back(run_once(fresh, backend, tokens, /*n_past=*/0));
        LOOM_CHECK(fresh.builds() == 1 && fresh.reuses() == 0);
    }

    for (size_t i = 0; i < steps.size(); ++i) {
        LOOM_CHECK(bit_identical(reused_out[i], rebuilt_out[i]));
    }
    // ...and the comparison above can actually fail: consecutive steps really do produce different
    // logits, so a reused graph that returned its previous pass's numbers would be caught rather than
    // matching by construction. The last step repeats the first's tokens and must reproduce it exactly,
    // which is the same claim from the other side -- nothing about a reused graph is path-dependent.
    LOOM_CHECK(!bit_identical(reused_out[0], reused_out[1]));
    LOOM_CHECK(bit_identical(reused_out[0], reused_out[steps.size() - 1]));
    // The point of the whole item: five calls, one graph.
    LOOM_CHECK(reused.builds() == 1);
    LOOM_CHECK(reused.reuses() == steps.size() - 1);
}

// The other half: a prefill followed by a decode loop, where `n_past` moves every step. Its job is to
// prove that neither moving the declared inputs out of the gallocr pool (P4.0.13) nor addressing the KV
// write by cell index (P4.0.15) changed what the engine computes -- the same sequence must still agree,
// step for step, with one driven through a builder that is thrown away and rebuilt between every call.
void test_changing_shape_sequence_matches_rebuild(loom::GgufModel& model, const loom::GraphTopology& topo,
                                                   ggml_backend_t backend) {
    const std::vector<int32_t> prompt = {1, 2, 3};
    const std::vector<int32_t> decoded = {4, 5, 6};

    auto shared_kv = make_cache(model, backend);
    loom::GraphBuilder shared(topo, model, backend, shared_kv.get());
    std::vector<std::vector<float>> shared_out;
    shared_out.push_back(run_once(shared, backend, prompt, /*n_past=*/0));
    for (size_t s = 0; s < decoded.size(); ++s) {
        shared_out.push_back(run_once(shared, backend, {decoded[s]}, static_cast<uint32_t>(prompt.size() + s)));
    }
    // Four calls, TWO builds: the prefill is one shape (n_tokens = 3) and every decode step is the
    // other (n_tokens = 1), because n_past has stopped being part of the graph and n_kv rounds to the
    // same bucket for all three of them. This is the assertion P4.0.15 inverted -- it read
    // `builds() == 4 && reuses() == 0` when a decode step's write offset was baked in.
    LOOM_CHECK(shared.builds() == 2 && shared.reuses() == 2);

    // Same sequence against the same cache, but every step gets a builder that has never built before.
    auto per_step_kv = make_cache(model, backend);
    std::vector<std::vector<float>> per_step_out;
    {
        loom::GraphBuilder b(topo, model, backend, per_step_kv.get());
        per_step_out.push_back(run_once(b, backend, prompt, /*n_past=*/0));
    }
    for (size_t s = 0; s < decoded.size(); ++s) {
        loom::GraphBuilder b(topo, model, backend, per_step_kv.get());
        per_step_out.push_back(run_once(b, backend, {decoded[s]}, static_cast<uint32_t>(prompt.size() + s)));
    }

    for (size_t i = 0; i < shared_out.size(); ++i) {
        LOOM_CHECK(bit_identical(shared_out[i], per_step_out[i]));
    }
}

// Exactly ONE graph is retained, and it is the most recent -- not an LRU keyed by shape. A retained
// OutputStore is reshaped by the build that fills it, so only the last build's ggml_cpy destinations
// are guaranteed to still be the store's current tensors; going back to an earlier shape therefore has
// to rebuild rather than resurrect.
void test_cache_holds_exactly_one_graph(loom::GgufModel& model, const loom::GraphTopology& topo,
                                         ggml_backend_t backend) {
    auto kv = make_cache(model, backend);
    loom::GraphBuilder builder(topo, model, backend, kv.get());

    const loom::GraphBuilder::BuildResult& first = builder.build({{"n_tokens", 4}, {"n_past", 0}});
    // The bucketing itself, stated once and directly, since everything else here follows from it: a
    // 4-token prefill is built at a mask spanning a whole bucket, not 4 cells, and the builder reports
    // both widths so a caller can place a 4-wide mask into it.
    LOOM_CHECK(first.input_tensors.at("kq_mask")->ne[0] ==
                static_cast<int64_t>(loom::GraphBuilder::kKvBucket));
    LOOM_CHECK(first.n_kv_real == 4);
    LOOM_CHECK(first.kv_padded_inputs == std::vector<std::string>{"kq_mask"});

    ggml_cgraph* a1 = first.graph;
    ggml_cgraph* a2 = builder.build({{"n_tokens", 4}, {"n_past", 0}}).graph;
    LOOM_CHECK(a1 == a2);
    LOOM_CHECK(builder.builds() == 1 && builder.reuses() == 1);

    // A different n_tokens is a different graph even at the same (bucketed) n_kv -- `n_tokens` is a
    // real shape on q, on the mask and on every intermediate, and P4.0.15 removed n_past from the key,
    // not that.
    ggml_cgraph* b1 = builder.build({{"n_tokens", 1}, {"n_past", 3}}).graph;
    LOOM_CHECK(builder.builds() == 2);

    ggml_cgraph* a3 = builder.build({{"n_tokens", 4}, {"n_past", 0}}).graph;
    LOOM_CHECK(builder.builds() == 3); // evicted by b1, so this is a rebuild
    (void)b1;
    (void)a3;
}

// **The item's own assertion (BACKLOG.md P4.0.15): a decode loop long enough to cross bucket
// boundaries reuses its graph, and reuses it without changing a single output bit.**
//
// The oracle is the same one the shape-change test uses -- a builder per step, which cannot reuse
// anything by construction -- driven against its own cache in the same state. If the cell-index write
// landed at the wrong cell, or a reused graph kept the previous step's indices, the two would diverge
// at the first decode step and stay diverged, because a KV cache makes every later step depend on it.
void test_decode_loop_reuses_and_matches_rebuild(loom::GgufModel& model, const loom::GraphTopology& topo,
                                                  ggml_backend_t backend) {
    // Sized off the bucket itself rather than off the number 32, so this states the contract instead of
    // restating the constant. A cache of three buckets, and enough steps to CROSS exactly one boundary
    // rather than spend the whole loop inside one: n_kv runs from 7 to prompt + kSteps, which lands in
    // (kKvBucket, 2*kKvBucket]. The graph is therefore rebuilt exactly once in the middle -- and if the
    // cell indices were somehow tied to the graph rather than rewritten under it, that rebuild is where
    // the damage would show.
    constexpr uint32_t kBucket = loom::GraphBuilder::kKvBucket;
    constexpr uint32_t kCacheCells = 3 * kBucket;
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6};
    constexpr int kSteps = static_cast<int>(kBucket) + 8;

    auto reused_kv = make_cache(model, backend, kCacheCells);
    loom::GraphBuilder reused(topo, model, backend, reused_kv.get());
    auto oracle_kv = make_cache(model, backend, kCacheCells);

    std::vector<std::vector<float>> reused_out, oracle_out;
    reused_out.push_back(run_once(reused, backend, prompt, /*n_past=*/0));
    {
        loom::GraphBuilder fresh(topo, model, backend, oracle_kv.get());
        oracle_out.push_back(run_once(fresh, backend, prompt, /*n_past=*/0));
    }
    for (int s = 0; s < kSteps; ++s) {
        const auto n_past = static_cast<uint32_t>(prompt.size() + s);
        // A token that changes from step to step, so a stale reused graph would show up as a stale
        // logit rather than being hidden by feeding the same input twice. Wrapped into the toy
        // fixture's 16-entry vocabulary -- GET_ROWS bounds-checks the id.
        const std::vector<int32_t> step = {static_cast<int32_t>(7 + s) % 16};
        reused_out.push_back(run_once(reused, backend, step, n_past));
        loom::GraphBuilder fresh(topo, model, backend, oracle_kv.get());
        oracle_out.push_back(run_once(fresh, backend, step, n_past));
        LOOM_CHECK(fresh.builds() == 1 && fresh.reuses() == 0);
    }

    for (size_t i = 0; i < reused_out.size(); ++i) {
        LOOM_CHECK(bit_identical(reused_out[i], oracle_out[i]));
    }
    // ...and the comparison can fail: consecutive decode steps really do produce different logits.
    LOOM_CHECK(!bit_identical(reused_out[1], reused_out[2]));
    // Three graphs for 1 + kSteps calls: the prefill shape, the decode shape inside the first bucket,
    // and the decode shape inside the second. Everything else is reused -- 38 of 41 calls.
    LOOM_CHECK(reused.builds() == 3);
    LOOM_CHECK(reused.reuses() == static_cast<uint64_t>(1 + kSteps) - 3);
}

// **Padded cells contribute exactly zero, and not because they are zero.** Rounding n_kv up to a bucket
// means attention reads cells past the end of the sequence; the claim that they cost nothing rests
// entirely on the mask saying -inf there. A cache full of zeros cannot tell the two apart -- a zeroed
// cell reached through a *finite* mask would still contribute a zero -- so this primes the tail of the
// cache with large non-zero K and V first, and requires the run to be bit-identical to the same run
// against a cache whose tail is untouched.
void test_padded_cells_contribute_nothing(loom::GgufModel& model, const loom::GraphTopology& topo,
                                           ggml_backend_t backend) {
    const std::vector<int32_t> prompt = {1, 2, 3};
    const std::vector<int32_t> decoded = {4, 5, 6};

    const auto run_sequence = [&](loom::KvCache& kv) {
        loom::GraphBuilder builder(topo, model, backend, &kv);
        std::vector<std::vector<float>> out;
        out.push_back(run_once(builder, backend, prompt, /*n_past=*/0));
        for (size_t s = 0; s < decoded.size(); ++s) {
            out.push_back(run_once(builder, backend, {decoded[s]}, static_cast<uint32_t>(prompt.size() + s)));
        }
        return out;
    };

    auto clean_kv = make_cache(model, backend);
    const std::vector<std::vector<float>> clean = run_sequence(*clean_kv);

    // The same cache geometry, with cells [6, kNCtxMax) -- everything the bucket reaches but the
    // sequence never writes -- filled with values a finite mask could not possibly hide.
    auto dirty_kv = make_cache(model, backend);
    const uint32_t n_used = static_cast<uint32_t>(prompt.size() + decoded.size());
    const uint32_t n_garbage = kNCtxMax - n_used;
    const uint32_t n_embd_k = model.hparam_u32("n_head_kv") * model.hparam_u32("n_embd_head_k");
    const uint32_t n_embd_v = model.hparam_u32("n_head_kv") * model.hparam_u32("n_embd_head_v");
    for (uint32_t il = 0; il < dirty_kv->n_layer(); ++il) {
        loom_test::GgmlScratch s(backend);
        ggml_tensor* k_cur = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, n_embd_k, n_garbage);
        ggml_tensor* v_cur = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, n_embd_v, n_garbage);
        ggml_set_input(k_cur);
        ggml_set_input(v_cur);
        ggml_tensor* cells = loom::KvCache::new_cell_index(s.ctx.get(), n_garbage);
        ggml_cgraph* gf = ggml_new_graph(s.ctx.get());
        ggml_build_forward_expand(gf, dirty_kv->write_k(s.ctx.get(), k_cur, il, cells));
        ggml_build_forward_expand(gf, dirty_kv->write_v(s.ctx.get(), v_cur, il, cells));
        ggml_gallocr_alloc_graph(s.galloc.get(), gf);
        loom_test::set_f32(k_cur, std::vector<float>(static_cast<size_t>(n_embd_k) * n_garbage, 1000.0f));
        loom_test::set_f32(v_cur, std::vector<float>(static_cast<size_t>(n_embd_v) * n_garbage, -1000.0f));
        loom::KvCache::fill_cell_index(cells, /*n_past=*/n_used);
        s.compute(gf);
    }

    const std::vector<std::vector<float>> dirty = run_sequence(*dirty_kv);
    LOOM_CHECK(clean.size() == dirty.size());
    for (size_t i = 0; i < clean.size(); ++i) {
        LOOM_CHECK(bit_identical(clean[i], dirty[i]));
    }
    // And the priming really happened -- otherwise the loop above would be comparing two clean caches
    // and proving nothing. Read the tail back through the cache's own view.
    ggml_context_ptr read_ctx(ggml_init(ggml_init_params{16 * 1024, nullptr, /*no_alloc=*/true}));
    ggml_tensor* view = dirty_kv->read_k(read_ctx.get(), 0, kNCtxMax);
    std::vector<float> cells_k(static_cast<size_t>(ggml_nelements(view)));
    ggml_backend_tensor_get(view, cells_k.data(), 0, cells_k.size() * sizeof(float));
    LOOM_CHECK(cells_k[static_cast<size_t>(n_used) * n_embd_k] == 1000.0f);
}

// The hazard test_graph_reuse_safety.cpp documents, asserted absent here: no declared input may share
// an address with anything gallocr placed. It cannot, because the inputs are not in gallocr's pool at
// all -- they carry their own backend buffer, which is what makes handing the same graph back safe
// instead of merely fast. If this starts failing, reuse is no longer sound and BACKLOG.md P4.0.13's
// reasoning needs revisiting before the retained graph does.
void test_declared_inputs_are_never_aliased(loom::GgufModel& model, const loom::GraphTopology& topo,
                                             ggml_backend_t backend) {
    auto kv = make_cache(model, backend);
    loom::GraphBuilder builder(topo, model, backend, kv.get());
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", 4}, {"n_past", 0}});

    LOOM_CHECK(!r.input_tensors.empty());
    for (const auto& [name, input] : r.input_tensors) {
        LOOM_CHECK(input->data != nullptr);   // allocated before the graph was even built
        LOOM_CHECK(input->buffer != nullptr); // ...and on a real backend buffer, not a raw pointer
        bool aliased = false;
        for (int i = 0; i < ggml_graph_n_nodes(r.graph); ++i) {
            ggml_tensor* node = ggml_graph_node(r.graph, i);
            if (node != input && node->data == input->data) aliased = true;
        }
        LOOM_CHECK(!aliased);
        if (aliased) std::fprintf(stderr, "declared input '%s' was aliased by a graph node\n", name.c_str());
    }
}

} // namespace

int main() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    const std::string gguf_path = std::string(LOOM_TEST_FIXTURE_DIR) + "/toy_llm.gguf";
    auto model = loom::GgufModel::load(gguf_path, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    test_fixed_shape_loop_matches_rebuild(*model, topo, backend.get());
    test_changing_shape_sequence_matches_rebuild(*model, topo, backend.get());
    test_cache_holds_exactly_one_graph(*model, topo, backend.get());
    test_declared_inputs_are_never_aliased(*model, topo, backend.get());
    test_decode_loop_reuses_and_matches_rebuild(*model, topo, backend.get());
    test_padded_cells_contribute_nothing(*model, topo, backend.get());

    LOOM_TEST_REPORT_AND_RETURN();
}
