// `$LOOM_PROFILE` attributes a run's time to the ops it actually ran (loom/core/profile.h).
//
// The contract pinned here is narrow on purpose: most of what a profiler emits is timings, and a test
// asserting on wall-clock numbers is a test of the machine it runs on. What IS checkable, and what the
// feature would be worthless -- or worse than worthless -- without:
//
//   1. **The node-by-node walk computes the same thing as the plain compute.** This is the claim that
//      matters. `profile::compute` runs each node through its own `ggml_graph_view`, and if that did not
//      hand a node the buffers it would have had mid-`ggml_backend_graph_compute`, the profiler would
//      quietly corrupt every run it observed -- the worst failure mode a measurement tool can have.
//      Compared BIT-EXACTLY rather than within a tolerance: it is the same arithmetic in the same order,
//      so anything but bit-identity is a real difference, and a tolerance would hide exactly the kind of
//      partial-staleness bug this exists to rule out.
//   2. **Every executed node lands in a bucket**, so the rollup cannot silently drop work.
//   3. **The env var routes GraphBuilder::compute in BOTH directions** -- profiling off records nothing,
//      profiling on records something. Checked by registering this binary twice with different
//      ENVIRONMENT rather than by re-execing, because `profile::enabled()` caches its answer on first
//      call and a single process therefore only ever gets one of the two answers.
//
// Deliberately not asserted: that any particular op is heaviest, or that the floor is any given size.
// Those are properties of the host, not of this code.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// The fixture declares one i32 input, "tokens". Rewritten before EVERY compute -- the discipline
// tests/test_graph_reuse_safety.cpp exists to enforce, and the only thing that makes two runs of one
// retained graph comparable at all.
void write_inputs(const loom::GraphBuilder::BuildResult& built) {
    for (const auto& entry : built.input_tensors) {
        const size_t n = ggml_nelements(entry.second);
        LOOM_CHECK(entry.second->type == GGML_TYPE_I32);
        std::vector<int32_t> ids(n);
        for (size_t i = 0; i < n; ++i) ids[i] = static_cast<int32_t>((i * 3 + 1) % 4);
        ggml_backend_tensor_set(entry.second, ids.data(), 0, n * sizeof(int32_t));
    }
}

std::vector<float> read_output(const loom::GraphBuilder::BuildResult& built) {
    std::vector<float> out(ggml_nelements(built.output));
    ggml_backend_tensor_get(built.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/builder_test.gguf";
    auto model = loom::GgufModel::load(path, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    loom::GraphBuilder builder(topo, *model, backend.get());
    const loom::GraphBuilder::BuildResult& built = builder.build({{"n_tokens", 8}, {"n_past", 0}});

    // --- 1. Node-by-node == whole-graph, bit for bit. ---
    //
    // Both halves call the ggml entry points directly rather than going through GraphBuilder::compute,
    // so this half of the test says nothing about the environment and runs identically in both
    // registrations. `profile::compute` is a plain public function; it does not consult `enabled()`.
    write_inputs(built);
    LOOM_CHECK(loom::profile::compute(backend.get(), built.graph) == GGML_STATUS_SUCCESS);
    const std::vector<float> node_by_node = read_output(built);

    write_inputs(built);
    LOOM_CHECK(ggml_backend_graph_compute(backend.get(), built.graph) == GGML_STATUS_SUCCESS);
    const std::vector<float> whole_graph = read_output(built);

    LOOM_CHECK(!whole_graph.empty());
    LOOM_CHECK(node_by_node.size() == whole_graph.size());
    for (size_t i = 0; i < whole_graph.size(); ++i) {
        LOOM_CHECK(std::memcmp(&node_by_node[i], &whole_graph[i], sizeof(float)) == 0);
    }
    std::fprintf(stderr, "%zu outputs, node-by-node bit-identical to whole-graph\n", whole_graph.size());

    // --- 2. The buckets account for every node that ran. ---
    const loom::profile::Totals totals = loom::profile::totals();
    LOOM_CHECK(totals.nodes > 0);
    LOOM_CHECK(totals.seconds > 0.0);
    uint64_t summed = 0;
    for (const loom::profile::Row& row : loom::profile::rows()) summed += row.calls;
    LOOM_CHECK(summed == totals.nodes);

    // The report renders on the shapes a real graph produces rather than throwing or truncating oddly.
    const std::string text = loom::profile::report();
    LOOM_CHECK(text.find("loom profile") != std::string::npos);
    LOOM_CHECK(text.find("by op") != std::string::npos);

    // --- 3. GraphBuilder::compute routes on $LOOM_PROFILE, in whichever direction this registration is
    // testing. Both directions are real failure modes: a branch stuck ON makes every shipped run pay
    // per-node dispatch, and one stuck OFF makes the feature silently do nothing. ---
    loom::profile::reset();
    LOOM_CHECK(loom::profile::totals().nodes == 0);
    write_inputs(built);
    builder.compute();
    const uint64_t recorded = loom::profile::totals().nodes;

    if (loom::profile::enabled()) {
        std::fprintf(stderr, "LOOM_PROFILE set: builder.compute() recorded %llu nodes\n",
                      static_cast<unsigned long long>(recorded));
        LOOM_CHECK(recorded > 0);
        // Same graph both times, so the profiled route must see exactly the node count the direct call
        // to profile::compute above did.
        LOOM_CHECK(recorded == totals.nodes);
    } else {
        std::fprintf(stderr, "LOOM_PROFILE unset: builder.compute() recorded %llu nodes\n",
                      static_cast<unsigned long long>(recorded));
        LOOM_CHECK(recorded == 0);
    }

    // And the output is still right whichever route ran.
    const std::vector<float> after = read_output(built);
    for (size_t i = 0; i < whole_graph.size(); ++i) {
        LOOM_CHECK(std::memcmp(&after[i], &whole_graph[i], sizeof(float)) == 0);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
