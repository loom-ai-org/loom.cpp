// GraphBuilder's `ggml_backend_sched` path (BACKLOG.md P4.7), run without a GPU.
//
// **What this pins, and what it cannot.** A hybrid builder is one whose `Backends` names two DIFFERENT
// backends, and the scheduler machinery -- create, size, allocate, split, compute, reuse, release -- is
// the same machinery whether the second backend is a Vulkan device or, as here, a second CPU. So this
// test constructs `Backends{cpu_a, cpu_b}`: an arrangement no host would ever ask for, and the only one
// that exercises that machinery on a machine with no device in it. Everything below is therefore about
// PLUMBING -- that the scheduled path allocates, runs, reuses and yields the same numbers as the plain
// gallocr path.
//
// What it deliberately does NOT pin is the fallback itself: with two CPU backends every op is supported
// everywhere, so nothing is forced across a split. "These custom ops ran on the CPU while the matmuls
// ran on the device" is a claim about a real device and lives in tests/gate/test_e2e_device_parity.cpp,
// which skips when there is none.
//
// The parity assertion is EXACT, not approximate, and that is not an accident of the two backends being
// the same kind: a scheduled graph must compute the same graph, and any difference here would be the
// scheduler mis-splitting or mis-allocating rather than arithmetic drift.
//
// **Measured limit of the stand-in.** Replacing `ggml_backend_sched_graph_compute` with the plain
// `ggml_backend_graph_compute` in GraphBuilder::compute() -- i.e. running a scheduled graph as though it
// were not one -- does NOT turn this test red, because two CPU backends put every buffer in host memory
// and the wrong call still finds it. The parity comparison itself IS live (feeding the scheduled run
// different tokens fails it), so what this test proves is that the scheduled path reaches the same
// arithmetic, not that the scheduled CALL is the one being made. Only a real device can prove that,
// which is one more thing the gate test is for.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

constexpr int kNVocab = 6;

std::vector<float> run_once(loom::GgufModel& model, const loom::GraphTopology& topo,
                             loom::Backends backends, const std::vector<int32_t>& tokens) {
    loom::GraphBuilder builder(topo, model, backends);
    const auto& r = builder.build({{"n_tokens", static_cast<double>(tokens.size())}, {"n_past", 0.0}});
    ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens.data(), 0,
                             tokens.size() * sizeof(int32_t));
    builder.compute();
    std::vector<float> out(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

} // namespace

int main() {
    ggml_backend_ptr cpu_a(ggml_backend_cpu_init());
    ggml_backend_ptr cpu_b(ggml_backend_cpu_init());
    LOOM_CHECK(cpu_a != nullptr && cpu_b != nullptr);
    LOOM_CHECK(cpu_a.get() != cpu_b.get());

    const loom::Backends single = cpu_a.get();
    const loom::Backends pair(cpu_a.get(), cpu_b.get());
    LOOM_CHECK(!single.hybrid());
    LOOM_CHECK(pair.hybrid());

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/builder_test.gguf";
    auto model = loom::GgufModel::load(path, single);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    const std::vector<int32_t> tokens{1, 4, 2, 0, 5};

    // --- The scheduled graph computes the same numbers as the gallocr one ---------------------------
    const std::vector<float> expected = run_once(*model, topo, single, tokens);
    const std::vector<float> scheduled = run_once(*model, topo, pair, tokens);
    LOOM_CHECK(expected.size() == static_cast<size_t>(kNVocab) * tokens.size());
    LOOM_CHECK(scheduled.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        LOOM_CHECK(scheduled[i] == expected[i]);
    }

    // --- ...and the scheduler reports on itself ------------------------------------------------------
    {
        loom::GraphBuilder builder(topo, *model, pair);
        const auto& r = builder.build({{"n_tokens", static_cast<double>(tokens.size())}, {"n_past", 0.0}});
        ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens.data(), 0,
                                 tokens.size() * sizeof(int32_t));
        builder.compute();

        LOOM_CHECK(builder.buffer_size() > 0);
        // One split means "ran entirely on one backend", which is what two interchangeable CPUs get. The
        // assertion worth making here is only that the scheduler HAS a plan, not how many pieces it is
        // in -- how many pieces is the device question, and it is asked in the gate test.
        LOOM_CHECK(builder.splits() >= 1);
        const std::vector<std::string> assignment = builder.node_backends();
        LOOM_CHECK(assignment.size() == static_cast<size_t>(ggml_graph_n_nodes(r.graph)));
        for (const std::string& name : assignment) LOOM_CHECK(!name.empty());

        // A CPU-only builder has no scheduler and so nothing to report -- the same two accessors are
        // deliberately empty/zero there rather than fabricating a single-backend "assignment".
        loom::GraphBuilder cpu_only(topo, *model, single);
        cpu_only.build({{"n_tokens", 1.0}, {"n_past", 0.0}});
        LOOM_CHECK(cpu_only.splits() == 0);
        LOOM_CHECK(cpu_only.node_backends().empty());
    }

    // --- Graph reuse survives the scheduler ----------------------------------------------------------
    // P4.0.13's whole claim is that a fixed-shape loop rebuilds exactly once. It would be easy for the
    // scheduled path to keep that at the ggml_cgraph level and lose it underneath -- a reset and a
    // re-split per iteration -- so this asserts the counters AND that every iteration still answers.
    {
        loom::GraphBuilder builder(topo, *model, pair);
        std::vector<float> first;
        for (int iteration = 0; iteration < 5; ++iteration) {
            const auto& r = builder.build({{"n_tokens", static_cast<double>(tokens.size())}, {"n_past", 0.0}});
            ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens.data(), 0,
                                     tokens.size() * sizeof(int32_t));
            builder.compute();
            std::vector<float> out(static_cast<size_t>(ggml_nelements(r.output)));
            ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
            if (iteration == 0) first = out;
            for (size_t i = 0; i < out.size(); ++i) LOOM_CHECK(out[i] == expected[i]);
        }
        LOOM_CHECK(builder.builds() == 1);
        LOOM_CHECK(builder.reuses() == 4);
        LOOM_CHECK(!first.empty());
        // The gallocr shrink has no scheduler equivalent and none is faked: this reports 0 by
        // construction, which is worth stating so a later reading of `shrinks()` is not misread as
        // "nothing needed shrinking".
        LOOM_CHECK(builder.shrinks() == 0);
    }

    // --- A rebuild at a new shape releases the previous allocation and takes a new one -----------------
    // ggml_backend_sched_alloc_graph asserts it is not called twice without a reset in between, so a
    // builder that walks through several shapes is the cheapest way to state that the reset is there.
    {
        loom::GraphBuilder builder(topo, *model, pair);
        for (uint32_t n_tokens : {1u, 7u, 3u, 12u, 2u}) {
            const std::vector<int32_t> step(n_tokens, 1);
            const auto& r = builder.build({{"n_tokens", static_cast<double>(n_tokens)}, {"n_past", 0.0}});
            ggml_backend_tensor_set(r.input_tensors.at("tokens"), step.data(), 0,
                                     step.size() * sizeof(int32_t));
            builder.compute();
            LOOM_CHECK(r.output->ne[0] == kNVocab);
            LOOM_CHECK(r.output->ne[1] == static_cast<int64_t>(n_tokens));
        }
        LOOM_CHECK(builder.builds() == 5);
    }

    // --- compute() before build() is an error, not a crash ---------------------------------------------
    {
        loom::GraphBuilder builder(topo, *model, pair);
        LOOM_CHECK_THROWS(builder.compute(), loom::Error);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
