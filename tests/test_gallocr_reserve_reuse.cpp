// Confirms the worst-case-reserve-once contract from SPECIFICATION.md §5 / GraphBuilder::reserve():
// after reserving for a worst-case n_ctx_max, ordinary build() calls at smaller shapes don't grow the
// gallocr-managed compute buffer any further.
//
// The Phase-3 bucketed graph-reuse fast path (reused ggml_cgraph* vs. from-scratch rebuild must be
// bit-identical) is a separate, later addition to this same file once KvCache/ATTENTION land -- this is
// the baseline half described in the implementation plan.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

int main() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/builder_test.gguf";
    auto model = loom::GgufModel::load(path, backend.get());
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
    loom::GraphBuilder builder(topo, *model, backend.get());

    constexpr uint32_t kNCtxMax = 16;
    builder.reserve(kNCtxMax);
    const size_t reserved_size = builder.buffer_size();
    LOOM_CHECK(reserved_size > 0);

    bool grew = false;
    for (uint32_t n_tokens = 1; n_tokens <= kNCtxMax; ++n_tokens) {
        builder.build(n_tokens, /*n_past=*/0);
        if (builder.buffer_size() > reserved_size) grew = true;
    }
    LOOM_CHECK(!grew);

    // A decode-shaped call (n_tokens=1 at the far end of the reserved range) shouldn't grow it either.
    builder.build(1, kNCtxMax - 1);
    LOOM_CHECK(builder.buffer_size() <= reserved_size);

    LOOM_TEST_REPORT_AND_RETURN();
}
