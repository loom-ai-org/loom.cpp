// A graph's layout is the graph's own (ADR-117): the same shape built before and after a larger one of the
// same topology puts every node at the same offset in the compute buffer.
//
// `ggml_gallocr_alloc_graph` re-plans only when a graph does not fit the last plan, so before the fix a
// graph built after a larger one took the larger one's offsets. Layout is not only placement: the CPU
// backend's conv fusion (ggml-0036) falls back to the unfused nodes when its result overlaps its input,
// and the two round differently, so Chatterbox's HiFT gave one mel 1.1e-07 apart before and after a
// longer one, and a stream interleaved with another stopped being bit-identical to itself.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <ggml-backend.h>

#include <cstdint>
#include <vector>

namespace {

// Each computed node's offset from its buffer's base; -1 for one outside the compute buffer (an input,
// a weight, a cache view).
std::vector<int64_t> node_offsets(const loom::GraphBuilder::BuildResult& r, ggml_backend_buffer_t compute) {
    std::vector<int64_t> out;
    for (int i = 0; i < ggml_graph_n_nodes(r.graph); ++i) {
        const ggml_tensor* t = ggml_graph_node(r.graph, i);
        if (t->buffer != compute || t->view_src != nullptr) {
            out.push_back(-1);
            continue;
        }
        out.push_back(static_cast<const char*>(t->data) -
                      static_cast<const char*>(ggml_backend_buffer_get_base(t->buffer)));
    }
    return out;
}

ggml_backend_buffer_t compute_buffer(const loom::GraphBuilder::BuildResult& r) {
    for (int i = 0; i < ggml_graph_n_nodes(r.graph); ++i) {
        const ggml_tensor* t = ggml_graph_node(r.graph, i);
        if (t->buffer != nullptr && t->view_src == nullptr && !(t->flags & GGML_TENSOR_FLAG_INPUT)) {
            return t->buffer;
        }
    }
    return nullptr;
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/builder_test.gguf";
    auto model = loom::GgufModel::load(path, backend.get());
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
    loom::GraphBuilder builder(topo, *model, backend.get());

    const auto& small = builder.build({{"n_tokens", 3}, {"n_past", 0}});
    ggml_backend_buffer_t buf = compute_buffer(small);
    LOOM_CHECK(buf != nullptr);
    const std::vector<int64_t> first = node_offsets(small, buf);

    builder.build({{"n_tokens", 16}, {"n_past", 0}});

    const auto& again = builder.build({{"n_tokens", 3}, {"n_past", 0}});
    const std::vector<int64_t> second = node_offsets(again, compute_buffer(again));
    size_t moved = 0;
    for (size_t i = 0; i < std::min(first.size(), second.size()); ++i) moved += first[i] != second[i];
    std::fprintf(stderr, "%zu nodes, %zu at another offset after a larger graph\n", first.size(), moved);
    LOOM_CHECK(first.size() == second.size());
    LOOM_CHECK(moved == 0);

    LOOM_TEST_REPORT_AND_RETURN();
}
