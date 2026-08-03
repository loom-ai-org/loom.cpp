// Regression test for the axis-collision bug (BACKLOG.md, found 2026-08-03).
//
// `op_add`/`op_mul`/`op_mul_mat` carry "dynamically heal transposed layouts" heuristics that infer an
// operand's intended layout from its SIZES. Those are ambiguous whenever two axes happen to be equal,
// and the transformer case makes that happen for real: RoPE multiplies cos [head_dim, n_tokens, 1] into
// q [head_dim, n_tokens, n_head], and attention multiplies q by k, both [head_dim, n_tokens, n_head].
// When a prompt's length equals the head count, the heuristics fired on already-correct operands and
// silently permuted them -- turning per-token rotation into per-head rotation, and transposing
// attention's own operands.
//
// The real-model symptom was that EVERY MIL-exported causal LM produced wrong logits at exactly
// n_tokens == n_head (and n_head_kv): Qwen3-0.6B was off by 13-23 logits at 8 and 16 tokens while being
// correct to 2e-5 everywhere else. Nothing caught it because the only numeric gate on that path used
// 3- and 7-token prompts.
//
// These checks are deliberately at the PRIMITIVE level rather than through a model: they run with no
// checkpoint, in milliseconds, and they fail for the original reason rather than for a downstream one.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

// A topology computing `out = x * y` (or a matmul) so the primitives run exactly as they do in a real
// graph -- through GraphBuilder, not by calling ggml directly.
std::string mul_topology(int head_dim, const char* n_tokens_expr, int n_head) {
    return std::string(R"({"version":1,"inputs":[)") +
           R"({"name":"x","dtype":"f32","shape":[")" + std::to_string(head_dim) + R"(","n_tokens",")" +
           std::to_string(n_head) + R"("]},)" +
           R"({"name":"y","dtype":"f32","shape":[")" + std::to_string(head_dim) + R"(","n_tokens","1"]}],)" +
           R"("output":"out","nodes":[{"op":"MUL","inputs":["x","y"],"outputs":["out"]}]})";
}

// x[d, t, h] * y[d, t, 1] must broadcast over the HEAD axis: every head sees the same per-token vector.
// That is RoPE's contract, and the collision case is n_tokens == n_head.
bool check_mul_broadcasts_over_heads(int head_dim, int n_tokens, int n_head) {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(std::string(LOOM_TEST_FIXTURE_DIR) + "/minimal.gguf", backend.get());
    loom::GraphTopology topo = loom::GraphTopology::parse(mul_topology(head_dim, "n_tokens", n_head));

    loom::GraphBuilder builder(topo, *model, backend.get());
    loom::GraphBuilder::BuildResult r = builder.build({{"n_tokens", static_cast<double>(n_tokens)}});

    // x[d,t,h] = 1000*h + 10*t + d, y[d,t] = t + 1 -- every index distinguishable, so a permuted
    // operand cannot coincidentally produce the right answer.
    std::vector<float> x(static_cast<size_t>(head_dim) * n_tokens * n_head);
    for (int h = 0; h < n_head; ++h)
        for (int t = 0; t < n_tokens; ++t)
            for (int d = 0; d < head_dim; ++d)
                x[(static_cast<size_t>(h) * n_tokens + t) * head_dim + d] = 1000.0f * h + 10.0f * t + d;
    std::vector<float> y(static_cast<size_t>(head_dim) * n_tokens);
    for (int t = 0; t < n_tokens; ++t)
        for (int d = 0; d < head_dim; ++d)
            y[static_cast<size_t>(t) * head_dim + d] = static_cast<float>(t + 1);

    ggml_backend_tensor_set(r.input_tensors.at("x"), x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("y"), y.data(), 0, y.size() * sizeof(float));
    ggml_backend_graph_compute(backend.get(), r.graph);

    std::vector<float> got(static_cast<size_t>(ggml_nelements(r.outputs.front())));
    ggml_backend_tensor_get(r.outputs.front(), got.data(), 0, got.size() * sizeof(float));

    bool ok = true;
    for (int h = 0; h < n_head && ok; ++h) {
        for (int t = 0; t < n_tokens && ok; ++t) {
            for (int d = 0; d < head_dim; ++d) {
                const size_t i = (static_cast<size_t>(h) * n_tokens + t) * head_dim + d;
                const float want = x[i] * y[static_cast<size_t>(t) * head_dim + d];
                if (std::abs(got[i] - want) > 1e-3f) {
                    std::fprintf(stderr,
                                 "  MUL[d=%d,t=%d,h=%d] (head_dim=%d n_tokens=%d n_head=%d): "
                                 "want %.1f got %.1f\n",
                                 d, t, h, head_dim, n_tokens, n_head, want, got[i]);
                    ok = false;
                    break;
                }
            }
        }
    }
    return ok;
}

} // namespace

int main() {
    // n_tokens deliberately swept ACROSS the head count, so the collision case is covered alongside its
    // neighbours -- a test that only used one length would have passed before the fix just as the real
    // model's own 3-and-7-token gate did.
    constexpr int kHeadDim = 4;
    for (int n_head : {3, 4}) {
        for (int n_tokens = 1; n_tokens <= 6; ++n_tokens) {
            const bool ok = check_mul_broadcasts_over_heads(kHeadDim, n_tokens, n_head);
            if (!ok) {
                std::fprintf(stderr, "FAILED at n_tokens=%d n_head=%d%s\n", n_tokens, n_head,
                             n_tokens == n_head ? "  (the axis-collision case)" : "");
            }
            LOOM_CHECK(ok);
        }
    }
    std::fprintf(stderr, "MUL broadcasts over heads correctly at every n_tokens, including "
                          "n_tokens == n_head\n");
    LOOM_TEST_REPORT_AND_RETURN();
}
