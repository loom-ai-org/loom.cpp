// LOOM_CPU_REPACK places Q4_0 / Q8_0 matrices in ggml's CPU_REPACK buffer, whose interleaved layout
// only a direct F32 mul_mat can read (GgufModel::route_repacked_uses). This loads one fixture twice --
// switch off, switch on -- and builds the same graph over each, where every weight is read three ways:
//
//   * MUL_MAT        -- the repacked kernel's job. Equal to the plain one up to summation order, so it is
//                       compared to 1e-5 relative, not bit for bit.
//   * GET_ROWS       -- not served by CPU_REPACK: must be pointed at a plain copy, and is then EXACT.
//   * RESHAPE view   -- a view of a pre-allocated weight got its data pointer at creation, into the
//     -> GET_ROWS       repacked bytes; it must be re-pointed into the copy, and is then EXACT too.
//
// Read as interleaved blocks, either of the last two comes back wrong by O(1), so a missed route cannot
// pass. Where this CPU has no repack layout for a type (Q8_0 on x86, both on a core without dotprod),
// nothing is placed and the two arms are the same run -- the test says so and still compares them.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Outputs {
    std::vector<float> mm, rows, view;
};

Outputs run(loom::GgufModel& model, ggml_backend_t backend, const std::string& weight) {
    const std::string json = R"({
      "version": 1,
      "inputs": [{"name": "tokens", "dtype": "i32", "shape": ["n_tokens"]}],
      "outputs": ["mm", "rows", "view"],
      "nodes": [
        {"op": "GET_ROWS", "inputs": ["act.weight", "tokens"], "outputs": ["x"]},
        {"op": "MUL_MAT", "inputs": [")" + weight + R"(", "x"], "outputs": ["mm"]},
        {"op": "GET_ROWS", "inputs": [")" + weight + R"(", "tokens"], "outputs": ["rows"]},
        {"op": "RESHAPE", "inputs": [")" + weight + R"("], "outputs": ["w2"], "attrs": {"shape": [128, 16]}},
        {"op": "GET_ROWS", "inputs": ["w2", "tokens"], "outputs": ["view"]}
      ]
    })";
    loom::GraphTopology topo = loom::GraphTopology::parse(json);
    loom::GraphBuilder builder(topo, model, backend);
    const std::vector<int32_t> tokens = {0, 5, 3, 7, 1};
    const auto& r = builder.build({{"n_tokens", tokens.size()}, {"n_past", 0}});
    ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens.data(), 0, tokens.size() * sizeof(int32_t));
    LOOM_CHECK(ggml_backend_graph_compute(backend, r.graph) == GGML_STATUS_SUCCESS);
    auto get = [](ggml_tensor* t) {
        std::vector<float> v(static_cast<size_t>(ggml_nelements(t)));
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        return v;
    };
    return {get(r.outputs[0]), get(r.outputs[1]), get(r.outputs[2])};
}

std::unique_ptr<loom::GgufModel> load(const std::string& path, ggml_backend_t backend, const char* repack) {
    if (repack) setenv("LOOM_CPU_REPACK", repack, 1); else unsetenv("LOOM_CPU_REPACK");
    auto model = loom::GgufModel::load(path, backend);
    unsetenv("LOOM_CPU_REPACK");
    return model;
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/repack_test.gguf";

    auto plain = load(path, backend.get(), nullptr);
    auto packed = load(path, backend.get(), "1");
    LOOM_CHECK(plain->repacked_count() == 0);

    for (const char* weight : {"w_q4_0.weight", "w_q8_0.weight"}) {
        const Outputs a = run(*plain, backend.get(), weight);
        const Outputs b = run(*packed, backend.get(), weight);

        LOOM_CHECK(a.rows == b.rows);
        LOOM_CHECK(a.view == b.view);
        LOOM_CHECK(a.mm.size() == b.mm.size());
        double scale = 0, worst = 0;
        for (size_t i = 0; i < a.mm.size(); ++i) {
            scale = std::max(scale, std::fabs(static_cast<double>(a.mm[i])));
            worst = std::max(worst, std::fabs(static_cast<double>(a.mm[i]) - b.mm[i]));
        }
        LOOM_CHECK(worst <= 1e-5 * scale);
        std::printf("%s: mul_mat max rel %.2e, GET_ROWS %s, view %s\n", weight, worst / scale,
                    a.rows == b.rows ? "exact" : "DIFFERS", a.view == b.view ? "exact" : "DIFFERS");
    }

    // Each weight placed is read by GET_ROWS and by a view, so each needed exactly one plain copy.
    LOOM_CHECK(packed->plain_copy_count() == packed->repacked_count());
    if (packed->repacked_count() == 0) {
        std::printf("note: this CPU has no CPU_REPACK layout for Q4_0 or Q8_0, so both arms ran plain\n");
    } else {
        std::printf("%zu weight(s) repacked on this CPU\n", packed->repacked_count());
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
