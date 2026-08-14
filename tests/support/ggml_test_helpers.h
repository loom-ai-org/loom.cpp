#pragma once

// Shared scratch-graph helper for tests that exercise a handful of ggml ops in isolation (as opposed to
// tests that go through the full GraphBuilder). Each test case gets its own GgmlScratch so unrelated
// graphs never interact.

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include "cpu_backend.h"

#include <vector>

namespace loom_test {

struct GgmlScratch {
    ggml_backend_ptr backend_owned; // null when constructed with an external backend
    ggml_backend_t backend;         // always valid: backend_owned.get(), or the external one passed in
    ggml_context_ptr ctx;
    ggml_gallocr_ptr galloc;

    explicit GgmlScratch(size_t mem_size = 16 * 1024 * 1024)
        : backend_owned(loom_test::cpu_backend()),
          backend(backend_owned.get()),
          ctx(ggml_init(ggml_init_params{mem_size, nullptr, /*no_alloc=*/true})),
          galloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {}

    // Use an existing backend instead of creating a new one -- needed when the graph being built
    // touches tensors allocated on that specific backend already (e.g. a KvCache's persistent buffers).
    explicit GgmlScratch(ggml_backend_t external_backend, size_t mem_size = 16 * 1024 * 1024)
        : backend_owned(nullptr),
          backend(external_backend),
          ctx(ggml_init(ggml_init_params{mem_size, nullptr, /*no_alloc=*/true})),
          galloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {}

    // Builds forward from `out` and allocates real storage for every tensor in the graph. Input tensors
    // (those created with ggml_set_input()) only get a valid `data` pointer after this call -- set their
    // contents afterwards, via set_f32()/set_i32() below, then run ggml_backend_graph_compute().
    ggml_cgraph* expand(ggml_tensor* out) {
        ggml_cgraph* gf = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_alloc_graph(galloc.get(), gf);
        return gf;
    }

    void compute(ggml_cgraph* gf) { ggml_backend_graph_compute(backend, gf); }
};

inline void set_f32(ggml_tensor* t, const std::vector<float>& data) {
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
}

inline void set_i32(ggml_tensor* t, const std::vector<int32_t>& data) {
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(int32_t));
}

inline std::vector<float> get_f32(ggml_tensor* t) {
    std::vector<float> out(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    return out;
}

} // namespace loom_test
