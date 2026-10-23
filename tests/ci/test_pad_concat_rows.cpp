// PAD and CONCAT a row at a time (P4.41, ggml-0050) are EXACT: the same values moved by memcpy/memset.
//
// ggml's loops form 64-bit addresses and range tests per element: ~180 ns an element for PAD and ~100 for
// CONCAT on a Pi Zero W, where Pocket-TTS's Mimi decoder pads 2.8M-element activations before its causal
// convolutions and concatenates its RoPE pair swaps. ggml-0050 copies whole rows whenever the source
// rows are contiguous floats, on every ISA. This pins, against the per-element definition written out
// here and compared bit for bit (NaNs, -0 and denormals included):
//
//   1. PAD with left and right padding on each of the four axes, alone and together, from a
//      contiguous source, from a strided view (rows contiguous: the fast path) and from a transposed
//      view (rows not contiguous: ggml's own loop).
//   2. CONCAT along each of the four axes, from contiguous sources, from strided views with rows of
//      one float (the RoPE pair swap's shape) and from a transposed view; and of I32 tensors.

#include "cpu_backend.h"
#include "test_util.h"

#include <ggml.h>
#include <ggml-backend.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

bool compute(ggml_cgraph* gf) {
    static ggml_backend_t backend = loom_test::cpu_backend();
    if (backend == nullptr) return false;
    return ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
}

// element (i0..i3) of any tensor whose element is 4 bytes, through its strides
uint32_t at(const ggml_tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    uint32_t v;
    std::memcpy(&v, (const char*) t->data + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3], 4);
    return v;
}

void fill(ggml_tensor* t, uint32_t seed) {
    uint32_t* d = (uint32_t*) t->data;
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; ++i) {
        float f = 0.01f * (float) ((int64_t) ((i + seed) * 7919 % 211) - 105);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        if (i % 17 == 3) u = 0x7fc00001u + (uint32_t) i;   // NaN payloads must survive a copy
        if (i % 19 == 5) u = 0x80000000u;                    // -0
        if (i % 23 == 7) u = 0x00000003u;                    // a denormal
        d[i] = u;
    }
}

// the source of a case: a fresh contiguous tensor, or a view of a larger one
enum class Src { contiguous, strided, transposed };

ggml_tensor* make_src(ggml_context* ctx, Src kind, int64_t n0, int64_t n1, int64_t n2, int64_t n3, uint32_t seed,
                      ggml_type type = GGML_TYPE_F32) {
    if (kind == Src::contiguous) {
        ggml_tensor* t = ggml_new_tensor_4d(ctx, type, n0, n1, n2, n3);
        fill(t, seed);
        return t;
    }
    if (kind == Src::strided) {
        // every other float of a row twice as long, from offset 1: rows of n0 contiguous floats when the
        // view takes them whole, rows of one float when n0 == 1 (Pocket's pair swap)
        ggml_tensor* big = ggml_new_tensor_4d(ctx, type, 2 * n0 + 1, n1, n2, n3);
        fill(big, seed);
        return ggml_view_4d(ctx, big, n0, n1, n2, n3, big->nb[1], big->nb[2], big->nb[3], 4);
    }
    ggml_tensor* t = ggml_new_tensor_4d(ctx, type, n1, n0, n2, n3);
    fill(t, seed);
    return ggml_transpose(ctx, t);   // ne = [n0, n1, n2, n3], nb[0] != 4
}

int check_pad(Src kind, int64_t n0, int64_t n1, int64_t n2, int64_t n3, const int lp[4], const int rp[4]) {
    ggml_init_params ip = { (size_t) 64 << 20, nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* a = make_src(ctx, kind, n0, n1, n2, n3, 5);
    ggml_tensor* y = ggml_pad_ext(ctx, a, lp[0], rp[0], lp[1], rp[1], lp[2], rp[2], lp[3], rp[3]);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    LOOM_CHECK(compute(gf));
    int bad = 0;
    for (int64_t i3 = 0; i3 < y->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < y->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < y->ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < y->ne[0]; ++i0) {
                    const int64_t s0 = i0 - lp[0], s1 = i1 - lp[1], s2 = i2 - lp[2], s3 = i3 - lp[3];
                    const bool in = s0 >= 0 && s0 < n0 && s1 >= 0 && s1 < n1 && s2 >= 0 && s2 < n2 && s3 >= 0 && s3 < n3;
                    const uint32_t want = in ? at(a, s0, s1, s2, s3) : 0u;
                    bad += at(y, i0, i1, i2, i3) != want;
                }
    ggml_free(ctx);
    return bad;
}

int check_concat(Src kind, int dim, int64_t n0, int64_t n1, int64_t n2, int64_t n3, ggml_type type = GGML_TYPE_F32) {
    ggml_init_params ip = { (size_t) 64 << 20, nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    int64_t m[4] = { n0, n1, n2, n3 };
    m[dim] += 3;   // src1 is longer along the joined axis
    ggml_tensor* a = make_src(ctx, kind, n0, n1, n2, n3, 7, type);
    ggml_tensor* b = make_src(ctx, kind, m[0], m[1], m[2], m[3], 11, type);
    ggml_tensor* y = ggml_concat(ctx, a, b, dim);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    LOOM_CHECK(compute(gf));
    const int64_t na[4] = { n0, n1, n2, n3 };
    int bad = 0;
    for (int64_t i3 = 0; i3 < y->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < y->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < y->ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < y->ne[0]; ++i0) {
                    int64_t i[4] = { i0, i1, i2, i3 };
                    uint32_t want;
                    if (i[dim] < na[dim]) {
                        want = at(a, i0, i1, i2, i3);
                    } else {
                        i[dim] -= na[dim];
                        want = at(b, i[0], i[1], i[2], i[3]);
                    }
                    bad += at(y, i0, i1, i2, i3) != want;
                }
    ggml_free(ctx);
    return bad;
}

} // namespace

int main() {
    const char* names[] = { "contiguous", "strided", "transposed" };
    for (Src kind : { Src::contiguous, Src::strided, Src::transposed }) {
        const int none[4] = { 0, 0, 0, 0 };
        const int causal[4] = { 2, 0, 0, 0 };          // Mimi's SEANet: k - 1 zeros on the left of time
        const int left0[4] = { 3, 0, 0, 0 };
        const int right0[4] = { 5, 0, 0, 0 };
        const int all[4] = { 1, 2, 2, 1 };
        const int all_r[4] = { 2, 1, 1, 2 };
        const int lp_rows[4] = { 0, 2, 1, 1 };         // whole zero rows, planes and volumes
        const int rp_rows[4] = { 0, 1, 2, 0 };
        int bad = 0;
        bad += check_pad(kind, 37, 5, 3, 2, causal, none);
        bad += check_pad(kind, 37, 5, 3, 2, left0, right0);
        bad += check_pad(kind, 9, 4, 3, 2, all, all_r);
        bad += check_pad(kind, 9, 4, 3, 2, lp_rows, rp_rows);
        bad += check_pad(kind, 1, 7, 2, 1, causal, none);
        std::printf("  pad    %-10s %s\n", names[(int) kind], bad ? "MISMATCH" : "exact");
        LOOM_CHECK(bad == 0);

        bad = 0;
        for (int dim = 0; dim < 4; ++dim) {
            bad += check_concat(kind, dim, 6, 5, 3, 2);
            bad += check_concat(kind, dim, 1, 64, 4, 1);   // rows of one float: the RoPE pair swap
        }
        bad += check_concat(kind, 0, 6, 5, 3, 2, GGML_TYPE_I32);
        bad += check_concat(kind, 1, 32, 7, 4, 1, GGML_TYPE_I32);
        std::printf("  concat %-10s %s\n", names[(int) kind], bad ? "MISMATCH" : "exact");
        LOOM_CHECK(bad == 0);
    }
    LOOM_TEST_REPORT_AND_RETURN();
}
