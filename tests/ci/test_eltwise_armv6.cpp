// The ARMv6 elementwise passes (ADR-079 P4.32f f2, ggml-0037) are BIT-IDENTICAL to ggml's own loops.
//
// On the VFP11 ggml's ADD, SUB, MUL, SCALE, RELU and LEAKY_RELU paid 43-69 ns an element on a Pi Zero W:
// one VFP op per element waiting on the one before it through its store, no prefetch, and for the RELUs
// a `vcmpe` + `vmrs` per element. ggml-0037 runs them eight floats a step with `pld`, and does the RELUs'
// compares on the integer bits. Nothing about the arithmetic changes, so the claim is exact equality,
// and this pins it. Registered twice, GGML_CPU_ARMV6_ELTWISE=0 and =1 (ggml reads it once):
//
//   1. **The numbers**: every case against the same float operations done here, EXACTLY, on inputs that
//      stay in the normal range (the test's own thread is not in RunFast; ggml's may be).
//   2. **Bit-identity, edges included**: every case again with ±0, ±denormals, ±inf and NaN spliced into
//      its input, and the two registrations' outputs compared byte for byte (whichever runs second,
//      DEPENDS). That is the comparison that decides the integer compares match the VFP's in whatever
//      FPSCR mode ggml computes in.
//
// Shapes cover the kernel's eight-float steps and their tails, a src1 row of one float (a [1, C] bias,
// a [1, 1] scale -- the scalar form), a src1 row broadcast over rows (a time mask), a src1 row repeated
// within a row, and the in-place forms; and (ggml-0051, P4.41) 3-D and 4-D broadcasts that wrap the row
// counters, and strided one-float rows times a scalar. On every other ISA both registrations run stock
// ggml and pass.

#include "cpu_backend.h"
#include "test_util.h"

#include <ggml.h>
#include <ggml-backend.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

namespace {

bool compute(ggml_cgraph* gf) {
    static ggml_backend_t backend = loom_test::cpu_backend();
    if (backend == nullptr) return false;
    return ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
}

bool eltwise_on() {
    const char* e = std::getenv("GGML_CPU_ARMV6_ELTWISE");
#if defined(__ARM_FEATURE_SIMD32) && !defined(__ARM_NEON) && !defined(__aarch64__)
    return !(e != nullptr && std::atoi(e) == 0);
#else
    (void) e;
    return false;
#endif
}

float bits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// every value the integer compares have to classify exactly as the VFP does
const uint32_t kEdges[] = {
    0x00000000u, 0x80000000u,                            // +-0
    0x00000001u, 0x80000001u, 0x007fffffu, 0x807fffffu,  // +-denormals, smallest and largest
    0x00400000u, 0x80400000u,
    0x00800000u, 0x80800000u,                            // +-smallest normal
    0x7f7fffffu, 0xff7fffffu,                            // +-largest finite
    0x7f800000u, 0xff800000u,                            // +-inf
    0x7fc00000u, 0xffc00000u, 0x7f800001u, 0xff800001u,  // NaNs, quiet and signalling
    0x3f800000u, 0xbf800000u,                            // +-1
};
constexpr int kNEdges = sizeof kEdges / sizeof kEdges[0];

void fill(ggml_tensor* t, int seed, bool edges) {
    float* d = (float*) t->data;
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; ++i) {
        d[i] = 0.01f * (float) (((i + seed) * 7919) % 113 - 56) + 0.001f * (float) seed;
    }
    if (edges) {
        for (int64_t i = 0; i < n; i += 5) d[i] = bits(kEdges[(i / 5 + seed) % kNEdges]);
    }
}

struct Case {
    const char* name;
    // builds the node from (a, b); b may be unused
    std::function<ggml_tensor*(ggml_context*, ggml_tensor*, ggml_tensor*)> node;
    int64_t a0, a1, b0, b1;
    // the reference for element (i0, i1), given a's and b's values there
    std::function<float(float, float)> ref;
};

const float kSlope = 0.01f;   // piper / VITS
const float kScale = 1.0f / 3.0f;

float leaky_ref(float x, float) { return ((x > 0.f) ? x : 0.f) + kSlope * ((x < 0.0f) ? x : 0.f); }

// runs one case; appends dst to `all`; returns the number of reference mismatches (normal inputs only)
int run_case(const Case& cs, bool edges, std::vector<float>& all) {
    ggml_init_params ip = { (size_t) 16 << 20, nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cs.a0, cs.a1);
    ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cs.b0, cs.b1);
    fill(a, 3, edges);
    fill(b, 11, edges);
    // a's input, kept: the in-place cases overwrite it
    std::vector<float> av((float*) a->data, (float*) a->data + ggml_nelements(a));
    ggml_tensor* out = cs.node(ctx, a, b);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    LOOM_CHECK(compute(gf));

    int bad = 0;
    const float* o = (const float*) out->data;
    const float* bv = (const float*) b->data;
    for (int64_t i1 = 0; i1 < cs.a1; ++i1) {
        for (int64_t i0 = 0; i0 < cs.a0; ++i0) {
            const float x = av[i1 * cs.a0 + i0];
            const float y = bv[(i1 % cs.b1) * cs.b0 + i0 % cs.b0];
            const float want = cs.ref(x, y);
            if (!edges && std::memcmp(&want, &o[i1 * cs.a0 + i0], 4) != 0) ++bad;
        }
    }
    all.insert(all.end(), o, o + ggml_nelements(out));
    ggml_free(ctx);
    return bad;
}

// ggml-0051 (P4.41): the ARMv6 rows are walked with counters, so the cases that matter are the ones where
// the counters wrap -- 3-D and 4-D sources, src1 broadcast along rows, planes and volumes -- and the
// one-float rows done inline: a strided [1, 64, n] view times a [1, 1] scalar is Pocket-TTS's RoPE
// negation. Returns the reference mismatches (normal inputs only); appends dst to `all`.
int run_case4(const char* name, char op, bool strided, const int64_t a[4], const int64_t b[4], bool edges,
              std::vector<float>& all) {
    ggml_init_params ip = { (size_t) 16 << 20, nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* src;
    if (strided) {
        // element 1 of every pair of a [2 a0, a1, a2, a3] tensor, as Pocket's `pairs[..., 1:2]`
        ggml_tensor* big = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2 * a[0], a[1], a[2], a[3]);
        fill(big, 3, edges);
        src = ggml_view_4d(ctx, big, a[0], a[1], a[2], a[3], big->nb[1], big->nb[2], big->nb[3], 4);
    } else {
        src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a[0], a[1], a[2], a[3]);
        fill(src, 3, edges);
    }
    ggml_tensor* bt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b[0], b[1], b[2], b[3]);
    fill(bt, 11, edges);
    ggml_tensor* out = op == '+' ? ggml_add(ctx, src, bt) : op == '-' ? ggml_sub(ctx, src, bt) : ggml_mul(ctx, src, bt);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    LOOM_CHECK(compute(gf));
    int bad = 0;
    for (int64_t i3 = 0; i3 < a[3]; ++i3)
        for (int64_t i2 = 0; i2 < a[2]; ++i2)
            for (int64_t i1 = 0; i1 < a[1]; ++i1)
                for (int64_t i0 = 0; i0 < a[0]; ++i0) {
                    const float x = *(const float*) ((const char*) src->data + i0 * src->nb[0] + i1 * src->nb[1] + i2 * src->nb[2] + i3 * src->nb[3]);
                    const float y = *(const float*) ((const char*) bt->data + (i0 % b[0]) * bt->nb[0] + (i1 % b[1]) * bt->nb[1] +
                                                     (i2 % b[2]) * bt->nb[2] + (i3 % b[3]) * bt->nb[3]);
                    const float want = op == '+' ? x + y : op == '-' ? x - y : x * y;
                    const float got = ((const float*) out->data)[((i3 * a[2] + i2) * a[1] + i1) * a[0] + i0];
                    if (!edges && std::memcmp(&want, &got, 4) != 0) ++bad;
                }
    all.insert(all.end(), (const float*) out->data, (const float*) out->data + ggml_nelements(out));
    ggml_free(ctx);
    if (!edges) std::printf("  %-32s %s\n", name, bad ? "MISMATCH" : "exact");
    return bad;
}

} // namespace

int main() {
    const bool on = eltwise_on();
    std::printf("GGML_CPU_ARMV6_ELTWISE path %s\n", on ? "ON" : "off (stock ggml)");

    auto add = [](ggml_context* c, ggml_tensor* a, ggml_tensor* b) { return ggml_add(c, a, b); };
    auto sub = [](ggml_context* c, ggml_tensor* a, ggml_tensor* b) { return ggml_sub(c, a, b); };
    auto mul = [](ggml_context* c, ggml_tensor* a, ggml_tensor* b) { return ggml_mul(c, a, b); };
    auto add_ip = [](ggml_context* c, ggml_tensor* a, ggml_tensor* b) { return ggml_add_inplace(c, a, b); };
    auto mul_ip = [](ggml_context* c, ggml_tensor* a, ggml_tensor* b) { return ggml_mul_inplace(c, a, b); };
    auto f_add = [](float x, float y) { return x + y; };
    auto f_sub = [](float x, float y) { return x - y; };
    auto f_mul = [](float x, float y) { return x * y; };

    const std::vector<Case> cases = {
        // same shape: the eight-float steps, their tails, and a row the length of amy's largest stage
        { "add, same shape, n=1",            add, 1, 3, 1, 3, f_add },
        { "add, same shape, n=7",            add, 7, 3, 7, 3, f_add },
        { "add, same shape, n=8",            add, 8, 3, 8, 3, f_add },
        { "add, same shape, n=9",            add, 9, 3, 9, 3, f_add },
        { "add, same shape, n=1003",         add, 1003, 5, 1003, 5, f_add },
        { "sub, same shape, n=203",          sub, 203, 4, 203, 4, f_sub },
        { "mul, same shape, n=203",          mul, 203, 4, 203, 4, f_mul },
        { "add, n=130048 x 2",               add, 130048, 2, 130048, 2, f_add },
        // a src1 row of one float: a [1, C] bias and a [1, 1] scale
        { "add, [1, C] bias",                add, 557, 6, 1, 6, f_add },
        { "sub, [1, C]",                     sub, 13, 6, 1, 6, f_sub },
        { "mul, [1, 1] scale",               mul, 557, 6, 1, 1, f_mul },
        // a src1 row broadcast over rows (a time mask), and repeated within a row
        { "mul, [T, 1] mask",                mul, 551, 7, 551, 1, f_mul },
        { "add, src1 row repeated x3",       add, 3 * 41, 4, 41, 4, f_add },
        // in place
        { "add in place",                    add_ip, 301, 3, 301, 3, f_add },
        { "mul in place, [1, C]",            mul_ip, 301, 3, 1, 3, f_mul },
        // unary
        { "leaky_relu",                      [](ggml_context* c, ggml_tensor* a, ggml_tensor*) { return ggml_leaky_relu(c, a, kSlope, false); },
                                             1003, 4, 1, 1, leaky_ref },
        { "leaky_relu in place",             [](ggml_context* c, ggml_tensor* a, ggml_tensor*) { return ggml_leaky_relu(c, a, kSlope, true); },
                                             97, 3, 1, 1, leaky_ref },
        { "relu",                            [](ggml_context* c, ggml_tensor* a, ggml_tensor*) { return ggml_relu(c, a); },
                                             1003, 4, 1, 1, [](float x, float) { return (x > 0.f) ? x : 0.f; } },
        { "scale",                           [](ggml_context* c, ggml_tensor* a, ggml_tensor*) { return ggml_scale(c, a, kScale); },
                                             1003, 4, 1, 1, [](float x, float) { return x * kScale; } },
        { "scale in place",                  [](ggml_context* c, ggml_tensor* a, ggml_tensor*) { return ggml_scale_inplace(c, a, kScale); },
                                             59, 3, 1, 1, [](float x, float) { return x * kScale; } },
    };

    std::vector<float> all;
    for (const Case& cs : cases) {
        const int bad = run_case(cs, false, all);
        std::printf("  %-32s %s\n", cs.name, bad ? "MISMATCH" : "exact");
        LOOM_CHECK(bad == 0);
    }
    for (const Case& cs : cases) run_case(cs, true, all);   // edges: judged byte for byte below

    struct Case4 { const char* name; char op; bool strided; int64_t a[4], b[4]; };
    const Case4 cases4[] = {
        { "mul, strided [1,64,37] x scalar",  '*', true,  { 1, 64, 37, 1 }, { 1, 1, 1, 1 } },
        { "mul, strided [1,5,3,2] x [1,5]",   '*', true,  { 1, 5, 3, 2 },   { 1, 5, 1, 1 } },
        { "add, [9,4,3,2] + [9,2,3,1]",       '+', false, { 9, 4, 3, 2 },   { 9, 2, 3, 1 } },
        { "sub, [1,6,4,3] - [1,3,2,3]",       '-', false, { 1, 6, 4, 3 },   { 1, 3, 2, 3 } },
        { "mul, [1,7,5,2] x [1,1,5,1]",       '*', false, { 1, 7, 5, 2 },   { 1, 1, 5, 1 } },
        { "add, [16,3,4,5] + [1,1,1,5]",      '+', false, { 16, 3, 4, 5 },  { 1, 1, 1, 5 } },
    };
    for (const Case4& c : cases4) LOOM_CHECK(run_case4(c.name, c.op, c.strided, c.a, c.b, false, all) == 0);
    for (const Case4& c : cases4) run_case4(c.name, c.op, c.strided, c.a, c.b, true, all);

    if (const char* dir = std::getenv("LOOM_TEST_TMPDIR")) {
        char mine[1024], other[1024];
        std::snprintf(mine, sizeof mine, "%s/eltwise_armv6_%s.f32", dir, on ? "on" : "off");
        std::snprintf(other, sizeof other, "%s/eltwise_armv6_%s.f32", dir, on ? "off" : "on");
        if (FILE* f = std::fopen(mine, "wb")) {
            std::fwrite(all.data(), sizeof(float), all.size(), f);
            std::fclose(f);
        }
        if (FILE* f = std::fopen(other, "rb")) {
            std::vector<float> them(all.size());
            const size_t n = std::fread(them.data(), sizeof(float), them.size(), f);
            std::fclose(f);
            LOOM_CHECK(n == them.size());
            size_t diff = 0;
            for (size_t i = 0; i < n; ++i) diff += std::memcmp(&all[i], &them[i], 4) != 0;
            std::printf("  vs the other registration: %zu of %zu floats differ\n", diff, n);
            LOOM_CHECK(diff == 0);
        }
    }
    LOOM_TEST_REPORT_AND_RETURN();
}
