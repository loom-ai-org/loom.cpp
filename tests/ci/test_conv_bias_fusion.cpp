// The convolution's bias ADD is folded into the convolution (BACKLOG.md P4.15).
//
// `cmake/patches/ggml-0005-conv2d-bias-fusion.patch` teaches ggml's CPU backend to recognise
// CONV_2D -> (reshape) -> ADD(per-channel bias) and compute it as one node, adding the bias to each
// batch of the result while that batch is still in cache instead of making a second pass over the
// whole output. On a VITS synthesis that ADD is 12% of the run.
//
// Fusion is the kind of change that is invisible until it is wrong, so this pins both halves:
//
//   1. **It produces the same numbers.** Compared against a double-precision reference AND against
//      the same graph computed with `GGML_CPU_DISABLE_FUSION=1`, which must agree BIT for bit -- the
//      fused kernel adds the same bias to the same sums, so anything else is a real difference.
//   2. **It actually fuses**, which no output comparison can show. The convolution's own result
//      tensor is poisoned before the run: when the pattern is fused that tensor is never written, so
//      the poison survives, and when it is not, it holds the convolution's output. Registered twice
//      with different environments (ggml reads `GGML_CPU_DISABLE_FUSION` once, on first use), so each
//      run asserts the direction its environment selected -- a fusion that silently stopped matching
//      the pattern would otherwise just look like a performance regression nobody attributed.
//
// Every tensor here is allocated in the ggml context rather than by a graph allocator, because a
// graph allocator is free to give the ADD the same memory as the convolution it consumes -- which
// would make the poison check meaningless.

#include "test_util.h"

#include <ggml.h>
#include <ggml-cpu.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr float kPoison = -123456.0f;

// A conv shape with the aspect ratio the vocoder convolutions have -- long activation, few channels,
// and an output length that is NOT a multiple of the tile height, so the row tail runs too.
constexpr int64_t KW = 5, IC = 8, OC = 6, IL = 251;

double reference(const std::vector<float>& K, const std::vector<float>& X,
                 const std::vector<float>& B, int64_t ol, int64_t oc, int64_t p0) {
    double acc = (double) B[oc];
    for (int64_t ic = 0; ic < IC; ++ic) {
        for (int64_t kx = 0; kx < KW; ++kx) {
            const int64_t sx = ol + kx - p0;
            if (sx < 0 || sx >= IL) continue;
            acc += (double) K[oc * (IC * KW) + ic * KW + kx] * (double) X[ic * IL + sx];
        }
    }
    return acc;
}

} // namespace

int main() {
    const bool fusion_disabled = [] {
        const char* e = std::getenv("GGML_CPU_DISABLE_FUSION");
        return e != nullptr && std::atoi(e) == 1;
    }();

    const int64_t p0 = (KW - 1) / 2;
    const int64_t OL = IL;   // 'same' padding, stride 1

    std::vector<float> K((size_t) KW * IC * OC), X((size_t) IL * IC), B((size_t) OC);
    for (size_t i = 0; i < K.size(); ++i) K[i] = 0.02f - 0.001f * (float) (i % 53);
    for (size_t i = 0; i < X.size(); ++i) X[i] = 0.01f + 0.001f * (float) (i % 97);
    for (size_t i = 0; i < B.size(); ++i) B[i] = 0.5f + 0.25f * (float) i;

    // no_alloc = false: every tensor gets its own storage in this context, so the convolution's
    // result cannot share memory with the add's.
    ggml_init_params ip = { (size_t) 64 * 1024 * 1024, nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    LOOM_CHECK(ctx != nullptr);

    ggml_tensor* tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, KW, 1, IC, OC);
    ggml_tensor* tx = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, IL, 1, IC, 1);
    ggml_tensor* tb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, OC, 1);
    std::memcpy(tk->data, K.data(), K.size() * sizeof(float));
    std::memcpy(tx->data, X.data(), X.size() * sizeof(float));
    std::memcpy(tb->data, B.data(), B.size() * sizeof(float));

    // Exactly what src/ops/primitives_conv.cpp emits for a CONV_1D on aarch64: the convolution, a
    // reshape of its result to [OL, OC, N], and the bias add. The reshape is why the fusion has to
    // look past view nodes rather than only at the next node in the graph.
    ggml_tensor* conv = ggml_conv_2d_direct(ctx, tk, tx, 1, 1, (int) p0, 0, 1, 1);
    ggml_tensor* out  = ggml_add(ctx, ggml_reshape_3d(ctx, conv, conv->ne[0], OC, 1), tb);

    for (int64_t i = 0; i < ggml_nelements(conv); ++i) ((float*) conv->data)[i] = kPoison;
    std::memset(out->data, 0, ggml_nbytes(out));

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    LOOM_CHECK(ggml_graph_compute_with_ctx(ctx, gf, 2) == GGML_STATUS_SUCCESS);

    // 1. the numbers
    double worst = 0.0;
    for (int64_t oc = 0; oc < OC; ++oc) {
        for (int64_t ol = 0; ol < OL; ++ol) {
            const double ref = reference(K, X, B, ol, oc, p0);
            const double got = (double) ((const float*) out->data)[oc * OL + ol];
            const double rel = std::fabs(got - ref) / (std::fabs(ref) + 1e-6);
            if (rel > worst) worst = rel;
        }
    }
    LOOM_CHECK(worst < 1e-5);
    if (worst >= 1e-5) std::fprintf(stderr, "  worst relative error %.3e\n", worst);

    // 2. whether the intermediate was written, which is what says the pattern fused
    bool intermediate_written = false;
    for (int64_t i = 0; i < ggml_nelements(conv); ++i) {
        if (((const float*) conv->data)[i] != kPoison) intermediate_written = true;
    }
    if (fusion_disabled) {
        LOOM_CHECK(intermediate_written);
        if (!intermediate_written) {
            std::fprintf(stderr, "with GGML_CPU_DISABLE_FUSION=1 the convolution still did not write "
                                 "its own result -- this test can no longer tell the two apart\n");
        }
    } else {
        LOOM_CHECK(!intermediate_written);
        if (intermediate_written) {
            std::fprintf(stderr, "CONV_2D + bias ADD was NOT fused: ggml wrote the intermediate. The "
                                 "pattern in ggml-0005 no longer matches what the engine emits -- see "
                                 "ggml_cpu_conv_2d_bias_add_idx and BACKLOG.md P4.15\n");
        }
    }

    // 3. the two paths agree bit for bit -- run the same graph again in this process with the OTHER
    //    setting is not possible (ggml caches the env), so the comparison is against a file the
    //    fusion-disabled registration leaves behind. Whichever runs second does the comparing.
    const char* dir = std::getenv("LOOM_TEST_TMPDIR");
    if (dir != nullptr) {
        char path[1024];
        std::snprintf(path, sizeof path, "%s/conv_bias_fusion_%s.f32", dir,
                      fusion_disabled ? "unfused" : "fused");
        if (FILE* f = std::fopen(path, "wb")) {
            std::fwrite(out->data, 1, ggml_nbytes(out), f);
            std::fclose(f);
        }
        char other[1024];
        std::snprintf(other, sizeof other, "%s/conv_bias_fusion_%s.f32", dir,
                      fusion_disabled ? "fused" : "unfused");
        if (FILE* f = std::fopen(other, "rb")) {
            std::vector<float> them(ggml_nelements(out));
            const size_t n = std::fread(them.data(), sizeof(float), them.size(), f);
            std::fclose(f);
            LOOM_CHECK(n == them.size());
            LOOM_CHECK(std::memcmp(them.data(), out->data, ggml_nbytes(out)) == 0);
        }
    }

    // 4. AND IT MUST STAY CORRECT when the add's destination IS the convolution's own input.
    //
    // A graph allocator hands the ADD a block the convolution's input has just been freed from -- in
    // the unfused order nothing reads that input by the time the ADD runs -- and for a VITS vocoder it
    // does that to EVERY large convolution, so this is the case the fusion mostly has to serve rather
    // than a corner of it. Writing the result there while still reading the input for later batches
    // corrupts it in a way that still looks like a convolution (max_abs_diff 0.54 on a Matcha vocoder,
    // which is how the first version of this was caught -- by the gates, not by this test), so the
    // kernel stages each batch until the next has read what it would overwrite.
    //
    // Reproduced here by pointing the add's destination at the input tensor's own memory (the two are
    // the same size by construction below). The reference is computed from the copies in K/X/B, so it
    // stays well-defined even though the input is being overwritten.
    {
        // Long enough that the convolution needs SEVERAL batches: with one batch the whole input is
        // read into the patch buffer before a single output element is written, so an aliased
        // destination would do no harm and the numbers below would pass either way.
        constexpr int64_t IL2 = 20000;
        ggml_tensor* k2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, KW, 1, IC, IC);
        ggml_tensor* x2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, IL2, 1, IC, 1);
        ggml_tensor* b2 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, IC, 1);
        std::vector<float> K2((size_t) KW * IC * IC), X2((size_t) IL2 * IC), B2((size_t) IC);
        for (size_t i = 0; i < K2.size(); ++i) K2[i] = 0.02f - 0.001f * (float) (i % 53);
        for (size_t i = 0; i < X2.size(); ++i) X2[i] = 0.01f + 0.001f * (float) (i % 97);
        for (size_t i = 0; i < B2.size(); ++i) B2[i] = 0.5f + 0.25f * (float) i;
        std::memcpy(k2->data, K2.data(), K2.size() * sizeof(float));
        std::memcpy(x2->data, X2.data(), X2.size() * sizeof(float));
        std::memcpy(b2->data, B2.data(), B2.size() * sizeof(float));

        ggml_tensor* conv2 = ggml_conv_2d_direct(ctx, k2, x2, 1, 1, (int) p0, 0, 1, 1);
        ggml_tensor* out2  = ggml_add(ctx, ggml_reshape_3d(ctx, conv2, conv2->ne[0], IC, 1), b2);
        LOOM_CHECK(ggml_nbytes(out2) == ggml_nbytes(x2));
        out2->data = x2->data;                       // the aliasing a graph allocator can produce

        for (int64_t i = 0; i < ggml_nelements(conv2); ++i) ((float*) conv2->data)[i] = kPoison;

        ggml_cgraph* gf2 = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf2, out2);
        LOOM_CHECK(ggml_graph_compute_with_ctx(ctx, gf2, 2) == GGML_STATUS_SUCCESS);

        double worst2 = 0.0;
        for (int64_t oc = 0; oc < IC; ++oc) {
            for (int64_t ol = 0; ol < IL2; ++ol) {
                double ref = (double) B2[oc];
                for (int64_t ic = 0; ic < IC; ++ic) {
                    for (int64_t kx = 0; kx < KW; ++kx) {
                        const int64_t sx = ol + kx - p0;
                        if (sx < 0 || sx >= IL2) continue;
                        ref += (double) K2[oc * (IC * KW) + ic * KW + kx] * (double) X2[ic * IL2 + sx];
                    }
                }
                const double got = (double) ((const float*) out2->data)[oc * IL2 + ol];
                const double rel = std::fabs(got - ref) / (std::fabs(ref) + 1e-6);
                if (rel > worst2) worst2 = rel;
            }
        }
        LOOM_CHECK(worst2 < 1e-5);
        if (worst2 >= 1e-5) {
            std::fprintf(stderr, "  aliased destination: worst relative error %.3e -- the fusion took a "
                                 "pattern where the convolution overwrites its own input\n", worst2);
        }

        // and it must still be FUSED: staging is what makes the aliased case work, so falling back to
        // the unfused path here would quietly cost the win on every convolution that matters.
        bool written2 = false;
        for (int64_t i = 0; i < ggml_nelements(conv2); ++i) {
            if (((const float*) conv2->data)[i] != kPoison) written2 = true;
        }
        LOOM_CHECK(written2 == fusion_disabled);
    }

    ggml_free(ctx);
    LOOM_TEST_REPORT_AND_RETURN();
}
