// The AArch64 dynamic int8 mul_mat (P4.37, ggml-0047): CPU_NEON_I8, a buffer type for Q8_0 weights whose
// blocks carry one scale per row (ADR-076's Q8_0_PC), on cores without DOTPROD (the Pi 4's Cortex-A72).
// The kernel is XNNPACK's qd8-f32-qc8w 2x8c2s4 NEON GEMM, with the activations quantized per 32 k as
// ggml's own Q8_0 path does rather than per row.
//
// Its contract is integer arithmetic plus a fixed float prologue and epilogue, so it is checked BIT FOR BIT
// against a reference written here: per row and 32-k block, s = absmax / 127 and q = round(x * (127 /
// absmax)) to nearest even; the block's int32 sum of q w, accumulated in float as fma(sum, s, f) block by
// block; then f * sw[n]. Registered twice, GGML_CPU_NEON_I8=0 and =1 (read once). On AArch64
// without DOTPROD the first must find no such buffer type and the second must find it and pass every
// check below. Elsewhere the type never exists, and both runs check only that.
//
//   1. Shapes on 1, 2 and 4 threads: N not a multiple of the 8-column panel, odd and batched row counts,
//      and both schedules (few rows dealt panel by panel; many rows in L2 blocks, split by panels or,
//      with few panels, by row pairs).
//   2. The int16 bound: every activation at -127 against weights of -128, K = 3072, so every pair of
//      products sits at 2 x 16256.
//   3. A NaN poisons its own row only.
//   4. A weight whose blocks have different scales is not packed and computes exactly as the same bytes
//      in a plain CPU buffer.
//   5. A convolution on a weight in this buffer is declined, so the engine leaves a conv kernel's weight
//      in the plain buffer and its direct path.
//   6. GET_ROWS on a weight in this buffer (a tied embedding's lookup) gives the same bytes as ggml's own
//      Q8_0 GET_ROWS on the same weight in a plain buffer, at 1, 2 and 4 threads.

#include "cpu_backend.h"
#include "test_util.h"

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

ggml_backend_t backend() {
    static ggml_backend_t b = loom_test::cpu_backend();
    return b;
}

bool compute(ggml_cgraph* gf, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend()));
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_n_threads != nullptr) set_n_threads(backend(), n_threads);
    return ggml_backend_graph_compute(backend(), gf) == GGML_STATUS_SUCCESS;
}

ggml_backend_buffer_type_t i8_buft() {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend());
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (get_extra == nullptr) return nullptr;
    for (ggml_backend_buffer_type_t* p = get_extra(dev); p != nullptr && *p != nullptr; ++p) {
        if (std::strcmp(ggml_backend_buft_name(*p), "CPU_NEON_I8") == 0) return *p;
    }
    return nullptr;
}

struct Block { uint16_t d; int8_t qs[32]; };
static_assert(sizeof(Block) == 34, "block_q8_0");

// A Q8_0 weight [N][K] in [-128, 127] with one F16 scale per row (or, `uniform` false, a different one per
// block).
std::vector<Block> make_weight(int64_t K, int64_t N, int seed, bool uniform) {
    std::vector<Block> w((size_t) (N * K / 32));
    uint32_t s = 2166136261u ^ (uint32_t) seed;
    auto next = [&] { s = s * 1664525u + 1013904223u; return s >> 8; };
    for (int64_t n = 0; n < N; ++n) {
        const float scale = 0.002f + 0.0005f * (float) (n % 13);
        for (int64_t b = 0; b < K / 32; ++b) {
            Block& blk = w[(size_t) (n * (K / 32) + b)];
            blk.d = ggml_fp32_to_fp16(uniform ? scale : scale * (1.0f + 0.25f * (float) (b % 3)));
            for (int i = 0; i < 32; ++i) blk.qs[i] = (int8_t) ((int) (next() % 256) - 128);
        }
    }
    return w;
}

// The contract, for one row: what CPU_NEON_I8 must write, to the bit.
void reference(const std::vector<Block>& w, int64_t K, int64_t N, const float* x, float* y) {
    const int64_t nb = K / 32;
    std::vector<int32_t> q((size_t) K);
    std::vector<float> sb((size_t) nb);
    for (int64_t b = 0; b < nb; ++b) {
        float amax = 0.0f;
        for (int64_t k = b * 32; k < b * 32 + 32; ++k) amax = std::max(amax, std::fabs(x[k]));
        sb[(size_t) b] = amax / 127.0f;
        const float mul = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (int64_t k = b * 32; k < b * 32 + 32; ++k) q[(size_t) k] = (int32_t) std::nearbyint(x[k] * mul);
    }
    for (int64_t n = 0; n < N; ++n) {
        float f = 0.0f;
        for (int64_t b = 0; b < nb; ++b) {
            int32_t acc = 0;
            for (int64_t k = b * 32; k < b * 32 + 32; ++k) acc += (int32_t) w[(size_t) (n * nb + b)].qs[k % 32] * q[(size_t) k];
            f = std::fma((float) acc, sb[(size_t) b], f);
        }
        y[n] = f * ggml_fp16_to_fp32(w[(size_t) (n * nb)].d);
    }
}

struct Run {
    std::vector<float> y;
    bool ok = false;
};

// y = mul_mat(w, x) with w in `wbuft`, x [K][T][B] (filled by `fill`), on `n_threads`.
template <typename Fill>
Run run(ggml_backend_buffer_type_t wbuft, const std::vector<Block>& w, int64_t K, int64_t N, int64_t T, int64_t B,
        int n_threads, Fill fill) {
    Run r;
    ggml_context* cw = ggml_init({ ggml_tensor_overhead() * 2, nullptr, true });
    ggml_context* cx = ggml_init({ ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true });
    ggml_tensor* wt = ggml_new_tensor_2d(cw, GGML_TYPE_Q8_0, K, N);
    ggml_tensor* xt = ggml_new_tensor_3d(cx, GGML_TYPE_F32, K, T, B);
    ggml_tensor* yt = ggml_mul_mat(cx, wt, xt);
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, wbuft);
    ggml_backend_buffer_t bx = ggml_backend_alloc_ctx_tensors_from_buft(cx, ggml_backend_cpu_buffer_type());
    if (bw != nullptr && bx != nullptr) {
        ggml_backend_tensor_set(wt, w.data(), 0, ggml_nbytes(wt));
        fill((float*) xt->data);
        std::memset(yt->data, 0xff, ggml_nbytes(yt));   // an output never written reads as NaN
        ggml_cgraph* gf = ggml_new_graph(cx);
        ggml_build_forward_expand(gf, yt);
        r.ok = compute(gf, n_threads);
        r.y.assign((const float*) yt->data, (const float*) yt->data + ggml_nelements(yt));
    }
    ggml_backend_buffer_free(bw);
    ggml_backend_buffer_free(bx);
    ggml_free(cw);
    ggml_free(cx);
    return r;
}

// Activations with a sign bias and a magnitude that vary by row and block.
void fill_x(float* x, int64_t K, int64_t rows, int seed) {
    for (int64_t r = 0; r < rows; ++r) {
        const float bias = 0.4f * (float) ((r + seed) % 5) - 0.8f;
        for (int64_t k = 0; k < K; ++k) {
            const int64_t i = r * K + k;
            x[i] = (0.01f * (float) ((((i + seed) * 7919) % 211) - 105) + bias) * (1.0f + (float) ((k / 32) % 4));
        }
    }
}

bool same_bits(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }

}  // namespace

int main() {
    std::fesetround(FE_TONEAREST);
    const char* e = std::getenv("GGML_CPU_NEON_I8");
    const bool want_on = !(e != nullptr && std::atoi(e) == 0);
#if defined(__aarch64__)
    const bool eligible = !ggml_cpu_has_dotprod();
#else
    const bool eligible = false;
#endif
    LOOM_CHECK(backend() != nullptr);
    ggml_backend_buffer_type_t buft = i8_buft();
    LOOM_CHECK((buft != nullptr) == (eligible && want_on));
    if (buft == nullptr) {
        std::printf("CPU_NEON_I8 %s: nothing more to check here\n",
                    eligible ? "switched off" : "absent on this ISA or core");
        LOOM_TEST_REPORT_AND_RETURN();
    }

    // 1. The shapes. Few rows (rows x K <= 256 KB) are dealt panel by panel; past that, blocks of panels
    //    split by panel range when there are at least four panels a thread, else by row pairs.
    struct Shape { int64_t K, N, T, B; };
    const Shape shapes[] = { {32, 8, 1, 1}, {64, 5, 1, 1}, {768, 768, 1, 1}, {288, 13, 3, 1}, {1152, 20, 5, 1},
                             {3072, 12, 2, 1}, {96, 36, 7, 2}, {256, 8, 17, 1}, {32, 4, 16, 1},
                             {1024, 136, 300, 1},   // many rows, 17 panels: by panel range at 1-4 threads
                             {1024, 20, 301, 1},    // many rows, 3 panels: by row pairs, an odd count
                             {512, 72, 260, 2} };   // many rows over a batched src1
    for (const Shape& s : shapes) {
        const std::vector<Block> w = make_weight(s.K, s.N, (int) (s.K + s.N), true);
        for (int threads : {1, 2, 4}) {
            std::vector<float> x;
            Run r = run(buft, w, s.K, s.N, s.T, s.B, threads, [&](float* p) {
                fill_x(p, s.K, s.T * s.B, (int) s.N);
                x.assign(p, p + s.K * s.T * s.B);
            });
            LOOM_CHECK(r.ok);
            int bad = 0;
            std::vector<float> ref((size_t) s.N);
            for (int64_t c = 0; c < s.T * s.B; ++c) {
                reference(w, s.K, s.N, x.data() + c * s.K, ref.data());
                for (int64_t n = 0; n < s.N; ++n) bad += !same_bits(r.y[(size_t) (c * s.N + n)], ref[(size_t) n]);
            }
            if (bad) std::printf("K=%lld N=%lld T=%lld B=%lld threads=%d: %d outputs differ from the contract\n",
                                 (long long) s.K, (long long) s.N, (long long) s.T, (long long) s.B, threads, bad);
            LOOM_CHECK(bad == 0);
        }
    }

    // 2. The int16 bound: x all -1.5 puts every q at -127, against weights of -128.
    for (int64_t T : {1, 2, 3}) {
        const int64_t K = 3072, N = 16;
        std::vector<Block> w((size_t) (N * K / 32));
        for (Block& b : w) { b.d = ggml_fp32_to_fp16(0.001f); std::memset(b.qs, 0x80, 32); }
        std::vector<float> x;
        Run r = run(buft, w, K, N, T, 1, 1, [&](float* p) {
            for (int64_t k = 0; k < K * T; ++k) p[k] = -1.5f;
            x.assign(p, p + K * T);
        });
        LOOM_CHECK(r.ok);
        std::vector<float> ref((size_t) N);
        reference(w, K, N, x.data(), ref.data());
        LOOM_CHECK(ref[0] > 0.0f);
        for (int64_t i = 0; i < N * T; ++i) LOOM_CHECK(same_bits(r.y[(size_t) i], ref[(size_t) (i % N)]));
    }

    // 3. A NaN poisons its own row only, on both schedules.
    for (int64_t T : {3, 300}) {
        const int64_t K = 1024, N = 40;
        const std::vector<Block> w = make_weight(K, N, 3, true);
        std::vector<float> x;
        Run r = run(buft, w, K, N, T, 1, 2, [&](float* p) {
            fill_x(p, K, T, 5);
            p[K + 17] = NAN;
            x.assign(p, p + K * T);
        });
        LOOM_CHECK(r.ok);
        std::vector<float> ref((size_t) N);
        int bad = 0;
        for (int64_t c = 0; c < T; ++c) {
            if (c != 1) reference(w, K, N, x.data() + c * K, ref.data());
            for (int64_t n = 0; n < N; ++n) {
                const float got = r.y[(size_t) (c * N + n)];
                bad += !(c == 1 ? std::isnan(got) : same_bits(got, ref[(size_t) n]));
            }
        }
        if (bad) std::printf("NaN row, T=%lld: %d outputs wrong\n", (long long) T, bad);
        LOOM_CHECK(bad == 0);
    }

    // 4. Blocks with different scales in a row: stored as given, computed by ggml's own Q8_0 mul_mat.
    {
        const int64_t K = 256, N = 12, T = 5;
        const std::vector<Block> w = make_weight(K, N, 9, false);
        auto fill = [&](float* p) { fill_x(p, K, T, 11); };
        Run a = run(buft, w, K, N, T, 1, 1, fill);
        Run b = run(ggml_backend_cpu_buffer_type(), w, K, N, T, 1, 1, fill);
        LOOM_CHECK(a.ok && b.ok);
        LOOM_CHECK(a.y.size() == b.y.size() && std::memcmp(a.y.data(), b.y.data(), a.y.size() * 4) == 0);
    }

    // 5. A pointwise convolution on a packed kernel is declined.
    {
        const int64_t IC = 256, OC = 16, T = 40;
        ggml_context* cw = ggml_init({ ggml_tensor_overhead() * 2, nullptr, true });
        ggml_context* cx = ggml_init({ ggml_tensor_overhead() * 4, nullptr, true });
        ggml_tensor* wt = ggml_new_tensor_2d(cw, GGML_TYPE_Q8_0, IC, OC);
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, buft);
        LOOM_CHECK(bw != nullptr);
        ggml_tensor* xt = ggml_new_tensor_4d(cx, GGML_TYPE_F32, T, 1, IC, 1);
        ggml_tensor* conv = ggml_conv_2d_direct_packed(cx, wt, xt, 1, 1, 0, 0, 1, 1, 1, 1, (int) IC);
        LOOM_CHECK(!ggml_backend_dev_supports_op(ggml_backend_get_device(backend()), conv));
        ggml_backend_buffer_free(bw);
        ggml_free(cw);
        ggml_free(cx);
    }

    // 6. GET_ROWS from the packed panels against ggml's own, on the same bytes.
    for (const int64_t N : {13, 64}) {
        const int64_t K = 96;
        const std::vector<Block> w = make_weight(K, N, 17, true);
        const std::vector<int32_t> idx = { 0, (int32_t) N - 1, 7, 8, 3, (int32_t) N / 2 };
        std::vector<std::vector<float>> outs;
        for (ggml_backend_buffer_type_t wbuft : { ggml_backend_cpu_buffer_type(), buft }) {
            for (int threads : {1, 2, 4}) {
                ggml_context* cw = ggml_init({ ggml_tensor_overhead() * 2, nullptr, true });
                ggml_context* cx = ggml_init({ ggml_tensor_overhead() * 6 + ggml_graph_overhead(), nullptr, true });
                ggml_tensor* wt = ggml_new_tensor_2d(cw, GGML_TYPE_Q8_0, K, N);
                ggml_tensor* i1 = ggml_new_tensor_1d(cx, GGML_TYPE_I32, (int64_t) idx.size());
                ggml_tensor* y1 = ggml_get_rows(cx, wt, i1);
                ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, wbuft);
                ggml_backend_buffer_t bx = ggml_backend_alloc_ctx_tensors_from_buft(cx, ggml_backend_cpu_buffer_type());
                LOOM_CHECK(bw != nullptr && bx != nullptr);
                ggml_backend_tensor_set(wt, w.data(), 0, ggml_nbytes(wt));
                std::memcpy(i1->data, idx.data(), idx.size() * 4);
                LOOM_CHECK(ggml_backend_dev_supports_op(ggml_backend_get_device(backend()), y1));
                ggml_cgraph* gf = ggml_new_graph(cx);
                ggml_build_forward_expand(gf, y1);
                std::memset(y1->data, 0xff, ggml_nbytes(y1));
                LOOM_CHECK(compute(gf, threads));
                std::vector<float> o((const float*) y1->data, (const float*) y1->data + ggml_nelements(y1));
                outs.push_back(o);
                ggml_backend_buffer_free(bw);
                ggml_backend_buffer_free(bx);
                ggml_free(cw);
                ggml_free(cx);
            }
        }
        for (size_t i = 1; i < outs.size(); ++i) {
            const bool same = std::memcmp(outs[i].data(), outs[0].data(), outs[0].size() * 4) == 0;
            if (!same) std::printf("GET_ROWS N=%lld, run %zu: differs from ggml's own on a plain buffer\n", (long long) N, i);
            LOOM_CHECK(same);
        }
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
