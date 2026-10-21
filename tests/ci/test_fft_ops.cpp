// The engine's FFT and the STFT / ISTFT primitives built on it (ADR-080).
//
// Four things are held here:
//
//   * **The FFT against an f64 DFT**, forward and inverse, at every size the zoo uses (16, 20, 256, 400,
//     512, 1024, 1920) and at sizes that reach the code paths those do not: one point (radix "1"),
//     radix 3 and 5 alone, and the generic odd butterfly (7, 11). Bound: 2e-6 of the largest output,
//     where the measured worst is ~5e-7 -- a wrong twiddle or a swapped butterfly sign is O(1).
//   * **ISTFT against the lowering it replaces**: the three CONV_TRANSPOSE_1D over the windowed DFT basis
//     plus ADD and DIV that loom-exporter's istft.py traced before it, built here from the same formula
//     and run through ggml. Both are also held against an f64 reference, so a defect shared by the two
//     spellings cannot hide. center=false keeps sample 0, where a periodic Hann's square sum is 0 --
//     both must give the same NaN there, which is the module's own 0/0.
//   * **STFT against the two CONV_1D** coremltools lowers `torch.stft` to (cos and -sin bases, the
//     window folded in), through the engine's own CONV_1D primitive, and both against f64.
//   * **Thread-count invariance, bit for bit**: the kernel splits the output among threads and sums
//     every sample's frames in frame order, so 1 and 4 threads must agree exactly.

#include "ggml_test_helpers.h"
#include "test_util.h"
#include "cpu_backend.h"

#include "loom/loom.h"
#include "loom/ops/fft.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <vector>

using loom_test::GgmlScratch;
using loom_test::get_f32;
using loom_test::set_f32;

namespace {

const loom::PrimitiveFn& op(const std::string& name) {
    return loom::PrimitiveRegistry::instance().get(name);
}

uint32_t g_rng = 12345;
float rnd() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return static_cast<float>(g_rng >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
}

double angle(long k, long n, long N) { return 2.0 * M_PI * static_cast<double>((k * n) % N) / N; }

void test_fft_against_f64_dft() {
    for (int n : {2, 4, 6, 10, 16, 20, 98, 154, 256, 400, 512, 1024, 1920}) {
        const loom::fft::RealFft& plan = loom::fft::RealFft::plan(n);
        const int B = n / 2 + 1;
        std::vector<std::complex<float>> work(plan.work_size());

        std::vector<float> x(n), re(B), im(B);
        for (float& v : x) v = rnd();
        plan.forward(x.data(), re.data(), im.data(), work.data());
        double err = 0, peak = 0;
        for (int k = 0; k < B; ++k) {
            double sr = 0, si = 0;
            for (int j = 0; j < n; ++j) {
                sr += x[j] * std::cos(angle(k, j, n));
                si -= x[j] * std::sin(angle(k, j, n));
            }
            err = std::max({err, std::fabs(re[k] - sr), std::fabs(im[k] - si)});
            peak = std::max({peak, std::fabs(sr), std::fabs(si)});
        }
        const double fwd = err / peak;

        // a spectrum with non-zero DC and Nyquist imaginary parts, which a real inverse must ignore
        for (float& v : re) v = rnd();
        for (float& v : im) v = rnd();
        std::vector<float> y(n);
        plan.inverse(re.data(), im.data(), y.data(), work.data());
        err = peak = 0;
        for (int j = 0; j < n; ++j) {
            double s = 0;
            for (int k = 0; k < B; ++k) {
                const double w = (k == 0 || k == n / 2) ? 1.0 / n : 2.0 / n;
                s += w * (re[k] * std::cos(angle(k, j, n)) - im[k] * std::sin(angle(k, j, n)));
            }
            err = std::max(err, std::fabs(y[j] - s));
            peak = std::max(peak, std::fabs(s));
        }
        const double inv = err / peak;
        std::printf("  fft n=%-5d forward %.2e  inverse %.2e\n", n, fwd, inv);
        LOOM_CHECK(fwd <= 2e-6);
        LOOM_CHECK(inv <= 2e-6);
    }
}

std::vector<double> periodic_hann(int n) {
    std::vector<double> w(n);
    for (int i = 0; i < n; ++i) w[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / n);
    return w;
}

bool compute(ggml_backend_t backend, ggml_cgraph* gf, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_n_threads != nullptr) set_n_threads(backend, n_threads);
    return ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
}

struct Case {
    int n, hop, frames;
    bool center;
};

void test_istft_matches_the_conv_transpose_lowering(const Case& c) {
    const int n = c.n, hop = c.hop, F = c.frames, B = n / 2 + 1;
    const std::vector<double> win = periodic_hann(n);
    std::vector<float> re(static_cast<size_t>(F) * B), im(re.size());
    for (float& v : re) v = rnd();
    for (float& v : im) v = rnd();

    // f64 reference over the full signal, then the crop
    const int full = (F - 1) * hop + n, off = c.center ? n / 2 : 0, len = c.center ? full - n : full;
    std::vector<double> sig(full, 0.0), ws(full, 0.0);
    for (int f = 0; f < F; ++f)
        for (int j = 0; j < n; ++j) {
            double s = 0;
            for (int k = 0; k < B; ++k) {
                const double w = (k == 0 || k == n / 2) ? 1.0 / n : 2.0 / n;
                s += w * (re[static_cast<size_t>(k) * F + f] * std::cos(angle(k, j, n)) -
                          im[static_cast<size_t>(k) * F + f] * std::sin(angle(k, j, n)));
            }
            sig[f * hop + j] += s * win[j];
            ws[f * hop + j] += win[j] * win[j];
        }

    GgmlScratch s(64 * 1024 * 1024);
    ggml_context* ctx = s.ctx.get();
    ggml_tensor* tre = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, B, 1);
    ggml_tensor* tim = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, B, 1);
    ggml_tensor* twin = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    // the module's kernels, [K, OC=1, IC=B], and its all-ones frame row for the window-square sum
    ggml_tensor* kc = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, 1, B);
    ggml_tensor* ks = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, 1, B);
    ggml_tensor* kw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, 1, 1);
    ggml_tensor* ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F, 1);
    for (ggml_tensor* t : {tre, tim, twin, kc, ks, kw, ones}) ggml_set_input(t);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{ctx, env, nullptr};
    ggml_tensor* got = op("ISTFT")(pc, {tre, tim, twin},
                                   {{"n_fft", n}, {"hop", hop}, {"center", c.center ? 1 : 0}})[0];
    LOOM_CHECK(got->ne[0] == len && got->ne[1] == 1);

    ggml_tensor* a = ggml_conv_transpose_1d(ctx, kc, tre, hop, 0, 1);
    ggml_tensor* b = ggml_conv_transpose_1d(ctx, ks, tim, hop, 0, 1);
    ggml_tensor* w = ggml_conv_transpose_1d(ctx, kw, ones, hop, 0, 1);
    ggml_tensor* old = ggml_div(ctx, ggml_add(ctx, a, b), w);
    old = ggml_cont(ctx, ggml_view_1d(ctx, old, len, off * sizeof(float)));

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, got);
    ggml_build_forward_expand(gf, old);
    LOOM_CHECK(ggml_gallocr_alloc_graph(s.galloc.get(), gf));

    std::vector<float> vc(static_cast<size_t>(n) * B), vs(vc.size()), vw(n), vwin(n), vones(F, 1.0f);
    for (int k = 0; k < B; ++k)
        for (int j = 0; j < n; ++j) {
            const double wk = (k == 0 || k == n / 2) ? 1.0 / n : 2.0 / n;
            vc[static_cast<size_t>(k) * n + j] = static_cast<float>(wk * win[j] * std::cos(angle(k, j, n)));
            vs[static_cast<size_t>(k) * n + j] = static_cast<float>(-wk * win[j] * std::sin(angle(k, j, n)));
        }
    for (int j = 0; j < n; ++j) {
        vw[j] = static_cast<float>(win[j] * win[j]);
        vwin[j] = static_cast<float>(win[j]);
    }
    // Set before EACH compute: gallocr may hand an input's memory to a later node once the input is
    // consumed, so a second compute over the same allocation reads whatever was left there.
    auto set_inputs = [&] {
        set_f32(tre, re);
        set_f32(tim, im);
        set_f32(twin, vwin);
        set_f32(kc, vc);
        set_f32(ks, vs);
        set_f32(kw, vw);
        set_f32(ones, vones);
    };

    set_inputs();
    LOOM_CHECK(compute(s.backend, gf, 1));
    const std::vector<float> g1 = get_f32(got), o = get_f32(old);
    set_inputs();
    LOOM_CHECK(compute(s.backend, gf, 4));
    const std::vector<float> g4 = get_f32(got);

    double eg = 0, eo = 0, peak = 0;
    size_t nan_mismatch = 0, thread_diff = 0;
    for (int i = 0; i < len; ++i) {
        const double ref = sig[i + off] / ws[i + off];
        if (std::memcmp(&g1[i], &g4[i], sizeof(float)) != 0) {
            if (thread_diff < 3) std::printf("    DIFF i=%d of %d: %.9g vs %.9g\n", i, len, g1[i], g4[i]);
            ++thread_diff;
        }
        if (!std::isfinite(ref)) {   // 0/0 where the window square sums to zero
            if (std::isfinite(g1[i]) || std::isfinite(o[i])) ++nan_mismatch;
            continue;
        }
        eg = std::max(eg, std::fabs(g1[i] - ref));
        eo = std::max(eo, std::fabs(o[i] - ref));
        peak = std::max(peak, std::fabs(ref));
    }
    std::printf("  istft n=%-5d hop=%-4d F=%-3d center=%d  fft %.2e  conv %.2e  (of the peak)\n", n, hop, F,
                c.center ? 1 : 0, eg / peak, eo / peak);
    LOOM_CHECK(eg / peak <= 2e-6);
    LOOM_CHECK(eo / peak <= 2e-5);   // the old lowering is the looser of the two, by its N-term sums
    LOOM_CHECK(nan_mismatch == 0);
    LOOM_CHECK(thread_diff == 0);
}

void test_stft_matches_the_conv_lowering(int n, int hop, int frames, int win_length) {
    const int B = n / 2 + 1, L = (frames - 1) * hop + n;
    // torch.stft's win_length < n_fft: a periodic Hann of win_length, centred in n_fft zeros
    std::vector<double> win(n, 0.0);
    const int left = (n - win_length) / 2;
    for (int i = 0; i < win_length; ++i) win[left + i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / win_length);
    std::vector<float> x(L);
    for (float& v : x) v = rnd();

    GgmlScratch s(64 * 1024 * 1024);
    ggml_context* ctx = s.ctx.get();
    ggml_tensor* tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, 1);
    ggml_tensor* twin = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor* kc = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, 1, B);   // CONV_1D kernel [K, IC, OC]
    ggml_tensor* ks = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, 1, B);
    for (ggml_tensor* t : {tx, twin, kc, ks}) ggml_set_input(t);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{ctx, env, nullptr};
    const auto got = op("STFT")(pc, {tx, twin}, {{"n_fft", n}, {"hop", hop}});
    LOOM_CHECK(got.size() == 2);
    LOOM_CHECK(got[0]->ne[0] == frames && got[0]->ne[1] == B && got[0]->ne[2] == 1);
    ggml_tensor* x3 = ggml_reshape_3d(ctx, tx, L, 1, 1);
    const nlohmann::json conv = {{"s0", hop}, {"p0", 0}, {"d0", 1}};
    ggml_tensor* oc = op("CONV_1D")(pc, {kc, x3}, conv)[0];
    ggml_tensor* os = op("CONV_1D")(pc, {ks, x3}, conv)[0];
    ggml_tensor* gr = ggml_cont(ctx, got[0]);
    ggml_tensor* gi = ggml_cont(ctx, got[1]);

    ggml_cgraph* gf = ggml_new_graph(ctx);
    for (ggml_tensor* t : {gr, gi, oc, os}) ggml_build_forward_expand(gf, t);
    LOOM_CHECK(ggml_gallocr_alloc_graph(s.galloc.get(), gf));

    std::vector<float> vc(static_cast<size_t>(n) * B), vs(vc.size()), vwin(n);
    for (int k = 0; k < B; ++k)
        for (int j = 0; j < n; ++j) {
            vc[static_cast<size_t>(k) * n + j] = static_cast<float>(win[j] * std::cos(angle(k, j, n)));
            vs[static_cast<size_t>(k) * n + j] = static_cast<float>(-win[j] * std::sin(angle(k, j, n)));
        }
    for (int j = 0; j < n; ++j) vwin[j] = static_cast<float>(win[j]);
    auto set_inputs = [&] {
        set_f32(tx, x);
        set_f32(twin, vwin);
        set_f32(kc, vc);
        set_f32(ks, vs);
    };
    set_inputs();
    LOOM_CHECK(compute(s.backend, gf, 1));
    const std::vector<float> r1 = get_f32(gr), i1 = get_f32(gi), rc = get_f32(oc), ic = get_f32(os);
    set_inputs();
    LOOM_CHECK(compute(s.backend, gf, 4));
    const std::vector<float> r4 = get_f32(gr), i4 = get_f32(gi);

    double eg = 0, eo = 0, peak = 0;
    size_t thread_diff = 0;
    for (int f = 0; f < frames; ++f)
        for (int k = 0; k < B; ++k) {
            double sr = 0, si = 0;
            for (int j = 0; j < n; ++j) {
                const double v = x[static_cast<size_t>(f) * hop + j] * win[j];
                sr += v * std::cos(angle(k, j, n));
                si -= v * std::sin(angle(k, j, n));
            }
            const size_t i = static_cast<size_t>(k) * frames + f;
            eg = std::max({eg, std::fabs(r1[i] - sr), std::fabs(i1[i] - si)});
            eo = std::max({eo, std::fabs(rc[i] - sr), std::fabs(ic[i] - si)});
            peak = std::max({peak, std::fabs(sr), std::fabs(si)});
            if (std::memcmp(&r1[i], &r4[i], sizeof(float)) != 0 || std::memcmp(&i1[i], &i4[i], sizeof(float)) != 0) {
                ++thread_diff;
            }
        }
    std::printf("  stft  n=%-5d hop=%-4d F=%-3d win=%-4d   fft %.2e  conv %.2e  (of the peak)\n", n, hop, frames,
                win_length, eg / peak, eo / peak);
    LOOM_CHECK(eg / peak <= 2e-6);
    LOOM_CHECK(eo / peak <= 2e-5);
    LOOM_CHECK(thread_diff == 0);
}

void test_istft_rejects_bad_shapes() {
    GgmlScratch s;
    ggml_context* ctx = s.ctx.get();
    loom::SymbolEnv env;
    loom::PrimitiveContext pc{ctx, env, nullptr};
    ggml_tensor* re = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 10, 9, 1);
    ggml_tensor* win = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    ggml_tensor* win15 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 15);
    const nlohmann::json ok = {{"n_fft", 16}, {"hop", 4}, {"center", 1}};
    LOOM_CHECK_THROWS(op("ISTFT")(pc, {re, re, win15}, ok), loom::SchemaError);
    LOOM_CHECK_THROWS(op("ISTFT")(pc, {re, re, win}, {{"n_fft", 18}, {"hop", 4}, {"center", 1}}), loom::SchemaError);
    LOOM_CHECK_THROWS(op("ISTFT")(pc, {re, re, win}, {{"n_fft", 15}, {"hop", 4}, {"center", 1}}), loom::SchemaError);
    LOOM_CHECK_THROWS(op("ISTFT")(pc, {re, re}, ok), loom::SchemaError);
}

}  // namespace

int main() {
    test_fft_against_f64_dft();
    for (const Case& c : {Case{16, 4, 37, true}, Case{20, 5, 29, true}, Case{16, 4, 23, false},
                          Case{1024, 256, 31, true}, Case{1920, 480, 9, true}, Case{512, 160, 7, true}}) {
        test_istft_matches_the_conv_transpose_lowering(c);
    }
    test_stft_matches_the_conv_lowering(512, 160, 41, 400);   // NeMo: Citrinet, conformer
    test_stft_matches_the_conv_lowering(400, 160, 33, 400);   // Whisper, WakeHuBERT
    test_stft_matches_the_conv_lowering(256, 128, 17, 256);   // Silero
    test_stft_matches_the_conv_lowering(1024, 256, 9, 1024);  // Qwen3-TTS's speaker encoder
    test_stft_matches_the_conv_lowering(16, 4, 5, 12);
    test_istft_rejects_bad_shapes();
    LOOM_TEST_REPORT_AND_RETURN();
}
