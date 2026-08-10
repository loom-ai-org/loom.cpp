// Exercises each milestone-1 primitive in isolation (bypassing GraphTopology/GraphBuilder entirely) by
// invoking PrimitiveRegistry::instance().get(op) directly against hand-built input tensors with known
// values, and checking the result against a hand-computed expectation.

#include "ggml_test_helpers.h"
#include "test_util.h"

#include "loom/loom.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>

using loom_test::GgmlScratch;
using loom_test::get_f32;
using loom_test::set_f32;
using loom_test::set_i32;

namespace {

const loom::PrimitiveFn& op(const std::string& name) {
    return loom::PrimitiveRegistry::instance().get(name);
}

void test_get_rows() {
    GgmlScratch s;
    // 4 rows of 3 elements each.
    ggml_tensor* data = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 3, 4);
    ggml_set_input(data);
    ggml_tensor* idx = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_I32, 2);
    ggml_set_input(idx);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    auto outs = op("GET_ROWS")(pc, {data, idx}, {});
    LOOM_CHECK(outs.size() == 1);
    ggml_tensor* out = outs[0];
    LOOM_CHECK(out->ne[0] == 3);
    LOOM_CHECK(out->ne[1] == 2);

    ggml_cgraph* gf = s.expand(out);
    set_f32(data, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    set_i32(idx, {2, 0});
    s.compute(gf);

    auto result = get_f32(out);
    const std::vector<float> expected = {7, 8, 9, 1, 2, 3};
    LOOM_CHECK(result == expected);
}

void test_mul_mat_identity() {
    GgmlScratch s;
    // a: 2x2 identity. b: ne=[2,3] (3 "columns" of length 2). mul_mat(identity, b) == b.
    ggml_tensor* a = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 2, 2);
    ggml_set_input(a);
    ggml_tensor* b = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 2, 3);
    ggml_set_input(b);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    auto outs = op("MUL_MAT")(pc, {a, b}, {});
    ggml_tensor* out = outs[0];
    LOOM_CHECK(out->ne[0] == 2);
    LOOM_CHECK(out->ne[1] == 3);

    ggml_cgraph* gf = s.expand(out);
    set_f32(a, {1, 0, 0, 1});
    const std::vector<float> b_data = {1, 2, 3, 4, 5, 6};
    set_f32(b, b_data);
    s.compute(gf);

    LOOM_CHECK(get_f32(out) == b_data);
}

void test_add_mul_silu() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(a);
    ggml_tensor* b = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(b);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* added = op("ADD")(pc, {a, b}, {})[0];
    ggml_tensor* muled = op("MUL")(pc, {a, b}, {})[0];
    ggml_tensor* silued = op("SILU")(pc, {a}, {})[0];

    ggml_cgraph* gf = ggml_new_graph(s.ctx.get());
    ggml_build_forward_expand(gf, added);
    ggml_build_forward_expand(gf, muled);
    ggml_build_forward_expand(gf, silued);
    ggml_gallocr_alloc_graph(s.galloc.get(), gf);

    const std::vector<float> a_data = {1, 2, 3, 4};
    const std::vector<float> b_data = {10, 20, 30, 40};
    set_f32(a, a_data);
    set_f32(b, b_data);
    s.compute(gf);

    LOOM_CHECK((get_f32(added) == std::vector<float>{11, 22, 33, 44}));
    LOOM_CHECK((get_f32(muled) == std::vector<float>{10, 40, 90, 160}));

    auto silu_result = get_f32(silued);
    bool silu_ok = true;
    for (size_t i = 0; i < a_data.size(); ++i) {
        const float x = a_data[i];
        const float expected = x / (1.0f + std::exp(-x));
        if (std::fabs(silu_result[i] - expected) > 1e-5f) silu_ok = false;
    }
    LOOM_CHECK(silu_ok);
}

void test_swiglu() {
    GgmlScratch s;
    // SWIGLU(gate, up) == silu(gate) * up (ggml_swiglu_split semantics).
    ggml_tensor* gate = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 3);
    ggml_set_input(gate);
    ggml_tensor* up = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 3);
    ggml_set_input(up);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("SWIGLU")(pc, {gate, up}, {})[0];

    ggml_cgraph* gf = s.expand(out);
    const std::vector<float> gate_data = {-1.0f, 0.0f, 2.0f};
    const std::vector<float> up_data = {2.0f, 3.0f, 0.5f};
    set_f32(gate, gate_data);
    set_f32(up, up_data);
    s.compute(gf);

    auto result = get_f32(out);
    bool ok = true;
    for (size_t i = 0; i < gate_data.size(); ++i) {
        const float g = gate_data[i];
        const float expected = (g / (1.0f + std::exp(-g))) * up_data[i];
        if (std::fabs(result[i] - expected) > 1e-5f) ok = false;
    }
    LOOM_CHECK(ok);
}

void test_rms_norm() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(a);

    loom::SymbolEnv env;
    env.set("rms_norm_eps", 1e-5);
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    nlohmann::json attrs = {{"eps", "$rms_norm_eps"}};
    ggml_tensor* out = op("RMS_NORM")(pc, {a}, attrs)[0];

    ggml_cgraph* gf = s.expand(out);
    const std::vector<float> a_data = {1, 2, 3, 4};
    set_f32(a, a_data);
    s.compute(gf);

    double mean_sq = 0.0;
    for (float v : a_data) mean_sq += static_cast<double>(v) * v;
    mean_sq /= a_data.size();
    const double scale = 1.0 / std::sqrt(mean_sq + 1e-5);

    auto result = get_f32(out);
    bool ok = true;
    for (size_t i = 0; i < a_data.size(); ++i) {
        if (std::fabs(result[i] - a_data[i] * scale) > 1e-4) ok = false;
    }
    LOOM_CHECK(ok);
}

void test_layer_norm() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    nlohmann::json attrs = {{"eps", 1e-5}};
    ggml_tensor* out = op("LAYER_NORM")(pc, {a}, attrs)[0];

    ggml_cgraph* gf = s.expand(out);
    const std::vector<float> a_data = {1, 2, 3, 4};
    set_f32(a, a_data);
    s.compute(gf);

    double mean = 0.0;
    for (float v : a_data) mean += v;
    mean /= a_data.size();
    double var = 0.0;
    for (float v : a_data) var += (v - mean) * (v - mean);
    var /= a_data.size();
    const double inv_std = 1.0 / std::sqrt(var + 1e-5);

    auto result = get_f32(out);
    bool ok = true;
    for (size_t i = 0; i < a_data.size(); ++i) {
        const double expected = (a_data[i] - mean) * inv_std;
        if (std::fabs(result[i] - expected) > 1e-4) ok = false;
    }
    LOOM_CHECK(ok);
}

void test_sigmoid() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("SIGMOID")(pc, {a}, {})[0];

    ggml_cgraph* gf = s.expand(out);
    const std::vector<float> a_data = {-2.0f, -0.5f, 0.5f, 2.0f};
    set_f32(a, a_data);
    s.compute(gf);

    auto result = get_f32(out);
    bool ok = true;
    for (size_t i = 0; i < a_data.size(); ++i) {
        const double expected = 1.0 / (1.0 + std::exp(-static_cast<double>(a_data[i])));
        if (std::fabs(result[i] - expected) > 1e-5) ok = false;
    }
    LOOM_CHECK(ok);
}

void test_tanh() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("TANH")(pc, {a}, {})[0];

    ggml_cgraph* gf = s.expand(out);
    const std::vector<float> a_data = {-2.0f, -0.5f, 0.5f, 2.0f};
    set_f32(a, a_data);
    s.compute(gf);

    auto result = get_f32(out);
    bool ok = true;
    for (size_t i = 0; i < a_data.size(); ++i) {
        const double expected = std::tanh(static_cast<double>(a_data[i]));
        if (std::fabs(result[i] - expected) > 1e-5) ok = false;
    }
    LOOM_CHECK(ok);
}

void test_sin_cos() {
    // Added for Kokoro TTS (StyleTTS2-family): the NSF harmonic source's SineGen and the Generator's
    // "Snake1D" activation (x + sin(a*x)^2/a) both need raw sin/cos -- ggml already has ggml_sin/
    // ggml_cos natively (confirmed by reading ggml.h directly), so this is a thin wrapper, same pattern
    // as every other one-line primitive in this file. Two independent scratch graphs (rather than one
    // multi-output graph) since GgmlScratch::expand only supports a single declared output.
    const std::vector<float> a_data = {0.0f, 0.5f, 1.5707963f, 3.1415926f};
    {
        GgmlScratch s;
        ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 4);
        ggml_set_input(a);
        loom::SymbolEnv env;
        loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
        ggml_tensor* out = op("SIN")(pc, {a}, {})[0];
        ggml_cgraph* gf = s.expand(out);
        set_f32(a, a_data);
        s.compute(gf);
        auto result = get_f32(out);
        bool ok = true;
        for (size_t i = 0; i < a_data.size(); ++i) {
            if (std::fabs(result[i] - std::sin(static_cast<double>(a_data[i]))) > 1e-5) ok = false;
        }
        LOOM_CHECK(ok);
    }
    {
        GgmlScratch s;
        ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 4);
        ggml_set_input(a);
        loom::SymbolEnv env;
        loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
        ggml_tensor* out = op("COS")(pc, {a}, {})[0];
        ggml_cgraph* gf = s.expand(out);
        set_f32(a, a_data);
        s.compute(gf);
        auto result = get_f32(out);
        bool ok = true;
        for (size_t i = 0; i < a_data.size(); ++i) {
            if (std::fabs(result[i] - std::cos(static_cast<double>(a_data[i]))) > 1e-5) ok = false;
        }
        LOOM_CHECK(ok);
    }
}

void test_relu() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("RELU")(pc, {a}, {})[0];

    ggml_cgraph* gf = s.expand(out);
    set_f32(a, {-2.0f, -0.5f, 0.5f, 2.0f});
    s.compute(gf);

    LOOM_CHECK((get_f32(out) == std::vector<float>{0.0f, 0.0f, 0.5f, 2.0f}));
}

void test_leaky_relu() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("LEAKY_RELU")(pc, {a}, {{"slope", 0.1}})[0];

    ggml_cgraph* gf = s.expand(out);
    set_f32(a, {-2.0f, -0.5f, 0.5f, 2.0f});
    s.compute(gf);

    const auto result = get_f32(out);
    const std::vector<float> expected = {-0.2f, -0.05f, 0.5f, 2.0f};
    bool ok = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(result[i] - expected[i]) > 1e-6f) ok = false;
    }
    LOOM_CHECK(ok);
}

void test_step() {
    GgmlScratch s;
    // ggml_step is strict: x>0 -> 1, else 0 (confirmed against ggml-cpu/vec.h's ggml_vec_step_f32
    // directly) -- 0.0 itself maps to 0, not 1. STEP(SUB(a,b)) composes a strict "a>b" comparison;
    // "a>=b" needs 1-STEP(b-a) instead (used by generate_path/the rational-quadratic spline bucketize).
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("STEP")(pc, {a}, {})[0];

    ggml_cgraph* gf = s.expand(out);
    set_f32(a, {-1.0f, 0.0f, 0.0001f, 5.0f});
    s.compute(gf);

    LOOM_CHECK((get_f32(out) == std::vector<float>{0.0f, 0.0f, 1.0f, 1.0f}));
}

void test_group_norm() {
    GgmlScratch s;
    // 4 channels (ne[2]), 2 groups, spatial ne[0]=2 x ne[1]=1 -- matches the [T,1,C,1] reshape convention
    // this project's own Layout A [T,C] tensors need before calling ggml_group_norm (confirmed to group
    // over ne[2] and reduce over ne[0]*ne[1] directly against ggml-cpu/ops.cpp's
    // ggml_compute_forward_group_norm_f32). No learned affine here -- GROUP_NORM itself never applies one,
    // same as RMS_NORM/LAYER_NORM.
    ggml_tensor* a = ggml_new_tensor_4d(s.ctx.get(), GGML_TYPE_F32, 2, 1, 4, 1);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("GROUP_NORM")(pc, {a}, {{"n_groups", 2}, {"eps", 1e-5}})[0];

    ggml_cgraph* gf = s.expand(out);
    // channel0={1,2} channel1={3,4} channel2={10,20} channel3={30,40}
    set_f32(a, {1, 2, 3, 4, 10, 20, 30, 40});
    s.compute(gf);

    const auto result = get_f32(out);
    // group0 (channels 0,1): mean=2.5, var=1.25 -> std=sqrt(1.25+eps)
    // group1 (channels 2,3): mean=25, var=125 -> std=sqrt(125+eps)
    const double std0 = std::sqrt(1.25 + 1e-5);
    const double std1 = std::sqrt(125.0 + 1e-5);
    const std::vector<float> expected = {
        static_cast<float>((1 - 2.5) / std0), static_cast<float>((2 - 2.5) / std0),
        static_cast<float>((3 - 2.5) / std0), static_cast<float>((4 - 2.5) / std0),
        static_cast<float>((10 - 25.0) / std1), static_cast<float>((20 - 25.0) / std1),
        static_cast<float>((30 - 25.0) / std1), static_cast<float>((40 - 25.0) / std1),
    };
    bool ok = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(result[i] - expected[i]) > 1e-4f) ok = false;
    }
    LOOM_CHECK(ok);
}

void test_cumsum() {
    GgmlScratch s;
    // ne=[4,2]: two independent rows, cumsum along ne[0] (confirmed against ggml-cpu's real
    // ggml_compute_forward_cumsum_f32, which sums along ne00 per row) -- verifies both the per-row
    // math and that rows don't cross-contaminate.
    ggml_tensor* a = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 4, 2);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("CUMSUM")(pc, {a}, {})[0];

    ggml_cgraph* gf = s.expand(out);
    set_f32(a, {1.0f, 2.0f, 3.0f, 4.0f, /*row1*/ 10.0f, 0.0f, -5.0f, 1.0f});
    s.compute(gf);

    LOOM_CHECK((get_f32(out) == std::vector<float>{1.0f, 3.0f, 6.0f, 10.0f, /*row1*/ 10.0f, 10.0f, 5.0f, 6.0f}));
}

void test_glu() {
    GgmlScratch s;
    // ne=[OL=2, channels=4, N=1]: channels 0-1 are the "value" half, channels 2-3 the "gate" half
    // (all zero -> sigmoid(0)=0.5), so the expected output is exactly half the value half.
    ggml_tensor* a = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 2, 4, 1);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("GLU")(pc, {a}, {})[0];
    LOOM_CHECK(out->ne[0] == 2);
    LOOM_CHECK(out->ne[1] == 2);

    ggml_cgraph* gf = s.expand(out);
    set_f32(a, {1, 2, 3, 4, 0, 0, 0, 0});
    s.compute(gf);

    LOOM_CHECK((get_f32(out) == std::vector<float>{0.5f, 1.0f, 1.5f, 2.0f}));
}

void test_conv_1d_dw() {
    GgmlScratch s;
    // 2 channels, K=3, "same" padding (p0=1,s0=1,d0=1) so OL == IL == 4.
    // kernel: ch0=[1,1,1] (sliding sum), ch1=[1,0,-1] (edge filter).
    ggml_tensor* kernel = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 3, 1, 2);
    ggml_set_input(kernel);
    ggml_tensor* data = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 4, 2, 1);
    ggml_set_input(data);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    nlohmann::json attrs = {{"s0", 1}, {"p0", 1}, {"d0", 1}};
    ggml_tensor* out = op("CONV_1D_DW")(pc, {kernel, data}, attrs)[0];
    LOOM_CHECK(out->ne[0] == 4);
    LOOM_CHECK(out->ne[1] == 2);

    ggml_cgraph* gf = s.expand(out);
    set_f32(kernel, {1, 1, 1, 1, 0, -1});
    set_f32(data, {1, 2, 3, 4, 5, 6, 7, 8});
    s.compute(gf);

    // ch0 (sum filter over [0,1,2,3,4,0] padded): [3,6,9,7]
    // ch1 (edge filter over [0,5,6,7,8,0] padded): [-6,-2,-2,7]
    LOOM_CHECK((get_f32(out) == std::vector<float>{3, 6, 9, 7, -6, -2, -2, 7}));
}

void test_depthwise_conv_transpose_1d_via_composition() {
    // Kokoro's AdainResBlk1d "pool" (a weight-normed, DEPTHWISE ConvTranspose1d, kernel=3, stride=2,
    // padding=1, output_padding=1 -- real shapes confirmed against the checkpoint's own
    // `predictor.F0.1.pool.weight_v` (512,1,3)) has no direct ggml primitive (ggml_conv_transpose_1d
    // is non-grouped only, and this project has no CONV_TRANSPOSE_1D_DW). Composes entirely from
    // EXISTING primitives instead: zero-stuff the input (RESHAPE to insert a dummy fastest axis +
    // PAD_1D by stride-1 on it + RESHAPE flattens back, giving a length-L_in*stride "overstuffed"
    // signal with a trailing (stride-1) extra zeros vs. the textbook zero-stuffing scheme) -> VIEW+CONT
    // truncates to the textbook (L_in-1)*stride+1 length -> PAD_1D by (kernel-1-padding) each side plus
    // output_padding on the right -> CONV_1D_DW with the kernel REVERSED along its own length axis (a
    // conversion-time transform of the real weight, not a runtime op -- transposed convolution is
    // cross-correlation with a flipped kernel). Verified against real
    // torch.nn.functional.conv_transpose1d(groups=channels) on a small hand-picked example (C=2,L_in=4)
    // BEFORE trusting this composition for anything real, matching this project's usual discipline for
    // novel compositions (INTERPOLATE_1D's bilinear-degenerates-to-1D-linear trick, the ISTFT-via-
    // CONV_TRANSPOSE_1D derivation, etc.).
    constexpr int64_t kC = 2, kLIn = 4, kStride = 2, kKernel = 3, kPadding = 1, kOutputPadding = 1;
    GgmlScratch s;
    ggml_tensor* data = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, kLIn, kC);
    ggml_set_input(data);
    ggml_tensor* kernel_flipped = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, kKernel, 1, kC);
    ggml_set_input(kernel_flipped);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};

    ggml_tensor* d3 = op("RESHAPE")(pc, {data}, {{"shape", {1, kLIn, kC}}})[0];
    ggml_tensor* stuffed3 = op("PAD_1D")(pc, {d3}, {{"lp0", 0}, {"rp0", kStride - 1}})[0]; // [stride,L_in,C]
    ggml_tensor* overstuffed = op("RESHAPE")(pc, {stuffed3}, {{"shape", {kLIn * kStride, kC}}})[0];
    constexpr int64_t kStdLen = (kLIn - 1) * kStride + 1;
    ggml_tensor* truncated_view = op("VIEW")(pc, {overstuffed}, {{"shape", {kStdLen, kC}}})[0];
    ggml_tensor* truncated = op("CONT")(pc, {truncated_view}, {})[0];
    constexpr int64_t kPadEach = kKernel - 1 - kPadding;
    ggml_tensor* padded = op("PAD_1D")(pc, {truncated}, {{"lp0", kPadEach}, {"rp0", kPadEach + kOutputPadding}})[0];
    ggml_tensor* out = op("CONV_1D_DW")(pc, {kernel_flipped, padded}, {{"s0", 1}, {"p0", 0}, {"d0", 1}})[0];

    LOOM_CHECK(out->ne[0] == 8);
    LOOM_CHECK(out->ne[1] == kC);

    ggml_cgraph* gf = s.expand(out);
    set_f32(data, {1.5409960746765137f, -0.293428897857666f, -2.1787893772125244f, 0.5684312582015991f,
                   -1.0845223665237427f, -1.3985954523086548f, 0.40334683656692505f, 0.8380263447761536f});
    // kernel, PRE-FLIPPED along its own length axis (conversion-time transform, not a runtime op):
    // real w = [[-0.7192576,-0.4033435,-0.5966353],[0.1820365,-0.8566746,1.1006042]] -> reversed per row.
    set_f32(kernel_flipped, {-0.5966353416442871f, -0.40334352850914f, -0.7192575931549072f,
                             1.1006041765213013f, -0.8566746115684509f, 0.18203648924827576f});
    s.compute(gf);

    const std::vector<float> expected = {
        -0.6215507984161377f, -0.7083617448806763f, 0.11835264414548874f, 1.7421808242797852f,
        0.8788005709648132f, 0.8910942673683167f, -0.22927306592464447f, -0.3391461670398712f,
        0.9290827512741089f, -1.4482252597808838f, 1.1981412172317505f, -1.4658761024475098f,
        -0.345537006855011f, 0.5964765548706055f, -0.7179158926010132f, 0.9223352670669556f,
    };
    auto result = get_f32(out);
    bool ok = true;
    for (size_t i = 0; i < expected.size(); ++i) if (std::fabs(result[i] - expected[i]) > 1e-5f) ok = false;
    LOOM_CHECK(ok);
}

void test_reshape_permute_cont() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 6);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    nlohmann::json reshape_attrs = {{"shape", {2, 3}}};
    ggml_tensor* reshaped = op("RESHAPE")(pc, {a}, reshape_attrs)[0];
    LOOM_CHECK(reshaped->ne[0] == 2);
    LOOM_CHECK(reshaped->ne[1] == 3);

    nlohmann::json permute_attrs = {{"axes", {1, 0, 2, 3}}};
    ggml_tensor* permuted = op("PERMUTE")(pc, {reshaped}, permute_attrs)[0];
    LOOM_CHECK(permuted->ne[0] == 3);
    LOOM_CHECK(permuted->ne[1] == 2);

    ggml_tensor* conted = op("CONT")(pc, {permuted}, {})[0];
    LOOM_CHECK(conted->ne[0] == 3);
    LOOM_CHECK(conted->ne[1] == 2);

    ggml_cgraph* gf = s.expand(conted);
    // a laid out as 2 cols x 3 rows (ne0=2,ne1=3): rows = [1,2],[3,4],[5,6]
    set_f32(a, {1, 2, 3, 4, 5, 6});
    s.compute(gf);

    // permute(1,0,2,3) + cont transposes to 3 cols x 2 rows: rows = [1,3,5],[2,4,6]
    LOOM_CHECK((get_f32(conted) == std::vector<float>{1, 3, 5, 2, 4, 6}));
}

void test_reshape_infers_minus_one_dim() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 12);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    // 12 elements, explicit dim0=3 -> dim1 must be inferred as 4.
    nlohmann::json attrs = {{"shape", {3, -1}}};
    ggml_tensor* out = op("RESHAPE")(pc, {a}, attrs)[0];
    LOOM_CHECK(out->ne[0] == 3);
    LOOM_CHECK(out->ne[1] == 4);

    LOOM_CHECK_THROWS(op("RESHAPE")(pc, {a}, nlohmann::json{{"shape", {-1, -1}}}), loom::SchemaError);
    LOOM_CHECK_THROWS(op("RESHAPE")(pc, {a}, nlohmann::json{{"shape", {5, -1}}}), loom::SchemaError);
}

void test_view() {
    GgmlScratch s;
    // 4 rows of 3 elements; VIEW the middle two rows (offset = 1 row, shape = [3, 2]).
    ggml_tensor* a = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 3, 4);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    nlohmann::json attrs = {{"shape", {3, 2}}, {"offset", static_cast<int64_t>(a->nb[1])}};
    ggml_tensor* viewed = op("VIEW")(pc, {a}, attrs)[0];
    ggml_tensor* out = op("CONT")(pc, {viewed}, {})[0];
    LOOM_CHECK(out->ne[0] == 3);
    LOOM_CHECK(out->ne[1] == 2);

    ggml_cgraph* gf = s.expand(out);
    set_f32(a, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    s.compute(gf);

    LOOM_CHECK((get_f32(out) == std::vector<float>{4, 5, 6, 7, 8, 9}));
}

// A VIEW whose parent is a PERMUTE with ne[0] == 1 -- the exact shape LFM2-350M presents at
// n_tokens == 1, and the one that made its decode step fail with "needs 16380 bytes but parent has
// 12288" long after the conv state it was blamed on had been fixed (BACKLOG.md P4.0.10).
//
// The trap is that `ggml_is_contiguous` REPORTS TRUE here: its stride test is skipped entirely when
// ne[0] == blck_size (ggml.c:1467), so a permuted tensor with nb=[12288,4] passes, op_view's `cont`
// does not fire, and the bounds check saw nb[0] = 12288 where it meant one element. The view itself was
// always correct -- with ne[0] == 1, nb[0] addresses nothing -- so this is a false rejection, and the
// numbers below prove the data is right as well as the check passing.
void test_view_of_permuted_parent_with_unit_leading_axis() {
    GgmlScratch s;
    // [C=6, T=1] -> PERMUTE -> [T=1, C=6], which is the in_proj output layout at one token.
    ggml_tensor* a = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 6, 1);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* permuted = op("PERMUTE")(pc, {a}, {{"axes", {1, 0, 2, 3}}})[0];
    LOOM_CHECK(permuted->ne[0] == 1 && permuted->ne[1] == 6);
    // The precondition that makes this test meaningful: ggml calls this contiguous even though its
    // leading stride is not the element size. If ggml ever tightens that, this test still passes.
    LOOM_CHECK(permuted->nb[0] != ggml_type_size(permuted->type));

    // Split the 6 channels into two halves of 3, exactly as LFM2 splits B/C/x -- the second at a
    // non-zero offset, which is where a wrong stride would also corrupt the data rather than only
    // over-count the bounds.
    const int64_t elem = static_cast<int64_t>(ggml_type_size(permuted->type));
    ggml_tensor* first = op("CONT")(pc, {op("VIEW")(pc, {permuted},
        {{"shape", {1, 3}}, {"offset", 0}})[0]}, {})[0];
    ggml_tensor* second = op("CONT")(pc, {op("VIEW")(pc, {permuted},
        {{"shape", {1, 3}}, {"offset", 3 * elem}})[0]}, {})[0];

    ggml_cgraph* gf = ggml_new_graph(s.ctx.get());
    ggml_build_forward_expand(gf, first);
    ggml_build_forward_expand(gf, second);
    ggml_gallocr_alloc_graph(s.galloc.get(), gf);
    set_f32(a, {10, 20, 30, 40, 50, 60});
    s.compute(gf);

    LOOM_CHECK((get_f32(first) == std::vector<float>{10, 20, 30}));
    LOOM_CHECK((get_f32(second) == std::vector<float>{40, 50, 60}));
}

// The bounds check must still REJECT a genuinely out-of-range view -- the fix narrows what counts as
// out of range by one element's worth of stride, it does not remove the check.
void test_view_out_of_bounds_still_throws() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 3, 4);
    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    // One row past the end.
    LOOM_CHECK_THROWS(op("VIEW")(pc, {a}, nlohmann::json{{"shape", {3, 4}},
                                                          {"offset", static_cast<int64_t>(a->nb[1])}}),
                       loom::SchemaError);
}

void test_unknown_op_throws() {
    LOOM_CHECK_THROWS(loom::PrimitiveRegistry::instance().get("NOT_A_REAL_OP"), loom::UnknownOpError);
}

void test_rope_identity_at_position_zero() {
    GgmlScratch s;
    // a: [n_embd_head=4, n_head=1, n_tokens=2]. Both tokens at position 0 -> rotation angle 0 -> identity.
    ggml_tensor* a = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 4, 1, 2);
    ggml_set_input(a);
    ggml_tensor* pos = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_I32, 2);
    ggml_set_input(pos);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    nlohmann::json attrs = {
        {"n_dims", 4}, {"mode", 2 /* GGML_ROPE_TYPE_NEOX */}, {"n_ctx_orig", 32},
        {"freq_base", 10000.0}, {"freq_scale", 1.0}, {"ext_factor", 0.0},
        {"attn_factor", 1.0}, {"beta_fast", 32.0}, {"beta_slow", 1.0},
    };
    ggml_tensor* out = op("ROPE")(pc, {a, pos}, attrs)[0];

    ggml_cgraph* gf = s.expand(out);
    const std::vector<float> a_data = {1, 2, 3, 4, 5, 6, 7, 8};
    set_f32(a, a_data);
    set_i32(pos, {0, 0});
    s.compute(gf);

    auto result = get_f32(out);
    bool ok = true;
    for (size_t i = 0; i < a_data.size(); ++i) {
        if (std::fabs(result[i] - a_data[i]) > 1e-4f) ok = false;
    }
    LOOM_CHECK(ok);
}

void test_conv_1d() {
    GgmlScratch s;
    // kernel: K=2, IC=1, OC=1, weights=[1,1] (adjacent-pair sum filter).
    ggml_tensor* kernel = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 2, 1, 1);
    ggml_set_input(kernel);
    // data: IL=4, IC=1, N=1.
    ggml_tensor* data = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 4, 1, 1);
    ggml_set_input(data);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    nlohmann::json attrs = {{"s0", 1}, {"p0", 0}, {"d0", 1}};
    ggml_tensor* out = op("CONV_1D")(pc, {kernel, data}, attrs)[0];
    LOOM_CHECK(out->ne[0] == 3); // OL = (4 - (2-1) - 1)/1 + 1 = 3
    LOOM_CHECK(out->ne[1] == 1);

    ggml_cgraph* gf = s.expand(out);
    set_f32(kernel, {1, 1});
    set_f32(data, {1, 2, 3, 4});
    s.compute(gf);

    LOOM_CHECK((get_f32(out) == std::vector<float>{3, 5, 7}));
}

void test_conv_2d() {
    GgmlScratch s;
    // kernel: KW=2, KH=2, IC=1, OC=1, all weights=1 (2x2 sum filter).
    ggml_tensor* kernel = ggml_new_tensor_4d(s.ctx.get(), GGML_TYPE_F32, 2, 2, 1, 1);
    ggml_set_input(kernel);
    // data: IW=3, IH=3, IC=1, N=1, values 1..9 row-major (w fastest).
    ggml_tensor* data = ggml_new_tensor_4d(s.ctx.get(), GGML_TYPE_F32, 3, 3, 1, 1);
    ggml_set_input(data);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    nlohmann::json attrs = {{"s0", 1}, {"s1", 1}, {"p0", 0}, {"p1", 0}, {"d0", 1}, {"d1", 1}};
    ggml_tensor* out = op("CONV_2D")(pc, {kernel, data}, attrs)[0];
    LOOM_CHECK(out->ne[0] == 2); // OW = 3 - 2 + 1 = 2
    LOOM_CHECK(out->ne[1] == 2); // OH = 2

    ggml_cgraph* gf = s.expand(out);
    set_f32(kernel, {1, 1, 1, 1});
    set_f32(data, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    s.compute(gf);

    LOOM_CHECK((get_f32(out) == std::vector<float>{12, 16, 24, 28}));
}

void test_conv_2d_dw() {
    GgmlScratch s;
    // 2 channels, each convolved independently with its OWN 2x2 kernel -- verifies both the per-channel
    // math and that channels don't get cross-mixed. Channel 0 reuses test_conv_2d's own 2x2 all-ones sum
    // filter over data 1..9 (expected {12,16,24,28}, already hand-verified there). Channel 1 uses a
    // "pick only the top-left tap" kernel [1,0,0,0] over data {10,20,...,90} -- cross-correlation with a
    // single nonzero tap at (kw=0,kh=0) just selects data[ow,oh] directly, so the expected output is
    // exactly the input's own top-left 2x2 block: {10,20,40,50}.
    ggml_tensor* kernel = ggml_new_tensor_4d(s.ctx.get(), GGML_TYPE_F32, 2, 2, 1, 2); // [KW,KH,1,C]
    ggml_set_input(kernel);
    ggml_tensor* data = ggml_new_tensor_4d(s.ctx.get(), GGML_TYPE_F32, 3, 3, 2, 1); // [W,H,C,N]
    ggml_set_input(data);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    nlohmann::json attrs = {{"s0", 1}, {"s1", 1}, {"p0", 0}, {"p1", 0}, {"d0", 1}, {"d1", 1}};
    ggml_tensor* out = op("CONV_2D_DW")(pc, {kernel, data}, attrs)[0];
    LOOM_CHECK(out->ne[0] == 2); // OW = 3 - 2 + 1 = 2
    LOOM_CHECK(out->ne[1] == 2); // OH = 2
    LOOM_CHECK(out->ne[2] == 2); // channels preserved

    ggml_cgraph* gf = s.expand(out);
    set_f32(kernel, {1, 1, 1, 1, /*ch1*/ 1, 0, 0, 0});
    set_f32(data, {1, 2, 3, 4, 5, 6, 7, 8, 9, /*ch1*/ 10, 20, 30, 40, 50, 60, 70, 80, 90});
    s.compute(gf);

    LOOM_CHECK((get_f32(out) == std::vector<float>{12, 16, 24, 28, /*ch1*/ 10, 20, 40, 50}));
}

void test_conv_transpose_1d() {
    GgmlScratch s;
    // kernel: K=2, OC=1, IC=1, values=[1,1]. data: IL=2, IC=1, values=[3,4]. stride=1 (K > stride, so
    // this exercises the accumulating-overlap path, not just non-overlapping scatter).
    ggml_tensor* kernel = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 2, 1, 1);
    ggml_set_input(kernel);
    ggml_tensor* data = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 2, 1);
    ggml_set_input(data);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("CONV_TRANSPOSE_1D")(pc, {kernel, data}, {{"s0", 1}})[0];
    LOOM_CHECK(out->ne[0] == 3); // OL = (2-1)*1 + 2 = 3

    ggml_cgraph* gf = s.expand(out);
    set_f32(kernel, {1, 2});
    set_f32(data, {3, 4});
    s.compute(gf);

    // out[i10*s0+i00] += data[i10]*kernel[i00]: out[0]+=3*1=3; out[1]+=3*2=6; out[1]+=4*1=4 (->10); out[2]+=4*2=8.
    LOOM_CHECK((get_f32(out) == std::vector<float>{3, 10, 8}));
}

void test_conv_transpose_2d() {
    GgmlScratch s;
    // kernel: KW=2, KH=2, OC=1, IC=1, values=[1,2,3,4]. data: a single pixel (IW=IH=1,IC=1) = 2.
    // With a single input pixel and stride == kernel size (no overlap), the output is exactly the
    // kernel scaled by that one value -- output[row,col] = data * kernel[row,col], flattened identically
    // to the kernel's own storage order.
    ggml_tensor* kernel = ggml_new_tensor_4d(s.ctx.get(), GGML_TYPE_F32, 2, 2, 1, 1);
    ggml_set_input(kernel);
    ggml_tensor* data = ggml_new_tensor_4d(s.ctx.get(), GGML_TYPE_F32, 1, 1, 1, 1);
    ggml_set_input(data);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("CONV_TRANSPOSE_2D")(pc, {kernel, data}, {{"s0", 2}})[0];
    LOOM_CHECK(out->ne[0] == 2); // OW = (1-1)*2 + 2 = 2
    LOOM_CHECK(out->ne[1] == 2); // OH = 2

    ggml_cgraph* gf = s.expand(out);
    set_f32(kernel, {1, 2, 3, 4});
    set_f32(data, {2});
    s.compute(gf);

    LOOM_CHECK((get_f32(out) == std::vector<float>{2, 4, 6, 8}));
}

void test_pool_1d() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};

    ggml_tensor* max_out = op("POOL_1D")(pc, {a}, {{"op", "max"}, {"k0", 2}, {"s0", 2}, {"p0", 0}})[0];
    ggml_tensor* avg_out = op("POOL_1D")(pc, {a}, {{"op", "avg"}, {"k0", 2}, {"s0", 2}, {"p0", 0}})[0];
    LOOM_CHECK(max_out->ne[0] == 2);

    ggml_cgraph* gf = ggml_new_graph(s.ctx.get());
    ggml_build_forward_expand(gf, max_out);
    ggml_build_forward_expand(gf, avg_out);
    ggml_gallocr_alloc_graph(s.galloc.get(), gf);
    set_f32(a, {1, 2, 3, 4});
    s.compute(gf);

    LOOM_CHECK((get_f32(max_out) == std::vector<float>{2, 4}));
    LOOM_CHECK((get_f32(avg_out) == std::vector<float>{1.5f, 3.5f}));
}

void test_pool_2d() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 4, 4);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    nlohmann::json attrs = {{"op", "max"}, {"k0", 2}, {"k1", 2}, {"s0", 2}, {"s1", 2}, {"p0", 0.0}, {"p1", 0.0}};
    ggml_tensor* out = op("POOL_2D")(pc, {a}, attrs)[0];
    LOOM_CHECK(out->ne[0] == 2);
    LOOM_CHECK(out->ne[1] == 2);

    ggml_cgraph* gf = s.expand(out);
    set_f32(a, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    s.compute(gf);

    LOOM_CHECK((get_f32(out) == std::vector<float>{6, 8, 14, 16}));
}

void test_gelu() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("GELU")(pc, {a}, {})[0];

    ggml_cgraph* gf = s.expand(out);
    const std::vector<float> a_data = {-2.0f, -0.5f, 0.5f, 2.0f};
    set_f32(a, a_data);
    s.compute(gf);

    auto result = get_f32(out);
    bool ok = true;
    for (size_t i = 0; i < a_data.size(); ++i) {
        const float x = a_data[i];
        const float expected = 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
        if (std::fabs(result[i] - expected) > 1e-5f) ok = false;
    }
    LOOM_CHECK(ok);
}

void test_attention_without_kv_cache() {
    GgmlScratch s;
    // n_head = n_head_kv = 1, n_embd_head = 2, n_tokens = 2. Both q rows identical (and equal to both k
    // rows) -> QK^T scores are equal across the kv axis for every query, so softmax is exactly uniform
    // (0.5/0.5) regardless of scale -- every output row must equal the plain average of the two v rows.
    ggml_tensor* q = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 2, 1, 2);
    ggml_set_input(q);
    ggml_tensor* k = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 2, 1, 2);
    ggml_set_input(k);
    ggml_tensor* v = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 2, 1, 2);
    ggml_set_input(v);
    ggml_tensor* mask = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 2, 2);
    ggml_set_input(mask);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, /*kv_cache=*/nullptr};
    nlohmann::json attrs = {{"kv_cache", false}, {"scale", 0.5}};
    ggml_tensor* out = op("ATTENTION")(pc, {q, k, v, mask}, attrs)[0];
    LOOM_CHECK(out->ne[0] == 2); // n_embd_head * n_head
    LOOM_CHECK(out->ne[1] == 2); // n_tokens

    ggml_cgraph* gf = s.expand(out);
    set_f32(q, {1, 1, 1, 1});           // both tokens: [1, 1]
    set_f32(k, {1, 1, 1, 1});           // both tokens: [1, 1] (same as q)
    set_f32(v, {2, 4, 6, 8});           // token0 v=[2,4], token1 v=[6,8]
    set_f32(mask, {0, 0, 0, 0});        // fully unmasked
    s.compute(gf);

    const std::vector<float> expected_row = {4, 6}; // average of [2,4] and [6,8]
    auto result = get_f32(out);
    LOOM_CHECK(result.size() == 4);
    bool ok = true;
    for (int row = 0; row < 2; ++row) {
        for (int i = 0; i < 2; ++i) {
            if (std::fabs(result[row * 2 + i] - expected_row[i]) > 1e-4f) ok = false;
        }
    }
    LOOM_CHECK(ok);

    // No KvCache and no SchemaError: kv_cache=false must not require pc.kv_cache.
    LOOM_CHECK_THROWS(
        op("ATTENTION")(pc, {q, k, v, mask}, nlohmann::json{{"kv_cache", true}, {"scale", 0.5}, {"layer", 0}}),
        loom::SchemaError);
}

void test_rel_shift() {
    // Cross-checked against actual PyTorch execution of the exact rel_shift algorithm on this same
    // qlen=2, pos_len=3 example (see the implementation plan): input rows [0,1,2] and [3,4,5] (row-major,
    // i.e. ne=[pos_len=3, qlen=2, n_head=1]) shift to [1,2,0] and [3,4,5].
    GgmlScratch s;
    ggml_tensor* x = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 3, 2, 1);
    ggml_set_input(x);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("REL_SHIFT")(pc, {x}, {})[0];
    LOOM_CHECK(out->ne[0] == 3);
    LOOM_CHECK(out->ne[1] == 2);

    ggml_cgraph* gf = s.expand(out);
    set_f32(x, {0, 1, 2, 3, 4, 5});
    s.compute(gf);

    LOOM_CHECK((get_f32(out) == std::vector<float>{1, 2, 0, 3, 4, 5}));
}

void test_rel_pos_attention() {
    // n_head=1, head_dim=2, n_tokens=2, n_pos=3 (=2*n_tokens-1). pos_bias_u/v and the projected
    // positional embedding p are all zero, so matrix_bd contributes nothing (rel_shift's own numerics
    // are independently covered by test_rel_shift) -- this isolates and verifies the rest of the
    // mechanics: dual-matrix scoring plumbing, masked softmax, and the weighted value sum.
    GgmlScratch s;
    ggml_tensor* q = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 2, 1, 2);
    ggml_set_input(q);
    ggml_tensor* k = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 2, 1, 2);
    ggml_set_input(k);
    ggml_tensor* v = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 2, 1, 2);
    ggml_set_input(v);
    ggml_tensor* p = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 2, 1, 3);
    ggml_set_input(p);
    ggml_tensor* pos_bias_u = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 2, 1);
    ggml_set_input(pos_bias_u);
    ggml_tensor* pos_bias_v = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 2, 1);
    ggml_set_input(pos_bias_v);
    ggml_tensor* mask = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 2, 2);
    ggml_set_input(mask);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    const double scale = 1.0 / std::sqrt(2.0);
    nlohmann::json attrs = {{"scale", scale}};
    ggml_tensor* out = op("REL_POS_ATTENTION")(pc, {q, k, v, p, pos_bias_u, pos_bias_v, mask}, attrs)[0];
    LOOM_CHECK(out->ne[0] == 2);
    LOOM_CHECK(out->ne[1] == 2);

    ggml_cgraph* gf = s.expand(out);
    set_f32(q, {1, 0, 0, 1});   // token0=(1,0), token1=(0,1)
    set_f32(k, {1, 0, 0, 1});   // same as q (self-attention)
    set_f32(v, {10, 20, 30, 40}); // token0=(10,20), token1=(30,40)
    set_f32(p, {0, 0, 0, 0, 0, 0});
    set_f32(pos_bias_u, {0, 0});
    set_f32(pos_bias_v, {0, 0});
    set_f32(mask, {0, 0, 0, 0});
    s.compute(gf);

    // matrix_ac[kv,q]=dot(k[kv],q[q]): for q=0 -> [dot(k0,q0)=1, dot(k1,q0)=0]; for q=1 -> [0,1].
    // matrix_bd=0 (p=0). scores = matrix_ac * scale, softmax over kv, then weighted sum of v.
    auto softmax2 = [&](double a, double b) {
        const double ea = std::exp(a * scale), eb = std::exp(b * scale);
        const double sum = ea + eb;
        return std::make_pair(ea / sum, eb / sum);
    };
    auto [p00, p01] = softmax2(1.0, 0.0); // query 0's weights over kv={0,1}
    auto [p10, p11] = softmax2(0.0, 1.0); // query 1's weights over kv={0,1}

    const std::vector<float> expected = {
        static_cast<float>(p00 * 10 + p01 * 30), static_cast<float>(p00 * 20 + p01 * 40), // query 0
        static_cast<float>(p10 * 10 + p11 * 30), static_cast<float>(p10 * 20 + p11 * 40), // query 1
    };
    auto result = get_f32(out);
    bool ok = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(result[i] - expected[i]) > 1e-4f) ok = false;
    }
    LOOM_CHECK(ok);
}

void test_rel_to_abs_shaw() {
    // Cross-checked against actual PyTorch execution of the real
    // attentions.py::_relative_position_to_absolute_position on this same length=4 example (values
    // 0..27, row-major, ne=[2*length-1=7, length=4, n_head=1]) before trusting this translation.
    GgmlScratch s;
    ggml_tensor* x = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 7, 4, 1);
    ggml_set_input(x);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("REL_TO_ABS_SHAW")(pc, {x}, {})[0];
    LOOM_CHECK(out->ne[0] == 4);
    LOOM_CHECK(out->ne[1] == 4);

    ggml_cgraph* gf = s.expand(out);
    std::vector<float> input_data(28);
    for (size_t i = 0; i < input_data.size(); ++i) input_data[i] = static_cast<float>(i);
    set_f32(x, input_data);
    s.compute(gf);

    LOOM_CHECK((get_f32(out) == std::vector<float>{3, 4, 5, 6, 9, 10, 11, 12, 15, 16, 17, 18, 21, 22, 23, 24}));
}

void test_abs_to_rel_shaw() {
    // Cross-checked against actual PyTorch execution of the real
    // attentions.py::_absolute_position_to_relative_position on this same length=4 example (values
    // 0..15, row-major, ne=[length=4, length=4, n_head=1]) before trusting this translation.
    GgmlScratch s;
    ggml_tensor* x = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 4, 4, 1);
    ggml_set_input(x);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("ABS_TO_REL_SHAW")(pc, {x}, {})[0];
    LOOM_CHECK(out->ne[0] == 7);
    LOOM_CHECK(out->ne[1] == 4);

    ggml_cgraph* gf = s.expand(out);
    std::vector<float> input_data(16);
    for (size_t i = 0; i < input_data.size(); ++i) input_data[i] = static_cast<float>(i);
    set_f32(x, input_data);
    s.compute(gf);

    LOOM_CHECK((get_f32(out) == std::vector<float>{
        0, 0, 0, 0, 1, 2, 3,
        0, 0, 4, 5, 6, 7, 0,
        0, 8, 9, 10, 11, 0, 0,
        12, 13, 14, 15, 0, 0, 0,
    }));
}

void test_rel_pos_attention_shaw() {
    // n_head=1, head_dim=2, n_tokens=2 (so 2*n_tokens-1=3). emb_rel_k/v are zero, so the relative terms
    // contribute nothing (the skew tricks' own numerics are independently covered by
    // test_rel_to_abs_shaw/test_abs_to_rel_shaw) -- this isolates and verifies the rest of the
    // mechanics: unbiased dual-matrix scoring plumbing, masked softmax, weighted value sum, and that the
    // (zero) relative-value term adds cleanly. Same q/k/v/mask/expected-output values as
    // test_rel_pos_attention for direct comparability -- confirms REL_POS_ATTENTION_SHAW's content-term
    // path degenerates to the same plain dot-product attention when there's no positional signal.
    GgmlScratch s;
    ggml_tensor* q = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 2, 1, 2);
    ggml_set_input(q);
    ggml_tensor* k = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 2, 1, 2);
    ggml_set_input(k);
    ggml_tensor* v = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 2, 1, 2);
    ggml_set_input(v);
    ggml_tensor* emb_rel_k = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 2, 3);
    ggml_set_input(emb_rel_k);
    ggml_tensor* emb_rel_v = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 2, 3);
    ggml_set_input(emb_rel_v);
    ggml_tensor* mask = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 2, 2);
    ggml_set_input(mask);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    const double scale = 1.0 / std::sqrt(2.0);
    nlohmann::json attrs = {{"scale", scale}};
    ggml_tensor* out = op("REL_POS_ATTENTION_SHAW")(pc, {q, k, v, emb_rel_k, emb_rel_v, mask}, attrs)[0];
    LOOM_CHECK(out->ne[0] == 2);
    LOOM_CHECK(out->ne[1] == 2);

    ggml_cgraph* gf = s.expand(out);
    set_f32(q, {1, 0, 0, 1});
    set_f32(k, {1, 0, 0, 1});
    set_f32(v, {10, 20, 30, 40});
    set_f32(emb_rel_k, {0, 0, 0, 0, 0, 0});
    set_f32(emb_rel_v, {0, 0, 0, 0, 0, 0});
    set_f32(mask, {0, 0, 0, 0});
    s.compute(gf);

    auto softmax2 = [&](double a, double b) {
        const double ea = std::exp(a * scale), eb = std::exp(b * scale);
        const double sum = ea + eb;
        return std::make_pair(ea / sum, eb / sum);
    };
    auto [p00, p01] = softmax2(1.0, 0.0);
    auto [p10, p11] = softmax2(0.0, 1.0);

    const std::vector<float> expected = {
        static_cast<float>(p00 * 10 + p01 * 30), static_cast<float>(p00 * 20 + p01 * 40),
        static_cast<float>(p10 * 10 + p11 * 30), static_cast<float>(p10 * 20 + p11 * 40),
    };
    auto result = get_f32(out);
    bool ok = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(result[i] - expected[i]) > 1e-4f) ok = false;
    }
    LOOM_CHECK(ok);
}

void test_sub_div_scale() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(a);
    ggml_tensor* b = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(b);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* sub_out = op("SUB")(pc, {a, b}, {})[0];
    ggml_tensor* div_out = op("DIV")(pc, {a, b}, {})[0];
    nlohmann::json scale_attrs = {{"s", 2.5}};
    ggml_tensor* scale_out = op("SCALE")(pc, {a}, scale_attrs)[0];

    ggml_cgraph* gf = ggml_new_graph(s.ctx.get());
    ggml_build_forward_expand(gf, sub_out);
    ggml_build_forward_expand(gf, div_out);
    ggml_build_forward_expand(gf, scale_out);
    ggml_gallocr_alloc_graph(s.galloc.get(), gf);

    set_f32(a, {10, 20, 30, 40});
    set_f32(b, {1, 2, 5, 8});
    s.compute(gf);

    LOOM_CHECK((get_f32(sub_out) == std::vector<float>{9, 18, 25, 32}));
    LOOM_CHECK((get_f32(div_out) == std::vector<float>{10, 10, 6, 5}));
    LOOM_CHECK((get_f32(scale_out) == std::vector<float>{25, 50, 75, 100}));
}

void test_sqr_sqrt_log() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* sqr_out = op("SQR")(pc, {a}, {})[0];
    ggml_tensor* sqrt_out = op("SQRT")(pc, {a}, {})[0];
    ggml_tensor* log_out = op("LOG")(pc, {a}, {})[0];

    ggml_cgraph* gf = ggml_new_graph(s.ctx.get());
    ggml_build_forward_expand(gf, sqr_out);
    ggml_build_forward_expand(gf, sqrt_out);
    ggml_build_forward_expand(gf, log_out);
    ggml_gallocr_alloc_graph(s.galloc.get(), gf);

    const std::vector<float> a_data = {1, 4, 9, 16};
    set_f32(a, a_data);
    s.compute(gf);

    LOOM_CHECK((get_f32(sqr_out) == std::vector<float>{1, 16, 81, 256}));
    LOOM_CHECK((get_f32(sqrt_out) == std::vector<float>{1, 2, 3, 4}));

    auto log_result = get_f32(log_out);
    bool log_ok = true;
    for (size_t i = 0; i < a_data.size(); ++i) {
        if (std::fabs(log_result[i] - std::log(a_data[i])) > 1e-5f) log_ok = false;
    }
    LOOM_CHECK(log_ok);
}

void test_atan2() {
    // ATAN2(y,x) via ggml_map_custom2 -- no native ggml op, needed for Kokoro's Generator (real STFT
    // phase = atan2(imag,real)). Covers all 4 quadrants plus the x=0 edge cases against std::atan2.
    GgmlScratch s;
    ggml_tensor* y = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 6);
    ggml_tensor* x = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 6);
    ggml_set_input(y);
    ggml_set_input(x);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("ATAN2")(pc, {y, x}, {})[0];

    ggml_cgraph* gf = s.expand(out);
    const std::vector<float> y_data = {1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 0.0f};
    const std::vector<float> x_data = {1.0f, -1.0f, -1.0f, 1.0f, 0.0f, -1.0f};
    set_f32(y, y_data);
    set_f32(x, x_data);
    s.compute(gf);

    const auto result = get_f32(out);
    bool ok = true;
    for (size_t i = 0; i < y_data.size(); ++i) {
        if (std::fabs(result[i] - std::atan2(y_data[i], x_data[i])) > 1e-6f) ok = false;
    }
    LOOM_CHECK(ok);
}

void test_exp() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("EXP")(pc, {a}, {})[0];

    ggml_cgraph* gf = s.expand(out);
    const std::vector<float> a_data = {0.0f, 1.0f, -1.0f, 2.0f};
    set_f32(a, a_data);
    s.compute(gf);

    const auto result = get_f32(out);
    bool ok = true;
    for (size_t i = 0; i < a_data.size(); ++i) {
        if (std::fabs(result[i] - std::exp(a_data[i])) > 1e-5f) ok = false;
    }
    LOOM_CHECK(ok);
}

void test_floor() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 5);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("FLOOR")(pc, {a}, {})[0];

    ggml_cgraph* gf = s.expand(out);
    set_f32(a, {1.9f, -1.1f, 2.0f, 0.0f, -0.5f});
    s.compute(gf);

    LOOM_CHECK((get_f32(out) == std::vector<float>{1.0f, -2.0f, 2.0f, 0.0f, -1.0f}));
}

void test_sum_rows() {
    GgmlScratch s;
    // ne=[3,2]: row0=[1,2,3], row1=[4,5,6] (ggml "rows" are along ne[0]).
    ggml_tensor* a = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 3, 2);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("SUM_ROWS")(pc, {a}, {})[0];
    LOOM_CHECK(out->ne[0] == 1);
    LOOM_CHECK(out->ne[1] == 2);

    ggml_cgraph* gf = s.expand(out);
    set_f32(a, {1, 2, 3, 4, 5, 6});
    s.compute(gf);

    LOOM_CHECK((get_f32(out) == std::vector<float>{6, 15}));
}

void test_pad_1d() {
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 3);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    nlohmann::json attrs = {{"lp0", 1}, {"rp0", 2}};
    ggml_tensor* out = op("PAD_1D")(pc, {a}, attrs)[0];
    LOOM_CHECK(out->ne[0] == 6);

    ggml_cgraph* gf = s.expand(out);
    set_f32(a, {5, 6, 7});
    s.compute(gf);

    LOOM_CHECK((get_f32(out) == std::vector<float>{0, 5, 6, 7, 0, 0}));
}

void test_pad_1d_reflect() {
    // Same lp0=1/rp0=2 shape as test_pad_1d, but reflect (not zero) fill -- verified against
    // numpy.pad([5,6,7], (1,2), mode="reflect") == [6,5,6,7,6,5] (excludes the edge element itself from
    // being duplicated, matching PyTorch's own "reflect" -- not "symmetric" -- convention, needed for
    // MIL's "pad" op mode="reflect" as produced by STFT's center-framing decomposition, see
    // EXPORT-IMPROVEMENT-BACKLOG.md item 4).
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 3);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    nlohmann::json attrs = {{"lp0", 1}, {"rp0", 2}};
    ggml_tensor* out = op("PAD_1D_REFLECT")(pc, {a}, attrs)[0];
    LOOM_CHECK(out->ne[0] == 6);

    ggml_cgraph* gf = s.expand(out);
    set_f32(a, {5, 6, 7});
    s.compute(gf);

    LOOM_CHECK((get_f32(out) == std::vector<float>{6, 5, 6, 7, 6, 5}));
}

void test_concat() {
    // ggml_concat(a,b,dim) -- real in-graph channel concatenation needed by Kokoro's Decoder (torch.cat
    // sites whose operands are themselves graph-computed, unlike the host-side style-concatenation used
    // for TextEncoder/DurationEncoder). ne=[T=2,C=3] "a" concatenated with ne=[T=2,C=2] "b" along dim=1
    // (channels) -> ne=[2,5]; separately verifies dim=0 (T-axis) concatenation too.
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 2, 3);
    ggml_tensor* b = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 2, 2);
    ggml_set_input(a);
    ggml_set_input(b);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out_dim1 = op("CONCAT")(pc, {a, b}, {{"dim", 1}})[0];
    LOOM_CHECK(out_dim1->ne[0] == 2 && out_dim1->ne[1] == 5);

    ggml_cgraph* gf = s.expand(out_dim1);
    set_f32(a, {1, 2, 3, 4, 5, 6});     // ne=[2,3]: rows (t=0,1) x (c=0,1,2)
    set_f32(b, {10, 20, 30, 40});       // ne=[2,2]
    s.compute(gf);
    LOOM_CHECK((get_f32(out_dim1) == std::vector<float>{1, 2, 3, 4, 5, 6, 10, 20, 30, 40}));

    // dim=0 (T-axis) concatenation: ne=[2,2] "c" ++ ne=[3,2] "d" (same channel count) -> ne=[5,2].
    GgmlScratch s2;
    ggml_tensor* c = ggml_new_tensor_2d(s2.ctx.get(), GGML_TYPE_F32, 2, 2);
    ggml_tensor* d = ggml_new_tensor_2d(s2.ctx.get(), GGML_TYPE_F32, 3, 2);
    ggml_set_input(c);
    ggml_set_input(d);
    loom::PrimitiveContext pc2{s2.ctx.get(), env, nullptr};
    ggml_tensor* out_dim0 = op("CONCAT")(pc2, {c, d}, {{"dim", 0}})[0];
    LOOM_CHECK(out_dim0->ne[0] == 5 && out_dim0->ne[1] == 2);
    ggml_cgraph* gf2 = s2.expand(out_dim0);
    set_f32(c, {1, 2, /*row1*/ 3, 4});
    set_f32(d, {10, 20, 30, /*row1*/ 40, 50, 60});
    s2.compute(gf2);
    LOOM_CHECK((get_f32(out_dim0) == std::vector<float>{1, 2, 10, 20, 30, /*row1*/ 3, 4, 40, 50, 60}));
}

void test_repeat() {
    // ggml_repeat_4d wrapper (StyleTTS2's diffusion Transformer1d needs to broadcast a single per-batch
    // style vector [channels] up to [channels, T] before CONCAT-ing with a per-position context
    // embedding -- CONCAT itself requires matching shape on every non-concat axis). ne=[3,1] "a"
    // repeated to ne=[3,4] (a plain numeric "shape" attr here; a "$"-symbol string entry follows the
    // same SymbolEnv::eval() path already exercised by RESHAPE/VIEW elsewhere in this file).
    GgmlScratch s;
    ggml_tensor* a = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 3, 1);
    ggml_set_input(a);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("REPEAT")(pc, {a}, {{"shape", {3, 4}}})[0];
    LOOM_CHECK(out->ne[0] == 3 && out->ne[1] == 4);

    ggml_cgraph* gf = s.expand(out);
    set_f32(a, {1, 2, 3});
    s.compute(gf);
    LOOM_CHECK((get_f32(out) == std::vector<float>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}));

    // Symbol-resolved target shape ("$n_tokens", matching RESHAPE's own string-symbol convention).
    loom::SymbolEnv env2;
    env2.set("n_tokens", 5.0);
    GgmlScratch s2;
    ggml_tensor* b = ggml_new_tensor_1d(s2.ctx.get(), GGML_TYPE_F32, 1);
    ggml_set_input(b);
    loom::PrimitiveContext pc2{s2.ctx.get(), env2, nullptr};
    ggml_tensor* out2 = op("REPEAT")(pc2, {b}, {{"shape", {"1", "$n_tokens"}}})[0];
    LOOM_CHECK(out2->ne[0] == 1 && out2->ne[1] == 5);
    ggml_cgraph* gf2 = s2.expand(out2);
    set_f32(b, {7.0f});
    s2.compute(gf2);
    LOOM_CHECK((get_f32(out2) == std::vector<float>{7, 7, 7, 7, 7}));
}

void test_interpolate_1d() {
    // Cross-checked against real torch.nn.functional.interpolate on the exact same input (input {1,2,3,4},
    // ne=[4,1]) before trusting the "bilinear-with-ne1-held-fixed degenerates to 1D linear" claim
    // documented in op_interpolate_1d's own comment:
    //   F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
    //     -> [1.0, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.0]
    //   F.interpolate(x, scale_factor=0.5, mode='linear', align_corners=False) -> [1.5, 3.5]
    //   F.interpolate(x, scale_factor=2, mode='nearest') -> [1, 1, 2, 2, 3, 3, 4, 4]
    const std::vector<float> a_data = {1, 2, 3, 4};

    {
        GgmlScratch s;
        ggml_tensor* a = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 4, 1);
        ggml_set_input(a);
        loom::SymbolEnv env;
        loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
        ggml_tensor* out = op("INTERPOLATE_1D")(pc, {a}, {{"ne0", 8}, {"mode", "linear"}})[0];
        ggml_cgraph* gf = s.expand(out);
        set_f32(a, a_data);
        s.compute(gf);
        const std::vector<float> expected = {1.0f, 1.25f, 1.75f, 2.25f, 2.75f, 3.25f, 3.75f, 4.0f};
        auto result = get_f32(out);
        bool ok = true;
        for (size_t i = 0; i < expected.size(); ++i) if (std::fabs(result[i] - expected[i]) > 1e-5) ok = false;
        LOOM_CHECK(ok);
    }
    {
        GgmlScratch s;
        ggml_tensor* a = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 4, 1);
        ggml_set_input(a);
        loom::SymbolEnv env;
        loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
        ggml_tensor* out = op("INTERPOLATE_1D")(pc, {a}, {{"ne0", 2}, {"mode", "linear"}})[0];
        ggml_cgraph* gf = s.expand(out);
        set_f32(a, a_data);
        s.compute(gf);
        const std::vector<float> expected = {1.5f, 3.5f};
        auto result = get_f32(out);
        bool ok = true;
        for (size_t i = 0; i < expected.size(); ++i) if (std::fabs(result[i] - expected[i]) > 1e-5) ok = false;
        LOOM_CHECK(ok);
    }
    {
        GgmlScratch s;
        ggml_tensor* a = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 4, 1);
        ggml_set_input(a);
        loom::SymbolEnv env;
        loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
        ggml_tensor* out = op("INTERPOLATE_1D")(pc, {a}, {{"ne0", 8}, {"mode", "nearest"}})[0];
        ggml_cgraph* gf = s.expand(out);
        set_f32(a, a_data);
        s.compute(gf);
        const std::vector<float> expected = {1, 1, 2, 2, 3, 3, 4, 4};
        LOOM_CHECK((get_f32(out) == expected));
    }
}

void test_rq_spline_inverse() {
    // Cross-checked against real execution of piper's own transforms.py::piecewise_rational_quadratic_transform
    // (inverse=True, tails="linear") on this exact hand-picked (not random) num_bins=3/tail_bound=2.0/T=3
    // fixture -- expected outputs computed by literally calling that real function (see BACKLOG.md), not
    // hand-derived, since the RQS formula itself is intricate enough that a hand computation would risk
    // just re-deriving the same potential mistake twice.
    constexpr int64_t kNumBins = 3;
    constexpr int64_t kT = 3;
    GgmlScratch s;

    ggml_tensor* inputs = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, kT);
    ggml_set_input(inputs);
    ggml_tensor* uw = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, kNumBins, kT);
    ggml_set_input(uw);
    ggml_tensor* uh = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, kNumBins, kT);
    ggml_set_input(uh);
    ggml_tensor* ud = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, kNumBins - 1, kT);
    ggml_set_input(ud);
    ggml_tensor* boundary_deriv_const = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, kNumBins + 1);
    ggml_set_input(boundary_deriv_const);
    ggml_tensor* eps_bump = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, kNumBins);
    ggml_set_input(eps_bump);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    nlohmann::json attrs = {{"tail_bound", 2.0}, {"min_bin_width", 1e-3}, {"min_bin_height", 1e-3}, {"min_derivative", 1e-3}};
    ggml_tensor* out = op("RQ_SPLINE_INVERSE")(pc, {inputs, uw, uh, ud, boundary_deriv_const, eps_bump}, attrs)[0];
    LOOM_CHECK(out->ne[0] == kT);

    ggml_cgraph* gf = s.expand(out);
    set_f32(inputs, {-0.3f, 0.7f, 1.5f});
    set_f32(uw, {1.0f, 0.0f, -1.0f, 0.5f, 0.5f, 0.5f, -0.5f, 1.0f, 0.2f});
    set_f32(uh, {0.0f, 1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 0.3f, -0.3f, 0.1f});
    set_f32(ud, {0.2f, -0.2f, 0.0f, 0.5f, -0.4f, 0.3f});
    const float boundary_const = 0.5397424172369522f; // log(exp(1-1e-3)-1)
    set_f32(boundary_deriv_const, {boundary_const, 0.0f, 0.0f, boundary_const});
    set_f32(eps_bump, {0.0f, 0.0f, 1e-6f});
    s.compute(gf);

    const std::vector<float> expected = {0.9890786746197022f, 0.025953779486384554f, 1.6059942347562157f};
    const auto result = get_f32(out);
    bool ok = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(result[i] - expected[i]) > 1e-4f) ok = false;
    }
    if (!ok) {
        std::fprintf(stderr, "RQ_SPLINE_INVERSE mismatch: got [%f %f %f], expected [%f %f %f]\n",
                     static_cast<double>(result[0]), static_cast<double>(result[1]), static_cast<double>(result[2]),
                     static_cast<double>(expected[0]), static_cast<double>(expected[1]), static_cast<double>(expected[2]));
    }
    LOOM_CHECK(ok);
}

void test_rq_spline_inverse_outside_tail_bound() {
    // Same fixture as test_rq_spline_inverse, but inputs[2]=3.0 (genuinely outside tail_bound=2.0,
    // never exercised by that test since its own inputs were all in-domain) -- isolates whether the
    // outside-tail-bound identity-passthrough blend is correct, cross-checked against real execution.
    constexpr int64_t kNumBins = 3;
    constexpr int64_t kT = 3;
    GgmlScratch s;

    ggml_tensor* inputs = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, kT);
    ggml_set_input(inputs);
    ggml_tensor* uw = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, kNumBins, kT);
    ggml_set_input(uw);
    ggml_tensor* uh = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, kNumBins, kT);
    ggml_set_input(uh);
    ggml_tensor* ud = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, kNumBins - 1, kT);
    ggml_set_input(ud);
    ggml_tensor* boundary_deriv_const = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, kNumBins + 1);
    ggml_set_input(boundary_deriv_const);
    ggml_tensor* eps_bump = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, kNumBins);
    ggml_set_input(eps_bump);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    nlohmann::json attrs = {{"tail_bound", 2.0}, {"min_bin_width", 1e-3}, {"min_bin_height", 1e-3}, {"min_derivative", 1e-3}};
    ggml_tensor* out = op("RQ_SPLINE_INVERSE")(pc, {inputs, uw, uh, ud, boundary_deriv_const, eps_bump}, attrs)[0];

    ggml_cgraph* gf = s.expand(out);
    set_f32(inputs, {-0.3f, 0.7f, 3.0f});
    set_f32(uw, {1.0f, 0.0f, -1.0f, 0.5f, 0.5f, 0.5f, -0.5f, 1.0f, 0.2f});
    set_f32(uh, {0.0f, 1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 0.3f, -0.3f, 0.1f});
    set_f32(ud, {0.2f, -0.2f, 0.0f, 0.5f, -0.4f, 0.3f});
    const float boundary_const = 0.5397424172369522f;
    set_f32(boundary_deriv_const, {boundary_const, 0.0f, 0.0f, boundary_const});
    set_f32(eps_bump, {0.0f, 0.0f, 1e-6f});
    s.compute(gf);

    const std::vector<float> expected = {0.9890784621238708f, 0.025953590869903564f, 3.0f};
    const auto result = get_f32(out);
    bool ok = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(result[i] - expected[i]) > 1e-4f) ok = false;
    }
    LOOM_CHECK(ok);
}

bool approx_eq(const std::vector<float>& got, const std::vector<float>& expected, float tol) {
    if (got.size() != expected.size()) return false;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - expected[i]) > tol) return false;
    }
    return true;
}

void test_wn() {
    // Cross-checked against real execution of piper's own modules.WN (gin_channels=0, i.e. no speaker
    // conditioning, matching the real checkpoint) on a small hand-picked (hidden_channels=4,
    // kernel_size=3, dilation_rate=2, n_layers=2, T=5) fixture -- weights/inputs are random (seeded,
    // `torch.manual_seed(0)`) but the expected output was obtained by literally running the real module,
    // not hand-derived (see BACKLOG.md).
    constexpr int64_t kHidden = 4, kT = 5, kK = 3, kNLayers = 2;
    GgmlScratch s;
    ggml_context* ctx = s.ctx.get();

    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kT, kHidden);
    ggml_set_input(x);
    ggml_tensor* in_w0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kK, kHidden, 2 * kHidden);
    ggml_tensor* in_b0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * kHidden);
    ggml_tensor* rs_w0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kHidden, 2 * kHidden);
    ggml_tensor* rs_b0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * kHidden);
    ggml_tensor* in_w1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kK, kHidden, 2 * kHidden);
    ggml_tensor* in_b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * kHidden);
    ggml_tensor* rs_w1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kHidden, kHidden);
    ggml_tensor* rs_b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    for (ggml_tensor* t : {in_w0, in_b0, rs_w0, rs_b0, in_w1, in_b1, rs_w1, rs_b1}) ggml_set_input(t);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{ctx, env, nullptr};
    nlohmann::json attrs = {{"kernel_size", kK}, {"dilation_rate", 2}, {"n_layers", kNLayers}};
    ggml_tensor* out = op("WN")(pc, {x, in_w0, in_b0, rs_w0, rs_b0, in_w1, in_b1, rs_w1, rs_b1}, attrs)[0];
    LOOM_CHECK(out->ne[0] == kT);
    LOOM_CHECK(out->ne[1] == kHidden);

    ggml_cgraph* gf = s.expand(out);
    set_f32(x, {-0.36090219020843506f, 1.3102548122406006f, -0.9650526642799377f, 0.8806114196777344f, 1.0630345344543457f, 0.7378342747688293f, -0.2454289048910141f, -0.30603769421577454f, 1.9465994834899902f, 2.0370163917541504f, 1.037927269935608f, 0.33297106623649597f, 0.332072377204895f, -0.15596836805343628f, -1.3332977294921875f, -1.4993096590042114f, 0.8772966861724854f, -0.48217466473579407f, -0.17328432202339172f, -0.10489042848348618f});
    set_f32(in_w0, {0.6111522316932678f, 0.1929127424955368f, -0.2197708934545517f, -0.14631402492523193f, -0.07018730789422989f, 0.2121955007314682f, 0.17401443421840668f, 0.08049062639474869f, -0.24893678724765778f, 0.3803502023220062f, 0.0820687785744667f, -0.18439629673957825f, -0.007048321887850761f, 0.3515009582042694f, 0.11960615962743759f, -0.05961465835571289f, -0.21616026759147644f, -0.664426326751709f, -0.20511886477470398f, 0.15490730106830597f, 0.16764099895954132f, 0.23752754926681519f, -0.0554027333855629f, -0.21953175961971283f, -0.331714004278183f, 0.04311944916844368f, 0.1750790923833847f, 0.40444672107696533f, -0.24411942064762115f, 0.24599848687648773f, -0.1899520605802536f, 0.3884265124797821f, -0.07462754100561142f, -0.3696002662181854f, 0.1877012699842453f, -0.36693817377090454f, -0.1869656890630722f, -0.06487361341714859f, -0.14660266041755676f, 0.23608683049678802f, -0.21984028816223145f, 0.15429013967514038f, 0.11929459124803543f, 0.19303688406944275f, -0.35939356684684753f, 0.14353135228157043f, -0.36885735392570496f, -0.4110109806060791f, 0.12602409720420837f, 0.338710755109787f, 0.12790782749652863f, -0.3408348858356476f, -0.038769129663705826f, -0.016378426924347878f, 0.12250402569770813f, 0.3379097878932953f, -0.18237857520580292f, -0.10874497145414352f, -0.4521622061729431f, -0.15260960161685944f, -0.37277930974960327f, 0.3853650391101837f, 0.07313168793916702f, 0.15911059081554413f, -0.8272602558135986f, -0.2497074455022812f, 0.1469956785440445f, 0.08724652230739594f, 0.1932690292596817f, 1.1790012121200562f, -0.03732728958129883f, 0.08860251307487488f, 0.7259133458137512f, 0.4936833381652832f, -0.09260819852352142f, -0.4544171094894409f, 0.58370041847229f, -0.3871091604232788f, -0.7048428058624268f, -0.6206585764884949f, -0.09298187494277954f, -0.01681855134665966f, 0.15522386133670807f, -0.47887104749679565f, 0.16234254837036133f, 0.46168750524520874f, 0.32581472396850586f, 0.3739215135574341f, 0.16891655325889587f, -0.4305202066898346f, 0.21581993997097015f, -0.41122424602508545f, 0.2619691789150238f, 0.01953801140189171f, 0.231972336769104f, -0.2910415530204773f});
    set_f32(in_b0, {0.03474516049027443f, 0.05803529545664787f, -0.012889565899968147f, 0.10062911361455917f, 0.053378283977508545f, -0.008121379651129246f, -0.049508798867464066f, 0.19327621161937714f});
    set_f32(in_w1, {-0.3276289701461792f, -0.018325554206967354f, -0.44782793521881104f, -0.529322624206543f, 0.08826039731502533f, 0.23919673264026642f, 0.37926462292671204f, 0.28064772486686707f, 0.6429688930511475f, -0.28877997398376465f, -0.16906988620758057f, 0.4933788478374481f, 0.4350663125514984f, 1.2304481267929077f, 0.33546778559684753f, -0.4700542688369751f, -0.1665043979883194f, 0.23602502048015594f, 0.2044760137796402f, 0.45532816648483276f, -0.22934189438819885f, 0.07225218415260315f, 0.0499277301132679f, -0.6695444583892822f, -0.5660734176635742f, 0.22283777594566345f, -0.027737347409129143f, -0.4292701780796051f, 0.5057772994041443f, -0.36530762910842896f, 0.2294890135526657f, 0.3591354489326477f, -0.6724103689193726f, 0.13120470941066742f, -0.16660857200622559f, -0.017382973805069923f, -0.0035520680248737335f, 0.29390016198158264f, -0.319815456867218f, 0.5315890908241272f, -0.1141795963048935f, 0.33933359384536743f, -0.027061620727181435f, -0.2693699300289154f, -0.2855278253555298f, 0.08145638555288315f, 0.201473668217659f, 0.5549947023391724f, -0.19080011546611786f, 0.332449346780777f, -0.0016822589095681906f, 0.43754827976226807f, 0.6767191886901855f, 0.36863869428634644f, -0.14563825726509094f, 0.13607139885425568f, -0.1561989188194275f, 0.30766570568084717f, -0.16787242889404297f, 0.13030144572257996f, 0.08269744366407394f, 0.032907191663980484f, 0.10782723128795624f, -0.22612154483795166f, 0.06986258178949356f, 0.010922487825155258f, 0.12211213260889053f, 0.3754940927028656f, -0.3597327470779419f, -0.007705850061029196f, 0.5407124757766724f, -0.31789591908454895f, -0.0013179760426282883f, -0.6669789552688599f, 0.2220071405172348f, -0.2148292362689972f, -0.06132223829627037f, -0.09492432326078415f, 0.38812312483787537f, 0.4035786986351013f, 0.030805960297584534f, -0.14490234851837158f, -0.07823868840932846f, 0.14288993179798126f, 0.1028236672282219f, -0.4813477694988251f, -0.17619159817695618f, 0.18011654913425446f, 0.026257313787937164f, 0.2110944390296936f, -0.4029712378978729f, -0.006110383663326502f, -0.428119421005249f, 0.17776624858379364f, -0.3474574089050293f, 0.010728277266025543f});
    set_f32(in_b1, {-0.14087329804897308f, -0.08164220303297043f, 0.06984715908765793f, 0.13983400166034698f, -0.09498212486505508f, -0.06232306361198425f, 0.03649945184588432f, 0.03820106014609337f});
    set_f32(rs_w0, {0.4846949279308319f, 0.1027119904756546f, 0.0024532137904316187f, 0.09279777109622955f, 0.10926898568868637f, 0.026440218091011047f, -0.3920733630657196f, -0.21191059052944183f, 0.11430606245994568f, -0.13653281331062317f, -0.6736576557159424f, -0.3123721182346344f, -0.05718448385596275f, 0.21499529480934143f, -0.6000568270683289f, -0.7228974103927612f, -0.010035743936896324f, -0.42803847789764404f, 0.06961196660995483f, -0.5195377469062805f, -0.16382186114788055f, -0.18905343115329742f, -0.1903950423002243f, 0.2923993766307831f, 0.27940165996551514f, -0.1220245435833931f, -0.2727717161178589f, 0.16790032386779785f, -0.23206116259098053f, 0.1788642555475235f, -0.3751256763935089f, 0.34367799758911133f});
    set_f32(rs_b0, {-0.08985810726881027f, -0.12028484791517258f, 0.31501227617263794f, -0.02164987288415432f, 0.26095685362815857f, -0.027559498324990273f, 0.12810425460338593f, -0.01586892455816269f});
    set_f32(rs_w1, {-0.08879224210977554f, 0.41142433881759644f, 0.2921537756919861f, 0.2296629548072815f, 0.38865068554878235f, 0.26725849509239197f, -0.14695441722869873f, -0.3517972230911255f, 0.20302674174308777f, -0.024189766496419907f, 0.04743693396449089f, 0.029995078220963478f, -0.01785724237561226f, 0.6035525798797607f, -0.10103750973939896f, 0.09779410809278488f});
    set_f32(rs_b1, {0.06867282837629318f, 0.1043618693947792f, 0.11209886521100998f, 0.08756598085165024f});
    s.compute(gf);

    const std::vector<float> expected = {0.17084145545959473f, 0.5691779851913452f, 0.36216408014297485f, 0.5524346828460693f, 0.6482698917388916f, 0.38555648922920227f, 0.07586196064949036f, 0.12389619648456573f, 0.1796228140592575f, 0.22955842316150665f, 0.21294480562210083f, 0.07460697740316391f, 0.5642104148864746f, 0.1776123046875f, 0.41373080015182495f, 0.001019902527332306f, 0.46994107961654663f, -0.12869518995285034f, 0.2652955651283264f, 0.14895686507225037f};
    LOOM_CHECK(approx_eq(get_f32(out), expected, 1e-4f));
}

void test_residual_coupling_layer_reverse() {
    // Cross-checked against real execution of piper's own modules.ResidualCouplingLayer, reverse=True,
    // mean_only=True (the real checkpoint's own configuration -- see BACKLOG.md), on a small hand-picked
    // fixture (channels=4, hidden_channels=4, kernel_size=3, dilation_rate=2, n_layers=2, T=5).
    constexpr int64_t kChannels = 4, kHidden = 4, kT = 5, kK = 3, kNLayers = 2, kHalf = 2;
    GgmlScratch s;
    ggml_context* ctx = s.ctx.get();

    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kT, kChannels);
    ggml_tensor* pre_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kHalf, kHidden);
    ggml_tensor* pre_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    ggml_tensor* in_w0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kK, kHidden, 2 * kHidden);
    ggml_tensor* in_b0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * kHidden);
    ggml_tensor* rs_w0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kHidden, 2 * kHidden);
    ggml_tensor* rs_b0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * kHidden);
    ggml_tensor* in_w1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kK, kHidden, 2 * kHidden);
    ggml_tensor* in_b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * kHidden);
    ggml_tensor* rs_w1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kHidden, kHidden);
    ggml_tensor* rs_b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    ggml_tensor* post_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kHidden, kHalf);
    ggml_tensor* post_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHalf);
    for (ggml_tensor* t : {x, pre_w, pre_b, in_w0, in_b0, rs_w0, rs_b0, in_w1, in_b1, rs_w1, rs_b1, post_w, post_b}) {
        ggml_set_input(t);
    }

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{ctx, env, nullptr};
    nlohmann::json attrs = {{"kernel_size", kK}, {"dilation_rate", 2}, {"n_layers", kNLayers}};
    ggml_tensor* out = op("RESIDUAL_COUPLING_LAYER_REVERSE")(
        pc, {x, pre_w, pre_b, in_w0, in_b0, rs_w0, rs_b0, in_w1, in_b1, rs_w1, rs_b1, post_w, post_b}, attrs)[0];
    LOOM_CHECK(out->ne[0] == kT);
    LOOM_CHECK(out->ne[1] == kChannels);

    ggml_cgraph* gf = s.expand(out);
    set_f32(x, {0.8701950311660767f, 2.814647912979126f, 0.7322447299957275f, -0.1503330022096634f, 0.6218945980072021f, 2.284700870513916f, 1.5749047994613647f, -0.31051355600357056f, 1.4957236051559448f, 1.0033701658248901f, -1.0747631788253784f, -0.021185562014579773f, -1.6455464363098145f, -1.4646426439285278f, 0.35013866424560547f, -0.1059054508805275f, -0.7404946088790894f, 1.1576905250549316f, -1.7930618524551392f, -0.1147308498620987f});
    set_f32(pre_w, {0.01078505627810955f, -0.09681537747383118f, -0.07493160665035248f, 0.3214437663555145f, -0.13985484838485718f, 0.23920026421546936f, -0.35233110189437866f, -0.4206389784812927f});
    set_f32(pre_b, {0.1003800556063652f, -0.07704495638608932f, 0.13836102187633514f, 0.03989950940012932f});
    set_f32(in_w0, {0.6892627477645874f, 0.15817642211914062f, 0.045414578169584274f, 0.15564732253551483f, -0.17688629031181335f, 0.3002040684223175f, 0.35805609822273254f, -0.0059278844855725765f, 0.17036493122577667f, -0.42576101422309875f, -0.09679030627012253f, 0.04803809896111488f, -0.44506871700286865f, 0.24701225757598877f, -0.3419817388057709f, -0.008950045332312584f, 0.6246023774147034f, 0.035022929310798645f, -0.04446876421570778f, 0.26791608333587646f, -0.2285495549440384f, -0.4616364538669586f, -0.18673650920391083f, -0.10688454657793045f, -0.7368581891059875f, 0.006209224928170443f, -0.037595391273498535f, 0.5058717727661133f, 0.6828716397285461f, -0.28703486919403076f, 0.5354036092758179f, 0.057985566556453705f, 0.020980043336749077f, 0.001869930187240243f, -0.42165905237197876f, 0.03583567589521408f, -0.09936635941267014f, -0.2522205412387848f, 0.08193396031856537f, -0.10759387165307999f, -0.0657113716006279f, -0.0721995085477829f, -0.18752871453762054f, 0.24484477937221527f, -0.17132078111171722f, -0.03583584725856781f, 0.1024860218167305f, -0.08630435168743134f, -0.4236229360103607f, -0.029932113364338875f, 0.022480860352516174f, 0.17709669470787048f, 0.11956692487001419f, -0.19076718389987946f, 0.01406063325703144f, 0.5460425019264221f, -0.6163046956062317f, -0.024198729544878006f, -0.00016060149937402457f, -0.34363192319869995f, 0.6018244028091431f, -0.05311451107263565f, -0.41148945689201355f, -0.2124132215976715f, 0.015229483135044575f, -0.4967232644557953f, -0.13844646513462067f, 0.011423110030591488f, -0.39721453189849854f, 0.1664479821920395f, -0.32452401518821716f, -0.09905383735895157f, 0.014989381656050682f, 0.1356382668018341f, 0.24437099695205688f, -0.04182564839720726f, -0.11030112951993942f, -0.13720637559890747f, -0.21669688820838928f, 0.15778928995132446f, -0.5729297995567322f, 0.31614404916763306f, -0.3141920566558838f, 0.06753461807966232f, 0.20394985377788544f, 0.15866026282310486f, -0.23885783553123474f, -0.09438708424568176f, 0.011070610024034977f, 0.18690773844718933f, 0.2519274055957794f, -0.2645954191684723f, 0.28951355814933777f, 0.26064980030059814f, 0.23027411103248596f, 0.10990618914365768f});
    set_f32(in_b0, {-0.01060530450195074f, -0.02152823843061924f, -0.02236071042716503f, -0.10871756076812744f, -0.03180688992142677f, -0.07416480779647827f, -0.06897430866956711f, -0.14381596446037292f});
    set_f32(in_w1, {-0.2043232023715973f, -0.3929688036441803f, 0.19808079302310944f, -0.423618882894516f, 0.5503743290901184f, 0.13685846328735352f, 0.13889464735984802f, 0.2640746533870697f, 0.032734040170907974f, -0.25289952754974365f, 0.18155090510845184f, 0.39529675245285034f, 0.3213014304637909f, -0.36651453375816345f, 0.30294564366340637f, -0.0735427662730217f, -0.23982985317707062f, -0.03770048916339874f, 0.20277778804302216f, 0.25875046849250793f, 0.5390390753746033f, -0.4523196518421173f, 0.048840250819921494f, -0.1914798468351364f, 0.1611376404762268f, 0.20450478792190552f, -0.4489001929759979f, -0.2947425842285156f, 0.21551814675331116f, 0.1320635974407196f, -0.030735881999135017f, -0.5179604887962341f, -0.6309975385665894f, 0.2633151710033417f, -0.0711565613746643f, 0.4697175920009613f, -0.34328123927116394f, -0.7989959120750427f, -0.0032787593081593513f, 0.07739502191543579f, 0.2434602975845337f, 0.04887678846716881f, -0.20156067609786987f, -0.5078544020652771f, -0.3224779963493347f, 0.2563023865222931f, -0.2488568127155304f, 0.1717686504125595f, -0.3720364570617676f, -0.021201573312282562f, -0.018960095942020416f, 0.17759191989898682f, -0.13366824388504028f, -0.365694135427475f, -0.4518306851387024f, 0.062473416328430176f, -0.8218052983283997f, 0.007764474488794804f, -0.058871086686849594f, -0.23461715877056122f, -0.6422109603881836f, 0.10012178868055344f, 0.08854605257511139f, 0.14794977009296417f, -0.2629183232784271f, -0.532200813293457f, -0.24311278760433197f, -0.44278863072395325f, 0.20010744035243988f, 0.20064117014408112f, 0.1900424361228943f, -0.00967409648001194f, 0.008250097744166851f, 0.1743946075439453f, 0.07049524784088135f, 0.3215011954307556f, -0.15685968101024628f, -0.2163870632648468f, 0.30417513847351074f, -0.08644863963127136f, 0.1393986940383911f, 0.11744412034749985f, -0.40218111872673035f, 0.22789376974105835f, 0.21863770484924316f, -0.020020617172122f, -0.22450374066829681f, 0.42671993374824524f, 0.03318819776177406f, 0.010556439869105816f, -0.16936755180358887f, 0.2747652232646942f, 0.08969460427761078f, 0.05312744528055191f, 0.12434279918670654f, 0.1258580982685089f});
    set_f32(in_b1, {-0.03430599346756935f, 0.06325703114271164f, -0.11305767297744751f, 0.06477969139814377f, 0.16279223561286926f, -0.032473403960466385f, 0.12786836922168732f, -0.05936909839510918f});
    set_f32(rs_w0, {-0.4555489122867584f, 0.014258594252169132f, -0.003094265703111887f, 0.43060410022735596f, 0.07552525401115417f, 0.22224682569503784f, 0.4329383671283722f, -0.06160425394773483f, -0.41966283321380615f, 0.012241151183843613f, -0.33522602915763855f, 0.21088294684886932f, -0.20024296641349792f, 0.03761468827724457f, 0.23700706660747528f, -0.00285415630787611f, 0.015637267380952835f, 0.1020166352391243f, -0.06372488290071487f, 0.4688749611377716f, -0.2721477448940277f, -0.46985557675361633f, -0.01624004729092121f, 0.060865338891744614f, 0.6353916525840759f, -0.05481315031647682f, -0.23305366933345795f, 0.5329280495643616f, -0.1509835124015808f, 0.016972778365015984f, 0.056474167853593826f, 0.26803427934646606f});
    set_f32(rs_b0, {-0.024914976209402084f, -0.04716075584292412f, 0.0043088640086352825f, -0.04292250797152519f, 0.07924213260412216f, 0.018812550231814384f, 0.012507474981248379f, 0.10838896036148071f});
    set_f32(rs_w1, {-0.30632930994033813f, 0.3238396942615509f, 0.34407180547714233f, -0.051987651735544205f, 0.019113607704639435f, -0.3809749484062195f, -0.11167012155056f, 0.6251184940338135f, 0.026761969551444054f, 0.5402256846427917f, -0.6187979578971863f, 0.09665101021528244f, -0.34171122312545776f, 0.37253889441490173f, -0.4657468795776367f, -0.20436039566993713f});
    set_f32(rs_b1, {-0.01785287633538246f, -0.09177599102258682f, -0.13696996867656708f, 0.04527019336819649f});
    set_f32(post_w, {0.05291367694735527f, 0.2777854800224304f, -0.04088832810521126f, 0.31743770837783813f, -0.4917338490486145f, -0.04861591383814812f, 0.14806026220321655f, 0.21400637924671173f});
    set_f32(post_b, {-0.016227176412940025f, 0.07490350306034088f});
    s.compute(gf);

    const std::vector<float> expected = {0.8701950311660767f, 2.814647912979126f, 0.7322447299957275f, -0.1503330022096634f, 0.6218945980072021f, 2.284700870513916f, 1.5749047994613647f, -0.31051355600357056f, 1.4957236051559448f, 1.0033701658248901f, -1.0657532215118408f, 0.0023529157042503357f, -1.655283808708191f, -1.4621962308883667f, 0.391849547624588f, -0.1822173297405243f, -0.8781994581222534f, 0.9377689361572266f, -2.0133631229400635f, -0.21498388051986694f};
    LOOM_CHECK(approx_eq(get_f32(out), expected, 1e-4f));
}

void test_flip_via_get_rows() {
    // VITS's `Flip` (modules.py) reverses the entire channel axis (`torch.flip(x,[1])`). No new
    // primitive is needed: GET_ROWS already selects rows (here, channels, for our [T, C] convention)
    // by an arbitrary I32 index list, so a conversion-time-baked reversed-index constant
    // ([C-1, C-2, ..., 0]) reproduces Flip exactly. Verified against the same x/expected-output pair
    // used for test_residual_coupling_layer_reverse's input fixture (cross-checked against real
    // execution of piper's own modules.Flip).
    constexpr int64_t kChannels = 4, kT = 5;
    GgmlScratch s;
    ggml_context* ctx = s.ctx.get();

    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kT, kChannels);
    ggml_set_input(x);
    ggml_tensor* idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kChannels);
    ggml_set_input(idx);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{ctx, env, nullptr};
    ggml_tensor* out = op("GET_ROWS")(pc, {x, idx}, {})[0];
    LOOM_CHECK(out->ne[0] == kT);
    LOOM_CHECK(out->ne[1] == kChannels);

    ggml_cgraph* gf = s.expand(out);
    set_f32(x, {0.8701950311660767f, 2.814647912979126f, 0.7322447299957275f, -0.1503330022096634f, 0.6218945980072021f, 2.284700870513916f, 1.5749047994613647f, -0.31051355600357056f, 1.4957236051559448f, 1.0033701658248901f, -1.0747631788253784f, -0.021185562014579773f, -1.6455464363098145f, -1.4646426439285278f, 0.35013866424560547f, -0.1059054508805275f, -0.7404946088790894f, 1.1576905250549316f, -1.7930618524551392f, -0.1147308498620987f});
    set_i32(idx, {3, 2, 1, 0});
    s.compute(gf);

    const std::vector<float> expected = {-0.1059054508805275f, -0.7404946088790894f, 1.1576905250549316f, -1.7930618524551392f, -0.1147308498620987f, -1.0747631788253784f, -0.021185562014579773f, -1.6455464363098145f, -1.4646426439285278f, 0.35013866424560547f, 2.284700870513916f, 1.5749047994613647f, -0.31051355600357056f, 1.4957236051559448f, 1.0033701658248901f, 0.8701950311660767f, 2.814647912979126f, 0.7322447299957275f, -0.1503330022096634f, 0.6218945980072021f};
    LOOM_CHECK(approx_eq(get_f32(out), expected, 1e-5f));
}

void test_elementwise_affine_reverse() {
    // Cross-checked against real execution of piper's own modules.ElementwiseAffine, reverse=True.
    constexpr int64_t kChannels = 2, kT = 5;
    GgmlScratch s;
    ggml_context* ctx = s.ctx.get();
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kT, kChannels);
    ggml_tensor* m = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kChannels);
    ggml_tensor* logs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kChannels);
    for (ggml_tensor* t : {x, m, logs}) ggml_set_input(t);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{ctx, env, nullptr};
    ggml_tensor* out = op("ELEMENTWISE_AFFINE_REVERSE")(pc, {x, m, logs}, {})[0];
    LOOM_CHECK(out->ne[0] == kT);
    LOOM_CHECK(out->ne[1] == kChannels);

    ggml_cgraph* gf = s.expand(out);
    set_f32(x, {-0.4519059658050537f, -0.16613022983074188f, -1.522768497467041f, 0.38168391585350037f, -1.0276086330413818f, -0.563052773475647f, -0.8922905325889587f, -0.05825017765164375f, -0.19550958275794983f, -0.9656359553337097f});
    set_f32(m, {0.1984056532382965f, 0.08007723838090897f});
    set_f32(logs, {0.012335452251136303f, 0.1242634654045105f});
    s.compute(gf);

    const std::vector<float> expected = {-0.6423389911651611f, -0.3600667715072632f, -1.7000731229782104f, 0.18103133141994476f, -1.2109837532043457f, -0.5679783821105957f, -0.8587437868118286f, -0.12216345965862274f, -0.24338370561599731f, -0.9235185980796814f};
    LOOM_CHECK(approx_eq(get_f32(out), expected, 1e-4f));
}

void test_dds_conv() {
    // Cross-checked against real execution of piper's own modules.DDSConv on a small hand-picked
    // fixture (channels=4, kernel_size=3, n_layers=2, T=5).
    constexpr int64_t kChannels = 4, kT = 5, kK = 3, kNLayers = 2;
    GgmlScratch s;
    ggml_context* ctx = s.ctx.get();

    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kT, kChannels);
    ggml_tensor* sep_w0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kK, 1, kChannels);
    ggml_tensor* sep_b0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kChannels);
    ggml_tensor* ln1_g0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kChannels);
    ggml_tensor* ln1_b0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kChannels);
    ggml_tensor* oo_w0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kChannels, kChannels);
    ggml_tensor* oo_b0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kChannels);
    ggml_tensor* ln2_g0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kChannels);
    ggml_tensor* ln2_b0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kChannels);
    ggml_tensor* sep_w1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kK, 1, kChannels);
    ggml_tensor* sep_b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kChannels);
    ggml_tensor* ln1_g1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kChannels);
    ggml_tensor* ln1_b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kChannels);
    ggml_tensor* oo_w1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kChannels, kChannels);
    ggml_tensor* oo_b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kChannels);
    ggml_tensor* ln2_g1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kChannels);
    ggml_tensor* ln2_b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kChannels);
    std::vector<ggml_tensor*> all = {x, sep_w0, sep_b0, ln1_g0, ln1_b0, oo_w0, oo_b0, ln2_g0, ln2_b0,
                                      sep_w1, sep_b1, ln1_g1, ln1_b1, oo_w1, oo_b1, ln2_g1, ln2_b1};
    for (ggml_tensor* t : all) ggml_set_input(t);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{ctx, env, nullptr};
    nlohmann::json attrs = {{"kernel_size", kK}, {"n_layers", kNLayers}, {"eps", 1e-5}};
    ggml_tensor* out = op("DDS_CONV")(pc, all, attrs)[0];
    LOOM_CHECK(out->ne[0] == kT);
    LOOM_CHECK(out->ne[1] == kChannels);

    ggml_cgraph* gf = s.expand(out);
    set_f32(x, {-0.19700005650520325f, -0.033396217972040176f, 0.7192915081977844f, 1.064414620399475f, -0.7530738711357117f, -0.43190088868141174f, 0.6693012118339539f, 0.6505143642425537f, -0.869227409362793f, 0.886859655380249f, -1.2211220264434814f, -0.6752681732177734f, 0.33310893177986145f, 1.1079213619232178f, 0.18567782640457153f, -0.27635684609413147f, -0.593854546546936f, -0.3060556948184967f, 0.677129328250885f, 0.6808614134788513f});
    set_f32(sep_w0, {-0.11904474347829819f, -0.19711796939373016f, -0.4928257465362549f, 0.2940875291824341f, -0.012644119560718536f, -0.2461727410554886f, 0.09398985654115677f, -0.3405499756336212f, 0.1132001131772995f, -0.08472587913274765f, -0.7700210213661194f, -0.4290982484817505f});
    set_f32(sep_b0, {0.05009211227297783f, 0.0543767474591732f, -0.04057423770427704f, 0.11340516060590744f});
    set_f32(ln1_g0, {0.9056583642959595f, 1.066835880279541f, 1.1162776947021484f, 0.9677112102508545f});
    set_f32(ln1_b0, {0.18781517446041107f, -0.05666264519095421f, 0.04016321897506714f, -0.011528476141393185f});
    set_f32(oo_w0, {-0.18721090257167816f, -0.29318884015083313f, 0.2624528706073761f, 0.2961844205856323f, 0.07514028251171112f, -0.23790977895259857f, 0.15693031251430511f, 0.36708173155784607f, -0.12103846669197083f, -0.28774356842041016f, -0.001562311197631061f, -0.02365894243121147f, -0.11673584580421448f, -0.023880107328295708f, 0.22814343869686127f, -0.30075526237487793f});
    set_f32(oo_b0, {0.12707822024822235f, -0.00020225078333169222f, -0.10951849073171616f, 0.06016450747847557f});
    set_f32(ln2_g0, {1.0729644298553467f, 0.9868826270103455f, 0.9363210797309875f, 1.104292631149292f});
    set_f32(ln2_b0, {0.049028560519218445f, 0.10318265110254288f, -0.05988920480012894f, 0.16015303134918213f});
    set_f32(sep_w1, {-0.33346158266067505f, 0.10502027720212936f, -0.23108184337615967f, -0.04417986795306206f, 0.1881536990404129f, 0.3280358910560608f, 0.02817094512283802f, 0.37141990661621094f, -0.40376827120780945f, 0.15356919169425964f, -0.20798330008983612f, -0.05002805218100548f});
    set_f32(sep_b1, {-0.09998821467161179f, -0.164756640791893f, 0.08098284900188446f, 0.005542439874261618f});
    set_f32(ln1_g1, {1.0317047834396362f, 1.0562853813171387f, 1.0866177082061768f, 0.9647220969200134f});
    set_f32(ln1_b1, {0.034822966903448105f, 0.11370787769556046f, -0.03338823467493057f, -0.14724226295948029f});
    set_f32(oo_w1, {-0.03285498544573784f, 0.09377072006464005f, 0.4511384963989258f, 0.15113124251365662f, -0.17055478692054749f, 0.2512788772583008f, 0.5350981950759888f, -0.058627400547266006f, -0.34303995966911316f, -0.19536913931369781f, -0.030949190258979797f, 0.2081093192100525f, -0.1623840630054474f, 0.2685522139072418f, -0.2647521197795868f, 0.15954338014125824f});
    set_f32(oo_b1, {-0.1286470741033554f, 0.0823109969496727f, -0.06100623682141304f, -0.12959890067577362f});
    set_f32(ln2_g1, {0.8926532864570618f, 0.8782684206962585f, 1.0647212266921997f, 0.9958832859992981f});
    set_f32(ln2_b1, {-0.017749307677149773f, -0.05000392720103264f, 0.08672749251127243f, -0.027319222688674927f});
    s.compute(gf);

    const std::vector<float> expected = {0.7591858506202698f, 0.9001830220222473f, 1.4790091514587402f, 1.2676560878753662f, -0.20570723712444305f, 1.1412701606750488f, 2.2206478118896484f, 1.5578324794769287f, 0.6611259579658508f, 2.7047805786132812f, -1.509482979774475f, -0.9500081539154053f, 0.07445654273033142f, 0.8279659152030945f, -0.04820296913385391f, 0.13774490356445312f, -0.18611657619476318f, 0.9312532544136047f, 1.9432650804519653f, 1.0476534366607666f};
    LOOM_CHECK(approx_eq(get_f32(out), expected, 1e-4f));
}

void test_conv_flow_reverse() {
    // Cross-checked against real execution of piper's own modules.ConvFlow, reverse=True, WITH a
    // nonzero `g` conditioning input (the only way ConvFlow is ever actually invoked in this
    // checkpoint's real topology -- StochasticDurationPredictor.forward's reverse branch always calls
    // each flow with `g=x`, its own conditioning features), on a small hand-picked fixture
    // (in_channels=2, filter_channels=4, kernel_size=3, n_layers=2, num_bins=4, tail_bound=2.0, T=5).
    // Exercises both the in-domain spline branch and the outside-tail-bound identity branch (x1[2]=2.9382
    // > tail_bound=2.0 in this fixture).
    constexpr int64_t kFilter = 4, kT = 5, kK = 3, kNLayers = 2, kNumBins = 4;
    GgmlScratch s;
    ggml_context* ctx = s.ctx.get();

    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kT, 2);
    ggml_tensor* pre_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, kFilter);
    ggml_tensor* pre_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* sep_w0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kK, 1, kFilter);
    ggml_tensor* sep_b0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* ln1_g0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* ln1_b0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* oo_w0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kFilter, kFilter);
    ggml_tensor* oo_b0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* ln2_g0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* ln2_b0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* sep_w1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kK, 1, kFilter);
    ggml_tensor* sep_b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* ln1_g1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* ln1_b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* oo_w1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kFilter, kFilter);
    ggml_tensor* oo_b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* ln2_g1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* ln2_b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* proj_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kFilter, 3 * kNumBins - 1);
    ggml_tensor* proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * kNumBins - 1);
    ggml_tensor* boundary_deriv_const = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kNumBins + 1);
    ggml_tensor* eps_bump = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kNumBins);
    ggml_tensor* g = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kT, kFilter);
    std::vector<ggml_tensor*> all = {x, pre_w, pre_b, sep_w0, sep_b0, ln1_g0, ln1_b0, oo_w0, oo_b0, ln2_g0, ln2_b0,
                                      sep_w1, sep_b1, ln1_g1, ln1_b1, oo_w1, oo_b1, ln2_g1, ln2_b1,
                                      proj_w, proj_b, boundary_deriv_const, eps_bump, g};
    for (ggml_tensor* t : all) ggml_set_input(t);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{ctx, env, nullptr};
    nlohmann::json attrs = {{"kernel_size", kK}, {"n_layers", kNLayers}, {"num_bins", kNumBins},
                             {"tail_bound", 2.0}, {"ln_eps", 1e-5}};
    ggml_tensor* out = op("CONV_FLOW_REVERSE")(pc, all, attrs)[0];
    LOOM_CHECK(out->ne[0] == kT);
    LOOM_CHECK(out->ne[1] == 2);

    ggml_cgraph* gf = s.expand(out);
    set_f32(x, {-0.7496130466461182f, -1.0645033121109009f, 1.0059064626693726f, -1.0691108703613281f, 0.31916582584381104f, 0.5580326914787292f, -0.6536681056022644f, 2.9381799697875977f, 0.7845586538314819f, -0.18172304332256317f});
    set_f32(pre_w, {-0.27702388167381287f, 0.18881076574325562f, 0.14715667068958282f, 0.36787569522857666f});
    set_f32(pre_b, {-0.20087504386901855f, -0.09414206445217133f, 0.04695797339081764f, -0.12195540964603424f});
    set_f32(sep_w0, {0.17967255413532257f, 0.539918839931488f, 0.3229123651981354f, 0.32524365186691284f, -0.0472554974257946f, 0.2985955774784088f, -0.033040016889572144f, 0.2704795002937317f, 0.5163682699203491f, 0.22764626145362854f, -0.29296061396598816f, 0.5757042765617371f});
    set_f32(sep_b0, {-0.07275336235761642f, -0.22896018624305725f, -0.10281645506620407f, -0.06007272005081177f});
    set_f32(ln1_g0, {1.0351353883743286f, 1.0306215286254883f, 0.8916435837745667f, 0.8038790822029114f});
    set_f32(ln1_b0, {-0.07468719780445099f, -0.03221055492758751f, -0.051207978278398514f, -0.0025955636519938707f});
    set_f32(oo_w0, {0.28174903988838196f, 0.10685445368289948f, -0.19684995710849762f, 0.20249903202056885f, 0.38729187846183777f, 0.08630422502756119f, 0.33939701318740845f, -0.000576831167563796f, 0.2420041561126709f, -0.23350341618061066f, 0.005302452482283115f, 0.3560906946659088f, -0.1787063628435135f, -0.4721554219722748f, 0.2702849805355072f, 0.31497785449028015f});
    set_f32(oo_b0, {-0.1294471174478531f, -0.15406039357185364f, -0.01920928619801998f, -0.09268879145383835f});
    set_f32(ln2_g0, {1.0012000799179077f, 0.8564978241920471f, 1.0300531387329102f, 1.0386089086532593f});
    set_f32(ln2_b0, {-0.0554354302585125f, -0.1767720729112625f, 0.003508932190015912f, 0.12522640824317932f});
    set_f32(sep_w1, {0.007611261680722237f, -0.019893044605851173f, -0.4502426087856293f, 0.0019290128257125616f, 0.39373520016670227f, 0.15663942694664001f, -0.12154239416122437f, 0.032797202467918396f, 0.5200806856155396f, 0.13849253952503204f, 0.25821393728256226f, 0.11983378231525421f});
    set_f32(sep_b1, {-0.02651146426796913f, 9.199998021358624e-05f, -0.05645310878753662f, -0.14245443046092987f});
    set_f32(ln1_g1, {1.0217094421386719f, 0.9297191500663757f, 1.1620115041732788f, 0.7937030792236328f});
    set_f32(ln1_b1, {0.14200641214847565f, -0.06444717943668365f, 0.0038804709911346436f, -0.09435078501701355f});
    set_f32(oo_w1, {0.1024622693657875f, -0.10497551411390305f, -0.07031884789466858f, -0.1635577380657196f, 0.08253643661737442f, -0.3550170660018921f, -0.2460491955280304f, 0.09170494228601456f, 0.220586359500885f, 0.5586596131324768f, 0.1332489550113678f, 0.14688503742218018f, 0.37541407346725464f, -0.5314837694168091f, -0.0062791733071208f, 0.1753685176372528f});
    set_f32(oo_b1, {-0.0511038675904274f, -0.2337372601032257f, -0.022863944992423058f, 0.018055720254778862f});
    set_f32(ln2_g1, {0.8322675228118896f, 0.8924780488014221f, 1.0555880069732666f, 0.8745896816253662f});
    set_f32(ln2_b1, {-0.0698826014995575f, -0.03089888207614422f, 0.05011402443051338f, 0.1268770843744278f});
    set_f32(proj_w, {0.06535575538873672f, 0.10294780880212784f, -0.04674355313181877f, 0.12099821865558624f, -0.09292705357074738f, 0.1596456617116928f, -0.003517786506563425f, 0.019276363775134087f, -0.011456212028861046f, 0.04030787944793701f, 0.0783390998840332f, 0.16240975260734558f, -0.19683785736560822f, 0.09298764914274216f, 0.1598103642463684f, 0.04261689633131027f, 0.06325045973062515f, 0.04409002885222435f, -0.20394562184810638f, 0.10628663748502731f, 0.07760129868984222f, 0.008345716632902622f, 0.17073935270309448f, -0.02075815759599209f, 0.01671871542930603f, -0.06524643301963806f, 0.2319086641073227f, -0.031556662172079086f, 0.03312002122402191f, 0.1035284772515297f, -0.016316445544362068f, 0.025800129398703575f, 0.08273085951805115f, -0.024229655042290688f, -0.01800033263862133f, 0.09583788365125656f, 0.07199329137802124f, 0.07556533068418503f, 0.01770401932299137f, 0.14330007135868073f, 0.007244384381920099f, -0.07704640179872513f, 0.045658718794584274f, -0.1315045952796936f});
    set_f32(proj_b, {0.0595579519867897f, -0.13605843484401703f, -0.07178483158349991f, 0.10850757360458374f, 0.0640895664691925f, -0.2292538434267044f, 0.02502293325960636f, 0.2299896776676178f, 0.11097162216901779f, 0.04603404924273491f, -0.08856828510761261f});
    const float boundary_const = 0.5397424172369522f;
    set_f32(boundary_deriv_const, {boundary_const, 0.0f, 0.0f, 0.0f, boundary_const});
    set_f32(eps_bump, {0.0f, 0.0f, 0.0f, 1e-6f});
    set_f32(g, {0.3550335466861725f, 0.007120965048670769f, 0.3335949778556824f, -0.2839277982711792f, -0.11796847730875015f, 0.28043296933174133f, 0.15122726559638977f, 0.19960062205791473f, -0.1804848164319992f, -0.2797794044017792f, 0.022235441952943802f, 0.06338673830032349f, -0.05461089685559273f, 0.07982761412858963f, -0.3805844783782959f, -0.018001293763518333f, 0.3741171956062317f, -0.14020471274852753f, 0.46497073769569397f, 0.13505025207996368f});
    s.compute(gf);

    const std::vector<float> expected = {-0.7496130466461182f, -1.0645033121109009f, 1.0059064626693726f, -1.0691108703613281f, 0.31916582584381104f, 0.5190484523773193f, -0.608812689781189f, 2.9381799697875977f, 0.6768510937690735f, -0.25991201400756836f};
    const auto result = get_f32(out);
    if (!approx_eq(result, expected, 1e-4f)) {
        std::fprintf(stderr, "CONV_FLOW_REVERSE got:");
        for (float v : result) std::fprintf(stderr, " %f", static_cast<double>(v));
        std::fprintf(stderr, "\nexpected:");
        for (float v : expected) std::fprintf(stderr, " %f", static_cast<double>(v));
        std::fprintf(stderr, "\n");
    }
    LOOM_CHECK(approx_eq(result, expected, 1e-4f));
}

void test_sdp_reverse_assembly() {
    // StochasticDurationPredictor.forward, reverse=True (models.py:108-117), assembled by chaining
    // already-verified primitives in the exact real order -- no new primitive needed here, since SDP
    // itself has no repeated-substructure math beyond what CONV_1D/DDS_CONV/CONV_FLOW_REVERSE/
    // ELEMENTWISE_AFFINE_REVERSE/GET_ROWS(as Flip) already cover; this test verifies the WIRING/ORDERING
    // (which flows survive the real "remove a useless vflow" filter, what gets fed as `g`, the final
    // z0=logw split), cross-checked against a hand-replica of the real control flow using piper's own
    // sub-module classes directly (see BACKLOG.md) with n_flows=2 -- the minimum that leaves exactly one
    // ConvFlow (`CF1`) surviving; `CF0`'s weights are provably never touched at inference, matching the
    // real code's own comment.
    //
    // Real flows list (n_flows=2): [EA, CF0, Flip0, CF1, Flip1]. Reversed: [Flip1, CF1, Flip0, CF0, EA].
    // flows_final = reversed[:-2] + [reversed[-1]] = [Flip1, CF1, Flip0, EA] -- applied to host-provided
    // noise `z` in that exact order, conditioned throughout on `g = pre->convs->proj(enc)`.
    constexpr int64_t kInCh = 2, kFilter = 2, kK = 3, kT = 2, kNumBins = 3;
    GgmlScratch s;
    ggml_context* ctx = s.ctx.get();

    ggml_tensor* enc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kT, kInCh);
    ggml_tensor* pre_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kInCh, kFilter);
    ggml_tensor* pre_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* c_sep_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kK, 1, kFilter);
    ggml_tensor* c_sep_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* c_ln1_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* c_ln1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* c_oo_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kFilter, kFilter);
    ggml_tensor* c_oo_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* c_ln2_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* c_ln2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* proj_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kFilter, kFilter);
    ggml_tensor* proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* z_noise = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kT, 2);
    ggml_tensor* cf_pre_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, kFilter);
    ggml_tensor* cf_pre_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* cf_sep_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kK, 1, kFilter);
    ggml_tensor* cf_sep_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* cf_ln1_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* cf_ln1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* cf_oo_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kFilter, kFilter);
    ggml_tensor* cf_oo_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* cf_ln2_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* cf_ln2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* cf_proj_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kFilter, 3 * kNumBins - 1);
    ggml_tensor* cf_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * kNumBins - 1);
    ggml_tensor* boundary_deriv_const = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kNumBins + 1);
    ggml_tensor* eps_bump = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kNumBins);
    ggml_tensor* ea_m = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_tensor* ea_logs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);

    std::vector<ggml_tensor*> all = {enc, pre_w, pre_b, c_sep_w, c_sep_b, c_ln1_g, c_ln1_b, c_oo_w, c_oo_b,
                                      c_ln2_g, c_ln2_b, proj_w, proj_b, z_noise, cf_pre_w, cf_pre_b, cf_sep_w,
                                      cf_sep_b, cf_ln1_g, cf_ln1_b, cf_oo_w, cf_oo_b, cf_ln2_g, cf_ln2_b,
                                      cf_proj_w, cf_proj_b, boundary_deriv_const, eps_bump, ea_m, ea_logs};
    for (ggml_tensor* t : all) ggml_set_input(t);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{ctx, env, nullptr};

    // g_cond = proj(convs(pre(enc)))
    ggml_tensor* pre_out = op("CONV_1D")(pc, {pre_w, ggml_reshape_3d(ctx, enc, kT, kInCh, 1)}, {{"s0", 1}, {"p0", 0}, {"d0", 1}})[0];
    pre_out = ggml_add(ctx, pre_out, ggml_reshape_3d(ctx, pre_b, 1, kFilter, 1));
    ggml_tensor* convs_in = ggml_reshape_2d(ctx, pre_out, kT, kFilter);
    nlohmann::json dds_attrs = {{"kernel_size", kK}, {"n_layers", 1}, {"eps", 1e-5}};
    ggml_tensor* convs_out = op("DDS_CONV")(pc, {convs_in, c_sep_w, c_sep_b, c_ln1_g, c_ln1_b, c_oo_w, c_oo_b, c_ln2_g, c_ln2_b}, dds_attrs)[0];
    ggml_tensor* proj_out = op("CONV_1D")(pc, {proj_w, ggml_reshape_3d(ctx, convs_out, kT, kFilter, 1)}, {{"s0", 1}, {"p0", 0}, {"d0", 1}})[0];
    proj_out = ggml_add(ctx, proj_out, ggml_reshape_3d(ctx, proj_b, 1, kFilter, 1));
    ggml_tensor* g_cond = ggml_reshape_2d(ctx, proj_out, kT, kFilter);

    // z = Flip(z_noise)
    ggml_tensor* flip_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    ggml_set_input(flip_idx);
    ggml_tensor* z = op("GET_ROWS")(pc, {z_noise, flip_idx}, {})[0];
    // z = CF1_reverse(z, g=g_cond)
    nlohmann::json cf_attrs = {{"kernel_size", kK}, {"n_layers", 1}, {"num_bins", kNumBins}, {"tail_bound", 2.0}, {"ln_eps", 1e-5}};
    z = op("CONV_FLOW_REVERSE")(pc, {z, cf_pre_w, cf_pre_b, cf_sep_w, cf_sep_b, cf_ln1_g, cf_ln1_b, cf_oo_w, cf_oo_b,
                                      cf_ln2_g, cf_ln2_b, cf_proj_w, cf_proj_b, boundary_deriv_const, eps_bump, g_cond},
                                 cf_attrs)[0];
    // z = Flip(z)
    z = op("GET_ROWS")(pc, {z, flip_idx}, {})[0];
    // z = EA_reverse(z)
    z = op("ELEMENTWISE_AFFINE_REVERSE")(pc, {z, ea_m, ea_logs}, {})[0];
    // logw = z0 (first channel)
    ggml_tensor* logw = ggml_cont(ctx, ggml_view_2d(ctx, z, kT, 1, z->nb[1], 0));
    logw = ggml_reshape_1d(ctx, logw, kT);

    ggml_cgraph* gf = s.expand(logw);
    set_f32(enc, {0.40568798780441284f, 1.618118405342102f, 0.39315488934516907f, -0.2147795557975769f});
    set_f32(pre_w, {-0.30470243096351624f, -0.2666245698928833f, 0.04493391513824463f, -0.06266818195581436f});
    set_f32(pre_b, {-0.0387020967900753f, 0.09912377595901489f});
    set_f32(c_sep_w, {0.1403709203004837f, -0.06147957593202591f, -0.2222721427679062f, 0.10855189710855484f, 0.5759698152542114f, -0.06761626899242401f});
    set_f32(c_sep_b, {-0.034169748425483704f, 0.03040127456188202f});
    set_f32(c_ln1_g, {0.6405531167984009f, 1.0019150972366333f});
    set_f32(c_ln1_b, {0.010517301969230175f, 0.09603401273488998f});
    set_f32(c_oo_w, {-0.20670409500598907f, -0.33801552653312683f, -0.08572651445865631f, -0.3280532956123352f});
    set_f32(c_oo_b, {0.11351022869348526f, 0.07592452317476273f});
    set_f32(c_ln2_g, {0.9432819485664368f, 0.9429352283477783f});
    set_f32(c_ln2_b, {0.1598038524389267f, 0.011148621328175068f});
    set_f32(proj_w, {-0.011759010143578053f, 0.4233461916446686f, -0.19668330252170563f, 0.25728172063827515f});
    set_f32(proj_b, {-0.16270242631435394f, -0.13951388001441956f});
    set_f32(z_noise, {1.012095332145691f, -0.25424548983573914f, -0.34233081340789795f, -0.40055227279663086f});
    set_f32(cf_pre_w, {-0.04004606232047081f, 0.102446548640728f});
    set_f32(cf_pre_b, {-0.007157138083130121f, -0.009089037775993347f});
    set_f32(cf_sep_w, {-0.39890769124031067f, -0.16277480125427246f, 0.16412954032421112f, 0.19291912019252777f, -0.237144336104393f, -0.2717511057853699f});
    set_f32(cf_sep_b, {-0.026072561740875244f, -0.05465104058384895f});
    set_f32(cf_ln1_g, {1.1450257301330566f, 0.9306433796882629f});
    set_f32(cf_ln1_b, {0.099666528403759f, 0.06130759045481682f});
    set_f32(cf_oo_w, {0.6352202296257019f, -0.513540506362915f, 0.04954032599925995f, 0.47456029057502747f});
    set_f32(cf_oo_b, {0.044846098870038986f, 0.0033029892947524786f});
    set_f32(cf_ln2_g, {1.0776387453079224f, 0.9697084426879883f});
    set_f32(cf_ln2_b, {-0.12753024697303772f, -0.047575175762176514f});
    set_f32(cf_proj_w, {-0.1175333634018898f, 0.03580570966005325f, 0.047876790165901184f, 0.13537000119686127f, -0.01593310758471489f, -0.04249436780810356f, 0.09442310035228729f, -0.018493453040719032f, 0.018516134470701218f, 0.1068691834807396f, 0.13065344095230103f, 0.0459834523499012f, 0.026177797466516495f, -0.07599348574876785f, -0.2046138495206833f, -0.15294533967971802f});
    set_f32(cf_proj_b, {-0.09350629895925522f, 0.02138027921319008f, -0.12842117249965668f, -0.06916777044534683f, -0.05359472706913948f, 0.033552203327417374f, 0.02469366230070591f, 0.0032430763822048903f});
    const float boundary_const = 0.5397424172369522f;
    set_f32(boundary_deriv_const, {boundary_const, 0.0f, 0.0f, boundary_const});
    set_f32(eps_bump, {0.0f, 0.0f, 1e-6f});
    set_f32(ea_m, {-0.047744836658239365f, -0.10099808126688004f});
    set_f32(ea_logs, {-0.24751631915569305f, -0.09316029399633408f});
    set_i32(flip_idx, {1, 0});
    s.compute(gf);

    const std::vector<float> expected = {1.6295220851898193f, -0.11275245994329453f};
    const auto result = get_f32(logw);
    if (!approx_eq(result, expected, 1e-4f)) {
        std::fprintf(stderr, "SDP_REVERSE got: %f %f, expected: %f %f\n",
                     static_cast<double>(result[0]), static_cast<double>(result[1]),
                     static_cast<double>(expected[0]), static_cast<double>(expected[1]));
    }
    LOOM_CHECK(approx_eq(result, expected, 1e-4f));
}

void test_hifigan_generator() {
    // VITS's HiFi-GAN `Generator` (models.py), confirmed via the real checkpoint's own state-dict shapes
    // (`model_g.dec.resblocks.*.convs.{0,1}` only, no `convs2` -- ResBlock2/"low_quality" config, not
    // ResBlock1) to use `resblock="2"`. No new primitive needed -- entirely CONV_1D/CONV_TRANSPOSE_1D/
    // LEAKY_RELU/TANH/ADD, all already verified individually; this test verifies the WIRING (upsample
    // stages, resblock kernel/dilation fan-out-then-average, the two distinct LeakyReLU slopes -- 0.1
    // inside resblocks/after each upsample, PyTorch's *default* 0.01 only once at the very end before
    // conv_post) and one real primitive gap: `ggml_conv_transpose_1d` only supports padding=0 (asserted
    // in ggml's own source), unlike PyTorch's `ConvTranspose1d(..., padding=p)`. Fixed not by a new
    // primitive but by computing the padding=0 ("full") output and cropping `p` samples off each end --
    // verified as an exact identity (not an approximation) against real PyTorch execution comparing
    // `ConvTranspose1d(padding=p)` to `ConvTranspose1d(padding=0)` sliced by `[:, :, p:-p]` before relying
    // on it here (see BACKLOG.md).
    //
    // Small-scale (2 upsample stages instead of the real 3, 2 resblock kernel sizes instead of 3):
    // initial_channel=4, upsample_initial_channel=8, upsample_rates=(2,2), upsample_kernel_sizes=(4,4),
    // resblock_kernel_sizes=(3,5), resblock_dilation_sizes=((1,2),(2,6)), T=4 -- cross-checked against
    // real execution of piper's own models.Generator.
    constexpr int64_t kT = 4;
    GgmlScratch s;
    ggml_context* ctx = s.ctx.get();
    loom::SymbolEnv env;
    loom::PrimitiveContext pc{ctx, env, nullptr};
    auto leaky = [&](ggml_tensor* t, float slope) { return op("LEAKY_RELU")(pc, {t}, {{"slope", slope}})[0]; };
    auto conv_bias = [&](ggml_tensor* w, ggml_tensor* b, ggml_tensor* data3d, int pad, int dil) {
        ggml_tensor* o = op("CONV_1D")(pc, {w, data3d}, {{"s0", 1}, {"p0", pad}, {"d0", dil}})[0];
        return ggml_add(ctx, o, ggml_reshape_3d(ctx, b, 1, b->ne[0], 1));
    };
    auto crop = [&](ggml_tensor* full2d, int64_t pad, int64_t oc) {
        int64_t new_len = full2d->ne[0] - 2 * pad;
        return ggml_cont(ctx, ggml_view_2d(ctx, full2d, new_len, oc, full2d->nb[1], pad * static_cast<int64_t>(sizeof(float))));
    };
    auto resblock2 = [&](ggml_tensor* x2d, int64_t ch, ggml_tensor* w0, ggml_tensor* b0, int d0,
                          ggml_tensor* w1, ggml_tensor* b1, int d1, int k) {
        ggml_tensor* x = x2d;
        {
            ggml_tensor* xt = leaky(x, 0.1f);
            xt = conv_bias(w0, b0, ggml_reshape_3d(ctx, xt, xt->ne[0], ch, 1), (k * d0 - d0) / 2, d0);
            x = ggml_add(ctx, ggml_reshape_2d(ctx, xt, xt->ne[0], ch), x);
        }
        {
            ggml_tensor* xt = leaky(x, 0.1f);
            xt = conv_bias(w1, b1, ggml_reshape_3d(ctx, xt, xt->ne[0], ch, 1), (k * d1 - d1) / 2, d1);
            x = ggml_add(ctx, ggml_reshape_2d(ctx, xt, xt->ne[0], ch), x);
        }
        return x;
    };

    ggml_tensor* x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kT, 4);
    ggml_tensor* conv_pre_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 4, 8);
    ggml_tensor* conv_pre_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor* up0_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 8);
    ggml_tensor* up0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* up1_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 2, 4);
    ggml_tensor* up1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_tensor* rb0_c0_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 4, 4);
    ggml_tensor* rb0_c0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* rb0_c1_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 4, 4);
    ggml_tensor* rb0_c1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* rb1_c0_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5, 4, 4);
    ggml_tensor* rb1_c0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* rb1_c1_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5, 4, 4);
    ggml_tensor* rb1_c1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* rb2_c0_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 2, 2);
    ggml_tensor* rb2_c0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_tensor* rb2_c1_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 2, 2);
    ggml_tensor* rb2_c1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_tensor* rb3_c0_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5, 2, 2);
    ggml_tensor* rb3_c0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_tensor* rb3_c1_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5, 2, 2);
    ggml_tensor* rb3_c1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_tensor* conv_post_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 2, 1);

    std::vector<ggml_tensor*> all = {x_in, conv_pre_w, conv_pre_b, up0_w, up0_b, up1_w, up1_b,
                                      rb0_c0_w, rb0_c0_b, rb0_c1_w, rb0_c1_b, rb1_c0_w, rb1_c0_b, rb1_c1_w, rb1_c1_b,
                                      rb2_c0_w, rb2_c0_b, rb2_c1_w, rb2_c1_b, rb3_c0_w, rb3_c0_b, rb3_c1_w, rb3_c1_b,
                                      conv_post_w};
    for (ggml_tensor* t : all) ggml_set_input(t);

    // conv_pre: K=7, pad=3 ("same").
    ggml_tensor* x = conv_bias(conv_pre_w, conv_pre_b, ggml_reshape_3d(ctx, x_in, kT, 4, 1), 3, 1);
    x = ggml_reshape_2d(ctx, x, kT, 8); // [T=4, C=8]

    // Stage 0: upsample_rates[0]=2, upsample_kernel_sizes[0]=4, pad=(4-2)/2=1.
    x = leaky(x, 0.1f);
    ggml_tensor* up0_full = op("CONV_TRANSPOSE_1D")(pc, {up0_w, x}, {{"s0", 2}})[0]; // [(4-1)*2+4=10, 4]
    up0_full = ggml_add(ctx, up0_full, ggml_reshape_2d(ctx, up0_b, 1, 4));
    x = crop(up0_full, 1, 4); // [8, 4]
    ggml_tensor* rb0 = resblock2(x, 4, rb0_c0_w, rb0_c0_b, 1, rb0_c1_w, rb0_c1_b, 2, 3);
    ggml_tensor* rb1 = resblock2(x, 4, rb1_c0_w, rb1_c0_b, 2, rb1_c1_w, rb1_c1_b, 6, 5);
    x = ggml_scale(ctx, ggml_add(ctx, rb0, rb1), 0.5f);

    // Stage 1: upsample_rates[1]=2, upsample_kernel_sizes[1]=4, pad=1.
    x = leaky(x, 0.1f);
    ggml_tensor* up1_full = op("CONV_TRANSPOSE_1D")(pc, {up1_w, x}, {{"s0", 2}})[0]; // [(8-1)*2+4=18, 2]
    up1_full = ggml_add(ctx, up1_full, ggml_reshape_2d(ctx, up1_b, 1, 2));
    x = crop(up1_full, 1, 2); // [16, 2]
    ggml_tensor* rb2 = resblock2(x, 2, rb2_c0_w, rb2_c0_b, 1, rb2_c1_w, rb2_c1_b, 2, 3);
    ggml_tensor* rb3 = resblock2(x, 2, rb3_c0_w, rb3_c0_b, 2, rb3_c1_w, rb3_c1_b, 6, 5);
    x = ggml_scale(ctx, ggml_add(ctx, rb2, rb3), 0.5f);

    // Final: LeakyReLU with PyTorch's *default* slope (0.01, not 0.1) -> conv_post (K=7, pad=3, no bias) -> tanh.
    x = leaky(x, 0.01f);
    ggml_tensor* out = op("CONV_1D")(pc, {conv_post_w, ggml_reshape_3d(ctx, x, x->ne[0], 2, 1)}, {{"s0", 1}, {"p0", 3}, {"d0", 1}})[0];
    out = op("TANH")(pc, {out}, {})[0];
    out = ggml_reshape_1d(ctx, out, out->ne[0]);
    LOOM_CHECK(out->ne[0] == 16);

    ggml_cgraph* gf = s.expand(out);
    set_f32(x_in, {0.10469477623701096f, 0.48543310165405273f, -1.0852758884429932f, 1.4142487049102783f, -0.5493184328079224f, -0.8505457639694214f, -1.673621654510498f, 1.0049158334732056f, 0.5154784321784973f, 0.4688435196876526f, 0.06790226697921753f, -0.6663368940353394f, -1.694682240486145f, -1.8690822124481201f, -1.2805936336517334f, 0.4549793303012848f});
    set_f32(conv_pre_w, {0.32425636053085327f, -0.12275521457195282f, 0.31737571954727173f, -0.15833614766597748f, 0.16193881630897522f, -0.5450014472007751f, 0.5243209004402161f, -0.0628548413515091f, -0.10780873149633408f, -0.44440820813179016f, 0.046702589839696884f, 0.05957435443997383f, 0.3278753161430359f, -0.2048327475786209f, 0.18073472380638123f, 0.05612875893712044f, 0.5971975922584534f, 0.3121187686920166f, -0.07115577161312103f, 0.1249316930770874f, -0.11568056792020798f, 0.6231275796890259f, -0.26031848788261414f, 0.32211801409721375f, -0.153251051902771f, 0.015433127991855145f, -0.3982302248477936f, 0.182294100522995f, 0.5721282958984375f, 0.06693879514932632f, -0.243233323097229f, -0.1717810034751892f, -0.10103669762611389f, -0.37796005606651306f, 0.29509076476097107f, 0.09177376329898834f, 0.11037363857030869f, 0.346702516078949f, 0.12842071056365967f, 0.49986279010772705f, 0.2884584069252014f, -0.2817590534687042f, -0.11607522517442703f, 0.07014872878789902f, 0.0024729501456022263f, 0.003069168422371149f, -0.5752966403961182f, 0.3186078667640686f, 0.06496314704418182f, 0.10933021456003189f, 0.09796957671642303f, 0.43969017267227173f, 0.0042160991579294205f, 0.49094924330711365f, -0.25831836462020874f, -0.0124678248539567f, 0.09542858600616455f, 0.4304625689983368f, 0.5173574090003967f, 0.20092277228832245f, -0.09870455414056778f, -0.5960872769355774f, -0.5220977067947388f, -0.26811808347702026f, -0.1470012664794922f, 0.5144419074058533f, -0.6507248282432556f, -0.5103680491447449f, -0.6318864226341248f, -0.22917139530181885f, -0.09587717801332474f, 0.17322728037834167f, 0.14589916169643402f, -0.056082841008901596f, 0.15414264798164368f, -0.07037850469350815f, -0.14149627089500427f, -0.08494559675455093f, 0.46349194645881653f, -0.1914028376340866f, -0.5165942907333374f, -0.0926591157913208f, -3.024571378773544e-05f, -0.2423214167356491f, 0.19084079563617706f, -0.5179260969161987f, 0.2678687274456024f, 0.549274742603302f, -0.4493664503097534f, 0.19099237024784088f, 0.07696941494941711f, -0.18294164538383484f, 0.13449177145957947f, 0.22393934428691864f, 0.024851566180586815f, 0.21682581305503845f, 0.13589051365852356f, -0.3854689598083496f, -0.035939112305641174f, 0.6715861558914185f, 0.03129839524626732f, 0.060513727366924286f, 0.44163453578948975f, 0.3266445994377136f, -0.14176113903522491f, 0.33726784586906433f, -0.14455462992191315f, -0.48707637190818787f, 0.046314068138599396f, 0.08242225646972656f, -0.18899409472942352f, -0.15404072403907776f, -0.15626879036426544f, -0.45094892382621765f, -0.16178232431411743f, -0.47766542434692383f, 0.2546994686126709f, 0.05076718330383301f, -0.09008657932281494f, -0.10500968247652054f, 0.30526021122932434f, 0.27533286809921265f, -0.022669928148388863f, -0.4284486770629883f, 0.11741302162408829f, 0.5038482546806335f, 0.27258768677711487f, -0.08910292387008667f, -0.01688908413052559f, -0.6652588248252869f, 0.10145936906337738f, -0.6928080916404724f, -0.4012131989002228f, -0.46185302734375f, 0.10293613374233246f, 0.3065170347690582f, 0.1860632747411728f, -0.12294001132249832f, -0.22138287127017975f, 0.23818661272525787f, 0.1316625326871872f, -0.13715961575508118f, 0.26526397466659546f, 0.3638290762901306f, -0.06246529892086983f, 0.38503313064575195f, -0.256017804145813f, -0.02730620838701725f, -0.3111141622066498f, 0.41603875160217285f, -0.15589597821235657f, 0.29799750447273254f, -0.3082411289215088f, -0.005198461469262838f, 0.6071972250938416f, 0.34308770298957825f, 0.046386606991291046f, -0.03694578632712364f, -0.3873243033885956f, 0.13867038488388062f, 0.332206666469574f, -0.6301074028015137f, 0.6295794248580933f, -0.288957417011261f, -0.07840801030397415f, -0.1446535587310791f, 0.3253706097602844f, 0.0204195324331522f, 0.5113922357559204f, 0.1802465319633484f, -0.28832072019577026f, -0.3522685766220093f, -0.021898185834288597f, -0.3627372980117798f, 0.10832828283309937f, -0.20973972976207733f, -0.07961756736040115f, 0.08716803789138794f, 0.0021233325824141502f, 0.0061596534214913845f, 0.020725924521684647f, 0.2942892611026764f, -0.24568195641040802f, 0.03056996688246727f, -0.405729740858078f, 0.18520718812942505f, -0.1105390265583992f, 0.18075019121170044f, 0.43504396080970764f, 0.32036352157592773f, -0.3099727928638458f, 0.30224889516830444f, -0.46522220969200134f, 0.06167561560869217f, -0.12115246057510376f, -0.41071316599845886f, -0.16742616891860962f, 0.2207617610692978f, 0.14647948741912842f, -0.23248416185379028f, 0.2809480130672455f, -0.5093629360198975f, 0.32450929284095764f, -0.33170047402381897f, -0.503886878490448f, -0.4042936861515045f, -0.02037491649389267f, 0.1930779367685318f, 0.0291725005954504f, -0.12624119222164154f, -0.2827201187610626f, 0.07514264434576035f, -0.25798341631889343f, 0.000745734665542841f, -0.17641890048980713f, -0.04749513417482376f, 0.13987715542316437f, -0.014224653132259846f, 0.09218980371952057f, 0.5360912680625916f, 0.16302698850631714f, 0.12832634150981903f, 0.02434081956744194f, -0.20497506856918335f});
    set_f32(conv_pre_b, {-0.06644625961780548f, -0.19082240760326385f, -0.12831458449363708f, 0.03241082653403282f, -0.17428471148014069f, -0.11987948417663574f, -0.04844796285033226f, 0.1896217316389084f});
    set_f32(up0_w, {0.06727274507284164f, -0.3312308192253113f, -0.2246958464384079f, -0.10196308046579361f, -0.11291953921318054f, 0.23176053166389465f, 0.14122992753982544f, -0.1767924726009369f, -0.14189551770687103f, -0.09593751281499863f, -0.15466511249542236f, 0.15236283838748932f, -0.10961967706680298f, -0.5053971409797668f, 0.20764751732349396f, 0.11914200335741043f, 0.23810188472270966f, -0.5346435904502869f, -0.5882574319839478f, -0.2726501226425171f, 0.24999867379665375f, -0.023033032193779945f, 0.2805076241493225f, -0.14284248650074005f, 0.28574278950691223f, 0.37528568506240845f, 0.2680774927139282f, -0.5083094239234924f, 0.08211773633956909f, 0.3528771996498108f, -0.20678827166557312f, -0.08469396829605103f, -0.007018801290541887f, 0.036389805376529694f, 0.25599732995033264f, 0.1760520339012146f, 0.02146187052130699f, 0.18842081725597382f, -0.16017580032348633f, 0.08926768600940704f, -0.1681617647409439f, -0.4899221658706665f, -0.0006166492239572108f, -0.11662733554840088f, 0.23467795550823212f, -0.10122840851545334f, 0.14650845527648926f, -0.3883354663848877f, -0.10572115331888199f, -0.6784427762031555f, 0.288512259721756f, -0.33846521377563477f, 0.46745529770851135f, -0.0023907357826828957f, -0.04298035800457001f, -0.09693706780672073f, 0.2104259431362152f, -0.043569523841142654f, -0.17093387246131897f, -0.2933458089828491f, -0.5215452909469604f, -0.30000221729278564f, -0.46277615427970886f, 0.41775211691856384f, -0.1539594978094101f, -0.5467300415039062f, 0.20228104293346405f, -0.17728327214717865f, 0.23438692092895508f, 0.17653554677963257f, -0.4303986728191376f, -0.08223208039999008f, -0.25544095039367676f, 0.01292296964675188f, -0.3628087341785431f, -0.4724749028682709f, 0.572763204574585f, -0.4404984414577484f, 0.1791774183511734f, -0.2346692830324173f, -0.1612377166748047f, -0.015904176980257034f, -0.36189693212509155f, 0.3752315044403076f, 0.19249799847602844f, 0.5223947167396545f, -0.16914089024066925f, -0.16695749759674072f, -0.17928186058998108f, 0.10536190122365952f, -0.29588279128074646f, 0.3308253586292267f, 0.6023120284080505f, -0.1427142173051834f, -0.4598858058452606f, -0.48144659399986267f, -0.25646910071372986f, 0.047399964183568954f, -0.24485458433628082f, -0.26165327429771423f, -0.2572004497051239f, -0.14298632740974426f, -0.12261012941598892f, 0.1680782586336136f, 0.20458640158176422f, 0.021600279957056046f, 0.13947290182113647f, 0.01536425482481718f, 0.05521155893802643f, 0.378684401512146f, 0.1441299319267273f, -0.1338237226009369f, 0.3063565194606781f, 0.06787970662117004f, 0.040410593152046204f, 0.14131395518779755f, 0.12426091730594635f, 0.138262540102005f, -0.3390781879425049f, 0.2431860864162445f, 0.422251433134079f, 0.4096081256866455f, -0.16842120885849f, -0.1287643015384674f, -0.46798640489578247f, 0.32280683517456055f, 0.02732953615486622f, 0.12604062259197235f});
    set_f32(up0_b, {-0.05764628201723099f, -0.03765898197889328f, 0.12351393699645996f, -0.0783136710524559f});
    set_f32(up1_w, {-0.11941514164209366f, -0.682668924331665f, 0.017570272088050842f, 0.721219539642334f, -0.3569745719432831f, -0.2520574629306793f, -0.44587579369544983f, 0.014971529133617878f, -0.4328857362270355f, -0.28694114089012146f, 0.36715665459632874f, -0.22103272378444672f, -0.28496792912483215f, 0.11953260004520416f, -0.42305412888526917f, 0.43024328351020813f, 0.4307360351085663f, -0.05009007081389427f, 0.22362424433231354f, 0.5624802708625793f, -0.20131535828113556f, 0.010747048072516918f, 0.36314857006073f, 0.1841735690832138f, -0.07142581790685654f, 0.0008925935253500938f, 0.4798847734928131f, 0.05677806958556175f, -0.03410143777728081f, 0.46952325105667114f, 0.09526214748620987f, 0.20041659474372864f});
    set_f32(up1_b, {0.10528966039419174f, -0.036553870886564255f});
    set_f32(rb0_c0_w, {-0.13138647377490997f, -0.23658083379268646f, 0.12337201088666916f, -0.16114550828933716f, -0.007280149031430483f, 0.3535495698451996f, -0.04484526813030243f, 0.3572528064250946f, -0.3462925851345062f, -0.1723397821187973f, -0.06583283841609955f, -0.11342433840036392f, 0.17326800525188446f, -0.20768418908119202f, 0.3503912687301636f, 0.5878909230232239f, -0.033836208283901215f, -0.05469425395131111f, -0.24072787165641785f, 0.06025409698486328f, -0.7197795510292053f, -0.4997040629386902f, -0.37981879711151123f, -0.3707151710987091f, -0.6246180534362793f, 0.7208583354949951f, 0.4668233096599579f, -0.28780102729797363f, -0.3127686381340027f, 0.08704707771539688f, 0.09484230726957321f, -0.05082632228732109f, -0.16257734596729279f, 0.6922188401222229f, 0.042066819965839386f, -0.14494824409484863f, -0.08244252949953079f, 0.2017616629600525f, -0.395504891872406f, -0.006768522784113884f, -0.9745211005210876f, 0.05214974284172058f, -0.05920178443193436f, 0.010639230720698833f, -0.12358754128217697f, -0.278245210647583f, -0.13917769491672516f, 0.21261325478553772f});
    set_f32(rb0_c0_b, {0.04715040698647499f, -0.04293670877814293f, -0.10832046717405319f, -0.13159184157848358f});
    set_f32(rb0_c1_w, {0.16154804825782776f, -0.31050142645835876f, -0.26423484086990356f, 0.6489802598953247f, -0.20020601153373718f, -0.8004103302955627f, -0.8359041810035706f, -0.4651607573032379f, -0.11374574154615402f, -0.19459325075149536f, 0.6993862986564636f, -0.4523547887802124f, -0.3026726245880127f, -0.598993182182312f, 0.0643201693892479f, 0.1696212738752365f, -0.039959169924259186f, -0.1343831568956375f, -0.06310519576072693f, 0.29257217049598694f, -0.41632959246635437f, 0.1609870195388794f, 0.2449205219745636f, -0.23522207140922546f, 0.11509030312299728f, -0.1887291520833969f, -0.1705951988697052f, -0.10739698261022568f, 0.3033122420310974f, -0.15704892575740814f, 0.27908945083618164f, -0.10337306559085846f, 0.2366100400686264f, 0.7109183669090271f, 0.4385503828525543f, 0.1597099006175995f, -0.02116955630481243f, 0.3507506251335144f, -0.13691945374011993f, 0.11621064692735672f, 0.2770155966281891f, -0.4463488757610321f, -0.24809075891971588f, 0.1534663736820221f, -0.49657997488975525f, -0.056396204978227615f, 0.0468299500644207f, -0.1710100919008255f});
    set_f32(rb0_c1_b, {0.06651653349399567f, 0.03846804425120354f, 0.10090448707342148f, 0.02178858406841755f});
    set_f32(rb1_c0_w, {-0.3309285044670105f, -0.06704898178577423f, 0.16964319348335266f, -0.07230866700410843f, -0.47449639439582825f, 0.24395494163036346f, -0.038289863616228104f, 0.010681511834263802f, 0.42620566487312317f, 0.1600228250026703f, 0.013597851619124413f, -0.24657480418682098f, -0.1326937973499298f, 0.07225899398326874f, -0.31755203008651733f, -0.023193975910544395f, 0.18793919682502747f, -0.2576095461845398f, 0.23672625422477722f, 0.04832923039793968f, 0.3029009699821472f, -0.5141538381576538f, 0.6231688857078552f, 0.09318587183952332f, -0.04405743628740311f, -0.01821988821029663f, 0.05150049552321434f, -0.268984854221344f, -0.27695444226264954f, 0.27209383249282837f, 0.1397029012441635f, -0.12217030674219131f, 0.06419353932142258f, -0.08173055946826935f, 0.09239919483661652f, -0.26891010999679565f, 0.14493045210838318f, -0.45092901587486267f, -0.42587950825691223f, 0.31792399287223816f, 0.2004922330379486f, 0.13542434573173523f, -0.26381099224090576f, -0.22102148830890656f, -0.3674614727497101f, -0.0419304221868515f, -0.11533699184656143f, 0.03758981078863144f, 0.04026325047016144f, 0.1813524067401886f, 0.04852770268917084f, -0.44401347637176514f, 0.36974209547042847f, 0.2171911895275116f, -0.49406054615974426f, 0.4509628117084503f, 0.3046662211418152f, -0.11671915650367737f, 0.026795003563165665f, -0.44131556153297424f, 0.3578309118747711f, -0.6855896711349487f, 0.3988107144832611f, -0.3330385684967041f, -0.17478442192077637f, -0.5426536202430725f, 0.20408998429775238f, 0.10585696250200272f, -0.05306561663746834f, 0.010264761745929718f, 0.22210679948329926f, 0.6765312552452087f, -0.3764759302139282f, -0.18302755057811737f, -0.24099572002887726f, 0.08015699684619904f, -0.027745874598622322f, -0.08651971071958542f, 0.01793755404651165f, 0.2292829155921936f});
    set_f32(rb1_c0_b, {0.014964543282985687f, -0.0911383256316185f, 0.09728308767080307f, -0.02901567332446575f});
    set_f32(rb1_c1_w, {-0.5647217631340027f, -0.3957134485244751f, 0.022483455017209053f, 0.3348514437675476f, 0.12335441261529922f, 0.17698884010314941f, 0.0658062994480133f, 0.22914274036884308f, 0.1949928253889084f, -0.23542647063732147f, -0.12881089746952057f, 0.4012192487716675f, 0.06090308725833893f, -0.4658640921115875f, 0.10199541598558426f, 0.18283510208129883f, -0.26032155752182007f, 0.058158550411462784f, -0.13846279680728912f, -0.2962549924850464f, -0.14675574004650116f, 0.09945676475763321f, 0.39782556891441345f, 0.02162926457822323f, -0.12622018158435822f, 0.2712095081806183f, 0.055774834007024765f, 0.1724555641412735f, -0.4084250032901764f, -0.7060266733169556f, -0.17194844782352448f, 0.4354586899280548f, 0.27602559328079224f, -0.014483603648841381f, 0.031443286687135696f, 0.533201277256012f, 0.5729814767837524f, -0.38512465357780457f, -0.1876494288444519f, -0.41466692090034485f, -0.2512489855289459f, 0.3842224180698395f, 0.22211593389511108f, -0.19807399809360504f, 0.227730393409729f, 0.07500521838665009f, 0.048978887498378754f, 0.467628538608551f, -0.1189684197306633f, 0.13118073344230652f, -0.15384680032730103f, -0.17648626863956451f, 0.5416335463523865f, -0.10306756943464279f, -0.35076025128364563f, -0.143259197473526f, 0.10754413157701492f, -0.2040763795375824f, 0.13003158569335938f, 0.3455694317817688f, 0.15411323308944702f, 0.17842400074005127f, 0.6366115808486938f, -0.44867396354675293f, 0.04961803928017616f, -0.5079440474510193f, -0.1831567883491516f, 0.2689231336116791f, -0.01068081520497799f, -0.16990938782691956f, 0.11860505491495132f, -0.008109388872981071f, -0.08173161745071411f, -0.5584269165992737f, -0.29519423842430115f, 0.30610936880111694f, -0.29018837213516235f, 0.21980136632919312f, 0.13520894944667816f, 0.3470756411552429f});
    set_f32(rb1_c1_b, {0.14693902432918549f, 0.004036751575767994f, -0.08912418782711029f, -0.105606809258461f});
    set_f32(rb2_c0_w, {0.028801998123526573f, -0.014671724289655685f, 0.40351566672325134f, -0.21513675153255463f, -0.32053032517433167f, -0.21332813799381256f, 0.15955215692520142f, -0.5836032032966614f, -0.32190465927124023f, -0.16352415084838867f, 0.028773896396160126f, -0.6166872382164001f});
    set_f32(rb2_c0_b, {0.09525768458843231f, -0.08485528826713562f});
    set_f32(rb2_c1_w, {-0.3745489716529846f, -0.3660585582256317f, -0.29083406925201416f, 0.5352700352668762f, -0.43195611238479614f, -0.6367340087890625f, 0.021837539970874786f, -0.12476561218500137f, -0.5660884380340576f, 0.11605466157197952f, -0.4221849739551544f, -0.36949384212493896f});
    set_f32(rb2_c1_b, {-0.18932296335697174f, -0.008464938960969448f});
    set_f32(rb3_c0_w, {-0.37213191390037537f, 0.23501521348953247f, -0.08399119973182678f, 0.014930937439203262f, -0.3648974299430847f, 0.058544937521219254f, -0.08304954320192337f, 0.17593999207019806f, -0.03181374818086624f, -0.40234246850013733f, -0.29651543498039246f, -0.10527031868696213f, 0.6850675940513611f, -0.23026920855045319f, -0.0005349860875867307f, -0.09953358769416809f, 0.08257745951414108f, -0.18305645883083344f, -0.295356810092926f, 0.21421080827713013f});
    set_f32(rb3_c0_b, {0.06628292798995972f, -0.0021927461493760347f});
    set_f32(rb3_c1_w, {-0.41453883051872253f, -0.2781943678855896f, -0.037992946803569794f, 0.6107516288757324f, 0.3409042954444885f, 0.25702279806137085f, 0.5807203054428101f, 0.0858064815402031f, 0.17126914858818054f, 0.2509654462337494f, -0.18889428675174713f, 0.09761513769626617f, -0.03320741280913353f, -0.23971012234687805f, -0.0504876971244812f, 0.01319195982068777f, -0.060915108770132065f, 0.21237081289291382f, 0.25220194458961487f, -0.3582276701927185f});
    set_f32(rb3_c1_b, {0.06505131721496582f, 0.014152111485600471f});
    set_f32(conv_post_w, {-0.04453146085143089f, -0.08658070117235184f, 0.17482832074165344f, 0.17478744685649872f, -0.10996519774198532f, 0.14122922718524933f, 0.12835870683193207f, -0.37328168749809265f, 0.24400447309017181f, -0.2825877070426941f, 0.13529355823993683f, 0.30121809244155884f, 0.365219384431839f, -0.08856014907360077f});
    s.compute(gf);

    const std::vector<float> expected = {0.06750647723674774f, 0.06339208036661148f, 0.06914437562227249f, 0.03997614607214928f, 0.07125523686408997f, 0.09070377051830292f, 0.1256563365459442f, -0.12171465903520584f, 0.14708009362220764f, 0.13669118285179138f, -0.07071546465158463f, 0.007865207269787788f, 0.0738728865981102f, -0.00545044569298625f, 0.05549495667219162f, 0.01891183853149414f};
    const auto result = get_f32(out);
    if (!approx_eq(result, expected, 1e-4f)) {
        std::fprintf(stderr, "GENERATOR got:");
        for (float v : result) std::fprintf(stderr, " %f", static_cast<double>(v));
        std::fprintf(stderr, "\nexpected:");
        for (float v : expected) std::fprintf(stderr, " %f", static_cast<double>(v));
        std::fprintf(stderr, "\n");
    }
    LOOM_CHECK(approx_eq(result, expected, 1e-4f));
}

void test_text_encoder_assembly() {
    // VITS's `TextEncoder` (models.py), composed from already-verified primitives -- no new primitive
    // needed. `conv_q`/`conv_k`/`conv_v`/`conv_o`/`proj` are all kernel_size=1 convs, which are just a
    // per-position linear projection -- expressed directly via MUL_MAT (this engine's standard
    // attention-convention idiom, e.g. Qwen3's own QKV projections) rather than CONV_1D, avoiding
    // unneeded transposes to/from conv's [T,C,N] layout. `attentions.Encoder`'s own `LayerNorm` (transpose
    // -> F.layer_norm -> transpose, same class DDSConv uses) needs NO transpose here, unlike DDSConv's
    // [T,C] convention: TextEncoder's attention pipeline is already channel-first ([C,T], matching
    // REL_POS_ATTENTION_SHAW's own convention and GET_ROWS's embedding-lookup output), which is exactly
    // `ggml_norm`'s own normalization axis (ne[0]). Only the FFN's kernel_size=3 convs genuinely need
    // CONV_1D, hence a transpose in and back out. `m`/`logs` are split along the CHANNEL axis (ne[0] in
    // this convention), via the same offset-view trick as everywhere else.
    //
    // Small-scale (n_vocab=5, hidden_channels=4, n_heads=2, filter_channels=6, kernel_size=3, n_layers=1,
    // out_channels=4, T=2, window_size=1 -- chosen so 2*window_size+1 == 2*T-1 == 3, i.e. no
    // pad/crop-to-dynamic-T logic is exercised here; that real, genuinely-dynamic-length concern is
    // deferred to the full conversion script/e2e test, task #78's remaining scope, same as noted in
    // BACKLOG.md), cross-checked against real execution of piper's own models.TextEncoder.
    constexpr int64_t kHidden = 4, kHeads = 2, kHeadDim = 2, kFilter = 6, kK = 3, kT = 2, kOut = 4, kVocab = 5;
    GgmlScratch s;
    ggml_context* ctx = s.ctx.get();
    loom::SymbolEnv env;
    loom::PrimitiveContext pc{ctx, env, nullptr};

    ggml_tensor* emb_table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kVocab);
    ggml_tensor* token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kT);
    ggml_tensor* q_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kHidden);
    ggml_tensor* q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    ggml_tensor* k_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kHidden);
    ggml_tensor* k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    ggml_tensor* v_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kHidden);
    ggml_tensor* v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    ggml_tensor* o_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kHidden);
    ggml_tensor* o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    ggml_tensor* emb_rel_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHeadDim, 2 * kT - 1);
    ggml_tensor* emb_rel_v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHeadDim, 2 * kT - 1);
    ggml_tensor* ln1_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    ggml_tensor* ln1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    ggml_tensor* ffn_w1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kK, kHidden, kFilter);
    ggml_tensor* ffn_b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFilter);
    ggml_tensor* ffn_w2 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kK, kFilter, kHidden);
    ggml_tensor* ffn_b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    ggml_tensor* ln2_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    ggml_tensor* ln2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    ggml_tensor* proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, 2 * kOut);
    ggml_tensor* proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * kOut);
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kT, kT);

    std::vector<ggml_tensor*> all = {emb_table, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b, emb_rel_k, emb_rel_v,
                                      ln1_g, ln1_b, ffn_w1, ffn_b1, ffn_w2, ffn_b2, ln2_g, ln2_b, proj_w, proj_b, mask};
    for (ggml_tensor* t : all) ggml_set_input(t);
    ggml_set_input(token_ids);

    // x = emb(token_ids) * sqrt(hidden_channels)
    ggml_tensor* x = op("GET_ROWS")(pc, {emb_table, token_ids}, {})[0]; // [4, T]
    x = ggml_scale(ctx, x, std::sqrt(static_cast<float>(kHidden)));

    // Self-attention (channel-first [C,T] convention throughout; 1x1 "convs" are plain MUL_MAT).
    ggml_tensor* q = ggml_add(ctx, ggml_mul_mat(ctx, q_w, x), ggml_reshape_2d(ctx, q_b, kHidden, 1));
    ggml_tensor* k = ggml_add(ctx, ggml_mul_mat(ctx, k_w, x), ggml_reshape_2d(ctx, k_b, kHidden, 1));
    ggml_tensor* v = ggml_add(ctx, ggml_mul_mat(ctx, v_w, x), ggml_reshape_2d(ctx, v_b, kHidden, 1));
    q = ggml_reshape_3d(ctx, q, kHeadDim, kHeads, kT);
    k = ggml_reshape_3d(ctx, k, kHeadDim, kHeads, kT);
    v = ggml_reshape_3d(ctx, v, kHeadDim, kHeads, kT);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    ggml_tensor* attn_out = op("REL_POS_ATTENTION_SHAW")(pc, {q, k, v, emb_rel_k, emb_rel_v, mask}, {{"scale", scale}})[0]; // [4, T]
    ggml_tensor* o = ggml_add(ctx, ggml_mul_mat(ctx, o_w, attn_out), ggml_reshape_2d(ctx, o_b, kHidden, 1));

    // Residual + post-norm 1 (LayerNorm over ne[0]=C, no transpose needed in this convention).
    x = ggml_add(ctx, x, o);
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_add(ctx, ggml_mul(ctx, x, ln1_g), ln1_b);

    // FFN (kernel_size=3, "same" padding=1) -- needs a transpose to/from CONV_1D's [T,C,N] convention.
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x)); // [T, C]
    ggml_tensor* h = op("CONV_1D")(pc, {ffn_w1, ggml_reshape_3d(ctx, xt, kT, kHidden, 1)}, {{"s0", 1}, {"p0", 1}, {"d0", 1}})[0];
    h = ggml_add(ctx, h, ggml_reshape_3d(ctx, ffn_b1, 1, kFilter, 1));
    h = op("RELU")(pc, {h}, {})[0];
    ggml_tensor* h2 = op("CONV_1D")(pc, {ffn_w2, h}, {{"s0", 1}, {"p0", 1}, {"d0", 1}})[0];
    h2 = ggml_add(ctx, h2, ggml_reshape_3d(ctx, ffn_b2, 1, kHidden, 1));
    ggml_tensor* ffn_out = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, h2, kT, kHidden))); // [C, T]

    // Residual + post-norm 2.
    x = ggml_add(ctx, x, ffn_out);
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_add(ctx, ggml_mul(ctx, x, ln2_g), ln2_b);

    // proj -> split into m/logs along the channel axis (ne[0]).
    ggml_tensor* stats = ggml_add(ctx, ggml_mul_mat(ctx, proj_w, x), ggml_reshape_2d(ctx, proj_b, 2 * kOut, 1));
    ggml_tensor* m = ggml_cont(ctx, ggml_view_2d(ctx, stats, kOut, kT, stats->nb[1], 0));
    ggml_tensor* logs = ggml_cont(ctx, ggml_view_2d(ctx, stats, kOut, kT, stats->nb[1], kOut * static_cast<int64_t>(sizeof(float))));

    // Three co-equal outputs (x, m, logs) -- none reachable from the others, so s.expand()'s
    // single-output convenience wrapper (which allocates immediately after building forward from just
    // one tensor) would leave m/logs's own nodes unallocated. Build forward from all three before the
    // one gallocr_alloc_graph call instead. Each ALSO needs ggml_set_output(): without it, gallocr's
    // liveness analysis sees `x` has no reader after `stats = mul_mat(proj_w, x)` computes and frees its
    // buffer for reuse by a later tensor in the same graph -- silently corrupting `x` once nothing reads
    // it again, exactly the bug class GraphBuilder's own single designated output already guards against
    // (`ggml_set_output(result.output)` in graph_builder.cpp) but which a hand-built multi-output test
    // graph must do explicitly for every one of its outputs.
    ggml_set_output(x);
    ggml_set_output(m);
    ggml_set_output(logs);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, x);
    ggml_build_forward_expand(gf, m);
    ggml_build_forward_expand(gf, logs);
    ggml_gallocr_alloc_graph(s.galloc.get(), gf);

    set_i32(token_ids, {2, 4});
    set_f32(emb_table, {0.2735578715801239f, -0.029098372906446457f, -0.1942010074853897f, 0.21115396916866302f, 0.29464927315711975f, -0.5971376895904541f, 0.0012183618964627385f, 0.34370148181915283f, 0.01544987689703703f, 0.34470057487487793f, -0.12315557152032852f, 0.02899819053709507f, 0.2535160481929779f, 0.028626857325434685f, -0.5067577958106995f, 0.12382772564888f, -0.5556535720825195f, 0.031837474554777145f, 0.047392673790454865f, -0.4971745014190674f});
    set_f32(q_w, {0.16759054362773895f, 0.15970514714717865f, 0.5759621858596802f, 0.3261508047580719f, 0.4735068678855896f, 0.31057167053222656f, -0.42894113063812256f, -0.1436341255903244f, 0.04492757096886635f, -0.032004643231630325f, 0.35695692896842957f, -0.013003856875002384f, 0.35007619857788086f, -0.1703319400548935f, 0.14687517285346985f, -0.0585513561964035f});
    set_f32(q_b, {0.0018562375335022807f, 0.24750427901744843f, -0.1207750216126442f, 0.068842314183712f});
    set_f32(k_w, {0.25316792726516724f, -0.18948189914226532f, 0.02738010324537754f, -0.3664647340774536f, 0.10411250591278076f, 0.20954985916614532f, -0.1471162587404251f, -0.7015507221221924f, 0.28883984684944153f, -0.3773203492164612f, 0.0351981557905674f, 0.07309666275978088f, -0.07987263053655624f, 0.11759988218545914f, 0.16871631145477295f, -0.6883307695388794f});
    set_f32(k_b, {-0.08381973952054977f, -0.12479590624570847f, -0.11161511391401291f, -0.020301884040236473f});
    set_f32(v_w, {-0.07005248218774796f, 0.029186084866523743f, -0.3007197380065918f, 0.39772945642471313f, 0.4511043429374695f, -0.06705683469772339f, -0.2166220098733902f, -0.2153405100107193f, 0.06337670981884003f, 0.10608106851577759f, 0.1588565707206726f, -0.01897706277668476f, 0.33497175574302673f, -0.1282695233821869f, 0.3161723017692566f, -0.2897670567035675f});
    set_f32(v_b, {-0.06864573061466217f, 0.22976069152355194f, -0.041616760194301605f, 0.07118459790945053f});
    set_f32(o_w, {-0.4921324849128723f, -0.13322371244430542f, -0.42208126187324524f, 0.4649789035320282f, -0.0915786400437355f, 0.21012715995311737f, -0.357303649187088f, -0.0037014083936810493f, -0.07684749364852905f, -0.09428457170724869f, -0.2689804136753082f, -0.21729077398777008f, -0.3195062279701233f, 0.8340452313423157f, -0.15301188826560974f, -0.5446470975875854f});
    set_f32(o_b, {-0.10522573441267014f, 0.048293713480234146f, 0.04918471351265907f, -0.1517430990934372f});
    set_f32(emb_rel_k, {-0.04063500091433525f, -0.059720247983932495f, 0.13462351262569427f, -0.27966010570526123f, -0.14148665964603424f, -0.2959136664867401f});
    set_f32(emb_rel_v, {-0.024137595668435097f, 0.1730950027704239f, -0.08115755021572113f, -0.08489614725112915f, -0.2138059139251709f, -0.028592733666300774f});
    set_f32(ln1_g, {0.9696376323699951f, 0.96767258644104f, 0.9705373644828796f, 1.1289485692977905f});
    set_f32(ln1_b, {0.03487204760313034f, 0.06428582966327667f, -0.06891416758298874f, 0.09101004153490067f});
    set_f32(ffn_w1, {-0.2714439332485199f, -0.008271753787994385f, 0.041289448738098145f, 0.13484232127666473f, 0.53193598985672f, 0.3697507381439209f, -0.5083812475204468f, -0.21981053054332733f, -0.1922120898962021f, -0.16295376420021057f, 0.17139510810375214f, -0.09396044164896011f, 0.60767662525177f, 0.39448288083076477f, -0.48059672117233276f, -0.2634173631668091f, 0.34921127557754517f, -0.3543115556240082f, -0.06502442806959152f, -0.36586788296699524f, -0.18340015411376953f, -0.048870593309402466f, 0.05859798938035965f, -0.3472621738910675f, 0.0702788233757019f, 0.27525201439857483f, 0.4604566991329193f, -0.054058875888586044f, -0.10568193346261978f, 0.4706215560436249f, -0.03159633278846741f, 0.6586230993270874f, 0.2297285497188568f, -0.17153030633926392f, 0.10353963822126389f, 0.2619233727455139f, -0.17634981870651245f, -0.06519851833581924f, 0.11930490285158157f, 0.12011496722698212f, 0.30717453360557556f, 0.0640571117401123f, -0.34942737221717834f, -0.22885631024837494f, 0.39617833495140076f, -0.17520076036453247f, 0.009876400232315063f, -0.10272278636693954f, -0.1499970406293869f, 0.10452484339475632f, -0.2981511950492859f, 0.4296700060367584f, -0.04059664160013199f, 0.4167649745941162f, 1.0697510242462158f, 0.4582308828830719f, -0.14517369866371155f, 0.24003469944000244f, -0.2285708338022232f, 0.16972704231739044f, -0.09724616259336472f, -0.08193118125200272f, 0.6763309240341187f, -0.39737096428871155f, -0.6257888674736023f, 0.4187425673007965f, 0.039433300495147705f, 0.3393603265285492f, -0.7068489789962769f, 0.04450235143303871f, -0.023179911077022552f, -0.06513185054063797f});
    set_f32(ffn_b1, {0.018316062167286873f, 0.09203507006168365f, 0.10527931898832321f, 0.10058531910181046f, -0.19082622230052948f, 0.17418813705444336f});
    set_f32(ffn_w2, {0.8277115225791931f, -0.3755279779434204f, -0.25592949986457825f, -0.3443884551525116f, -0.09622064977884293f, -0.03377624973654747f, -0.3496638238430023f, 0.2859630882740021f, 0.2623225152492523f, -0.016669629141688347f, -0.028164559975266457f, -0.1805533468723297f, 0.03473224863409996f, -0.24599270522594452f, -0.29890063405036926f, -0.23433826863765717f, -0.3329066336154938f, -0.19068841636180878f, 0.47949540615081787f, -0.0004641084815375507f, -0.006961026694625616f, -0.05735870078206062f, -0.18148890137672424f, 0.4106942415237427f, 0.2920328378677368f, -0.06295190006494522f, 0.09332707524299622f, 0.046693600714206696f, -0.5069254636764526f, -0.31703001260757446f, 0.25245901942253113f, 0.20999938249588013f, 0.17259567975997925f, -0.27805036306381226f, -0.05481220781803131f, 0.829495370388031f, -0.4501914978027344f, -0.03330798074603081f, 0.22439567744731903f, 0.20033538341522217f, -0.6904001235961914f, -0.06470808386802673f, 0.5508257746696472f, -0.34868133068084717f, -0.21180905401706696f, 0.40556564927101135f, -0.2539864778518677f, -0.08541908115148544f, -0.1606588363647461f, 0.010778840631246567f, -0.04999406263232231f, -0.22627045214176178f, 0.2598220705986023f, -0.04473583772778511f, -0.062381595373153687f, -0.07359520345926285f, 0.4276692271232605f, -0.29265928268432617f, 0.3608766496181488f, -0.1327328085899353f, -0.47788992524147034f, 0.13238000869750977f, 0.0024753448087722063f, 0.0859682708978653f, -0.43466973304748535f, -0.20552048087120056f, -0.1523999273777008f, -0.39764708280563354f, 0.05667705461382866f, 0.06310270726680756f, -0.4160870611667633f, -0.09999572485685349f});
    set_f32(ffn_b2, {0.10474057495594025f, -0.17189092934131622f, -0.10928753763437271f, 0.002550143515691161f});
    set_f32(ln2_g, {1.0179455280303955f, 1.0900791883468628f, 0.973031759262085f, 0.8840893507003784f});
    set_f32(ln2_b, {-0.15928658843040466f, -0.047316428273916245f, 0.11442150920629501f, -0.03517582640051842f});
    set_f32(proj_w, {-0.225281223654747f, 0.1394084393978119f, -0.15507099032402039f, 0.19313012063503265f, -0.5273115634918213f, 0.3183804750442505f, 0.42294031381607056f, -0.32663923501968384f, -0.03912394121289253f, 0.3300371468067169f, -0.1616499274969101f, 0.005678706802427769f, -0.025954926386475563f, -0.1670980006456375f, 0.1362776905298233f, 0.290764182806015f, 0.14662151038646698f, 0.3958304524421692f, -0.26786530017852783f, 0.3575526773929596f, -0.3261091113090515f, 0.4323309063911438f, 0.01708049140870571f, 0.008818165399134159f, -0.6837702989578247f, 0.02192666567862034f, -0.2089049518108368f, -0.22442680597305298f, 0.14644932746887207f, 0.037406276911497116f, 0.10548225790262222f, -0.32215383648872375f});
    set_f32(proj_b, {-0.019840670749545097f, -0.06577856838703156f, -0.21198606491088867f, -0.11890077590942383f, 0.09860371053218842f, -0.10659334808588028f, 0.014217301271855831f, -0.05214131623506546f});
    set_f32(mask, {0.0f, 0.0f, 0.0f, 0.0f});
    s.compute(gf);

    // Note: these are the real PyTorch reference values TRANSPOSED to match this test's ne=[C,T]
    // (channel-fastest) convention -- PyTorch's own x_out/m/logs are (B,C,T), T fastest (materialized
    // contiguous after the module's internal `torch.transpose(x,1,-1)`), which is the opposite axis
    // order from GET_ROWS's natural channel-fastest embedding output that this test's whole pipeline
    // stays in throughout. Same values, dumped via `.transpose(1,2).flatten()` instead of `.flatten()`
    // directly -- confirmed by first reproducing the *untransposed* mismatch (a same-value, different-
    // order permutation, not a real computation error) before concluding this was a comparison-order
    // issue rather than a genuine bug.
    const std::vector<float> expected_x = {-0.5915854573249817f, 1.298757791519165f, -1.244606375694275f, 0.48337045311927795f, -0.612420380115509f, 1.190181016921997f, 0.8164405226707458f, -1.2831271886825562f};
    const std::vector<float> expected_m = {0.5808459520339966f, -0.02461153268814087f, 0.4437328577041626f, -0.34963130950927734f, -0.0903693288564682f, 1.4005135297775269f, 0.06551410257816315f, -0.5637071132659912f};
    const std::vector<float> expected_logs = {1.0321695804595947f, 0.6308252215385437f, 0.5987263917922974f, -0.37720048427581787f, -0.19756203889846802f, 0.6103050112724304f, 0.5764784812927246f, 0.402174711227417f};
    LOOM_CHECK(approx_eq(get_f32(x), expected_x, 1e-4f));
    LOOM_CHECK(approx_eq(get_f32(m), expected_m, 1e-4f));
    LOOM_CHECK(approx_eq(get_f32(logs), expected_logs, 1e-4f));
}

// SSM (Mamba/SSD) / RWKV primitives (EXPORT-IMPROVEMENT-BACKLOG.md item 4) -- shape-only checks, mirroring
// ggml's own test-backend-ops.cpp shapes for these ops (test_ssm_conv/test_ssm_scan/test_rwkv_wkv6/
// test_rwkv_wkv7): no exporter/model currently produces these, so there's no numeric reference to check
// against yet, only that this project's thin wraps build a valid ggml graph and the CPU backend actually
// computes it (not just infers a shape) without asserting.

void test_ssm_conv() {
    GgmlScratch s;
    // d_conv=3, d_inner=3, n_t=2, n_s=1 -- sx->ne[0] must equal (d_conv - 1 + n_t).
    ggml_tensor* sx = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, /*d_conv-1+n_t=*/4, /*d_inner=*/3, /*n_s=*/1);
    ggml_tensor* c = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, /*d_conv=*/3, /*d_inner=*/3);
    ggml_set_input(sx);
    ggml_set_input(c);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("SSM_CONV")(pc, {sx, c}, {})[0];
    LOOM_CHECK(out->ne[0] == 3);  // d_inner
    LOOM_CHECK(out->ne[1] == 2);  // n_t
    LOOM_CHECK(out->ne[2] == 1);  // n_s

    ggml_cgraph* gf = s.expand(out);
    set_f32(sx, std::vector<float>(static_cast<size_t>(ggml_nelements(sx)), 1.0f));
    set_f32(c, std::vector<float>(static_cast<size_t>(ggml_nelements(c)), 1.0f));
    s.compute(gf); // must not assert/crash -- values aren't checked, no reference exists yet.
}

void test_ssm_scan() {
    GgmlScratch s;
    // d_state=16 (not a small arbitrary value) deliberately -- ggml's own CPU kernel comment notes
    // "d_state is usually 16" for this exact code path; a too-small head_dim/head_count/d_state combo
    // risks an unvalidated edge case in the kernel's SIMD blocking (confirmed the hard way: an earlier,
    // smaller-shaped version of this test file's own RWKV_WKV7 test corrupted the heap with head_size=4,
    // well below any real SIMD width ever exercised in ggml's own test-backend-ops.cpp defaults).
    const int64_t d_state = 16, head_dim = 2, n_head = 2, n_group = 1, n_seq_tokens = 3, n_seqs = 1;
    ggml_tensor* state = ggml_new_tensor_4d(s.ctx.get(), GGML_TYPE_F32, d_state, head_dim, n_head, n_seqs);
    ggml_tensor* x = ggml_new_tensor_4d(s.ctx.get(), GGML_TYPE_F32, head_dim, n_head, n_seq_tokens, n_seqs);
    ggml_tensor* dt = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, n_head, n_seq_tokens, n_seqs);
    ggml_tensor* A = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, /*head_dim>1 so 1=*/1, n_head);
    ggml_tensor* B = ggml_new_tensor_4d(s.ctx.get(), GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs);
    ggml_tensor* C = ggml_new_tensor_4d(s.ctx.get(), GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs);
    ggml_tensor* ids = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_I32, n_seqs);
    for (ggml_tensor* t : {state, x, dt, A, B, C}) ggml_set_input(t);
    ggml_set_input(ids);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("SSM_SCAN")(pc, {state, x, dt, A, B, C, ids}, {})[0];
    // Concatenated y + new ssm_states, per ggml_ssm_scan's own doc comment.
    LOOM_CHECK(ggml_nelements(out) == ggml_nelements(x) + d_state * head_dim * n_head * n_seqs);

    ggml_cgraph* gf = s.expand(out);
    for (ggml_tensor* t : {state, x, dt, A, B, C}) {
        set_f32(t, std::vector<float>(static_cast<size_t>(ggml_nelements(t)), 0.1f));
    }
    set_i32(ids, {0});
    s.compute(gf);
}

void test_rwkv_wkv6() {
    GgmlScratch s;
    // head_size=64 matches ggml's own test-backend-ops.cpp default for this op exactly (not shrunk the
    // way most of this file's other shapes are) -- its CPU kernel's SIMD blocking has only ever been
    // exercised at that width; a smaller head_size corrupted the heap in an earlier version of the
    // sibling RWKV_WKV7 test below (see that test's own comment).
    const int64_t head_size = 64, head_count = 2, n_tokens = 3, n_seqs = 1;
    ggml_tensor* k = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, head_size, head_count, n_tokens);
    ggml_tensor* v = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, head_size, head_count, n_tokens);
    ggml_tensor* r = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, head_size, head_count, n_tokens);
    ggml_tensor* tf = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, head_size, head_count);
    ggml_tensor* td = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, head_size, head_count, n_tokens);
    ggml_tensor* state = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, head_size * head_size * head_count, n_seqs);
    for (ggml_tensor* t : {k, v, r, tf, td, state}) ggml_set_input(t);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("RWKV_WKV6")(pc, {k, v, r, tf, td, state}, {})[0];
    LOOM_CHECK(ggml_nelements(out) == head_size * head_count * n_tokens + head_size * head_size * head_count * n_seqs);

    ggml_cgraph* gf = s.expand(out);
    for (ggml_tensor* t : {k, v, r, tf, td, state}) {
        set_f32(t, std::vector<float>(static_cast<size_t>(ggml_nelements(t)), 0.1f));
    }
    s.compute(gf);
}

void test_rwkv_wkv7() {
    GgmlScratch s;
    // head_size=64 (ggml's own test-backend-ops.cpp default) -- a smaller head_size (originally 4)
    // corrupted the heap here ("corrupted size vs. prev_size while consolidating" during
    // ggml_graph_compute), almost certainly this op's SIMD-blocked CPU kernel assuming a SIMD-width-
    // aligned head_size rather than handling an arbitrary remainder safely. Confirmed empirically, not
    // theorized -- ggml's own test suite never exercises this op below 64 either.
    const int64_t head_size = 64, head_count = 2, n_tokens = 3, n_seqs = 1;
    ggml_tensor* r = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, head_size, head_count, n_tokens);
    ggml_tensor* w = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, head_size, head_count, n_tokens);
    ggml_tensor* k = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, head_size, head_count, n_tokens);
    ggml_tensor* v = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, head_size, head_count, n_tokens);
    ggml_tensor* a = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, head_size, head_count, n_tokens);
    ggml_tensor* b = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, head_size, head_count, n_tokens);
    ggml_tensor* state = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, head_size * head_size * head_count, n_seqs);
    for (ggml_tensor* t : {r, w, k, v, a, b, state}) ggml_set_input(t);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* out = op("RWKV_WKV7")(pc, {r, w, k, v, a, b, state}, {})[0];
    LOOM_CHECK(ggml_nelements(out) == head_size * head_count * n_tokens + head_size * head_size * head_count * n_seqs);

    ggml_cgraph* gf = s.expand(out);
    for (ggml_tensor* t : {r, w, k, v, a, b, state}) {
        set_f32(t, std::vector<float>(static_cast<size_t>(ggml_nelements(t)), 0.1f));
    }
    s.compute(gf);
}

} // namespace

int main() {
    test_get_rows();
    test_mul_mat_identity();
    test_add_mul_silu();
    test_swiglu();
    test_rms_norm();
    test_layer_norm();
    test_sigmoid();
    test_tanh();
    test_sin_cos();
    test_relu();
    test_leaky_relu();
    test_step();
    test_group_norm();
    test_cumsum();
    test_glu();
    test_conv_1d_dw();
    test_depthwise_conv_transpose_1d_via_composition();
    test_reshape_permute_cont();
    test_reshape_infers_minus_one_dim();
    test_view();
    test_view_of_permuted_parent_with_unit_leading_axis();
    test_view_out_of_bounds_still_throws();
    test_unknown_op_throws();
    test_rope_identity_at_position_zero();
    test_conv_1d();
    test_conv_2d();
    test_conv_2d_dw();
    test_conv_transpose_1d();
    test_conv_transpose_2d();
    test_pool_1d();
    test_pool_2d();
    test_gelu();
    test_attention_without_kv_cache();
    test_rel_shift();
    test_rel_to_abs_shaw();
    test_abs_to_rel_shaw();
    test_rel_pos_attention_shaw();
    test_rel_pos_attention();
    test_sub_div_scale();
    test_sqr_sqrt_log();
    test_atan2();
    test_exp();
    test_floor();
    test_sum_rows();
    test_pad_1d();
    test_pad_1d_reflect();
    test_concat();
    test_repeat();
    test_interpolate_1d();
    test_rq_spline_inverse();
    test_rq_spline_inverse_outside_tail_bound();
    test_wn();
    test_residual_coupling_layer_reverse();
    test_flip_via_get_rows();
    test_elementwise_affine_reverse();
    test_dds_conv();
    test_conv_flow_reverse();
    test_sdp_reverse_assembly();
    test_hifigan_generator();
    test_text_encoder_assembly();
    test_ssm_conv();
    test_ssm_scan();
    test_rwkv_wkv6();
    test_rwkv_wkv7();

    LOOM_TEST_REPORT_AND_RETURN();
}
