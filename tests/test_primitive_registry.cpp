// Exercises each milestone-1 primitive in isolation (bypassing GraphTopology/GraphBuilder entirely) by
// invoking PrimitiveRegistry::instance().get(op) directly against hand-built input tensors with known
// values, and checking the result against a hand-computed expectation.

#include "ggml_test_helpers.h"
#include "test_util.h"

#include "loom/loom.h"

#include <nlohmann/json.hpp>

#include <cmath>

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

} // namespace

int main() {
    test_get_rows();
    test_mul_mat_identity();
    test_add_mul_silu();
    test_swiglu();
    test_rms_norm();
    test_layer_norm();
    test_sigmoid();
    test_relu();
    test_glu();
    test_conv_1d_dw();
    test_reshape_permute_cont();
    test_reshape_infers_minus_one_dim();
    test_view();
    test_unknown_op_throws();
    test_rope_identity_at_position_zero();
    test_conv_1d();
    test_conv_2d();
    test_conv_transpose_1d();
    test_conv_transpose_2d();
    test_pool_1d();
    test_pool_2d();
    test_gelu();
    test_attention_without_kv_cache();
    test_rel_shift();
    test_rel_pos_attention();
    test_sub_div_scale();
    test_sqr_sqrt_log();
    test_sum_rows();
    test_pad_1d();

    LOOM_TEST_REPORT_AND_RETURN();
}
