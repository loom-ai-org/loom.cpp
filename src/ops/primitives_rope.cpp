#include "loom/ops/primitive_registry.h"
#include "loom/loom_errors.h"

#include <nlohmann/json.hpp>

namespace loom {
namespace {

using Json = nlohmann::json;

std::vector<ggml_tensor*> op_rope(PrimitiveContext& pc, const std::vector<ggml_tensor*>& in, const Json& attrs) {
    if (in.size() < 2 || in.size() > 3) {
        throw SchemaError("ROPE expects 2 or 3 inputs (a, positions[, freq_factors]), got " + std::to_string(in.size()));
    }
    ggml_tensor* a = in[0];
    ggml_tensor* pos = in[1];
    ggml_tensor* freq_factors = in.size() == 3 ? in[2] : nullptr;

    const int n_dims     = static_cast<int>(resolve_attr_int(attrs, "n_dims", pc.symbols));
    const int mode       = static_cast<int>(resolve_attr_int(attrs, "mode", pc.symbols));
    const int n_ctx_orig = static_cast<int>(resolve_attr_int(attrs, "n_ctx_orig", pc.symbols));
    const float freq_base   = static_cast<float>(resolve_attr_number(attrs, "freq_base", pc.symbols));
    const float freq_scale  = static_cast<float>(resolve_attr_number(attrs, "freq_scale", pc.symbols));
    const float ext_factor  = static_cast<float>(resolve_attr_number(attrs, "ext_factor", pc.symbols));
    const float attn_factor = static_cast<float>(resolve_attr_number(attrs, "attn_factor", pc.symbols));
    const float beta_fast   = static_cast<float>(resolve_attr_number(attrs, "beta_fast", pc.symbols));
    const float beta_slow   = static_cast<float>(resolve_attr_number(attrs, "beta_slow", pc.symbols));

    return {ggml_rope_ext(pc.ctx, a, pos, freq_factors, n_dims, mode, n_ctx_orig,
                           freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)};
}

} // namespace

LOOM_REGISTER_OP(ROPE, op_rope)

} // namespace loom
