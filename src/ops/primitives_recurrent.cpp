#include "loom/loom_errors.h"
#include "loom/ops/primitive_registry.h"

#include <nlohmann/json.hpp>

// SSM (Mamba/SSD) and RWKV primitives (EXPORT-IMPROVEMENT-BACKLOG.md item 4). ggml already has real,
// dedicated compute kernels for these -- ggml_ssm_conv/ggml_ssm_scan (Mamba's selective-scan recurrence),
// ggml_rwkv_wkv6/ggml_rwkv_wkv7 (RWKV's linear-attention-style recurrence) -- unlike LSTM/GRU, which have
// no native ggml op at all and need a hand-composed per-timestep cell (see recurrent.py). Registered here
// as thin, mechanical wraps -- no exporter-side MIL-graph auto-detection exists for these yet, and none is
// attempted: unlike torch.nn.LSTM/GRU (which coremltools' torch frontend maps directly to a single opaque
// MIL op), Mamba/RWKV aren't torch built-ins, so a traced HF model using them would either fail to trace
// (a custom CUDA kernel call) or unroll into a large sequence of raw elementwise ops with no reliable
// "this is an SSM scan" signal to pattern-match on -- and no model on the current roadmap needs this yet.
// These primitives exist so a FUTURE bespoke hand-authored driver (the same precedent Kokoro's own STFT
// path already set) has real ggml-backed ops available the moment a concrete model needs them, without
// that being blocked on inventing MIL-graph detection for a pattern nobody has traced yet.

namespace loom {
namespace {

using Json = nlohmann::json;
using Inputs = std::vector<ggml_tensor*>;
using Outputs = std::vector<ggml_tensor*>;

void expect_n_inputs(const char* op, const Inputs& in, size_t n) {
    if (in.size() != n) {
        throw SchemaError(std::string(op) + " expects " + std::to_string(n) + " input(s), got " + std::to_string(in.size()));
    }
}

Outputs op_ssm_conv(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SSM_CONV", in, 2);
    return {ggml_ssm_conv(pc.ctx, /*sx=*/in[0], /*c=*/in[1])};
}

Outputs op_ssm_scan(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SSM_SCAN", in, 7);
    return {ggml_ssm_scan(pc.ctx, /*s=*/in[0], /*x=*/in[1], /*dt=*/in[2], /*A=*/in[3], /*B=*/in[4], /*C=*/in[5],
                           /*ids=*/in[6])};
}

Outputs op_rwkv_wkv6(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("RWKV_WKV6", in, 6);
    return {ggml_rwkv_wkv6(pc.ctx, /*k=*/in[0], /*v=*/in[1], /*r=*/in[2], /*tf=*/in[3], /*td=*/in[4],
                            /*state=*/in[5])};
}

Outputs op_rwkv_wkv7(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("RWKV_WKV7", in, 7);
    return {ggml_rwkv_wkv7(pc.ctx, /*r=*/in[0], /*w=*/in[1], /*k=*/in[2], /*v=*/in[3], /*a=*/in[4], /*b=*/in[5],
                            /*state=*/in[6])};
}

} // namespace

LOOM_REGISTER_OP(SSM_CONV, op_ssm_conv)
LOOM_REGISTER_OP(SSM_SCAN, op_ssm_scan)
LOOM_REGISTER_OP(RWKV_WKV6, op_rwkv_wkv6)
LOOM_REGISTER_OP(RWKV_WKV7, op_rwkv_wkv7)

} // namespace loom
