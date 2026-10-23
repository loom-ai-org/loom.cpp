#include "loom/core/kv_cache.h"
#include "loom/loom_errors.h"
#include "loom/ops/primitive_registry.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace loom {
namespace {

using Json = nlohmann::json;

// The whole-sequence form: every query against every key, the mask deciding which of them count.
ggml_tensor* dense_attention(ggml_context* ctx, ggml_tensor* qp, ggml_tensor* kp, ggml_tensor* vt,
                             ggml_tensor* kq_mask, float scale) {
    ggml_tensor* kq = ggml_mul_mat(ctx, kp, qp); // [n_kv, n_tokens, n_head]
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(ctx, kq, kq_mask, scale, /*max_bias=*/0.0f);
    return ggml_mul_mat(ctx, vt, kq);            // [n_embd_head_v, n_tokens, n_head]
}

// Queries per block of a windowed node. Pi Zero W, Pocket-TTS 2m (window 50 over 960 positions), the whole
// synthesis: 16 / 24 / 32 / 48 / 64 = 9.56 / 9.62 / 9.61 / 9.67-9.70 / 9.73-9.75 s. Smaller blocks compute
// less of the band's masked corner and run more nodes; 32 is as fast as 16 with half of them.
constexpr int64_t kWindowBlock = 32;

// `$LOOM_ATTENTION_WINDOW=0` runs a windowed node densely, for A/B runs; `$LOOM_ATTENTION_WINDOW_BLOCK`
// overrides the query block. Both read once per process.
bool window_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("LOOM_ATTENTION_WINDOW");
        return !(e && std::strcmp(e, "0") == 0);
    }();
    return on;
}

int64_t window_block() {
    static const int64_t block = [] {
        const char* e = std::getenv("LOOM_ATTENTION_WINDOW_BLOCK");
        const long v = e ? std::strtol(e, nullptr, 10) : 0;
        return static_cast<int64_t>(v > 0 ? v : kWindowBlock);
    }();
    return block;
}

// attrs["window"] (P4.41, ADR-085): a promise from the export that this node's mask is -inf wherever
// a key is not in the query's causal window, `q - window < k <= q`. Nothing in the mask's VALUES is
// trusted beyond that: inside the band they are applied exactly as the dense form applies them.
int64_t attention_window(const Json& attrs, PrimitiveContext& pc, bool use_cache) {
    if (!attrs.contains("window")) return 0;
    const int64_t window = resolve_attr_int(attrs, "window", pc.symbols);
    if (window <= 0) throw SchemaError("ATTENTION: window must be positive, got " + std::to_string(window));
    if (use_cache) {
        // A cached step's keys are the cache's cells, whose positions are the driver's business; the
        // only caller so far (Mimi's decoder, a single uncached pass) has no use for the pairing.
        throw SchemaError("ATTENTION: window is only defined with kv_cache=false");
    }
    return window_enabled() ? window : 0;
}

// The windowed form: each block of `block` queries against only the keys its rows can see, the
// keys [q0 - window + 1, q0 + nq). Mimi's decoder transformer (window 50 over 960 positions for 4.8 s
// of audio) spent 6.6 s of a 19 s Pi Zero synthesis on the dense form, ~95% of it on pairs its mask
// zeroes (P4.41).
//
// Every in-band score, mask value and softmax input is the dense form's, and an out-of-band column
// contributes an exact 0 there (exp of -inf, or of -1e30 after the max), so the softmax rows agree.
// The A @ V products are not bit-identical: K shrinks from n_kv to the band, which regroups the sums.
ggml_tensor* banded_attention(ggml_context* ctx, ggml_tensor* qp, ggml_tensor* kp, ggml_tensor* vt,
                              ggml_tensor* kq_mask, float scale, int64_t window) {
    const int64_t n_tokens = qp->ne[1];
    const int64_t n_kv = kp->ne[1];
    if (n_kv != n_tokens) {
        throw SchemaError("ATTENTION: a windowed node attends over its own keys (n_kv " + std::to_string(n_kv) +
                          " != n_tokens " + std::to_string(n_tokens) + ")");
    }
    const int64_t block = window_block();
    std::vector<ggml_tensor*> parts;
    for (int64_t q0 = 0; q0 < n_tokens; q0 += block) {
        const int64_t nq = std::min(block, n_tokens - q0);
        const int64_t k0 = std::max<int64_t>(0, q0 - window + 1);
        const int64_t nk = q0 + nq - k0;
        // Q's block COPIED to contiguous rows: ggml hands a batched mul_mat to tinyBLAS only when src1 is
        // contiguous, and the strided view ran the per-element dot instead -- 33 MMAC/s against A @ V's
        // 280 on a Pi Zero W, 1.6 s of 2.1 (P4.41). K's block stays a view: src0's rows may be strided.
        ggml_tensor* qb = ggml_cont(ctx, ggml_view_3d(ctx, qp, qp->ne[0], nq, qp->ne[2], qp->nb[1], qp->nb[2],
                                                      q0 * qp->nb[1]));
        ggml_tensor* kb = ggml_view_3d(ctx, kp, kp->ne[0], nk, kp->ne[2], kp->nb[1], kp->nb[2], k0 * kp->nb[1]);
        ggml_tensor* kq = ggml_mul_mat(ctx, kb, qb); // [nk, nq, n_head]
        ggml_prec_set_acc(kq, GGML_PREC_F32);
        // ggml_soft_max_ext takes a contiguous mask only, so the block's [nk, nq] corner is copied out.
        ggml_tensor* mb = ggml_cont(ctx, ggml_view_2d(ctx, kq_mask, nk, nq, kq_mask->nb[1],
                                                      q0 * kq_mask->nb[1] + k0 * kq_mask->nb[0]));
        kq = ggml_soft_max_ext(ctx, kq, mb, scale, /*max_bias=*/0.0f);
        ggml_tensor* vb = ggml_view_3d(ctx, vt, nk, vt->ne[1], vt->ne[2], vt->nb[1], vt->nb[2], k0 * vt->nb[0]);
        parts.push_back(ggml_mul_mat(ctx, vb, kq)); // [n_embd_head_v, nq, n_head]
    }
    // Pairwise, so each level copies the result once: a left fold copies its growing prefix per block.
    while (parts.size() > 1) {
        std::vector<ggml_tensor*> next;
        for (size_t i = 0; i + 1 < parts.size(); i += 2) next.push_back(ggml_concat(ctx, parts[i], parts[i + 1], 1));
        if (parts.size() % 2) next.push_back(parts.back());
        parts.swap(next);
    }
    return parts[0];
}

// Composite (non-flash) scaled dot-product attention. Mirrors llama.cpp's
// llm_graph_context::build_attn_mha non-flash / !v_trans branch (llama-graph.cpp:2384-2517) -- see the
// implementation plan for why composite-over-flash was chosen for milestone 1 (exact fp32 verification
// against a numpy reference, no forced F16 K/V cast).
//
// Inputs: q [n_embd_head_k, n_head, n_tokens], k/v [n_embd_head_k(_v), n_head_kv, n_tokens] (this step's
// fresh, already-RoPE'd-if-applicable projections for a causal decoder; just the current forward pass's
// projections for a non-autoregressive encoder), kq_mask [n_kv, n_tokens]. n_head_kv/n_embd_head_k(_v)
// are read directly off k/v/q's shapes rather than duplicated in attrs.
//
// attrs["kv_cache"] (bool, default true) selects between the two: true (the default, matching every
// milestone-1 LLM topology, which never sets this attr) reads/writes a persistent KvCache for causal
// autoregressive decoding; false skips the cache entirely and attends directly over this call's own k/v
// (n_kv == n_tokens) -- the shape a non-autoregressive encoder needs, where the whole (fixed-length)
// sequence is attended over in one shot and there's no "past" to persist.
//
// attrs["window"] (int, kv_cache=false only) declares a causal sliding window and computes only the band
// it allows -- see banded_attention.
std::vector<ggml_tensor*> op_attention(PrimitiveContext& pc, const std::vector<ggml_tensor*>& in, const Json& attrs) {
    if (in.size() != 4) {
        throw SchemaError("ATTENTION expects 4 inputs (q, k, v, kq_mask), got " + std::to_string(in.size()));
    }

    ggml_tensor* q = in[0];
    ggml_tensor* k = in[1];
    ggml_tensor* v = in[2];
    ggml_tensor* kq_mask = in[3];

    const bool use_cache = attrs.value("kv_cache", true);
    const float scale = static_cast<float>(resolve_attr_number(attrs, "scale", pc.symbols));

    const int64_t n_embd_head_k = k->ne[0];
    const int64_t n_head_kv = k->ne[1];
    const int64_t n_embd_head_v = v->ne[0];

    ggml_tensor* k_cache;
    ggml_tensor* v_cache;

    if (use_cache) {
        if (!pc.kv_cache) {
            throw SchemaError("ATTENTION: no KvCache was provided to GraphBuilder, but the topology uses ATTENTION with kv_cache=true");
        }
        if (!pc.kv_cells) {
            throw SchemaError("ATTENTION: a KvCache was provided but no cell-index tensor was, which "
                               "means the axes this graph was built for bind no 'n_past' -- a cached "
                               "attention has nowhere to write without one");
        }
        const uint32_t layer = static_cast<uint32_t>(resolve_attr_int(attrs, "layer", pc.symbols));
        const int64_t n_embd_k_gqa = n_embd_head_k * n_head_kv;
        const int64_t n_embd_v_gqa = n_embd_head_v * n_head_kv;

        const auto n_tokens = static_cast<uint32_t>(std::llround(pc.symbols.get("n_tokens")));
        const auto n_kv_u32 = static_cast<uint32_t>(std::llround(pc.symbols.get("n_kv")));

        // Scatter this step's fresh K/V into the persistent cache at the cells `pc.kv_cells` names --
        // [n_past, n_past + n_tokens) for this single-sequence cache, but as a runtime VALUE rather than
        // a build-time offset, which is what lets a decode loop reuse one graph (BACKLOG.md P4.0.15).
        // These writes have no data-dependency edge to anything downstream (the cache read below is a
        // plain memory view, not a consumer of the set_rows node), so the caller must route them through
        // side_effects to guarantee they're included in the graph and executed before the read.
        ggml_tensor* k_flat = ggml_reshape_2d(pc.ctx, k, n_embd_k_gqa, n_tokens);
        ggml_tensor* v_flat = ggml_reshape_2d(pc.ctx, v, n_embd_v_gqa, n_tokens);
        ggml_tensor* k_write = pc.kv_cache->write_k(pc.ctx, k_flat, layer, pc.kv_cells);
        ggml_tensor* v_write = pc.kv_cache->write_v(pc.ctx, v_flat, layer, pc.kv_cells);
        if (pc.side_effects) {
            pc.side_effects->push_back(k_write);
            pc.side_effects->push_back(v_write);
        }

        // Read back the whole valid prefix [0, n_kv) -- both prior history and what was just written.
        // `n_kv` is the BUCKET-PADDED length when GraphBuilder is bucketing (P4.0.15), so this view can
        // span cells past the real end of the sequence. Those cells contribute exactly zero because the
        // mask says -inf there and a -inf column's softmax weight is 0 -- never because their contents
        // are zero, which they are not after a second generation reuses the cache.
        k_cache = ggml_reshape_3d(pc.ctx, pc.kv_cache->read_k(pc.ctx, layer, n_kv_u32), n_embd_head_k, n_head_kv, n_kv_u32);
        v_cache = ggml_reshape_3d(pc.ctx, pc.kv_cache->read_v(pc.ctx, layer, n_kv_u32), n_embd_head_v, n_head_kv, n_kv_u32);
    } else {
        // No persistent state: attend directly over this call's own k/v, already in the right shape
        // (n_kv == n_tokens implicitly, via k_cache/v_cache's own ne[2]).
        k_cache = k;
        v_cache = v;
    }

    // Head-major layout, then QK^T (broadcasting n_head_kv -> n_head for GQA) in F32 precision.
    ggml_tensor* qp = ggml_permute(pc.ctx, q, 0, 2, 1, 3);        // [n_embd_head_k, n_tokens, n_head]
    ggml_tensor* kp = ggml_permute(pc.ctx, k_cache, 0, 2, 1, 3);  // [n_embd_head_k, n_kv, n_head_kv]
    ggml_tensor* vp = ggml_permute(pc.ctx, v_cache, 0, 2, 1, 3);  // [n_embd_head_v, n_kv, n_head_kv]

    // KvCache never pre-transposes V's storage (no v_trans optimization), so this branch always runs.
    ggml_tensor* vt = ggml_cont(pc.ctx, ggml_transpose(pc.ctx, vp)); // [n_kv, n_embd_head_v, n_head_kv]

    const int64_t window = attention_window(attrs, pc, use_cache);
    ggml_tensor* kqv = window > 0 ? banded_attention(pc.ctx, qp, kp, vt, kq_mask, scale, window)
                                  : dense_attention(pc.ctx, qp, kp, vt, kq_mask, scale);
    // kqv: [n_embd_head_v, n_tokens, n_head]

    ggml_tensor* cur = ggml_permute(pc.ctx, kqv, 0, 2, 1, 3); // [n_embd_head_v, n_head, n_tokens]
    cur = ggml_cont_2d(pc.ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]); // [n_embd, n_tokens]

    return {cur};
}

// Transformer-XL / Conformer relative-shift trick (verbatim algorithm confirmed against NeMo's
// RelPositionMultiHeadAttention.rel_shift and cross-checked numerically against actual PyTorch
// execution on a small example -- see the implementation plan). `x` has ne=[pos_len, qlen, n_head, 1].
// Despite looking like a transpose, every step here is a pure flat-memory reinterpretation (pad, then
// three reshape/view calls) -- verified against PyTorch's own pad+view+slice+view sequence
// element-for-element on a qlen=2,pos_len=3 example before trusting this translation.
ggml_tensor* rel_shift(ggml_context* ctx, ggml_tensor* x) {
    const int64_t pos_len = x->ne[0];
    const int64_t qlen = x->ne[1];
    const int64_t n_head = x->ne[2];

    ggml_tensor* padded = ggml_pad_ext(ctx, x, /*lp0=*/1, /*rp0=*/0, 0, 0, 0, 0, 0, 0); // [pos_len+1, qlen, n_head]
    ggml_tensor* reshaped = ggml_reshape_3d(ctx, padded, qlen, pos_len + 1, n_head);     // reinterpret, not a transpose
    ggml_tensor* sliced = ggml_view_3d(ctx, reshaped, qlen, pos_len, n_head,
                                       reshaped->nb[1], reshaped->nb[2], /*offset=*/reshaped->nb[1]);
    ggml_tensor* sliced_cont = ggml_cont(ctx, sliced);
    return ggml_reshape_3d(ctx, sliced_cont, pos_len, qlen, n_head); // reinterpret back, not a transpose
}

std::vector<ggml_tensor*> op_rel_shift(PrimitiveContext& pc, const std::vector<ggml_tensor*>& in, const Json&) {
    // Exposed as its own primitive purely so the trick above can be unit-tested in isolation
    // (tests/test_primitive_registry.cpp) before trusting it inside REL_POS_ATTENTION -- not meant to be
    // used directly in a real topology.
    if (in.size() != 1) {
        throw SchemaError("REL_SHIFT expects 1 input, got " + std::to_string(in.size()));
    }
    return {rel_shift(pc.ctx, in[0])};
}

// Conformer's relative-positional multi-head self-attention (Transformer-XL style: Dai et al. 2019,
// as adapted by the Conformer paper, Gulati et al. 2020 arXiv:2005.08100). Distinct from ATTENTION
// (which has no positional-bias concept at all), used by Conformer-CTC's encoder layers.
//
// Inputs: q [head_dim, n_head, n_tokens], k/v [head_dim, n_head, n_tokens] (self-attention: same source,
// no KV cache -- a single non-autoregressive encoder pass), p [head_dim, n_head, n_pos] (the shared
// sinusoidal positional embedding, already run through this layer's own linear_pos projection as a
// separate MUL_MAT node upstream -- kept out of this primitive the same way Q/K/V projections are kept
// as separate MUL_MAT nodes ahead of ATTENTION), pos_bias_u/pos_bias_v [head_dim, n_head] (this layer's
// own untied bias weights), kq_mask [n_tokens, n_tokens]. Output is the pre-linear_out attention
// context, same convention as ATTENTION's output (the output projection is a separate MUL_MAT node
// after this, same as ATTENTION).
std::vector<ggml_tensor*> op_rel_pos_attention(PrimitiveContext& pc, const std::vector<ggml_tensor*>& in, const Json& attrs) {
    if (in.size() != 7) {
        throw SchemaError("REL_POS_ATTENTION expects 7 inputs (q, k, v, p, pos_bias_u, pos_bias_v, kq_mask), got " +
                           std::to_string(in.size()));
    }
    ggml_tensor* q = in[0];
    ggml_tensor* k = in[1];
    ggml_tensor* v = in[2];
    ggml_tensor* p = in[3];
    ggml_tensor* pos_bias_u = in[4];
    ggml_tensor* pos_bias_v = in[5];
    ggml_tensor* kq_mask = in[6];

    const int64_t head_dim = q->ne[0];
    const float scale = static_cast<float>(resolve_attr_number(attrs, "scale", pc.symbols));

    // q + {u,v} broadcasts pos_bias_{u,v} (ne=[head_dim,n_head]) over q's n_tokens axis (ne[2]).
    ggml_tensor* q_bias_u = ggml_add(pc.ctx, q, pos_bias_u);
    ggml_tensor* q_bias_v = ggml_add(pc.ctx, q, pos_bias_v);

    ggml_tensor* qu = ggml_permute(pc.ctx, q_bias_u, 0, 2, 1, 3); // [head_dim, n_tokens, n_head]
    ggml_tensor* qv = ggml_permute(pc.ctx, q_bias_v, 0, 2, 1, 3);
    ggml_tensor* kp = ggml_permute(pc.ctx, k, 0, 2, 1, 3);        // [head_dim, n_tokens, n_head]
    ggml_tensor* pp = ggml_permute(pc.ctx, p, 0, 2, 1, 3);        // [head_dim, n_pos, n_head]

    ggml_tensor* matrix_ac = ggml_mul_mat(pc.ctx, kp, qu); // [n_tokens(kv), n_tokens(q), n_head]
    ggml_prec_set_acc(matrix_ac, GGML_PREC_F32);

    ggml_tensor* matrix_bd_raw = ggml_mul_mat(pc.ctx, pp, qv); // [n_pos, n_tokens(q), n_head]
    ggml_prec_set_acc(matrix_bd_raw, GGML_PREC_F32);
    ggml_tensor* matrix_bd_shifted = rel_shift(pc.ctx, matrix_bd_raw); // [n_pos, n_tokens(q), n_head]
    // Truncate to matrix_ac's kv length (matrix_bd[:, :, :matrix_ac.size(-1)] in the PyTorch original).
    ggml_tensor* matrix_bd = ggml_view_3d(pc.ctx, matrix_bd_shifted, matrix_ac->ne[0], matrix_bd_shifted->ne[1],
                                          matrix_bd_shifted->ne[2], matrix_bd_shifted->nb[1], matrix_bd_shifted->nb[2], 0);

    ggml_tensor* scores = ggml_add(pc.ctx, matrix_ac, ggml_cont(pc.ctx, matrix_bd));
    scores = ggml_soft_max_ext(pc.ctx, scores, kq_mask, scale, /*max_bias=*/0.0f);

    ggml_tensor* vp = ggml_permute(pc.ctx, v, 0, 2, 1, 3);
    ggml_tensor* vt = ggml_cont(pc.ctx, ggml_transpose(pc.ctx, vp)); // [n_tokens(kv), head_dim, n_head]
    ggml_tensor* kqv = ggml_mul_mat(pc.ctx, vt, scores);             // [head_dim, n_tokens(q), n_head]

    ggml_tensor* cur = ggml_permute(pc.ctx, kqv, 0, 2, 1, 3); // [head_dim, n_head, n_tokens]
    cur = ggml_cont_2d(pc.ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]); // [n_embd, n_tokens]

    (void)head_dim; // kept for readability/documentation of the shape contract, not otherwise needed
    return {cur};
}

} // namespace

LOOM_REGISTER_OP(ATTENTION, op_attention)
LOOM_REGISTER_OP(REL_SHIFT, op_rel_shift)
LOOM_REGISTER_OP(REL_POS_ATTENTION, op_rel_pos_attention)

} // namespace loom
