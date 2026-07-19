"""Shared TopologyBuilder + small composable builders for SupertonicTTS v2 conversion scripts (mirrors
this project's usual per-tool TopologyBuilder idiom, e.g. tools/convert_kokoro/*.py's own copies).
"""
from __future__ import annotations

import numpy as np
import torch


def to_f32(t):
    return t.detach().cpu().numpy().astype(np.float32)


class TopologyBuilder:
    def __init__(self):
        self.nodes = []
        self.weights = {}
        self._counter = 0

    def _fresh(self, hint):
        self._counter += 1
        return f"{hint}_{self._counter}"

    def node(self, op, inputs, attrs=None, out_hint="t", name=None):
        out = name if name is not None else self._fresh(out_hint)
        entry = {"op": op, "inputs": list(inputs), "outputs": [out]}
        if attrs:
            entry["attrs"] = attrs
        self.nodes.append(entry)
        return out

    def weight(self, name, array):
        arr = np.asarray(array)
        if name in self.weights and self.weights[name].shape != arr.shape:
            raise ValueError(f"weight {name!r} already registered with a different shape")
        self.weights[name] = arr
        return name

    def topology(self, inputs, output, nodes=None):
        return {"version": 1, "inputs": inputs, "output": output, "nodes": nodes if nodes is not None else self.nodes}


def write_gguf(path, topology, weights, architecture):
    from gguf import GGUFWriter
    import json

    w = GGUFWriter(str(path), architecture)
    w.add_string("model.graph_topology", json.dumps(topology))
    for name, arr in weights.items():
        w.add_tensor(name, arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {path}, {len(weights)} weights")


def add_replicate_pad(tb, x, lp, rp, channels, seq_len_expr, out_hint):
    """x: [T,channels] (CONV_1D-family convention, T=ne[0]). Replicate ("edge") padding: prepend `lp`
    copies of the FIRST row, append `rp` copies of the LAST row -- ggml has no native replicate-pad op
    (only zero-pad/reflect/circular), so this is a VIEW(boundary row)+REPEAT+CONCAT composition, the same
    "materialize the broadcast, then CONCAT" idiom REPEAT was built for (see StyleTTS2's diffusion
    sampler). `lp`/`rp` are static ints (kernel_size/dilation always known at conversion time).
    """
    pieces = []
    if lp > 0:
        first_row = tb.node("VIEW", [x], {"shape": [1, channels], "offset": 0}, f"{out_hint}_first")
        pieces.append(tb.node("REPEAT", [first_row], {"shape": [lp, channels]}, f"{out_hint}_lpad"))
    pieces.append(x)
    if rp > 0:
        # x has ne=[T,channels] (T=ne[0], the FASTEST axis) -- the t=(T-1) column is just `(T-1)` FLOATS
        # in, not `(T-1)*channels` (that formula would stride by whole channel-blocks, landing on a
        # completely wrong element -- a real bug caught here via a numerical mismatch, not by
        # inspection: reusing VIEW's own default nb1 keeps consecutive "columns" of this [1,channels]
        # slice correctly channel-strided, same as the t=0 case above).
        last_row = tb.node("VIEW", [x], {"shape": [1, channels], "offset": f"({seq_len_expr}-1)*4"},
                            f"{out_hint}_last")
        pieces.append(tb.node("REPEAT", [last_row], {"shape": [rp, channels]}, f"{out_hint}_rpad"))
    if len(pieces) == 1:
        return x
    out = pieces[0]
    for p in pieces[1:]:
        out = tb.node("CONCAT", [out, p], {"dim": 0}, f"{out_hint}_cat")
    return out


def add_convnext_block(tb, x, prefix, sd, sd_prefix, dim, interm_dim, kernel_size, dilation, causal,
                        seq_len_expr, out_hint):
    """x: [T,dim] (CONV_1D convention). Returns [T,dim]. Mirrors ConvNextBlock.forward exactly:
    replicate-pad -> depthwise conv -> LayerNorm(channel-first) -> pointwise conv 1 -> GELU ->
    pointwise conv 2 -> gamma scale -> residual add. `causal`: True pads (k-1)*dilation on the LEFT
    only; False pads symmetrically ((k-1)*dilation//2 each side).
    """
    def p(name):
        return f"{sd_prefix}.{name}" if sd_prefix else name

    pad_total = (kernel_size - 1) * dilation
    lp, rp = (pad_total, 0) if causal else (pad_total // 2, pad_total // 2)
    out_len_expr = seq_len_expr  # 'same' padding (symmetric OR causal-left-only both preserve length)

    residual = x
    padded = add_replicate_pad(tb, x, lp, rp, dim, seq_len_expr, f"{out_hint}_pad")

    dw_w = tb.weight(f"{prefix}.dwconv.weight", to_f32(sd[p("dwconv.weight")]))  # (dim,1,K) -> ggml [K,1,dim]
    dw_b = tb.weight(f"{prefix}.dwconv.bias", to_f32(sd[p("dwconv.bias")]))
    padded_len_expr = f"({seq_len_expr}+{lp}+{rp})"
    dw3 = tb.node("RESHAPE", [padded], {"shape": [padded_len_expr, dim, 1]}, f"{out_hint}_dw_in3")
    h = tb.node("CONV_1D_DW", [dw_w, dw3], {"s0": 1, "p0": 0, "d0": dilation}, f"{out_hint}_dw_raw")
    h = tb.node("RESHAPE", [h], {"shape": [out_len_expr, dim]}, f"{out_hint}_dw_2d")
    h = tb.node("ADD", [h, tb.node("RESHAPE", [dw_b], {"shape": [1, dim]}, f"{out_hint}_dw_bias_r")],
                None, f"{out_hint}_dw_biased")

    # LayerNorm normalizes over CHANNELS (real: x.transpose(1,2) -> LayerNorm(dim) -> transpose back) --
    # our [T,dim] tensor already has dim at ne[1], not ne[0], so cross to channel-first first.
    hp_ = tb.node("PERMUTE", [h], {"axes": [1, 0, 2, 3]}, f"{out_hint}_ct_p")
    hc = tb.node("CONT", [hp_], None, f"{out_hint}_ct")  # [dim, T]
    normed = tb.node("LAYER_NORM", [hc], {"eps": 1e-6}, f"{out_hint}_ln_normed")
    g = tb.weight(f"{prefix}.norm.gamma", to_f32(sd[p("norm.weight")]))
    b = tb.weight(f"{prefix}.norm.beta", to_f32(sd[p("norm.bias")]))
    normed = tb.node("MUL", [normed, g], None, f"{out_hint}_ln_mul")
    normed = tb.node("ADD", [normed, b], None, f"{out_hint}_ln_out")
    normed_p = tb.node("PERMUTE", [normed], {"axes": [1, 0, 2, 3]}, f"{out_hint}_tc_p")
    h = tb.node("CONT", [normed_p], None, f"{out_hint}_tc")  # back to [T,dim]

    pw1_w = tb.weight(f"{prefix}.pwconv1.weight", to_f32(sd[p("pwconv1.weight")]))  # (interm,dim,1)
    pw1_b = tb.weight(f"{prefix}.pwconv1.bias", to_f32(sd[p("pwconv1.bias")]))
    h3 = tb.node("RESHAPE", [h], {"shape": [out_len_expr, dim, 1]}, f"{out_hint}_pw1_in3")
    h = tb.node("CONV_1D", [pw1_w, h3], {"s0": 1, "p0": 0, "d0": 1}, f"{out_hint}_pw1_raw")
    h = tb.node("RESHAPE", [h], {"shape": [out_len_expr, interm_dim]}, f"{out_hint}_pw1_2d")
    h = tb.node("ADD", [h, tb.node("RESHAPE", [pw1_b], {"shape": [1, interm_dim]}, f"{out_hint}_pw1_bias_r")],
                None, f"{out_hint}_pw1_biased")
    h = tb.node("GELU", [h], None, f"{out_hint}_gelu")

    pw2_w = tb.weight(f"{prefix}.pwconv2.weight", to_f32(sd[p("pwconv2.weight")]))  # (dim,interm,1)
    pw2_b = tb.weight(f"{prefix}.pwconv2.bias", to_f32(sd[p("pwconv2.bias")]))
    h3b = tb.node("RESHAPE", [h], {"shape": [out_len_expr, interm_dim, 1]}, f"{out_hint}_pw2_in3")
    h = tb.node("CONV_1D", [pw2_w, h3b], {"s0": 1, "p0": 0, "d0": 1}, f"{out_hint}_pw2_raw")
    h = tb.node("RESHAPE", [h], {"shape": [out_len_expr, dim]}, f"{out_hint}_pw2_2d")
    h = tb.node("ADD", [h, tb.node("RESHAPE", [pw2_b], {"shape": [1, dim]}, f"{out_hint}_pw2_bias_r")],
                None, f"{out_hint}_pw2_biased")

    gamma = tb.weight(f"{prefix}.gamma", to_f32(sd[p("gamma")]).reshape(dim))  # (1,dim,1) -> (dim,)
    gamma_r = tb.node("RESHAPE", [gamma], {"shape": [1, dim]}, f"{out_hint}_gamma_r")
    h = tb.node("MUL", [h, gamma_r], None, f"{out_hint}_gamma_scaled")

    return tb.node("ADD", [residual, h], None, out_hint)


def add_style_cross_attention_delta(tb, kv_name, q_name, prefix, sd, sd_prefix, dim, stl_dim, n_style,
                                     kv_seq_len_expr, out_hint):
    """`StyleCrossAttention.compute_delta`, real source components.py -- ALWAYS 2 heads in this model
    (confirmed: every real instance -- DP/TTL style encoders, SpeechPrompted{Text}Encoder -- constructs
    with n_heads=2, never varies), hand-unrolled rather than looped since ggml has no generic
    "split-into-heads-along-a-new-axis" primitive and 2 is a fixed constant here, not a runtime symbol.
    `kv_name`: Layout B [dim, T_kv] (matches PyTorch's own (B,T,dim) `kv` argument exactly, byte-identical,
    no transpose). `q_name`: Layout B [stl_dim, n_style] (already stl_dim-wide -- either a baked learnable
    query constant or a previous stage's own delta+query output). Returns delta, Layout B [stl_dim,
    n_style] -- caller decides whether/what to add as residual (real `StyleCrossAttention.forward` adds
    `+q`, but `StyleEncoderCrossAttention`'s own 2nd stage discards the residual entirely, so this
    function intentionally does NOT do the addition itself).

    NOTE: `scale = sqrt(dim)` (the KV feature width), NOT sqrt(head_dim) -- confirmed directly from
    `StyleCrossAttention.__init__`'s own `self.scale = torch.sqrt(torch.tensor(dim, ...))`, a real quirk
    (this project's usual attention scale is 1/sqrt(head_dim)), not a bug to "fix".
    """
    def p(name):
        return f"{sd_prefix}.{name}" if sd_prefix else name

    n_heads = 2
    dh = stl_dim // n_heads
    scale = float(np.sqrt(dim))

    wk = tb.weight(f"{prefix}.W_key.weight", to_f32(sd[p("W_key.weight")]))
    bk = tb.weight(f"{prefix}.W_key.bias", to_f32(sd[p("W_key.bias")]))
    k = tb.node("ADD", [tb.node("MUL_MAT", [wk, kv_name], None, f"{out_hint}_k_mm"), bk], None, f"{out_hint}_k")

    wv = tb.weight(f"{prefix}.W_value.weight", to_f32(sd[p("W_value.weight")]))
    bv = tb.weight(f"{prefix}.W_value.bias", to_f32(sd[p("W_value.bias")]))
    v = tb.node("ADD", [tb.node("MUL_MAT", [wv, kv_name], None, f"{out_hint}_v_mm"), bv], None, f"{out_hint}_v")

    wq = tb.weight(f"{prefix}.W_query.weight", to_f32(sd[p("W_query.weight")]))
    bq = tb.weight(f"{prefix}.W_query.bias", to_f32(sd[p("W_query.bias")]))
    q = tb.node("ADD", [tb.node("MUL_MAT", [wq, q_name], None, f"{out_hint}_q_mm"), bq], None, f"{out_hint}_q_lin")
    q = tb.node("TANH", [q], None, f"{out_hint}_q_tanh")  # [stl_dim, n_style]

    head_outs = []
    for h in range(n_heads):
        off = h * dh * 4
        k_h = tb.node("VIEW", [k], {"shape": [dh, kv_seq_len_expr], "offset": off}, f"{out_hint}_k{h}")
        v_h = tb.node("VIEW", [v], {"shape": [dh, kv_seq_len_expr], "offset": off}, f"{out_hint}_v{h}")
        q_h = tb.node("VIEW", [q], {"shape": [dh, n_style], "offset": off}, f"{out_hint}_q{h}")

        scores = tb.node("MUL_MAT", [k_h, q_h], None, f"{out_hint}_scores{h}")  # [T_kv, n_style]
        scores = tb.node("SCALE", [scores], {"s": 1.0 / scale}, f"{out_hint}_scores_scaled{h}")
        attn = tb.node("SOFTMAX", [scores], None, f"{out_hint}_attn{h}")  # softmax over T_kv (ne[0])

        v_h_t = tb.node("CONT", [tb.node("PERMUTE", [v_h], {"axes": [1, 0, 2, 3]}, f"{out_hint}_v{h}_t_p")],
                         None, f"{out_hint}_v{h}_t")  # [T_kv, dh]
        out_h = tb.node("MUL_MAT", [v_h_t, attn], None, f"{out_hint}_out{h}")  # [dh, n_style]
        head_outs.append(out_h)

    out = tb.node("CONCAT", [head_outs[0], head_outs[1]], {"dim": 0}, f"{out_hint}_heads_cat")  # [stl_dim, n_style]
    ow = tb.weight(f"{prefix}.out_fc.weight", to_f32(sd[p("out_fc.weight")]))
    ob = tb.weight(f"{prefix}.out_fc.bias", to_f32(sd[p("out_fc.bias")]))
    return tb.node("ADD", [tb.node("MUL_MAT", [ow, out], None, f"{out_hint}_outfc_mm"), ob], None, out_hint)


def add_style_encoder_cross_attention(tb, x_name, prefix, sd, sd_prefix, dim, stl_dim, n_style,
                                       kv_seq_len_expr, out_hint):
    """`StyleEncoderCrossAttention.forward`, real source components.py: stage 1 (`attention1`, a
    LEARNABLE baked-constant query) computes `delta1 + query` (`StyleCrossAttention.forward`'s own
    residual, added HERE not inside the delta helper); stage 2 (`attention2`, query=stage-1's own
    output, kv=the SAME original `x_0` again) computes delta2 ONLY (the real code discards its own `+q`
    residual via `_`); final LayerNorm(stl_dim). `x_name`: Layout B [dim, T] (ConvNeXt-stack output,
    already channel-first-crossed by the caller via PERMUTE+CONT -- see e.g. add_dp_style_encoder).
    Returns [stl_dim, n_style].
    """
    def p(name):
        return f"{sd_prefix}.{name}" if sd_prefix else name

    query_const = tb.weight(f"{prefix}.attention1.query",
                             to_f32(sd[p("attention1.query")]).reshape(n_style, stl_dim))  # (1,n_style,stl_dim)->squeeze
    delta1 = add_style_cross_attention_delta(tb, x_name, query_const, f"{prefix}.attention1", sd,
                                              p("attention1"), dim, stl_dim, n_style, kv_seq_len_expr,
                                              f"{out_hint}_delta1")
    stage1_out = tb.node("ADD", [delta1, query_const], None, f"{out_hint}_stage1")  # [stl_dim, n_style]

    delta2 = add_style_cross_attention_delta(tb, x_name, stage1_out, f"{prefix}.attention2", sd,
                                              p("attention2"), dim, stl_dim, n_style, kv_seq_len_expr,
                                              f"{out_hint}_delta2")

    normed = tb.node("LAYER_NORM", [delta2], {"eps": 1e-6}, f"{out_hint}_ln_normed")
    g = tb.weight(f"{prefix}.norm.gamma", to_f32(sd[p("norm.weight")]))
    b = tb.weight(f"{prefix}.norm.beta", to_f32(sd[p("norm.bias")]))
    normed = tb.node("MUL", [normed, g], None, f"{out_hint}_ln_mul")
    return tb.node("ADD", [normed, b], None, out_hint)  # [stl_dim, n_style]


def apply_channel_layer_norm(tb, x_ta, prefix, sd, sd_prefix, dim, eps, seq_len_expr, out_hint):
    """`nn.LayerNorm(dim)` applied to a Layout A [T,dim] tensor (real `_apply_layer_norm`: transpose to
    channel-last, F.layer_norm, transpose back) -- crosses to Layout B [dim,T] (channels=ne[0], what
    LAYER_NORM needs) and back, same boundary-crossing pattern used throughout this project. Returns
    Layout A [T,dim]."""
    def p(name):
        return f"{sd_prefix}.{name}" if sd_prefix else name

    xp = tb.node("PERMUTE", [x_ta], {"axes": [1, 0, 2, 3]}, f"{out_hint}_cb_p")
    xc = tb.node("CONT", [xp], None, f"{out_hint}_cb")  # [dim, T]
    normed = tb.node("LAYER_NORM", [xc], {"eps": eps}, f"{out_hint}_normed")
    g = tb.weight(f"{prefix}.gamma", to_f32(sd[p("weight")]))
    b = tb.weight(f"{prefix}.beta", to_f32(sd[p("bias")]))
    normed = tb.node("MUL", [normed, g], None, f"{out_hint}_mul")
    normed = tb.node("ADD", [normed, b], None, f"{out_hint}_add")
    normed_p = tb.node("PERMUTE", [normed], {"axes": [1, 0, 2, 3]}, f"{out_hint}_ta_p")
    return tb.node("CONT", [normed_p], None, out_hint)  # [T, dim]


def add_feedforward_block(tb, x_ta, prefix, sd, sd_prefix, dim, interm_dim, seq_len_expr, out_hint):
    """`FeedForwardBlock`: Conv1d(dim,interm,1)+ReLU+Conv1d(interm,dim,1), real masking omitted (always a
    no-op here -- single unpadded utterance, same "no real masking" precedent as every other model in
    this project). x_ta: Layout A [T,dim]. Returns Layout A [T,dim]."""
    def p(name):
        return f"{sd_prefix}.{name}" if sd_prefix else name

    w1 = tb.weight(f"{prefix}.conv_1.weight", to_f32(sd[p("conv_1.weight")]))
    b1 = tb.weight(f"{prefix}.conv_1.bias", to_f32(sd[p("conv_1.bias")]))
    x3 = tb.node("RESHAPE", [x_ta], {"shape": [seq_len_expr, dim, 1]}, f"{out_hint}_in3")
    h = tb.node("CONV_1D", [w1, x3], {"s0": 1, "p0": 0, "d0": 1}, f"{out_hint}_c1_raw")
    h = tb.node("RESHAPE", [h], {"shape": [seq_len_expr, interm_dim]}, f"{out_hint}_c1_2d")
    h = tb.node("ADD", [h, tb.node("RESHAPE", [b1], {"shape": [1, interm_dim]}, f"{out_hint}_c1_bias_r")],
                None, f"{out_hint}_c1_biased")
    h = tb.node("RELU", [h], None, f"{out_hint}_relu")

    w2 = tb.weight(f"{prefix}.conv_2.weight", to_f32(sd[p("conv_2.weight")]))
    b2 = tb.weight(f"{prefix}.conv_2.bias", to_f32(sd[p("conv_2.bias")]))
    h3 = tb.node("RESHAPE", [h], {"shape": [seq_len_expr, interm_dim, 1]}, f"{out_hint}_in3b")
    h = tb.node("CONV_1D", [w2, h3], {"s0": 1, "p0": 0, "d0": 1}, f"{out_hint}_c2_raw")
    h = tb.node("RESHAPE", [h], {"shape": [seq_len_expr, dim]}, f"{out_hint}_c2_2d")
    h = tb.node("ADD", [h, tb.node("RESHAPE", [b2], {"shape": [1, dim]}, f"{out_hint}_c2_bias_r")], None, out_hint)
    return h


def add_multihead_relative_attention(tb, x_ta, prefix, sd, sd_prefix, dim, n_heads, window_size, table,
                                      seq_len_expr, seq_len_int, out_hint):
    """`MultiHeadRelativeAttention` (Shaw et al., real source: components.py), reusing VITS's own
    REL_POS_ATTENTION_SHAW primitive family directly -- confirmed the SAME algorithm on a small example
    (see BACKLOG.md's SupertonicTTS entry). x_ta: Layout A [T,dim]. `table`: a callable
    `(name, n_buckets_or_2Tminus1_array) -> weight_name`-style HOST helper isn't used here directly;
    instead `emb_rel_k`/`emb_rel_v` are passed in ALREADY windowed to `seq_len_int` (see
    `get_relative_embeddings`, imported from tools/convert_piper_vits/vits_common.py by the caller) --
    this function only wires the fixed-length-table case (matching a standalone conversion's own single
    target T; a real per-call dynamic-T driver would instead declare these as graph INPUTS, same as
    VITS's own convert_vits.py precedent). `table`: dict with 'k'/'v' -> already-windowed (2T-1,head_dim)
    numpy arrays. Returns Layout A [T,dim]."""
    def p(name):
        return f"{sd_prefix}.{name}" if sd_prefix else name

    head_dim = dim // n_heads
    x_cb_p = tb.node("PERMUTE", [x_ta], {"axes": [1, 0, 2, 3]}, f"{out_hint}_cb_p")
    x_cb = tb.node("CONT", [x_cb_p], None, f"{out_hint}_cb")  # [dim, T]

    qw = tb.weight(f"{prefix}.conv_q.weight", to_f32(sd[p("conv_q.weight")]).squeeze(-1))
    qb = tb.weight(f"{prefix}.conv_q.bias", to_f32(sd[p("conv_q.bias")]))
    kw = tb.weight(f"{prefix}.conv_k.weight", to_f32(sd[p("conv_k.weight")]).squeeze(-1))
    kb = tb.weight(f"{prefix}.conv_k.bias", to_f32(sd[p("conv_k.bias")]))
    vw = tb.weight(f"{prefix}.conv_v.weight", to_f32(sd[p("conv_v.weight")]).squeeze(-1))
    vb = tb.weight(f"{prefix}.conv_v.bias", to_f32(sd[p("conv_v.bias")]))
    ow = tb.weight(f"{prefix}.conv_o.weight", to_f32(sd[p("conv_o.weight")]).squeeze(-1))
    ob = tb.weight(f"{prefix}.conv_o.bias", to_f32(sd[p("conv_o.bias")]))

    q = tb.node("ADD", [tb.node("MUL_MAT", [qw, x_cb], None, f"{out_hint}_q_mm"), qb], None, f"{out_hint}_q_b")
    k = tb.node("ADD", [tb.node("MUL_MAT", [kw, x_cb], None, f"{out_hint}_k_mm"), kb], None, f"{out_hint}_k_b")
    v = tb.node("ADD", [tb.node("MUL_MAT", [vw, x_cb], None, f"{out_hint}_v_mm"), vb], None, f"{out_hint}_v_b")
    q = tb.node("RESHAPE", [q], {"shape": [head_dim, n_heads, seq_len_expr]}, f"{out_hint}_q_r")
    k = tb.node("RESHAPE", [k], {"shape": [head_dim, n_heads, seq_len_expr]}, f"{out_hint}_k_r")
    v = tb.node("RESHAPE", [v], {"shape": [head_dim, n_heads, seq_len_expr]}, f"{out_hint}_v_r")

    ek = tb.weight(f"{prefix}.emb_rel_k", table["k"])
    ev = tb.weight(f"{prefix}.emb_rel_v", table["v"])
    mask = tb.weight(f"{prefix}.mask_zero", np.zeros((seq_len_int, seq_len_int), dtype=np.float32))
    attn = tb.node("REL_POS_ATTENTION_SHAW", [q, k, v, ek, ev, mask],
                   {"scale": 1.0 / float(np.sqrt(head_dim))}, f"{out_hint}_attn")
    out_cb = tb.node("ADD", [tb.node("MUL_MAT", [ow, attn], None, f"{out_hint}_o_mm"), ob], None, f"{out_hint}_out_cb")

    out_p = tb.node("PERMUTE", [out_cb], {"axes": [1, 0, 2, 3]}, f"{out_hint}_ta_p")
    return tb.node("CONT", [out_p], None, out_hint)  # [T, dim]


def build_dp_text_encoder(tb, sd, dim, interm_dim, n_cn_layers, n_attn_layers, n_heads, window_size,
                           tables, seq_len_expr, seq_len_int, out_hint):
    """`DPTextEncoder.forward`: char embedding -> prepend sentence_token -> ConvNeXt stack -> (rel-pos
    attention + LayerNorm + FFN + LayerNorm) x n_attn_layers -> big residual with convnext_out ->
    extract position 0 (the sentence token) -> proj_out (no bias). Real masking always a no-op here
    (single unpadded utterance). `tables`: list of {'k':...,'v':...} per attn layer (already windowed to
    `seq_len_int+1`, the POST-prepend length -- see caller for how these are built via
    get_relative_embeddings). Returns utt_emb, a flat [dim] vector."""
    t_plus_1_expr = f"({seq_len_expr}+1)"
    t_plus_1_int = seq_len_int + 1

    emb_w = tb.weight("dp_te.char_embedder.weight", to_f32(sd["char_embedder.weight"]))
    x = tb.node("GET_ROWS", [emb_w, "txt_ids"], None, "dp_te_emb")  # [dim, T] Layout B
    sentence_token = tb.weight("dp_te.sentence_token", to_f32(sd["sentence_token"]).reshape(1, dim))  # (1,dim,1)->(1,dim)
    x = tb.node("CONCAT", [sentence_token, x], {"dim": 1}, "dp_te_prepend")  # [dim, T+1] (T-axis concat)

    xp = tb.node("PERMUTE", [x], {"axes": [1, 0, 2, 3]}, "dp_te_ta_p")
    x = tb.node("CONT", [xp], None, "dp_te_ta")  # [T+1, dim] Layout A

    for i in range(n_cn_layers):
        x = add_convnext_block(tb, x, f"dp_te.convnext.{i}", sd, f"convnext.{i}", dim, interm_dim, 5, 1,
                                causal=False, seq_len_expr=t_plus_1_expr, out_hint=f"dp_te_cn{i}")
    convnext_out = x

    for i in range(n_attn_layers):
        attn_out = add_multihead_relative_attention(tb, x, f"dp_te.attn{i}", sd, f"attn_layers.{i}", dim,
                                                     n_heads, window_size, tables[i], t_plus_1_expr,
                                                     t_plus_1_int, f"dp_te_attn{i}")
        x = tb.node("ADD", [x, attn_out], None, f"dp_te_res1_{i}")
        x = apply_channel_layer_norm(tb, x, f"dp_te.norm1.{i}", sd, f"norm_layers_1.{i}", dim, 1e-6,
                                      t_plus_1_expr, f"dp_te_ln1_{i}")
        ffn_out = add_feedforward_block(tb, x, f"dp_te.ffn{i}", sd, f"ffn_layers.{i}", dim, interm_dim,
                                         t_plus_1_expr, f"dp_te_ffn{i}")
        x = tb.node("ADD", [x, ffn_out], None, f"dp_te_res2_{i}")
        x = apply_channel_layer_norm(tb, x, f"dp_te.norm2.{i}", sd, f"norm_layers_2.{i}", dim, 1e-6,
                                      t_plus_1_expr, f"dp_te_ln2_{i}")

    x = tb.node("ADD", [x, convnext_out], None, "dp_te_big_res")  # [T+1, dim]

    pos0 = tb.node("VIEW", [x], {"shape": [1, dim], "offset": 0}, "dp_te_pos0")
    pos0_cont = tb.node("CONT", [pos0], None, "dp_te_pos0_cont")  # VIEW's own nb1 != contiguous stride
    pos0_vec = tb.node("RESHAPE", [pos0_cont], {"shape": [dim]}, "dp_te_pos0_vec")
    proj_w = tb.weight("dp_te.proj_out.weight", to_f32(sd["proj_out.weight"]).squeeze(-1))
    utt_emb = tb.node("MUL_MAT", [proj_w, pos0_vec], None, out_hint)
    return utt_emb


def build_style_encoder(tb, x_ta, sd, name_prefix, input_dim, embed_dim, interm_dim, n_style, stl_dim,
                         n_cn_layers, seq_len_expr, out_hint):
    """`{DP,TTL}StyleEncoder.forward` (structurally IDENTICAL classes, only dims differ -- real source:
    duration/encoders.py's `DPStyleEncoder`, text_to_latent_encoding/encoders.py's `TTLStyleEncoder`):
    Linear(input_dim,embed_dim) -> ConvNeXt stack -> StyleEncoderCrossAttention. x_ta: Layout A
    [T,input_dim] (the compressed-latent crop, T=$n_tokens by convention). `name_prefix` scopes this
    call's own weight names (so DP/TTL instances sharing this builder don't collide). Returns [stl_dim,
    n_style] Layout B (matching StyleEncoderCrossAttention's own output convention)."""
    x_cb_p = tb.node("PERMUTE", [x_ta], {"axes": [1, 0, 2, 3]}, f"{out_hint}_cb_p")
    x_cb = tb.node("CONT", [x_cb_p], None, f"{out_hint}_cb")  # [input_dim, T]
    w = tb.weight(f"{name_prefix}.proj_in.weight", to_f32(sd["proj_in.weight"]))
    b = tb.weight(f"{name_prefix}.proj_in.bias", to_f32(sd["proj_in.bias"]))
    h = tb.node("ADD", [tb.node("MUL_MAT", [w, x_cb], None, f"{out_hint}_proj_mm"), b], None, f"{out_hint}_proj")
    hp = tb.node("PERMUTE", [h], {"axes": [1, 0, 2, 3]}, f"{out_hint}_ta_p")
    h = tb.node("CONT", [hp], None, f"{out_hint}_ta")  # [T, embed_dim] Layout A

    for i in range(n_cn_layers):
        h = add_convnext_block(tb, h, f"{name_prefix}.convnext.{i}", sd, f"convnext.{i}", embed_dim,
                                interm_dim, 5, 1, causal=False, seq_len_expr=seq_len_expr,
                                out_hint=f"{out_hint}_cn{i}")

    hp2 = tb.node("PERMUTE", [h], {"axes": [1, 0, 2, 3]}, f"{out_hint}_style_cb_p")
    h_cb = tb.node("CONT", [hp2], None, f"{out_hint}_style_cb")  # [embed_dim, T]
    return add_style_encoder_cross_attention(tb, h_cb, f"{name_prefix}.style_token_layer", sd,
                                              "style_token_layer", embed_dim, stl_dim, n_style,
                                              seq_len_expr, out_hint)


def add_speech_prompted_cross_attention_delta(tb, x_cb, stl_emb_cb, prefix, sd, sd_prefix, txt_dim,
                                               stl_dim, n_style, seq_len_expr, out_hint):
    """`SpeechPromptedCrossAttention.compute_delta`, real source components.py -- ALWAYS 2 heads (every
    real instance is n_heads=2). Real masking (`txt_msk`-gated) is OMITTED -- always a no-op for a single
    unpadded utterance, same precedent as every other model in this project. `x_cb`: Layout B
    [txt_dim,T] (text queries). `stl_emb_cb`: Layout B [stl_dim,n_style] (style keys/values). Returns
    delta, Layout B [txt_dim,T] (caller adds the residual -- real `.forward()` does `x +
    compute_delta(...)`, but `SpeechPromptedTextEncoder`'s own 2-stage assembly needs the delta alone to
    implement its own "query from stage N, residual from original x_0" pattern).

    NOTE: `scale = sqrt(stl_dim)` here (NOT `sqrt(txt_dim)` like `StyleCrossAttention`'s own
    `scale=sqrt(dim)` -- confirmed directly from `SpeechPromptedCrossAttention.__init__`'s own
    `self.scale = torch.sqrt(torch.tensor(stl_dim, ...))`), and the KEY is a LEARNABLE parameter (not
    derived from any input) rather than data-dependent -- gated via `tanh` and FOLDED AT CONVERSION TIME
    (numpy `np.tanh`, same "fold at conversion time" precedent as weight-norm), since it never depends on
    any runtime input.
    """
    def p(name):
        return f"{sd_prefix}.{name}" if sd_prefix else name

    n_heads = 2
    dh = stl_dim // n_heads
    scale = float(np.sqrt(stl_dim))

    wq = tb.weight(f"{prefix}.W_query.weight", to_f32(sd[p("W_query.weight")]))
    bq = tb.weight(f"{prefix}.W_query.bias", to_f32(sd[p("W_query.bias")]))
    q = tb.node("ADD", [tb.node("MUL_MAT", [wq, x_cb], None, f"{out_hint}_q_mm"), bq], None, f"{out_hint}_q")

    wv = tb.weight(f"{prefix}.W_value.weight", to_f32(sd[p("W_value.weight")]))
    bv = tb.weight(f"{prefix}.W_value.bias", to_f32(sd[p("W_value.bias")]))
    v = tb.node("ADD", [tb.node("MUL_MAT", [wv, stl_emb_cb], None, f"{out_hint}_v_mm"), bv], None, f"{out_hint}_v")

    key_np = to_f32(sd[p("key")])  # (n_heads, 1, dh, n_style), pre-tanh
    head_outs = []
    for h in range(n_heads):
        off = h * dh * 4
        q_h = tb.node("VIEW", [q], {"shape": [dh, seq_len_expr], "offset": off}, f"{out_hint}_q{h}")
        v_h = tb.node("VIEW", [v], {"shape": [dh, n_style], "offset": off}, f"{out_hint}_v{h}")

        # key_np[h,0] is numpy (dh,n_style); tanh folded now (never depends on any runtime input); stored
        # WITHOUT transposing so GGUFWriter's own axis reversal gives ggml ne=[n_style,dh] -- WRONG for a
        # direct contraction against q_h (ne[0]=dh), so register the TRANSPOSE instead: numpy (n_style,dh)
        # -> ggml ne=[dh,n_style], matching q_h's own ne[0]=dh contraction axis.
        key_h = tb.weight(f"{prefix}.key{h}", np.tanh(key_np[h, 0]).T.copy())  # ne=[dh,n_style]

        scores = tb.node("MUL_MAT", [key_h, q_h], None, f"{out_hint}_scores{h}")  # [n_style, T]
        scores = tb.node("SCALE", [scores], {"s": 1.0 / scale}, f"{out_hint}_scores_scaled{h}")
        attn = tb.node("SOFTMAX", [scores], None, f"{out_hint}_attn{h}")  # softmax over n_style (ne[0])

        v_h_t = tb.node("CONT", [tb.node("PERMUTE", [v_h], {"axes": [1, 0, 2, 3]}, f"{out_hint}_v{h}_t_p")],
                         None, f"{out_hint}_v{h}_t")  # [n_style, dh]
        out_h = tb.node("MUL_MAT", [v_h_t, attn], None, f"{out_hint}_out{h}")  # [dh, T]
        head_outs.append(out_h)

    out = tb.node("CONCAT", [head_outs[0], head_outs[1]], {"dim": 0}, f"{out_hint}_heads_cat")  # [stl_dim, T]
    ow = tb.weight(f"{prefix}.out_fc.weight", to_f32(sd[p("out_fc.weight")]))
    ob = tb.weight(f"{prefix}.out_fc.bias", to_f32(sd[p("out_fc.bias")]))
    return tb.node("ADD", [tb.node("MUL_MAT", [ow, out], None, f"{out_hint}_outfc_mm"), ob], None, out_hint)


def build_speech_prompted_text_encoder(tb, x_ta, stl_emb_cb, prefix, sd, sd_prefix, txt_dim, stl_dim,
                                        n_style, seq_len_expr, out_hint):
    """`SpeechPromptedTextEncoder.forward`: stage 1 (query+residual both from x_0) -> stage 2 (query from
    stage-1 output, residual from the SAME ORIGINAL x_0 again) -> LayerNorm -- the same
    "later stage queries forward, residual stays anchored to the original" pattern as
    `StyleEncoderCrossAttention`. x_ta: Layout A [T,txt_dim] (`TTLTextPreEncoder`'s own native output
    convention). stl_emb_cb: Layout B [stl_dim,n_style]. `prefix` scopes this call's own weight names.
    Returns Layout A [T,txt_dim]."""
    def p(name):
        return f"{sd_prefix}.{name}" if sd_prefix else name

    xp = tb.node("PERMUTE", [x_ta], {"axes": [1, 0, 2, 3]}, f"{out_hint}_x0_cb_p")
    x0_cb = tb.node("CONT", [xp], None, f"{out_hint}_x0_cb")  # [txt_dim, T]

    delta1 = add_speech_prompted_cross_attention_delta(tb, x0_cb, stl_emb_cb, f"{prefix}.attention1", sd,
                                                        p("attention1"), txt_dim, stl_dim, n_style,
                                                        seq_len_expr, f"{out_hint}_delta1")
    x1_cb = tb.node("ADD", [x0_cb, delta1], None, f"{out_hint}_x1_cb")

    delta2 = add_speech_prompted_cross_attention_delta(tb, x1_cb, stl_emb_cb, f"{prefix}.attention2", sd,
                                                        p("attention2"), txt_dim, stl_dim, n_style,
                                                        seq_len_expr, f"{out_hint}_delta2")
    xt_cb = tb.node("ADD", [x0_cb, delta2], None, f"{out_hint}_xt_cb")  # residual from ORIGINAL x0, not x1

    xt_p = tb.node("PERMUTE", [xt_cb], {"axes": [1, 0, 2, 3]}, f"{out_hint}_xt_ta_p")
    xt_ta = tb.node("CONT", [xt_p], None, f"{out_hint}_xt_ta")  # [T, txt_dim] Layout A
    return apply_channel_layer_norm(tb, xt_ta, f"{prefix}.norm", sd, p("norm"), txt_dim, 1e-6,
                                     seq_len_expr, out_hint)


def build_ttl_text_pre_encoder(tb, sd, dim, interm_dim, n_cn_layers, n_attn_layers, n_heads, window_size,
                                tables, seq_len_expr, seq_len_int, out_hint):
    """`TTLTextPreEncoder.forward`: char embedding -> ConvNeXt stack -> (rel-pos attention + LayerNorm +
    FFN + LayerNorm) x n_attn_layers -> big residual with convnext_out. Real masking always a no-op here
    (single unpadded utterance). Unlike `DPTextEncoder`, there is NO sentence-token prepend/pooling --
    the full [T,dim] sequence is returned directly (real source: no `sentence_token`/`proj_out` at all in
    this class). `tables`: list of {'k':...,'v':...} per attn layer, windowed to `seq_len_int` (no +1
    here, unlike DPTextEncoder's own post-prepend length). Returns Layout A [T,dim]."""
    emb_w = tb.weight("ttl_te.char_embedder.weight", to_f32(sd["char_embedder.weight"]))
    x = tb.node("GET_ROWS", [emb_w, "txt_ids"], None, "ttl_te_emb")  # [dim, T] Layout B
    xp = tb.node("PERMUTE", [x], {"axes": [1, 0, 2, 3]}, "ttl_te_ta_p")
    x = tb.node("CONT", [xp], None, "ttl_te_ta")  # [T, dim] Layout A

    for i in range(n_cn_layers):
        x = add_convnext_block(tb, x, f"ttl_te.convnext.{i}", sd, f"convnext.{i}", dim, interm_dim, 5, 1,
                                causal=False, seq_len_expr=seq_len_expr, out_hint=f"ttl_te_cn{i}")
    convnext_out = x

    for i in range(n_attn_layers):
        attn_out = add_multihead_relative_attention(tb, x, f"ttl_te.attn{i}", sd, f"attn_layers.{i}", dim,
                                                     n_heads, window_size, tables[i], seq_len_expr,
                                                     seq_len_int, f"ttl_te_attn{i}")
        x = tb.node("ADD", [x, attn_out], None, f"ttl_te_res1_{i}")
        x = apply_channel_layer_norm(tb, x, f"ttl_te.norm1.{i}", sd, f"norm_layers_1.{i}", dim, 1e-6,
                                      seq_len_expr, f"ttl_te_ln1_{i}")
        ffn_out = add_feedforward_block(tb, x, f"ttl_te.ffn{i}", sd, f"ffn_layers.{i}", dim, interm_dim,
                                         seq_len_expr, f"ttl_te_ffn{i}")
        x = tb.node("ADD", [x, ffn_out], None, f"ttl_te_res2_{i}")
        x = apply_channel_layer_norm(tb, x, f"ttl_te.norm2.{i}", sd, f"norm_layers_2.{i}", dim, 1e-6,
                                      seq_len_expr, f"ttl_te_ln2_{i}")

    return tb.node("ADD", [x, convnext_out], None, out_hint)  # [T, dim]


def add_rope(tb, x_h, cos_t, sin_t, half_dim, seq_len_expr, out_hint):
    """Applies fractional RoPE to one head's [head_dim, T] Layout B slice, given precomputed [half_dim, T]
    `cos_t`/`sin_t` tables (see `add_vf_text_cross_attention`'s own docstring for how those are built).
    Real `_rope`: `x1,x2 = x[..:half], x[half:]`; returns `cat([x1*cos-x2*sin, x1*sin+x2*cos])`."""
    x1 = tb.node("VIEW", [x_h], {"shape": [half_dim, seq_len_expr], "offset": 0}, f"{out_hint}_x1")
    x2 = tb.node("VIEW", [x_h], {"shape": [half_dim, seq_len_expr], "offset": half_dim * 4}, f"{out_hint}_x2")
    rot1 = tb.node("SUB", [tb.node("MUL", [x1, cos_t], None, f"{out_hint}_x1cos"),
                           tb.node("MUL", [x2, sin_t], None, f"{out_hint}_x2sin")], None, f"{out_hint}_rot1")
    rot2 = tb.node("ADD", [tb.node("MUL", [x1, sin_t], None, f"{out_hint}_x1sin"),
                           tb.node("MUL", [x2, cos_t], None, f"{out_hint}_x2cos")], None, f"{out_hint}_rot2")
    return tb.node("CONCAT", [rot1, rot2], {"dim": 0}, out_hint)  # [head_dim, T]


def add_vf_text_cross_attention(tb, lat_cb, txt_cb, prefix, sd, sd_prefix, lat_dim, txt_dim, n_heads,
                                 head_dim, lat_frac_pos, txt_frac_pos, lat_seq_len_expr, txt_seq_len_expr,
                                 out_hint):
    """`VFTextCrossAttention.forward` (real source: vector_field_estimator.py) -- 4-head cross-attention,
    latent queries / text keys+values, with FRACTIONAL RoPE: `position = (index / actual_length) *
    theta`, NOT the usual integer-position RoPE (confirmed directly from source: `pos_q =
    (increments[:,:L] / lat_len) * theta` where `lat_len`/`txt_len` are the REAL (unpadded, single
    utterance) sequence lengths -- always `L`/`T` exactly here, same "no real masking" precedent as
    everywhere else). `lat_frac_pos`/`txt_frac_pos`: declared graph inputs, HOST-COMPUTED
    `[i/L for i in range(L)]` / `[i/T for i in range(T)]` (L,T only known at call time -- same
    "host computes small per-call values, feeds in as declared input" precedent as VITS's own
    `get_relative_embeddings` tables). `lat_cb`/`txt_cb`: Layout B [lat_dim,L]/[txt_dim,T] (this whole
    module operates channel-first throughout, no ConvNeXt-style Layout A crossings needed). Returns
    Layout B [lat_dim, L].
    """
    def p(name):
        return f"{sd_prefix}.{name}" if sd_prefix else name

    proj_dim = n_heads * head_dim
    half_dim = head_dim // 2
    scale = float(np.sqrt(proj_dim))

    wq = tb.weight(f"{prefix}.W_query.weight", to_f32(sd[p("W_query.weight")]))
    bq = tb.weight(f"{prefix}.W_query.bias", to_f32(sd[p("W_query.bias")]))
    q = tb.node("ADD", [tb.node("MUL_MAT", [wq, lat_cb], None, f"{out_hint}_q_mm"), bq], None, f"{out_hint}_q")
    wk = tb.weight(f"{prefix}.W_key.weight", to_f32(sd[p("W_key.weight")]))
    bk = tb.weight(f"{prefix}.W_key.bias", to_f32(sd[p("W_key.bias")]))
    k = tb.node("ADD", [tb.node("MUL_MAT", [wk, txt_cb], None, f"{out_hint}_k_mm"), bk], None, f"{out_hint}_k")
    wv = tb.weight(f"{prefix}.W_value.weight", to_f32(sd[p("W_value.weight")]))
    bv = tb.weight(f"{prefix}.W_value.bias", to_f32(sd[p("W_value.bias")]))
    v = tb.node("ADD", [tb.node("MUL_MAT", [wv, txt_cb], None, f"{out_hint}_v_mm"), bv], None, f"{out_hint}_v")

    # angle[d,pos] = theta[d] * frac_pos[pos] -- an OUTER PRODUCT via MUL_MAT with a size-1 contraction
    # dim (same trick as StyleTTS2's own sinusoidal time embedding, generalized from a scalar `time` to a
    # full position VECTOR here).
    theta = tb.weight(f"{prefix}.theta", to_f32(sd[p("theta")]).reshape(half_dim))
    theta_2d = tb.node("RESHAPE", [theta], {"shape": [1, half_dim]}, f"{out_hint}_theta_2d")

    def cos_sin_table(frac_pos_name, seq_len_expr, tag):
        frac_2d = tb.node("RESHAPE", [frac_pos_name], {"shape": [1, seq_len_expr]}, f"{out_hint}_{tag}_frac2d")
        angle = tb.node("MUL_MAT", [theta_2d, frac_2d], None, f"{out_hint}_{tag}_angle")  # [half_dim, T]
        return tb.node("COS", [angle], None, f"{out_hint}_{tag}_cos"), tb.node("SIN", [angle], None,
                                                                                f"{out_hint}_{tag}_sin")

    cos_q, sin_q = cos_sin_table(lat_frac_pos, lat_seq_len_expr, "q")
    cos_k, sin_k = cos_sin_table(txt_frac_pos, txt_seq_len_expr, "k")

    head_outs = []
    for h in range(n_heads):
        off = h * head_dim * 4
        q_h = tb.node("VIEW", [q], {"shape": [head_dim, lat_seq_len_expr], "offset": off}, f"{out_hint}_q{h}")
        k_h = tb.node("VIEW", [k], {"shape": [head_dim, txt_seq_len_expr], "offset": off}, f"{out_hint}_k{h}")
        v_h = tb.node("VIEW", [v], {"shape": [head_dim, txt_seq_len_expr], "offset": off}, f"{out_hint}_v{h}")

        q_rot = add_rope(tb, q_h, cos_q, sin_q, half_dim, lat_seq_len_expr, f"{out_hint}_qrot{h}")
        k_rot = add_rope(tb, k_h, cos_k, sin_k, half_dim, txt_seq_len_expr, f"{out_hint}_krot{h}")

        scores = tb.node("MUL_MAT", [k_rot, q_rot], None, f"{out_hint}_scores{h}")  # [T, L]
        scores = tb.node("SCALE", [scores], {"s": 1.0 / scale}, f"{out_hint}_scores_scaled{h}")
        attn = tb.node("SOFTMAX", [scores], None, f"{out_hint}_attn{h}")  # softmax over T (ne[0])

        v_h_t = tb.node("CONT", [tb.node("PERMUTE", [v_h], {"axes": [1, 0, 2, 3]}, f"{out_hint}_v{h}_t_p")],
                         None, f"{out_hint}_v{h}_t")  # [T, head_dim]
        out_h = tb.node("MUL_MAT", [v_h_t, attn], None, f"{out_hint}_out{h}")  # [head_dim, L]
        head_outs.append(out_h)

    out = head_outs[0]
    for oh in head_outs[1:]:
        out = tb.node("CONCAT", [out, oh], {"dim": 0}, f"{out_hint}_heads_cat")  # [proj_dim, L]
    ow = tb.weight(f"{prefix}.out_fc.weight", to_f32(sd[p("out_fc.weight")]))
    ob = tb.weight(f"{prefix}.out_fc.bias", to_f32(sd[p("out_fc.bias")]))
    out = tb.node("ADD", [tb.node("MUL_MAT", [ow, out], None, f"{out_hint}_outfc_mm"), ob], None,
                  f"{out_hint}_outfc")  # [lat_dim, L]

    x_seq = tb.node("ADD", [lat_cb, out], None, f"{out_hint}_res")
    normed = tb.node("LAYER_NORM", [x_seq], {"eps": 1e-6}, f"{out_hint}_ln_normed")
    g = tb.weight(f"{prefix}.norm.gamma", to_f32(sd[p("norm.weight")]))
    b = tb.weight(f"{prefix}.norm.beta", to_f32(sd[p("norm.bias")]))
    normed = tb.node("MUL", [normed, g], None, f"{out_hint}_ln_mul")
    return tb.node("ADD", [normed, b], None, out_hint)  # Layout B [lat_dim, L]


def add_vf_style_cross_attention(tb, lat_cb, stl_emb_cb, prefix, sd, sd_prefix, lat_dim, stl_dim,
                                  n_style, n_heads, head_dim, lat_seq_len_expr, out_hint):
    """`VFStyleCrossAttention.forward` (real source: vector_field_estimator.py) -- 2-head cross-attention,
    latent queries / TTL-style keys+values, keys derived via `tanh(W_key(fixed_constant_prototype))` --
    the prototype itself is a LEARNED parameter (`self.key`, shape (1,n_style,stl_dim)), so `tanh(W_key(
    key))` is NOT foldable at conversion time the way `SpeechPromptedCrossAttention`'s own raw learnable
    key was (that one skipped `W_key` entirely) -- `W_key` must run in-graph here. `lat_cb`: Layout B
    [lat_dim,L]. `stl_emb_cb`: Layout B [stl_dim,n_style]. Returns Layout B [lat_dim,L]."""
    def p(name):
        return f"{sd_prefix}.{name}" if sd_prefix else name

    proj_dim = n_heads * head_dim
    scale = float(np.sqrt(proj_dim))

    wq = tb.weight(f"{prefix}.W_query.weight", to_f32(sd[p("W_query.weight")]))
    bq = tb.weight(f"{prefix}.W_query.bias", to_f32(sd[p("W_query.bias")]))
    q = tb.node("ADD", [tb.node("MUL_MAT", [wq, lat_cb], None, f"{out_hint}_q_mm"), bq], None, f"{out_hint}_q")
    wv = tb.weight(f"{prefix}.W_value.weight", to_f32(sd[p("W_value.weight")]))
    bv = tb.weight(f"{prefix}.W_value.bias", to_f32(sd[p("W_value.bias")]))
    v = tb.node("ADD", [tb.node("MUL_MAT", [wv, stl_emb_cb], None, f"{out_hint}_v_mm"), bv], None, f"{out_hint}_v")

    # `key` numpy shape (n_style,stl_dim) after the squeeze/reshape below -> GGUFWriter's own axis
    # reversal ALREADY gives ggml ne=[stl_dim,n_style] -- i.e. Layout B directly, no further crossing
    # needed (unlike a real *input* tensor dumped from a native (B,T,C) PyTorch buffer, a weight
    # constant's ggml layout is simply whatever numpy shape we choose to register it with -- a real bug
    # caught here: an earlier version PERMUTE+CONT'd this a second time, wrongly turning an
    # already-correct Layout B tensor into Layout A).
    key_cb = tb.weight(f"{prefix}.key", to_f32(sd[p("key")]).reshape(n_style, stl_dim))  # ne=[stl_dim,n_style]
    wk = tb.weight(f"{prefix}.W_key.weight", to_f32(sd[p("W_key.weight")]))
    bk = tb.weight(f"{prefix}.W_key.bias", to_f32(sd[p("W_key.bias")]))
    k = tb.node("ADD", [tb.node("MUL_MAT", [wk, key_cb], None, f"{out_hint}_k_mm"), bk], None, f"{out_hint}_k_lin")
    k = tb.node("TANH", [k], None, f"{out_hint}_k")  # [proj_dim, n_style]

    head_outs = []
    for h in range(n_heads):
        off = h * head_dim * 4
        q_h = tb.node("VIEW", [q], {"shape": [head_dim, lat_seq_len_expr], "offset": off}, f"{out_hint}_q{h}")
        k_h = tb.node("VIEW", [k], {"shape": [head_dim, n_style], "offset": off}, f"{out_hint}_k{h}")
        v_h = tb.node("VIEW", [v], {"shape": [head_dim, n_style], "offset": off}, f"{out_hint}_v{h}")

        scores = tb.node("MUL_MAT", [k_h, q_h], None, f"{out_hint}_scores{h}")  # [n_style, L]
        scores = tb.node("SCALE", [scores], {"s": 1.0 / scale}, f"{out_hint}_scores_scaled{h}")
        attn = tb.node("SOFTMAX", [scores], None, f"{out_hint}_attn{h}")  # softmax over n_style (ne[0])

        v_h_t = tb.node("CONT", [tb.node("PERMUTE", [v_h], {"axes": [1, 0, 2, 3]}, f"{out_hint}_v{h}_t_p")],
                         None, f"{out_hint}_v{h}_t")  # [n_style, head_dim]
        out_h = tb.node("MUL_MAT", [v_h_t, attn], None, f"{out_hint}_out{h}")  # [head_dim, L]
        head_outs.append(out_h)

    out = head_outs[0]
    for oh in head_outs[1:]:
        out = tb.node("CONCAT", [out, oh], {"dim": 0}, f"{out_hint}_heads_cat")  # [proj_dim, L]
    ow = tb.weight(f"{prefix}.out_fc.weight", to_f32(sd[p("out_fc.weight")]))
    ob = tb.weight(f"{prefix}.out_fc.bias", to_f32(sd[p("out_fc.bias")]))
    out = tb.node("ADD", [tb.node("MUL_MAT", [ow, out], None, f"{out_hint}_outfc_mm"), ob], None,
                  f"{out_hint}_outfc")  # [lat_dim, L]

    x_seq = tb.node("ADD", [lat_cb, out], None, f"{out_hint}_res")
    normed = tb.node("LAYER_NORM", [x_seq], {"eps": 1e-6}, f"{out_hint}_ln_normed")
    g = tb.weight(f"{prefix}.norm.gamma", to_f32(sd[p("norm.weight")]))
    b = tb.weight(f"{prefix}.norm.beta", to_f32(sd[p("norm.bias")]))
    normed = tb.node("MUL", [normed, g], None, f"{out_hint}_ln_mul")
    return tb.node("ADD", [normed, b], None, out_hint)  # Layout B [lat_dim, L]


def build_ttl_text_encoder(tb, sd, stl_emb_cb, dim, interm_dim, n_cn_layers, n_attn_layers, n_heads,
                            window_size, tables, n_style, seq_len_expr, seq_len_int, out_hint):
    """`TTLTextEncoder.forward`: `TTLTextPreEncoder` -> `SpeechPromptedTextEncoder` (conditions on the
    TTL style embedding). `stl_emb_cb`: Layout B [dim,n_style] (`TTLStyleEncoder`'s own output --
    confirmed against the real checkpoint that `stl_dim==dim==256` here, both `txt_dim` and `stl_dim`
    arguments to `SpeechPromptedCrossAttention` are this same `dim`). Returns Layout A [T,dim]."""
    x = build_ttl_text_pre_encoder(tb, sd, dim, interm_dim, n_cn_layers, n_attn_layers, n_heads,
                                    window_size, tables, seq_len_expr, seq_len_int, f"{out_hint}_pre")
    return build_speech_prompted_text_encoder(tb, x, stl_emb_cb, "ttl_te.spe", sd,
                                               "speech_prompted_text_encoder", dim, dim, n_style,
                                               seq_len_expr, out_hint)


def mish(tb, x, out_hint):
    sp = tb.node("SOFTPLUS", [x], None, f"{out_hint}_softplus")
    t = tb.node("TANH", [sp], None, f"{out_hint}_tanh")
    return tb.node("MUL", [x, t], None, out_hint)


def build_vftime_encoder(tb, sd, sd_prefix, n_freqs, mlp_hidden, mlp_out, scale, out_hint):
    """`VFTimeEncoder.forward` (real source: vector_field_estimator.py): sinusoidal `t*scale*freqs`
    embedding, concat sin/cos, 2-layer MLP w/ Mish. `t`: declared graph input, scalar [1]. Returns a flat
    [mlp_out] vector (real code's own trailing `.unsqueeze(-1)` is dropped -- every caller here
    immediately treats it as a plain channel vector anyway)."""
    def p(name):
        return f"{sd_prefix}.{name}" if sd_prefix else name

    freqs = tb.weight(f"{out_hint}.freqs", to_f32(sd[p("freqs")]).reshape(n_freqs))
    angles_raw = tb.node("MUL", [freqs, "t"], None, f"{out_hint}_angles_raw")
    angles = tb.node("SCALE", [angles_raw], {"s": scale}, f"{out_hint}_angles")
    sin_a = tb.node("SIN", [angles], None, f"{out_hint}_sin_a")
    cos_a = tb.node("COS", [angles], None, f"{out_hint}_cos_a")
    embed = tb.node("CONCAT", [sin_a, cos_a], {"dim": 0}, f"{out_hint}_embed")  # [2*n_freqs]

    w1 = tb.weight(f"{out_hint}.linear1.weight", to_f32(sd[p("linear1.weight")]))
    b1 = tb.weight(f"{out_hint}.linear1.bias", to_f32(sd[p("linear1.bias")]))
    h = tb.node("ADD", [tb.node("MUL_MAT", [w1, embed], None, f"{out_hint}_h1_mm"), b1], None, f"{out_hint}_h1")
    h = mish(tb, h, f"{out_hint}_h1_mish")

    w2 = tb.weight(f"{out_hint}.linear2.weight", to_f32(sd[p("linear2.weight")]))
    b2 = tb.weight(f"{out_hint}.linear2.bias", to_f32(sd[p("linear2.bias")]))
    return tb.node("ADD", [tb.node("MUL_MAT", [w2, h], None, f"{out_hint}_h2_mm"), b2], None, out_hint)


def cross_ta_to_cb(tb, x_ta, out_hint):
    """[T,C] Layout A -> [C,T] Layout B."""
    p = tb.node("PERMUTE", [x_ta], {"axes": [1, 0, 2, 3]}, f"{out_hint}_p")
    return tb.node("CONT", [p], None, out_hint)


def cross_cb_to_ta(tb, x_cb, out_hint):
    """[C,T] Layout B -> [T,C] Layout A."""
    p = tb.node("PERMUTE", [x_cb], {"axes": [1, 0, 2, 3]}, f"{out_hint}_p")
    return tb.node("CONT", [p], None, out_hint)


def build_vector_field_estimator(tb, sd, z_t, txt_emb_cb, stl_emb_cb, t_name, lat_frac_pos, txt_frac_pos,
                                  hp, lat_seq_len_expr, txt_seq_len_expr, out_hint):
    """`VectorFieldEstimator.compute_velocity` (real source: vector_field_estimator.py): proj_in ->
    4 groups x (4x dilated ConvNeXt -> +time_cond -> 1x ConvNeXt -> VFTextCrossAttention -> 1x ConvNeXt
    -> VFStyleCrossAttention) -> 4x ConvNeXt -> proj_out. Real masking always a no-op (single unpadded
    utterance). `z_t`: Layout A [L,latent_dim] (the noisy/CFM-interpolated latent, native (B,144,L)
    memory layout). `txt_emb_cb`/`stl_emb_cb`: Layout B (already crossed by the caller, matching
    `TTLTextEncoder`'s/`TTLStyleEncoder`'s own output conventions directly -- no redundant crossing
    here). `t_name`: declared scalar graph input. `lat_frac_pos`/`txt_frac_pos`: declared graph inputs,
    host-computed fractional positions for THIS call's L/T (see `add_vf_text_cross_attention`'s own
    docstring). Returns Layout A [L, latent_dim] (the velocity field `v`, same convention as `z_t`)."""
    latent_dim = hp["latent_dim"]
    hidden_dim = hp["hidden_dim"]
    interm_dim = hp["interm_dim"]
    txt_dim = hp["txt_dim"]
    n_groups = hp["n_groups"]
    n_cn_layers = hp["n_cn_layers"]
    time_emb_dim = hp["time_emb_dim"]
    n_text_heads, text_head_dim = hp["n_text_heads"], hp["text_head_dim"]
    n_style_heads, style_head_dim = hp["n_style_heads"], hp["style_head_dim"]
    n_style = hp["n_style"]
    stl_dim = hp["stl_dim"]

    proj_in_w = tb.weight("vfe.proj_in.weight", to_f32(sd["proj_in.weight"]))  # (hidden,latent,1)
    z3 = tb.node("RESHAPE", [z_t], {"shape": [lat_seq_len_expr, latent_dim, 1]}, "vfe_z3")
    x = tb.node("CONV_1D", [proj_in_w, z3], {"s0": 1, "p0": 0, "d0": 1}, "vfe_proj_in_raw")
    x = tb.node("RESHAPE", [x], {"shape": [lat_seq_len_expr, hidden_dim]}, "vfe_x0")  # no bias (real: zeroed)

    time_emb = build_vftime_encoder(tb, sd, "time_encoder", 32, 256, time_emb_dim, 1000.0, "vfe_time_emb")

    for g in range(n_groups):
        for j in range(n_cn_layers):
            x = add_convnext_block(tb, x, f"vfe.big_cn{g}_{j}", sd, f"big_convnext.{g}.{j}", hidden_dim,
                                    interm_dim, 5, 2 ** j, causal=False, seq_len_expr=lat_seq_len_expr,
                                    out_hint=f"vfe_g{g}_big{j}")

        tl_w = tb.weight(f"vfe.time_linear{g}.weight", to_f32(sd[f"time_linears.{g}.weight"]))
        tl_b = tb.weight(f"vfe.time_linear{g}.bias", to_f32(sd[f"time_linears.{g}.bias"]))
        t_cond = tb.node("ADD", [tb.node("MUL_MAT", [tl_w, time_emb], None, f"vfe_g{g}_tcond_mm"), tl_b],
                          None, f"vfe_g{g}_tcond")  # [hidden_dim]
        t_cond_r = tb.node("RESHAPE", [t_cond], {"shape": [1, hidden_dim]}, f"vfe_g{g}_tcond_r")
        x = tb.node("ADD", [x, t_cond_r], None, f"vfe_g{g}_plus_tcond")  # broadcast over T (ne[0])

        x = add_convnext_block(tb, x, f"vfe.small_cn1_{g}", sd, f"small_convnext1.{g}", hidden_dim,
                                interm_dim, 5, 1, causal=False, seq_len_expr=lat_seq_len_expr,
                                out_hint=f"vfe_g{g}_small1")

        x_cb = cross_ta_to_cb(tb, x, f"vfe_g{g}_cb_in1")
        x_cb = add_vf_text_cross_attention(tb, x_cb, txt_emb_cb, f"vfe.text_attn{g}", sd, f"text_attn.{g}",
                                            hidden_dim, txt_dim, n_text_heads, text_head_dim, lat_frac_pos,
                                            txt_frac_pos, lat_seq_len_expr, txt_seq_len_expr,
                                            f"vfe_g{g}_textattn_cb")
        x = cross_cb_to_ta(tb, x_cb, f"vfe_g{g}_ta_out1")

        x = add_convnext_block(tb, x, f"vfe.small_cn2_{g}", sd, f"small_convnext2.{g}", hidden_dim,
                                interm_dim, 5, 1, causal=False, seq_len_expr=lat_seq_len_expr,
                                out_hint=f"vfe_g{g}_small2")

        x_cb2 = cross_ta_to_cb(tb, x, f"vfe_g{g}_cb_in2")
        x_cb2 = add_vf_style_cross_attention(tb, x_cb2, stl_emb_cb, f"vfe.style_attn{g}", sd,
                                              f"style_attn.{g}", hidden_dim, stl_dim, n_style,
                                              n_style_heads, style_head_dim, lat_seq_len_expr,
                                              f"vfe_g{g}_styleattn_cb")
        x = cross_cb_to_ta(tb, x_cb2, f"vfe_g{g}_ta_out2")

    for j in range(n_cn_layers):
        x = add_convnext_block(tb, x, f"vfe.last_cn{j}", sd, f"last_convnext.{j}", hidden_dim, interm_dim,
                                5, 1, causal=False, seq_len_expr=lat_seq_len_expr, out_hint=f"vfe_last{j}")

    proj_out_w = tb.weight("vfe.proj_out.weight", to_f32(sd["proj_out.weight"]))  # (latent,hidden,1)
    x3 = tb.node("RESHAPE", [x], {"shape": [lat_seq_len_expr, hidden_dim, 1]}, "vfe_x_out3")
    v = tb.node("CONV_1D", [proj_out_w, x3], {"s0": 1, "p0": 0, "d0": 1}, "vfe_proj_out_raw")
    return tb.node("RESHAPE", [v], {"shape": [lat_seq_len_expr, latent_dim]}, out_hint)  # no bias (real: zeroed)


def add_prelu(tb, x, weight_name, out_hint):
    """`PReLU(num_parameters=1)`: `relu(x) - weight*relu(-x)` (a single learned scalar slope) -- verified
    identity: x>0 -> relu(x)=x,relu(-x)=0 -> x; x<0 -> relu(x)=0,relu(-x)=-x -> -weight*(-x)=weight*x."""
    relu_pos = tb.node("RELU", [x], None, f"{out_hint}_pos")
    neg_x = tb.node("SCALE", [x], {"s": -1.0}, f"{out_hint}_negx")
    relu_neg = tb.node("RELU", [neg_x], None, f"{out_hint}_relu_neg")
    relu_neg_scaled = tb.node("MUL", [relu_neg, weight_name], None, f"{out_hint}_relu_neg_scaled")
    return tb.node("SUB", [relu_pos, relu_neg_scaled], None, out_hint)


def build_speech_decoder(tb, sd, latent, hp, lat_seq_len_expr, out_hint):
    """`SpeechDecoder.forward` (real source: speech_autoencoding/speech_autoencoder.py) -- the FINAL
    stage of the whole SupertonicTTS pipeline: pre-scale -> codebook-decompress (interleave 6 codebooks
    into the time axis, the SAME operation `TemporalLatentCompressor.decompress` performs, done INLINE
    here since the real decoder takes the COMPRESSED 144-channel latent directly) -> denormalize -> causal
    embed conv -> 10x causal ConvNeXt (dilations (1,2,4,1,2,4,1,1,1,1)) -> folded BatchNorm (eval-mode
    BatchNorm1d reduces to a per-channel affine, `scale=weight/sqrt(running_var+eps)`,
    `shift=bias-running_mean*scale`, folded at CONVERSION time -- same "fold at conversion time"
    precedent as weight-norm elsewhere) -> causal head convs w/ PReLU -> DIRECT WAVEFORM (no
    ISTFT/vocoder-GAN stack at all -- the 512-channel head output at each of `T*6` frame positions IS 512
    raw consecutive audio samples). `latent`: Layout A [T,144] (native (B,144,T), T=n_tokens by
    convention). Returns a FLAT [T*6*512] waveform vector (Layout B-derived: real
    `x.transpose(1,2).reshape(B,-1)` puts channels LAST/fastest before flattening, genuinely different
    from every other tensor in this pipeline which stays Layout-A-native until this final step).
    """
    lat_ch = hp["lat_channels"]  # 24
    n_cb = hp["n_codebooks"]  # 6
    hidden_dim = hp["hidden_dim"]  # 512
    interm_dim = hp["interm_dim"]  # 2048
    cn_dilations = hp["cn_dilations"]  # (1,2,4,1,2,4,1,1,1,1)
    norm_scale = float(to_f32(sd["norm_scale"]))
    t6_expr = f"({lat_seq_len_expr}*{n_cb})"

    # --- pre-scale (scalar) ---
    x = tb.node("SCALE", [latent], {"s": 1.0 / norm_scale}, f"{out_hint}_prescaled")  # [T, 144]

    # --- decompress: (T,144) -> (T,6,24) -> permute -> (6,T,24) -> (6T,24) ---
    x3 = tb.node("RESHAPE", [x], {"shape": [lat_seq_len_expr, n_cb, lat_ch]}, f"{out_hint}_split3")
    x3p = tb.node("PERMUTE", [x3], {"axes": [1, 0, 2, 3]}, f"{out_hint}_split3_p")
    x3c = tb.node("CONT", [x3p], None, f"{out_hint}_split3_c")  # [n_cb, T, lat_ch]
    x = tb.node("RESHAPE", [x3c], {"shape": [t6_expr, lat_ch]}, f"{out_hint}_decompressed")  # [T*6, 24]

    # --- denormalize (per-channel, 24ch) ---
    lat_std = tb.weight(f"{out_hint}.lat_std", to_f32(sd["lat_std"]).reshape(lat_ch))
    lat_mean = tb.weight(f"{out_hint}.lat_mean", to_f32(sd["lat_mean"]).reshape(lat_ch))
    lat_std_r = tb.node("RESHAPE", [lat_std], {"shape": [1, lat_ch]}, f"{out_hint}_lat_std_r")
    lat_mean_r = tb.node("RESHAPE", [lat_mean], {"shape": [1, lat_ch]}, f"{out_hint}_lat_mean_r")
    x = tb.node("MUL", [x, lat_std_r], None, f"{out_hint}_denorm_mul")
    x = tb.node("ADD", [x, lat_mean_r], None, f"{out_hint}_denorm")  # [T*6, 24]

    # --- causal embed: Conv1d(24,512,k=7), replicate-pad(6,0) ---
    padded = add_replicate_pad(tb, x, 6, 0, lat_ch, t6_expr, f"{out_hint}_embed_pad")
    embed_w = tb.weight(f"{out_hint}.embed.weight", to_f32(sd["embed.weight"]))
    embed_b = tb.weight(f"{out_hint}.embed.bias", to_f32(sd["embed.bias"]))
    padded_len_expr = f"({t6_expr}+6)"
    p3 = tb.node("RESHAPE", [padded], {"shape": [padded_len_expr, lat_ch, 1]}, f"{out_hint}_embed_in3")
    h = tb.node("CONV_1D", [embed_w, p3], {"s0": 1, "p0": 0, "d0": 1}, f"{out_hint}_embed_raw")
    h = tb.node("RESHAPE", [h], {"shape": [t6_expr, hidden_dim]}, f"{out_hint}_embed_2d")
    h = tb.node("ADD", [h, tb.node("RESHAPE", [embed_b], {"shape": [1, hidden_dim]}, f"{out_hint}_embed_bias_r")],
                None, f"{out_hint}_embed_biased")

    # --- 10x causal ConvNeXt (k=7) ---
    for i, dilation in enumerate(cn_dilations):
        h = add_convnext_block(tb, h, f"{out_hint}.convnext{i}", sd, f"convnext.{i}", hidden_dim, interm_dim,
                                7, dilation, causal=True, seq_len_expr=t6_expr, out_hint=f"{out_hint}_cn{i}")

    # --- folded BatchNorm (eval-mode -> per-channel affine) ---
    bn_w = to_f32(sd["final_norm.weight"])
    bn_b = to_f32(sd["final_norm.bias"])
    bn_mean = to_f32(sd["final_norm.running_mean"])
    bn_var = to_f32(sd["final_norm.running_var"])
    bn_eps = 1e-5
    bn_scale = bn_w / np.sqrt(bn_var + bn_eps)
    bn_shift = bn_b - bn_mean * bn_scale
    # Registered 1D (ne=[hidden_dim]) then RESHAPE'd in-graph to ne=[1,hidden_dim] -- registering a 2D
    # numpy array directly here would be subject to GGUFWriter's own axis-reversal convention, giving
    # ne=[hidden_dim,1] (backwards from what broadcasting against h[T*6,hidden_dim] needs) -- a real bug
    # caught here via a `ggml_can_repeat` assertion, same "RESHAPE in-graph, don't reshape-then-register"
    # pattern already used for `lat_std`/`lat_mean` just above.
    bn_scale_w = tb.weight(f"{out_hint}.bn_scale", bn_scale.astype(np.float32).reshape(hidden_dim))
    bn_shift_w = tb.weight(f"{out_hint}.bn_shift", bn_shift.astype(np.float32).reshape(hidden_dim))
    bn_scale_r = tb.node("RESHAPE", [bn_scale_w], {"shape": [1, hidden_dim]}, f"{out_hint}_bn_scale_r")
    bn_shift_r = tb.node("RESHAPE", [bn_shift_w], {"shape": [1, hidden_dim]}, f"{out_hint}_bn_shift_r")
    h = tb.node("MUL", [h, bn_scale_r], None, f"{out_hint}_bn_mul")
    h = tb.node("ADD", [h, bn_shift_r], None, f"{out_hint}_bn_out")  # [T*6, 512]

    # --- causal head: Conv1d(512,2048,k=3) + PReLU + Conv1d(2048,512,k=1,no bias) ---
    h_padded = add_replicate_pad(tb, h, 2, 0, hidden_dim, t6_expr, f"{out_hint}_head_pad")
    h1_w = tb.weight(f"{out_hint}.head_layer1.weight", to_f32(sd["head_layer1.weight"]))
    h1_b = tb.weight(f"{out_hint}.head_layer1.bias", to_f32(sd["head_layer1.bias"]))
    head_padded_len_expr = f"({t6_expr}+2)"
    hp3 = tb.node("RESHAPE", [h_padded], {"shape": [head_padded_len_expr, hidden_dim, 1]}, f"{out_hint}_h1_in3")
    h1 = tb.node("CONV_1D", [h1_w, hp3], {"s0": 1, "p0": 0, "d0": 1}, f"{out_hint}_h1_raw")
    h1 = tb.node("RESHAPE", [h1], {"shape": [t6_expr, interm_dim]}, f"{out_hint}_h1_2d")
    h1 = tb.node("ADD", [h1, tb.node("RESHAPE", [h1_b], {"shape": [1, interm_dim]}, f"{out_hint}_h1_bias_r")],
                 None, f"{out_hint}_h1_biased")

    prelu_w = tb.weight(f"{out_hint}.head_prelu.weight", to_f32(sd["head_prelu.weight"]).reshape(1))
    h1 = add_prelu(tb, h1, prelu_w, f"{out_hint}_h1_prelu")

    h2_w = tb.weight(f"{out_hint}.head_layer2.weight", to_f32(sd["head_layer2.weight"]))
    h2_3 = tb.node("RESHAPE", [h1], {"shape": [t6_expr, interm_dim, 1]}, f"{out_hint}_h2_in3")
    h2 = tb.node("CONV_1D", [h2_w, h2_3], {"s0": 1, "p0": 0, "d0": 1}, f"{out_hint}_h2_raw")
    h2 = tb.node("RESHAPE", [h2], {"shape": [t6_expr, hidden_dim]}, f"{out_hint}_h2_2d")  # [T*6, 512], no bias

    # --- flatten to waveform: cross to Layout B (channel-fastest), matching real
    #     `x.transpose(1,2).reshape(B,-1)`'s own flat byte order exactly ---
    h2_cb = cross_ta_to_cb(tb, h2, f"{out_hint}_wav_cb")  # [512, T*6]
    total_samples_expr = f"({t6_expr}*{hidden_dim})"
    return tb.node("RESHAPE", [h2_cb], {"shape": [total_samples_expr]}, out_hint)
