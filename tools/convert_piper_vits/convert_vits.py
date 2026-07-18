"""Converts a real piper/VITS checkpoint into loom-engine GGUF files.

Real checkpoint layout confirmed directly against `pipertts_en-GB_miro/epoch=9772-step=1494014.ckpt`
(see BACKLOG.md for the full trail of findings this converter encodes):
  - single-speaker (gin_channels=0): no speaker embedding table, no `cond` conv layers anywhere.
  - StochasticDurationPredictor (`dp`) is real: `dp.flows.0` is ElementwiseAffine (m/logs params),
    `dp.flows.{1,3,5,7}` are ConvFlow, `dp.flows.{2,4,6,8}` are Flip (no weights). At REVERSE-mode
    inference, `dp.flows.1` (the first ConvFlow) and `dp.log_flow` are never touched at all -- see
    BACKLOG.md's "remove a useless vflow" finding -- so their weights are never written.
  - ResidualCouplingBlock (`flow`) uses ALL of its flows in reverse (nothing dropped, unlike `dp`):
    `flow.flows.{0,2,4,6}` are ResidualCouplingLayer (mean_only=True), `{1,3,5,7}` are Flip.
  - HiFi-GAN vocoder (`dec`) uses resblock="2" (confirmed from `dec.resblocks.*.convs.{0,1}` only,
    no `convs2`), 3 upsample stages, 3 resblock kernel sizes per stage (9 resblocks total).

Produces three GGUF files (`out_dir/vits_stats.gguf`, `out_dir/vits_logw.gguf`,
`out_dir/vits_flow_vocoder.gguf`), matching this model's natural two-phase split: phase 1 (TextEncoder +
StochasticDurationPredictor, split into two single-output files since GraphTopology supports only one
declared output per topology) must run BEFORE the host can compute the duration-dependent output frame
count; phase 2 (coupling flow + vocoder) is sized by that host-computed count. See `loom::VitsDriver` for
the host-side glue tying the phases together.

Tensor-layout conventions used throughout (both already established and verified this whole VITS
effort -- see BACKLOG.md / primitives_flow.cpp doc comments):
  - TextEncoder's attention pipeline is channel-first, `[C, T]` (C = ne[0], matching GET_ROWS's own
    embedding-lookup output and REL_POS_ATTENTION_SHAW's convention).
  - WN / DDSConv / ConvFlow / the coupling flow / the vocoder are all `[T, C]` (T = ne[0], matching
    CONV_1D's own data convention). A PERMUTE+CONT ("transpose_2d" below) crosses between the two at
    each boundary (SDP's `pre` conv, TextEncoder's own FFN).
"""
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

from vits_common import fold_weight_norm, load_piper_checkpoint, to_f32

HP = {
    "n_vocab": 256,
    "hidden_channels": 192,
    "inter_channels": 192,  # == TextEncoder's proj-split out_channels
    "filter_channels": 768,  # TextEncoder FFN
    "n_heads": 2,
    "n_layers": 6,  # TextEncoder
    "kernel_size": 3,  # TextEncoder FFN / SDP's own DDSConv / each ConvFlow's DDSConv
    "window_size": 4,
    "sdp_n_flows": 4,
    "sdp_ddsconv_n_layers": 3,
    "sdp_num_bins": 10,
    "sdp_tail_bound": 5.0,
    "flow_kernel_size": 5,
    "flow_dilation_rate": 1,
    "flow_wn_n_layers": 4,
    "flow_n_flows": 4,
    "resblock_kernel_sizes": (3, 5, 7),
    "resblock_dilation_sizes": ((1, 2), (2, 6), (3, 12)),
    "upsample_rates": (8, 8, 4),
    "upsample_initial_channel": 256,
    "upsample_kernel_sizes": (16, 16, 8),
    "ln_eps": 1e-5,
}


class TopologyBuilder:
    """Mirrors `convert_parakeet_tdt.py`'s `node()`-closure idiom: appends `{op, inputs, outputs,
    attrs}` dicts and returns the (single) output name for chaining. Also collects `{gguf_name:
    numpy array}` weight tensors as they're referenced, so the topology and the weight set can never
    drift apart -- every weight a topology names gets registered exactly once, at the point some
    helper first uses it.
    """

    def __init__(self):
        self.nodes = []
        self.weights = {}
        self.int32_weights = set()
        self._counter = 0

    def _fresh(self, hint):
        self._counter += 1
        return f"{hint}_{self._counter}"

    def node(self, op, inputs, attrs=None, out_hint="t"):
        out = self._fresh(out_hint)
        entry = {"op": op, "inputs": list(inputs), "outputs": [out]}
        if attrs:
            entry["attrs"] = attrs
        self.nodes.append(entry)
        return out

    def weight(self, name, array, is_int32=False):
        """Registers a weight tensor under `name`. Returns `name` for chaining directly into
        `.node(...)`'s `inputs` list.
        """
        if name in self.weights:
            existing = self.weights[name]
            if existing.shape != np.asarray(array).shape:
                raise ValueError(f"weight {name!r} already registered with a different shape")
        else:
            self.weights[name] = np.asarray(array)
            if is_int32:
                self.int32_weights.add(name)
        return name

    def transpose_2d(self, x, out_hint="t2d"):
        """Swaps ne[0]/ne[1] of a 2D tensor: PERMUTE (op only supports exactly 4 axes) + CONT."""
        p = self.node("PERMUTE", [x], {"axes": [1, 0, 2, 3]}, out_hint + "_p")
        return self.node("CONT", [p], None, out_hint)

    def topology(self, inputs, output):
        return {"version": 1, "inputs": inputs, "output": output, "nodes": self.nodes}


def add_conv(tb, prefix, sd, name):
    """Registers a plain (non-weight_norm'd) conv's weight+bias verbatim under `{prefix}.weight`/
    `{prefix}.bias` -- covers kernel_size=1 ("1x1") convs, depthwise convs (groups=channels), and
    kernel_size=3/5 convs alike: PyTorch's own conv weight storage order ((OC or channels), IC-or-1,
    K) already matches ggml's CONV_1D/CONV_1D_DW/MUL_MAT-as-conv1x1 kernel convention byte-for-byte in
    every case encountered this whole VITS effort (see primitives_conv.cpp/primitives_flow.cpp doc
    comments) -- no reshaping needed, just a straight copy.
    """
    tb.weight(f"{prefix}.weight", to_f32(sd[f"{name}.weight"]))
    tb.weight(f"{prefix}.bias", to_f32(sd[f"{name}.bias"]))
    return f"{prefix}.weight", f"{prefix}.bias"


def add_conv1x1_as_matmul(tb, prefix, sd, name):
    """Registers a kernel_size=1 conv's weight+bias for use via a plain `MUL_MAT` node (TextEncoder's
    attention projections and `proj`, SDP's `pre`) instead of `CONV_1D`. PyTorch's own weight shape is
    (out, in, 1) -- unlike `add_conv` (used for genuine CONV_1D consumers, where that trailing K=1 dim
    is required), a MUL_MAT-treated weight needs it SQUEEZED to a plain (out, in) 2D array first: GGUF
    stores tensor dims in the same reversed order ggml uses (numpy shape (d0,d1,d2) -> ggml
    ne=[d2,d1,d0]), so a (out,in,1) array without squeezing would load back as ne=[1,in,out] --
    MUL_MAT would then contract against the wrong axis (ne[0]=1) instead of `in`. Caught by
    test_e2e_vits_smoke's `ggml_can_mul_mat` assertion before this ever reached a numerical check.
    """
    w = to_f32(sd[f"{name}.weight"])
    if w.ndim == 3:
        assert w.shape[-1] == 1, f"{name}.weight: expected a squeezable kernel_size=1 conv, got {w.shape}"
        w = w.reshape(w.shape[0], w.shape[1])
    tb.weight(f"{prefix}.weight", w)
    tb.weight(f"{prefix}.bias", to_f32(sd[f"{name}.bias"]))
    return f"{prefix}.weight", f"{prefix}.bias"


def add_conv_no_bias(tb, prefix, sd, name):
    """Like `add_conv`, for the one conv in this whole model with no bias at all: `dec.conv_post`
    (`nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)`, real code confirmed)."""
    tb.weight(f"{prefix}.weight", to_f32(sd[f"{name}.weight"]))
    return f"{prefix}.weight"


def add_wn_conv(tb, prefix, sd, name):
    """Registers a weight_norm'd conv's FOLDED weight (+ plain bias)."""
    folded = fold_weight_norm(sd[f"{name}.weight_g"], sd[f"{name}.weight_v"])
    tb.weight(f"{prefix}.weight", folded)
    tb.weight(f"{prefix}.bias", to_f32(sd[f"{name}.bias"]))
    return f"{prefix}.weight", f"{prefix}.bias"


def add_layer_norm(tb, prefix, sd, name):
    tb.weight(f"{prefix}.gamma", to_f32(sd[f"{name}.gamma"]))
    tb.weight(f"{prefix}.beta", to_f32(sd[f"{name}.beta"]))
    return f"{prefix}.gamma", f"{prefix}.beta"


def add_dds_conv(tb, x, prefix, sd, name, n_layers, kernel_size, out_hint="dds"):
    """Emits one DDS_CONV node. Input ordering per primitives_flow.cpp's op_dds_conv doc comment:
    x, then per layer [sep_w, sep_b, ln1_g, ln1_b, oo_w, oo_b, ln2_g, ln2_b]. `x` must already be in
    [T, channels] convention.
    """
    inputs = [x]
    for i in range(n_layers):
        sep_w, sep_b = add_conv(tb, f"{prefix}.convs_sep.{i}", sd, f"{name}.convs_sep.{i}")
        ln1_g, ln1_b = add_layer_norm(tb, f"{prefix}.norms_1.{i}", sd, f"{name}.norms_1.{i}")
        oo_w, oo_b = add_conv(tb, f"{prefix}.convs_1x1.{i}", sd, f"{name}.convs_1x1.{i}")
        ln2_g, ln2_b = add_layer_norm(tb, f"{prefix}.norms_2.{i}", sd, f"{name}.norms_2.{i}")
        inputs += [sep_w, sep_b, ln1_g, ln1_b, oo_w, oo_b, ln2_g, ln2_b]
    return tb.node("DDS_CONV", inputs, {"kernel_size": kernel_size, "n_layers": n_layers, "eps": HP["ln_eps"]}, out_hint)


def add_conv_flow_reverse(tb, x, g, prefix, sd, name, out_hint="cf"):
    """Emits one CONV_FLOW_REVERSE node (real ConvFlow always in_channels=2, half_channels=1 in this
    checkpoint -- see primitives_flow.cpp's op_conv_flow_reverse doc comment for the full input
    ordering this reproduces). `x` must be [T, 2]; `g` must be [T, filter_channels].
    """
    n_layers = HP["sdp_ddsconv_n_layers"]
    num_bins = HP["sdp_num_bins"]
    tail_bound = HP["sdp_tail_bound"]
    pre_w, pre_b = add_conv(tb, f"{prefix}.pre", sd, f"{name}.pre")
    inputs = [x, pre_w, pre_b]
    for i in range(n_layers):
        sep_w, sep_b = add_conv(tb, f"{prefix}.convs.convs_sep.{i}", sd, f"{name}.convs.convs_sep.{i}")
        ln1_g, ln1_b = add_layer_norm(tb, f"{prefix}.convs.norms_1.{i}", sd, f"{name}.convs.norms_1.{i}")
        oo_w, oo_b = add_conv(tb, f"{prefix}.convs.convs_1x1.{i}", sd, f"{name}.convs.convs_1x1.{i}")
        ln2_g, ln2_b = add_layer_norm(tb, f"{prefix}.convs.norms_2.{i}", sd, f"{name}.convs.norms_2.{i}")
        inputs += [sep_w, sep_b, ln1_g, ln1_b, oo_w, oo_b, ln2_g, ln2_b]
    proj_w, proj_b = add_conv(tb, f"{prefix}.proj", sd, f"{name}.proj")
    inputs += [proj_w, proj_b]

    # RQ_SPLINE_INVERSE's own baked constants (conversion-time, depend only on num_bins/min_derivative
    # -- see primitives_spline.cpp's doc comment). const = log(exp(1-min_derivative)-1).
    min_derivative = 1e-3
    boundary_const = float(np.log(np.exp(1 - min_derivative) - 1))
    boundary_deriv_const = np.zeros(num_bins + 1, dtype=np.float32)
    boundary_deriv_const[0] = boundary_const
    boundary_deriv_const[-1] = boundary_const
    eps_bump = np.zeros(num_bins, dtype=np.float32)
    eps_bump[-1] = 1e-6
    bdc_name = tb.weight(f"{prefix}.boundary_deriv_const", boundary_deriv_const)
    eps_name = tb.weight(f"{prefix}.eps_bump", eps_bump)
    inputs += [bdc_name, eps_name, g]

    attrs = {"kernel_size": HP["kernel_size"], "n_layers": n_layers, "num_bins": num_bins,
             "tail_bound": tail_bound, "ln_eps": HP["ln_eps"]}
    return tb.node("CONV_FLOW_REVERSE", inputs, attrs, out_hint)


def add_flip(tb, x, channels, out_hint="flip"):
    """`Flip` needs no new primitive: GET_ROWS with a conversion-time-baked reversed-index I32
    constant reverses the channel axis exactly (see BACKLOG.md).
    """
    idx_name = f"const.flip_idx_{channels}"
    tb.weight(idx_name, np.arange(channels - 1, -1, -1, dtype=np.int32), is_int32=True)
    return tb.node("GET_ROWS", [x, idx_name], {}, out_hint)


def build_text_encoder(tb, sd, token_ids_name):
    """Emits TextEncoder's emb -> attentions.Encoder (n_layers unrolled inline, not via repeat_for,
    since each layer's `emb_rel_k`/`emb_rel_v` are declared graph INPUTS -- dynamic-T host-computed
    pad/crop via vits_common.get_relative_embeddings, not baked weights, so they need a per-layer
    declared-input name rather than a uniformly-named weight tensor a repeat_for block could
    reference). Returns (x_cond, stats) output names, both in [C, T] (channel-first) convention.
    """
    h = HP["hidden_channels"]
    n_head = HP["n_heads"]
    head_dim = h // n_head
    n_layers = HP["n_layers"]
    k = HP["kernel_size"]

    emb_w = tb.weight("enc_p.emb.weight", to_f32(sd["enc_p.emb.weight"]))
    x = tb.node("GET_ROWS", [emb_w, token_ids_name], {}, "emb")
    x = tb.node("SCALE", [x], {"s": float(np.sqrt(h))}, "emb_scaled")

    mask_name = "attn_mask"  # declared input, host-filled with zeros (no padding, single utterance)

    for i in range(n_layers):
        p = f"enc_p.encoder.attn_layers.{i}"
        q_w, q_b = add_conv1x1_as_matmul(tb, f"{p}.conv_q", sd, f"{p}.conv_q")
        k_w, k_b = add_conv1x1_as_matmul(tb, f"{p}.conv_k", sd, f"{p}.conv_k")
        v_w, v_b = add_conv1x1_as_matmul(tb, f"{p}.conv_v", sd, f"{p}.conv_v")
        o_w, o_b = add_conv1x1_as_matmul(tb, f"{p}.conv_o", sd, f"{p}.conv_o")

        q = tb.node("MUL_MAT", [q_w, x], None, "q")
        q = tb.node("ADD", [q, q_b], None, "q_b")
        kk = tb.node("MUL_MAT", [k_w, x], None, "k")
        kk = tb.node("ADD", [kk, k_b], None, "k_b")
        v = tb.node("MUL_MAT", [v_w, x], None, "v")
        v = tb.node("ADD", [v, v_b], None, "v_b")
        q = tb.node("RESHAPE", [q], {"shape": [head_dim, n_head, -1]}, "q_r")
        kk = tb.node("RESHAPE", [kk], {"shape": [head_dim, n_head, -1]}, "k_r")
        v = tb.node("RESHAPE", [v], {"shape": [head_dim, n_head, -1]}, "v_r")

        # The declared inputs below (dynamic-T, host-computed via vits_common.get_relative_embeddings)
        # are what the topology actually reads at build/compute time. The RAW learned tables are also
        # registered here under a DIFFERENT name, `*_raw` -- not referenced by any node, so the
        # topology JSON never mentions them, but still written to the GGUF's tensor table purely so
        # the C++ driver can read the fixed-size learned parameter back out via GgufModel::weight(...)
        # and compute the per-call dynamic-T table itself (real inference T varies per input text; the
        # padding/cropping can't be baked in at conversion time).
        tb.weight(f"{p}.emb_rel_k_raw", to_f32(sd[f"{p}.emb_rel_k"]).reshape(2 * HP["window_size"] + 1, -1))
        tb.weight(f"{p}.emb_rel_v_raw", to_f32(sd[f"{p}.emb_rel_v"]).reshape(2 * HP["window_size"] + 1, -1))

        emb_rel_k_name = f"emb_rel_k_{i}"  # declared input: host-computed dynamic-T table
        emb_rel_v_name = f"emb_rel_v_{i}"
        attn = tb.node("REL_POS_ATTENTION_SHAW", [q, kk, v, emb_rel_k_name, emb_rel_v_name, mask_name],
                        {"scale": 1.0 / float(np.sqrt(head_dim))}, "attn")
        o = tb.node("MUL_MAT", [o_w, attn], None, "o")
        o = tb.node("ADD", [o, o_b], None, "o_b")

        x = tb.node("ADD", [x, o], None, "res1")
        x = tb.node("LAYER_NORM", [x], {"eps": HP["ln_eps"]}, "ln1_normed")
        ln1_g, ln1_b = add_layer_norm(tb, f"enc_p.encoder.norm_layers_1.{i}", sd, f"enc_p.encoder.norm_layers_1.{i}")
        x = tb.node("MUL", [x, ln1_g], None, "ln1_mul")
        x = tb.node("ADD", [x, ln1_b], None, "ln1")

        ff1_w, ff1_b = add_conv(tb, f"enc_p.encoder.ffn_layers.{i}.conv_1", sd, f"enc_p.encoder.ffn_layers.{i}.conv_1")
        ff2_w, ff2_b = add_conv(tb, f"enc_p.encoder.ffn_layers.{i}.conv_2", sd, f"enc_p.encoder.ffn_layers.{i}.conv_2")
        xt = tb.transpose_2d(x, "ffn_xt")  # [C,T] -> [T,C] for CONV_1D
        hft = tb.node("CONV_1D", [ff1_w, tb.node("RESHAPE", [xt], {"shape": [-1, h, 1]}, "ffn_xt3")],
                       {"s0": 1, "p0": (k - 1) // 2, "d0": 1}, "ffn_h")
        hft = tb.node("ADD", [hft, tb.node("RESHAPE", [ff1_b], {"shape": [1, HP["filter_channels"], 1]}, "ffn_b1_r")],
                       None, "ffn_h_b")
        hft = tb.node("RELU", [hft], None, "ffn_relu")
        hft2 = tb.node("CONV_1D", [ff2_w, hft], {"s0": 1, "p0": (k - 1) // 2, "d0": 1}, "ffn_h2")
        hft2 = tb.node("ADD", [hft2, tb.node("RESHAPE", [ff2_b], {"shape": [1, h, 1]}, "ffn_b2_r")], None, "ffn_h2_b")
        ffn_out = tb.node("RESHAPE", [hft2], {"shape": [-1, h]}, "ffn_out_2d")
        ffn_out = tb.transpose_2d(ffn_out, "ffn_out_ct")  # back to [C,T]

        x = tb.node("ADD", [x, ffn_out], None, "res2")
        x = tb.node("LAYER_NORM", [x], {"eps": HP["ln_eps"]}, "ln2_normed")
        ln2_g, ln2_b = add_layer_norm(tb, f"enc_p.encoder.norm_layers_2.{i}", sd, f"enc_p.encoder.norm_layers_2.{i}")
        x = tb.node("MUL", [x, ln2_g], None, "ln2_mul")
        x = tb.node("ADD", [x, ln2_b], None, "ln2")

    proj_w, proj_b = add_conv1x1_as_matmul(tb, "enc_p.proj", sd, "enc_p.proj")
    stats = tb.node("MUL_MAT", [proj_w, x], None, "stats")
    stats = tb.node("ADD", [stats, proj_b], None, "stats_b")
    return x, stats


def build_sdp_reverse(tb, x_cond, sd):
    """Emits StochasticDurationPredictor's reverse-mode flow chain (see BACKLOG.md: applied order is
    [Flip3, CF3, Flip2, CF2, Flip1, CF1, Flip0, EA], `dp.flows.1`/`dp.log_flow` never touched).
    `z_noise` (a declared input: host samples `randn(T,2)*noise_scale`) is the SDP's own initial
    noise, already in [T,2] convention. Returns the `logw` output name, shape [T, 1].
    """
    ddsconv_n = HP["sdp_ddsconv_n_layers"]
    filt = HP["hidden_channels"]  # SDP forces filter_channels = in_channels internally

    pre_w, pre_b = add_conv1x1_as_matmul(tb, "dp.pre", sd, "dp.pre")
    h = tb.node("MUL_MAT", [pre_w, x_cond], None, "dp_pre")  # [C,T], same convention as x_cond
    h = tb.node("ADD", [h, pre_b], None, "dp_pre_b")
    h = tb.transpose_2d(h, "dp_pre_tc")  # -> [T, filt] for DDS_CONV's own convention

    h2 = add_dds_conv(tb, h, "dp.convs", sd, "dp.convs", ddsconv_n, HP["kernel_size"], "dp_convs")  # [T, filt]

    proj_w, proj_b = add_conv(tb, "dp.proj", sd, "dp.proj")
    h2_3d = tb.node("RESHAPE", [h2], {"shape": [-1, filt, 1]}, "dp_convs_3d")
    g_cond = tb.node("CONV_1D", [proj_w, h2_3d], {"s0": 1, "p0": 0, "d0": 1}, "dp_proj")
    g_cond = tb.node("ADD", [g_cond, tb.node("RESHAPE", [proj_b], {"shape": [1, filt, 1]}, "dp_proj_b_r")],
                      None, "dp_proj_b")
    g_cond = tb.node("RESHAPE", [g_cond], {"shape": [-1, filt]}, "g_cond")  # [T, filt], CONV_FLOW_REVERSE's own `g`

    z = "z_noise"  # declared input [T, 2]
    z = add_flip(tb, z, 2, "z_flip3")
    z = add_conv_flow_reverse(tb, z, g_cond, "dp.flows.7", sd, "dp.flows.7", "z_cf3")
    z = add_flip(tb, z, 2, "z_flip2")
    z = add_conv_flow_reverse(tb, z, g_cond, "dp.flows.5", sd, "dp.flows.5", "z_cf2")
    z = add_flip(tb, z, 2, "z_flip1")
    z = add_conv_flow_reverse(tb, z, g_cond, "dp.flows.3", sd, "dp.flows.3", "z_cf1")
    z = add_flip(tb, z, 2, "z_flip0")
    m = tb.weight("dp.flows.0.m_flat", to_f32(sd["dp.flows.0.m"]).reshape(-1))
    logs = tb.weight("dp.flows.0.logs_flat", to_f32(sd["dp.flows.0.logs"]).reshape(-1))
    z = tb.node("ELEMENTWISE_AFFINE_REVERSE", [z, m, logs], {}, "z_ea")

    logw = tb.node("VIEW", [z], {"shape": ["$n_tokens", 1], "offset": 0}, "logw")
    return logw


def build_text_sdp_topologies(sd):
    """Returns (stats_topology, logw_topology, weights) -- GraphTopology supports exactly one
    declared output per topology (see graph_topology.h), so `stats` (TextEncoder's proj output, m/
    logs split host-side by channel offset) and `logw` (needs the SDP's own flow chain on top) are
    two separate topologies sharing one GGUF weight set, mirroring TdtDecoder's established
    multi-topology-per-model precedent.
    """
    k_channels = HP["hidden_channels"] // HP["n_heads"]
    token_input = {"name": "tokens", "dtype": "i32", "shape": ["$n_tokens"]}
    mask_input = {"name": "attn_mask", "dtype": "f32", "shape": ["$n_tokens", "$n_tokens"]}
    rel_inputs = []
    for i in range(HP["n_layers"]):
        rel_inputs.append({"name": f"emb_rel_k_{i}", "dtype": "f32", "shape": [str(k_channels), "2*$n_tokens-1"]})
        rel_inputs.append({"name": f"emb_rel_v_{i}", "dtype": "f32", "shape": [str(k_channels), "2*$n_tokens-1"]})

    tb1 = TopologyBuilder()
    _, stats = build_text_encoder(tb1, sd, "tokens")
    stats_topo = tb1.topology([token_input, mask_input] + rel_inputs, stats)

    tb2 = TopologyBuilder()
    x_cond, _ = build_text_encoder(tb2, sd, "tokens")
    logw = build_sdp_reverse(tb2, x_cond, sd)
    z_input = {"name": "z_noise", "dtype": "f32", "shape": ["$n_tokens", "2"]}
    logw_topo = tb2.topology([token_input, mask_input] + rel_inputs + [z_input], logw)

    # GgufModel::load requires exactly one "model.graph_topology" KV per file (see gguf_model.cpp),
    # so `stats` and `logw` -- GraphTopology supports only one declared output each -- become two
    # INDEPENDENT GGUF files below, each with its own full (partially redundant) TextEncoder weight
    # copy, mirroring convert_parakeet_tdt.py's established "one file per topology, weights duplicated
    # across files as needed" precedent (not a single file with multiple custom-named topology KVs,
    # which GgufModel::load doesn't support).
    return stats_topo, logw_topo, tb1.weights, tb1.int32_weights, tb2.weights, tb2.int32_weights


def build_flow_vocoder_topology(sd):
    """Flow (ResidualCouplingBlock, reverse) + HiFi-GAN Generator. Single topology, single output
    (the waveform) -- `z_p` (frame-expanded, noise-sampled prior) is a declared input, computed
    host-side (frame expansion via `generate_path` needs the duration predictor's own output length,
    a genuinely data-dependent value -- see BACKLOG.md's task #71 note). This topology's own
    `$n_tokens` (its `build()` call's dynamic-length argument) IS that host-computed frame count.
    """
    tb = TopologyBuilder()
    channels = HP["inter_channels"]
    k = HP["flow_kernel_size"]
    dil = HP["flow_dilation_rate"]
    wn_n_layers = HP["flow_wn_n_layers"]
    n_flows = HP["flow_n_flows"]

    z = "z_p"  # declared input [T, channels]
    for flow_idx in reversed(range(n_flows)):  # reversed(self.flows): [Flip3,RCL3,Flip2,RCL2,Flip1,RCL1,Flip0,RCL0]
        z = add_flip(tb, z, channels, f"flow_flip_{flow_idx}")
        rcl_idx = 2 * flow_idx
        prefix = f"flow.flows.{rcl_idx}"
        pre_w, pre_b = add_conv(tb, f"{prefix}.pre", sd, f"{prefix}.pre")
        inputs = [z, pre_w, pre_b]
        for i in range(wn_n_layers):
            in_w, in_b = add_wn_conv(tb, f"{prefix}.enc.in_layers.{i}", sd, f"{prefix}.enc.in_layers.{i}")
            rs_w, rs_b = add_wn_conv(tb, f"{prefix}.enc.res_skip_layers.{i}", sd, f"{prefix}.enc.res_skip_layers.{i}")
            inputs += [in_w, in_b, rs_w, rs_b]
        post_w, post_b = add_conv(tb, f"{prefix}.post", sd, f"{prefix}.post")
        inputs += [post_w, post_b]
        attrs = {"kernel_size": k, "dilation_rate": dil, "n_layers": wn_n_layers}
        z = tb.node("RESIDUAL_COUPLING_LAYER_REVERSE", inputs, attrs, "rcl")

    # HiFi-GAN Generator. `z` is [T, channels] -- transpose to CONV_1D's [T,C,N] 3D form directly
    # (already the right axis order, just needs the batch dim).
    conv_pre_w, conv_pre_b = add_conv(tb, "dec.conv_pre", sd, "dec.conv_pre")
    z3 = tb.node("RESHAPE", [z], {"shape": [-1, channels, 1]}, "gen_in")
    x = tb.node("CONV_1D", [conv_pre_w, z3], {"s0": 1, "p0": 3, "d0": 1}, "conv_pre_out")
    x = tb.node("ADD", [x, tb.node("RESHAPE", [conv_pre_b], {"shape": [1, HP["upsample_initial_channel"], 1]}, "cpb_r")],
                 None, "conv_pre_b")
    x = tb.node("RESHAPE", [x], {"shape": [-1, HP["upsample_initial_channel"]]}, "conv_pre_2d")

    upsample_rates = HP["upsample_rates"]
    upsample_kernel_sizes = HP["upsample_kernel_sizes"]
    resblock_kernel_sizes = HP["resblock_kernel_sizes"]
    resblock_dilation_sizes = HP["resblock_dilation_sizes"]
    num_kernels = len(resblock_kernel_sizes)
    upsample_initial_channel = HP["upsample_initial_channel"]

    running_product = 1
    for stage in range(len(upsample_rates)):
        u = upsample_rates[stage]
        kk = upsample_kernel_sizes[stage]
        pad = (kk - u) // 2
        ch_out = upsample_initial_channel // (2 ** (stage + 1))
        up_w, up_b = add_wn_conv(tb, f"dec.ups.{stage}", sd, f"dec.ups.{stage}")

        x = tb.node("LEAKY_RELU", [x], {"slope": 0.1}, "up_lrelu")
        x_full = tb.node("CONV_TRANSPOSE_1D", [up_w, x], {"s0": u}, "up_full")  # [T,channels] 2D (batch-less)
        x_full = tb.node("ADD", [x_full, tb.node("RESHAPE", [up_b], {"shape": [1, ch_out]}, "up_b_r")],
                          None, "up_biased")
        running_product *= u
        crop_shape_expr = f"$n_tokens*{running_product}"
        x = tb.node("VIEW", [x_full], {"shape": [crop_shape_expr, ch_out], "offset": pad * 4}, "up_cropped")

        summed = None
        for j in range(num_kernels):
            resblock_idx = stage * num_kernels + j
            rk = resblock_kernel_sizes[j]
            d0, d1 = resblock_dilation_sizes[j]
            prefix = f"dec.resblocks.{resblock_idx}"
            c0_w, c0_b = add_wn_conv(tb, f"{prefix}.convs.0", sd, f"{prefix}.convs.0")
            c1_w, c1_b = add_wn_conv(tb, f"{prefix}.convs.1", sd, f"{prefix}.convs.1")

            y = tb.node("LEAKY_RELU", [x], {"slope": 0.1}, "rb_lrelu1")
            y3 = tb.node("RESHAPE", [y], {"shape": [-1, ch_out, 1]}, "rb_y3")
            y = tb.node("CONV_1D", [c0_w, y3], {"s0": 1, "p0": (rk * d0 - d0) // 2, "d0": d0}, "rb_c0")
            y = tb.node("ADD", [y, tb.node("RESHAPE", [c0_b], {"shape": [1, ch_out, 1]}, "rb_c0b_r")], None, "rb_c0_b")
            y = tb.node("RESHAPE", [y], {"shape": [-1, ch_out]}, "rb_c0_2d")
            xres = tb.node("ADD", [y, x], None, "rb_res1")

            y2 = tb.node("LEAKY_RELU", [xres], {"slope": 0.1}, "rb_lrelu2")
            y2_3 = tb.node("RESHAPE", [y2], {"shape": [-1, ch_out, 1]}, "rb_y2_3")
            y2 = tb.node("CONV_1D", [c1_w, y2_3], {"s0": 1, "p0": (rk * d1 - d1) // 2, "d0": d1}, "rb_c1")
            y2 = tb.node("ADD", [y2, tb.node("RESHAPE", [c1_b], {"shape": [1, ch_out, 1]}, "rb_c1b_r")], None, "rb_c1_b")
            y2 = tb.node("RESHAPE", [y2], {"shape": [-1, ch_out]}, "rb_c1_2d")
            rb_out = tb.node("ADD", [y2, xres], None, "rb_res2")
            summed = rb_out if summed is None else tb.node("ADD", [summed, rb_out], None, "rb_sum")
        x = tb.node("SCALE", [summed], {"s": 1.0 / num_kernels}, "rb_avg")

    conv_post_w = add_conv_no_bias(tb, "dec.conv_post", sd, "dec.conv_post")
    # conv_post has no bias in the real module (`nn.Conv1d(ch,1,7,1,padding=3,bias=False)`).
    # `ch_out` still holds the LAST upsample stage's output channel count (Python loop variables
    # survive their loop) -- exactly conv_post's expected input channel count.
    x = tb.node("LEAKY_RELU", [x], {"slope": 0.01}, "final_lrelu")
    x3 = tb.node("RESHAPE", [x], {"shape": [-1, ch_out, 1]}, "final_3d")
    x = tb.node("CONV_1D", [conv_post_w, x3], {"s0": 1, "p0": 3, "d0": 1}, "wav_pre_tanh")
    wav = tb.node("TANH", [x], {}, "wav")
    wav = tb.node("RESHAPE", [wav], {"shape": [-1]}, "wav_1d")

    z_input = {"name": "z_p", "dtype": "f32", "shape": ["$n_tokens", str(channels)]}
    topo = tb.topology([z_input], wav)
    return topo, tb.weights, tb.int32_weights


def write_gguf(path, architecture, hparams, topology, weights, int32_names=()):
    w = GGUFWriter(str(path), architecture)
    w.add_string("loom.architecture", architecture)
    for key, value in hparams.items():
        if isinstance(value, float):
            w.add_float32(f"loom.{key}", value)
        elif isinstance(value, int):
            w.add_uint32(f"loom.{key}", value)
        else:
            w.add_string(f"loom.{key}", str(value))
    w.add_string("model.graph_topology", json.dumps(topology))
    for name, arr in weights.items():
        if name in int32_names:
            w.add_tensor(name, arr.astype(np.int32))
        else:
            w.add_tensor(name, arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {path} ({len(weights)} tensors)")


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <model.ckpt> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    full_sd = load_piper_checkpoint(ckpt_path)
    sd = {k[len("model_g."):]: v for k, v in full_sd.items() if k.startswith("model_g.")}

    stats_topo, logw_topo, stats_weights, stats_int32, logw_weights, logw_int32 = build_text_sdp_topologies(sd)
    write_gguf(out_dir / "vits_stats.gguf", "vits_stats", HP, stats_topo, stats_weights, stats_int32)
    write_gguf(out_dir / "vits_logw.gguf", "vits_logw", HP, logw_topo, logw_weights, logw_int32)

    flow_vocoder_topo, flow_vocoder_weights, flow_vocoder_int32 = build_flow_vocoder_topology(sd)
    write_gguf(out_dir / "vits_flow_vocoder.gguf", "vits_flow_vocoder", HP,
               flow_vocoder_topo, flow_vocoder_weights, flow_vocoder_int32)


if __name__ == "__main__":
    main()
