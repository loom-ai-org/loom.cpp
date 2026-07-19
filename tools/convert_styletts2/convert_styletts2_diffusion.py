"""Converts StyleTTS2's style-diffusion denoiser network: the plain `Transformer1d` (real source:
`Modules/diffusion/modules.py`) that `build_model` substitutes in place of `AudioDiffusionConditional`'s
own default deep multi-scale U-Net (`diffusion.unet = transformer` -- confirmed directly against the real
checkpoint's state dict, which has NO U-Net-shaped keys at all, only `unet.blocks.{0,1,2}.*`/`to_time`/
`to_mapping`/`to_out`/`fixed_embedding`, exactly matching `Transformer1d`'s own module list). This is the
ONE genuinely new architecture piece StyleTTS2 needed beyond what Kokoro already forced out (see PLAN.md).

`config.yml`'s `multispeaker: false` confirms `StyleTransformer1d` (the AdaLayerNorm/style-conditioned
variant) is NOT used here -- only the plain `Transformer1d`, confirmed further by the real key list having
no `to_features.*` (which would exist if `context_features` had been passed to it -- it wasn't, per
`build_model`'s own non-multispeaker branch).

KDiffusion's own `c_skip`/`c_out`/`c_in`/`c_noise` preconditioning (real source:
Modules/diffusion/sampler.py's `KDiffusion.get_scale_weights`/`denoise_fn`) is DELIBERATELY NOT built as
graph nodes here: `sigma` is always a plain host-known float scalar at each ADPM2 sampling step (from the
Karras schedule), so all four quantities are computed as ordinary host floats in
`StyleTTS2Driver`/`style_diffusion_sampler`'s own `DenoiseFn` callback -- `c_in` scales `x_noisy` BEFORE
this graph runs, `c_noise` is fed in as the `time` input directly, and `c_skip`/`c_out` combine this
graph's raw output back into `x_denoised` AFTER it returns. Same "host does small scalar math, graph does
the real tensor work" precedent as VITS's SDP/Kokoro's SineGen noise-amplitude scaling.

Real per-block math (`TransformerBlock.forward`, NO cross-attention since `context_features` was never
passed to `Attention.__init__` in this non-multispeaker config -- confirmed the real key list has no
`cross_attention.*` at all): `x = self.attention(x) + x; x = self.feed_forward(x) + x`. `Attention.forward`
normalizes Q from `x` via one LayerNorm (`.norm`) and K/V from the SAME `x` via a SEPARATE, independently
learned LayerNorm (`.norm_context`) -- `context = default(context, x)` when no real cross-attention
context is given -- a real quirk (two different learned affine transforms of the same input), not
redundant, not simplified away.

Axis convention: `embedding` (StyleTTS2's own name for the raw `bert_dur` BERT hidden states, i.e.
CustomAlbert's own `last_hidden_state`, UNPROJECTED by `bert_encoder`) is fed in directly at
`ne=[768,T]` -- Layout B, byte-identical to CustomAlbert's own raw output convention (see
convert_kokoro_bert_encoder.py's own derivation of this same axis fact). `x` (the single per-utterance
noisy style "pseudo-token", `ne=[256]`) is broadcast to `[256,T]` via the new REPEAT primitive (added
specifically for this -- CONCAT requires matching shape on every non-concat axis, so there is no way to
avoid materializing the broadcast before concatenating with `embedding` along the channel axis) then
CONCAT'd with `embedding` (channel order `[x(256), embedding(768)]`, matching the real
`torch.cat([x.expand(...), embedding], axis=-1)` operand order exactly -- getting this backwards would
silently transpose which weight columns apply to which half).

The final `mean(axis=1)` (real: mean over the TOKEN axis) is built as PERMUTE+CONT (channel-first ->
time-first) + SUM_ROWS (reduces over ne[0], now T) + SCALE(1/T) -- the mirror-image of every other
`[C,T]<->[T,C]` boundary crossing already used throughout this project. `to_out`'s Conv1d(1024,256,
kernel=1) applied to a single ("mean-pooled") position degenerates to a plain Linear (MUL_MAT+bias, kernel
dim folded away at conversion time) -- same "1x1-conv-as-matmul" precedent as Kokoro's F0_proj/N_proj.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

HP = {
    "channels": 256,          # style_dim*2
    "context_embedding_features": 768,
    "num_layers": 3,
    "num_heads": 8,
    "head_features": 64,
    "mid_features": 512,      # num_heads * head_features
    "multiplier": 2,
    "ln_eps": 1e-5,           # nn.LayerNorm's own default
    "sigma_data": 0.45731624995853165,  # config.yml's model_params.diffusion.dist.sigma_data
}


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

    def node(self, op, inputs, attrs=None, out_hint="t"):
        out = self._fresh(out_hint)
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

    def topology(self, inputs, output):
        return {"version": 1, "inputs": inputs, "output": output, "nodes": self.nodes}


def add_linear(tb, prefix, sd, name, has_bias=True):
    w = tb.weight(f"{prefix}.weight", to_f32(sd[f"{name}.weight"]))
    b = tb.weight(f"{prefix}.bias", to_f32(sd[f"{name}.bias"])) if has_bias else None
    return w, b


def apply_layer_norm(tb, x, prefix, sd, name, eps, out_hint):
    normed = tb.node("LAYER_NORM", [x], {"eps": eps}, f"{out_hint}_normed")
    g = tb.weight(f"{prefix}.gamma", to_f32(sd[f"{name}.weight"]))
    b = tb.weight(f"{prefix}.beta", to_f32(sd[f"{name}.bias"]))
    xm = tb.node("MUL", [normed, g], None, f"{out_hint}_mul")
    return tb.node("ADD", [xm, b], None, out_hint)


def build_diffusion_net(tb, sd, sd_prefix, hp):
    """Returns the output name, ne=[channels] (a single 256-vector: the raw, un-preconditioned network
    prediction `x_pred` -- KDiffusion's c_skip/c_out combination happens in the host driver, not here)."""
    channels = hp["channels"]
    ctx_feat = hp["context_embedding_features"]
    features = channels + ctx_feat  # 1024
    n_head = hp["num_heads"]
    head_dim = hp["head_features"]
    mid = hp["mid_features"]
    eps = hp["ln_eps"]

    def p(name):
        return f"{sd_prefix}.{name}"

    # --- time embedding: LearnedPositionalEmbedding(dim=channels) -> Linear(dim+1,features) -> GELU ---
    half_dim = channels // 2
    lpe_weights = tb.weight("diff.to_time.lpe_weights", to_f32(sd[p("to_time.0.0.weights")]))  # [half_dim]
    freqs_raw = tb.node("MUL", [lpe_weights, "time"], None, "freqs_raw")  # [half_dim] (time[1] broadcasts)
    freqs = tb.node("SCALE", [freqs_raw], {"s": float(2.0 * np.pi)}, "freqs")
    sin_f = tb.node("SIN", [freqs], None, "sin_f")
    cos_f = tb.node("COS", [freqs], None, "cos_f")
    fouriered = tb.node("CONCAT", [sin_f, cos_f], {"dim": 0}, "fouriered")           # [channels]
    fouriered_full = tb.node("CONCAT", ["time", fouriered], {"dim": 0}, "fouriered_full")  # [channels+1]

    tt_w, tt_b = add_linear(tb, "diff.to_time.linear", sd, p("to_time.0.1"))
    time_emb = tb.node("ADD", [tb.node("MUL_MAT", [tt_w, fouriered_full], None, "tt_mm"), tt_b], None, "tt_biased")
    time_emb = tb.node("GELU", [time_emb], None, "time_emb")  # [features] (== to_time's own output width)

    # --- to_mapping: 2x (Linear(features,features) + GELU) -- use_context_features=False, so `mapping`
    #     is JUST to_mapping(time_emb), no extra summed item. ---
    m0_w, m0_b = add_linear(tb, "diff.to_mapping.0", sd, p("to_mapping.0"))
    mapping = tb.node("ADD", [tb.node("MUL_MAT", [m0_w, time_emb], None, "map0_mm"), m0_b], None, "map0_biased")
    mapping = tb.node("GELU", [mapping], None, "map0_gelu")
    m2_w, m2_b = add_linear(tb, "diff.to_mapping.2", sd, p("to_mapping.2"))
    mapping = tb.node("ADD", [tb.node("MUL_MAT", [m2_w, mapping], None, "map2_mm"), m2_b], None, "map2_biased")
    mapping = tb.node("GELU", [mapping], None, "mapping")  # [features]

    # --- x_full = cat([x.expand(T), embedding], channel-axis) : channel order [x(256), embedding(768)] ---
    x_rep = tb.node("REPEAT", ["x_in"], {"shape": [channels, "$n_tokens"]}, "x_rep")  # [channels, T]
    x_full = tb.node("CONCAT", [x_rep, "embedding"], {"dim": 0}, "x_full")  # [features, T]

    for i in range(hp["num_layers"]):
        bp = f"blocks.{i}"
        x_full = tb.node("ADD", [x_full, mapping], None, f"blk{i}_premap")  # mapping[features] broadcasts over T

        # --- self-attention: Q normed via .norm, K/V normed via a SEPARATE .norm_context (same input x_full) ---
        qn = apply_layer_norm(tb, x_full, f"diff.blk{i}.attn.norm", sd, p(f"{bp}.attention.norm"), eps, f"blk{i}_qn")
        kvn = apply_layer_norm(tb, x_full, f"diff.blk{i}.attn.norm_ctx", sd, p(f"{bp}.attention.norm_context"),
                                eps, f"blk{i}_kvn")

        qw, _ = add_linear(tb, f"diff.blk{i}.attn.to_q", sd, p(f"{bp}.attention.to_q"), has_bias=False)
        kvw, _ = add_linear(tb, f"diff.blk{i}.attn.to_kv", sd, p(f"{bp}.attention.to_kv"), has_bias=False)
        q = tb.node("MUL_MAT", [qw, qn], None, f"blk{i}_q")       # [mid, T]
        kv = tb.node("MUL_MAT", [kvw, kvn], None, f"blk{i}_kv")   # [2*mid, T]
        k = tb.node("CONT", [tb.node("VIEW", [kv], {"shape": [mid, "$n_tokens"], "offset": 0}, f"blk{i}_k_v")],
                    None, f"blk{i}_k")
        v = tb.node("CONT", [tb.node("VIEW", [kv], {"shape": [mid, "$n_tokens"], "offset": mid * 4}, f"blk{i}_v_v")],
                    None, f"blk{i}_v")

        q_r = tb.node("RESHAPE", [q], {"shape": [head_dim, n_head, "$n_tokens"]}, f"blk{i}_q_r")
        k_r = tb.node("RESHAPE", [k], {"shape": [head_dim, n_head, "$n_tokens"]}, f"blk{i}_k_r")
        v_r = tb.node("RESHAPE", [v], {"shape": [head_dim, n_head, "$n_tokens"]}, f"blk{i}_v_r")

        attn = tb.node("ATTENTION", [q_r, k_r, v_r, "attn_mask"],
                        {"kv_cache": False, "scale": 1.0 / float(np.sqrt(head_dim))}, f"blk{i}_attn")  # [mid, T]

        ow, ob = add_linear(tb, f"diff.blk{i}.attn.to_out", sd, p(f"{bp}.attention.attention.to_out"))
        attn_proj = tb.node("ADD", [tb.node("MUL_MAT", [ow, attn], None, f"blk{i}_o_mm"), ob], None, f"blk{i}_o")
        x_full = tb.node("ADD", [x_full, attn_proj], None, f"blk{i}_res1")

        # --- feed_forward: Linear(features,mid_ff)+GELU+Linear(mid_ff,features) ---
        f0w, f0b = add_linear(tb, f"diff.blk{i}.ff.0", sd, p(f"{bp}.feed_forward.0"))
        ff = tb.node("ADD", [tb.node("MUL_MAT", [f0w, x_full], None, f"blk{i}_ff0_mm"), f0b], None, f"blk{i}_ff0")
        ff = tb.node("GELU", [ff], None, f"blk{i}_ff_gelu")
        f2w, f2b = add_linear(tb, f"diff.blk{i}.ff.2", sd, p(f"{bp}.feed_forward.2"))
        ff = tb.node("ADD", [tb.node("MUL_MAT", [f2w, ff], None, f"blk{i}_ff2_mm"), f2b], None, f"blk{i}_ff2")
        x_full = tb.node("ADD", [x_full, ff], None, f"blk{i}_res2")

    # --- mean over T (token axis), then to_out: Rearrange + Conv1d(features,channels,k=1) == a plain Linear ---
    xp = tb.node("PERMUTE", [x_full], {"axes": [1, 0, 2, 3]}, "mean_in_p")
    xc = tb.node("CONT", [xp], None, "mean_in")               # [T, features]
    summed = tb.node("SUM_ROWS", [xc], None, "mean_summed")    # [1, features]
    mean = tb.node("SCALE", [summed], {"s": "1/$n_tokens"}, "mean_scaled")
    mean_vec = tb.node("RESHAPE", [mean], {"shape": [features]}, "mean_vec")

    to_out_w_raw = to_f32(sd[p("to_out.1.weight")])  # (channels, features, 1) -- Conv1d kernel=1
    to_out_w = tb.weight("diff.to_out.weight", to_out_w_raw.reshape(channels, features))
    to_out_b = tb.weight("diff.to_out.bias", to_f32(sd[p("to_out.1.bias")]))
    model_out = tb.node("ADD", [tb.node("MUL_MAT", [to_out_w, mean_vec], None, "to_out_mm"), to_out_b], None,
                         "model_out")  # [channels]
    return model_out


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <epoch_2nd_00100.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    net = sd_all["net"] if "net" in sd_all else sd_all
    sd = net["diffusion"]

    tb = TopologyBuilder()
    out = build_diffusion_net(tb, sd, "module.unet", HP)

    inputs = [
        {"name": "x_in", "dtype": "f32", "shape": [str(HP["channels"])]},
        {"name": "time", "dtype": "f32", "shape": ["1"]},
        {"name": "embedding", "dtype": "f32", "shape": [str(HP["context_embedding_features"]), "$n_tokens"]},
        {"name": "attn_mask", "dtype": "f32", "shape": ["$n_tokens", "$n_tokens"]},
    ]
    topo = tb.topology(inputs, out)

    writer = GGUFWriter(str(out_dir / "styletts2_diffusion.gguf"), "loom-styletts2-diffusion")
    writer.add_string("model.graph_topology", json.dumps(topo))
    for name, arr in tb.weights.items():
        writer.add_tensor(name, arr.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {out_dir / 'styletts2_diffusion.gguf'}, {len(tb.weights)} weights")


if __name__ == "__main__":
    main()
