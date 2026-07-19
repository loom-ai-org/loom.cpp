"""Converts Kokoro's `CustomAlbert` (PL-BERT phoneme-conditioning transformer, a real HF `AlbertModel`
subclass that just returns `last_hidden_state`) into a standalone loom-engine GGUF, verified in isolation
before assembling the rest of Kokoro around it (DurationEncoder/ProsodyPredictor/TextEncoder/Decoder all
depend on its output, but it has no dependency on any style/prosody conditioning itself, so it's the
cleanest first piece -- same "verify each stage before trusting it in the whole" discipline as every
other model in this project).

Real architecture confirmed directly against `transformers/models/albert/{configuration,modeling}_albert
.py`'s source (reading the .py files off disk directly -- actually IMPORTING `transformers` is blocked in
this venv by the same broken huggingface-hub version pin already hit for NeMo's own toolkit) and the real
checkpoint's own state dict + `config.json`'s `plbert` block:
  - `AlbertConfig(vocab_size=178, hidden_size=768, num_attention_heads=12, intermediate_size=2048,
    max_position_embeddings=512, num_hidden_layers=12, dropout=0.1)` -- every other AlbertConfig field is
    HF's own default: `embedding_size=128` (embedding factorization -- 128-dim embeddings,
    `embedding_hidden_mapping_in` Linear projects up to `hidden_size=768`), `num_hidden_groups=1` (REAL
    cross-layer parameter sharing -- confirmed via the state dict having exactly ONE
    `encoder.albert_layer_groups.0.albert_layers.0.*`, applied 12 times), `hidden_act="gelu_new"` (a
    tanh-approximation GELU, algebraically IDENTICAL to `ggml_gelu_f32`'s own formula but NOT the same as
    this project's `GELU` primitive, which is bound to the exact erf-based formula -- composed here
    directly from `TANH`/`SQR`/`MUL`/`ADD`/`SCALE` instead of reusing either existing GELU primitive,
    since ggml's own tanh-GELU (`ggml_gelu`) unconditionally routes through an imprecise F16 lookup table
    on CPU, confirmed by reading `ggml-cpu/vec.h` directly), `layer_norm_eps=1e-12` (NOT this project's
    usual 1e-5), `position_embedding_type="absolute"` (no relative-position complexity at all).
  - Post-LN throughout (standard BERT/ALBERT convention, confirmed from `AlbertAttention.forward`/
    `AlbertLayer.forward` directly): `attention.LayerNorm(hidden_states + dense(attn_ctx))`, then
    SEPARATELY `full_layer_layer_norm(hidden_states + ffn_output(gelu_new(ffn(hidden_states))))`.
  - `token_type_ids` defaults to all-zeros (HF's own registered-buffer default, confirmed from
    `AlbertEmbeddings.forward`) -- since this project's real usage (Kokoro's `forward_with_tokens`) never
    passes `token_type_ids` either, the token-type embedding is always just row 0 of
    `token_type_embeddings.weight`, baked here as a plain constant added once (not a per-call GET_ROWS).
  - Real usage is always a single, unpadded utterance (batch_size=1, no padding) -- the additive
    attention mask HF would otherwise compute from a 0/1 padding mask is always all-zeros here, same
    "host-filled all-zeros, no real masking" precedent as VITS/Whisper's encoder.

This is verified standalone (own GGUF, own declared output = last_hidden_state at hidden_size=768) --
NOT yet wired to `bert_encoder`'s own downstream Linear(768,512) or anything else in KModel, which is
applied on top once this piece is trusted.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

HP = {
    "vocab_size": 178,
    "embedding_size": 128,
    "hidden_size": 768,
    "n_head": 12,
    "head_dim": 768 // 12,
    "intermediate_size": 2048,
    "n_layer": 12,  # logical layers, ALL sharing the SAME physical weights (num_hidden_groups=1)
    "ln_eps": 1e-12,
}


class TopologyBuilder:
    def __init__(self):
        self.nodes = []
        self.weights = {}
        self._counter = 0

    def _fresh(self, hint):
        self._counter += 1
        return f"{hint}_{self._counter}"

    def node(self, op, inputs, attrs=None, out_hint="t", name=None):
        # `name`, when given, is used VERBATIM as the output name instead of auto-freshening --
        # required inside a `repeat_for` block for any tensor that must carry state across loop
        # iterations (e.g. ALBERT's "cur", threaded through 12 logical layers that all share the SAME
        # physical weights). GraphBuilder's repeat_for has NO per-iteration symbol-table scoping (a
        # single flat symtab is just overwritten each iteration, confirmed by reading graph_builder.cpp
        # directly) -- so the producer BEFORE the loop and the producer at the END of the loop body must
        # emit the exact same literal name for the carried tensor, which auto-freshened hints (a
        # monotonic counter suffix) can never do. Found via a real crash ("references unresolved input
        # 'cur'") before adding this.
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


def to_f32(t):
    return t.detach().cpu().numpy().astype(np.float32)


def add_linear(tb, prefix, sd, name):
    w = tb.weight(f"{prefix}.weight", to_f32(sd[f"{name}.weight"]))
    b = tb.weight(f"{prefix}.bias", to_f32(sd[f"{name}.bias"]))
    return w, b


def apply_layer_norm(tb, x, prefix, sd, name, eps, out_hint, out_name=None):
    """Plain LAYER_NORM (ggml_norm, reduces over ne[0]) + the checkpoint's own learned gamma/beta --
    this project's usual LayerNorm pattern (LAYER_NORM -> MUL -> ADD), used here on a channel-first
    ([hidden,T]) tensor, i.e. genuinely normalizing over the CHANNEL axis (standard LayerNorm), not the
    AdaIN/InstanceNorm reuse documented elsewhere for Kokoro's decoder (that one needs a [T,C] tensor
    instead -- the SAME primitive, different axis convention depending on what's fed in). `out_name`,
    when given, is passed through verbatim (not auto-freshened) -- needed when this call's result must
    carry a literal name across a repeat_for loop boundary (see TopologyBuilder.node's own comment)."""
    normed = tb.node("LAYER_NORM", [x], {"eps": eps}, f"{out_hint}_normed")
    g = tb.weight(f"{prefix}.gamma", to_f32(sd[f"{name}.weight"]))
    b = tb.weight(f"{prefix}.beta", to_f32(sd[f"{name}.bias"]))
    xm = tb.node("MUL", [normed, g], None, f"{out_hint}_mul")
    return tb.node("ADD", [xm, b], None, out_hint, name=out_name)


def gelu_new(tb, x, out_hint):
    """HF's "gelu_new": 0.5*x*(1+tanh(sqrt(2/pi)*x*(1+0.044715*x^2))) -- algebraically identical to
    ggml_gelu_f32's own C formula (confirmed by hand-checking both on a small vector), composed directly
    from exact-F32 primitives rather than reusing ggml's own tanh-GELU (which routes through an F16
    lookup table on CPU, confirmed via ggml-cpu/vec.h) or this project's erf-based GELU primitive (a
    different function entirely)."""
    sqrt_2_over_pi = float(np.sqrt(2.0 / np.pi))
    x_sq = tb.node("SQR", [x], None, f"{out_hint}_sq")
    x_cubed_term = tb.node("SCALE", [x_sq], {"s": 0.044715}, f"{out_hint}_cubeterm")
    inner = tb.node("ADD", [x_cubed_term, tb.weight("gelu_new.one", np.array([1.0], dtype=np.float32))],
                     None, f"{out_hint}_inner_add")
    inner = tb.node("MUL", [inner, x], None, f"{out_hint}_inner_mul")
    inner = tb.node("SCALE", [inner], {"s": sqrt_2_over_pi}, f"{out_hint}_inner_scaled")
    t = tb.node("TANH", [inner], None, f"{out_hint}_tanh")
    t = tb.node("ADD", [t, tb.weight("gelu_new.one", np.array([1.0], dtype=np.float32))], None, f"{out_hint}_tanh_p1")
    out = tb.node("MUL", [t, x], None, f"{out_hint}_mul2")
    return tb.node("SCALE", [out], {"s": 0.5}, out_hint)


def build_albert(tb, sd, hp):
    h = hp["hidden_size"]
    n_head = hp["n_head"]
    head_dim = hp["head_dim"]
    eps = hp["ln_eps"]

    word_emb_w = tb.weight("bert.word_embeddings.weight", to_f32(sd["module.embeddings.word_embeddings.weight"]))
    pos_emb_w = tb.weight("bert.position_embeddings.weight", to_f32(sd["module.embeddings.position_embeddings.weight"]))
    # token_type_ids is always the all-zeros default -- bake row 0 as a fixed constant bias, not a GET_ROWS.
    type_row0 = tb.weight("bert.token_type_row0",
                           to_f32(sd["module.embeddings.token_type_embeddings.weight"][0]))

    word_emb = tb.node("GET_ROWS", [word_emb_w, "tokens"], None, "word_emb")   # [128, T]
    pos_emb = tb.node("GET_ROWS", [pos_emb_w, "positions"], None, "pos_emb")   # [128, T]
    x = tb.node("ADD", [word_emb, pos_emb], None, "emb_sum1")
    x = tb.node("ADD", [x, type_row0], None, "emb_sum2")
    x = apply_layer_norm(tb, x, "bert.emb_ln", sd, "module.embeddings.LayerNorm", eps, "emb_ln_out")

    map_w, map_b = add_linear(tb, "bert.embedding_hidden_mapping_in", sd, "module.encoder.embedding_hidden_mapping_in")
    x = tb.node("ADD", [tb.node("MUL_MAT", [map_w, x], None, "map_mm"), map_b], None, "cur", name="cur")

    # --- 12x logical layers, ALL referencing the SAME physical weights (num_hidden_groups=1) --
    #     no {i} in any weight name below, so repeat_for's per-iteration substitution simply leaves
    #     these names unchanged across every one of the 12 iterations, exactly as intended.
    p = "module.encoder.albert_layer_groups.0.albert_layers.0"
    layer_nodes = []
    saved_nodes, tb.nodes = tb.nodes, layer_nodes

    qw, qb = add_linear(tb, "bert.layer.attention.query", sd, f"{p}.attention.query")
    kw, kb = add_linear(tb, "bert.layer.attention.key", sd, f"{p}.attention.key")
    vw, vb = add_linear(tb, "bert.layer.attention.value", sd, f"{p}.attention.value")
    ow, ob = add_linear(tb, "bert.layer.attention.dense", sd, f"{p}.attention.dense")

    q = tb.node("ADD", [tb.node("MUL_MAT", [qw, "cur"], None, "q"), qb], None, "q_b")
    k = tb.node("ADD", [tb.node("MUL_MAT", [kw, "cur"], None, "k"), kb], None, "k_b")
    v = tb.node("ADD", [tb.node("MUL_MAT", [vw, "cur"], None, "v"), vb], None, "v_b")
    q = tb.node("RESHAPE", [q], {"shape": [head_dim, n_head, "$n_tokens"]}, "q_r")
    k = tb.node("RESHAPE", [k], {"shape": [head_dim, n_head, "$n_tokens"]}, "k_r")
    v = tb.node("RESHAPE", [v], {"shape": [head_dim, n_head, "$n_tokens"]}, "v_r")
    attn = tb.node("ATTENTION", [q, k, v, "attn_mask"],
                   {"kv_cache": False, "scale": 1.0 / float(np.sqrt(head_dim))}, "attn_out")
    attn_proj = tb.node("ADD", [tb.node("MUL_MAT", [ow, attn], None, "attn_proj_mm"), ob], None, "attn_proj")
    x_after_attn = tb.node("ADD", ["cur", attn_proj], None, "res_attn")
    x_after_attn = apply_layer_norm(tb, x_after_attn, "bert.layer.attention.ln", sd, f"{p}.attention.LayerNorm",
                                     eps, "attn_ln_out")

    fw, fb = add_linear(tb, "bert.layer.ffn", sd, f"{p}.ffn")
    fow, fob = add_linear(tb, "bert.layer.ffn_output", sd, f"{p}.ffn_output")
    ffn_h = tb.node("ADD", [tb.node("MUL_MAT", [fw, x_after_attn], None, "ffn_h_mm"), fb], None, "ffn_h")
    ffn_h = gelu_new(tb, ffn_h, "ffn_gelu")
    ffn_out = tb.node("ADD", [tb.node("MUL_MAT", [fow, ffn_h], None, "ffn_out_mm"), fob], None, "ffn_out")
    x_final = tb.node("ADD", [x_after_attn, ffn_out], None, "res_ffn")
    x_final = apply_layer_norm(tb, x_final, "bert.layer.full_ln", sd, f"{p}.full_layer_layer_norm", eps, "cur",
                                out_name="cur")

    tb.nodes = saved_nodes
    tb.nodes.append({"repeat_for": str(hp["n_layer"]), "index_var": "i", "nodes": layer_nodes})

    return "cur"


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <kokoro-v1_0.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd_all["bert"]

    tb = TopologyBuilder()
    out = build_albert(tb, sd, HP)

    inputs = [
        {"name": "tokens", "dtype": "i32", "shape": ["$n_tokens"]},
        {"name": "positions", "dtype": "i32", "shape": ["$n_tokens"]},
        {"name": "attn_mask", "dtype": "f32", "shape": ["$n_tokens", "$n_tokens"]},
    ]
    topo = tb.topology(inputs, out)

    writer = GGUFWriter(str(out_dir / "kokoro_albert.gguf"), "loom-kokoro-albert")
    writer.add_string("model.graph_topology", json.dumps(topo))
    for name, arr in tb.weights.items():
        writer.add_tensor(name, arr.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {out_dir / 'kokoro_albert.gguf'}, {len(tb.weights)} weights")


if __name__ == "__main__":
    main()
