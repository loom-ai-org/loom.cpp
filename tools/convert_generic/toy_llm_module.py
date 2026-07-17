"""Plain torch.nn.Module reimplementation of the Milestone-1 toy LLM (see
tools/fixture_gen/toy_llm_common.py / reference_forward.py), written to be export-friendly: every op
either maps 1:1 onto an existing loom primitive via a plain ATen call (embedding, rms_norm, linear, view,
silu, mul, add), or -- where loom has a fused primitive with no ATen equivalent at all -- calls a small
custom op registered via torch.library.custom_op, so torch.export() keeps it as a single opaque graph
node instead of decomposing it into element-wise pieces a generic op-mapping table would have to
pattern-match back together.

Two such custom ops:
  - loom::rope_neox -- ggml's exact NEOX-paired rotation (no ATen equivalent).
  - loom::attention  -- loom's ATTENTION primitive bakes in KV-cache side effects (append this step's K/V
    to a persistent cache, read back the full valid prefix) that have no ATen equivalent regardless of
    calling convention -- even a real aten.scaled_dot_product_attention-shaped subgraph would still need
    this runtime-only cache/layer information injected by the converter, since it isn't derivable from the
    graph. Using a custom op here removes an orthogonal wrinkle (SDPA's (batch,heads,seq,dim) calling
    convention doesn't match loom's native head-minor layout, which would need its own transpose-wrapper
    handling) without weakening what this POC is actually testing.

Weights are loaded directly from tools/fixture_gen/toy_llm_common.generate_weights() (same seed), so
numerical parity with the existing hand-written-topology fixture/tests is guaranteed by construction, not
re-derived.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fixture_gen"))
import toy_llm_common as common


@torch.library.custom_op("loom::rope_neox", mutates_args=())
def rope_neox(x: Tensor, positions: Tensor, n_dims: int, freq_base: float, freq_scale: float) -> Tensor:
    """x: [n_tokens, n_head, head_dim] (PyTorch-natural layout, matching ggml's reversed
    [head_dim, n_head, n_tokens]). Bit-for-bit the same math as reference_forward.py::rope_neox, just
    vectorized over the head_dim/2 pairs instead of a Python loop."""
    half = n_dims // 2
    pos = positions.to(torch.float64)
    i = torch.arange(half, dtype=torch.float64)
    theta = pos[:, None] * freq_scale * (freq_base ** (-2.0 * i / n_dims))[None, :]  # [n_tokens, half]
    cos_t = theta.cos().to(torch.float32)[:, None, :]  # [n_tokens, 1, half]
    sin_t = theta.sin().to(torch.float32)[:, None, :]
    x0 = x[:, :, :half]
    x1 = x[:, :, half:n_dims]
    out0 = x0 * cos_t - x1 * sin_t
    out1 = x0 * sin_t + x1 * cos_t
    return torch.cat([out0, out1], dim=-1)


@rope_neox.register_fake
def _(x, positions, n_dims, freq_base, freq_scale):
    return torch.empty_like(x)


@torch.library.custom_op("loom::attention", mutates_args=())
def attention(q: Tensor, k: Tensor, v: Tensor, kq_mask: Tensor, scale: float) -> Tensor:
    """q/k/v: [n_tokens, n_head(_kv), head_dim] (loom's native ATTENTION layout) -- q's head count may
    exceed k/v's (GQA), same broadcast contract as loom's own C++ ATTENTION primitive
    (ggml_mul_mat's own broadcast rule, requires n_head % n_head_kv == 0). Reference-only body (the real
    numbers this POC is checked against come from reference_forward*.py / loom-engine's C++ ATTENTION
    primitive, not from running this function) -- kept numerically faithful anyway for self-consistency."""
    qt = q.transpose(0, 1)  # [n_head, n_tokens, head_dim]
    kt = k.transpose(0, 1)  # [n_head_kv, n_tokens, head_dim]
    vt = v.transpose(0, 1)
    out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=kq_mask, scale=scale,
                                          enable_gqa=(qt.shape[0] != kt.shape[0]))
    out = out.transpose(0, 1).reshape(q.shape[0], -1)  # [n_tokens, n_head*head_dim]
    return out


@attention.register_fake
def _(q, k, v, kq_mask, scale):
    return q.new_empty(q.shape[0], q.shape[1] * q.shape[2])


class Layer(torch.nn.Module):
    def __init__(self, hp: dict, w: dict, i: int):
        super().__init__()
        n_embd, n_head, n_head_kv, head_dim, n_ff = (
            hp["n_embd"], hp["n_head"], hp["n_head_kv"], hp["n_embd_head_k"], hp["n_ff"],
        )
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.rope_dims = hp["rope_dims"]
        self.rope_freq_base = hp["rope_freq_base"]
        self.rope_freq_scale = hp["rope_freq_scale"]
        self.eps = hp["rms_norm_eps"]
        self.scale = 1.0 / (head_dim ** 0.5)

        self.attn_norm = torch.nn.Parameter(torch.from_numpy(w[f"blk.{i}.attn_norm.weight"]))
        self.attn_q = torch.nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.attn_k = torch.nn.Linear(n_embd, n_head_kv * head_dim, bias=False)
        self.attn_v = torch.nn.Linear(n_embd, n_head_kv * head_dim, bias=False)
        self.attn_output = torch.nn.Linear(n_head * head_dim, n_embd, bias=False)
        self.attn_q.weight.data = torch.from_numpy(w[f"blk.{i}.attn_q.weight"]).clone()
        self.attn_k.weight.data = torch.from_numpy(w[f"blk.{i}.attn_k.weight"]).clone()
        self.attn_v.weight.data = torch.from_numpy(w[f"blk.{i}.attn_v.weight"]).clone()
        self.attn_output.weight.data = torch.from_numpy(w[f"blk.{i}.attn_output.weight"]).clone()

        self.ffn_norm = torch.nn.Parameter(torch.from_numpy(w[f"blk.{i}.ffn_norm.weight"]))
        self.ffn_gate = torch.nn.Linear(n_embd, n_ff, bias=False)
        self.ffn_up = torch.nn.Linear(n_embd, n_ff, bias=False)
        self.ffn_down = torch.nn.Linear(n_ff, n_embd, bias=False)
        self.ffn_gate.weight.data = torch.from_numpy(w[f"blk.{i}.ffn_gate.weight"]).clone()
        self.ffn_up.weight.data = torch.from_numpy(w[f"blk.{i}.ffn_up.weight"]).clone()
        self.ffn_down.weight.data = torch.from_numpy(w[f"blk.{i}.ffn_down.weight"]).clone()

    def forward(self, cur: Tensor, positions: Tensor, kq_mask: Tensor) -> Tensor:
        n_tokens = cur.shape[0]

        attn_normed = F.rms_norm(cur, (cur.shape[-1],), weight=None, eps=self.eps) * self.attn_norm
        q = self.attn_q(attn_normed).view(n_tokens, self.n_head, self.head_dim)
        k = self.attn_k(attn_normed).view(n_tokens, self.n_head_kv, self.head_dim)
        v = self.attn_v(attn_normed).view(n_tokens, self.n_head_kv, self.head_dim)

        q = torch.ops.loom.rope_neox(q, positions, self.rope_dims, self.rope_freq_base, self.rope_freq_scale)
        k = torch.ops.loom.rope_neox(k, positions, self.rope_dims, self.rope_freq_base, self.rope_freq_scale)

        attn_out = torch.ops.loom.attention(q, k, v, kq_mask, self.scale)
        attn_proj = self.attn_output(attn_out)
        cur = cur + attn_proj

        ffn_normed = F.rms_norm(cur, (cur.shape[-1],), weight=None, eps=self.eps) * self.ffn_norm
        gate = self.ffn_gate(ffn_normed)
        up = self.ffn_up(ffn_normed)
        act = F.silu(gate) * up
        ffn_out = self.ffn_down(act)
        cur = cur + ffn_out
        return cur


class ToyLLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hp = common.hparams()
        w = common.generate_weights()
        self.eps = hp["rms_norm_eps"]

        self.token_embd = torch.nn.Parameter(torch.from_numpy(w["token_embd.weight"]))
        self.layers = torch.nn.ModuleList([Layer(hp, w, i) for i in range(hp["n_layer"])])
        self.output_norm = torch.nn.Parameter(torch.from_numpy(w["output_norm.weight"]))
        self.output = torch.nn.Linear(hp["n_embd"], hp["n_vocab"], bias=False)
        self.output.weight.data = torch.from_numpy(w["output.weight"]).clone()

    def forward(self, tokens: Tensor, positions: Tensor, kq_mask: Tensor) -> Tensor:
        cur = F.embedding(tokens, self.token_embd)
        for layer in self.layers:
            cur = layer(cur, positions, kq_mask)
        cur = F.rms_norm(cur, (cur.shape[-1],), weight=None, eps=self.eps) * self.output_norm
        logits = self.output(cur)
        return logits
