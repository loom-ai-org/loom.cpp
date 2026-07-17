"""Plain torch.nn.Module reimplementation of real Qwen3-0.6B-Base (see tools/convert_qwen3/convert_qwen3.py
for the hand-written topology this is checked against), built to test how much of the toy LLM's generic
op-mapping table (tools/convert_generic/aten_to_loom.py) survives unchanged against a real checkpoint with
genuine GQA (16 query / 8 KV heads), per-head QK-norm, and tied embeddings.

Deliberately reuses toy_llm_module.py's loom::rope_neox / loom::attention custom ops verbatim (same
`torch.ops.loom.*` registration, imported not redefined) -- the whole point of this exercise is to see
whether that reuse actually holds, not to write a second copy.

Weights load directly from the real checkpoint's model.safetensors (BF16 -> F32, no `transformers`
dependency, same precedent as tools/convert_qwen3/convert_qwen3.py).
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_llm_module as toy  # registers torch.ops.loom.rope_neox / torch.ops.loom.attention as a side effect


def hparams(config: dict) -> dict:
    return {
        "n_vocab": config["vocab_size"], "n_embd": config["hidden_size"],
        "n_layer": config["num_hidden_layers"], "n_head": config["num_attention_heads"],
        "n_head_kv": config["num_key_value_heads"], "n_embd_head_k": config["head_dim"],
        "n_ff": config["intermediate_size"], "rope_dims": config["head_dim"],
        "rope_freq_base": float(config["rope_theta"]), "rope_freq_scale": 1.0,
        "rms_norm_eps": float(config["rms_norm_eps"]),
    }


class Qwen3Layer(torch.nn.Module):
    def __init__(self, hp: dict, state: dict, i: int):
        super().__init__()
        n_embd, n_head, n_head_kv, head_dim, n_ff = (
            hp["n_embd"], hp["n_head"], hp["n_head_kv"], hp["n_embd_head_k"], hp["n_ff"],
        )
        self.n_head, self.n_head_kv, self.head_dim = n_head, n_head_kv, head_dim
        self.rope_dims = hp["rope_dims"]
        self.rope_freq_base = hp["rope_freq_base"]
        self.rope_freq_scale = hp["rope_freq_scale"]
        self.eps = hp["rms_norm_eps"]
        self.scale = 1.0 / (head_dim ** 0.5)

        def f32(key):
            return state[key].to(torch.float32).clone()

        p = f"model.layers.{i}"
        self.attn_norm = torch.nn.Parameter(f32(f"{p}.input_layernorm.weight"))
        self.attn_q = torch.nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.attn_k = torch.nn.Linear(n_embd, n_head_kv * head_dim, bias=False)
        self.attn_v = torch.nn.Linear(n_embd, n_head_kv * head_dim, bias=False)
        self.attn_output = torch.nn.Linear(n_head * head_dim, n_embd, bias=False)
        self.attn_q.weight.data = f32(f"{p}.self_attn.q_proj.weight")
        self.attn_k.weight.data = f32(f"{p}.self_attn.k_proj.weight")
        self.attn_v.weight.data = f32(f"{p}.self_attn.v_proj.weight")
        self.attn_output.weight.data = f32(f"{p}.self_attn.o_proj.weight")
        self.attn_q_norm = torch.nn.Parameter(f32(f"{p}.self_attn.q_norm.weight"))
        self.attn_k_norm = torch.nn.Parameter(f32(f"{p}.self_attn.k_norm.weight"))

        self.ffn_norm = torch.nn.Parameter(f32(f"{p}.post_attention_layernorm.weight"))
        self.ffn_gate = torch.nn.Linear(n_embd, n_ff, bias=False)
        self.ffn_up = torch.nn.Linear(n_embd, n_ff, bias=False)
        self.ffn_down = torch.nn.Linear(n_ff, n_embd, bias=False)
        self.ffn_gate.weight.data = f32(f"{p}.mlp.gate_proj.weight")
        self.ffn_up.weight.data = f32(f"{p}.mlp.up_proj.weight")
        self.ffn_down.weight.data = f32(f"{p}.mlp.down_proj.weight")

    def forward(self, cur: Tensor, positions: Tensor, kq_mask: Tensor) -> Tensor:
        n_tokens = cur.shape[0]

        attn_normed = F.rms_norm(cur, (cur.shape[-1],), weight=None, eps=self.eps) * self.attn_norm
        q = self.attn_q(attn_normed).view(n_tokens, self.n_head, self.head_dim)
        k = self.attn_k(attn_normed).view(n_tokens, self.n_head_kv, self.head_dim)
        v = self.attn_v(attn_normed).view(n_tokens, self.n_head_kv, self.head_dim)

        # Per-head QK-norm: RMS_NORM(weight=None) + separate mul, exactly the same op pair the toy LLM
        # already used for attn_norm/ffn_norm -- applied here to q/k instead of cur, before RoPE. Needs
        # zero new op-mapping table entries.
        q = F.rms_norm(q, (self.head_dim,), weight=None, eps=self.eps) * self.attn_q_norm
        k = F.rms_norm(k, (self.head_dim,), weight=None, eps=self.eps) * self.attn_k_norm

        q = torch.ops.loom.rope_neox(q, positions, self.rope_dims, self.rope_freq_base, self.rope_freq_scale)
        k = torch.ops.loom.rope_neox(k, positions, self.rope_dims, self.rope_freq_base, self.rope_freq_scale)

        # q has n_head heads, k/v have n_head_kv < n_head -- GQA. loom::attention's own custom-op contract
        # never assumed n_head == n_head_kv (the toy LLM just never exercised the difference); the real
        # broadcast happens inside loom's C++ ATTENTION primitive (ggml_mul_mat's own broadcast rule).
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


class Qwen3LLM(torch.nn.Module):
    def __init__(self, hf_dir: Path):
        super().__init__()
        config = json.loads((hf_dir / "config.json").read_text())
        hp = hparams(config)
        self.eps = hp["rms_norm_eps"]

        state = load_file(str(hf_dir / "model.safetensors"))
        self.token_embd = torch.nn.Parameter(state["model.embed_tokens.weight"].to(torch.float32).clone())
        self.layers = torch.nn.ModuleList([Qwen3Layer(hp, state, i) for i in range(hp["n_layer"])])
        self.output_norm = torch.nn.Parameter(state["model.norm.weight"].to(torch.float32).clone())
        # tie_word_embeddings=true: no separate lm_head weight in the checkpoint at all -- the final
        # logits projection reuses self.token_embd directly (same nn.Parameter, not a copy), same as
        # convert_qwen3.py's hand-written topology referencing "token_embd.weight" twice by name.
        assert config["tie_word_embeddings"]

    def forward(self, tokens: Tensor, positions: Tensor, kq_mask: Tensor) -> Tensor:
        cur = F.embedding(tokens, self.token_embd)
        for layer in self.layers:
            cur = layer(cur, positions, kq_mask)
        cur = F.rms_norm(cur, (cur.shape[-1],), weight=None, eps=self.eps) * self.output_norm
        logits = F.linear(cur, self.token_embd)
        return logits
