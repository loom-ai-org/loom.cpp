"""Custom op for Conformer-CTC's relative-position self-attention (REL_POS_ATTENTION), registered so
torch.export() keeps it as one opaque graph node -- same rationale/precedent as toy_llm_module.py's
loom::rope_neox / loom::attention: loom's REL_POS_ATTENTION primitive (src/ops/primitives_attention.cpp)
has no ATen equivalent at all (the Transformer-XL-style relative shift trick + dual pos_bias_u/v terms
aren't a single ATen op, or even a common composition of ATen ops a generic mapping table could
pattern-match). The eager body below mirrors the C++ primitive's real math (rel_shift included) for
self-consistency, matching every other custom op in this project -- but only register_fake's shape
contract is actually exercised by torch.export() itself.
"""
import torch
import torch.nn.functional as F
from torch import Tensor


def _rel_shift(x: Tensor) -> Tensor:
    """x: [n_head, qlen, pos_len]. Mirrors rel_shift() in src/ops/primitives_attention.cpp:109-120 exactly:
    pad one zero column at the front of the last axis, reinterpret the last two axes swapped, drop the
    first "row" of that reinterpretation, reinterpret back."""
    n_head, qlen, pos_len = x.shape
    padded = F.pad(x, (1, 0))  # [n_head, qlen, pos_len+1]
    reshaped = padded.reshape(n_head, pos_len + 1, qlen)
    sliced = reshaped[:, 1:, :]  # [n_head, pos_len, qlen]
    return sliced.reshape(n_head, qlen, pos_len)


@torch.library.custom_op("loom::rel_pos_attention", mutates_args=())
def rel_pos_attention(q: Tensor, k: Tensor, v: Tensor, p: Tensor, pos_bias_u: Tensor, pos_bias_v: Tensor,
                       kq_mask: Tensor, scale: float) -> Tensor:
    """q/k/v: [n_tokens, n_head, head_dim]; p: [n_pos, n_head, head_dim] (shared sinusoidal positional
    embedding, already linear_pos-projected upstream); pos_bias_u/v: [n_head, head_dim]; kq_mask:
    [n_tokens(kv), n_tokens(q)]. Output: [n_tokens, n_head*head_dim] (pre-linear_out context), same
    convention as loom::attention's output."""
    qu = (q + pos_bias_u).permute(1, 0, 2)  # [n_head, n_tokens, head_dim]
    qv = (q + pos_bias_v).permute(1, 0, 2)
    kp = k.permute(1, 0, 2)                 # [n_head, n_tokens, head_dim]
    pp = p.permute(1, 0, 2)                 # [n_head, n_pos, head_dim]

    matrix_ac = torch.matmul(qu, kp.transpose(-1, -2))          # [n_head, n_tokens(q), n_tokens(kv)]
    matrix_bd_raw = torch.matmul(qv, pp.transpose(-1, -2))      # [n_head, n_tokens(q), n_pos]
    matrix_bd_shifted = _rel_shift(matrix_bd_raw)
    matrix_bd = matrix_bd_shifted[:, :, :matrix_ac.shape[-1]]

    scores = (matrix_ac + matrix_bd) * scale
    scores = scores + kq_mask.transpose(0, 1).unsqueeze(0)  # kq_mask: [n_tokens(kv), n_tokens(q)] -> broadcast
    probs = torch.softmax(scores, dim=-1)

    vp = v.permute(1, 0, 2)  # [n_head, n_tokens(kv), head_dim]
    ctx = torch.matmul(probs, vp)  # [n_head, n_tokens(q), head_dim]
    return ctx.permute(1, 0, 2).reshape(q.shape[0], -1)  # [n_tokens, n_head*head_dim]


@rel_pos_attention.register_fake
def _(q, k, v, p, pos_bias_u, pos_bias_v, kq_mask, scale):
    return q.new_empty(q.shape[0], q.shape[1] * q.shape[2])
