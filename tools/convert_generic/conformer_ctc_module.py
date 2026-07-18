"""Plain torch.nn.Module reimplementation of the real Conformer-CTC encoder (see
tools/convert_nemo/convert_conformer_ctc.py's build_topology(), which this mirrors node-for-node), written
to be export-friendly for the generic ATen-graph -> loom-topology converter (tools/convert_generic/).

Scope, deliberately narrower than the full model (see BACKLOG.md's gating-criterion note): the mel
frontend (STFT-via-conv, CMVN) exercises no new ops relative to the toy-LLM/Qwen3 generic-converter POCs
already done -- this module's declared input is "mel_input" (the same intermediate tensor name/shape
convert_conformer_ctc.py already produces), not raw waveform. The actually-new ops worth testing are the
Conformer encoder's: subsampling Conv2d, LayerNorm, depthwise Conv1d, GLU, and relative-position
self-attention (no ATen equivalent -- see conformer_ops.py's loom::rel_pos_attention custom op).

No batch dimension is carried through the transformer/attention portion (same "batch=1 always, don't track
it" convention as toy_llm_module.py/qwen3_module.py) -- cur is [T, n_embd] throughout. A batch dim of 1 is
added locally, only where Conv1d/Conv2d's own API requires it, and dropped again immediately after.

Ground-truth for the subsampling module's exact tensor convention was confirmed against NeMo's real source
(not assumed from convert_conformer_ctc.py's comments-about-NeMo alone): `ConvSubsampling.forward`
(nemo/collections/asr/parts/submodules/subsampling.py:385-431) receives x already unsqueezed to
[B,1,T,F] (confirmed via subsampling.py:728's own `x = x.unsqueeze(1) # (batch, 1, time, features)` and
conformer_encoder.py:628's `torch.transpose(audio_signal, 1, 2)` immediately before calling pre_encode,
which is what turns the frontend's natural [B,F,T] into [B,T,F] first) -- i.e. H=time, W=freq in NCHW --
and flattens post-conv via `x.transpose(1,2).reshape(b,t,-1)` on a [B,C,T',F'] tensor, i.e.
channel-slower/freq-faster, exactly matching convert_conformer_ctc.py's own comment.

Weights load via tools/convert_nemo/nemo_common.py's load_nemo()/hparams()/fold_batchnorm() -- the exact
same functions (and thus the exact same folded weights) convert_conformer_ctc.py and
reference_forward_conformer.py already use, so numerical parity with the existing hand-written-topology
fixture is guaranteed by construction, not re-derived.

Requires: pip install torch numpy pyyaml
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "convert_nemo"))
import conformer_ops  # noqa: F401 -- registers loom::rel_pos_attention as a side effect
import nemo_common as common


class ConformerLayer(torch.nn.Module):
    def __init__(self, hp: dict, w: dict, i: int):
        super().__init__()
        n_embd, n_head, head_dim, ff_hidden = hp["n_embd"], hp["n_head"], hp["head_dim"], hp["ff_hidden"]
        conv_k, conv_pad = hp["conv_kernel_size"], hp["conv_padding"]
        self.n_head, self.head_dim, self.eps, self.scale = n_head, head_dim, hp["ln_eps"], 1.0 / (head_dim ** 0.5)

        p = f"encoder.layers.{i}"

        def lin(out_f, in_f, key):
            layer = torch.nn.Linear(in_f, out_f, bias=True)
            layer.weight.data = w[f"{key}.weight"].clone()
            layer.bias.data = w[f"{key}.bias"].clone()
            return layer

        def ln_weight(key):
            return torch.nn.Parameter(w[f"{key}.weight"].clone()), torch.nn.Parameter(w[f"{key}.bias"].clone())

        self.ff1_norm_w, self.ff1_norm_b = ln_weight(f"{p}.norm_feed_forward1")
        self.ff1_linear1 = lin(ff_hidden, n_embd, f"{p}.feed_forward1.linear1")
        self.ff1_linear2 = lin(n_embd, ff_hidden, f"{p}.feed_forward1.linear2")
        self.ff1_linear2.weight.data *= 0.5  # half-step residual, folded (see build_topology's half_step_ff)
        self.ff1_linear2.bias.data *= 0.5

        self.sa_norm_w, self.sa_norm_b = ln_weight(f"{p}.norm_self_att")
        self.q_proj = lin(n_head * head_dim, n_embd, f"{p}.self_attn.linear_q")
        self.k_proj = lin(n_head * head_dim, n_embd, f"{p}.self_attn.linear_k")
        self.v_proj = lin(n_head * head_dim, n_embd, f"{p}.self_attn.linear_v")
        self.out_proj = lin(n_embd, n_head * head_dim, f"{p}.self_attn.linear_out")
        self.pos_proj = torch.nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.pos_proj.weight.data = w[f"{p}.self_attn.linear_pos.weight"].clone()
        self.pos_bias_u = torch.nn.Parameter(w[f"{p}.self_attn.pos_bias_u"].clone())
        self.pos_bias_v = torch.nn.Parameter(w[f"{p}.self_attn.pos_bias_v"].clone())

        self.conv_norm_w, self.conv_norm_b = ln_weight(f"{p}.norm_conv")
        self.pw1 = torch.nn.Conv1d(n_embd, 2 * n_embd, kernel_size=1, stride=1, padding=0)
        self.pw1.weight.data = w[f"{p}.conv.pointwise_conv1.weight"].clone()
        self.pw1.bias.data = w[f"{p}.conv.pointwise_conv1.bias"].clone()
        self.dw = torch.nn.Conv1d(n_embd, n_embd, kernel_size=conv_k, stride=1, padding=conv_pad, groups=n_embd)
        self.dw.weight.data = w[f"{p}.conv.depthwise_conv.weight"].clone()
        self.dw.bias.data = w[f"{p}.conv.depthwise_conv.bias"].clone()
        bn_scale, bn_shift = common.fold_batchnorm({k: v for k, v in w.items()}, f"{p}.conv.batch_norm", hp["bn_eps"])
        self.bn_scale, self.bn_shift = torch.nn.Parameter(bn_scale.clone()), torch.nn.Parameter(bn_shift.clone())
        self.pw2 = torch.nn.Conv1d(n_embd, n_embd, kernel_size=1, stride=1, padding=0)
        self.pw2.weight.data = w[f"{p}.conv.pointwise_conv2.weight"].clone()
        self.pw2.bias.data = w[f"{p}.conv.pointwise_conv2.bias"].clone()

        self.ff2_norm_w, self.ff2_norm_b = ln_weight(f"{p}.norm_feed_forward2")
        self.ff2_linear1 = lin(ff_hidden, n_embd, f"{p}.feed_forward2.linear1")
        self.ff2_linear2 = lin(n_embd, ff_hidden, f"{p}.feed_forward2.linear2")
        self.ff2_linear2.weight.data *= 0.5
        self.ff2_linear2.bias.data *= 0.5

        self.out_norm_w, self.out_norm_b = ln_weight(f"{p}.norm_out")

    def _layer_norm(self, x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        normed = F.layer_norm(x, (x.shape[-1],), weight=None, bias=None, eps=self.eps)
        return normed * weight + bias

    def forward(self, cur: Tensor, p_emb: Tensor, kq_mask: Tensor) -> Tensor:
        n_tokens = cur.shape[0]

        ff1_in = self._layer_norm(cur, self.ff1_norm_w, self.ff1_norm_b)
        ff1 = self.ff1_linear2(F.silu(self.ff1_linear1(ff1_in)))
        cur = cur + ff1

        sa_normed = self._layer_norm(cur, self.sa_norm_w, self.sa_norm_b)
        q = self.q_proj(sa_normed).reshape(n_tokens, self.n_head, self.head_dim)
        k = self.k_proj(sa_normed).reshape(n_tokens, self.n_head, self.head_dim)
        v = self.v_proj(sa_normed).reshape(n_tokens, self.n_head, self.head_dim)
        p = self.pos_proj(p_emb).reshape(p_emb.shape[0], self.n_head, self.head_dim)
        attn_ctx = torch.ops.loom.rel_pos_attention(q, k, v, p, self.pos_bias_u, self.pos_bias_v, kq_mask, self.scale)
        cur = cur + self.out_proj(attn_ctx)

        conv_in = self._layer_norm(cur, self.conv_norm_w, self.conv_norm_b)
        conv_in = conv_in.permute(1, 0).unsqueeze(0)  # [T,C] -> [1,C,T] for Conv1d
        glu_out = F.glu(self.pw1(conv_in), dim=1)
        dw_out = self.dw(glu_out)
        bn_out = dw_out * self.bn_scale.reshape(1, -1, 1) + self.bn_shift.reshape(1, -1, 1)
        conv_result = self.pw2(F.silu(bn_out))
        conv_result = conv_result.squeeze(0).permute(1, 0)  # [1,C,T] -> [T,C]
        cur = cur + conv_result

        ff2_in = self._layer_norm(cur, self.ff2_norm_w, self.ff2_norm_b)
        ff2 = self.ff2_linear2(F.silu(self.ff2_linear1(ff2_in)))
        cur = cur + ff2

        return self._layer_norm(cur, self.out_norm_w, self.out_norm_b)


class ConformerCTC(torch.nn.Module):
    def __init__(self, nemo_path: str):
        super().__init__()
        config, state, _ = common.load_nemo(nemo_path)
        hp = common.hparams(config)
        self.hp = hp
        n_embd = hp["n_embd"]
        xscale = float(np.sqrt(n_embd))

        self.conv0 = torch.nn.Conv2d(1, hp["subsampling_conv_channels"], kernel_size=3, stride=2, padding=1)
        self.conv0.weight.data = state["encoder.pre_encode.conv.0.weight"].clone()
        self.conv0.bias.data = state["encoder.pre_encode.conv.0.bias"].clone()
        self.conv1 = torch.nn.Conv2d(hp["subsampling_conv_channels"], hp["subsampling_conv_channels"],
                                      kernel_size=3, stride=2, padding=1)
        self.conv1.weight.data = state["encoder.pre_encode.conv.2.weight"].clone()
        self.conv1.bias.data = state["encoder.pre_encode.conv.2.bias"].clone()

        flat_dim = state["encoder.pre_encode.out.weight"].shape[1]
        self.sub_out = torch.nn.Linear(flat_dim, n_embd, bias=True)
        self.sub_out.weight.data = state["encoder.pre_encode.out.weight"].clone() * xscale
        self.sub_out.bias.data = state["encoder.pre_encode.out.bias"].clone() * xscale

        self.layers = torch.nn.ModuleList([ConformerLayer(hp, state, i) for i in range(hp["n_layers"])])

        self.decoder = torch.nn.Linear(n_embd, hp["num_classes"], bias=True)
        self.decoder.weight.data = state["decoder.decoder_layers.0.weight"].squeeze(-1).clone()
        self.decoder.bias.data = state["decoder.decoder_layers.0.bias"].clone()

    def forward(self, mel_input: Tensor, pos_emb_raw: Tensor, kq_mask: Tensor) -> Tensor:
        # mel_input: [1, 1, T_mel, n_mels] (NCHW; ground-truthed against real NeMo source, see module
        # docstring). pos_emb_raw: [n_pos, n_embd]. kq_mask: [n_subsampled(kv), n_subsampled(q)].
        x = F.relu(self.conv0(mel_input))
        x = F.relu(self.conv1(x))
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)  # channel-slower/freq-faster flatten (ground-truthed)
        cur = self.sub_out(x).reshape(t, -1)  # drop the batch=1 dim -- [T', n_embd], no-batch convention

        for layer in self.layers:
            cur = layer(cur, pos_emb_raw, kq_mask)

        return self.decoder(cur)
