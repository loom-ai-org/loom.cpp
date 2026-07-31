"""Exports a real piper/VITS checkpoint through `TTSVitsExportConfig` (BACKLOG.md P3.3, migrated from
`export_vits_mil.py`), tracing the REAL `piper_train.vits.models.SynthesizerTrn` submodules directly
(TextEncoder, StochasticDurationPredictor, ResidualCouplingBlock, HiFi-GAN Generator) -- not the
hand-built bespoke topology `tools/convert_piper_vits/convert_vits.py` constructs op-by-op via its own
custom ggml primitives (CONV_FLOW_REVERSE, RESIDUAL_COUPLING_LAYER_REVERSE, DDS_CONV,
ELEMENTWISE_AFFINE_REVERSE, REL_POS_ATTENTION_SHAW). See BACKLOG.md for the full trail of findings this
module's own workarounds encode.

Three phases, same split as the bespoke script (`GraphTopology` supports exactly one declared output per
topology, and the duration predictor's own output determines the total output frame count -- a genuinely
data-dependent value the host must compute BEFORE phase 2 can run):
  - stats:        TextEncoder -> [m_p; logs_p] concatenated, T-fast (ne=[T, 2*inter_channels]).
  - logw:         TextEncoder + StochasticDurationPredictor(reverse) -> duration logits, ne=[T].
  - flow_vocoder: ResidualCouplingBlock(reverse) + HiFi-GAN Generator -> waveform, ne=[n_samples].
                  Takes `z_p` [1, inter_channels, T'] as a declared input (T' = the host-computed frame
                  count from `logw`'s own output, via `generate_path`) -- same host-side glue
                  loom::VitsDriver already implements for the bespoke topology's own phase 2.

Two real, general (not VITS-specific) exporter bugs found and fixed getting this to trace/build/run at
all (see BACKLOG.md for the full writeups):
  - `_resolve_scalar_expr`'s cycle guard treated any DAG diamond (the same upstream scalar reached via
    two different arithmetic paths, e.g. `end = start + 2*length - 1` referencing `length` twice) as a
    false cycle, silently returning None and corrupting `_get_relative_embeddings`'s dynamic slice.
  - `_infer_dynamic_dim_expr` didn't walk through `leaky_relu` or `conv_transpose`, breaking HiFi-GAN's
    3-stage upsample chain's own dynamic-length tracking (stage 2 silently read only stage 1's FIRST
    1/8th).

One new custom op + MIL dialect translation: StochasticDurationPredictor's ConvFlow uses a boolean-mask-
indexed rational-quadratic spline transform that `torch.jit.trace` cannot correctly capture (genuinely
data-dependent output shape) -- bridged via `torch.ops.loom.spline_inverse`
(`vits_spline_op.py`, registered automatically by this package's own `__init__.py`) into MIL's
`loom_spline` op (`dialect.py`) into the already-independently-verified RQ_SPLINE_INVERSE ggml primitive
(`src/ops/primitives_spline.cpp`).

Numerically verified against `tools/convert_piper_vits/reference_forward_vits.py`'s real-checkpoint
reference at T=62 ("Hello world, this is a test."): stats max abs diff ~3.3e-6 (m_p) / ~9.5e-7 (logs_p),
logw max abs diff ~7.9e-6, flow_vocoder waveform max abs diff ~5.4e-8.

NOT YET wired into `loom::VitsDriver` (`src/core/vits_driver.cpp`) -- its host glue currently assumes the
bespoke topology's own tensor conventions (channel-fast `stats`, `logw`'s own emb_rel_k/v-based
TextEncoder re-derivation), which differ from this module's simpler ones in at least the `stats` layout
(T-fast here, channel-fast there). See BACKLOG.md for the itemized remaining integration work.

Usage:
  loom-export /path/to/piper.ckpt -o vits_mil.gguf --task tts-multi-phase --model vits
"""
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct

from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase

sys.path.insert(0, "/home/flavio/Dev/piper/src/python")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "convert_piper_vits"))

from piper_train.vits.models import SynthesizerTrn  # noqa: E402
from piper_train.vits import modules as vits_modules  # noqa: E402
from piper_train.vits.attentions import MultiHeadAttention  # noqa: E402
from vits_common import load_piper_checkpoint  # noqa: E402

HP = dict(
    n_vocab=256, spec_channels=513, segment_size=8192,
    inter_channels=192, hidden_channels=192, filter_channels=768,
    n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1, resblock="2",
    resblock_kernel_sizes=(3, 5, 7), resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
    upsample_rates=(8, 8, 4), upsample_initial_channel=256, upsample_kernel_sizes=(16, 16, 8),
    n_speakers=1, gin_channels=0, use_sdp=True,
)

# `spec_channels`/`segment_size` are training-only (PosteriorEncoder's input width / random-slice
# length) -- irrelevant to inference numerics, `enc_q` is never called in the reverse-mode path any of
# these three wrappers trace. `spec_channels=513` is real, though (n_fft=1024 -> n_fft/2+1): the
# checkpoint's own `enc_q.pre.weight` tensor shape pins it, needed for `load_state_dict(strict=True)`.


# ---- Trace-friendly MultiHeadAttention patches ----
# coremltools' torch frontend has a hard *runtime* limitation (not just a translator gap -- see its own
# source comment in ops.py's `pad` converter): dynamic pad AMOUNTS only work for rank-1 tensors. Every
# dynamic pad site in MultiHeadAttention's relative-position machinery operates on rank>=3 tensors, so
# each needs a trace-friendly replacement. Verified bit-identical to the real piper code across lengths
# both above and below window_size+1 (both of the real code's own branches) via a standalone pure-eager
# equivalence check before ever tracing anything.
_REL_EMB_MAX_PAD = 2048  # generous static bound; must be >= (real per-call T) - window_size - 1


def _get_relative_embeddings_traceable(self, relative_embeddings, length):
    """Pads the FIXED-size learned table by a generous STATIC amount unconditionally, then dynamically
    SLICES out the real window (dynamic slicing is well-supported; only dynamic PAD AMOUNTS are the
    problem). The real code's window in the table's OWN (pre-pad) coordinates is always
    `[window_size+1-length, window_size+length)` regardless of whether length is above or below
    window_size+1 (both of the real code's branches reduce to this same formula) -- so padding to any
    sufficiently large fixed bound first and slicing with an offset is exact, not an approximation, for
    every length up to that bound.
    """
    window_size = self.window_size
    pad = _REL_EMB_MAX_PAD
    padded = F.pad(relative_embeddings, (0, 0, pad, pad, 0, 0))  # static amount -> any rank is fine
    start = pad + (window_size + 1) - length
    end = start + 2 * length - 1
    return padded[:, start:end]


def _dynamic_zero_pad_last(x, right):
    """Appends `right` (possibly dynamic) zeros to x's last dim via CONCAT, avoiding coremltools' own
    dynamic-pad rank restriction entirely. Builds the zero block from a same-dynamically-sized SLICE of
    x itself (`x[..., :right]`) multiplied by 0, rather than constructing a raw dynamic-shape
    torch.zeros(...) argument list (whose own frontend conversion has a separate, unrelated bug: `.narrow`
    with a dynamic length fails, `x[..., :right]` slicing doesn't).
    """
    if isinstance(right, int) and right == 0:
        return x
    zeros = x[..., :right] * 0.0
    return torch.cat([x, zeros], dim=-1)


def _dynamic_zero_pad_front_last(x, left):
    if isinstance(left, int) and left == 0:
        return x
    zeros = x[..., :left] * 0.0
    return torch.cat([zeros, x], dim=-1)


def _relative_position_to_absolute_position_traceable(self, x):
    batch, heads, length, _ = x.size()
    x = F.pad(x, (0, 1, 0, 0, 0, 0, 0, 0))  # amount=1 is a compile-time constant -- fine as-is
    x_flat = x.view([batch, heads, length * 2 * length])
    x_flat = _dynamic_zero_pad_last(x_flat, length - 1)
    x_final = x_flat.view([batch, heads, length + 1, (2 * length) - 1])[:, :, :length, length - 1:]
    return x_final


def _absolute_position_to_relative_position_traceable(self, x):
    batch, heads, length, _ = x.size()
    x = _dynamic_zero_pad_last(x, length - 1)
    x_flat = x.view([batch, heads, (length * length) + (length * (length - 1))])
    x_flat = _dynamic_zero_pad_front_last(x_flat, length)
    x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
    return x_final


MultiHeadAttention._get_relative_embeddings = _get_relative_embeddings_traceable
MultiHeadAttention._relative_position_to_absolute_position = _relative_position_to_absolute_position_traceable
MultiHeadAttention._absolute_position_to_relative_position = _absolute_position_to_relative_position_traceable


def _wn_forward_traceable(wn, x, x_mask, g=None, **kwargs):
    """Real WN.forward's body, with `n_channels_tensor = torch.IntTensor([self.hidden_channels])` (the
    only thing coremltools' torch frontend rejects here -- 'PyTorch convert function for op
    'intimplicit' not implemented', from constructing a Tensor around a plain trace-time-constant int)
    replaced by using that same constant int directly. `fused_add_tanh_sigmoid_multiply`'s own body
    (commons.py) is inlined rather than called, since it immediately unwraps the tensor back to a plain
    int via `n_channels[0]` anyway -- mathematically identical either way, hidden_channels is always a
    static architecture constant.
    """
    output = torch.zeros_like(x)
    n_channels = wn.hidden_channels
    if g is not None:
        g = wn.cond_layer(g)
    for i in range(wn.n_layers):
        x_in = wn.in_layers[i](x)
        if g is not None:
            cond_offset = i * 2 * wn.hidden_channels
            g_l = g[:, cond_offset:cond_offset + 2 * wn.hidden_channels, :]
        else:
            g_l = torch.zeros_like(x_in)
        in_act = x_in + g_l
        t_act = torch.tanh(in_act[:, :n_channels, :])
        s_act = torch.sigmoid(in_act[:, n_channels:, :])
        acts = t_act * s_act
        acts = wn.drop(acts)
        res_skip_acts = wn.res_skip_layers[i](acts)
        if i < wn.n_layers - 1:
            res_acts = res_skip_acts[:, :wn.hidden_channels, :]
            x = (x + res_acts) * x_mask
            output = output + res_skip_acts[:, wn.hidden_channels:, :]
        else:
            output = output + res_skip_acts
    return output * x_mask


vits_modules.WN.forward = _wn_forward_traceable


def _text_encoder_forward(enc, tokens):
    """Real TextEncoder.forward's body (unmodified self.emb/self.encoder/self.proj submodule calls),
    with `sequence_mask(x_lengths, T)` replaced by a direct all-ones mask. Two reasons this is safe: (1)
    this whole project's "single unpadded utterance" convention means x_lengths == T always, so the real
    mask is always all-ones anyway; (2) `torch.full((1,), T, ...)` with a DYNAMIC value T fails to
    convert on its own merits regardless ('full' op converter calls `int(inputs[1].val)` unconditionally,
    no dynamic-value fallback) -- confirmed independently of anything loom-specific.
    `torch.ones_like` hits the "dynamic shape, static value" fill path instead, which converts fine.
    """
    x = enc.emb(tokens) * math.sqrt(enc.hidden_channels)
    x = torch.transpose(x, 1, -1)
    x_mask = torch.ones_like(x[:, :1, :])
    x_cond = enc.encoder(x * x_mask, x_mask)
    return x_cond, x_mask


def _conv_flow_reverse_traceable(flow, x, x_mask, g):
    """Real ConvFlow.forward's reverse branch, with the boolean-mask-indexed
    piecewise_rational_quadratic_transform call replaced by the loom.spline_inverse custom op (see
    `vits_spline_op.py`) -- everything else (pre/convs/proj, the reshape/permute into per-bin logits) is
    the real, unmodified computation.
    """
    x0, x1 = torch.split(x, [flow.half_channels] * 2, 1)
    h = flow.pre(x0)
    h = flow.convs(h, x_mask, g=g)
    h = flow.proj(h) * x_mask
    b, c, t = x0.shape
    h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)
    num_bins = flow.num_bins
    filt = flow.filter_channels
    uw = h[..., :num_bins] / math.sqrt(filt)
    uh = h[..., num_bins:2 * num_bins] / math.sqrt(filt)
    ud = h[..., 2 * num_bins:]
    x1_out = torch.ops.loom.spline_inverse(x1, uw, uh, ud)
    return torch.cat([x0, x1_out], 1) * x_mask


class StatsWrapper(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, tokens):
        x_cond, x_mask = _text_encoder_forward(self.enc, tokens)
        stats = self.enc.proj(x_cond) * x_mask  # [1, 2*out_channels, T]
        # ggml ne=[T, 2*out_channels] (T-fast) -- deliberately NOT transposed to a channel-fast layout: a
        # bare PERMUTE as the topology's own declared OUTPUT is a live, non-contiguous view
        # (ggml_backend_tensor_get does a raw contiguous byte copy, which silently ignores a dangling
        # permute's logical reordering and returns the PRE-permute byte order instead) -- confirmed
        # empirically (a `.transpose(1,2)` here reported a channel-fast ne=[2*out_channels,T] while the
        # actual read-out bytes stayed T-fast regardless). Keeping the natural (untransposed) trace
        # output sidesteps this rather than depending on the exporter to insert a CONT after a permute
        # that happens to be the graph's own final node.
        return stats


class LogwWrapper(torch.nn.Module):
    """Real StochasticDurationPredictor.forward's reverse branch (models.py:108-117; see BACKLOG.md's
    "remove a useless vflow" note for the `flows[:-2]+[flows[-1]]` slicing this replicates), with the
    internal `z = torch.randn(...)` sampling replaced by an explicit `z_noise` graph input (host-
    supplied, ALREADY scaled by noise_scale_w -- matches reference_forward_vits.py's own convention of
    passing noise_scale=1.0 to dp() once the injected noise already carries the real scale).
    """
    def __init__(self, enc, dp):
        super().__init__()
        self.enc = enc
        self.dp = dp

    def forward(self, tokens, z_noise):
        x_cond, x_mask = _text_encoder_forward(self.enc, tokens)

        dp = self.dp
        h = torch.detach(x_cond)
        h = dp.pre(h)
        h = dp.convs(h, x_mask, g=None)
        h = dp.proj(h) * x_mask  # g_cond, [1, filter_channels, T]

        order = [dp.flows[8], dp.flows[7], dp.flows[6], dp.flows[5],
                 dp.flows[4], dp.flows[3], dp.flows[2], dp.flows[0]]
        z = z_noise
        for flow in order:
            if isinstance(flow, vits_modules.Flip):
                z = flow(z, x_mask, g=h, reverse=True)
            elif isinstance(flow, vits_modules.ConvFlow):
                z = _conv_flow_reverse_traceable(flow, z, x_mask, h)
            else:  # ElementwiseAffine
                z = flow(z, x_mask, reverse=True)

        z0, _z1 = torch.split(z, [1, 1], 1)
        return z0  # [1, 1, T] -> ggml ne=[T,1,1] == [T]


class FlowVocoderWrapper(torch.nn.Module):
    """Real ResidualCouplingBlock(reverse) + HiFi-GAN Generator -- neither has any boolean-mask
    indexing or data-dependent branching (WN's `if g is not None` is a static Python None check, always
    the same branch since gin_channels=0/g=None for this single-speaker checkpoint), so both trace
    essentially unmodified.
    """
    def __init__(self, flow, dec):
        super().__init__()
        self.flow = flow
        self.dec = dec

    def forward(self, z_p):
        y_mask = torch.ones_like(z_p[:, :1, :])
        z = self.flow(z_p, y_mask, g=None, reverse=True)
        wav = self.dec(z, g=None)
        return wav.reshape(-1)


@dataclass(kw_only=True)
class TTSVitsExportConfig(BaseMultiPhaseModelExportConfig):
    """VITS's own three-phase split (stats/logw/flow_vocoder) -- see module docstring. `phases()` loads
    the real checkpoint once per call, mirroring the original script's `main()` (load once, build all
    three phases off the one loaded model instance).
    """

    checkpoint_path: str
    driver_script_path: Path = Path(__file__).resolve().parent.parent / "convert_piper_vits" / "vits_driver_mil.lua"
    dummy_t: int = 42

    def phases(self) -> List[ExportPhase]:
        print(f"Loading checkpoint {self.checkpoint_path}...")
        full_sd = load_piper_checkpoint(self.checkpoint_path)
        sd = {k[len("model_g."):]: v for k, v in full_sd.items() if k.startswith("model_g.")}

        model = SynthesizerTrn(**HP)
        missing, unexpected = model.load_state_dict(sd, strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        model.eval()

        seq_dim = ct.RangeDim(1, 512)
        dummy_tokens = torch.randint(1, HP["n_vocab"], (1, self.dummy_t), dtype=torch.long)
        dummy_z_noise = torch.randn(1, 2, self.dummy_t) * 0.8
        dummy_zp = torch.randn(1, HP["inter_channels"], 8) * 0.5

        print("Tracing all three phases...")
        return [
            ExportPhase(
                name="stats", wrapper=StatsWrapper(model.enc_p).eval(), dummy_inputs=(dummy_tokens,),
                mil_inputs=[ct.TensorType(name="tokens", shape=(1, seq_dim), dtype=np.int32)],
            ),
            ExportPhase(
                name="logw", wrapper=LogwWrapper(model.enc_p, model.dp).eval(),
                dummy_inputs=(dummy_tokens, dummy_z_noise),
                mil_inputs=[
                    ct.TensorType(name="tokens", shape=(1, seq_dim), dtype=np.int32),
                    ct.TensorType(name="z_noise", shape=(1, 2, seq_dim), dtype=np.float32),
                ],
            ),
            ExportPhase(
                name="flow_vocoder", wrapper=FlowVocoderWrapper(model.flow, model.dec).eval(),
                dummy_inputs=(dummy_zp,),
                mil_inputs=[ct.TensorType(name="z_p", shape=(1, HP["inter_channels"], seq_dim), dtype=np.float32)],
            ),
        ]


def _is_vits(path: Path) -> bool:
    """No self-describing config for this checkpoint format (a raw piper Lightning `.ckpt`) -- always
    False, requiring an explicit `--task tts-multi-phase --model vits` (BACKLOG.md P3.3's stated scope
    limit, same as `optimum` itself needing `--task` for sufficiently custom architectures)."""
    return False


def _build_vits(path: Path, output_path: str) -> TTSVitsExportConfig:
    return TTSVitsExportConfig(architecture="vits_mil", output_path=output_path, checkpoint_path=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="tts-multi-phase",
        config_class=BaseMultiPhaseModelExportConfig,
        recognizers=[ModelRecognizer(name="vits", detect=_is_vits, build_config=_build_vits)],
    ))
