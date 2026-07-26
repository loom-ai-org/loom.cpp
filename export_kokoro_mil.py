#!/usr/bin/env python3
"""
WORK IN PROGRESS -- exports ONLY the "decoder_vocoder" phase so far (Decoder.encode/decode +
SineGen + STFT + Generator, the single riskiest/most error-prone chunk of Kokoro, per
tools/convert_kokoro/convert_kokoro_{decoder_core,sinegen,stft,generator}.py's own bespoke
counterparts -- this MIL trace replaces all four of those in one combined topology). Traces the REAL
kokoro.istftnet.Decoder module directly. Structurally verified end-to-end: builds AND computes a
finite waveform for a real, tiny dummy T (see tests/test_e2e_kokoro_mil_decoder_vocoder_smoke.cpp) --
NOT yet numerically verified against a real reference (none exists for this phase yet). NOT yet done:
CustomAlbert+bert_encoder MIL phase, duration_predictor/text_encoder/f0n LSTM-bound pieces (reuse the
existing bespoke GGUFs -- LSTM has no ggml-native op, see tools/loom_mil_compiler/recurrent.py), a
combined kokoro_driver_mil.lua, and that numerical verification. See BACKLOG.md for the full status
and the many real, general exporter/primitive bugs this phase surfaced (instance_norm, depthwise
CONV_TRANSPOSE composition, ATAN, INTERPOLATE_1D/upsample dynamic-dim tracking, CUMSUM, scalar-weight
GGUF serialization, get_var_info's gating overhaul, the `symbol_overrides` mechanism for topologies
with more than one independently-varying dynamic input, and -- the one that took longest to root-cause
-- `ggml_is_contiguous()` vacuously passing a permuted tensor with `ne[0]==1`, silently corrupting
every elementwise binary/unary primitive's own contiguity guard until replaced with a stricter
`ensure_packed()` check across primitives_basic.cpp/primitives_mil.cpp).

Usage:
  ~/.venvs/piper/bin/python3 export_kokoro_mil.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct
from coremltools.converters.mil.frontend.torch import ops as _torch_ops
from coremltools.converters.mil.mil import Builder as _mb

# --- coremltools/numpy version-incompatibility workaround (self-contained, doesn't touch the
# installed package) ---------------------------------------------------------------------------
# coremltools 9.0's own `_cast` op converter does `dtype(x.val)` unconditionally once it's decided
# `x` is a compile-time constant "scalar or length-1 tensor" (its own docstring/shape check already
# allows a genuine (1,1,...,1)-shaped array, not just a true 0-d one) -- but numpy>=1.25 rejects
# int()/float() on anything but a strictly 0-d array, even a 1-element one. First hit tracing
# SineGen's dynamic-length F.interpolate calls (their own internal size computation produces exactly
# this shape). Squeeze to a genuine 0-d value first, matching the intent of the pre-existing check.
_orig_cast = _torch_ops._cast


def _patched_cast(context, node, dtype, dtype_name):
    inputs = _torch_ops._get_inputs(context, node, expected=1)
    x = inputs[0]
    if x.can_be_folded_to_const() and not isinstance(x.val, dtype):
        val = np.asarray(x.val).reshape(())
        context.add(_mb.const(val=dtype(val), name=node.name))
        return
    return _orig_cast(context, node, dtype, dtype_name)


_torch_ops._cast = _patched_cast

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools" / "convert_kokoro"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
from kokoro_stft_common import build_forward_dft_kernels, build_inverse_synth_kernels  # noqa: E402

# --- transformers/huggingface-hub version-pin workaround (self-contained) ----------------------
# The `kokoro` package's own CustomAlbert needs a real `transformers.AlbertModel`, but this venv's
# huggingface_hub (1.24.0, needed for other tools sharing this venv, e.g. NeMo) is newer than
# transformers 4.57.6's own `<1.0` advisory pin -- an ImportError at package-metadata-check time, not
# a genuine runtime incompatibility (confirmed: AlbertModel imports and runs fine once the check
# itself is bypassed). Stubbing out just the version-check submodule (not modifying either installed
# package) sidesteps it -- same self-contained-monkeypatch convention as the _cast fix above.
import types  # noqa: E402
_stub = types.ModuleType("transformers.utils.versions")
_stub.require_version = lambda *a, **k: None
_stub.require_version_core = lambda *a, **k: None
sys.modules["transformers.utils.versions"] = _stub

from kokoro.model import KModel  # noqa: E402
from kokoro.istftnet import AdainResBlk1d, SineGen, SourceModuleHnNSF, Generator, Decoder  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools" / "loom_mil_compiler" / ".."))
import loom_mil_compiler  # noqa: E402  registers the "loom" backend + torch-frontend patches
from loom_mil_compiler.exporter import LoomGGUFExporter  # noqa: E402

CKPT_PATH = "/home/flavio/.claude/tmp/kokoro_model/kokoro-v1_0.pth"
CONFIG_PATH = "/home/flavio/.claude/tmp/kokoro_model/config.json"

_STFT_N_FFT = 20  # gen_istft_n_fft, real checkpoint config
_STFT_HOP = 5  # gen_istft_hop_size
_HARMONIC_NUM = 8  # real SourceModuleHnNSF default
_UPSAMPLE_SCALE = 300  # math.prod(upsample_rates) * gen_istft_hop_size = 10*6*5


# ---- Trace-friendly AdainResBlk1d/SineGen/SourceModuleHnNSF/Generator/Decoder patches ----

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _adain_resblk1d_forward_traceable(self, x, s):
    """Real forward, with `torch.rsqrt(torch.tensor(2))` (an int64-typed constant tensor MIL's own
    rsqrt op rejects -- 'expects tensor of dtype fp16/fp32 but got tensor[int32]') replaced by the
    equivalent plain float constant, matching how tools/convert_kokoro/convert_kokoro_f0n.py's own
    `add_adain_resblk1d` already reduces this identical expression to a constant SCALE."""
    out = self._residual(x, s)
    return (out + self._shortcut(x)) * _INV_SQRT2


AdainResBlk1d.forward = _adain_resblk1d_forward_traceable


class VerifiedSTFT(torch.nn.Module):
    """Replaces istftnet.py's CustomSTFT -- same conv-based DFT kernels
    (tools/convert_kokoro/kokoro_stft_common.py, already verified against real torch.stft to ~1e-6),
    NOT the real (disable_complex=False) TorchSTFT this checkpoint would use in production (torch.stft
    on complex dtype isn't traceable by coremltools at all) and NOT CustomSTFT's own approximate
    reconstruction either (adds the real torch.istft's own window-sum-square/NOLA normalization
    CustomSTFT never applies -- confirmed reading custom_stft.py, no wsum division anywhere; needed to
    match the checkpoint's REAL production numerics, not CustomSTFT's own admittedly-approximate one).
    Kernels registered as buffers (not closed-over plain tensors) -- a plain module-level
    `torch.from_numpy(...)` closed over by a free function traced fine in isolation but lost its shape
    metadata once embedded in the full Decoder trace (a real, narrow coremltools quirk; CustomSTFT's
    own registered-buffer convention side-steps it)."""

    def __init__(self, n_fft, hop):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        cos_k, neg_sin_k, boundary_mask = build_forward_dft_kernels(n_fft)
        cos_synth, neg_sin_synth = build_inverse_synth_kernels(n_fft)
        self.register_buffer("cos_k", torch.from_numpy(cos_k))
        self.register_buffer("neg_sin_k", torch.from_numpy(neg_sin_k))
        self.register_buffer("boundary_mask", torch.from_numpy(boundary_mask).view(1, -1, 1))
        self.register_buffer("cos_synth", torch.from_numpy(cos_synth))
        self.register_buffer("neg_sin_synth", torch.from_numpy(neg_sin_synth))

    def transform(self, waveform):
        pad = self.n_fft // 2
        waveform = F.pad(waveform, (pad, pad), mode="reflect")
        x = waveform.unsqueeze(1)
        re = F.conv1d(x, self.cos_k, bias=None, stride=self.hop, padding=0)
        im_raw = F.conv1d(x, self.neg_sin_k, bias=None, stride=self.hop, padding=0)
        im = im_raw - im_raw * self.boundary_mask  # exact +0.0 at DC/Nyquist bins, matching real torch.stft
        mag = torch.sqrt(re ** 2 + im ** 2)
        phase = torch.atan2(im, re)
        return mag, phase

    def inverse(self, magnitude, phase, wsum):
        """wsum depends only on the frame count (a fixed function of input length, not of any real
        data) -- host-precomputed and fed in as a declared input, same convention as every other
        host-derived-constant in this project (VITS's own z_p noise, this project's kq_mask, ...)."""
        re = magnitude * torch.cos(phase)
        im = magnitude * torch.sin(phase)
        re_contrib = F.conv_transpose1d(re, self.cos_synth, bias=None, stride=self.hop, padding=0)
        im_contrib = F.conv_transpose1d(im, self.neg_sin_synth, bias=None, stride=self.hop, padding=0)
        numerator = re_contrib + im_contrib
        normalized = numerator / wsum
        pad = self.n_fft // 2
        return normalized[..., pad:-pad]


def _f02sine_traceable(self, f0_values, rand_ini):
    """Real _f02sine's math unmodified except: (1) rand_ini is a declared input, not torch.rand(...);
    (2) both F.interpolate(mode='linear') calls go through a rank-4 'unsqueeze->bilinear->squeeze'
    trick (coremltools' torch frontend maps 1D linear interpolate to the SAME torch_upsample_bilinear
    dialect op as 2D, which hard-requires rank 4); (3) the intermediate cumsum stays in that same
    rank-4 shape (dim=3, the time axis, instead of transposing back to rank 3 and re-unsqueezing for
    the second interpolate call) -- confirmed empirically that squeezing to rank 3 then re-unsqueezing
    between two chained interpolate calls loses the second one's STATIC rank in coremltools' own MIL
    type inference ('input to the torch_upsample_bilinear op must have rank 4', even though the real
    eager-mode rank is always 4) -- a real, narrow coremltools shape-propagation gap, not a rank bug in
    this code. Verified bit-level equivalent to the original (dim=1 cumsum on a rank-3 tensor) up to
    cumsum's own floating-point reduction-order noise (~6e-5 abs, from summing along a different axis
    -- non-associativity of float addition, not a logic difference).
    """
    rad_values = (f0_values / self.sampling_rate) % 1
    rand_ini_full = torch.cat([torch.zeros_like(rand_ini[:, :1]), rand_ini[:, 1:]], dim=1)
    rad0 = rad_values[:, 0, :] + rand_ini_full
    rad_values = torch.cat([rad0.unsqueeze(1), rad_values[:, 1:, :]], dim=1)  # (B,L,dim)

    x4 = rad_values.transpose(1, 2).unsqueeze(2)  # (B,dim,1,L)
    down4 = F.interpolate(x4, scale_factor=(1, 1 / self.upsample_scale), mode="bilinear",
                           recompute_scale_factor=True, align_corners=False)  # (B,dim,1,L')
    phase4 = torch.cumsum(down4, dim=3) * 2 * torch.pi
    phase4 = phase4 * self.upsample_scale
    up4 = F.interpolate(phase4, scale_factor=(1, self.upsample_scale), mode="bilinear",
                         recompute_scale_factor=True, align_corners=False)  # (B,dim,1,L)
    # NOT up4.squeeze(2): a squeeze() immediately downstream of this rank-4-forced interpolate loses
    # static RANK tracking several ops further downstream in coremltools' own MIL type inference (a
    # real, narrow, reproduced-in-isolation coremltools bug -- an explicit reshape built from real
    # shape queries doesn't trip it, confirmed by bisection).
    b, c, ln = up4.shape[0], up4.shape[1], up4.shape[3]
    phase = up4.reshape(b, c, ln).transpose(1, 2)  # (B,L,dim)
    return torch.sin(phase)


def _sine_gen_forward_traceable(self, f0, rand_ini, noise_in):
    """Real code: `f0 * torch.arange(1, dim+1).view(1,1,-1)` -- f0 is (B,L,1), the arange is (1,1,dim),
    broadcasting along TWO DIFFERENT axes at once (L on f0's side, dim on the arange's side). MIL
    itself handles this arbitrary broadcast fine, but ggml_mul's simpler "one operand's shape must be
    a per-axis divisor of the other's" repeat model genuinely cannot express it in one call regardless
    of operand order (confirmed: SchemaError even after op_mul's own commutative-swap heuristic).
    Composed instead via `dim` unrolled per-harmonic SCALE + CONCAT calls (dim=9, a small static
    constant) -- the exact same fix tools/convert_kokoro/convert_kokoro_sinegen.py's own bespoke
    topology already needed for this identical shape, per its own docstring ("no generic outer-product/
    broadcast-repeat primitive existed").
    """
    fn = torch.cat([f0 * float(k) for k in range(1, self.harmonic_num + 2)], dim=-1)
    sine_waves = self._f02sine(fn, rand_ini) * self.sine_amp
    uv = self._f02uv(f0)
    noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
    noise = noise_amp * noise_in
    sine_waves = sine_waves * uv + noise
    return sine_waves, uv, noise


SineGen._f02sine = _f02sine_traceable
SineGen.forward = _sine_gen_forward_traceable


def _source_module_forward_traceable(self, x, rand_ini, noise_in):
    sine_wavs, uv, _ = self.l_sin_gen(x, rand_ini, noise_in)
    sine_merge = self.l_tanh(self.l_linear(sine_wavs))
    noise = noise_in[..., :1] * 0.0  # unused downstream (Generator never reads noise/uv outputs)
    return sine_merge, noise, uv


SourceModuleHnNSF.forward = _source_module_forward_traceable


def _generator_forward_traceable(self, x, s, f0, rand_ini, noise_in, wsum):
    f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
    har_source, noi_source, uv = self.m_source(f0, rand_ini, noise_in)
    har_source = har_source.transpose(1, 2)  # (B,1,L)
    har_source = har_source.reshape(har_source.shape[0], har_source.shape[2])  # not .squeeze(1), see _f02sine
    har_spec, har_phase = self.verified_stft.transform(har_source)
    har = torch.cat([har_spec, har_phase], dim=1)
    for i in range(self.num_upsamples):
        x = F.leaky_relu(x, negative_slope=0.1)
        x_source = self.noise_convs[i](har)
        x_source = self.noise_res[i](x_source, s)
        x = self.ups[i](x)
        if i == self.num_upsamples - 1:
            x = self.reflection_pad(x)
        x = x + x_source
        xs = None
        for j in range(self.num_kernels):
            block_out = self.resblocks[i * self.num_kernels + j](x, s)
            xs = block_out if xs is None else xs + block_out
        x = xs / self.num_kernels
    x = F.leaky_relu(x)
    x = self.conv_post(x)
    spec = torch.exp(x[:, :self.post_n_fft // 2 + 1, :])
    phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
    return self.verified_stft.inverse(spec, phase, wsum)


Generator.forward = _generator_forward_traceable


def _decoder_forward_traceable(self, asr, F0_curve, N, s, rand_ini, noise_in, wsum):
    F0 = self.F0_conv(F0_curve.unsqueeze(1))
    N = self.N_conv(N.unsqueeze(1))
    x = torch.cat([asr, F0, N], axis=1)
    x = self.encode(x, s)
    asr_res = self.asr_res(asr)
    res = True
    for block in self.decode:
        if res:
            x = torch.cat([x, asr_res, F0, N], axis=1)
        x = block(x, s)
        if block.upsample_type != "none":
            res = False
    return self.generator(x, s, F0_curve, rand_ini, noise_in, wsum)


Decoder.forward = _decoder_forward_traceable


class DecoderVocoderWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.dec = decoder

    def forward(self, asr, f0_curve, n_curve, s, rand_ini, noise_in, wsum):
        return self.dec(asr, f0_curve, n_curve, s, rand_ini, noise_in, wsum)


def compute_wsum_np(t_frames, n_fft=_STFT_N_FFT, hop=_STFT_HOP, upsample_scale=_UPSAMPLE_SCALE):
    """Matches kokoro_stft_common.compute_wsum, driven by t_frames (=T_frames, this wrapper's own
    "n_tokens" base symbol) rather than a raw frame count directly."""
    t_f0 = 2 * t_frames
    length = t_f0 * upsample_scale
    pad = n_fft // 2
    padded_len = length + 2 * pad
    t_har = (padded_len - n_fft) // hop + 1
    out_len_full = (t_har - 1) * hop + n_fft
    window = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)
    wsum = np.zeros(out_len_full, dtype=np.float32)
    for t in range(t_har):
        wsum[t * hop:t * hop + n_fft] += window ** 2
    return wsum


def build_decoder_vocoder_topology(decoder, dummy_t_frames=40, dim_in=512):
    """Traces DecoderVocoderWrapper and runs it through the exporter far enough to get back its
    topology JSON + weight dict, mirroring export_vits_mil.py's own `_build_topology` helper."""
    decoder.generator.verified_stft = VerifiedSTFT(_STFT_N_FFT, _STFT_HOP)
    wrapper = DecoderVocoderWrapper(decoder).eval()

    dim = _HARMONIC_NUM + 1
    t_f0 = 2 * dummy_t_frames
    length = t_f0 * _UPSAMPLE_SCALE
    dummy_args = (
        torch.randn(1, dim_in, dummy_t_frames) * 0.3,
        torch.randn(1, t_f0) * 50 + 100,
        torch.rand(1, t_f0),
        torch.randn(1, 128) * 0.5,
        torch.rand(1, dim),
        torch.randn(1, length, dim),
        torch.from_numpy(compute_wsum_np(dummy_t_frames)),
    )
    with torch.no_grad():
        wrapper(*dummy_args)  # eager sanity check before tracing

    traced = torch.jit.trace(wrapper, dummy_args)
    seq = ct.RangeDim(1, 2000)
    mil_inputs = [
        ct.TensorType(name="asr", shape=(1, dim_in, seq), dtype=np.float32),
        ct.TensorType(name="f0_curve", shape=(1, ct.RangeDim(1, 4000)), dtype=np.float32),
        ct.TensorType(name="n_curve", shape=(1, ct.RangeDim(1, 4000)), dtype=np.float32),
        ct.TensorType(name="s", shape=(1, 128), dtype=np.float32),
        ct.TensorType(name="rand_ini", shape=(1, dim), dtype=np.float32),
        ct.TensorType(name="noise_in", shape=(1, ct.RangeDim(1, 600000), dim), dtype=np.float32),
        ct.TensorType(name="wsum", shape=(ct.RangeDim(1, 600000),), dtype=np.float32),
    ]
    prog = ct.convert(traced, inputs=mil_inputs, convert_to="milinternal",
                       compute_precision=ct.precision.FLOAT32)
    main_func = prog.functions["main"]

    # f0_curve/n_curve/noise_in/wsum are genuinely independent LEAF inputs whose real length is a
    # fixed multiple of asr's own T (2x/600x/600x+20 respectively) -- not derivable from the graph
    # (see LoomGGUFExporter.symbol_overrides' own docstring in exporter.py). Built from the REAL
    # traced symbol names (not guessed) by reading each input Var's own dynamic shape entry directly.
    def root_symbol(name, axis):
        return str(main_func.inputs[name].shape[axis])

    symbol_overrides = {
        root_symbol("f0_curve", 1): "2*n_tokens",
        root_symbol("n_curve", 1): "2*n_tokens",
        root_symbol("noise_in", 1): "600*n_tokens",
        root_symbol("wsum", 0): "600*n_tokens+20",
    }
    exporter = LoomGGUFExporter(prog, symbol_overrides=symbol_overrides)
    topo = exporter.generate_graph_topology(main_func, "decoder_vocoder")
    print(f"  decoder_vocoder: {len(topo['nodes'])} nodes, {len(exporter.weights)} weights")
    return topo, exporter.weights


def main():
    print(f"Loading checkpoint {CKPT_PATH}...")
    cfg = json.load(open(CONFIG_PATH))
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=cfg, model=CKPT_PATH, disable_complex=True)
    model.eval()

    print("Tracing decoder_vocoder phase...")
    topo, weights = build_decoder_vocoder_topology(model.decoder)

    out_exporter = LoomGGUFExporter(None, output_path="kokoro_decoder_vocoder_mil.gguf",
                                     architecture="loom-kokoro-decoder-vocoder-mil")
    out_exporter.topologies = {"decoder_vocoder": topo}
    out_exporter.weights = weights
    out_exporter.write_gguf("")  # no driver script yet -- see module docstring for remaining work


if __name__ == "__main__":
    main()
