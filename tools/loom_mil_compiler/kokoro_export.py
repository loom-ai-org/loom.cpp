"""Exports Kokoro-82M's two LSTM-free phases through `TTSKokoroExportConfig` (BACKLOG.md P3.3, migrated
from `export_kokoro_mil.py`), into ONE combined `kokoro_mil.gguf` alongside the embedded
`kokoro_driver_mil.lua` orchestration script:
  - "albert_bert_encoder": CustomAlbert (a real transformers.AlbertModel) + bert_encoder (Linear(768,512))
    -- replaces convert_kokoro_albert.py + convert_kokoro_bert_encoder.py's two hand-built topologies.
  - "decoder_vocoder": Decoder.encode/decode + SineGen + STFT + Generator -- replaces FOUR bespoke
    scripts, convert_kokoro_{decoder_core,sinegen,stft,generator}.py, in one combined trace.
Kokoro leans heavily on `torch.nn.LSTM` elsewhere (TextEncoder, DurationEncoder, predictor.lstm,
F0Ntrain's shared LSTM) -- ggml has no native LSTM op, so those pieces are a deliberate scoping
exclusion: `kokoro_driver_mil.lua` wires these two MIL-traced topologies together with the EXISTING
bespoke, hand-built LSTM-bound topologies (`tools/convert_kokoro/convert_kokoro_lua_all.py`'s own
`kokoro.gguf`), not a full re-trace of the whole model.

Both phases numerically verified against real-checkpoint references
(`tools/convert_kokoro/reference_forward_kokoro_{albert_bert_encoder_mil,decoder_vocoder_mil}.py`):
albert_bert_encoder to ~1.8e-6 mean/~1.5e-5 max abs diff; decoder_vocoder to ~5.1e-4 mean/~2.5e-2 max abs
diff (see `test_e2e_kokoro_mil_decoder_vocoder_reference.cpp`'s own comments for why the looser max bound
-- a real, bounded, HiFi-GAN-vocoder amplification ceiling, same category as StyleTTS2's own documented
one, not a further-fixable bug). Getting both phases to trace/build/compute at all surfaced the usual
long tail of general (not Kokoro-specific) exporter/primitive bugs -- see BACKLOG.md for the full trail
-- but TWO were only caught by this numerical-verification pass specifically (structural verification
alone, i.e. "builds and produces a finite waveform," did not, and would not, catch either):
  - `op_sub`'s scalar-broadcast shortcut (`SUB(a,b)` with `nelements(a)==1 < nelements(b)`) computed
    `ggml_neg(b)` unconditionally -- correct ONLY for the `(0.0 - b)` idiom it was named after, silently
    WRONG (dropping `a` entirely) for any other nonzero scalar `a`. First hit by HF's ubiquitous
    `1.0 - mask` attention-masking idiom (CustomAlbert's own `get_extended_attention_mask`). Fixed
    generally in src/ops/primitives_basic.cpp: explicitly REPEAT the smaller operand before subtracting,
    valid for any value.
  - coremltools' own `torch.atan2(y,x)` decomposition (NOT MIL's fused `atan2` op -- for this model it
    traces to a plain `atan(y/x)` op plus a manually-composed quadrant correction) has a real gap: it
    covers the x<0 case's y>0/y<0 STRICT branches but not the y==0,x<0 boundary (real
    `atan2(0.0,-1)==+pi`; the decomposition silently returns 0). `VerifiedSTFT.transform`'s own
    `im = im_raw - im_raw*boundary_mask` (zeroing the imaginary part at DC/Nyquist bins, matching real
    torch.stft) produces exactly this trigger condition at ~5.9% of all phase elements in one real trace
    -- fixed by nudging to a tiny positive epsilon instead of an exact 0.0 (see `boundary_eps`'s own
    comment below for the full reasoning and confirmation that `im` is always +0.0-derived here, never
    -0.0).

Usage:
  loom-export /path/to/kokoro/dir -o kokoro_mil.gguf --task tts-multi-phase --model kokoro
"""
import json
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct
from coremltools.converters.mil.frontend.torch import ops as _torch_ops
from coremltools.converters.mil.mil import Builder as _mb

from .checkpoint_probe import read_json
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .patcher import ModelPatcher


class KokoroModelPatcher(ModelPatcher):
    """Import-order stubs needed before Kokoro's real package (and the `transformers.AlbertModel` it
    wraps) can be imported at all -- BACKLOG.md P3.3's `ModelPatcher` hook. Both patches are
    self-contained (don't modify either installed package) and were already applied at plain module-load
    time in the original script; wrapping them in this named hook doesn't change when they run, only
    documents what they're for."""

    @staticmethod
    def prepare_environment() -> None:
        # coremltools 9.0's own `_cast` op converter does `dtype(x.val)` unconditionally once it's
        # decided `x` is a compile-time constant "scalar or length-1 tensor" (its own docstring/shape
        # check already allows a genuine (1,1,...,1)-shaped array, not just a true 0-d one) -- but
        # numpy>=1.25 rejects int()/float() on anything but a strictly 0-d array, even a 1-element one.
        # First hit tracing SineGen's dynamic-length F.interpolate calls (their own internal size
        # computation produces exactly this shape). Squeeze to a genuine 0-d value first, matching the
        # intent of the pre-existing check.
        orig_cast = _torch_ops._cast

        def _patched_cast(context, node, dtype, dtype_name):
            inputs = _torch_ops._get_inputs(context, node, expected=1)
            x = inputs[0]
            if x.can_be_folded_to_const() and not isinstance(x.val, dtype):
                val = np.asarray(x.val).reshape(())
                context.add(_mb.const(val=dtype(val), name=node.name))
                return
            return orig_cast(context, node, dtype, dtype_name)

        _torch_ops._cast = _patched_cast

        # The `kokoro` package's own CustomAlbert needs a real `transformers.AlbertModel`, but this
        # venv's huggingface_hub is newer than transformers' own `<1.0` advisory pin -- an ImportError at
        # package-metadata-check time, not a genuine runtime incompatibility (confirmed: AlbertModel
        # imports and runs fine once the check itself is bypassed).
        stub = types.ModuleType("transformers.utils.versions")
        stub.require_version = lambda *a, **k: None
        stub.require_version_core = lambda *a, **k: None
        sys.modules["transformers.utils.versions"] = stub


KokoroModelPatcher.prepare_environment()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "convert_kokoro"))
from kokoro_stft_common import build_forward_dft_kernels, build_inverse_synth_kernels  # noqa: E402

from kokoro.model import KModel  # noqa: E402
from kokoro.istftnet import AdainResBlk1d, SineGen, SourceModuleHnNSF, Generator, Decoder  # noqa: E402

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
        # +boundary_eps (not exact +0.0) at DC/Nyquist bins, PLUS a uniform tiny positive nudge
        # everywhere else -- still matches real torch.stft's own convention (a purely-real DC/Nyquist
        # bin, im "=0") to float32 precision, but sidesteps a genuine coremltools bug: torch.atan2(y,x)
        # traces (for THIS model) not to MIL's fused `atan2` op but to `atan(y/x)` plus a manually-
        # composed quadrant correction (GREATER/LESS-based additive +-pi/+-pi/2 terms) that only covers
        # the y>0/y<0 STRICT branches for the x<0 case -- omitting the y==0, x<0 boundary entirely (real
        # atan2(0.0,-1)=+pi; this decomposition silently returns 0 instead, confirmed via a standalone
        # MIL-op-level probe). `im_raw - im_raw*mask` at the boundary bins produces an EXACT 0.0 im (any
        # real x satisfies x-x=+0.0 under round-to-nearest) -- the FIRST, most reliably reproduced trigger
        # (confirmed: ~5.9% of all phase elements in one real trace) -- but real (non-boundary) `im_raw`
        # can ALSO land on exactly 0.0 by coincidence for sufficiently periodic/structured content (first
        # caught end-to-end, not by the boundary-only fix above: a real predicted F0/asr fixture produced
        # a short but violent ~40-sample resonance burst, up to ~17x the signal's own typical amplitude,
        # traced to exactly this same atan2 gap firing at a NON-boundary bin). Nudging `im` uniformly (not
        # just at the boundary positions) closes both cases -- the epsilon is physically negligible
        # (float32-representable, ~1e-20 against real STFT magnitudes of ~1e-3 to 10) and only changes
        # the OUTCOME for values already at or below float32's own precision floor, matching real atan2's
        # "+0.0 treated as the positive side" convention (confirmed: torch.atan2(0.0,-1.0)=+pi,
        # torch.atan2(-0.0,-1.0)=-pi -- this project's im is always +0.0-derived, never -0.0, at the
        # boundary bins; genuinely negative real im near zero elsewhere is far larger in magnitude than
        # this epsilon and keeps its own true sign).
        boundary_eps = 1e-20
        im = im_raw - im_raw * self.boundary_mask + self.boundary_mask * boundary_eps + boundary_eps
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
    -- non-associativity of float addition, not a logic difference). (4) `% 1` replaced with the
    algebraically identical `x - floor(x)` -- a genuine coremltools frontend bug, not this project's
    exporter: `torch.remainder(x, 1)` (`x % 1` on a tensor with divisor Python-int 1) traces to a raw
    MIL `sub(x, x)` (always exactly 0!), confirmed by reading the raw traced MIL ops directly -- looks
    like coremltools' own `remainder` lowering computes `x - floor_divide(x,y)*y` but short-circuits
    `floor_divide(x,1)` to plain `x` (a valid `real_div`-by-1 optimization, invalid for `floor_divide`,
    which must still floor). First caught by this exact line collapsing SineGen's entire phase signal to
    a constant, verified via a standalone MIL-op-level probe before writing this workaround.
    """
    rad_values = (f0_values / self.sampling_rate)
    rad_values = rad_values - torch.floor(rad_values)
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


class AlbertBertEncoderWrapper(torch.nn.Module):
    """Traces CustomAlbert (KModel.bert, a real transformers.AlbertModel returning last_hidden_state)
    + bert_encoder (plain Linear(768,512)) as ONE combined topology, replacing
    convert_kokoro_albert.py + convert_kokoro_bert_encoder.py's two hand-built graphs. Real call site
    (model.py's forward_with_tokens): `bert_dur = self.bert(input_ids, attention_mask=(~text_mask)
    .int()); d_en = self.bert_encoder(bert_dur).transpose(-1,-2)`. `attention_mask` is always all-ones
    here (real usage is always a single, unpadded utterance -- same "no real masking" convention
    convert_kokoro_albert.py's own docstring already established for the bespoke topology), so it's
    synthesized in-graph from input_ids' own dynamic shape rather than declared as a separate input.

    Deliberately does NOT apply the real code's own final `.transpose(-1,-2)`: a bare permute as a
    traced graph's own declared OUTPUT is a live, non-contiguous view that this exporter's raw
    contiguous byte copy would silently read in PRE-permute order (the exact bug export_vits_mil.py's
    own StatsWrapper docstring found and worked around for VITS's `stats` output). Returns the natural
    (T,512) time-major layout instead (ggml ne=[512,T], flat[t*512+c] -- kokoro_driver.lua's own
    "row_major" convention) -- kokoro_driver_mil.lua converts to per-timestep rows via
    `from_row_major`, no transpose needed on the Lua side either.
    """

    def __init__(self, bert, bert_encoder):
        super().__init__()
        self.bert = bert
        self.bert_encoder = bert_encoder

    def forward(self, input_ids):
        attention_mask = torch.ones_like(input_ids)
        # Explicit all-zeros token_type_ids, matching the real code's own default -- but passed
        # explicitly rather than left None. When None, AlbertEmbeddings.forward derives it from a
        # REGISTERED BUFFER via `self.token_type_ids[:, :seq_length].expand(input_shape[0], seq_length)`,
        # which MIL traces as a `slice_by_index` feeding a `tile` op whose `reps` is itself a runtime-
        # computed ratio (not a compile-time constant) -- a real exporter bug (a SECOND `get_var_info`
        # lookup of that slice's own output resolves its dynamic axis to the WRONG symbol, observed
        # concretely producing a bogus target shape of 512 -- bert_encoder's hidden_dim, from a totally
        # unrelated later op -- instead of n_tokens) not worth root-causing here since the real
        # `AlbertEmbeddings.forward` already supports this exact bypass as a first-class argument.
        token_type_ids = torch.zeros_like(input_ids)
        bert_dur = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        d_en = self.bert_encoder(bert_dur)  # (1,T,512) -- NOT .transpose(-1,-2), see class docstring
        return d_en.reshape(d_en.shape[1], d_en.shape[2])  # (T,512)


def build_albert_bert_encoder_phase(bert, bert_encoder, dummy_t=7, vocab_size=178) -> ExportPhase:
    """Builds the `albert_bert_encoder` phase's `ExportPhase` (wrapper + dummy inputs + MIL input
    declarations) -- mirrors `build_decoder_vocoder_phase`'s own shape. Was
    `build_albert_bert_encoder_topology`, which additionally traced/converted/generated the topology
    itself; that part is now `BaseMultiPhaseModelExportConfig.export()`'s shared loop, so this function
    only builds the phase's own inputs -- same eager sanity-check-before-tracing behavior, just with the
    trace itself deferred to the shared loop."""
    wrapper = AlbertBertEncoderWrapper(bert, bert_encoder).eval()

    dummy_input_ids = torch.randint(0, vocab_size, (1, dummy_t), dtype=torch.long)
    with torch.no_grad():
        wrapper(dummy_input_ids)  # eager sanity check before tracing

    seq = ct.RangeDim(1, 2000)
    return ExportPhase(
        name="albert_bert_encoder", wrapper=wrapper, dummy_inputs=(dummy_input_ids,),
        mil_inputs=[ct.TensorType(name="tokens", shape=(1, seq), dtype=np.int32)],
    )


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


def build_decoder_vocoder_phase(decoder, dummy_t_frames=40, dim_in=512) -> ExportPhase:
    """Builds the `decoder_vocoder` phase's `ExportPhase` -- kept as a MODULE-LEVEL function (not a
    method on `TTSKokoroExportConfig`) so `styletts2_export.py` can still call it directly: StyleTTS2 and
    Kokoro share the identical iSTFTNet decoder/vocoder architecture (BACKLOG.md P3.3's "real
    cross-model dependency" note -- was `import export_kokoro_mil as ekm; ekm.build_decoder_vocoder_topology(decoder)`).
    Was `build_decoder_vocoder_topology`, which additionally traced/converted/generated the topology
    itself; see `build_albert_bert_encoder_phase`'s own docstring for why that part moved to the shared
    export loop.
    """
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
    # "asr"'s own axis is this phase's one true dynamic quantity -- the post-encoder acoustic/ASR
    # frame count (EXPORT-ROADMAP.md R1, axes.py's N_ENC_FRAMES), not a token count. f0_curve/n_curve/
    # noise_in/wsum are genuinely independent LEAF inputs whose real length is a fixed multiple of
    # that same frame count (2x/600x/600x+20 respectively) -- not derivable from the graph (see
    # LoomGGUFExporter's `declared_axes` docstring in exporter.py). Declared by input name and axis
    # position; the exporter reads each input's own real traced symbol itself.
    return ExportPhase(
        name="decoder_vocoder", wrapper=wrapper, dummy_inputs=dummy_args, mil_inputs=mil_inputs,
        root_axis="n_enc_frames",
        declared_axes={
            "f0_curve": {1: "2*n_enc_frames"},
            "n_curve": {1: "2*n_enc_frames"},
            "noise_in": {1: "600*n_enc_frames"},
            "wsum": {0: "600*n_enc_frames+20"},
        },
    )


@dataclass(kw_only=True)
class TTSKokoroExportConfig(BaseMultiPhaseModelExportConfig):
    """Kokoro's own two-phase split (albert_bert_encoder/decoder_vocoder) -- see module docstring.
    `model_dir` is the directory containing both the real checkpoint (`kokoro-v1_0.pth`) and its
    `config.json` (the real Kokoro-82M HF repo layout)."""

    model_dir: str
    driver_script_path: Path = Path(__file__).resolve().parent.parent / "convert_kokoro" / "kokoro_driver_mil.lua"

    def phases(self) -> List[ExportPhase]:
        ckpt_path = Path(self.model_dir) / "kokoro-v1_0.pth"
        config_path = Path(self.model_dir) / "config.json"
        print(f"Loading checkpoint {ckpt_path}...")
        cfg = json.load(open(config_path))
        model = KModel(repo_id="hexgrad/Kokoro-82M", config=cfg, model=str(ckpt_path), disable_complex=True)
        model.eval()

        print("Tracing albert_bert_encoder phase...")
        albert_phase = build_albert_bert_encoder_phase(model.bert, model.bert_encoder)

        print("Tracing decoder_vocoder phase...")
        decoder_vocoder_phase = build_decoder_vocoder_phase(model.decoder)

        return [albert_phase, decoder_vocoder_phase]


# Kokoro's `config.json` carries no `model_type`-style single field, but its own key set is a real
# signature: the two nested sub-configs (`istftnet` -- the decoder/vocoder this family exports as its
# second phase; `plbert` -- the ALBERT encoder it exports as its first) plus the three top-level
# hyperparameters that shape both. Checked as a subset, so a checkpoint adding keys still matches.
_KOKORO_CONFIG_KEYS = frozenset({"istftnet", "plbert", "n_token", "style_dim", "vocab"})


def _is_kokoro(path: Path) -> bool:
    """Real structural check (BACKLOG.md P4.0.1): the model directory `TTSKokoroExportConfig.phases()`
    itself requires -- `kokoro-v1_0.pth` beside a `config.json` carrying Kokoro's own key signature.

    Both halves are load-bearing. The checkpoint name alone is weak, and the config alone genuinely
    cannot be trusted: StyleTTS2 loads this *same* `config.json` (`TTSStyleTTS2ExportConfig.
    kokoro_config_path` -- the two models share the iSTFTNet decoder/vocoder architecture, so they share
    its declaration too), so a directory recognized on the config alone could just as well be a
    StyleTTS2 export environment. Requiring the checkpoint file this config's `build_config` will
    actually open is what makes the answer unambiguous."""
    if not path.is_dir() or not (path / "kokoro-v1_0.pth").is_file():
        return False
    config = read_json(path / "config.json")
    return config is not None and _KOKORO_CONFIG_KEYS.issubset(config)


def _build_kokoro(path: Path, output_path: str) -> TTSKokoroExportConfig:
    return TTSKokoroExportConfig(architecture="loom-kokoro-mil", output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="tts-multi-phase",
        config_class=BaseMultiPhaseModelExportConfig,
        recognizers=[ModelRecognizer(name="kokoro", detect=_is_kokoro, build_config=_build_kokoro)],
    ))
