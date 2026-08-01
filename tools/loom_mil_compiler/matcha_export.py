"""Export the real Matcha-TTS checkpoint (`matcha_ljspeech.ckpt` + HiFi-GAN v1 `generator_v1`) through
`TTSMatchaExportConfig` (BACKLOG.md P3.3, migrated from `export_matcha_mil.py`), tracing the REAL
`matcha.models.components.{text_encoder,decoder}`/`matcha.hifigan.models` submodules directly -- not the
hand-built bespoke topology `tools/convert_matcha/convert_matcha_*.py` constructs op-by-op via its own
custom `TopologyBuilder` DSL.

Four phases, same "GraphTopology supports exactly one declared output" split as the bespoke scripts:
  - encoder_mu:    TextEncoder up through proj_m -> mu, ne=[T,n_feats] (T-fast, matches the real
                   module's own native (1,C,T) torch layout directly -- see module docstring below for
                   why this is a DIFFERENT convention than the bespoke topology's own mu output).
  - encoder_logw:  TextEncoder (re-traced) up through proj_w -> logw, ne=[T].
  - decoder:       Decoder U-Net (CFM estimator), one Euler velocity evaluation
                   `dphi_dt = estimator(z, mask=ones, mu, t, spks=None, cond=None)`.
  - vocoder:       HiFi-GAN v1 Generator, mel -> waveform.

Layout note (IMPORTANT, differs from the bespoke conversion): the bespoke TextEncoder topology
(convert_matcha_text_encoder.py) emits `mu` C-fast (ne=[n_feats,T], "rows_flat" convention, T rows of
n_feats contiguous floats each) because it builds `mu` via an explicit MUL_MAT against a [C,T]-convention
`x`. Tracing the REAL `TextEncoder.forward` instead preserves its own native torch tensor layout
(1, n_feats, T) unchanged -- which is T-FAST (ne=[T,n_feats], matching the Decoder's/HiFi-GAN's own
[T,C] CONV_1D-friendly convention, exactly like `z`/`dphi_dt`/`mel` already are). This is actually a
SIMPLER pipeline than the bespoke one (no channel-first/T-first bridging transpose needed feeding the
Decoder), at the cost of needing the duration-expansion (per-token repeat) step to operate natively in
T-fast layout -- see `matcha_driver_mil.lua`'s own comment for the resulting direct (no
`loom.expand_by_duration` reuse) repeat loop. Deliberately did NOT add a `.transpose()` in the wrapper to
match the bespoke convention instead: a bare transpose as a topology's own final declared output is a
live, non-contiguous GGML PERMUTE view once compiled (`ggml_backend_tensor_get` does a raw contiguous
byte copy, silently ignoring it) -- same danger already documented in `vits_export.py`'s own
`StatsWrapper`, sidestepped there (and here) by keeping the trace's natural, untransposed output.

Trace-friendliness patches needed (same category of fix as VITS's/StyleTTS2's own):
  - `TextEncoder`'s real `sequence_mask(x_lengths, T)` (dynamic `T`, "torch.full doesn't accept a
    non-constant size" issue) replaced by a direct all-ones mask, exact same fix/reasoning as
    `vits_export.py`'s own `_text_encoder_forward` -- valid because this whole project's "single,
    unpadded utterance" convention means x_lengths == T always.
  - Real `huggingface-hub`/`transformers`/`diffusers` version-pin import chain workaround (same
    self-contained stub pattern as `kokoro_export.py`/`styletts2_export.py`), needed because
    `matcha.models.components.{decoder,transformer}` import `diffusers.models.attention_processor.Attention`
    transitively through `diffusers.models.lora` -> `transformers` -> its own version gate, AND
    `diffusers.utils.dynamic_modules_utils` itself imports a `huggingface_hub.cached_download` symbol
    removed in the huggingface_hub version installed here (never actually CALLED by anything this module
    exercises -- stubbed to a no-op).
  - Real `matcha.utils.__init__` pulls in a chain of training-only tooling (hydra/rootutils/gdown/rich)
    that `matcha.models.components.text_encoder` transitively imports (`import matcha.utils as utils`,
    just for `utils.get_pylogger`) -- bypassed with a lightweight stand-in module registered directly in
    `sys.modules["matcha.utils"]` before the real `matcha` package is ever imported, so the real
    `matcha.models.components.*` modules pick up the stand-in instead of the heavy real one. (`gdown`/
    `rootutils` themselves ARE installed in this venv -- see BACKLOG.md -- but the stand-in avoids the
    dependency entirely rather than relying on it.)

Usage:
  loom-export /path/to/matcha/ckpt/dir -o matcha_mil.gguf --task text-to-speech --model matcha
"""
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch

from .checkpoint_probe import probe_torch_checkpoint
from .flow_matching_export import FlowMatchingSpec
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase, TTSFlowMatchingModelExportConfig
from .patcher import ModelPatcher
from .spec_protocol import Unchecked


class MatchaModelPatcher(ModelPatcher):
    """Import-order stubs needed before Matcha's real package can be imported at all -- BACKLOG.md P3.3's
    `ModelPatcher` hook. Self-contained (doesn't modify either installed package), same timing as the
    original script's own top-of-file side effects."""

    @staticmethod
    def prepare_environment() -> None:
        stub_versions = types.ModuleType("transformers.utils.versions")
        stub_versions.require_version = lambda *a, **k: None
        stub_versions.require_version_core = lambda *a, **k: None
        sys.modules["transformers.utils.versions"] = stub_versions

        import huggingface_hub
        huggingface_hub.cached_download = lambda *a, **k: None

        # matcha.utils stand-in (bypasses the real package's training-tooling-heavy __init__ chain).
        stub_matcha_utils = types.ModuleType("matcha.utils")
        stub_matcha_utils.get_pylogger = lambda name=__name__: logging.getLogger(name)
        # Give the stand-in a real `__path__` pointing at the actual package directory so plain submodule
        # imports (`matcha.utils.model`, which only needs numpy/torch, no heavy training tooling) still
        # resolve normally through the standard import machinery instead of also needing to be stubbed.
        stub_matcha_utils.__path__ = ["/home/flavio/Dev/Matcha-TTS/matcha/utils"]
        sys.modules["matcha.utils"] = stub_matcha_utils


MatchaModelPatcher.prepare_environment()

sys.path.insert(0, "/home/flavio/Dev/Matcha-TTS")

import coremltools as ct  # noqa: E402

from . import group_norm_op  # noqa: E402,F401  patches nn.GroupNorm.forward globally

from matcha.models.components.text_encoder import TextEncoder, RotaryPositionalEmbeddings  # noqa: E402
from matcha.models.components.decoder import Decoder  # noqa: E402
from matcha.hifigan.config import v1 as HIFIGAN_V1  # noqa: E402
from matcha.hifigan.models import Generator  # noqa: E402


def _rope_build_cache_traceable(self, seq_len, device):
    """Real `RotaryPositionalEmbeddings._build_cache`'s formula, unconditionally rebuilt every call
    (dropping the real code's own "already built for a long-enough sequence, skip" mutable-state fast
    path -- mathematically identical, just recomputed; the fast path exists only as a speed optimization
    across REPEATED calls with a shared module instance, but that same statefulness makes
    `torch.jit.trace`'s own internal double-invocation sanity check (`check_trace`) see two DIFFERENT
    graphs for the same module and fail with 'Graphs differed across invocations!'). Takes `seq_len`
    directly (a plain int-valued scalar tensor) rather than deriving it internally via `x.shape[0]` on
    the real code's own already-REARRANGED ("t b h d") tensor -- see `_rope_forward_traceable`'s own
    docstring for why: this exporter's shape-inference has a hardcoded (and, for every OTHER model,
    correct) "torch axis 0 of a rank>=2 tensor is always the batch axis, therefore always static 1"
    shortcut (`_try_derive_gather_shape_value`, exporter.py) that fires WRONGLY here once axis 0 has been
    rearranged to mean the sequence length instead of batch, silently baking a `RANGE_1D` `end=1`
    (confirmed directly: a standalone traced-MIL dump showed exactly `end='1'`) instead of the real
    dynamic length -- corrupting every position beyond the first.
    """
    theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(device)
    seq_idx = torch.arange(seq_len, device=device).float().to(device)
    idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
    idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
    self.cos_cached = idx_theta2.cos()[:, None, None, :]
    self.sin_cached = idx_theta2.sin()[:, None, None, :]


def _rope_forward_traceable(self, x):
    """Real `RotaryPositionalEmbeddings.forward`, with two changes: (1) `seq_len` for `_build_cache` is
    read from the ORIGINAL (pre-rearrange) `x`'s own `t` axis -- torch axis 2 of the real "b h t d"
    input -- instead of the rearranged "t b h d" tensor's axis 0, sidestepping the "axis 0 = batch"
    shortcut described in `_rope_build_cache_traceable`'s own docstring (axis 2 hits this exporter's
    general dynamic-dim backward walk instead, the same one correctly used everywhere else); (2) the
    real code's own `self.cos_cached[:x.shape[0]]`/`self.sin_cached[:x.shape[0]]` slices are dropped
    (bare `self.cos_cached`/`self.sin_cached` used directly) -- always the identity slice given
    `_build_cache` now always rebuilds to exactly `seq_len` with no "already have a longer cache" fast
    path, and a real dynamic-length slice of a graph-internal tensor is exactly the kind of thing this
    whole session's other fixes (fill/ones_like, GroupNorm's reshape-back, RESHAPE targets) kept finding
    subtly wrong length expressions for.
    """
    from einops import rearrange

    seq_len = x.shape[2]
    x = rearrange(x, "b h t d -> t b h d")
    self._build_cache(seq_len, x.device)
    x_rope, x_pass = x[..., : self.d], x[..., self.d :]
    neg_half_x = self._neg_half(x_rope)
    x_rope = (x_rope * self.cos_cached) + (neg_half_x * self.sin_cached)
    return rearrange(torch.cat((x_rope, x_pass), dim=-1), "t b h d -> b h t d")


RotaryPositionalEmbeddings._build_cache = _rope_build_cache_traceable
RotaryPositionalEmbeddings.forward = _rope_forward_traceable


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


# ---- Trace-friendly TextEncoder body (real submodule calls, sequence_mask -> ones) ----
def _text_encoder_forward_traceable(enc, tokens):
    """Real `TextEncoder.forward`'s body up to (and including) `self.encoder(...)`, with
    `sequence_mask(x_lengths, T)` replaced by a direct all-ones mask derived arithmetically from `x`
    itself (`x[:, :1, :] * 0.0 + 1.0`, not `torch.ones_like` -- see `_decoder_forward_traceable`'s own
    docstring for why: a separate `ones_like`/`fill`-style op needs its OWN shape re-inferred by the
    exporter, which was found to go wrong when the same wrapper calls it more than once at different
    dynamic lengths; deriving the mask via elementwise arithmetic on `x` sidesteps that by construction)
    -- see module docstring for why replacing `sequence_mask` at all is safe (this project's standing
    "single, unpadded utterance" convention) and necessary (dynamic-T `torch.full`/`torch.arange`-vs-
    length comparison doesn't convert as a real op with a *traced* dynamic T).
    """
    import math
    x = enc.emb(tokens) * math.sqrt(enc.n_channels)
    x = torch.transpose(x, 1, -1)
    x_mask = x[:, :1, :] * 0.0 + 1.0
    x = enc.prenet(x, x_mask)
    x = enc.encoder(x, x_mask)
    return x, x_mask


class MuWrapper(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, tokens):
        x, x_mask = _text_encoder_forward_traceable(self.enc, tokens)
        mu = self.enc.proj_m(x) * x_mask  # (1, n_feats, T)
        return mu.squeeze(0)  # (n_feats, T) -> ggml ne=[T,n_feats], T-fast (see module docstring)


class LogwWrapper(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, tokens):
        x, x_mask = _text_encoder_forward_traceable(self.enc, tokens)
        x_dp = torch.detach(x)
        logw = self.enc.proj_w(x_dp, x_mask)  # (1, 1, T)
        return logw.reshape(-1)  # (T,) -> ggml ne=[T]


def _block1d_forward_nomask(block1d, x):
    """Real `Block1D.forward`'s body (`self.block(x*mask) * mask`) with the mask multiplies DROPPED
    entirely, not replaced by a traced all-ones tensor -- see `_decoder_forward_traceable`'s own
    docstring for why even constructing an all-ones mask tensor (whether via `torch.ones_like` or plain
    arithmetic on `x`) turned out to be trace-hostile here: every mask multiply in this whole U-Net is a
    real, exact no-op (mask is always all-ones for this project's "single, unpadded utterance"
    convention, same simplification `tools/convert_matcha/convert_matcha_decoder.py`'s own bespoke
    topology already makes -- see that module's docstring), so the correct trace-friendly fix is simply
    never constructing the mask tensor at all, calling `block1d.block` (the real
    `Sequential(Conv1d,GroupNorm,Mish)`) directly on `x` unmultiplied.
    """
    return block1d.block(x)


def _resnet_block1d_forward_nomask(resnet, x, time_emb):
    """Real `ResnetBlock1D.forward`'s body, mask-free (see `_block1d_forward_nomask`) -- `res_conv(x)`
    instead of the real `res_conv(x*mask)` for the same reason.
    """
    h = _block1d_forward_nomask(resnet.block1, x)
    h = h + resnet.mlp(time_emb).unsqueeze(-1)
    h = _block1d_forward_nomask(resnet.block2, h)
    return h + resnet.res_conv(x)


def _decoder_forward_traceable(dec, x, mu, t):
    """Real `Decoder.forward`'s body (unmodified real submodule calls otherwise: transformer_blocks/
    downsample/upsample/final_proj), with the real `mask`/`attention_mask` machinery removed rather than
    replaced by a traced all-ones tensor. `attention_mask=None` sidesteps
    `diffusers.attention_processor.Attention.prepare_attention_mask`'s own dynamic-shape-dependent
    reshape/repeat chain entirely -- mathematically identical (an all-ones additive-bias mask is a no-op
    either way). The resnet/Block1D mask multiplies are dropped via `_resnet_block1d_forward_nomask`/
    `_block1d_forward_nomask` above rather than traced as `x * ones_like(...)`: BOTH an explicit
    `torch.ones_like(x[:, :1, :])` (a `fill` MIL op) AND the seemingly-safer arithmetic
    `x[:, :1, :] * 0.0 + 1.0` (a slice/VIEW + MUL + ADD) were independently confirmed, via real
    `ggml_mul`/`ggml_reshape` shape-mismatch crashes, to get a STALE dynamic-length expression from this
    exporter's per-symbol shape-inference cache one call site bleeding into another (this function calls
    the mask-construction code once per down/mid/up-block stage, each at a genuinely different T after
    downsampling) -- never actually constructing a mask tensor at all sidesteps the whole question rather
    than depending on any particular construction tracing correctly.
    """
    from einops import rearrange

    t_emb = dec.time_embeddings(t)
    t_emb = dec.time_mlp(t_emb)

    x = torch.cat([x, mu], dim=1)  # spks is always None in this project (n_spks=1, no speaker embedding)

    hiddens = []
    for resnet, transformer_blocks, downsample in dec.down_blocks:
        x = _resnet_block1d_forward_nomask(resnet, x, t_emb)
        x = rearrange(x, "b c t -> b t c")
        for transformer_block in transformer_blocks:
            x = transformer_block(hidden_states=x, attention_mask=None, timestep=t_emb)
        x = rearrange(x, "b t c -> b c t")
        hiddens.append(x)
        x = downsample(x)

    for resnet, transformer_blocks in dec.mid_blocks:
        x = _resnet_block1d_forward_nomask(resnet, x, t_emb)
        x = rearrange(x, "b c t -> b t c")
        for transformer_block in transformer_blocks:
            x = transformer_block(hidden_states=x, attention_mask=None, timestep=t_emb)
        x = rearrange(x, "b t c -> b c t")

    for resnet, transformer_blocks, upsample in dec.up_blocks:
        x = torch.cat([x, hiddens.pop()], dim=1)
        x = _resnet_block1d_forward_nomask(resnet, x, t_emb)
        x = rearrange(x, "b c t -> b t c")
        for transformer_block in transformer_blocks:
            x = transformer_block(hidden_states=x, attention_mask=None, timestep=t_emb)
        x = rearrange(x, "b t c -> b c t")
        x = upsample(x)

    x = _block1d_forward_nomask(dec.final_block, x)
    return dec.final_proj(x)


class DecoderWrapper(torch.nn.Module):
    """See `_decoder_forward_traceable` for the trace-friendly mask handling. `Downsample1D`/
    `Upsample1D` use real `nn.Conv1d`/`nn.ConvTranspose1d` with static kernel/stride/padding chosen (by
    the real architecture) to exactly round-trip a length back to the original T, so -- unlike the
    bespoke ggml `CONV_TRANSPOSE_1D` primitive, which has no padding parameter and needed a manual
    VIEW-crop workaround -- no cropping logic is needed here at all: real `nn.ConvTranspose1d`'s own
    `padding` argument already produces the exact right length directly.
    """
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, z, mu, t):
        dphi_dt = _decoder_forward_traceable(self.dec, z, mu, t)
        return dphi_dt.squeeze(0)  # (n_feats, T) -> ggml ne=[T,n_feats], T-fast


class VocoderWrapper(torch.nn.Module):
    """Real `matcha.hifigan.models.Generator.forward`, unmodified -- `remove_weight_norm()` is called
    once on the loaded module BEFORE tracing (see `load_vocoder`) so the trace bakes plain folded conv
    weights directly, matching `add_wn_conv`'s own folding in the bespoke conversion, rather than tracing
    the weight_norm `g*v/||v||` recomputation into the graph on every forward call.
    """
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def forward(self, mel):
        wav = self.gen(mel)
        return wav.reshape(-1)  # (T*256,)


def load_text_encoder(matcha_ckpt_path: str):
    ckpt = torch.load(matcha_ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    sd = ckpt["state_dict"]

    encoder_params = AttrDict(dict(hp["encoder"]["encoder_params"]))
    dp_params = AttrDict(dict(hp["encoder"]["duration_predictor_params"]))
    enc = TextEncoder(
        hp["encoder"]["encoder_type"], encoder_params, dp_params,
        hp["n_vocab"], hp["n_spks"], hp["spk_emb_dim"],
    )
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = enc.load_state_dict(enc_sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    enc.eval()
    return enc, hp, sd


def load_decoder(hp, sd):
    n_feats = hp["n_feats"]
    dec = Decoder(in_channels=2 * n_feats, out_channels=n_feats, **hp["decoder"])
    dec_sd = {k[len("decoder.estimator."):]: v for k, v in sd.items() if k.startswith("decoder.estimator.")}
    missing, unexpected = dec.load_state_dict(dec_sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    dec.eval()
    return dec


def load_vocoder(hifigan_ckpt_path: str):
    ckpt = torch.load(hifigan_ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["generator"]
    h = AttrDict(dict(HIFIGAN_V1))
    gen = Generator(h)
    gen.load_state_dict(sd, strict=True)
    gen.eval()
    gen.remove_weight_norm()
    return gen


@dataclass(kw_only=True)
class TTSMatchaExportConfig(TTSFlowMatchingModelExportConfig):
    """Matcha's own four-phase split (encoder_mu/encoder_logw/decoder/vocoder) plus the Euler CFM
    sampler over `decoder` -- see module docstring. `model_dir` is the directory containing both
    checkpoints (`matcha_ljspeech.ckpt` and the HiFi-GAN `generator_v1`, the real Matcha-TTS release
    layout)."""

    model_dir: str

    __unchecked__ = {
        "model_dir": Unchecked(
            "the directory holding matcha_ljspeech.ckpt and generator_v1. path to the real checkpoint(s). The recognizer's own detect() already established the structure this config depends on -- it probes the checkpoint's pickle opcodes without unpickling (checkpoint_probe) rather than trusting the filename -- and phases() raises on anything it cannot load. A 'this path exists' link would check the weaker property while reading as if it checked the stronger one."
        ),
    }
    driver_script_path: Path = Path(__file__).resolve().parent.parent / "convert_matcha" / "matcha_driver_mil.lua"

    def phases(self) -> List[ExportPhase]:
        matcha_ckpt_path = str(Path(self.model_dir) / "matcha_ljspeech.ckpt")
        hifigan_ckpt_path = str(Path(self.model_dir) / "generator_v1")

        print(f"Loading Matcha checkpoint {matcha_ckpt_path}...")
        enc, hp, sd = load_text_encoder(matcha_ckpt_path)

        dummy_T = 12
        dummy_tokens = torch.randint(1, hp["n_vocab"], (1, dummy_T), dtype=torch.long)
        seq_dim = ct.RangeDim(4, 512)

        print("Tracing TextEncoder (mu, logw)...")
        mu_phase = ExportPhase(
            name="encoder_mu", wrapper=MuWrapper(enc).eval(), dummy_inputs=(dummy_tokens,),
            mil_inputs=[ct.TensorType(name="tokens", shape=(1, seq_dim), dtype=np.int32)],
        )
        logw_phase = ExportPhase(
            name="encoder_logw", wrapper=LogwWrapper(enc).eval(), dummy_inputs=(dummy_tokens,),
            mil_inputs=[ct.TensorType(name="tokens", shape=(1, seq_dim), dtype=np.int32)],
        )

        print("Tracing Decoder U-Net...")
        n_feats = hp["n_feats"]
        dec = load_decoder(hp, sd)
        dummy_dec_T = 8  # multiple of 4, matches reference_forward_matcha_decoder.py's own fixture
        dummy_z = torch.randn(1, n_feats, dummy_dec_T)
        dummy_mu = torch.randn(1, n_feats, dummy_dec_T)
        dummy_t = torch.tensor([0.37])
        dec_seq_dim = ct.RangeDim(4, 2048)
        decoder_phase = ExportPhase(
            name="decoder", wrapper=DecoderWrapper(dec).eval(), dummy_inputs=(dummy_z, dummy_mu, dummy_t),
            mil_inputs=[
                ct.TensorType(name="z", shape=(1, n_feats, dec_seq_dim), dtype=np.float32),
                ct.TensorType(name="mu", shape=(1, n_feats, dec_seq_dim), dtype=np.float32),
                ct.TensorType(name="t", shape=(1,), dtype=np.float32),
            ],
        )

        print("Tracing HiFi-GAN vocoder...")
        gen = load_vocoder(hifigan_ckpt_path)
        dummy_voc_T = 4
        dummy_mel = torch.randn(1, n_feats, dummy_voc_T)
        voc_seq_dim = ct.RangeDim(1, 4096)
        vocoder_phase = ExportPhase(
            name="vocoder", wrapper=VocoderWrapper(gen).eval(), dummy_inputs=(dummy_mel,),
            mil_inputs=[ct.TensorType(name="mel", shape=(1, n_feats, voc_seq_dim), dtype=np.float32)],
        )

        return [mu_phase, logw_phase, decoder_phase, vocoder_phase]

    def samplers(self) -> List[FlowMatchingSpec]:
        # The Euler CFM sampling loop is the shared "N-step refinement over loop-carried state" family
        # (EXPORT-IMPROVEMENT.md item 4), so the driver declares it rather than hand-writing it -- and
        # gets the spec cross-checked against "decoder"'s real declared inputs at export time.
        return [FlowMatchingSpec(
            func_name="sample_decoder",
            estimator="decoder",
            carried_input="z",
            time_input="t",
            fixed_inputs=["mu"],
            note="Deterministic Euler CFM sampling over the Matcha-TTS Decoder U-Net vector-field\n"
                 "estimator: z <- z + v(z, mu, t) * dt, uniform dt = 1/n_steps.",
        )]


def _is_matcha(path: Path) -> bool:
    """Real structural check (BACKLOG.md P4.0.1): the model directory `TTSMatchaExportConfig.phases()`
    itself requires -- both checkpoints present, and the Matcha one really being Matcha's own Lightning
    checkpoint.

    **A Lightning signature alone is NOT a discriminator here**, which is the finding that shaped this
    check: `pytorch-lightning_version` + `state_dict` is exactly what piper-VITS's own `.ckpt` declares
    too (confirmed by probing both real checkpoints -- Matcha 2.0.8, VITS 1.9.5). The two are told apart
    by their first state-dict key instead: Matcha's is `mel_mean` (its stored mel normalization
    statistics, which VITS has no equivalent of), VITS's are `model_g.`-prefixed. `vits_export._is_vits`
    keys on the other side of that same pair; the two checks are meant to be read together.

    The filename is the LJSpeech release's own (`phases()` hardcodes it, so detecting anything looser
    would recognize directories this config then fails to export); a differently-named Matcha voice
    needs both this and `phases()` widened together."""
    if not path.is_dir():
        return False
    ckpt = path / "matcha_ljspeech.ckpt"
    if not ckpt.is_file() or not (path / "generator_v1").is_file():
        return False
    probe = probe_torch_checkpoint(ckpt)
    if probe is None:
        return False
    return {"pytorch-lightning_version", "state_dict", "mel_mean"}.issubset(probe.strings)


def _build_matcha(path: Path, output_path: str) -> TTSMatchaExportConfig:
    return TTSMatchaExportConfig(architecture="matcha_mil", output_path=output_path, model_dir=str(path))


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech",
        config_class=TTSFlowMatchingModelExportConfig,
        recognizers=[ModelRecognizer(name="matcha", detect=_is_matcha, build_config=_build_matcha)],
    ))
