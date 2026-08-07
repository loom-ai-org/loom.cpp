"""Exports StyleTTS2's (yl4579/StyleTTS2-LJSpeech) MIL-traceable phases into ONE combined
`styletts2_mil.gguf` (BACKLOG.md P3.3, migrated from `export_styletts2_mil.py`) alongside the embedded
`styletts2_driver/` orchestration script:
  - "albert": CustomAlbert (a real transformers.AlbertModel) alone -- input_ids -> raw bert_dur, (T,768)
    time-major. UNLIKE `kokoro_export.py`'s own combined "albert_bert_encoder" (which fuses in the
    bert_encoder Linear too), this phase stops at bert_dur because StyleTTS2's diffusion sampler needs
    the RAW (unprojected) bert_dur as its own conditioning input (real source: `sampler(noise,
    embedding=bert_dur[0].unsqueeze(0), ...)` in Demo/Inference_LJSpeech.ipynb) -- bert_encoder's
    Linear(768,512) projection is a SEPARATE, later consumer of this same tensor. bert_encoder itself
    stays on the existing bespoke kokoro_bert_encoder.gguf topology (tools/convert_kokoro/
    convert_kokoro_bert_encoder.py) -- a single Linear is zero-risk to hand-build and there's nothing a
    trace would improve about it, so it's out of scope here.
  - "decoder_vocoder": Decoder.encode/decode + SineGen + STFT + Generator -- DIRECT reuse of
    `kokoro_export.py`'s own `DecoderVocoderWrapper`/`build_decoder_vocoder_phase`/`VerifiedSTFT`
    (including its trace-friendly AdainResBlk1d/SineGen/SourceModuleHnNSF/Generator/Decoder monkeypatches,
    applied globally at `import .kokoro_export` time) -- the REAL payoff of "using Kokoro's lessons":
    Kokoro's own istftnet.py Decoder/Generator/SineGen classes ARE StyleTTS2's own classes (Kokoro is a
    fork of this exact architecture), so tracing them with StyleTTS2's own checkpoint weights instead of
    Kokoro's needs zero new code, only a different state dict loaded into the same real nn.Module classes.
  - "diffusion": StyleTTS2's own genuinely new piece (no Kokoro equivalent) -- the plain `Transformer1d`
    denoiser network (real source: styletts2 repo's Modules/diffusion/modules.py) that `build_model`
    substitutes in place of AudioDiffusionConditional's own default U-Net (confirmed directly against the
    real checkpoint's state dict, which has NO U-Net-shaped keys -- see convert_styletts2_diffusion.py's
    own module docstring, which this phase supersedes with a MIL trace of the REAL Transformer1d.run()
    method instead of a hand-derived topology). embedding_scale=1.0 only (the real demo's own
    basic-synthesis default, confirmed via Demo/Inference_LJSpeech.ipynb's own `inference()` --
    `embedding_scale != 1.0` classifier-free-guidance branch traces this SAME network TWICE and is out of
    scope, matching convert_styletts2_diffusion.py's own identical scoping decision).

Everything else (DurationEncoder/predictor.lstm/duration_proj, F0Ntrain, TextEncoder's BiLSTM) stays on
the existing bespoke, hand-built LSTM-bound topologies from convert_styletts2_reused.py -- ggml has no
native LSTM op, same deliberate scoping exclusion Kokoro's own MIL export already established.

StyleTTS2's diffusion sampler is ADPM2 over a Karras sigma schedule -- two network evaluations per step,
per-step noise injection, and real preconditioning math around the call -- so it is NOT a
`TTSFlowMatchingModelExportConfig` (`flow_matching_export.py`'s own docstring documents why this can't be
generalized the way Matcha's/Supertonic's plain Euler CFM integration can). This class stays a plain
`BaseMultiPhaseModelExportConfig` with the ADPM2 loop hand-written in `styletts2_driver/` and only
`EstimatorSpec`-checked via `estimators()`: the sampler's per-step `run_subgraph` call still gets the same
export-time validation against the real traced "diffusion" topology, generating no codegen.

Numerically verified against real-checkpoint references (see `tools/convert_styletts2/
reference_forward_styletts2_{albert_mil,diffusion}.py` and `kokoro_export.py`'s own already-verified
decoder_vocoder reference reused as-is): see `test_e2e_styletts2_mil_*.cpp` for the actual tolerances.

Usage:
  loom-export /path/to/styletts2.pth -o styletts2_mil.gguf --task text-to-speech --model styletts2
"""
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import coremltools as ct

from .checkpoint_probe import probe_torch_checkpoint
from .flow_matching_export import EstimatorSpec
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .patcher import ModelPatcher
from .spec_protocol import Unchecked


class StyleTTS2ModelPatcher(ModelPatcher):
    """Import-order stub needed before `transformers.AlbertModel` (via `kokoro.modules.CustomAlbert`)
    can be imported at all -- same self-contained pattern as `kokoro_export.py`'s own."""

    @staticmethod
    def prepare_environment() -> None:
        stub = types.ModuleType("transformers.utils.versions")
        stub.require_version = lambda *a, **k: None
        stub.require_version_core = lambda *a, **k: None
        sys.modules["transformers.utils.versions"] = stub


StyleTTS2ModelPatcher.prepare_environment()

from transformers import AlbertConfig  # noqa: E402
from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder  # noqa: E402
from kokoro.istftnet import Decoder  # noqa: E402

# `kokoro_export`'s own import applies its trace-friendly monkeypatches (AdainResBlk1d/SineGen/
# SourceModuleHnNSF/Generator/Decoder) globally as an import side effect -- needed before tracing
# StyleTTS2's OWN Decoder instance below, since it's the exact same class. `build_decoder_vocoder_phase`
# is reused directly rather than re-implemented (BACKLOG.md P3.3's "real cross-model dependency" note).
from . import kokoro_export  # noqa: E402

sys.path.insert(0, "/home/flavio/Dev/styletts2")  # read-only reference clone, see memory: readonly repos
from Modules.diffusion.modules import Transformer1d, AttentionBase  # noqa: E402
from einops import rearrange  # noqa: E402
from einops_exts import rearrange_many  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "convert_styletts2"))
from convert_styletts2_diffusion import HP as DIFF_HP  # noqa: E402


def _attention_base_forward_traceable(self, q, k, v):
    """Real forward (Modules/diffusion/modules.py's AttentionBase.forward) is composed of two
    `torch.einsum` calls ('... n d, ... m d -> ... n m' / '... n m, ... m d -> ... n d', both rank-4
    batched over (b,h)) -- coremltools' own GENERIC einsum solver has a real bug for this exact equation
    shape: its diagonal-einsum pre-pass (`solve_diagonal_einsum_one_step`) builds a `perm` sized for the
    wrong rank (5 instead of the real rank-4 operand), raising 'perm should have the same length as
    rank(x): 5 != 4' at MIL-build time (confirmed empirically -- a genuine coremltools bug, not a shape
    error in this code). Replaced with the algebraically identical batched-matmul formulation
    (`q @ k.transpose(-2,-1)` / `attn @ v`) -- MIL's own `matmul` op handles the same (b,h,n,d) batch-dim
    broadcast directly, sidestepping the einsum solver entirely (already exercised extensively elsewhere
    in this project's exports, e.g. every SDPA-based attention).
    """
    q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)
    sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale
    attn = sim.softmax(dim=-1)
    out = torch.matmul(attn, v)
    out = rearrange(out, "b h n d -> b n (h d)")
    return self.to_out(out)


AttentionBase.forward = _attention_base_forward_traceable


def _transformer1d_run_traceable(self, x, time, embedding, features):
    """Real `Transformer1d.run()` (Modules/diffusion/modules.py) broadcasts the single noisy-style
    "pseudo-token" `x` (and the per-batch `mapping` vector) out to the real, dynamic token count via
    `x.expand(-1, embedding.size(1), -1)` -- traces to a MIL `tile` op whose own `reps` tensor is a
    runtime shape query (`embedding.size(1)`), not a compile-time constant. This exporter's generic
    tile->REPEAT translation has no reliable way to tell "this axis's target IS the genuine n_tokens
    quantity" apart from "some other static architectural axis" (e.g. num_heads) from local shape
    information alone -- confirmed the hard way: a general fix attempted directly in exporter.py's shared
    tile-shape-inference heuristic got this case right but regressed CustomAlbert's own attention-mask
    head-broadcast (identical "static-1 axis, unreadable reps" shape there, but needing the OPPOSITE
    resolution, since that axis is num_heads, not n_tokens) -- reverted in favor of this narrower,
    single-model fix instead.

    Replaced with a batched-matmul outer product (`ones_like(embedding[...,:1]) @ x`) instead of an
    elementwise-broadcast MUL: a (1,T,1) ones tensor times a (1,1,C) tensor via `torch.matmul` gives
    (1,T,C) directly (a genuine matrix multiply contracting over the size-1 axis, not an elementwise
    broadcast at all) -- tried the elementwise-MUL version first, but ggml's own MUL primitive only
    supports a SINGLE-direction broadcast (one operand's shape a per-axis divisor of the other's), not
    NumPy-style MUTUAL broadcasting (256-vs-1 on one axis, 1-vs-T on another, simultaneously, which is
    exactly what this is) -- confirmed empirically ('MUL: incompatible shapes a=[256,1,1,1]
    b=[1,6,1,1]'). Matmul's own dynamic-shape inference in this exporter is already well-established
    (every attention QK^T/AV op works this same way), so this sidesteps BOTH the tile ambiguity above AND
    ggml's narrower MUL-broadcast rules. Otherwise byte-identical to the real method (features=None
    throughout, since StyleTTS2's `multispeaker: false` means `get_mapping` never uses `features` -- see
    DiffusionNetWrapper's own docstring).
    """
    mapping = self.get_mapping(time, features)
    ones_col = torch.ones_like(embedding[:, :, :1])  # (1,T,1) -- pure shape-broadcast helper, no real data
    x_rep = torch.matmul(ones_col, x)  # (1,T,1) @ (1,1,C) -> (1,T,C)
    x_full = torch.cat([x_rep, embedding], axis=-1)
    mapping_full = torch.matmul(ones_col, mapping.unsqueeze(1))  # (1,T,1) @ (1,1,features) -> (1,T,features)
    for block in self.blocks:
        x_full = x_full + mapping_full
        x_full = block(x_full)
    # Real code: `x.mean(axis=1)` (reduce over T, torch axis 1). Coremltools' `reduce_mean` translation
    # maps straight to ggml's own MEAN primitive, which (like REDUCE_SUM elsewhere in this project's
    # hand-built topologies) always reduces ne[0] -- for a plain (1,T,features) tensor that's the
    # CHANNEL axis, not T (confirmed empirically: 'RESHAPE ... has 1024 elements but input has 6',
    # T=6 -- the wrong axis got collapsed). Transposing first so T becomes the LAST torch axis (== ne[0])
    # makes the exact same `.mean()` call reduce the axis MEAN actually reduces, matching this project's
    # own "PERMUTE so the target axis lands on ne[0], THEN reduce" precedent (e.g.
    # convert_styletts2_diffusion.py's own to_out mean-pooling, or Kokoro's STFT reductions).
    x_full = x_full.transpose(1, 2).mean(dim=-1)  # (1,features)
    # Real `self.to_out` = nn.Sequential(Rearrange("b t c -> b c t"), Conv1d(features,channels,kernel=1))
    # applied to a single (mean-pooled) position -- degenerates to a plain Linear (kernel dim folded away),
    # same "1x1-conv-as-matmul" precedent already used by convert_styletts2_diffusion.py's own hand-built
    # topology (and Kokoro's F0_proj/N_proj). Composed explicitly as a Linear here rather than calling
    # `self.to_out[1]` (the real Conv1d module) directly: a length-1 CONV_1D traces to a MIL PERMUTE
    # feeding straight into CONV_1D's im2col lowering, and something in that specific (length=1,
    # kernel=1) combination produces a silently WRONG result in this exporter (confirmed empirically via
    # isolated bisection against an eager PyTorch reference -- builds and runs without error, but the
    # values are wrong; not yet root-caused further since the equivalent Linear composition sidesteps it
    # entirely and is already an established, correct pattern elsewhere in this project). A `.contiguous()`
    # workaround was tried first for a RELATED (crashing, not silently-wrong) issue in this same spot and
    # didn't survive tracing (MIL has no notion of contiguity, so coremltools drops such calls as a
    # no-op) -- switching to Linear avoids needing one at all.
    to_out_w = self.to_out[1].weight.reshape(self.to_out[1].weight.shape[0], -1)  # (channels,features)
    to_out_b = self.to_out[1].bias  # (channels,)
    x_full = torch.nn.functional.linear(x_full, to_out_w, to_out_b)  # (1,channels)
    x_full = x_full.unsqueeze(-1)  # (1,channels,1) -- matches real to_out's own (B,channels,1) output shape
    x_full = x_full.transpose(-1, -2)
    return x_full


Transformer1d.run = _transformer1d_run_traceable


def load_submodule(module, state_dict):
    """Same fallback convention as kokoro.model.KModel.__init__'s own checkpoint-loading loop: try the
    state dict as-is first, and only strip a `module.` DDP-training prefix (confirmed the real prefix
    both checkpoints use, see convert_styletts2_reused.py) if the strict load fails."""
    try:
        module.load_state_dict(state_dict)
    except Exception:
        stripped = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
        module.load_state_dict(stripped, strict=False)


class AlbertWrapper(torch.nn.Module):
    """Traces CustomAlbert (KModel.bert, a real transformers.AlbertModel returning last_hidden_state)
    ALONE -- see this file's own module docstring for why this stops short of `kokoro_export.py`'s
    combined AlbertBertEncoderWrapper (StyleTTS2's diffusion sampler needs this exact raw tensor as its
    own conditioning input). Same all-ones attention_mask / all-zeros token_type_ids / no-final-transpose
    conventions as AlbertBertEncoderWrapper, for the identical reasons documented there.
    """

    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids):
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        bert_dur = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return bert_dur.reshape(bert_dur.shape[1], bert_dur.shape[2])  # (T,768), row-major/time-major


def build_albert_phase(bert, dummy_t=7, vocab_size=178) -> ExportPhase:
    wrapper = AlbertWrapper(bert).eval()
    dummy_input_ids = torch.randint(0, vocab_size, (1, dummy_t), dtype=torch.long)
    with torch.no_grad():
        wrapper(dummy_input_ids)  # eager sanity check before tracing

    seq = ct.RangeDim(1, 2000)
    return ExportPhase(
        name="albert", wrapper=wrapper, dummy_inputs=(dummy_input_ids,),
        mil_inputs=[ct.TensorType(name="tokens", shape=(1, seq), dtype=np.int32)],
    )


class DiffusionNetWrapper(torch.nn.Module):
    """Traces the real `Transformer1d.run()` method directly (real source: styletts2 repo's
    Modules/diffusion/modules.py), NOT `Transformer1d.forward()` -- `.run()` is exactly the
    embedding_scale==1.0 branch `forward()` would otherwise dispatch to (confirmed:
    Demo/Inference_LJSpeech.ipynb's own `inference()` always calls the sampler with embedding_scale=1;
    the `!= 1.0` branch runs `.run()` TWICE for classifier-free guidance, out of scope, matching
    convert_styletts2_diffusion.py's own identical scoping decision). `features=None` throughout
    (config.yml's `multispeaker: false` -- confirmed no `to_features.*` keys in the real checkpoint, so
    `use_context_features` is always False for this model).

    `x_in` is StyleTTS2Driver's own noisy style "pseudo-token" (KDiffusion's `c_in`-scaled sampler state,
    a single 256-vector -- style_dim*2) -- reshaped to the real (B=1,T=1,channels) rank-3 shape `.run()`
    expects before its own internal `x.expand(-1, embedding.size(1), -1)` broadcast over T. `time` is the
    real `c_noise` scalar (already computed host-side, same "host does the small KDiffusion preconditioning
    scalar math" convention as the old bespoke topology) as a rank-1 (1,) tensor -- LearnedPositionalEmbedding
    needs a real per-batch-element rank, not a bare scalar. `embedding` is the raw, UNPROJECTED bert_dur
    (this file's own "albert" phase output, byte-identical (T,768) row-major convention -- matches
    `.run()`'s own `embedding.size(1)` == T expectation directly, no reshape needed).

    Output: the real `to_out` Conv1d(1x1) already collapses T back to exactly 1 position (the real
    `.run()` does `x.mean(axis=1)` BEFORE `to_out`) -- reshaped away to a plain (channels,) vector, same
    "no bare permute/singleton-dim as a traced output" caution as every other wrapper in this project's
    MIL exporters (a raw non-contiguous view read in pre-permute order by this exporter's own byte copy).
    """

    def __init__(self, transformer):
        super().__init__()
        self.net = transformer

    def forward(self, x_in, time, embedding):
        x = x_in.reshape(1, 1, -1)
        emb = embedding.reshape(1, embedding.shape[0], embedding.shape[1])
        out = self.net.run(x, time, emb, features=None)  # (1,1,channels)
        return out.reshape(out.shape[-1])


def build_diffusion_phase(transformer, dummy_t=6) -> ExportPhase:
    channels = DIFF_HP["channels"]
    ctx_feat = DIFF_HP["context_embedding_features"]
    wrapper = DiffusionNetWrapper(transformer).eval()

    dummy_x = torch.randn(channels) * 0.5
    dummy_time = torch.tensor([0.7])
    dummy_embedding = torch.randn(dummy_t, ctx_feat)
    with torch.no_grad():
        wrapper(dummy_x, dummy_time, dummy_embedding)  # eager sanity check before tracing

    seq = ct.RangeDim(1, 2000)
    return ExportPhase(
        name="diffusion", wrapper=wrapper, dummy_inputs=(dummy_x, dummy_time, dummy_embedding),
        mil_inputs=[
            ct.TensorType(name="x_in", shape=(channels,), dtype=np.float32),
            ct.TensorType(name="time", shape=(1,), dtype=np.float32),
            ct.TensorType(name="embedding", shape=(seq, ctx_feat), dtype=np.float32),
        ],
    )


# StyleTTS2's own Karras sigma schedule, and the one group of numbers here that is neither in the
# checkpoint nor in a config file: the repo wires `KarrasSchedule(sigma_min=0.0001, sigma_max=3.0,
# rho=9.0)` -- its own comment calls them "empirical parameters" -- into the DiffusionSampler at every
# call site (train_second.py:175, train_finetune.py:174, and the Colab demo). Unlike VITS's three
# scales they are NOT bound as overridable defaults: the repo's own inference entry point exposes
# `diffusion_steps` and `embedding_scale` and not these, so treating them as a per-utterance knob would
# be inventing an interface rather than preserving one. `diffusion_steps` stays an `infer` input.
SIGMA_MIN = 0.0001
SIGMA_MAX = 3.0
RHO = 9.0


@dataclass(kw_only=True)
class TTSStyleTTS2ExportConfig(BaseMultiPhaseModelExportConfig):
    """StyleTTS2's own three-phase split (albert/decoder_vocoder/diffusion) -- see module docstring.
    `kokoro_config_path` supplies the real hyperparameters this checkpoint shares byte-identically with
    Kokoro's own KokoroConfig (see `styletts2_driver.h`'s own top comment / `tools/convert_styletts2/
    PLAN.md`) -- a genuinely separate dependency from `checkpoint_path` (StyleTTS2's own weights)."""

    checkpoint_path: str
    kokoro_config_path: str = "/home/flavio/Dev/models/kokoro_model/config.json"
    # StyleTTS2's own training config, which is the only place `sigma_data` exists. Defaults to the
    # `config.yml` the release ships beside the checkpoint; `phases()` raises naming it if absent.
    styletts2_config_path: Optional[str] = None

    # Read in `phases()` and bound into the driver as `ExportConstants` (P4.0.8's first follow-up).
    style_dim: Optional[int] = field(default=None, init=False, repr=False)
    d_model: Optional[int] = field(default=None, init=False, repr=False)
    hidden_per_dir: Optional[int] = field(default=None, init=False, repr=False)
    sigma_data: Optional[float] = field(default=None, init=False, repr=False)
    # A DIRECTORY of `.lua` fragments -- StyleTTS2 is peeled (P4.0.6/C.8). See `driver_components`.
    driver_script_path: Path = Path(__file__).resolve().parent.parent / "convert_styletts2" / "styletts2_driver"

    def driver_components(self) -> List:
        """StyleTTS2's driver, as components (P4.0.6/C.8 -- the last family, and the one the plan
        predicted would "stay partly raw").

        It does, and the boundary is exactly where the plan said it would be. Two of thirteen
        `run_subgraph` calls become IR; seven name their topology with a computed expression, two sit
        inside Lua `for` loops, one is external, and **one is inside a closure**: `denoise_fn` is a
        local function the ADPM2 sampler calls twice per step, so its `diffusion` call cannot be a
        statement in the entry function at all. That call is what `estimators()`' `EstimatorSpec`
        exists for -- it is checked without being generated, which is the split
        `flow_matching_export.py`'s own docstring argues for and this family is the reason it exists.

        The ADPM2 sampler itself stays a hand-written helper in `00_header.lua`, unchanged: two network
        evaluations per step, Karras preconditioning, per-step noise injection. No template emits that
        without becoming a worse thing to read than the loop."""
        from .driver_components import (
            ComputedCall, DriverReturn, ExportConstants, HelperCall, LuaFragment,
            SubgraphCallComponent,
        )
        from .lua_library import LuaLibrary
        from .driver_ir import Call, FieldAccess, Lit, Var

        fragment = self.driver_script_path
        external = self.external_topologies()
        t_text, t_frames = Var("T_text"), Var("T_frames")

        def block(name, **kwargs):
            return LuaFragment(fragment / name, external=external, **kwargs)

        return [
            block("00_header.lua", top_level=True),
            # The shared driver-side library. Every one of these was a byte-identical copy
            # in this family and the other one until they moved to lua/ -- 11 functions,
            # 112 lines, shipped twice. Only what is declared here is emitted.
            LuaLibrary(uses=(
                "array_slice", "array_sum",
                "to_row_major", "from_row_major", "to_layout_a",
                "from_layout_a", "run_bi_lstm", "run_resblk_stack",
                "run_proj1x1", "predict_durations", "compute_wsum",
                "karras_schedule", "adpm2_sample",
            )),
            # The eleven numbers the caller used to have to supply (P4.0.8's first follow-up).
            # Three come off the real TextEncoder/ProsodyPredictor, four are the istftnet geometry the
            # decoder_vocoder graph was traced with, three are StyleTTS2's own Karras schedule, and
            # SIGMA_DATA is read from the checkpoint's config.yml because it is estimated per training
            # run. `diffusion_steps` is NOT here -- it stays an input, because it is the one the repo's
            # own inference entry point exposes.
            ExportConstants(values={
                "STYLE_DIM": self.style_dim,
                "D_MODEL": self.d_model,
                "HIDDEN_PER_DIR": self.hidden_per_dir,
                "HARMONIC_NUM": kokoro_export._HARMONIC_NUM,
                "UPSAMPLE_SCALE": kokoro_export._UPSAMPLE_SCALE,
                "GEN_ISTFT_N_FFT": kokoro_export._STFT_N_FFT,
                "GEN_ISTFT_HOP": kokoro_export._STFT_HOP,
                "SIGMA_MIN": SIGMA_MIN,
                "SIGMA_MAX": SIGMA_MAX,
                "RHO": RHO,
                "SIGMA_DATA": self.sigma_data,
            }),
            block("01_lengths.lua", defines=("T_text",)),
            SubgraphCallComponent(
                topology="albert", outputs=("bert_out",), length=t_text,
                inputs={"tokens": FieldAccess("inputs", "input_ids")},
                note="--- CustomAlbert, ONE MIL-traced call -> bert_out, time-major (T,768)\n"
                     "    (== ne=[768,T] Layout B, see module docstring for why this convention,\n"
                     "    not styletts2_driver.lua's own explicit-transpose one). ---"),
            block("02_style_diffusion.lua",
                  reads=("bert_out", "T_text", "STYLE_DIM", "SIGMA_MIN", "SIGMA_MAX", "RHO",
                         "SIGMA_DATA"),
                  defines=("denoise_fn", "style_vec_dim", "noise0", "sigmas", "s_pred",
                           "s_decoder", "s_predictor")),
            # `drives` is D.2, and this family is where it pays most: nine of StyleTTS2's thirteen
            # call sites name their topology with an expression built at run time, either in a Lua
            # loop here or inside the loom_lua helper one level down. Declaring the namespaces is what
            # turns them back into ordinary checked calls -- see driver_components.HelperCall.
            block("03_duration_encoder.lua",
                  reads=("bert_out", "T_text", "D_MODEL", "STYLE_DIM", "s_predictor",
                         "HIDDEN_PER_DIR"),
                  defines=("d_en_flat", "x", "d", "top_out", "duration_logits", "pred_dur"),
                  drives=(
                      HelperCall("run_bi_lstm", tuple(f"duration_lstm_{i}" for i in range(3)),
                                 written='"duration_lstm_" .. i'),
                      ComputedCall(topologies=tuple(f"duration_adaln_{i}" for i in range(3)),
                                   inputs=("x", "style"), written='"duration_adaln_" .. i'),
                      HelperCall("run_bi_lstm", "top_lstm"),
                  )),
            block("04_frame_expansion.lua",
                  reads=("T_text", "pred_dur", "d", "D_MODEL", "STYLE_DIM", "HIDDEN_PER_DIR"),
                  defines=("T_frames", "d_channels", "en", "cnn_flat", "cnn_shape", "te_channels",
                           "cnn_rows", "t_en", "asr"),
                  drives=(HelperCall("run_bi_lstm", "text_encoder_lstm"),)),
            block("05_f0n.lua", reads=("en", "HIDDEN_PER_DIR", "s_predictor"),
                  defines=("shared_out", "f0_feat", "n_feat", "F0_curve", "N_curve"),
                  drives=(
                      HelperCall("run_bi_lstm", "f0n_shared_lstm"),
                      HelperCall("run_resblk_stack", "f0n_f0"),
                      HelperCall("run_resblk_stack", "f0n_n"),
                      HelperCall("run_proj1x1", "f0n_f0_proj"),
                      HelperCall("run_proj1x1", "f0n_n_proj"),
                  )),
            block("06_vocoder_inputs.lua",
                  reads=("T_frames", "HARMONIC_NUM", "UPSAMPLE_SCALE", "GEN_ISTFT_N_FFT",
                         "GEN_ISTFT_HOP"),
                  defines=("dim", "T_f0", "L", "rand_ini", "u", "noise_in", "wsum")),
            SubgraphCallComponent(
                topology="decoder_vocoder", outputs=("waveform",),
                axes={"n_enc_frames": t_frames, "n_past": Lit(0)},
                inputs={
                    "asr": Call("to_layout_a", [Var("asr"), t_frames, Lit(512)]),
                    "f0_curve": Var("F0_curve"), "n_curve": Var("N_curve"),
                    "s": Var("s_decoder"), "rand_ini": Var("rand_ini"),
                    "noise_in": Var("noise_in"), "wsum": Var("wsum"),
                },
                multiline=True),
            DriverReturn(values=("waveform",)),
        ]

    # No `hparams()`, and the absence is deliberate. Kokoro declares `loom.style_dim` because its host
    # has to BUILD `ref_s`; StyleTTS2 samples its own style vector inside the driver, so a host needs
    # nothing but `input_ids` to call `infer`. Declaring style_dim here anyway would be a KV nobody
    # reads -- which is precisely the decorative declaration P4.0.7's own catalogue work warns about.

    def external_topologies(self) -> Dict[str, str]:
        """Empty, as of P4.0.7 -- see `TTSKokoroExportConfig.external_topologies` for the full note.

        This listed `bert_encoder`, `text_encoder_cnn` and `duration_proj` while StyleTTS2's MIL export
        was partial. All three are now traced here, along with the six BiLSTMs and everything else the
        driver calls, so the emitted GGUF is self-contained."""
        return {}

    __unchecked__ = {
        "checkpoint_path": Unchecked(
            "path to StyleTTS2's own .pth. The recognizer's detect() already established the structure "
            "this config depends on, and it is the near-collision with Kokoro that makes that check the "
            "real one: both checkpoints are dicts of component name -> OrderedDict with no version "
            "marker, so detect() discriminates on what Kokoro's inference-only release strips. A path "
            "link would accept the Kokoro checkpoint."
        ),
        "kokoro_config_path": Unchecked(
            "a genuinely separate dependency -- the hyperparameters this checkpoint shares "
            "byte-identically with Kokoro's own KokoroConfig, which StyleTTS2's own release does not "
            "ship. phases() reads it as JSON and every field it needs raises by name if absent, which "
            "is more specific than a path check."
        ),
        "styletts2_config_path": Unchecked(
            "StyleTTS2's own config.yml, defaulted to the one beside the checkpoint. Same argument as "
            "the two paths above: phases() reads `model_params.diffusion.dist.sigma_data` out of it "
            "and raises naming the file and the key if either is missing, which says more than a "
            "'this path exists' link would"
        ),
        "style_dim": Unchecked(
            "DERIVED in phases() by `kokoro_export.prosody_dims` from the restored ProsodyPredictor "
            "itself -- see TTSKokoroExportConfig for the full note; these are the same classes with "
            "this checkpoint's weights"
        ),
        "d_model": Unchecked("same"),
        "hidden_per_dir": Unchecked("same"),
        "sigma_data": Unchecked(
            "READ off config.yml's `model_params.diffusion.dist.sigma_data`, the ONLY authority on it "
            "-- the same config declares `estimate_sigma_data: True`, so it is a statistic of the "
            "training data rather than a constant, and a different checkpoint has a different one. "
            "Nothing in the graph consumes it (the KDiffusion preconditioning around the diffusion "
            "call is host-side arithmetic in the driver), so its check is numeric: the frozen "
            "reference waveform in fixtures/legacy_driver_reference/"
        ),
    }

    def phases(self) -> List[ExportPhase]:
        print(f"Loading StyleTTS2 checkpoint {self.checkpoint_path}...")
        sd_all = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)["net"]

        kokoro_cfg = json.load(open(self.kokoro_config_path))
        kokoro_export.check_istftnet_geometry(kokoro_cfg)
        self.sigma_data = self._read_sigma_data()
        # Real hyperparameters confirmed byte-identical to Kokoro's own KokoroConfig -- see
        # styletts2_driver.h's own top comment / tools/convert_styletts2/PLAN.md.
        bert = CustomAlbert(AlbertConfig(vocab_size=kokoro_cfg["n_token"], **kokoro_cfg["plbert"]))
        decoder = Decoder(dim_in=kokoro_cfg["hidden_dim"], style_dim=kokoro_cfg["style_dim"],
                           dim_out=kokoro_cfg["n_mels"], disable_complex=True, **kokoro_cfg["istftnet"])
        load_submodule(bert, sd_all["bert"])
        load_submodule(decoder, sd_all["decoder"])
        bert.eval()
        decoder.eval()

        transformer = Transformer1d(num_layers=DIFF_HP["num_layers"], channels=DIFF_HP["channels"],
                                     num_heads=DIFF_HP["num_heads"], head_features=DIFF_HP["head_features"],
                                     multiplier=DIFF_HP["multiplier"],
                                     context_embedding_features=DIFF_HP["context_embedding_features"])
        diff_prefix = "module.unet."
        diff_sd = {k[len(diff_prefix):]: v for k, v in sd_all["diffusion"].items() if k.startswith(diff_prefix)}
        transformer.load_state_dict(diff_sd)
        transformer.eval()

        print("Tracing albert phase...")
        albert_phase = build_albert_phase(bert)

        print("Tracing decoder_vocoder phase...")
        dv_phase = kokoro_export.build_decoder_vocoder_phase(decoder)

        print("Tracing diffusion phase...")
        diffusion_phase = build_diffusion_phase(transformer)

        # bert_encoder: the Linear(768, 512) projecting raw bert_dur for the duration half. Kept out of
        # the "albert" phase deliberately (the diffusion sampler needs the UNPROJECTED bert_dur -- see
        # this module's own docstring), and traced here rather than borrowed from the pre-MIL gguf.
        bert_encoder = torch.nn.Linear(bert.config.hidden_size, kokoro_cfg["hidden_dim"])
        load_submodule(bert_encoder, sd_all["bert_encoder"])
        bert_encoder.eval()

        # TextEncoder + ProsodyPredictor: the same classes Kokoro uses, with this checkpoint's weights
        # (Kokoro is a StyleTTS2 derivative -- the bespoke converters reuse Kokoro's hyperparameter
        # dicts for StyleTTS2 wholesale), so their 21 phases come from one shared builder rather than a
        # second copy. Same reuse as `build_decoder_vocoder_phase` just above.
        text_encoder = TextEncoder(channels=kokoro_cfg["hidden_dim"],
                                    kernel_size=kokoro_cfg["text_encoder_kernel_size"],
                                    depth=kokoro_cfg["n_layer"], n_symbols=kokoro_cfg["n_token"])
        predictor = ProsodyPredictor(style_dim=kokoro_cfg["style_dim"], d_hid=kokoro_cfg["hidden_dim"],
                                      nlayers=kokoro_cfg["n_layer"], max_dur=kokoro_cfg["max_dur"],
                                      dropout=kokoro_cfg["dropout"])
        load_submodule(text_encoder, sd_all["text_encoder"])
        load_submodule(predictor, sd_all["predictor"])
        text_encoder.eval()
        predictor.eval()

        print("Tracing bert_encoder + TextEncoder/ProsodyPredictor phases...")
        bert_encoder_phase = ExportPhase(
            name="bert_encoder", wrapper=_BertEncoderWrapper(bert_encoder).eval(),
            dummy_inputs=(torch.randn(7, bert.config.hidden_size),),
            mil_inputs=[ct.TensorType(name="x", shape=(ct.RangeDim(1, 2000), bert.config.hidden_size),
                                      dtype=np.float32)],
        )

        self.style_dim, self.d_model, self.hidden_per_dir = (
            kokoro_export.prosody_dims(text_encoder, predictor)[k]
            for k in ("style_dim", "d_model", "hidden_per_dir"))

        return [albert_phase, dv_phase, diffusion_phase, bert_encoder_phase,
                *kokoro_export.build_prosody_phases(text_encoder, predictor)]

    def _read_sigma_data(self) -> float:
        """`model_params.diffusion.dist.sigma_data` out of StyleTTS2's own config.yml.

        There is no fallback and there should not be one: the same config says
        `estimate_sigma_data: True`, so this is a statistic of the training data, not a constant of the
        architecture. Substituting the LJSpeech release's 0.4573... for a checkpoint that does not
        state it would silently mis-precondition every denoiser call.
        """
        import yaml

        path = Path(self.styletts2_config_path or (Path(self.checkpoint_path).parent / "config.yml"))
        if not path.is_file():
            raise FileNotFoundError(
                f"StyleTTS2's own config.yml is required and is not at {path}. It is the only "
                f"authority on `model_params.diffusion.dist.sigma_data`, which the driver's KDiffusion "
                f"preconditioning needs and which is estimated per training run "
                f"(`estimate_sigma_data: True`). Pass styletts2_config_path if it lives elsewhere."
            )
        cfg = yaml.safe_load(path.read_text())
        try:
            return float(cfg["model_params"]["diffusion"]["dist"]["sigma_data"])
        except (KeyError, TypeError) as exc:
            raise KeyError(
                f"{path} has no model_params.diffusion.dist.sigma_data ({exc}). The driver's "
                f"KDiffusion preconditioning cannot be written without it."
            ) from exc

    def estimators(self) -> List[EstimatorSpec]:
        # The ADPM2/Karras sampler loop itself stays hand-written (EXPORT-IMPROVEMENT.md item 4 concedes
        # true one-offs, and this one is a second-order sampler with two network evaluations and real
        # preconditioning math per step -- see styletts2_driver/). But its per-step `run_subgraph`
        # call has the same failure mode as every generated one, so it is declared here and cross-checked
        # against the real traced "diffusion" topology at export time rather than at run time.
        return [EstimatorSpec(topology="diffusion", inputs=["x_in", "time", "embedding"])]


def _is_styletts2(path: Path) -> bool:
    """Real structural check (BACKLOG.md P4.0.1): a `.pth` file holding StyleTTS2's own `net` component
    dict, including the `diffusion` component this family's sampler is built around.

    Both keys are the answer to a near-collision worth recording. Kokoro is a StyleTTS2 derivative --
    this module borrows `kokoro_export.build_decoder_vocoder_phase` outright for that reason -- and the
    two checkpoints are the same kind of object: a dict of component name -> `OrderedDict` state dict,
    leading with the identical `bert` -> `module.embeddings.word_embeddings.weight` ALBERT keys, with no
    version marker, no config, and no class reference beyond `collections.OrderedDict`. Probing both
    real checkpoints (`checkpoint_probe.probe_torch_checkpoint`) found every component name in Kokoro's
    also present in StyleTTS2's, so the discriminator has to run the other way, on what Kokoro's
    inference-only release *strips*: the `net` wrapper (which `export()` itself indexes through) and the
    training-time components under it (`diffusion`, `mpd`, `msd`, `wd`). `diffusion` is the one to key
    on beyond `net` -- it is exactly why this family stays a plain `BaseMultiPhaseModelExportConfig`
    with a hand-written ADPM2 sampler rather than a `TTSFlowMatchingModelExportConfig`."""
    probe = probe_torch_checkpoint(path)
    if probe is None:
        return False
    return {"net", "diffusion", "bert", "decoder"}.issubset(probe.strings)


def _build_styletts2(path: Path, output_path: str) -> TTSStyleTTS2ExportConfig:
    return TTSStyleTTS2ExportConfig(
        architecture="loom-styletts2-mil", output_path=output_path, checkpoint_path=str(path),
    )


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-to-speech",
        config_class=BaseMultiPhaseModelExportConfig,
        recognizers=[ModelRecognizer(name="styletts2", detect=_is_styletts2, build_config=_build_styletts2)],
    ))


class _BertEncoderWrapper(torch.nn.Module):
    """`bert_encoder` -- Linear(768, 512), rows_flat in, **Layout A out**.

    The input side is rows_flat: the bespoke topology declares `x` as `["768", "$n_tokens"]`
    (ne=[768, T], flat[t*768+c] -- a contiguous torch `(T, 768)`), and the driver hands it `bert_out`
    straight from the "albert" phase, which is time-major for exactly this reason.

    The OUTPUT side crosses layouts, and this is the one place in the transfer where the two sides
    genuinely disagreed. The bespoke topology ends in `PERMUTE(axes=[1,0,2,3]) + CONT`, so it returns
    Layout A -- and the driver reads it that way (`d_en_flat[c * T_text + t + 1]`, with the fragment's
    own comment saying "Layout A [T,512]"). A wrapper that just returned the Linear's natural `(T, 512)`
    builds and runs and produces a transposed result: the equivalence test caught it as
    mean_abs_diff=0.717 against a reference whose values only reach 2.23, which is what a transpose
    looks like when nothing crashes."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):  # (T, 768) rows_flat -> (512, T) Layout A
        return self.linear(x).transpose(0, 1).contiguous()
