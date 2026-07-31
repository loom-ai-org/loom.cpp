"""Exports StyleTTS2's (yl4579/StyleTTS2-LJSpeech) MIL-traceable phases into ONE combined
`styletts2_mil.gguf` (BACKLOG.md P3.3, migrated from `export_styletts2_mil.py`) alongside the embedded
`styletts2_driver_mil.lua` orchestration script:
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
`BaseMultiPhaseModelExportConfig` with the ADPM2 loop hand-written in `styletts2_driver_mil.lua` and only
`EstimatorSpec`-checked via `estimators()`: the sampler's per-step `run_subgraph` call still gets the same
export-time validation against the real traced "diffusion" topology, generating no codegen.

Numerically verified against real-checkpoint references (see `tools/convert_styletts2/
reference_forward_styletts2_{albert_mil,diffusion}.py` and `kokoro_export.py`'s own already-verified
decoder_vocoder reference reused as-is): see `test_e2e_styletts2_mil_*.cpp` for the actual tolerances.

Usage:
  loom-export /path/to/styletts2.pth -o styletts2_mil.gguf --task tts-multi-phase --model styletts2
"""
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import coremltools as ct

from .flow_matching_export import EstimatorSpec
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .patcher import ModelPatcher


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
from kokoro.modules import CustomAlbert  # noqa: E402
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


@dataclass(kw_only=True)
class TTSStyleTTS2ExportConfig(BaseMultiPhaseModelExportConfig):
    """StyleTTS2's own three-phase split (albert/decoder_vocoder/diffusion) -- see module docstring.
    `kokoro_config_path` supplies the real hyperparameters this checkpoint shares byte-identically with
    Kokoro's own KokoroConfig (see `styletts2_driver.h`'s own top comment / `tools/convert_styletts2/
    PLAN.md`) -- a genuinely separate dependency from `checkpoint_path` (StyleTTS2's own weights)."""

    checkpoint_path: str
    kokoro_config_path: str = "/home/flavio/.claude/tmp/kokoro_model/config.json"
    driver_script_path: Path = Path(__file__).resolve().parent.parent / "convert_styletts2" / "styletts2_driver_mil.lua"

    def phases(self) -> List[ExportPhase]:
        print(f"Loading StyleTTS2 checkpoint {self.checkpoint_path}...")
        sd_all = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)["net"]

        kokoro_cfg = json.load(open(self.kokoro_config_path))
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

        return [albert_phase, dv_phase, diffusion_phase]

    def estimators(self) -> List[EstimatorSpec]:
        # The ADPM2/Karras sampler loop itself stays hand-written (EXPORT-IMPROVEMENT.md item 4 concedes
        # true one-offs, and this one is a second-order sampler with two network evaluations and real
        # preconditioning math per step -- see styletts2_driver_mil.lua). But its per-step `run_subgraph`
        # call has the same failure mode as every generated one, so it is declared here and cross-checked
        # against the real traced "diffusion" topology at export time rather than at run time.
        return [EstimatorSpec(topology="diffusion", inputs=["x_in", "time", "embedding"])]


def _is_styletts2(path: Path) -> bool:
    """No auto-detection this pass -- requires an explicit `--task tts-multi-phase --model styletts2`
    (BACKLOG.md P3.3's stated scope limit)."""
    return False


def _build_styletts2(path: Path, output_path: str) -> TTSStyleTTS2ExportConfig:
    return TTSStyleTTS2ExportConfig(
        architecture="loom-styletts2-mil", output_path=output_path, checkpoint_path=str(path),
    )


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="tts-multi-phase",
        config_class=BaseMultiPhaseModelExportConfig,
        recognizers=[ModelRecognizer(name="styletts2", detect=_is_styletts2, build_config=_build_styletts2)],
    ))
