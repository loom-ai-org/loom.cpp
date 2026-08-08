"""The audio-encoder + AR-cross-attention-decoder family (`EXPORT-ROADMAP.md` R5's family 2), on
Whisper -- BACKLOG.md P4.1.

**What is new here, and what is deliberately not.** Every earlier family either runs its model once
(`Flattened`: Qwen3, the NeMo encoders) or runs N phases in a fixed order with the loop over *data*
(`MultiPhase`: the TTS zoo, Parakeet's transducer). This family is the first whose second phase is a
KV-cached transformer decoding against a *first* phase's output -- an encoder run once, then a decode
loop that attends to its result at every step. That is one new fact for the export to carry (which
phase's attention is cached) and one new fact for the driver (which input holds the encoder's output);
everything else is the machinery those two families already have, which is why this module declares
phases and components rather than bringing a fourth `Decomposition`. See BACKLOG.md P4.1 for the
measurement behind that decision.

**The mel frontend is part of the exported graph, not the host's problem.** HF's `WhisperEncoder`
starts at `input_features` -- a log-mel spectrogram computed by `WhisperFeatureExtractor`, which is
numpy/torch code outside the model. Every other audio family in this tree traces its own frontend
(NeMo's `AudioToMelSpectrogramPreprocessor` is a real `nn.Module` and is traced with the encoder), and
the bespoke converter this replaces built the same mel in-graph from DFT-as-convolution kernels. So
`WhisperMelFrontend` reimplements the feature extractor's own arithmetic as a traceable module and the
exported encoder takes a **waveform**, keeping the "a host hands the engine audio, not features"
contract that makes the GGUF self-contained.

**The decoder is traced without a cache and decodes with one**, which is KV-CACHE.md's finding applied
unchanged: `fuse_loom_attention` turns each self-attention block into an `ATTENTION` node, the engine
supplies the past, and a decode step is the same graph at `n_tokens = 1`. Cross-attention is *not*
fused, and that is the correct outcome rather than a gap -- see `ASRWhisperExportConfig.phases`.
"""
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .decomposition import Decomposition, MultiPhase
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Unchecked


class WhisperMelFrontend(nn.Module):
    """`WhisperFeatureExtractor`'s log-mel spectrogram, as a traceable `nn.Module`.

    Numerically identical to the extractor, not merely equivalent: the filterbank is the extractor's
    own `mel_filters` array (read off the checkpoint's `preprocessor_config.json`, never recomputed
    here), and the arithmetic below is `_torch_extract_fbank_features` line for line -- Hann-windowed
    STFT, drop the final frame, power spectrum, filterbank, log10 with a 1e-10 floor, clamp to 8 dB
    below the clip's own maximum, then `(x + 4) / 4`.

    **The global maximum is why this cannot be simplified.** `torch.maximum(log_spec, log_spec.max() -
    8)` makes every output element depend on the loudest bin in the whole 30 s clip, so the frontend is
    not separable per frame and cannot be pushed onto the host per chunk without changing the numbers.

    Takes `(1, n_samples)` rather than `(n_samples,)`: a 1-D input makes `torch.stft` trace an
    `aten::size` -> `aten::Int` chain over the sample axis, and the batch axis avoids it entirely.
    """

    def __init__(self, n_fft: int, hop_length: int, mel_filters: np.ndarray):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.register_buffer("window", torch.hann_window(self.n_fft))
        # The extractor stores `(n_freq, n_mels)`; the matmul below wants `(n_mels, n_freq)`.
        self.register_buffer(
            "filters", torch.from_numpy(np.asarray(mel_filters, dtype=np.float32)).T.contiguous()
        )

    def forward(self, waveform):
        stft = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft,
            window=self.window, center=True, return_complex=True,
        )
        # Whisper drops the last STFT frame, so a 30 s clip yields 3000 frames and not 3001. The
        # extractor writes this as `stft[..., :-1].abs() ** 2`; taking the magnitude FIRST is the same
        # arithmetic on one fewer element and is the only order that converts -- coremltools' complex
        # dialect has no `slice_by_index` over a complex tensor, so slicing before `abs()` fails with
        # `expects tensor ... but got tensor[1, 201, 3001, complex64]`.
        magnitudes = (stft.abs() ** 2)[..., :-1]
        mel_spec = self.filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0


class _WhisperEncoderWrapper(nn.Module):
    """`waveform -> encoder hidden states`, the one tensor the encoder phase exports.

    The mel frontend is inside the wrapper rather than beside it precisely so the traced graph contains
    it: this is the module docstring's "self-contained GGUF" claim in code.
    """

    def __init__(self, model, mel: WhisperMelFrontend):
        super().__init__()
        self.mel = mel
        self.encoder = model.model.encoder

    def forward(self, waveform):
        return self.encoder(self.mel(waveform)).last_hidden_state


class _WhisperDecoderWrapper(nn.Module):
    """`(tokens, position_ids, attention_mask, xa) -> logits`.

    The four inputs are not a free choice. `position_ids` and `attention_mask` are passed explicitly
    for the reason `causal_lm_export._causal_mask` documents -- it is what keeps the token axis
    genuinely dynamic under `torch.jit.trace` -- and both names are already in
    `driver_components.POSITION_INPUT_NAMES`/`CAUSAL_MASK_INPUT_NAMES`, so the driver fills them in from
    `n_tokens`/`n_past` without this family declaring anything. `xa` is the encoder phase's output,
    passed unchanged at every step.

    `use_cache=False` is what makes the trace cache-free, which is the shape `fuse_loom_attention`
    matches; the cache appears at run time, in the engine, not in the graph.
    """

    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out

    def forward(self, tokens, position_ids, attention_mask, xa):
        hidden = self.decoder(
            input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask,
            encoder_hidden_states=xa, use_cache=False,
        ).last_hidden_state
        return self.proj_out(hidden)


def causal_mask(seq_len: int) -> torch.Tensor:
    """A 4D additive causal mask, the form transformers passes straight through to attention.

    Same tensor `causal_lm_export._causal_mask` builds, and for the same reason: an already-prepared 4D
    mask short-circuits `create_causal_mask` entirely, so the internal mask builder never derives a
    key length from a Python-level shape that tracing would bake in.
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


def decoder_prompt_constants(generation_config, n_audio_ctx: int, vocab_size: int) -> dict:
    """The token ids a Whisper decode prompt is built from, read off the checkpoint's own generation
    config -- for the DRIVER to build the prompt with, not for a host to hardcode.

    Five numbers, and which of them exist is itself the capability statement this family needs:

    * `SOT` -- `decoder_start_token_id`. Always present; a prompt is at minimum this one token.
    * `LANG_LO` / `LANG_HI` -- the half-open id window the 98 language tokens occupy. Present only on a
      **multilingual** checkpoint. `0`/`0` on an English-only one (`whisper-*.en` has no language tokens
      at all), which is how the driver knows it must neither ask for a language nor try to detect one.
      Derived as `min..max+1` over `lang_to_id`'s own values rather than assumed contiguous -- and
      cross-checked below, because a non-contiguous block would make a restricted argmax able to return
      something that is not a language.
    * `TRANSCRIBE` / `TRANSLATE` -- `task_to_id`, likewise absent (`0`) on an English-only checkpoint.
    * `NO_TIMESTAMPS` -- `no_timestamps_token_id`, `0` if this checkpoint has no such token.

    **Not `forced_decoder_ids`**, which is the obvious-looking source and is wrong: a multilingual
    checkpoint leaves its language slot `None` there (`[[1, None], [2, 50359]]` on whisper-small) because
    HF fills it in from detection at generation time. Copying that list yields a prompt with a hole in it.
    """
    lang_to_id = getattr(generation_config, "lang_to_id", None) or {}
    task_to_id = getattr(generation_config, "task_to_id", None) or {}
    lang_ids = sorted(int(v) for v in lang_to_id.values())
    lang_lo, lang_hi = (lang_ids[0], lang_ids[-1] + 1) if lang_ids else (0, 0)
    if lang_ids and len(lang_ids) != lang_hi - lang_lo:
        # The driver detects a language with ONE restricted argmax over [LANG_LO, LANG_HI). A gap in the
        # block means some id in that window is not a language token, so detection could return it --
        # silently, as a plausible-looking prompt token. Say so at export time instead.
        raise ValueError(
            f"this checkpoint's {len(lang_ids)} language tokens do not occupy a contiguous id block: "
            f"{lang_lo}..{lang_hi - 1} spans {lang_hi - lang_lo} ids. Language detection is a single "
            f"argmax restricted to that window, which is only sound while every id in it is a language."
        )
    # The timestamp block: Whisper lays it out immediately above `<|notimestamps|>` and runs it to the
    # end of the vocabulary, one token per encoder frame plus one for the closing edge (`<|0.00|>` ..
    # `<|30.00|>` in 0.02 s steps = n_audio_ctx + 1 of them). Derived from that layout rather than by
    # looking tokens up by name, and then CHECKED against the model's own frame count -- a driver takes a
    # timestamp-restricted argmax over this window, so a wrong bound would silently return a word.
    no_timestamps = int(getattr(generation_config, "no_timestamps_token_id", None) or 0)
    ts_lo = ts_hi = 0
    if no_timestamps:
        ts_lo, ts_hi = no_timestamps + 1, int(vocab_size)
        # Unconditional once the checkpoint has a `<|notimestamps|>` at all -- a guarded version of this
        # check is what let `TS_HI = 0` ship once: `vocab_size` is on the model config, not the
        # generation config, so reading it from the wrong object produced a zero, the guard skipped
        # itself, and the driver silently stopped forcing timestamps. A bound this arithmetic depends on
        # gets verified or the export fails; it does not get defaulted.
        if ts_hi - ts_lo != n_audio_ctx + 1:
            raise ValueError(
                f"this checkpoint's timestamp block is {ts_hi - ts_lo} tokens ({ts_lo}..{ts_hi - 1}), but "
                f"its encoder emits {n_audio_ctx} frames, which should give {n_audio_ctx + 1} timestamps "
                f"(one per frame boundary). The bounds come from `no_timestamps_token_id` + 1 and "
                f"`vocab_size`, so one of those is not what this family assumes."
            )
    return {
        "SOT": int(generation_config.decoder_start_token_id),
        "LANG_LO": lang_lo,
        "LANG_HI": lang_hi,
        "TRANSCRIBE": int(task_to_id.get("transcribe", 0)),
        "TRANSLATE": int(task_to_id.get("translate", 0)),
        "NO_TIMESTAMPS": no_timestamps,
        "TS_LO": ts_lo,
        "TS_HI": ts_hi,
    }


def load_feature_extractor(model_dir: str):
    """The checkpoint's own `WhisperFeatureExtractor`.

    Loaded from the directory rather than constructed with this family's idea of the defaults: the mel
    filterbank, the FFT geometry and the 30 s chunk length are all properties of the checkpoint, and
    `preprocessor_config.json` is where it states them.
    """
    from transformers import WhisperFeatureExtractor

    return WhisperFeatureExtractor.from_pretrained(model_dir)


@dataclass
class ASRWhisperExportConfig(BaseMultiPhaseModelExportConfig):
    """Whisper as two traced phases -- `encoder` (waveform -> audio states) and `decoder` (a cached,
    cross-attending transformer step) -- plus a driver that runs the first once and loops the second.

    **Why this is a `MultiPhase` config and not a fourth `Decomposition`.**
    `EXPORT-PREPARATION.md` §5 decision 2 reserved a decomposition of its own for this shape, on the
    reasoning that a new orchestration needs a driver builder the family cannot supply. Building it
    found the orchestration to be the one `MultiPhase` already has: two independently traced phases, a
    component list, and `MultiPhaseDriverBuilder`. What genuinely differs is *two facts*, and both are
    now fields on the pieces that own them -- `ExportPhase.fuse_attention` (this decoder is cached and
    this encoder must not be) and `PrefillDecodeLoop.bound` (this step's `xa` comes from the encoder,
    not from the caller). A `Decomposition` subclass would have restated `MultiPhase.export` verbatim
    around those two. See BACKLOG.md P4.1.
    """

    model_dir: str = ""
    architecture: str = "whisper"
    output_path: str = "whisper_mil.gguf"
    # The decoder's token axis. The encoder phase declares `n_samples` instead -- raw audio, never a
    # token count -- the same distinction the NeMo family draws (EXPORT-ROADMAP.md R1).
    root_axis: str = "n_tokens"
    driver_script_path: Path = Path(__file__).resolve().parent / "whisper_driver"
    decomposition: Decomposition = field(default_factory=MultiPhase)

    # Read off the checkpoint in `phases()`, which is the only moment the model and its feature
    # extractor are both in hand. Declared as fields rather than recomputed because the driver
    # components and `hparams()` need them after the trace.
    n_samples: Optional[int] = field(default=None, init=False, repr=False)
    sample_rate: Optional[int] = field(default=None, init=False, repr=False)
    n_audio_ctx: Optional[int] = field(default=None, init=False, repr=False)
    d_model: Optional[int] = field(default=None, init=False, repr=False)
    max_target_positions: Optional[int] = field(default=None, init=False, repr=False)
    decoder_bindings: tuple = field(default=(), init=False, repr=False)
    prompt_constants: dict = field(default_factory=dict, init=False, repr=False)

    __unchecked__ = {
        "model_dir": Unchecked(
            "path to the HF directory, already established by the recognizer's own detect(), which "
            "reads its config.json `model_type`. WhisperForConditionalGeneration.from_pretrained "
            "raises on anything it cannot load."
        ),
        "architecture": Unchecked("the GGUF's own architecture string; it names this export, and there "
                                  "is no second authority to compare it against"),
        "output_path": Unchecked("where to write. A caller's choice, not a claim about the model."),
        "root_axis": Unchecked("checked by the decoder ExportPhase's own Axis link, which is where the "
                               "value is actually used"),
        "driver_script_path": Unchecked("the one hand-written fragment here is a header comment; its "
                                        "contents are still parsed and cross-checked by LuaFragment"),
        "decomposition": Unchecked("MultiPhase by construction -- see the class docstring for why this "
                                   "shape did not need a fourth one"),
        "n_samples": Unchecked("READ off the checkpoint's own feature extractor in phases() "
                               "(`chunk_length * sampling_rate`), not declared"),
        "sample_rate": Unchecked("same -- the feature extractor's own `sampling_rate`"),
        "n_audio_ctx": Unchecked("same -- `config.max_source_positions`"),
        "d_model": Unchecked("same -- `config.d_model`"),
        "max_target_positions": Unchecked("same -- `config.max_target_positions`, which is the KV "
                                          "cache capacity a decode loop can address"),
        "decoder_bindings": Unchecked(
            "(name, kind) per decoder input, derived in phases() from the SAME mil_inputs list the "
            "trace is declared with, through `exporter._binding_kind` -- the one implementation of "
            "'is this input host-computed', which the flattened path already routes through -- so the "
            "driver cannot disagree with the trace about the order or the names, and this family "
            "cannot drift from the causal-LM one about which names the driver fills in. "
            "PrefillDecodeLoop's own `inputs` link re-checks them against the emitted topology anyway."
        ),
        "prompt_constants": Unchecked(
            "READ off the checkpoint's own generation config in phases() (`decoder_start_token_id`, "
            "`lang_to_id`, `task_to_id`, `no_timestamps_token_id`), never declared -- see "
            "`decoder_prompt_constants`, which DOES cross-check the one property a restricted argmax "
            "depends on: that the language tokens form a contiguous id block."
        ),
    }

    def prepare_environment(self) -> None:
        # transformers' hf-hub version gate, the same stub causal_lm_export installs at import time.
        mock_dep = types.ModuleType("dependency_versions_check")
        mock_dep.dep_version_check = lambda *args, **kwargs: None
        sys.modules.setdefault("transformers.dependency_versions_check", mock_dep)

    def load_model(self):
        from transformers import WhisperForConditionalGeneration

        print(f"Loading model from {self.model_dir}...")
        return WhisperForConditionalGeneration.from_pretrained(
            self.model_dir, torch_dtype=torch.float32
        ).eval()

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        from .exporter import _binding_kind

        model = self.load_model()
        extractor = load_feature_extractor(self.model_dir)
        cfg = model.config
        self.n_samples = int(extractor.n_samples)
        self.sample_rate = int(extractor.sampling_rate)
        self.n_audio_ctx = int(cfg.max_source_positions)
        self.d_model = int(cfg.d_model)
        self.max_target_positions = int(cfg.max_target_positions)
        self.prompt_constants = decoder_prompt_constants(model.generation_config, self.n_audio_ctx, cfg.vocab_size)

        mel = WhisperMelFrontend(extractor.n_fft, extractor.hop_length, np.array(extractor.mel_filters))

        # The trace length for the decoder. Free, and deliberately not 1: the graph must contain a real
        # token axis for the RangeDim below to make dynamic, and a length-1 trace gives coremltools a
        # size-1 axis it is entitled to fold away.
        trace_tokens = 8
        token_axis = ct.RangeDim(1, self.max_target_positions)
        decoder_inputs = [
            ct.TensorType(name="tokens", shape=(1, token_axis), dtype=np.int32),
            ct.TensorType(name="position_ids", shape=(1, token_axis), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, 1, token_axis, token_axis), dtype=np.float32),
            ct.TensorType(name="xa", shape=(1, self.n_audio_ctx, self.d_model), dtype=np.float32),
        ]
        self.decoder_bindings = tuple(
            (t.name, _binding_kind(t.name)) for t in decoder_inputs
        )

        return [
            ExportPhase(
                name="encoder",
                wrapper=_WhisperEncoderWrapper(model, mel).eval(),
                dummy_inputs=(torch.zeros(1, self.n_samples),),
                mil_inputs=[ct.TensorType(name="waveform", shape=(1, self.n_samples), dtype=np.float32)],
                # Every shape in this phase is a compile-time constant -- Whisper always sees exactly
                # 30 s of audio -- so this axis names the sample count without anything varying over it.
                root_axis="n_samples",
            ),
            ExportPhase(
                name="decoder",
                wrapper=_WhisperDecoderWrapper(model).eval(),
                dummy_inputs=(
                    torch.zeros((1, trace_tokens), dtype=torch.long),
                    torch.arange(trace_tokens).unsqueeze(0),
                    causal_mask(trace_tokens),
                    torch.zeros(1, self.n_audio_ctx, self.d_model),
                ),
                mil_inputs=decoder_inputs,
                root_axis=self.root_axis,
                # The self-attention blocks become cached ATTENTION nodes; the cross-attention blocks do
                # not, and that is correct rather than a miss. `fuse_loom_attention` anchors on the
                # `add(scores, mask)` that only a masked block has, and Whisper's cross-attention has no
                # mask at all -- it attends over the whole encoder output, every step. A cache there
                # would be wrong twice over: the K/V it would store are the encoder's, identical at
                # every step and already fully computed, and `layer` indices are assigned in occurrence
                # order, so a cached cross-attention block would consume cache slots the self-attention
                # blocks address.
                fuse_attention=True,
                kv_cache_size=self.max_target_positions,
            ),
        ]

    def hparams(self) -> dict:
        """What a HOST must know to call this driver at all.

        `n_samples` is the load-bearing one: Whisper is trained on exactly 30 s of audio and the encoder
        graph is built at that length, so a caller has to pad or trim to it before calling. Until this
        existed for the bespoke path, that number lived in a C++ test header
        (`test_e2e_whisper_lua_driver.cpp` sizes its input from a hardcoded `WhisperConfig`), which is
        precisely the "self-contained GGUF" claim being false (P4.0.8's first follow-up).

        `sample_rate` is here for two host jobs, both of which are otherwise a guess: the audio has to be
        resampled to it before `waveform` means anything, and a **timestamp token** has to be turned into
        a time. Whisper's timestamp tokens step by one encoder frame, so that step is
        `(n_samples / sample_rate) / n_audio_ctx` -- 0.02 s -- and a host that had to hardcode either
        number would be carrying a per-model constant, which is what these KVs exist to stop.
        """
        return {
            "n_samples": self.n_samples,
            "n_audio_ctx": self.n_audio_ctx,
            "n_text_ctx": self.max_target_positions,
            "sample_rate": self.sample_rate,
        }

    def backend_kwargs(self) -> dict:
        # The tokenizer travels with the model. `tokenizer_pre` is named rather than left to the hash
        # cascade because Whisper's tokenizer.json is not in llama.cpp's chkhsh table, so detection warns
        # and falls back to "qwen2" -- a pretokenizer whose digit and letter runs differ from the
        # byte-level GPT-2 one Whisper actually uses. Nothing in this family's own decode path notices
        # (ASR only ever maps ids BACK to text), which is exactly why a wrong value here would sit
        # unnoticed until something encoded text with it.
        return dict(tokenizer_dir=self.model_dir, tokenizer_pre="gpt-2", hparams=self.hparams())

    def driver_components(self) -> List:
        """Encoder once, then the decode loop -- two components and a header.

        The two calls are IR rather than text, which is what lets each be checked against the real
        traced topologies: the encoder's input names and output arity by `SubgraphCallComponent`, the
        decoder's by `PrefillDecodeLoop`'s own exact `inputs` link.

        The prompt is the exception, and it earns it: choosing a language is branching control flow
        (given / detect / neither), which is exactly what a `LuaFragment` is for -- the same split
        Parakeet's decode loop takes. Its own `run_subgraph_and_retain('decoder', ...)` detection call is
        parsed out of the text and declared against the traced topology regardless.
        """
        from .driver_components import (
            ExportConstants, LuaFragment, PrefillDecodeLoop, SubgraphCallComponent,
        )
        from .driver_ir import FieldAccess, Lit, OutputRef, Var

        return [
            LuaFragment(self.driver_script_path / "00_header.lua", top_level=True),
            ExportConstants(values=dict(self.prompt_constants)),
            SubgraphCallComponent(
                topology="encoder",
                # Retained, not bound to a local. The encoder emits `n_audio_ctx * d_model` floats --
                # 1.15M for whisper-small -- and the decode loop reads them at every step, so a Lua
                # table here would marshal them once per generated token. `OutputRef` below copies
                # backend-side instead (BACKLOG.md P4.0.12).
                outputs=(),
                retain=True,
                inputs={"waveform": FieldAccess("inputs", "waveform")},
                # A literal, not `#inputs.waveform`: this phase's sample count is a compile-time
                # constant (Whisper always sees exactly 30 s), so binding the axis to the length the
                # graph was built at states that, where reading the caller's array would imply the
                # encoder could run at some other length.
                axes={"n_samples": Lit(self.n_samples), "n_past": Lit(0)},
                note="Encoder: one fixed-shape pass over 30 s of audio -- mel frontend, conv stem, "
                     "transformer stack.",
            ),
            LuaFragment(
                self.driver_script_path / "01_prompt.lua",
                reads=("SOT", "LANG_LO", "LANG_HI", "TRANSCRIBE", "NO_TIMESTAMPS", "TS_LO", "TS_HI"),
                defines=("_prompt", "_language", "_gen0"),
            ),
            PrefillDecodeLoop(
                topology="decoder",
                bindings=self.decoder_bindings,
                inputs=tuple(name for name, _ in self.decoder_bindings),
                # The encoder's output, held constant across every step. Everything else the loop needs
                # it already computes: tokens from the previous step, positions and the causal mask from
                # n_tokens/n_past.
                bound={"xa": OutputRef("encoder")},
                # The prefix the fragment above built, not `inputs.tokens`: this driver owns its own
                # prompt, so a caller passes audio and at most a language.
                prompt=Var("_prompt"),
                # The forced opening timestamp, when there is one: the model chose it, so
                # it is part of what this call generated even though it is fed back in.
                generated_prefix=Var("_gen0"),
            ),
        ]


def _hf_model_type(path: Path) -> Optional[str]:
    """An HF-style directory's own `config.json`'s `model_type`, or None if `path` isn't one. Never
    raises: `detect()` runs against unidentified paths by construction."""
    config_path = path / "config.json"
    if not path.is_dir() or not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return config.get("model_type") if isinstance(config, dict) else None


def _is_whisper(path: Path) -> bool:
    """A real structural check (BACKLOG.md P3.2): an HF directory declaring `model_type == "whisper"`.

    Deliberately claims distil-whisper and every fine-tune too -- they are the same architecture with
    fewer decoder layers, which this family reads off the checkpoint rather than assuming. It does not
    collide with `hf-causal-lm`'s fallback, which requires a `*ForCausalLM` architecture entry and so
    rejects `WhisperForConditionalGeneration` by construction.
    """
    return _hf_model_type(path) == "whisper"


def _build_whisper(path: Path, output_path: str) -> ASRWhisperExportConfig:
    return ASRWhisperExportConfig(model_dir=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=ASRWhisperExportConfig,
        recognizers=[ModelRecognizer(name="whisper", detect=_is_whisper, build_config=_build_whisper)],
    ))
