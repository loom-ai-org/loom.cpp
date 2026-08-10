"""The audio-encoder + projector + causal-LM family (`EXPORT-ROADMAP.md` R5's family 3) -- BACKLOG.md
P4.3. The single largest coverage group on the roadmap: ~19 converters, ~36 models.

**What a family-3 model is, and why it is a composition rather than an architecture.** Every member is
three pieces the exporter already knows how to trace -- an audio encoder (family 1 or 2), a small
projector, and a causal LM (`causal_lm_export`) -- wired together by one idea: the projector's output
rows are *substituted into the LM's input embedding sequence* at the positions a placeholder token
occupies. Nothing about the LM changes; it reads embeddings it did not produce. So this module declares
phases and a driver shape, and the per-checkpoint leaves (`qwen3_asr_export`) declare only how their
checkpoint is loaded and where its three pieces live -- the same loader/template split P4.2 drew for
transducers.

**The finding that made this cheap: the prompt needs no concatenation anywhere.**
The obvious reading of "inject audio embeddings into the prompt" is that something must build one
`inputs_embeds` tensor out of text embeddings and audio embeddings -- which would need a backend-side
concatenation of two retained tensors, an engine op that does not exist and that `OutputStore` has no
shape for. It is not needed. Attention is causal and the decoder is KV-cached, so a call at
`n_past = k` over `n` new rows writes cells `[k, k+n)` and attends over `[0, k+n)` -- which means
feeding a prompt as N successive cached calls is *the same arithmetic* as feeding it concatenated. The
driver therefore runs one cached call per segment:

    text prefix   -> embed(tokens)              n_past 0
    audio         -> the encoder's own output   n_past 9
    text suffix   -> embed(tokens)              n_past 152   <- the decode loop's first iteration

Measured on Qwen3-ASR-0.6B against HF: segmented and concatenated prefill agree to 2.3e-04 on hidden
states whose absmax is 95.7 (2.4e-6 relative, float32 reduction-order noise) and pick the same first
token. `PromptSegments` below is that walk, and it is the whole of the "embedding-injection driver"
the roadmap asked for.

**The LM is split at the head, and that is a performance fact rather than a style choice.** The decoder
phase emits hidden states; a separate `lm_head` phase turns them into logits. Keeping the head in the
decoder graph would make the *audio* segment compute logits for every one of its rows -- 143 rows x
1024 hidden x 151936 vocab = 22 GFLOP, for rows whose argmax nobody reads. The driver runs `lm_head`
only where a token is actually needed, which is the loop.

**What a leaf supplies, and where the boundary actually is.** `load_model()`, `language_model()`,
`lm_head()`, `feature_extractor()`, `prompt_segment_constants()`, `audio_rows()` -- and, the two that
the SECOND leaf moved here, `audio_encoder()` and `audio_geometry()`. `mel_frontend()` is the one
softer hook: it has a default, because every family-3 extractor computes the same log-mel, and Granite
overrides only where its checkpoint writes those numbers down.

**A chunk-padded waveform costs nothing, and that took a second input and four separate statements**
(BACKLOG.md P4.3d/P4.3e). Every encoder here consumes audio in indivisible chunks -- one second for
Qwen3-ASR, twelve for Granite Speech -- so a host zero-pads up to the next boundary, and until P4.3e
that padding was read as speech. The `encoder` phase now takes `valid_samples` alongside the waveform;
a leaf's module masks it out of its attention keys, its convolutions and its projector windows;
`audio_rows` stops the prompt at the rows the real audio produced; and `WaveformValidLength` writes the
frontend's own STFT reflection over the head of the padding, which is what turns "close" into
bit-exact. `_check_padding_invariance` holds every leaf to the result on every export.

The first version of this module built the audio encoder itself, on the reasoning that a family-3
encoder is a family-3 encoder. Granite Speech disproved that: Qwen3-ASR's is a Qwen3-Omni
window-attention stack over one-second chunks of 128-bin mel, and Granite's is a **conformer** over
160-bin features feeding a **Q-Former** that turns each 15-frame window into
`window_size // downsample_rate` query rows. They share the log-mel frontend, and the
`(samples_per_chunk, frames_per_chunk)` contract, and nothing else -- so the encoder module is supplied
by the leaf and the contract is *checked* here, against the traced module's real output.

That contract holding across two encoders and two projectors this different is the substantive claim
this template makes, and it is why the second leaf needed no new driver component at all. Everything
else -- the phase list, their shapes and axes, the component list, the KV geometry -- is here.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .decomposition import Decomposition, MultiPhase
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase
from .spec_protocol import Unchecked


class LogMelFrontend(nn.Module):
    """A Whisper-style log-mel spectrogram as a traceable `nn.Module`, so the exported model takes a
    **waveform** and the GGUF stays self-contained.

    Deliberately a second implementation rather than an import of `whisper_export.WhisperMelFrontend`,
    and the difference is one line: that one is built at a fixed `n_samples` and this one is not, so
    the STFT here traces with a genuinely dynamic sample axis. The arithmetic is otherwise identical --
    Hann-windowed STFT, drop the final frame, power spectrum, filterbank, log10 with a 1e-10 floor,
    clamp to 8 dB below the clip's own maximum, then `(x + 4) / 4` -- because Qwen3-ASR's own
    `_torch_extract_fbank_features` is Whisper's line for line. Where they genuinely differ is the
    filterbank's *provenance*: Whisper reads `mel_filters` off `preprocessor_config.json`, and this
    family's extractor computes it with `mel_filter_bank(...)` at construction. Both arrive here as an
    array, which is why that difference does not reach this class.

    **Granite Speech shares every line of this, which was not obvious.** Its extractor is a torchaudio
    `MelSpectrogram` followed by `clip_(1e-10).log10_()`, a global `amax`, `maximum(x, mx - 8)` and
    `.div_(4).add_(1)` -- and `x/4 + 1` *is* Whisper's `(x + 4)/4`. Two differences reach this class
    and both are constructor arguments: the analysis window is shorter than the FFT (400 against 512,
    hence `win_length`), and the filterbank comes off a torchaudio `MelScale` rather than a
    transformers extractor -- which is again only an array. Reimplementing torchaudio's transform with
    `torch.stft` is required regardless of any of that: its own `complex_shape` cannot be lowered,
    the same wall `gigaam_export._TraceableMelSpectrogram` hit.

    **The final-frame drop is Granite's too, given the chunk contract.** Whisper drops the last STFT
    frame unconditionally; Granite computes `L // hop + 1` frames and drops the last one only when
    that count is odd. Under a whole number of 192000-sample chunks the count is `1200k + 1`, always
    odd, so the conditional fires every time and removes the same frame this line does.

    **The global maximum is why this cannot be made per-chunk.** `torch.maximum(log_spec,
    log_spec.max() - 8)` makes every output element depend on the loudest bin in the whole clip, so a
    host cannot stream this frontend without changing the numbers.
    """

    def __init__(self, n_fft: int, hop_length: int, mel_filters: np.ndarray,
                 win_length: Optional[int] = None):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        # Defaults to the FFT size, which is Whisper's and Qwen3-ASR's geometry. `torch.stft` centres a
        # shorter window inside the FFT itself, and `center=True` still pads by `n_fft // 2`, so a
        # shorter window changes the window buffer and nothing else about the frame grid.
        self.win_length = int(win_length) if win_length is not None else self.n_fft
        self.register_buffer("window", torch.hann_window(self.win_length))
        # The extractor stores `(n_freq, n_mels)`; the matmul below wants `(n_mels, n_freq)`.
        self.register_buffer(
            "filters", torch.from_numpy(np.asarray(mel_filters, dtype=np.float32)).T.contiguous()
        )

    def forward(self, waveform):
        stft = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=True, return_complex=True,
        )
        # Magnitude BEFORE the final-frame slice, which is the same arithmetic on one fewer element and
        # the only order that converts: coremltools' complex dialect has no `slice_by_index` over a
        # complex tensor. Recorded once already, in `whisper_export.WhisperMelFrontend`.
        magnitudes = (stft.abs() ** 2)[..., :-1]
        mel_spec = self.filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0


class _EmbedWrapper(nn.Module):
    """`token ids -> input embeddings`. Its own phase because the driver needs to build embeddings for
    text the LM has not seen yet -- both the prompt's text segments and, every step, the token the
    previous step produced."""

    def __init__(self, language_model):
        super().__init__()
        self.embed_tokens = language_model.embed_tokens

    def forward(self, tokens):
        return self.embed_tokens(tokens)


class _DecoderWrapper(nn.Module):
    """`(inputs_embeds, position_ids, attention_mask) -> last hidden state`.

    Takes embeddings rather than token ids, which is the one change family 3 makes to the causal-LM
    trace and the reason the same graph serves both a text segment and an audio segment: by the time
    the decoder runs, an audio row and a text row are the same kind of thing.

    `position_ids` and `attention_mask` are passed explicitly for the reason
    `causal_lm_export._causal_mask` documents at length -- an already-prepared 4D mask short-circuits
    transformers' own mask builder, which would otherwise derive a key length from a Python-level shape
    that tracing bakes in. Both names are already in `driver_components`' host-computed sets, so the
    driver fills them from `n_tokens`/`n_past` without this family declaring anything.

    **`position_ids` rather than `cache_position`, and that distinction is load-bearing here where it
    is not for `causal_lm_export`.** Handed `inputs_embeds` and an already-built 4D mask, this model
    consumes `cache_position` nowhere: it is used to *derive* position ids and to build a mask, and
    both jobs are already done. The trace therefore pruned it, and pruning it is not a tidy-up -- it
    means the rotary embedding folded to the eight positions the trace ran at, so every call at a
    different `n_past` would silently rotate by the wrong angle. Passing the positions the model
    actually indexes with is what keeps the axis genuinely dynamic; the exporter's own "supplies an
    input it does not declare" link is what caught it.
    """

    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model

    def forward(self, inputs_embeds, position_ids, attention_mask):
        return self.language_model(
            inputs_embeds=inputs_embeds, position_ids=position_ids,
            attention_mask=attention_mask, use_cache=False,
        ).last_hidden_state


class _LMHeadWrapper(nn.Module):
    """`hidden states -> logits`. Split out of the decoder so the prompt's audio segment does not pay
    for a vocabulary-wide projection of rows nobody reads; see the module docstring."""

    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden):
        return self.lm_head(hidden)


def split_prompt_on_audio(token_ids: List[int], audio_id: int) -> tuple:
    """`(prefix, suffix)` -- a rendered prompt's token ids either side of its audio placeholder run.

    Both leaves so far reach their two text segments this way, and both reach them from a rendered
    chat template rather than a transcription of one, so a template change moves the constants instead
    of silently disagreeing with them. What differs between the leaves is only how the template is
    rendered -- Qwen3-ASR's takes an audio *item* through the processor, Granite Speech's is a text
    template whose placeholder the caller writes -- so that stays in the leaves and this does not.

    The two ways this can be wrong are checked here, because neither would fail later: a template that
    renders no placeholder leaves the encoder's output with no position to occupy, and one that
    interleaves text inside the placeholder run means the audio is not one contiguous segment, which is
    the only shape `PromptSegments` feeds.
    """
    if audio_id not in token_ids:
        raise ValueError(
            f"this checkpoint's chat template rendered no audio placeholder (id {audio_id}) at all, so "
            f"there is no position for the encoder's output to occupy. The template is what this "
            f"family reads the prompt's shape from."
        )
    first = token_ids.index(audio_id)
    last = len(token_ids) - 1 - token_ids[::-1].index(audio_id)
    if not all(token == audio_id for token in token_ids[first:last + 1]):
        raise ValueError(
            f"this checkpoint's chat template puts non-audio tokens between its audio placeholders "
            f"(ids {first}..{last}), so the audio does not occupy one contiguous run of prompt "
            f"positions. PromptSegments feeds the encoder's output as a single segment."
        )
    return ([int(token) for token in token_ids[:first]],
            [int(token) for token in token_ids[last + 1:]])


def causal_mask(seq_len: int) -> torch.Tensor:
    """A 4D additive causal mask, the form transformers passes straight through to attention. Same
    tensor and same reason as `causal_lm_export._causal_mask` and `whisper_export.causal_mask`."""
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


@dataclass
class BaseSpeechLMExportConfig(BaseMultiPhaseModelExportConfig):
    """An audio encoder + projector + causal LM as four traced phases in one GGUF.

    Phases, and why there are four rather than two:

    * `encoder` -- waveform *and how much of it is real audio* to projected audio embeddings, dynamic
      over audio length, uncached. The second input is what keeps the caller's chunk padding out of the
      rows the language model reads (BACKLOG.md P4.3e); `audio_encoder` states the contract it imposes
      on a leaf, and `_check_padding_invariance` is what holds a leaf to it.
    * `embed` -- token ids to embeddings. The driver needs this separately from the decoder because it
      builds the prompt out of pieces, and because every decode step must embed the token the previous
      step chose.
    * `decoder` -- embeddings to hidden states, KV-cached. The only phase with `fuse_attention`.
    * `lm_head` -- hidden states to logits, uncached, run only where a token is needed.

    Like `ASRWhisperExportConfig`, this is a `MultiPhase` config rather than a fifth `Decomposition`:
    the orchestration is still N independently traced phases plus a component list, and what differs is
    carried by fields on the pieces that own them.
    """

    model_dir: str = ""
    architecture: str = "speech-lm"
    output_path: str = "speech_lm_mil.gguf"
    root_axis: str = "n_tokens"
    driver_script_path: Path = Path(__file__).resolve().parent / "speech_lm_driver"
    decomposition: Decomposition = field(default_factory=MultiPhase)
    # The KV cache capacity, in tokens. A prompt here is dominated by AUDIO rows -- 13 per second of
    # audio for Qwen3-ASR -- so this bounds audio length as much as text length, which is why it is not
    # simply the checkpoint's `max_position_embeddings` (65536 would allocate a cache far larger than
    # any edge target has memory for).
    max_seq_len: int = 4096
    # The longest audio the encoder's RangeDim admits, in chunks. Bounds nothing about the checkpoint:
    # it is the upper end of the dynamic axis, and the decode side is bounded by `max_seq_len` anyway.
    max_audio_chunks: int = 30

    # Read off the checkpoint in `phases()`, which is the only moment the model and its feature
    # extractor are both in hand.
    sample_rate: Optional[int] = field(default=None, init=False, repr=False)
    samples_per_chunk: Optional[int] = field(default=None, init=False, repr=False)
    frames_per_chunk: Optional[int] = field(default=None, init=False, repr=False)
    hidden_size: Optional[int] = field(default=None, init=False, repr=False)
    reflect_samples: Optional[int] = field(default=None, init=False, repr=False)
    decoder_bindings: tuple = field(default=(), init=False, repr=False)
    prompt_constants: dict = field(default_factory=dict, init=False, repr=False)

    __unchecked__ = {
        "model_dir": Unchecked(
            "path to the HF directory, already established by the recognizer's own detect(), which "
            "reads its config.json. The leaf's from_pretrained raises on anything it cannot load."
        ),
        "architecture": Unchecked("the GGUF's own architecture string; it names this export, and there "
                                  "is no second authority to compare it against"),
        "output_path": Unchecked("where to write. A caller's choice, not a claim about the model."),
        "root_axis": Unchecked("checked by each ExportPhase's own Axis link, which is where the value "
                               "is actually used"),
        "driver_script_path": Unchecked("the directory of hand-written .lua fragments; their CONTENTS "
                                        "are parsed and cross-checked by LuaFragment"),
        "decomposition": Unchecked("MultiPhase by construction -- see the class docstring"),
        "max_seq_len": Unchecked(
            "the KV cache capacity and the decoder's RangeDim upper bound. Deliberately NOT checked "
            "against the checkpoint's max_position_embeddings, for the reason "
            "LMCausalModelExportConfig.max_seq_len records: exporting a shorter context than the "
            "architecture allows is a legitimate choice, and here it is the only one -- this "
            "checkpoint declares 65536."
        ),
        "max_audio_chunks": Unchecked(
            "the encoder RangeDim's upper bound, in chunks of one second. A caller's ceiling, not a "
            "property of the checkpoint, which has no maximum audio length of its own."
        ),
        "sample_rate": Unchecked("READ off the checkpoint's own feature extractor in phases()"),
        "samples_per_chunk": Unchecked(
            "READ off the feature extractor and audio config in phases() as `hop_length * n_window * "
            "2`, the sample count whose mel is exactly one encoder chunk -- and CROSS-CHECKED there "
            "against the encoder's real output row count, because it is the number a host pads to and "
            "a wrong one would silently mis-shape every prompt."
        ),
        "frames_per_chunk": Unchecked("same -- `audio_config.max_position_embeddings`, the post-CNN "
                                      "frame count one chunk becomes, cross-checked with it"),
        "hidden_size": Unchecked("same -- the LM's own `hidden_size`, which is what the projector's "
                                 "output width must equal for a row to be substitutable at all"),
        "reflect_samples": Unchecked(
            "READ off the frontend this family builds, as `n_fft // 2` -- the amount "
            "`torch.stft(center=True)` reflects the signal by at each end, and therefore how much of "
            "a caller's zero padding has to be the mirror instead. See "
            "`driver_components.WaveformValidLength`, and the padding-invariance check in phases() "
            "for the property it exists to make true."
        ),
        "decoder_bindings": Unchecked(
            "(name, kind) per decoder input, derived in phases() from the SAME mil_inputs list the "
            "trace is declared with, through `exporter._binding_kind` -- so the driver cannot disagree "
            "with the trace about the order or the names."
        ),
        "prompt_constants": Unchecked(
            "READ off the checkpoint's own tokenizer and config in phases() (the audio placeholder id, "
            "the two eos ids, and the token ids the chat template's own prefix and suffix tokenize "
            "to), never declared -- see `prompt_segment_constants`, which resolves them by TEXT "
            "through the checkpoint's tokenizer rather than hardcoding a number."
        ),
    }

    # -- hooks a leaf supplies -----------------------------------------------------------------------

    def load_model(self):
        raise NotImplementedError

    def language_model(self, model):
        raise NotImplementedError

    def lm_head(self, model):
        raise NotImplementedError

    def feature_extractor(self):
        raise NotImplementedError

    def mel_frontend(self, extractor) -> LogMelFrontend:
        """The log-mel frontend `audio_encoder` is handed, built from THIS checkpoint's own extractor.

        Overridable rather than a hook a leaf must implement, because the default is what a
        transformers audio extractor spells: `n_fft`, `hop_length` and a `mel_filters` array as
        attributes, with the analysis window the full FFT. Granite Speech's extractor is a torchaudio
        `MelSpectrogram` module instead, so its filterbank and its shorter `win_length` are reached
        through different attribute names -- which is a difference in where the numbers are written
        down, not in the arithmetic, and is why the override is three lines and `LogMelFrontend` is
        still one class.
        """
        return LogMelFrontend(extractor.n_fft, extractor.hop_length,
                              np.asarray(extractor.mel_filters))

    def audio_encoder(self, model, mel: LogMelFrontend) -> nn.Module:
        """`(waveform, valid_samples) -> projected audio embeddings`, as ONE traceable module: this
        checkpoint's audio tower and its projector, with `mel` in front of them.

        **A leaf's job, and finding that out is what the second leaf was for.** The first version of
        this template built the encoder itself, on the reasoning that a family-3 encoder is a family-3
        encoder. It is not: Qwen3-ASR's is a Qwen3-Omni window-attention stack over one-second chunks
        of 128-bin mel (`qwen3_asr_export.WindowedAudioEncoder`), and Granite Speech's is a conformer
        over 160-bin features feeding a Q-Former. They share the log-mel frontend and the contract
        below, and nothing else -- so the module is supplied and the contract is enforced here.

        **`valid_samples` is a `(1,)` f32 tensor: how many of the waveform's samples are real audio**
        (BACKLOG.md P4.3e). The waveform is padded up to a whole chunk, and every encoder in this
        family has places where that padding leaks into rows the language model reads -- an attention
        key, a depthwise convolution's neighbourhood, a projector window, a convolutional stem that
        saw log-mel-of-silence where the checkpoint's extractor pads its features with literal zeros.
        A leaf's module has to keep the padding out of all of them, IN GRAPH, because the count is a
        run-time value: the raw sample count is the input rather than any per-checkpoint frame count so
        that this declaration is the same for every leaf, and the arithmetic that turns samples into
        that checkpoint's frames stays with the checkpoint's own encoder.

        Both the geometry and that masking are checked in `phases()` against the real thing rather than
        trusted: `audio_geometry`'s chunk count times its rows-per-chunk must be the row count the
        module actually emits, those rows must be the language model's own width, and appending whole
        chunks of silence to a waveform must not move a single row the driver will read.
        """
        raise NotImplementedError

    def audio_geometry(self, model, extractor) -> tuple:
        """`(samples_per_chunk, frames_per_chunk)` -- the two numbers the DRIVER does arithmetic with.

        A "chunk" is the unit of audio this encoder consumes indivisibly, and the unit a host pads up
        to: one second for Qwen3-ASR (`hop_length * n_window * 2`), **twelve** for Granite Speech
        (`lcm(context_size, window_size)` encoder frames, because its conformer's blocks and its
        Q-Former's windows must both divide the sequence -- a leaf can be this coarse and the template
        does not care). `frames_per_chunk` is how many prompt positions one chunk becomes -- 13 and
        120, which is also why "a chunk" is not a duration this family has any opinion about. The
        driver's audio segment is `floor(#waveform / samples_per_chunk) * frames_per_chunk` rows, which
        is the only place these numbers are used and the reason both are published as hparams.

        That the same two numbers describe both encoders is the substantive claim this template makes,
        and it is why a second leaf needed no new driver component at all.
        """
        raise NotImplementedError

    def audio_rows(self, valid_samples):
        """How many prompt positions the audio segment really occupies, as a `driver_ir` expression
        over `valid_samples` -- the caller's UNPADDED sample count (BACKLOG.md P4.3d).

        **Why this is not `chunks * frames_per_chunk`.** That product is what the padded waveform
        produces, and the encoder's contract is that a host pads up to a whole chunk -- so the rows
        past the caller's real audio are not audio at all. How many rows a given number of samples
        really becomes is a per-checkpoint formula: Granite Speech rounds up to a Q-Former window,
        Qwen3-ASR walks three stride-2 convs over its final partial chunk.

        **A leaf must state it -- there is no safe default any more, and P4.3e is why.** This used to
        fall back to the padded product, on the reasoning that a leaf without a formula still exported
        a model that merely read a little silence. That reasoning ended when the encoder learned to
        mask: the rows past `valid_samples` are now rows whose attention keys were all masked and whose
        features were zeroed, so they are not a reading of silence, they are nothing at all. A leaf
        that cannot say how many rows its audio produces cannot use this template.

        `phases()` checks whatever a leaf returns here against the geometry the same leaf declares --
        at a whole number of chunks the two must agree, since there is no padding to disagree about.

        **Every division must be `floordiv`.** Lua's `/` is float division, so a whole number of chunks
        would still produce `143.0` where the engine binds an integer extent -- and `rows` reaches
        `std::llround` on the engine side, where a fractional count would round rather than fail.
        """
        raise NotImplementedError

    def prompt_segment_constants(self, model) -> dict:
        """`{name: token id}` the driver builds its prompt from, read off this checkpoint's own
        tokenizer and config. A leaf's job because the chat template is a checkpoint fact."""
        raise NotImplementedError

    # -- the template --------------------------------------------------------------------------------

    def _valid_samples_expr(self):
        """`inputs.audio_samples or #inputs.waveform` -- the caller's real audio length, defaulting to
        the whole waveform.

        Its own method rather than an inline expression because `WaveformValidLength` binds it to a
        local that three separate things then read -- the encoder's `valid_samples` input, the mirror
        loop's bound, and the prompt's row count -- and "the caller's real length" should have exactly
        one definition. Same `or`-with-a-default idiom as `max_new_tokens`/`eos_token`, and for the
        same reason: a driver input a host may omit.
        """
        from .driver_ir import BinOp, FieldAccess, Len

        return BinOp("or", FieldAccess("inputs", "audio_samples"),
                     Len(FieldAccess("inputs", "waveform")))

    def _check_row_formula(self, trace_chunks: int) -> None:
        """`audio_rows` and `audio_geometry` must agree wherever there is no padding to disagree about.

        A leaf states two things that overlap: how many rows one whole chunk becomes
        (`frames_per_chunk`, already checked against the traced encoder above) and how many rows an
        arbitrary sample count becomes (`audio_rows`, a formula this template cannot check against the
        model at all, because the encoder's output is padded and its own row count is the padded one).
        At an exact multiple of `samples_per_chunk` the second must be the first times the chunk count
        -- there is no partial window, no partial block and no rounding -- so this is the one place the
        two declarations can be held against each other, and it catches an off-by-one in the formula's
        rounding, a wrong hop length and a `/` that should have been a `floordiv`.

        Evaluated through `driver_ir.evaluate` on a node built by the very method `driver_components()`
        builds the driver's own from, so what is checked is what ships rather than a Python restatement
        of it (BACKLOG.md P4.3e). The environment is empty because the expression's only free term is
        the sample count, substituted here as a literal where the driver substitutes its local.
        """
        from .driver_ir import Lit, evaluate

        for chunks in range(1, trace_chunks + 1):
            samples = self.samples_per_chunk * chunks
            rows = evaluate(self.audio_rows(Lit(samples)), {})
            if rows != chunks * self.frames_per_chunk:
                raise ValueError(
                    f"`audio_rows` says {samples} samples is {rows} prompt positions, but that is "
                    f"exactly {chunks} whole chunk(s) and `audio_geometry` says a chunk is "
                    f"{self.frames_per_chunk} rows, i.e. {chunks * self.frames_per_chunk}. At a whole "
                    f"number of chunks there is no padding for the two to disagree about, so one of "
                    f"these two declarations does not describe this checkpoint."
                )

    def _check_padding_invariance(self, encoder: nn.Module) -> None:
        """Appending whole chunks of silence to a waveform must not move a row the driver will read.

        **This is P4.3e's property, stated as a check the export runs.** A host pads up to a whole
        chunk, so the encoder always sees some audio that is not audio; `valid_samples` is what tells
        it where the real audio stopped, and a leaf's module is supposed to keep the padding out of
        every place it could reach a row -- an attention key, a depthwise convolution's neighbourhood,
        a projector window, a convolutional stem fed log-mel-of-silence where the checkpoint's own
        extractor pads with zeros. None of that is visible in a shape, and an encoder that ignored
        `valid_samples` entirely would export, run, and transcribe *almost* correctly -- which is the
        failure mode P4.2 already paid for once.

        So: the same audio at the same declared length, once in a waveform that exactly fits and once
        in one padded by two further chunks -- padded the way the DRIVER pads, mirror and all, since
        that repair is half of what makes the two equal. The audio is noise rather than zeros
        deliberately: on silence every frame is the same frame and a leak leaks nothing, so a zero
        probe would pass this without meaning it.

        The tolerance is float-accumulation noise, not a padding budget. A correctly masked encoder's
        rows agree to the last few bits -- the padded run's extra attention keys contribute exactly
        zero, and adding zero to a float sum is exact -- so both leaves land at 0.0e+00 here, and the
        failures this catches are orders of magnitude away (1.7e-01 for Granite Speech before the
        masking, 9.0e-05 for Qwen3-ASR from dropping the mirror alone).
        """
        from .driver_ir import Lit, evaluate

        chunks, extra = 2, 2
        valid = self.samples_per_chunk * chunks
        rows = evaluate(self.audio_rows(Lit(valid)), {})
        generator = torch.Generator().manual_seed(0)
        audio = torch.randn(1, valid, generator=generator) * 0.1
        padded = torch.cat([audio, torch.zeros(1, self.samples_per_chunk * extra)], dim=1)
        # `driver_components.WaveformValidLength`'s edge repair, in torch: the frontend's STFT reflects
        # by `n_fft // 2` at the signal's end, so a run whose audio simply stops there and a run whose
        # audio is followed by a caller's zeros disagree on the frames that straddle it -- which is a
        # real difference in the encoder's input, not a leak, and not what this check is asking about.
        for offset in range(1, (self.reflect_samples or 0) + 1):
            padded[0, valid + offset - 1] = audio[0, valid - 1 - offset]
        declared = torch.tensor([float(valid)])
        with torch.inference_mode():
            tight = encoder(audio, declared)[:rows]
            loose = encoder(padded, declared)[:rows]
        drift = float((tight - loose).abs().max())
        scale = max(float(tight.abs().max()), 1e-6)
        if drift > 1e-5 * scale:
            raise ValueError(
                f"this checkpoint's encoder is not padding-invariant: the same {valid} samples of "
                f"audio produce rows that differ by {drift:.3e} (reference absmax {scale:.3f}) "
                f"depending on whether {extra} further chunk(s) of silence follow them. A host pads up "
                f"to a whole chunk, so those rows are what the language model would read -- the module "
                f"has to keep the padding out of its attention keys, its convolutions and its "
                f"projector windows, using the `valid_samples` input it is given (BACKLOG.md P4.3e)."
            )

    def phases(self) -> List[ExportPhase]:
        import coremltools as ct

        from .exporter import _binding_kind

        model = self.load_model()
        extractor = self.feature_extractor()
        language_model = self.language_model(model)

        self.sample_rate = int(extractor.sampling_rate)
        self.samples_per_chunk, self.frames_per_chunk = (
            int(value) for value in self.audio_geometry(model, extractor)
        )
        self.hidden_size = int(language_model.config.hidden_size)
        self.prompt_constants = dict(self.prompt_segment_constants(model))

        mel = self.mel_frontend(extractor)
        self.reflect_samples = int(mel.n_fft) // 2
        encoder = self.audio_encoder(model, mel).eval()

        # The two numbers a host does arithmetic with, checked against the graph that produces them
        # rather than trusted from the leaf. A wrong `frames_per_chunk` would make every prompt place
        # its text suffix at the wrong `n_past` -- silently, since the shapes would still agree.
        #
        # This check is the whole reason `audio_geometry` may be a declaration at all: it is answered
        # against the traced module's real output on a known number of chunks, so a leaf that reads the
        # wrong config field is caught here rather than in a transcript.
        trace_chunks = 4
        probe = torch.zeros(1, self.samples_per_chunk * trace_chunks)
        probe_valid = torch.tensor([float(self.samples_per_chunk * trace_chunks)])
        with torch.inference_mode():
            probe_out = encoder(probe, probe_valid)
        rows, width = int(probe_out.shape[0]), int(probe_out.shape[1])
        if rows != trace_chunks * self.frames_per_chunk:
            raise ValueError(
                f"this checkpoint's encoder turns {trace_chunks} chunks of {self.samples_per_chunk} "
                f"samples into {rows} rows, but `audio_geometry` says one chunk is "
                f"{self.frames_per_chunk} rows, i.e. {trace_chunks} x {self.frames_per_chunk} = "
                f"{trace_chunks * self.frames_per_chunk}. The driver computes the audio segment's "
                f"length from those two numbers, so one of them is not what this checkpoint does."
            )
        if width != self.hidden_size:
            raise ValueError(
                f"the projector emits {width}-wide rows but the language model's hidden size is "
                f"{self.hidden_size}. An audio row is substituted into the LM's embedding sequence, so "
                f"the two must be equal for this composition to mean anything."
            )
        self._check_row_formula(trace_chunks)
        self._check_padding_invariance(encoder)

        trace_tokens = 8
        token_axis = ct.RangeDim(1, self.max_seq_len)
        decoder_inputs = [
            ct.TensorType(name="inputs_embeds", shape=(1, token_axis, self.hidden_size),
                          dtype=np.float32),
            ct.TensorType(name="position_ids", shape=(1, token_axis), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, 1, token_axis, token_axis),
                          dtype=np.float32),
        ]
        self.decoder_bindings = tuple((t.name, _binding_kind(t.name)) for t in decoder_inputs)

        return [
            ExportPhase(
                name="encoder",
                wrapper=encoder,
                dummy_inputs=(probe, probe_valid),
                mil_inputs=[
                    ct.TensorType(
                        name="waveform",
                        shape=(1, ct.RangeDim(self.samples_per_chunk,
                                              self.samples_per_chunk * self.max_audio_chunks)),
                        dtype=np.float32,
                    ),
                    # How many of those samples are real audio, as a one-element tensor rather than a
                    # scalar: the engine marshals a driver's Lua array into a declared input by
                    # element count, and `(1,)` is the shape a one-element array is. Float, because
                    # every use of it in every leaf's encoder is arithmetic against a `torch.arange`
                    # over a dynamic axis -- and a sample count is far inside f32's exact-integer
                    # range (2^24 is 17 minutes at 16 kHz, where `max_seq_len` caps the prompt long
                    # before that).
                    ct.TensorType(name="valid_samples", shape=(1,), dtype=np.float32),
                ],
                # Raw audio, never a token count -- the same distinction the NeMo family draws.
                root_axis="n_samples",
            ),
            ExportPhase(
                name="embed",
                wrapper=_EmbedWrapper(language_model).eval(),
                dummy_inputs=(torch.zeros((1, trace_tokens), dtype=torch.long),),
                mil_inputs=[ct.TensorType(name="tokens", shape=(1, ct.RangeDim(1, self.max_seq_len)),
                                          dtype=np.int32)],
                root_axis=self.root_axis,
            ),
            ExportPhase(
                name="decoder",
                wrapper=_DecoderWrapper(language_model).eval(),
                dummy_inputs=(
                    torch.zeros(1, trace_tokens, self.hidden_size),
                    torch.arange(trace_tokens).unsqueeze(0),
                    causal_mask(trace_tokens),
                ),
                mil_inputs=decoder_inputs,
                root_axis=self.root_axis,
                # The only cached phase. The encoder is a single full-sequence pass and must not be
                # cached; `embed` and `lm_head` have no attention at all.
                fuse_attention=True,
                kv_cache_size=self.max_seq_len,
            ),
            ExportPhase(
                name="lm_head",
                wrapper=_LMHeadWrapper(self.lm_head(model)).eval(),
                dummy_inputs=(torch.zeros(1, trace_tokens, self.hidden_size),),
                mil_inputs=[ct.TensorType(name="hidden",
                                          shape=(1, ct.RangeDim(1, self.max_seq_len),
                                                 self.hidden_size),
                                          dtype=np.float32)],
                root_axis=self.root_axis,
            ),
        ]

    def hparams(self) -> dict:
        """What a HOST must know to call this driver at all.

        `samples_per_chunk` is the load-bearing one: the encoder's contract is a whole number of chunks
        with every frame valid (each leaf's `audio_encoder` documents its own), so a caller pads up to a
        multiple of this. `sample_rate` is what makes that padding mean a duration, and
        `frames_per_chunk` is how many prompt positions one chunk of audio occupies -- which the host
        needs to size the audio segment.
        """
        return {
            "sample_rate": self.sample_rate,
            "samples_per_chunk": self.samples_per_chunk,
            "frames_per_chunk": self.frames_per_chunk,
            "n_ctx": self.max_seq_len,
        }

    def backend_kwargs(self) -> dict:
        return dict(tokenizer_dir=self.model_dir, hparams=self.hparams())

    def driver_components(self) -> List:
        """Encoder once, then the prompt as segments, then the decode loop.

        Four components and a header. `PromptSegments` walks the text/audio/text structure with a
        running `n_past`, and hands the loop both the segment it should start from and the `n_past` it
        should start at -- so the final text segment IS the loop's first iteration, exactly as a plain
        causal LM's prefill is.
        """
        from .driver_components import (
            ExportConstants, LuaFragment, PrefillDecodeLoop, PromptSegments, SubgraphCallComponent,
            WaveformValidLength,
        )
        from .driver_ir import ArrayLit, FieldAccess, Len, Lit, OutputRef, Var

        # The caller's real audio length, before it padded up to a whole chunk. Optional, and the
        # fallback is the padded length itself -- so a host that does not know or does not care says
        # "all of it", and one that passes it gets a prompt with no padding rows in it and an encoder
        # that never read the padding at all (BACKLOG.md P4.3d, P4.3e). Bound to a local by
        # `WaveformValidLength`, which also repairs the padding's head so the frontend's own STFT
        # reflection is what straddles the real end.
        waveform = FieldAccess("inputs", "waveform")
        valid_samples = Var("_audio_samples")

        return [
            LuaFragment(self.driver_script_path / "00_header.lua", top_level=True),
            # Everything except the stop tokens, which are bound into the loop that compares against
            # them rather than published as globals no fragment reads.
            ExportConstants(values={
                name: value for name, value in self.prompt_constants.items()
                if name not in ("EOS", "EOS_EXTRA")
            }),
            WaveformValidLength(
                waveform=waveform,
                valid=self._valid_samples_expr(),
                reflect_samples=self.reflect_samples or 0,
            ),
            SubgraphCallComponent(
                topology="encoder",
                # Retained, not bound to a local: the encoder emits `frames_per_chunk * hidden_size`
                # floats per second of audio, and the prompt's audio segment reads them backend-side
                # through an OutputRef rather than marshalling them into a Lua table (P4.0.12).
                outputs=(),
                retain=True,
                # `valid_samples` as a one-element array, which is how a `(1,)` input is spelled from
                # Lua. The encoder needs it to keep the caller's chunk padding out of the rows the
                # language model reads -- see `audio_encoder` and P4.3e.
                inputs={"waveform": waveform, "valid_samples": ArrayLit([valid_samples])},
                axes={"n_samples": Len(waveform), "n_past": Lit(0)},
                note="Encoder: mel frontend, chunked conv stem, window attention, projector -- one "
                     "pass over the whole (chunk-padded) waveform, masked back to the real audio.",
            ),
            PromptSegments(
                topology="decoder",
                bindings=self.decoder_bindings,
                embed_topology="embed",
                segments=(
                    ("text", Var("AUDIO_PREFIX")),
                    ("bound", OutputRef("encoder")),
                ),
                audio_rows=self.audio_rows(valid_samples),
            ),
            PrefillDecodeLoop(
                topology="decoder",
                bindings=self.decoder_bindings,
                inputs=tuple(name for name, _ in self.decoder_bindings),
                # The step's own tokens reach the decoder as EMBEDDINGS, through this topology, which
                # is the one structural difference between this loop and a plain causal LM's.
                embed_topology="embed",
                # Logits come from a separate phase, so the audio segment never computes any.
                head_topology="lm_head",
                # The loop starts where the prompt's segments left off, and its first iteration IS the
                # final text segment.
                initial_n_past=Var("_n_past"),
                prompt=Var("AUDIO_SUFFIX"),
                default_max_new_tokens=256,
                # The checkpoint's own stop tokens, so a host carries no per-model id. A chat-formatted
                # model declares two and stops on whichever it reaches.
                #
                # `.get` with the family-neutral defaults, not `[...]`: `component_registry.usage()`
                # builds every registered recognizer's component list WITHOUT a checkpoint, and an
                # empty report there is the claim that the catalogue's "used by" column is complete. A
                # hard index made this family the one recognizer that could not be read, which does not
                # fail loudly -- it quietly shortens the catalogue. -1 is `PrefillDecodeLoop`'s own
                # "no early stop" convention, so the list built without a model is inert rather than
                # wrong.
                default_eos_token=self.prompt_constants.get("EOS", -1),
                extra_eos_tokens=tuple(self.prompt_constants.get("EOS_EXTRA", ())),
            ),
        ]
