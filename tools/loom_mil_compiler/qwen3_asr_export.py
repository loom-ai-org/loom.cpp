"""Qwen3-ASR: family 3's first leaf, and the acceptance model for the composition template
(BACKLOG.md P4.3).

An 18-layer window-attention audio encoder (896-wide), a two-layer projector (896 -> 896 -> 1024), and
a 28-layer Qwen3 causal LM (1024-wide, GQA 16/8) -- 782M parameters, all four pieces in one checkpoint.

**This module is the loader, and almost nothing else**, which is the split P4.2 established for
transducers and this family inherits: `speech_lm_export.BaseSpeechLMExportConfig` holds the four
phases, their shapes and axes, the cross-checks and the whole component list, and a leaf says only how
its checkpoint is opened and where its pieces live -- plus, since this checkpoint's encoder is not
traceable as written, the rewrite of it below.

`WindowedAudioEncoder` lives HERE rather than in the template, and the second leaf is what moved it:
this is a Qwen3-Omni window-attention stack over one-second chunks, where Granite Speech's is a
conformer feeding a Q-Former. What the template kept is the log-mel frontend and the
`(samples_per_chunk, frames_per_chunk)` contract, which is all two encoders this different turn out to
share.

**Which checkpoint layout this loads, because there are two and only one works.** Qwen publishes
`Qwen/Qwen3-ASR-0.6B` (the native `qwen-asr` package layout: weights prefixed `thinker.*`, sub-configs
nested under `thinker_config`) and `Qwen/Qwen3-ASR-0.6B-hf`. transformers' own `qwen3_asr` module has
no `_checkpoint_conversion_mapping` and no `thinker` handling, so pointing it at the native layout does
not raise -- it reads an empty config and returns **class defaults** (a 1024-wide, 24-layer encoder
instead of this checkpoint's 896-wide, 18-layer one). `detect()` below therefore checks for the `-hf`
layout specifically rather than for `model_type` alone, so the native one fails detection with the
candidate list instead of being exported as a plausible wrong model.

**Requires transformers >= 5.13**, which is where `qwen3_asr` first ships.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .speech_lm_export import BaseSpeechLMExportConfig, LogMelFrontend, split_prompt_on_audio
from .spec_protocol import Unchecked


class WindowedAudioEncoder(nn.Module):
    """`(waveform, valid_samples) -> projected audio embeddings`: the mel frontend, a chunked
    convolutional stem, a window-attention transformer stack, and the projector, as one traced phase.

    **This is a rewrite of the checkpoint's own encoder forward, and it has to be.** The HF forward is
    untraceable twice over, and neither is incidental:

    * it packs valid frames with `valid_mask.flatten().nonzero()` + `index_select`, a data-dependent
      output shape;
    * it runs attention per window with `torch.split(q, lengths.tolist(), dim=2)`, whose lengths are
      Python integers that `torch.jit.trace` bakes in as constants -- so a traced graph would carry one
      specific audio length's window layout forever.

    Both are replaced by the same observation: `get_audio_cu_seqlens` cuts the post-CNN sequence into
    full `block`-sized windows plus a shorter final one, and attention runs independently inside each,
    which is *exactly* a block-diagonal additive mask. With every frame valid the packing step is the
    identity, and with the mask built in-graph from `torch.arange(T)` the whole phase has one dynamic
    axis and needs no second declared axis and no new engine binding.

    Verified against HF's own `get_audio_features` on real audio, in PyTorch, before anything was
    exported: `max abs diff 0.000e+00` -- bit-identical, not merely close. (Against HF's *sdpa* path it
    is 6.3e-05, which is that fused kernel's accumulation order and not a difference in this rewrite.)

    **The contract this phase imposes on its caller**: the waveform is a whole number of chunks. One
    chunk is `hop_length * n_window * 2` samples -- 16000, i.e. one second, for Qwen3-ASR -- so a host
    pads up to the next second. That is what makes the packing step an identity, and it makes the
    checkpoint's own feature extractor a faithful oracle: its mel-axis right-pad becomes a no-op on
    such a waveform, so HF and this phase see the identical mel.

    **`valid_samples` is where the padding stops** (BACKLOG.md P4.3e). HF does not run this encoder on
    padding at all: `Qwen3ASREncoder.forward` packs only the post-CNN positions its
    `input_features_mask` marks valid, and windows attention over that packed sequence. Two statements
    reproduce it here, and the first is the one that is easy to miss:

    * **the mel frames past the real audio are zeroed.** HF's extractor right-pads `input_features`
      with literal 0.0, and log-mel of silence is emphatically not zero -- so a chunk-padded waveform
      hands the convolutional stem a different final chunk than the extractor ever produced. The stem
      is a 2-D convolution over a whole chunk, so that difference reaches the valid rows of that chunk
      and not only the padded ones.
    * **the padded rows are masked as attention keys**, which is what HF's packing does by simply not
      including them. The windows themselves need no adjustment: every whole chunk contributes exactly
      `max_position_embeddings` rows in both worlds, so a valid row sits at the same index in the
      packed sequence as in the padded one and falls in the same window.

    Measured against HF on 10.625 s of audio in eleven one-second chunks: 1.0e-01 before, **exactly
    zero** after -- bit-identical, on rows whose absmax is 0.128.

    Both statements are about KEYS and FEATURES, never about which rows exist: the phase still emits
    `13` rows per chunk including the padded ones, and `audio_rows` is what stops the prompt before
    them.
    """

    def __init__(self, mel: LogMelFrontend, tower: nn.Module, projector: nn.Module, audio_config):
        super().__init__()
        self.mel = mel
        self.tower = tower
        self.projector = projector
        self.chunk_len = int(audio_config.n_window) * 2
        self.num_heads = int(audio_config.encoder_attention_heads)
        self.head_dim = int(audio_config.d_model) // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.rows_per_chunk = int(audio_config.max_position_embeddings)
        # `get_audio_cu_seqlens`' own window width, in post-CNN frames: the per-chunk frame count times
        # how many chunks an inference window spans.
        ratio = int(audio_config.n_window_infer) // self.chunk_len
        self.block = self.rows_per_chunk * ratio

    def _attention(self, layer, hidden, mask):
        attn = layer.self_attn
        total = hidden.shape[0]
        shape = (total, self.num_heads, self.head_dim)
        q = attn.q_proj(hidden).reshape(shape).transpose(0, 1).unsqueeze(0)
        k = attn.k_proj(hidden).reshape(shape).transpose(0, 1).unsqueeze(0)
        v = attn.v_proj(hidden).reshape(shape).transpose(0, 1).unsqueeze(0)
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling + mask
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        out = out.squeeze(0).transpose(0, 1).reshape(total, self.num_heads * self.head_dim)
        return attn.out_proj(out)

    def _valid_rows(self, valid_mel):
        """`Qwen3ASRProcessor._get_audio_token_length`, in graph: three stride-2 convolutions over the
        final partial chunk, plus `max_position_embeddings` rows for every whole one.

        `(n - 1) // 2 + 1` per convolution, and the floor has to reach negative numbers: an empty
        remainder gives `(0 - 1) // 2 + 1 = 0`, which is how a waveform that fills its last chunk
        exactly gets no extra rows. `torch.floor` on the float quotient is that floor -- Python's `//`
        and Lua's `math.floor(a / b)` agree with it, which is what lets `audio_rows` state the same
        formula for the driver.
        """
        whole = torch.floor(valid_mel / self.chunk_len)
        rows = valid_mel - whole * self.chunk_len
        for _ in range(3):
            rows = torch.floor((rows - 1.0) / 2.0) + 1.0
        return rows + whole * self.rows_per_chunk

    def forward(self, waveform, valid_samples):
        tower = self.tower
        mel = self.mel(waveform)
        n_mels = mel.shape[1]
        # Where the caller's real audio stops, in mel frames. `floor(L / hop)` is what the extractor's
        # own attention mask comes to: it rescales a sample-resolution mask by the hop and drops the
        # final entry whenever the sample count is not a multiple of it, which is `floor` either way.
        valid_mel = torch.floor(valid_samples.reshape(1) / self.mel.hop_length)
        frame_pos = torch.arange(mel.shape[2], dtype=mel.dtype, device=mel.device)
        # The extractor pads `input_features` with 0.0, not with the log-mel of silence a zero-padded
        # waveform produces -- and the conv stem spreads that difference across the whole final chunk.
        mel = mel * (frame_pos.unsqueeze(0) < valid_mel).to(mel.dtype).reshape(1, 1, -1)
        # Chunk the mel time axis and fold the chunks into the batch, which is what makes the
        # convolutional stem see a fixed 100-frame window regardless of clip length. `-1` rather than a
        # computed chunk count: the whole point is that this dimension stays dynamic.
        chunked = (mel.reshape(1, n_mels, -1, self.chunk_len)
                   .permute(0, 2, 1, 3)
                   .reshape(-1, 1, n_mels, self.chunk_len))
        conv = F.gelu(tower.conv2d1(chunked))
        conv = F.gelu(tower.conv2d2(conv))
        conv = F.gelu(tower.conv2d3(conv))
        channels, freq_bins, steps = conv.shape[1], conv.shape[2], conv.shape[3]
        hidden = tower.conv_out(
            conv.permute(0, 3, 1, 2).reshape(-1, steps, channels * freq_bins)
        )
        hidden = hidden + tower.positional_embedding.positional_embedding[:steps].to(hidden.dtype)
        # HF's `index_select` over the non-padding positions is a PREFIX here: every chunk before the
        # last contributes all of its rows in both worlds, so the packed sequence is the flattened one
        # cut at `_valid_rows`. That is why the rows keep their indices and the windows below need no
        # adjustment -- only the keys past the cut have to be masked out.
        hidden = hidden.reshape(-1, hidden.shape[-1])
        total = hidden.shape[0]
        # Float, not integer: the outer products below are real matmuls, and the window index is a
        # small whole number that float32 represents exactly.
        pos = torch.arange(total, device=hidden.device, dtype=hidden.dtype)
        window = torch.floor(pos / self.block)
        # Both operands expanded to the full square BEFORE the comparison, rather than left as
        # `(T, 1)` and `(1, T)` for broadcasting. ggml's elementwise ops repeat `b` into `a` and cannot
        # do a two-way broadcast, so the natural `window.unsqueeze(1) == window.unsqueeze(0)` spelling
        # aborts in `ggml_sub` inside the `equal` primitive.
        #
        # `.expand(total, total)` on both operands is the obvious repair and does NOT work: MIL's
        # `equal` broadcasts natively, so coremltools folds the expands away and emits the same
        # two-way broadcast again. An OUTER PRODUCT against a vector of ones survives, because a
        # `matmul` is not a broadcast and there is no pattern that rewrites it into one -- each side is
        # a genuine (T, T) tensor by the time the comparison sees it.
        #
        # The comparison is then against a SCALAR rather than between two tensors, which is the second
        # half of staying inside what ggml can repeat: a rank-0 constant repeats into any shape, where
        # two equal-rank tensors must match exactly. Window indices are whole numbers, so `< 0.5`
        # separates "same window" from "adjacent window" exactly.
        # `window * 0 + 1` rather than `torch.ones_like(window)`: the latter converts to a MIL `fill`,
        # whose length the exporter resolves through its own shape INPUT rather than from `window` --
        # and it resolved to a different expression for the same quantity, so the two sides of the
        # comparison disagreed about T. Built by arithmetic, the ones vector is elementwise on `window`
        # and cannot have a length `window` does not.
        ones = window * 0.0 + 1.0
        rows = window.unsqueeze(1) @ ones.unsqueeze(0)
        cols = ones.unsqueeze(1) @ window.unsqueeze(0)
        same = torch.abs(rows - cols) < 0.5
        # And the key is a row HF's packing would have kept. Broadcast the same way the window indices
        # are, and for the same reason.
        keep = (pos < self._valid_rows(valid_mel)).to(hidden.dtype)
        key_ok = (ones.unsqueeze(1) @ keep.unsqueeze(0)) > 0.5
        # A large finite negative rather than `-inf`: a window that lies entirely past the real audio
        # has every key masked, and softmax over a row of `-inf` is NaN -- which no later mask can
        # remove. `-finfo.max` softmaxes such a row to a uniform, finite, unread value, and is
        # indistinguishable from `-inf` wherever any key survives (the difference underflows f32).
        mask = torch.where(same & key_ok, 0.0, float(-torch.finfo(torch.float32).max))
        mask = mask.view(1, 1, -1, total)
        for layer in tower.layers:
            residual = hidden
            hidden = layer.self_attn_layer_norm(hidden)
            hidden = residual + self._attention(layer, hidden, mask)
            residual = hidden
            hidden = layer.final_layer_norm(hidden)
            hidden = residual + layer.fc2(layer.activation_fn(layer.fc1(hidden)))
        return self.projector(tower.ln_post(hidden))


@dataclass
class ASRQwen3SpeechLMExportConfig(BaseSpeechLMExportConfig):
    """Qwen3-ASR's loader over family 3's template."""

    architecture: str = "qwen3-asr"
    output_path: str = "qwen3_asr_mil.gguf"
    driver_script_path: Path = Path(__file__).resolve().parent / "speech_lm_driver"

    _processor: Optional[object] = field(default=None, init=False, repr=False)
    # `(hop_length, mel_frames_per_chunk, rows_per_chunk)` -- read in `audio_geometry`, used by
    # `audio_rows`. Defaults to this checkpoint's own values so `component_registry.usage()` can build
    # the component list without one, the same reason `prompt_constants` is read with `.get`.
    _row_formula: tuple = field(default=(160, 100, 13), init=False, repr=False)

    __unchecked__ = {
        "_processor": Unchecked(
            "the checkpoint's own AutoProcessor, cached so the feature extractor is loaded once "
            "rather than per phase. Loaded FROM the checkpoint, never constructed with this family's "
            "idea of the defaults -- the filterbank, FFT geometry and chunk length are all properties "
            "of the checkpoint and `processor_config.json` is where it states them."
        ),
        "_row_formula": Unchecked(
            "READ off the extractor and audio config in audio_geometry(), and checked where it is "
            "used: `phases()` evaluates the expression `audio_rows` builds from these three numbers "
            "and requires it to agree with `audio_geometry`'s own rows-per-chunk at every whole "
            "number of chunks"
        ),
    }

    def load_model(self):
        from transformers import Qwen3ASRForConditionalGeneration

        print(f"Loading model from {self.model_dir}...")
        # The DEFAULT attention implementation, deliberately, and getting this wrong is silent.
        #
        # Forcing `eager` here looks harmless -- `WindowedAudioEncoder` reimplements the audio tower's
        # attention as an explicit matmul/softmax and never calls the tower's own, so the setting
        # cannot change the encoder at all. But it also reaches the LANGUAGE MODEL, whose attention IS
        # traced as written, and `fuse_loom_attention` matches the SDPA shape: under `eager` the
        # decoder converted with **zero** fused ATTENTION nodes, so it had no KV cache, the mask was
        # never retyped to `n_kv`, and every phase still exported "successfully". The failure surfaced
        # only at run time, as a mask sized 143x143 where the cache wanted 143x152.
        #
        # `eager` belongs on the REFERENCE, not on the export: the encoder rewrite is bit-identical to
        # HF's eager path and differs from its sdpa path by 6.3e-05 in accumulation order alone, so the
        # fixture generator asks for eager and this does not.
        return Qwen3ASRForConditionalGeneration.from_pretrained(
            self.model_dir, dtype=torch.float32
        ).eval()

    def _processor_for(self):
        from transformers import AutoProcessor

        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_dir)
        return self._processor

    def feature_extractor(self):
        return self._processor_for().feature_extractor

    def language_model(self, model):
        return model.model.language_model

    def lm_head(self, model):
        return model.lm_head

    def audio_encoder(self, model, mel: LogMelFrontend) -> nn.Module:
        return WindowedAudioEncoder(
            mel, model.model.audio_tower, model.model.multi_modal_projector, model.config.audio_config,
        )

    def audio_geometry(self, model, extractor) -> tuple:
        """One chunk is `n_window * 2` mel frames -- one second at this checkpoint's 16 kHz / hop 160 --
        and the conv stem turns it into `max_position_embeddings` rows (13). Both are read off the
        checkpoint; `phases()` checks the pair against the encoder's real output."""
        audio_config = model.config.audio_config
        # The three numbers `audio_rows` needs, stashed at the one moment the model and its extractor
        # are both in hand -- the same point `phases()` reads everything else it publishes.
        self._row_formula = (int(extractor.hop_length), int(audio_config.n_window) * 2,
                             int(audio_config.max_position_embeddings))
        return (int(extractor.hop_length) * int(audio_config.n_window) * 2,
                int(audio_config.max_position_embeddings))

    def audio_rows(self, valid_samples):
        """`Qwen3ASRProcessor._get_audio_token_length`, as a driver expression over the caller's
        unpadded sample count (BACKLOG.md P4.3e).

        Four steps, HF's own: `floor(L / hop)` valid mel frames, the remainder of those past the last
        whole chunk, three stride-2 convolutions over that remainder (`(n - 1) // 2 + 1` each), and
        `max_position_embeddings` rows for every whole chunk.

        **P4.3d could not write this, and the difference is where the frame count comes from.** That
        attempt derived the mel-frame count as a closed form over samples and disagreed with the
        processor at 5 of 12 probe lengths, so this leaf kept the padded row count instead. The
        extractor's mask is `attention_mask[:, ::hop]` with its final entry dropped when the sample
        count is not a multiple of the hop -- which is `floor(L / hop)` in both branches, not the
        `ceil` the rescaling looks like. With that, the remaining three lines are HF's verbatim and
        agree at every length.

        Stating it is no longer optional: the encoder masks its own padding now, so the rows past this
        count are not a reading of trailing silence, they are rows whose keys were all masked.
        `phases()` checks it against `audio_geometry` at whole chunks, where the two cannot differ.
        """
        from .driver_ir import BinOp, Lit

        hop, chunk_frames, rows_per_chunk = self._row_formula
        mel_frames = BinOp("floordiv", valid_samples, Lit(hop))
        whole = BinOp("floordiv", mel_frames, Lit(chunk_frames))
        rows = BinOp("-", mel_frames, BinOp("*", whole, Lit(chunk_frames)))
        for _ in range(3):
            rows = BinOp("+", BinOp("floordiv", BinOp("-", rows, Lit(1)), Lit(2)), Lit(1))
        return BinOp("+", rows, BinOp("*", whole, Lit(rows_per_chunk)))

    def prompt_segment_constants(self, model) -> dict:
        """The token ids the driver's prompt is built from, resolved through the checkpoint's own chat
        template and tokenizer rather than hardcoded.

        Qwen3-ASR's template renders one fixed shape -- a system turn, a user turn holding the audio
        placeholder, and the assistant header -- so the prompt is exactly two runs of text with the
        audio between them:

            <|im_start|>system\\n<|im_end|>\\n<|im_start|>user\\n<|audio_start|>
            ... audio rows ...
            <|audio_end|><|im_end|>\\n<|im_start|>assistant\\n

        Rendering the template and splitting it on the audio placeholder is what produces those two
        runs, so the driver carries the checkpoint's real prompt rather than this module's transcription
        of it -- and a template change moves the constants instead of silently disagreeing with them.

        The two runs are returned as lists, which `ExportConstants` binds as Lua arrays -- so this
        family needs no hand-written prompt fragment at all, and every read of them is a real symbol
        `driver_ir.validate` resolves rather than text substituted into a template.
        """
        processor = self._processor_for()
        audio_id = int(model.config.audio_token_id)
        rendered = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "audio", "audio": _SILENCE}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        )
        ids = rendered["input_ids"][0].tolist()
        prefix, suffix = split_prompt_on_audio(ids, audio_id)
        # eos: the generation config lists both the base model's end-of-text and the chat turn's end,
        # and a decode loop has to stop on either. These two do NOT become ExportConstants -- they are
        # bound into the loop itself, since it is the loop that compares against them.
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [int(e) for e in (eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])]
        return {
            "EOS": eos_ids[0],
            "EOS_EXTRA": tuple(eos_ids[1:]),
            "AUDIO_PREFIX": prefix,
            "AUDIO_SUFFIX": suffix,
        }


# Half a second of silence, only ever used to make the chat template render. The template branches on
# whether an audio item is PRESENT, never on its contents, and the placeholder run it emits is
# discarded here -- only the text on either side of it is kept.
_SILENCE = np.zeros(8000, dtype=np.float32)


def _is_qwen3_asr_hf(path: Path) -> bool:
    """A real structural check (BACKLOG.md P3.2): an HF directory declaring `model_type == "qwen3_asr"`
    **in the transformers layout**.

    The second half of that is what makes this honest rather than merely specific. Qwen ships the same
    weights twice, and the native `qwen-asr` layout declares the identical `model_type` while nesting
    its real sub-configs under `thinker_config` -- where transformers reads class defaults off it
    without raising. Requiring `audio_config` and `text_config` at the top level is what tells the two
    apart, and it is the same property the loader depends on.
    """
    config_path = path / "config.json"
    if not path.is_dir() or not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(config, dict) or config.get("model_type") != "qwen3_asr":
        return False
    return "audio_config" in config and "text_config" in config


def _build_qwen3_asr(path: Path, output_path: str) -> ASRQwen3SpeechLMExportConfig:
    return ASRQwen3SpeechLMExportConfig(model_dir=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=ASRQwen3SpeechLMExportConfig,
        recognizers=[ModelRecognizer(
            name="qwen3-asr", detect=_is_qwen3_asr_hf, build_config=_build_qwen3_asr,
        )],
    ))
