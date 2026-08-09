"""Granite Speech 4.0.1b: family 3's second leaf, and the model that decided where the template's
boundary really is (BACKLOG.md P4.3c).

A 16-layer **conformer** encoder over 160-bin features (1024-wide, Shaw relative attention in blocks of
200 frames), a **BLIP-2 Q-Former** projector that turns each 15-frame window into three query rows, and
a 40-layer Granite causal LM (2048-wide, GQA 16/4) -- 2.31B parameters.

**Why this leaf and not Voxtral.** Voxtral-Mini-3B was the obvious second member (a Whisper encoder, a
frame-stacking projector, a Llama LM) and does not fit in this machine's memory; the measurements are in
BACKLOG.md P4.3b. Granite is the better test regardless: it varies BOTH halves the template claims to
abstract, where Voxtral would have varied neither by much. What survived the change is exactly the
template's stated contract -- the log-mel frontend, `(samples_per_chunk, frames_per_chunk)`, and the
four phases -- and the driver needed no new component at all.

**`has_lora_adapter = False` on this checkpoint**, so the LoRA path other Granite Speech variants carry
is not in play; `detect()` below refuses the ones where it is, rather than exporting a base model whose
audio path was never meant to run unadapted.

**Loads in the ordinary export environment**: `granite_speech` ships in transformers 4.57.6, so unlike
Qwen3-ASR this leaf needs no separate venv.
"""
import json
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .speech_lm_export import BaseSpeechLMExportConfig, LogMelFrontend, split_prompt_on_audio
from .spec_protocol import Unchecked


class ConformerQFormerEncoder(nn.Module):
    """`waveform -> projected audio embeddings`: the mel frontend, the 160-bin feature stacking, the
    conformer stack and the Q-Former projector, as one traced phase.

    **The rewrite is of two `math.ceil`s, and nothing else.** Both halves of this checkpoint pad a
    dynamic sequence up to a block boundary with Python-level arithmetic that `torch.jit.trace` bakes
    in as a constant -- `num_blocks = math.ceil(num_features / context_size)` with a `remainder`-driven
    right-pad in `GraniteSpeechConformerAttention`, and `nblocks = math.ceil(seq_len / window_size)` in
    `GraniteSpeechEncoderProjector`. This is the same wall Qwen3-ASR's encoder hit
    (`qwen3_asr_export.WindowedAudioEncoder`) and it takes the same fix: **require** the input to be a
    whole number of blocks, so every remainder is zero, and then spell the block count as `reshape(-1,
    block, ...)` so it stays dynamic. Everything else here calls the checkpoint's own submodules.

    Two smaller things are reimplemented for a reason:

    * `einsum("b m h c d, c r d -> b m h c r", ...)`, Shaw's relative-position term, becomes a batched
      matmul over the *query position* axis. ggml tensors are 4-D at most and that einsum's operands
      are 5-D; batching over `c` instead makes every operand 3-D or 4-D and changes no arithmetic.
    * the Q-Former's learnable query is a `(1, num_queries, hidden)` parameter that HF broadcasts
      against a batch of windows. The broadcast is two-way (batch 1 against N windows, N queries
      against 15 keys), which ggml's elementwise ops cannot do -- they repeat `b` into `a`. An **outer
      product against a ones column** materializes one copy of the query per window, which is the same
      trick, and for the same reason, as `WindowedAudioEncoder`'s window mask.

    **The contract this phase imposes on its caller**: the waveform is a whole number of chunks. One
    chunk here is `lcm(context_size, window_size)` encoder frames -- 600, i.e. 1200 mel frames, i.e.
    **192000 samples / 12 s** -- because the conformer's blocks and the Q-Former's windows must BOTH
    divide the sequence. That is coarse compared with Qwen3-ASR's one second, and it is a property of
    this checkpoint's two block sizes rather than of the family. The cost is Qwen3-ASR's, scaled by
    twelve: up to twelve seconds of trailing silence become real audio embeddings the LM reads -- up to
    **120 junk rows** against that leaf's 13 -- where HF would have masked them out. Trimming them
    needs a way to feed a *prefix* of a retained tensor, which nothing has today: BACKLOG.md **P4.3d**,
    open, and the reason it is worth doing once in `PromptSegments` rather than per leaf.

    The contract also makes the checkpoint's own feature extractor an exact oracle, because its
    "drop the final mel frame if the count is odd" fires on exactly the frame `LogMelFrontend` already
    drops (`1200k + 1` is always odd).
    """

    def __init__(self, mel: LogMelFrontend, encoder: nn.Module, projector: nn.Module):
        super().__init__()
        self.mel = mel
        self.encoder = encoder
        self.projector = projector
        config = encoder.config
        self.context_size = int(config.context_size)
        self.num_heads = int(config.num_heads)
        self.dim_head = int(config.dim_head)
        self.input_dim = int(config.input_dim)
        self.window_size = int(projector.window_size)
        self.num_queries = int(projector.num_queries)

    def _attention(self, attn, hidden):
        """`GraniteSpeechConformerAttention.forward` with the block count kept dynamic.

        `hidden` is `(1, N, hidden_dim)` with `N` a multiple of `context_size`, so HF's `remainder` is
        zero: its right-pad and its `masked_fill_` of the final block are both dead, and `num_blocks`
        is whatever `reshape(-1, ...)` infers.
        """
        block, heads, dim = self.context_size, self.num_heads, self.dim_head
        x = attn.pre_norm(hidden)
        query_states = attn.to_q(x)
        key_states, value_states = attn.to_kv(x).chunk(2, dim=-1)

        # (1, N, heads*dim) -> (n_blocks, block, heads, dim). `-1` is the block count.
        query_blocks = query_states.reshape(-1, block, heads, dim)
        key_blocks = key_states.reshape(-1, block, heads, dim)
        value_blocks = value_states.reshape(-1, block, heads, dim)

        # Shaw's relative-position term. HF spells it as a 5-D einsum over
        # `(batch, block, head, query_pos, dim) x (query_pos, key_pos, dim)`; batching the matmul over
        # the QUERY POSITION instead keeps every operand within ggml's four dimensions and contracts
        # the same axis. `rel_pos_emb` is a lookup into a static `(block, block)` buffer, so it is a
        # constant of the graph.
        rel_pos_emb = attn.rel_pos_emb(self.encoder.attention_dists)          # (block, block, dim)
        per_pos = query_blocks.permute(1, 0, 2, 3).reshape(block, -1, dim)    # (block, nb*heads, dim)
        pos_attn = torch.matmul(per_pos, rel_pos_emb.transpose(1, 2))         # (block, nb*heads, block)
        pos_attn = pos_attn.reshape(block, -1, heads, block).permute(1, 2, 0, 3) * attn.scale

        # HF calls SDPA with `attn_mask=pos_attn` and `scale=self.scale`, which is exactly this.
        q = query_blocks.permute(0, 2, 1, 3)
        k = key_blocks.permute(0, 2, 1, 3)
        v = value_blocks.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(2, 3)) * attn.scale + pos_attn
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        out = out.permute(0, 2, 1, 3).reshape(1, -1, heads * dim)
        return attn.to_out(out)

    def _conformer(self, features):
        """`GraniteSpeechCTCEncoder.forward`, with only the attention swapped out. The mid-stack
        self-conditioning (`out` -> softmax -> `out_mid`, added back at the half-way layer) is the
        checkpoint's own and traces as written."""
        encoder = self.encoder
        hidden = encoder.input_linear(features)
        for idx, layer in enumerate(encoder.layers, start=1):
            hidden = 0.5 * layer.ff1(hidden) + hidden
            hidden = self._attention(layer.attn, hidden) + hidden
            hidden = layer.conv(hidden) + hidden
            hidden = 0.5 * layer.ff2(hidden) + hidden
            hidden = layer.post_norm(hidden)
            if idx == encoder.num_layers // 2:
                hidden = hidden + encoder.out_mid(F.softmax(encoder.out(hidden), dim=-1))
        return hidden

    def _project(self, hidden):
        """`GraniteSpeechEncoderProjector.forward` with the window count kept dynamic and the two
        all-ones attention masks dropped.

        The Q-Former's masks are built by HF as `torch.ones(...)` over shapes derived from the batch,
        then turned into `(1 - mask) * -10000` -- identically zero, i.e. additive no-ops. Building them
        would put a MIL `fill` over a dynamic extent into the graph for no arithmetic at all, which is
        the failure `WindowedAudioEncoder` records in detail, so the layers are called directly instead
        of through `Blip2QFormerModel.forward`.
        """
        projector = self.projector
        width = self.projector.query.shape[-1]
        windows = hidden.reshape(-1, self.window_size, hidden.shape[-1])

        # One copy of the learnable query per window, as an outer product rather than an `expand`: the
        # window count is dynamic, and `query * ones_column` would be a two-way broadcast that ggml's
        # elementwise ops cannot perform. A matmul is not a broadcast, so nothing rewrites it back into
        # one. The ones column is built by arithmetic on `windows` for the same reason
        # `WindowedAudioEncoder` builds its own that way -- `torch.ones_like` becomes a MIL `fill`
        # whose extent the exporter resolves through a different expression for the same quantity.
        ones = windows[:, :1, :1].reshape(-1, 1) * 0.0 + 1.0
        query = torch.matmul(ones, projector.query.reshape(1, -1))
        query = query.reshape(-1, self.num_queries, width)

        qformer = projector.qformer
        hidden_states = qformer.layernorm(query)
        for layer in qformer.encoder.layer:
            hidden_states = layer.attention(hidden_states=hidden_states)
            hidden_states = layer.crossattention(
                hidden_states=hidden_states, encoder_hidden_states=windows,
            )
            hidden_states = layer.feed_forward_chunk_query(hidden_states)
        return projector.linear(hidden_states.reshape(-1, width))

    def forward(self, waveform):
        mel = self.mel(waveform)
        # The extractor's own layout: mel frames on the time axis, then consecutive PAIRS of frames
        # stacked into one 160-bin encoder frame. That halving is why one encoder frame is two mel
        # frames everywhere in this file's arithmetic.
        features = mel.transpose(1, 2).reshape(1, -1, self.input_dim)
        return self._project(self._conformer(features))


class _ScaledLMHead(nn.Module):
    """`hidden states -> logits`, including the `/ logits_scaling` that Granite applies OUTSIDE its
    `lm_head`.

    `GraniteForCausalLM.forward` ends with `logits = logits / self.config.logits_scaling` (8.0 here),
    which is one of the four multipliers that distinguish Granite from Llama. Exporting `lm_head` alone
    would drop it, and the driver would never notice: it takes an argmax, and dividing every logit by
    the same positive constant cannot move one. A host that sampled, or that read a probability, would
    notice -- so the phase emits the logits this checkpoint says it emits.
    """

    def __init__(self, lm_head: nn.Module, logits_scaling: float):
        super().__init__()
        self.lm_head = lm_head
        self.logits_scaling = float(logits_scaling)

    def forward(self, hidden):
        return self.lm_head(hidden) / self.logits_scaling


@dataclass
class ASRGraniteSpeechExportConfig(BaseSpeechLMExportConfig):
    """Granite Speech's loader over family 3's template."""

    architecture: str = "granite-speech"
    output_path: str = "granite_speech_mil.gguf"
    driver_script_path: Path = Path(__file__).resolve().parent / "speech_lm_driver"
    # A chunk is twelve seconds here rather than Qwen3-ASR's one, so the same ceiling in chunks would
    # be six minutes of audio -- far past what `max_seq_len` admits anyway (120 rows per chunk against
    # a 4096-token cache). Ten chunks is two minutes, and 1200 of the cache's 4096 positions.
    max_audio_chunks: int = 10
    # The instruction the model is asked to follow, which for this checkpoint is a genuine choice and
    # not a template constant: Granite Speech's chat template renders `USER: {content}\n ASSISTANT:`
    # and the audio placeholder is written INTO the content by the caller, so the text after it is
    # whatever task is wanted. This is the model card's own transcription prompt.
    instruction: str = "can you transcribe the speech into a written format?"

    _processor: Optional[object] = field(default=None, init=False, repr=False)

    __unchecked__ = {
        "instruction": Unchecked(
            "the task this export asks the model to perform, baked into the prompt's text suffix. A "
            "caller's choice -- this checkpoint follows instructions and its chat template carries no "
            "task of its own -- so there is no second authority to check it against. It IS checked to "
            "be non-empty by `prompt_segment_constants`, since an empty suffix would leave the decode "
            "loop with no first iteration."
        ),
        "_processor": Unchecked(
            "the checkpoint's own AutoProcessor, cached so the feature extractor is loaded once rather "
            "than per phase. Loaded FROM the checkpoint, never constructed with this family's idea of "
            "the defaults -- the filterbank, FFT geometry and window length are all properties of the "
            "checkpoint and `preprocessor_config.json` is where it states them."
        ),
    }

    def load_model(self):
        from transformers import GraniteSpeechForConditionalGeneration

        print(f"Loading model from {self.model_dir}...")
        # The DEFAULT attention implementation, for the reason `qwen3_asr_export` records at length:
        # `eager` would also reach the LANGUAGE model, whose attention is traced as written, and
        # `fuse_loom_attention` matches the SDPA shape -- under `eager` the decoder converts with zero
        # fused ATTENTION nodes and therefore no KV cache, and every phase still exports
        # "successfully". `eager` belongs on the reference generator, which is where it is.
        return GraniteSpeechForConditionalGeneration.from_pretrained(
            self.model_dir, dtype=torch.float32
        ).eval()

    def _processor_for(self):
        from transformers import AutoProcessor

        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_dir)
        return self._processor

    def feature_extractor(self):
        return self._processor_for().audio_processor

    def mel_frontend(self, extractor) -> LogMelFrontend:
        """This extractor states its geometry in `melspec_kwargs` and holds its filterbank inside a
        torchaudio `MelSpectrogram`, so the numbers are read from there rather than from the attribute
        names a transformers extractor uses. `mel_scale.fb` is already `(n_freq, n_mels)`, which is the
        layout `LogMelFrontend` transposes."""
        melspec = extractor.mel_filters
        return LogMelFrontend(
            n_fft=extractor.melspec_kwargs["n_fft"],
            hop_length=extractor.melspec_kwargs["hop_length"],
            mel_filters=melspec.mel_scale.fb.detach().cpu().numpy(),
            win_length=extractor.melspec_kwargs["win_length"],
        )

    def language_model(self, model):
        return model.language_model.model

    def lm_head(self, model):
        return _ScaledLMHead(model.language_model.lm_head,
                             model.config.text_config.logits_scaling)

    def audio_encoder(self, model, mel: LogMelFrontend) -> nn.Module:
        return ConformerQFormerEncoder(mel, model.encoder, model.projector)

    def audio_geometry(self, model, extractor) -> tuple:
        """`(samples_per_chunk, frames_per_chunk)`, and both numbers are FORCED rather than chosen.

        The conformer attends in blocks of `context_size` frames and the Q-Former consumes windows of
        `window_size` frames, and a chunk must contain a whole number of both -- so it is their least
        common multiple, `lcm(200, 15) = 600` encoder frames. One encoder frame is two mel frames (the
        extractor stacks consecutive pairs) and one mel frame is `hop_length` samples, so a chunk is
        192000 samples, twelve seconds. It becomes `600 / 15 * 3 = 120` prompt positions.

        Twelve seconds is coarse -- a host pads up to it -- and it is what this checkpoint's two block
        sizes cost, not a decision this family made. `phases()` checks the pair against the traced
        encoder's real output.
        """
        encoder_config = model.config.encoder_config
        projector = model.projector
        context_size = int(encoder_config.context_size)
        window_size = int(projector.window_size)
        encoder_frames = context_size * window_size // gcd(context_size, window_size)
        return (encoder_frames * 2 * int(extractor.melspec_kwargs["hop_length"]),
                encoder_frames // window_size * int(projector.num_queries))

    def prompt_segment_constants(self, model) -> dict:
        """The token ids the driver's prompt is built from, rendered through the checkpoint's own chat
        template and tokenizer rather than hardcoded.

        Granite Speech's template is a plain text one -- `USER: {content}\\n ASSISTANT:` -- and the
        audio placeholder is part of the content, so unlike Qwen3-ASR the split point is written by
        this method rather than produced by the processor's audio handling:

            USER: <|audio|>
            ... audio rows ...
            can you transcribe the speech into a written format?\\n ASSISTANT:

        The placeholder is emitted ONCE here, not once per audio row. HF's processor expands it to one
        copy per projected row because it then scatters the encoder's output into those positions; this
        driver writes those rows into the KV cache directly, as their own cached call, so the prompt it
        needs is the text on either side and the row count is arithmetic
        (`PromptSegments.audio_rows_per_chunk`).
        """
        if not self.instruction.strip():
            raise ValueError(
                "instruction is empty, so this prompt's text suffix would be empty too -- and the "
                "suffix is what PrefillDecodeLoop runs as its first iteration. Give the model a task."
            )
        processor = self._processor_for()
        tokenizer = processor.tokenizer
        audio_id = int(model.config.audio_token_id)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{processor.audio_token}{self.instruction}"}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        prefix, suffix = split_prompt_on_audio(ids, audio_id)
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [int(e) for e in (eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])]
        return {
            "EOS": eos_ids[0],
            "EOS_EXTRA": tuple(eos_ids[1:]),
            "AUDIO_PREFIX": prefix,
            "AUDIO_SUFFIX": suffix,
        }


def _is_granite_speech(path: Path) -> bool:
    """A real structural check (BACKLOG.md P3.2): an HF directory declaring
    `model_type == "granite_speech"` **without a LoRA adapter**.

    The second half is what keeps this honest. Granite Speech ships variants whose language model is
    only correct with a LoRA adapter merged in for audio inputs -- `has_lora_adapter` is the
    checkpoint's own statement of that, and the modeling code warns rather than raises when peft is
    missing. This exporter traces the base weights, so on such a checkpoint it would produce a model
    that runs and transcribes badly. Refusing detection puts that in the candidate list instead.
    """
    config_path = path / "config.json"
    if not path.is_dir() or not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(config, dict) or config.get("model_type") != "granite_speech":
        return False
    return not config.get("has_lora_adapter", False)


def _build_granite_speech(path: Path, output_path: str) -> ASRGraniteSpeechExportConfig:
    return ASRGraniteSpeechExportConfig(model_dir=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=ASRGraniteSpeechExportConfig,
        recognizers=[ModelRecognizer(
            name="granite-speech", detect=_is_granite_speech, build_config=_build_granite_speech,
        )],
    ))
