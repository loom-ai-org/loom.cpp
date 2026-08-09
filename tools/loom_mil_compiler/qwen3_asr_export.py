"""Qwen3-ASR: family 3's first leaf, and the acceptance model for the composition template
(BACKLOG.md P4.3).

An 18-layer window-attention audio encoder (896-wide), a two-layer projector (896 -> 896 -> 1024), and
a 28-layer Qwen3 causal LM (1024-wide, GQA 16/8) -- 782M parameters, all four pieces in one checkpoint.

**This module is the loader, and almost nothing else**, which is the split P4.2 established for
transducers and this family inherits: `speech_lm_export.BaseSpeechLMExportConfig` holds the four
phases, their shapes and axes, the cross-checks and the whole component list, and a leaf says only how
its checkpoint is opened and where its pieces live. The next family-3 member (Voxtral is the obvious
one -- a Whisper encoder, the same shape of projector, a Llama LM) should be about this long.

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

from .speech_lm_export import BaseSpeechLMExportConfig
from .spec_protocol import Unchecked


@dataclass
class ASRQwen3SpeechLMExportConfig(BaseSpeechLMExportConfig):
    """Qwen3-ASR's loader over family 3's template."""

    architecture: str = "qwen3-asr"
    output_path: str = "qwen3_asr_mil.gguf"
    driver_script_path: Path = Path(__file__).resolve().parent / "speech_lm_driver"

    _processor: Optional[object] = field(default=None, init=False, repr=False)

    __unchecked__ = {
        "_processor": Unchecked(
            "the checkpoint's own AutoProcessor, cached so the feature extractor is loaded once "
            "rather than per phase. Loaded FROM the checkpoint, never constructed with this family's "
            "idea of the defaults -- the filterbank, FFT geometry and chunk length are all properties "
            "of the checkpoint and `processor_config.json` is where it states them."
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

    def audio_config(self, model):
        return model.config.audio_config

    def audio_tower(self, model):
        return model.model.audio_tower

    def projector(self, model):
        return model.model.multi_modal_projector

    def language_model(self, model):
        return model.model.language_model

    def lm_head(self, model):
        return model.lm_head

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
        if audio_id not in ids:
            raise ValueError(
                f"this checkpoint's chat template rendered no audio placeholder (id {audio_id}) at "
                f"all, so there is no position for the encoder's output to occupy. The template is "
                f"what this family reads the prompt's shape from."
            )
        first = ids.index(audio_id)
        last = len(ids) - 1 - ids[::-1].index(audio_id)
        prefix, suffix = ids[:first], ids[last + 1:]
        if not all(token == audio_id for token in ids[first:last + 1]):
            # The driver feeds the audio as ONE contiguous segment at one `n_past`. Interleaved text
            # inside the placeholder run would need a segment list this family does not build, and
            # would otherwise be silently dropped.
            raise ValueError(
                f"this checkpoint's chat template puts non-audio tokens between its audio "
                f"placeholders (ids {first}..{last}), so the audio does not occupy one contiguous run "
                f"of prompt positions. PromptSegments feeds the encoder's output as a single segment."
            )
        # eos: the generation config lists both the base model's end-of-text and the chat turn's end,
        # and a decode loop has to stop on either. These two do NOT become ExportConstants -- they are
        # bound into the loop itself, since it is the loop that compares against them.
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [int(e) for e in (eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])]
        return {
            "EOS": eos_ids[0],
            "EOS_EXTRA": tuple(eos_ids[1:]),
            "AUDIO_PREFIX": [int(token) for token in prefix],
            "AUDIO_SUFFIX": [int(token) for token in suffix],
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
