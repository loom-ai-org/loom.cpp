#!/usr/bin/env python3
"""Ground truth for the MIL-exported Whisper (`test_e2e_whisper_mil_export.cpp`, BACKLOG.md P4.1).

Runs HF's own `WhisperForConditionalGeneration` -- the library, not this repo's exporter -- as a greedy
decoder over a deterministic synthetic waveform, and writes the three arrays the C++ test needs:

    ref_mil_waveform.npy    (n_samples,) f32  -- exactly 30 s at the checkpoint's own sample rate
    ref_mil_prompt.npy      (n_prompt,)  i32  -- the decoder prefix, with the language HF DETECTED
    ref_mil_generated.npy   (n_new,)     i32  -- the token ids HF greedily generates after it
    ref_mil_encoder.npy     (n_ctx, d)   f32  -- the encoder's own hidden states, so a failure can be
                                                 localized to a half instead of only reported
    ref_mil_language.npy    ()           i32  -- the detected language token id, or -1 on an
                                                 English-only checkpoint that has none

**Why a synthetic waveform rather than real speech.** The check is that two implementations of the same
arithmetic agree, and agreement on noise is a strictly harder test than agreement on speech: a real clip
makes the greedy path collapse onto a few high-confidence tokens, where a wrong logit rarely changes the
argmax. Noise keeps the distribution flat, so any divergence shows up as a different token immediately.
The same argument `test_e2e_matcha_mil_*` makes for its own random inputs.

**Greedy, and with `max_new_tokens` matched to the test.** Nothing here samples, so the comparison is an
exact integer-sequence equality with no tolerance question -- the same gate `test_e2e_whisper_lua_driver`
already uses against the C++ driver.

    python3 tools/fixture_gen/reference_forward_whisper_mil.py <model_dir> <out_dir> [n_new_tokens]
"""
import sys
import types
from pathlib import Path

# transformers' hf-hub version gate, the same stub every export path in this repo installs.
_mock = types.ModuleType("dependency_versions_check")
_mock.dep_version_check = lambda *args, **kwargs: None
sys.modules.setdefault("transformers.dependency_versions_check", _mock)

import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

# The waveform is a fixed PRNG draw rather than a file: the fixture has to be reproducible from this
# script alone, on a machine that has the checkpoint and nothing else.
SEED = 0
AMPLITUDE = 0.05


def detect_language(model, encoder_out, generation_config):
    """The language token id HF's own arithmetic picks for this audio, or None if the checkpoint has no
    language tokens (`whisper-*.en`).

    Written out rather than calling `model.detect_language`, and deliberately: it must be the *same*
    computation the exported driver performs -- one decoder step from `<|startoftranscript|>` alone, then
    an argmax over the language block ONLY. If the oracle used a different route (a helper that, say,
    also applied `forced_decoder_ids`) the comparison would stop being about this engine's arithmetic.
    """
    lang_to_id = getattr(generation_config, "lang_to_id", None) or {}
    if not lang_to_id:
        return None
    lang_ids = sorted(int(v) for v in lang_to_id.values())
    sot = torch.tensor([[int(generation_config.decoder_start_token_id)]], dtype=torch.long)
    logits = model(decoder_input_ids=sot, encoder_outputs=(encoder_out,)).logits[0, -1]
    # Restricted to the language block: unrestricted, the argmax is whichever ordinary word scores
    # highest, which is what the driver's `argmax_row_range` exists to avoid.
    best = max(lang_ids, key=lambda i: float(logits[i]))
    return int(best)


def decoder_prompt(generation_config, language):
    """The token prefix a decode starts from: start-of-transcript, the detected language and the
    transcribe task when this checkpoint has them, then no-timestamps.

    Not `forced_decoder_ids`, and that is a real trap rather than a stylistic choice: a multilingual
    checkpoint leaves the language slot `None` there (`[[1, None], [2, 50359]]` for whisper-small),
    because HF fills it in from language *detection* at generation time -- which is why `language` is a
    parameter here. An English-only checkpoint has no language or task tokens at all and correctly gets
    the short prefix.
    """
    prompt = [generation_config.decoder_start_token_id]
    task_to_id = getattr(generation_config, "task_to_id", None) or {}
    if language is not None:
        prompt.append(language)
        if "transcribe" in task_to_id:
            prompt.append(task_to_id["transcribe"])
    no_timestamps = getattr(generation_config, "no_timestamps_token_id", None)
    if no_timestamps is not None:
        prompt.append(no_timestamps)
    return [int(t) for t in prompt]


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <model_dir> <out_dir> [n_new_tokens]", file=sys.stderr)
        sys.exit(1)
    model_dir, out_dir = sys.argv[1], Path(sys.argv[2])
    n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    out_dir.mkdir(parents=True, exist_ok=True)

    model = WhisperForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    extractor = WhisperFeatureExtractor.from_pretrained(model_dir)

    rng = np.random.default_rng(SEED)
    waveform = (rng.standard_normal(extractor.n_samples) * AMPLITUDE).astype(np.float32)
    features = torch.from_numpy(
        extractor(waveform, sampling_rate=extractor.sampling_rate, return_tensors="np")["input_features"]
    )

    with torch.no_grad():
        encoder_out = model.model.encoder(features).last_hidden_state

        language = detect_language(model, encoder_out, model.generation_config)
        prompt = decoder_prompt(model.generation_config, language)

        tokens = torch.tensor([prompt], dtype=torch.long)
        generated = []
        for _ in range(n_new):
            logits = model(
                decoder_input_ids=tokens, encoder_outputs=(encoder_out,),
            ).logits
            next_token = int(logits[0, -1].argmax())
            generated.append(next_token)
            if next_token == model.generation_config.eos_token_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[next_token]], dtype=torch.long)], dim=1)

    np.save(out_dir / "ref_mil_waveform.npy", waveform)
    np.save(out_dir / "ref_mil_prompt.npy", np.array(prompt, dtype=np.int32))
    np.save(out_dir / "ref_mil_generated.npy", np.array(generated, dtype=np.int32))
    np.save(out_dir / "ref_mil_encoder.npy", encoder_out[0].numpy().astype(np.float32))
    np.save(out_dir / "ref_mil_language.npy",
            np.array([-1 if language is None else language], dtype=np.int32))
    print(f"wrote {out_dir}: waveform {waveform.shape}, language {language}, prompt {prompt}, "
          f"generated {generated}, encoder {tuple(encoder_out.shape)}")


if __name__ == "__main__":
    main()
