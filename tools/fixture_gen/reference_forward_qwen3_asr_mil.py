#!/usr/bin/env python3
"""Ground truth for the MIL-exported Qwen3-ASR (`test_e2e_qwen3_asr_mil_export.cpp`, BACKLOG.md P4.3).

Runs HF's own `Qwen3ASRForConditionalGeneration` -- the library, not this repo's exporter -- over real
speech, and writes the arrays the C++ test needs:

    ref_asr_waveform.npy   (n_samples,)   f32 -- a whole number of encoder chunks, all valid
    ref_asr_audio.npy      (n_rows, d)    f32 -- the projected audio embeddings, i.e. the whole
                                                 encoder+projector phase's output, so a failure can be
                                                 localized to that half rather than only observed at
                                                 the end
    ref_asr_generated.npy  (n_new,)       i32 -- the token ids HF greedily generates
    ref_asr_prompt_len.npy ()             i32 -- how many prompt positions HF used, which is the
                                                 driver's own `n_past` when the loop starts, and the
                                                 one number that would let a wrong prompt still decode
                                                 plausibly

**`attn_implementation="eager"`, and this is the load-bearing line in the file.**
`speech_lm_export.WindowedAudioEncoder` reimplements the audio tower's window attention as an explicit
matmul/softmax, and against HF's eager path that rewrite is bit-identical (`max abs diff 0.000e+00`);
against its sdpa path it differs by 6.3e-05 purely in the fused kernel's accumulation order. Asking for
eager here is what lets the encoder check below be stated as a tight bound rather than a loose one.
The EXPORT must not do this -- `fuse_loom_attention` matches the sdpa shape, and under eager the
language model converts with zero fused ATTENTION nodes and therefore no KV cache. See
`qwen3_asr_export.ASRQwen3SpeechLMExportConfig.load_model`.

**Real speech rather than the synthetic noise the Whisper fixture uses.** The argument there -- noise
keeps the distribution flat, so a wrong logit moves the argmax -- does not transfer: this is an
instruction-following ASR model whose prompt asks it to transcribe, and on noise it emits a refusal or
an empty transcript, which is a *shorter* sequence and a weaker check than a real utterance. The
tensor-level encoder comparison is what carries the "a plausible transcript can hide a wrong encoder"
concern (BACKLOG.md P4.2's own finding), and it is checked first for that reason.

**The waveform is padded to a whole chunk, which is the exported encoder's contract** (see
`WindowedAudioEncoder`): a chunk is `hop_length * n_window * 2` samples -- one second here. On such a
waveform the checkpoint's own feature extractor performs no mel-axis padding of its own, so HF and the
exported graph see the identical mel and this fixture is an exact oracle rather than an approximate
one. `samples/jfk.wav` is already 11.0 s, so nothing is added for the default input.

    python3 tools/fixture_gen/reference_forward_qwen3_asr_mil.py <model_dir> <out_dir> [wav] [n_new]
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

DEFAULT_WAV = Path(__file__).resolve().parents[2] / "samples" / "jfk.wav"
DEFAULT_MAX_NEW = 64


def pad_to_chunk(waveform: np.ndarray, samples_per_chunk: int) -> np.ndarray:
    """Zero-pad up to a whole number of encoder chunks -- what a host does before calling the driver."""
    remainder = waveform.shape[0] % samples_per_chunk
    if remainder == 0:
        return waveform
    return np.pad(waveform, (0, samples_per_chunk - remainder))


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    model_dir, out_dir = sys.argv[1], Path(sys.argv[2])
    wav_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_WAV
    max_new = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_MAX_NEW
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_dir)
    extractor = processor.feature_extractor
    audio_config = model.config.audio_config

    samples_per_chunk = int(extractor.hop_length) * int(audio_config.n_window) * 2
    waveform, sample_rate = sf.read(str(wav_path), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != int(extractor.sampling_rate):
        raise SystemExit(
            f"{wav_path} is {sample_rate} Hz but this checkpoint was trained at "
            f"{extractor.sampling_rate} Hz; resample before generating the fixture rather than letting "
            f"the mel frontend see the wrong rate."
        )
    waveform = pad_to_chunk(np.asarray(waveform, dtype=np.float32), samples_per_chunk)

    conversation = [{"role": "user", "content": [{"type": "audio", "audio": waveform}]}]
    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    prompt_len = int(inputs["input_ids"].shape[1])

    with torch.inference_mode():
        # The encoder+projector half on its own, which is what the exported `encoder` topology is.
        audio = model.model.get_audio_features(
            input_features=inputs["input_features"].to(torch.float32),
            input_features_mask=inputs["input_features_mask"],
        ).pooler_output
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new)
    new_tokens = generated[0][prompt_len:].tolist()

    np.save(out_dir / "ref_asr_waveform.npy", waveform.astype(np.float32))
    np.save(out_dir / "ref_asr_audio.npy", audio.to(torch.float32).numpy())
    np.save(out_dir / "ref_asr_generated.npy", np.asarray(new_tokens, dtype=np.int32))
    np.save(out_dir / "ref_asr_prompt_len.npy", np.asarray(prompt_len, dtype=np.int32))

    rows = audio.shape[0]
    print(f"waveform {waveform.shape[0]} samples ({waveform.shape[0] // samples_per_chunk} chunks)")
    print(f"audio embeddings {tuple(audio.shape)}  ({rows} of the {prompt_len} prompt positions)")
    print(f"generated {len(new_tokens)} tokens: {new_tokens}")
    print(f"text: {processor.tokenizer.decode(new_tokens)!r}")
    print(f"wrote fixtures to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
