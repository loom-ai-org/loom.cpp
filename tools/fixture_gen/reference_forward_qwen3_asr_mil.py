#!/usr/bin/env python3
"""Ground truth for the MIL-exported Qwen3-ASR (`test_e2e_qwen3_asr_mil_export.cpp`, BACKLOG.md P4.3).

Runs HF's own `Qwen3ASRForConditionalGeneration` -- the library, not this repo's exporter -- over real
speech, and writes the arrays the C++ test needs:

    ref_asr_waveform.npy   (n_padded,)    f32 -- what a HOST hands the driver: the real audio, zero-
                                                 padded up to a whole number of encoder chunks
    ref_asr_valid.npy      ()             i32 -- how many of those samples are real
    ref_asr_encoder_in.npy (n_padded,)    f32 -- what the DRIVER hands the encoder topology: the same
                                                 waveform with the frontend's own STFT reflection
                                                 written over the head of the padding, so the C++ test
                                                 can drive that topology directly without restating
                                                 `driver_components.WaveformValidLength`
    ref_asr_audio.npy      (n_rows, d)    f32 -- the projected audio embeddings for the REAL audio,
                                                 i.e. the whole encoder+projector phase's output, so a
                                                 failure can be localized to that half rather than only
                                                 observed at the end
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

**HF is given the UNPADDED audio, and that is what makes this an oracle at all** (BACKLOG.md P4.3e).
It used to be handed the padded waveform, so both sides read the same trailing silence and the fixture
could not see the padding question. HF never pads: `Qwen3ASREncoder` packs only the post-CNN positions
its `input_features_mask` marks valid. The exported encoder is given the padded waveform *and* the real
length, and is required to produce those same rows -- which it now does **exactly**, bit for bit, where
before the masking it was 1.0e-01 out on a reference whose absmax is 0.128.

**The default input is deliberately trimmed to leave a partial chunk.** A chunk here is one second
(`hop_length * n_window * 2` samples) and `samples/jfk.wav` is exactly 11.0 s, so on the raw file there
would be no padding at all and this fixture could not exercise the one thing it exists for. When the
wav's length is already a whole number of chunks, half a chunk is dropped -- derived from the
checkpoint's own geometry rather than written down as a sample count, so a checkpoint with a different
chunk drops a different amount and still lands mid-chunk.

    python3 tools/fixture_gen/reference_forward_qwen3_asr_mil.py <model_dir> <out_dir> \\
        [wav] [n_new] [valid_samples]
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
        return waveform.copy()
    return np.pad(waveform, (0, samples_per_chunk - remainder))


def reflect_tail(padded: np.ndarray, valid: int, n_fft: int) -> np.ndarray:
    """The driver's own edge repair (`driver_components.WaveformValidLength`), so the C++ test can
    drive the `encoder` topology directly on the array this returns.

    `torch.stft(center=True)` reflects the signal by `n_fft // 2` at each end, so the mel frames that
    straddle the caller's real end are computed against a mirror of the audio and not against the
    caller's zeros. Restated here for the same reason the chunk arithmetic below is: this file is HF's
    side of the comparison, and a fixture that took the exporter's word for a transform would not be
    an independent check of it.
    """
    out = padded.copy()
    room = min(n_fft // 2, out.shape[0] - valid)
    for i in range(1, room + 1):
        out[valid + i - 1] = padded[valid - 1 - i]
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    model_dir, out_dir = sys.argv[1], Path(sys.argv[2])
    wav_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_WAV
    max_new = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_MAX_NEW
    requested_valid = int(sys.argv[5]) if len(sys.argv) > 5 else None
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
    waveform = np.asarray(waveform, dtype=np.float32)
    # Land mid-chunk unless the caller says otherwise: on a wav that already fills its last chunk there
    # is no padding, and the padding is what this fixture is for. See the module docstring.
    if requested_valid is None:
        requested_valid = waveform.shape[0]
        if requested_valid % samples_per_chunk == 0:
            requested_valid -= samples_per_chunk // 2
    valid_samples = min(int(requested_valid), int(waveform.shape[0]))
    waveform = waveform[:valid_samples]
    padded = pad_to_chunk(waveform, samples_per_chunk)
    encoder_in = reflect_tail(padded, valid_samples, int(extractor.n_fft))

    # The REAL audio, not the padded waveform: HF's own pipeline never pads, and the whole point of
    # this fixture is that the exported encoder reproduces what HF computes for the real length.
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

    np.save(out_dir / "ref_asr_waveform.npy", padded.astype(np.float32))
    np.save(out_dir / "ref_asr_valid.npy", np.asarray(valid_samples, dtype=np.int32))
    np.save(out_dir / "ref_asr_encoder_in.npy", encoder_in.astype(np.float32))
    np.save(out_dir / "ref_asr_audio.npy", audio.to(torch.float32).numpy())
    np.save(out_dir / "ref_asr_generated.npy", np.asarray(new_tokens, dtype=np.int32))
    np.save(out_dir / "ref_asr_prompt_len.npy", np.asarray(prompt_len, dtype=np.int32))

    rows = audio.shape[0]
    print(f"waveform {valid_samples} real samples -> {padded.shape[0]} padded "
          f"({padded.shape[0] // samples_per_chunk} chunks of {samples_per_chunk})")
    print(f"audio embeddings {tuple(audio.shape)}  ({rows} of the {prompt_len} prompt positions; the "
          f"padded waveform's own row count is "
          f"{padded.shape[0] // samples_per_chunk * int(audio_config.max_position_embeddings)})")
    print(f"generated {len(new_tokens)} tokens: {new_tokens}")
    print(f"text: {processor.tokenizer.decode(new_tokens)!r}")
    print(f"wrote fixtures to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
