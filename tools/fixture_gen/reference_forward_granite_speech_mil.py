#!/usr/bin/env python3
"""Ground truth for the MIL-exported Granite Speech (`test_e2e_granite_speech_mil_export.cpp`,
BACKLOG.md P4.3c).

Runs HF's own `GraniteSpeechForConditionalGeneration` -- the library, not this repo's exporter -- over
real speech, and writes the arrays the C++ test needs:

    ref_asr_waveform.npy   (n_samples,)   f32 -- a whole number of encoder chunks, all valid
    ref_asr_audio.npy      (n_rows, d)    f32 -- the projected audio embeddings, i.e. the whole
                                                 conformer+Q-Former phase's output, so a failure can be
                                                 localized to that half rather than only observed at
                                                 the end
    ref_asr_generated.npy  (n_new,)       i32 -- the token ids HF greedily generates
    ref_asr_prompt_len.npy ()             i32 -- how many prompt positions HF used, which is the
                                                 driver's own `n_past` when the loop starts, and the
                                                 one number that would let a wrong prompt still decode
                                                 plausibly

**The default attention implementation, unlike the Qwen3-ASR fixture.** That one asks for `eager`
because its exported encoder rewrite is bit-identical to HF's eager path and 6.3e-05 from its sdpa one.
Here the flag would be inert on the half it would be asked for: `GraniteSpeechConformerAttention` pins
`SDPBackend.MATH` itself, and `Blip2QFormerModel` declares `_supports_sdpa = False`, so the encoder runs
the same kernels either way. Leaving the default means the language model is traced and referenced
through the same path, which is what the export does.

**The instruction must match the export's.** Granite Speech's chat template is a plain
`USER: {content}\\n ASSISTANT:` and the audio placeholder is written into the content by the caller, so
the text after the audio is a choice rather than a template constant. The driver bakes
`ASRGraniteSpeechExportConfig.instruction` into its prompt; this file defaults to the same string and
takes an override for the same reason the config does.

**The waveform is padded to a whole chunk, which is the exported encoder's contract** (see
`granite_speech_export.ConformerQFormerEncoder`): a chunk is `lcm(context_size, window_size)` encoder
frames -- 600 here, i.e. 192000 samples, twelve seconds. On such a waveform the extractor's own
"drop the final mel frame if the count is odd" removes exactly the frame the exported frontend already
drops, so HF and the exported graph see the identical features and this fixture is an exact oracle.
`samples/jfk.wav` is 11.0 s, so it is padded to one 12 s chunk.

    python3 tools/fixture_gen/reference_forward_granite_speech_mil.py <model_dir> <out_dir> \\
        [wav] [n_new] [instruction]
"""
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration

DEFAULT_WAV = Path(__file__).resolve().parents[2] / "samples" / "jfk.wav"
DEFAULT_MAX_NEW = 64
# `ASRGraniteSpeechExportConfig.instruction`'s own default, which is the model card's transcription
# prompt. Imported rather than restated would couple this generator to the exporter package; it is
# printed below so a mismatch with the driver's prompt is visible rather than silent.
DEFAULT_INSTRUCTION = "can you transcribe the speech into a written format?"


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
    instruction = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_INSTRUCTION
    out_dir.mkdir(parents=True, exist_ok=True)

    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_dir)
    extractor = processor.audio_processor

    # The chunk arithmetic, from the checkpoint rather than hardcoded -- the same expression
    # `ASRGraniteSpeechExportConfig.audio_geometry` evaluates.
    context_size = int(model.config.encoder_config.context_size)
    window_size = int(model.projector.window_size)
    encoder_frames = context_size * window_size // gcd(context_size, window_size)
    samples_per_chunk = encoder_frames * 2 * int(extractor.melspec_kwargs["hop_length"])

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

    prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{processor.audio_token}{instruction}"}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=prompt, audio=torch.from_numpy(waveform).unsqueeze(0),
                       return_tensors="pt")
    prompt_len = int(inputs["input_ids"].shape[1])

    with torch.inference_mode():
        # The conformer+Q-Former half on its own, which is what the exported `encoder` topology is.
        audio = model.get_audio_features(inputs["input_features"].to(torch.float32))
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new)
    new_tokens = generated[0][prompt_len:].tolist()
    audio = audio.reshape(-1, audio.shape[-1])

    np.save(out_dir / "ref_asr_waveform.npy", waveform.astype(np.float32))
    np.save(out_dir / "ref_asr_audio.npy", audio.to(torch.float32).numpy())
    np.save(out_dir / "ref_asr_generated.npy", np.asarray(new_tokens, dtype=np.int32))
    np.save(out_dir / "ref_asr_prompt_len.npy", np.asarray(prompt_len, dtype=np.int32))

    print(f"instruction: {instruction!r}")
    print(f"waveform {waveform.shape[0]} samples ({waveform.shape[0] // samples_per_chunk} chunks of "
          f"{samples_per_chunk})")
    print(f"audio embeddings {tuple(audio.shape)}  ({audio.shape[0]} of the {prompt_len} prompt "
          f"positions)")
    print(f"generated {len(new_tokens)} tokens: {new_tokens}")
    print(f"text: {processor.tokenizer.decode(new_tokens)!r}")
    print(f"wrote fixtures to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
