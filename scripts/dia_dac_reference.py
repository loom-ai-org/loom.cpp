#!/usr/bin/env python3
"""Regenerate the `transformers` oracle for `tests/gate/test_e2e_dia_dac_composition.cpp`.

**This is the family-10 pair, end to end, in the reference implementation**: text -> Dia -> nine
delayed code streams -> realign -> DAC -> waveform. Two files on the loom side and two models here,
chained the same way, so what the gate compares is the composition and not either half alone.

    ~/.venvs/piper/bin/python scripts/dia_dac_reference.py \
        --model ~/Dev/models/dia-1.6b --dac ~/Dev/models/dac-44khz \
        --out $LOOM_FIXTURES/dia_dac_ref

It writes one `.npy` per clip length -- `codes_<N>f.npy` (float32, frame-major, N x 9, the integers
written as floats because that is the one dtype `tests/support/npy_fixture.h` reads) and
`wav_<N>f.npy` (float32, the waveform) -- plus `prompt_ids.npy`. Nothing lands in the repo: these are
gate fixtures, which live under `$LOOM_FIXTURES` by the derived rule.

**Greedy, and guidance-free by default**, because this fixture is about the JOIN between the two
files rather than about the decoder: the codes only have to be a real generation, and the cheapest
real one is the best. `--guidance` is here so the pair can be re-checked under the checkpoint's own
decoding if that is ever wanted; the guided path's own oracle is `dia_reference_codes.py`, which
compares codes rather than waveforms and so isolates it. Greedy on both sides is not a simplification
but the only thing an exact comparison can be -- two samplers running one algorithm from different
RNG streams agree on nothing.

**Why the codes are captured as well as the waveform.** They are what makes a failure legible. The
composition has exactly three ways to break and they are not distinguishable from the waveform alone:
Dia emits the wrong codes, the realignment hands DAC the wrong layout, or DAC decodes correctly from
the right codes and the join dropped a frame. With the codes checked first, a waveform mismatch can
only be the third.

**Two clip lengths, and that is the point of the row rather than thoroughness.** Family 11's own
lesson is that a codec decoder can return one frame's worth of audio for every input and raise
nothing (`test_codec_output_length_follows_the_input`); one clip length cannot see that, because any
constant is consistent with itself. Two lengths whose sample counts differ by exactly the hop times
the frame difference is the check that a length actually follows its input.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import DacModel, DiaForConditionalGeneration, DiaProcessor


def generate_codes(model, processor, text: str, frames: int, guidance: float | None):
    """`frames` audio frames of realigned, frame-major codes, exactly as the loom driver returns them.

    The row/frame translation is the subtle half and `dia_reference_codes.py`'s docstring has it in
    full: `transformers` takes `max_new_tokens` in ROWS and `DiaEOSDelayPatternLogitsProcessor` forces
    an EOS at `max_length - max(delay) - 1`, so N frames is `N + max(delay) + 1` rows. The loom
    driver's own `max_new_tokens` counts frames and takes N directly.
    """
    delay = list(model.config.delay_pattern)
    n_channels = model.config.decoder_config.num_channels
    bos, pad = model.config.bos_token_id, model.config.pad_token_id

    encoded = processor(text=[text])
    with torch.no_grad():
        # `guidance_scale=None` is how `DiaGenerationMixin` spells "no CFG processor"; a scale of 1.0
        # says the same thing with a number, and its own check rejects anything <= 1.
        scale = guidance if (guidance or 1.0) > 1.0 else None
        out = model.generate(**encoded, do_sample=False, temperature=1.0, guidance_scale=scale,
                             max_new_tokens=frames + max(delay) + 1)
    seq = out[0]
    # `DiaProcessor.batch_decode`'s delay revert and its window, written out rather than called --
    # `batch_decode` goes on to run its own codec, and this script chains an explicit one so that the
    # two halves stay visible.
    start = int((seq[:, 0] == bos).sum())
    end = int(seq.shape[0] - (seq[:, 0] == pad).sum() - 1)
    codes = np.array([[int(seq[t + delay[k], k]) for k in range(n_channels)]
                      for t in range(start, end)], dtype=np.int64)
    if codes.shape[0] != frames:
        raise SystemExit(
            f"asked for {frames} frames and got {codes.shape[0]} -- the model emitted EOS on its own "
            f"before the ceiling. Capture that count instead, and tell the test about it."
        )
    return codes, encoded["input_ids"][0].tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the Dia checkpoint directory")
    ap.add_argument("--dac", required=True,
                    help="the DAC checkpoint directory. Dia's own audio_tokenizer_config.json names "
                         "`descript/dac_44khz`; a different codec is a different model.")
    ap.add_argument("--text", default="[S1] Hello world.",
                    help="the sentence to capture; must match the test's kPromptIds")
    ap.add_argument("--frames", type=int, nargs="+", default=[16, 32],
                    help="the clip lengths, in AUDIO frames")
    ap.add_argument("--guidance", type=float, default=None,
                    help="classifier-free guidance scale, in the checkpoint's own centring. Omit for "
                         "the guidance-free decode, which is what the composition fixtures use -- the "
                         "guided path has its own oracle in `dia_reference_codes.py`, and this one is "
                         "about the JOIN rather than about the sampler.")
    ap.add_argument("--out", required=True, help="directory to write the .npy fixtures into")
    args = ap.parse_args()

    processor = DiaProcessor.from_pretrained(args.model)
    dia = DiaForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager").eval()
    dac = DacModel.from_pretrained(args.dac, dtype=torch.float32).eval()

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_ids = None
    for frames in args.frames:
        codes, ids = generate_codes(dia, processor, args.text, frames, args.guidance)
        if prompt_ids is None:
            prompt_ids = ids
            np.save(out_dir / "prompt_ids.npy", np.array(ids, dtype=np.float32))
        with torch.no_grad():
            # [n_frames, n_codebooks] -> [1, n_codebooks, n_frames], which is what `decode` takes.
            # The loom side spells the same transpose inside `_CodecDecodeWrapper`, so its caller
            # hands over the frame-major array this one holds.
            wav = dac.decode(audio_codes=torch.tensor(codes, dtype=torch.long).T[None]).audio_values
        wav = wav.reshape(-1).numpy().astype(np.float32)
        np.save(out_dir / f"codes_{frames}f.npy", codes.astype(np.float32))
        np.save(out_dir / f"wav_{frames}f.npy", wav)
        print(f"{frames} frames: {codes.shape[0]}x{codes.shape[1]} codes -> {wav.size} samples "
              f"({wav.size / codes.shape[0]:.1f} per frame), peak {np.abs(wav).max():.4f}")

    print(f"\nprompt ids: {prompt_ids}")
    print(f"written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
