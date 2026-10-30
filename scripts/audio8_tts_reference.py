#!/usr/bin/env python3
"""Regenerate the reference oracle for `tests/gate/test_e2e_audio8_tts_composition.cpp`.

The Audio8 pair end to end in the reference implementation (the checkpoint's own remote code, F32,
CPU): text -> Audio8-TTS-Realistic-0.6b -> 10 codebooks per frame -> its `codec.pth` -> 44.1 kHz mono.

    ~/.venvs/piper/bin/python scripts/audio8_tts_reference.py \\
        --model ~/Dev/models/audio8-tts-realistic-0.6b --out $LOOM_FIXTURES/audio8_tts_ref

It writes, all float32 (the one dtype `tests/support/npy_fixture.h` reads):

* `text_ids.npy`: the sentence encoded with the checkpoint's own tokenizer -- the caller's side. The
  template around it is the export's, so that is what the gate checks.
* `codes_greedy.npy`: `generate(do_sample=False)`, frames x 10, run to its own EOS, so the gate also
  checks the frame the model stops on.
* `wav_<N>f.npy`: the reference codec's `decode_audio` of the first N greedy frames, at two lengths.
* `clone_ref_tokens.npy` / `clone_ref_codes.npy`: `--clip` (default `samples/jfk.wav`, 16 kHz, so the
  resample runs) as the processor formats its transcript and the codec encodes it.
* `clone_codes_<C>f.npy`: greedy with that reference, C frames.

The gate reads the reference through a voice FILE written independently by the exporter's tool --

    python -m loom_exporter.audio8_tts_voices <model> -o $OUT --wav samples/jfk.wav \\
        --text "<the transcript below>" --name voice_jfk --license "public domain"

-- and checks its inputs equal these arrays before comparing what the driver made of them.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

SENTENCE = "The quick brown fox jumps over the lazy dog."
JFK = ("And so my fellow Americans, ask not what your country can do for you, ask what you can do for "
       "your country.")
CODEC_FRAMES = (24, None)          # None: every greedy frame
CLONE_FRAMES = 24


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clip", type=Path, default=Path(__file__).resolve().parents[1] / "samples" / "jfk.wav")
    ap.add_argument("--transcript", default=JFK)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoProcessor

    def save(name, array):
        # C order: `tests/support/npy_fixture.h` refuses a Fortran-order file, which a `.T` view saves as.
        np.save(args.out / name, np.ascontiguousarray(np.asarray(array, dtype=np.float32)))
        print(f"  {name}: {np.asarray(array).shape}")

    proc = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(args.model), trust_remote_code=True, dtype=torch.float32).eval()
    save("text_ids.npy", proc.tokenizer.encode(SENTENCE, add_special_tokens=False))

    with torch.inference_mode():
        out = model.generate(**proc(text=[SENTENCE], return_tensors="pt"), do_sample=False,
                             max_new_tokens=512, return_dict_in_generate=True)
        if not bool(out.finished[0]):
            raise SystemExit("the greedy decode did not reach EOS in 512 frames")
        codes = out.codes[0]                                            # [10, F]
        save("codes_greedy.npy", codes.T.numpy())
        for n in CODEC_FRAMES:
            part = codes if n is None else codes[:, :n]
            wav, length = model.decode_audio(part[None])
            save(f"wav_{part.shape[1]}f.npy", wav[0, :int(length[0])].numpy())

        inputs = proc(text=[SENTENCE], reference_audio=[str(args.clip)], reference_text=[args.transcript],
                      return_tensors="pt")
        ref_codes, ref_len = model.encode_audio(inputs["reference_audio_values"], inputs["reference_audio_lengths"])
        save("clone_ref_codes.npy", ref_codes[0, :, :int(ref_len[0])].T.numpy())
        formatted = proc._format_reference_text(args.transcript)
        save("clone_ref_tokens.npy", proc.tokenizer.encode(formatted, add_special_tokens=False))
        clone = model.generate(**inputs, do_sample=False, max_new_tokens=CLONE_FRAMES)
        save(f"clone_codes_{CLONE_FRAMES}f.npy", clone[0].T.numpy())


if __name__ == "__main__":
    main()
