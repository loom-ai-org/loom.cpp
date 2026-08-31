#!/usr/bin/env python3
"""Turn a sentence into the phoneme ids ONE exported TTS model wants, reading the table out of that
model's own GGUF -- and, for Kokoro, pull the reference style vector it also needs.

WHY THIS EXISTS. The ASR oracle (Retro-006, and P4.13/P4.28's intelligibility work) is only an oracle
if the model is fed REAL speech: the gate tests drive these families with synthetic token ids like
{5, 42, 7, 88, ...}, which produce audio no ASR can or should transcribe. Getting from a sentence to
ids is four small facts per family, all of them in the file, and all of them easy to get wrong -- so
they live here rather than in a shell history.

    scripts/tts_ids.py "Hey, can you shut down the computer, my friend?" model.gguf ids.txt
    # kokoro additionally writes ids.txt.ref_s

THE FOUR FACTS, AND THE TWO THAT BITE.

  * `tokenizer.ggml.tokens` is the id-indexed symbol table. Every family here is IPA, so espeak-ng's
    output maps character by character; a symbol the table lacks is DROPPED and reported, because a
    silent drop is how you get audio that is subtly not the sentence.
  * `interleave_blank` says whether a blank goes between every phoneme (piper does, the rest do not).
  * **`bos_id`/`eos_id`/`blank_id` of -1 is a SENTINEL meaning "this model has none", not an id.**
    Matcha and StyleTTS2 declare `eos_id = -1` and StyleTTS2 `blank_id = -1`. Appending a literal -1
    reaches the engine as an out-of-range GET_ROWS.
  * **Kokoro's `loom.default_style.ref_s` is a VOICE PACK, not a vector**: [510, 256], indexed by the
    phoneme count, and the driver wants one row of 256. Passing the whole 130560-float tensor, or row
    0, is a different voice from the one the checkpoint means at that length.

Run with the piper venv (`~/.venvs/piper/bin/python3`), which has `phonemizer` and espeak-ng.
"""
import sys

import numpy as np
from gguf import GGUFReader


def _phoneme_field(reader, key, default=-1):
    """A `tokenizer.ggml.phoneme.*` scalar, read as SIGNED -- the -1 sentinels above are written as
    int32 and come back as 4294967295 if that is forgotten."""
    for f in reader.fields.values():
        if f.name == f"tokenizer.ggml.phoneme.{key}":
            return int(np.int32(f.parts[f.data[0]][0]))
    return default


def ids_for(sentence, gguf_path, voice_out=None):
    from phonemizer.backend import EspeakBackend

    reader = GGUFReader(gguf_path)
    tokens_field = next(f for f in reader.fields.values() if f.name == "tokenizer.ggml.tokens")
    tokens = [str(bytes(tokens_field.parts[i]), "utf-8") for i in tokens_field.data]
    sym2id = {t: i for i, t in enumerate(tokens)}

    bos = _phoneme_field(reader, "bos_id")
    eos = _phoneme_field(reader, "eos_id")
    blank = _phoneme_field(reader, "blank_id")
    interleave = bool(_phoneme_field(reader, "interleave_blank", 0))

    ipa = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True).phonemize(
        [sentence], strip=True)[0]
    core, missing = [], []
    for ch in ipa:
        (core if ch in sym2id else missing).append(sym2id.get(ch, ch))
    if missing:
        print(f"  WARNING: {len(missing)} symbol(s) not in this model's table and DROPPED: "
              f"{sorted(set(missing))}", file=sys.stderr)

    seq = ([bos] if bos >= 0 else [])
    for x in core:
        seq.append(x)
        if interleave and blank >= 0:
            seq.append(blank)
    if eos >= 0:
        seq.append(eos)

    print(f"  ipa: {ipa}")
    print(f"  bos={bos} eos={eos} blank={blank} interleave={interleave} "
          f"phonemes={len(core)} ids={len(seq)}")

    if voice_out is not None:
        pack = next((t for t in reader.tensors if t.name == "loom.default_style.ref_s"), None)
        if pack is not None:
            rows = np.array(pack.data).reshape(-1, 256)
            row = min(len(core), rows.shape[0] - 1)
            with open(voice_out, "w") as f:
                f.write(" ".join("%.8g" % v for v in rows[row]))
            print(f"  ref_s: pack {rows.shape}, wrote row {row} (the phoneme count) to {voice_out}")
    return seq


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    sentence, gguf, out = sys.argv[1], sys.argv[2], sys.argv[3]
    seq = ids_for(sentence, gguf, voice_out=out + ".ref_s")
    with open(out, "w") as f:
        f.write(" ".join(map(str, seq)))
    print(f"  wrote {out}")
