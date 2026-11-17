#!/usr/bin/env python3
"""tiny-voca's own `speak_text`, as the oracle `test_e2e_tiny_voca` is held to.

    DIZIONE_DATA=.../tiny-voca/phonemizer PYTHONPATH=<tiny-voca>/src:<dizione>/src \\
        tiny_voca_reference.py OUT_DIR "text" [--voice F2|M2] [--speed 1.0] [--pitch-shift 0] [--pitch-range 1]
        [--seed 0]

Writes one case directory:

    case.json          the arguments, the sentences as spoken, per-sentence token and frame counts
    tokens.npy         every sentence's phoneme ids, end to end (each `<bos> ... <eos>`)
    stress.npy         their stress values (0 none, 1 primary, 2 secondary)
    durations.npy      the frames per token the model predicted, end to end
    noise.npy          each sentence's noise draw (`randn(frames * hop)`, seeded `seed + sentence`)
    audio.npy          `speak_text`'s waveform (joined, peak-limited to 0.99)
    raw.npy            the same before the peak limit

`speak_text` reads one sentence at a time (`sentences(finished(text))`) and joins the audio; the noise of
sentence i comes from `torch.Generator().manual_seed(seed + i)`, which is what `noise.npy` records so the
engine can be handed the same numbers.
"""
import argparse
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("text")
ap.add_argument("--checkpoint", default=os.path.expanduser("~/Dev/models/tiny-voca/tiny-voca.pt"))
ap.add_argument("--voice", default="F2")
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--pitch-shift", type=float, default=0.0)
ap.add_argument("--pitch-range", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

import tiny_voca.infer as infer
from tiny_voca.model import TinyVoca

torch.set_num_threads(1)
os.makedirs(a.out, exist_ok=True)
infer.match_frontend(a.checkpoint)
model = infer.load_model(a.checkpoint)

record = {"ids": [], "stress": [], "durations": [], "noise": [], "frames": []}
synthesize = TinyVoca.synthesize


def recording(self, tokens, stress, lengths, speakers, speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0,
              pitch_jitter=0.0, voiced_vowels=True):
    wave, samples, details = synthesize(self, tokens, stress, lengths, speakers, speed, pitch_shift, seed,
                                        pitch_range, pitch_jitter, voiced_vowels)
    n = int(details["durations"].sum())
    record["ids"] += tokens[0].tolist()
    record["stress"] += stress[0].tolist()
    record["durations"] += details["durations"][0].tolist()
    record["noise"].append(torch.randn(n * self.config.hop, generator=torch.Generator().manual_seed(seed)).numpy())
    record["frames"].append(n)
    return wave, samples, details


TinyVoca.synthesize = recording
with torch.inference_mode():
    raw, spoken = infer.speak_text(model, a.text, a.voice, a.speed, a.pitch_shift, a.seed, a.pitch_range, peak=None)
audio = infer.limit_peak(raw)

save = lambda name, value: np.save(os.path.join(a.out, name), np.asarray(value, dtype=np.float32))
save("tokens.npy", record["ids"])
save("stress.npy", record["stress"])
save("durations.npy", record["durations"])
save("noise.npy", np.concatenate(record["noise"]))
save("audio.npy", audio.numpy())
save("raw.npy", raw.numpy())
case = dict(text=a.text, voice=a.voice, speed=a.speed, pitch_shift=a.pitch_shift, pitch_range=a.pitch_range,
            seed=a.seed, sentences=[str(p) for p in spoken], tokens=[len(p) for p in spoken],
            frames=record["frames"], samples=int(audio.shape[-1]))
json.dump(case, open(os.path.join(a.out, "case.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(case, ensure_ascii=False)[:300])
