#!/usr/bin/env python3
"""Paradee (loom-exporter `paradee_export.py`) by upstream's own training code, as the oracle
`test_e2e_paradee` is held to.

    LOOM_PARADEE_REPO=<paradee clone, default ~/Dev/paradee> paradee_reference.py OUT_DIR "<phonemes>" [--voice af_heart]
        [--speed 1.0] [--seed 0] [--model ~/Dev/models/paradee-8m-v1.3]

The phonemes are cut into sentences where upstream's runtime cuts its text (after `.`, `!`, `?` or `…`
followed by whitespace), each wrapped `[0, ids, 0]`, and each sentence is run through upstream's
`TextStudent` and `VDecoder` (`training/scripts/student.py`, `vhead.py`) -- the modules the export traces,
unstripped, with `torch.stft`/`istft` where the export has convolutions. Sentence i's noise is
`torch.randn(samples, generator=manual_seed(seed + i))`, which is what the driver draws (`aten_randn`).

Writes one case directory:

    case.json      the arguments, the voice's row, the sentences, per-sentence token and sample counts
    tokens.npy     every sentence's ids, end to end, each wrapped in 0s
    durations.npy  the frames per token the model predicted, end to end
    noise.npy      every sentence's noise draw, end to end
    audio.npy      the sentences' waveforms, joined
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("phonemes")
ap.add_argument("--model", default=os.path.expanduser("~/Dev/models/paradee-8m-v1.3"))
ap.add_argument("--voice", default="af_heart")
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

sys.path.insert(0, os.path.join(os.environ.get("LOOM_PARADEE_REPO", os.path.expanduser("~/Dev/paradee")), "training",
                                "scripts"))
from student import TextStudent, alignment   # noqa: E402
from vhead import FHOP, VDecoder             # noqa: E402

torch.set_num_threads(1)
cfg = json.load(open(os.path.join(a.model, "config.json")))
voices, vocab = cfg["voices"], cfg["vocab"]
ts = TextStudent("s+mlp", n_voices=len(voices)).eval()
ts.load_state_dict(torch.load(os.path.join(a.model, "pytorch", "text_side.pt"), map_location="cpu",
                              weights_only=True)["model"])
vd = VDecoder(head="v2", dim=160, blocks=4, n_voices=len(voices)).eval()
vd.load_state_dict(torch.load(os.path.join(a.model, "pytorch", "decoder.pt"), map_location="cpu",
                              weights_only=True)["model"])
voice = voices.index(a.voice)

sentences = [s for s in re.split(r"(?<=[.!?…])\s+", a.phonemes.strip()) if s.strip()]
record = {"ids": [], "durations": [], "noise": [], "audio": [], "tokens": [], "samples": []}
with torch.no_grad():
    for i, s in enumerate(sentences):
        ids = [0] + [vocab[c] for c in s if c in vocab] + [0]
        x = torch.LongTensor([ids])
        n = torch.tensor([len(ids)])
        vo = torch.tensor([voice])
        pd, _, _, asr = ts(x, n, voice=vo)
        dur = torch.clamp(torch.round(pd[0] / a.speed), min=1).long()
        aln = alignment(dur)[None]
        _, f0, nn_, _ = ts(x, n, aln, vo)
        noise = torch.randn(1, f0.shape[1] * FHOP, generator=torch.Generator().manual_seed(a.seed + i))
        wave = vd(asr @ aln, f0, nn_, noise=noise, voice=vo)[0]
        record["ids"] += ids
        record["durations"] += dur.tolist()
        record["noise"].append(noise[0].numpy())
        record["audio"].append(wave.numpy())
        record["tokens"].append(len(ids))
        record["samples"].append(int(wave.shape[0]))

os.makedirs(a.out, exist_ok=True)
save = lambda name, value: np.save(os.path.join(a.out, name), np.asarray(value, dtype=np.float32))
save("tokens.npy", record["ids"])
save("durations.npy", record["durations"])
save("noise.npy", np.concatenate(record["noise"]))
save("audio.npy", np.concatenate(record["audio"]))
case = dict(phonemes=a.phonemes, voice=a.voice, voice_index=voice, speed=a.speed, seed=a.seed,
            sentences=sentences, tokens=record["tokens"], samples=record["samples"])
json.dump(case, open(os.path.join(a.out, "case.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(case, ensure_ascii=False)[:300])
