#!/usr/bin/env python3
"""OmniVoice's own `generate`, with its draws recorded, as the oracle `test_e2e_omnivoice` is held to.

    omnivoice_reference.py OUT_DIR "text" [--instruct "female, british accent"] [--lang en]
        [--ref clip.wav --ref-text "what it says"] [--chunk-threshold S --chunk-duration S] [--seed N]

Writes one case directory:

    case.json              the arguments, the upstream wall time, the per-chunk frame counts
    tokens.npy             the text's ids (the tokenizer, no special tokens): the driver's `tokens`
    instruct.npy           the instruct's ids, when given                   -> `instruct`
    ref_text.npy           the transcript's ids, when cloning               -> `ref_text`
    waveform.npy           the clip at 24 kHz, when cloning                 -> `waveform`
    language.npy           the 1-based index into the sorted language codes  -> `language`
    draws.npy              every position-noise uniform, in draw order     -> `draws`
    input_ids_<i>.npy      chunk i's conditional prompt `[8, L]` (prefix, reference codes, MASK target)
    tokens_<i>.npy         chunk i's codes `[8, T_i]`
    picks.npy, pick_counts.npy   every step's top-k cell indices (for bisecting a divergence)
    audio.npy              the post-processed waveform `generate` returned

Needs the OmniVoice package on PYTHONPATH (a clone of github.com/k2-fsa/OmniVoice), transformers >= 5.3,
`pydub` and `accelerate`. The model runs at f32 on the CPU with sdpa attention. Upstream draws with
`torch.rand_like`; here every draw comes from one seeded generator and is recorded, so the engine can be
handed the same numbers (feedback: the same seed is not the same noise).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("text")
ap.add_argument("--model", default=os.path.expanduser("~/Dev/models/omnivoice"))
ap.add_argument("--instruct")
ap.add_argument("--lang")
ap.add_argument("--ref")
ap.add_argument("--ref-text")
ap.add_argument("--chunk-threshold", type=float)
ap.add_argument("--chunk-duration", type=float)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()

import omnivoice.models.omnivoice as om
from omnivoice import OmniVoice
from omnivoice.utils.lang_map import LANG_IDS

torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
save = lambda name, value: np.save(os.path.join(a.out, name), value)

draws, picks, prompts, codes, post_inputs = [], [], [], [], []
generator = torch.Generator().manual_seed(a.seed)


def gumbel(logits, temperature):
    scaled = logits / temperature
    u = torch.rand(scaled.shape, generator=generator, dtype=scaled.dtype)
    draws.append(u.numpy().reshape(-1).copy())
    return scaled + -torch.log(-torch.log(u + 1e-10) + 1e-10)


_topk = torch.topk


def topk(x, k, *args, **kwargs):
    result = _topk(x, k, *args, **kwargs)
    if x.dim() == 1:                      # the step's selection; `_filter_top_k` (sampling) is 2-D+
        picks.append(result.indices.numpy().copy())
    return result


om._gumbel_sample = gumbel
torch.topk = topk
prepare = om.OmniVoice._prepare_inference_inputs
om.OmniVoice._prepare_inference_inputs = lambda self, *x, **k: (
    lambda r: (prompts.append(r["input_ids"][0].numpy().copy()), r)[1])(prepare(self, *x, **k))
iterate = om.OmniVoice._generate_iterative
om.OmniVoice._generate_iterative = lambda self, task, cfg: (
    lambda r: (codes.extend(t.numpy().copy() for t in r), r)[1])(iterate(self, task, cfg))

model = OmniVoice.from_pretrained(a.model, dtype=torch.float32)
tok = model.text_tokenizer
ids = lambda s: np.array(tok(s, add_special_tokens=False).input_ids, np.float32)

kwargs = dict(text=a.text, num_step=a.steps)
save("tokens.npy", ids(a.text))
if a.instruct:
    kwargs["instruct"] = a.instruct
    save("instruct.npy", ids(a.instruct))
if a.lang:
    kwargs["language"] = a.lang
    save("language.npy", np.array([sorted(LANG_IDS).index(a.lang) + 1], np.float32))
if a.ref:
    import soundfile as sf

    wav, sr = sf.read(a.ref, dtype="float32")
    if sr != 24000 or wav.ndim != 1:
        raise SystemExit("the reference clip must be mono 24 kHz (what the engine's door takes)")
    kwargs["ref_audio"] = (torch.from_numpy(wav).view(1, -1), sr)
    kwargs["ref_text"] = a.ref_text
    save("waveform.npy", wav)
    save("ref_text.npy", ids(a.ref_text))
if a.chunk_threshold:
    kwargs["audio_chunk_threshold"] = a.chunk_threshold
if a.chunk_duration:
    kwargs["audio_chunk_duration"] = a.chunk_duration

start = time.time()
audio = model.generate(**kwargs)[0]
seconds = time.time() - start

save("audio.npy", audio.astype(np.float32))
save("draws.npy", np.concatenate(draws).astype(np.float32))
save("picks.npy", np.concatenate(picks))
save("pick_counts.npy", np.array([len(p) for p in picks]))
for i, p in enumerate(prompts):
    save(f"input_ids_{i}.npy", p.astype(np.float32))      # float32: what tests/support/npy_fixture.h reads
for i, c in enumerate(codes):
    save(f"tokens_{i}.npy", c.astype(np.float32))
case = dict(text=a.text, instruct=a.instruct, lang=a.lang, ref=a.ref, ref_text=a.ref_text,
            chunk_threshold=a.chunk_threshold, chunk_duration=a.chunk_duration, seed=a.seed, steps=a.steps,
            seconds=seconds, samples=int(audio.shape[-1]), frames=[int(c.shape[1]) for c in codes])
json.dump(case, open(os.path.join(a.out, "case.json"), "w"), indent=1)
print(json.dumps(case))
