---
type: retro
date: 2026-10-10
domain: text-frontends
tags: [phonemes, piper, vits, sanotts, framing, reference]
---

# Retro-081: Piper Has Two Framings

## The Issue

sanoTTS's piperlite voices (2026-10-10) are distilled from Piper/VITS teachers and keep Piper's
`phoneme_id_map`. Upstream's own front end built `[1, 0, p1, 0, ..., pn, 0, 2]`. The engine's Piper
assembly, which the VITS export declares, builds `[1, p1, 0, ..., pn, 0, 2]`. Both claim to be Piper's.

## Root Cause

Piper has two `phonemes_to_ids`, and they differ by exactly that one id:

* **piper-phonemize** (C++), which Piper's TRAINING preprocessing calls
  (`piper_train.preprocess` → `phoneme_ids_espeak`), puts a pad right after BOS when `interspersePad`
  is set. **piper1-gpl**, the maintained successor that sanoTTS uses, does the same.
* **The old `python_run` runtime** (`piper/voice.py`) does not. It is what
  `tools/convert_piper_vits/reference_forward_vits.py` copied, so it is what the VITS export and
  `PhonemeVocab` reproduce.

So **Piper's VITS voices were trained with the pad after BOS**, and loom's VITS door feeds them one id
short. VITS is robust to it: Whisper is word-perfect through the door on `vits-piper-en-gb-miro`. That
is why it was never noticed. A 1.46M-parameter student is not robust in the same way.

## What Changed

* `tokenizer.ggml.phoneme.blank_after_bos` (engine `PhonemeVocab`, exporter writer). It is optional and
  absent means the old assembly, so every existing file is unchanged.
* sanoTTS declares it. The VITS export does **not** yet: switching it changes every published Piper
  voice's audio, so it waits for a measured A/B and a republish (hub item).

## Takeaway

**When a reference has a runtime and a training pipeline, the training pipeline is the reference.**
A model learned whatever its preprocessing fed it. A later runtime that frames the input differently
is a second, unvalidated front end, however official it looks. Find the code that built the training
data before copying an assembly.
