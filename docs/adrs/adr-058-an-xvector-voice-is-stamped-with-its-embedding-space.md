---
type: adr
status: accepted
date: 2026-09-30
tags: [voices, voice-files, speecht5, family-9b, exporter, hosts]
supersedes: []
---

# ADR-058: An X-Vector Voice Is Stamped With Its Embedding Space

## Context

[ADR-045](adr-045-a-voice-is-a-file-of-driver-inputs-stamped-with-its-weights.md) made a voice a file
of driver inputs, stamped with `loom.voice.compat`, and `loom::load_voice` refuses a file whose stamp
differs from the model's. Every stamp so far has been a fingerprint of the WEIGHTS: Pocket-TTS's
voice is the flow LM's KV cache, and Voxtral's and MOSS-TTS's are rows those weights produced. A
voice made for other weights is wrong there, so refusing it is right.

SpeechT5's voice is different. It is a 512-float x-vector from SpeechBrain's
`spkrec-xvect-voxceleb` extractor, which the decoder prenet normalises and projects. That vector is
the EXTRACTOR's output, not the model's state. It fits any SpeechT5 trained on that extractor's
embeddings, and SpeechT5 is fine-tuned often, for other languages and speakers.

## Options

* **A weights fingerprint** (for example of the prenet's speaker projection), as ADR-045's other
  models do. Rejected. It would refuse the same CMU ARCTIC x-vector on every fine-tune, although it is
  exactly the input that fine-tune was trained on, and users would have to re-stamp identical files.
* **No stamp.** Rejected. `load_voice` requires one, and an x-vector from a different extractor (a
  192-d ECAPA vector, say) would then load and produce a wrong voice silently, or a shape error deep
  in the graph.
* **The embedding space: `xvector:<extractor>:<dim>`.** Chosen.

## Decision

A SpeechT5 export declares `loom.voice.compat = "xvector:speechbrain/spkrec-xvect-voxceleb:<dim>"`,
with `<dim>` read off the checkpoint's `speaker_embedding_dim`. `loom_exporter.speecht5_voices`
stamps every file it writes the same way. The file's one tensor is `speaker`, the driver input, and it
is not normalised, as the reference takes it. The engine is unchanged: it still compares two strings.

The released voices are the seven CMU ARCTIC speakers. Each is its speaker's `arctic_a0508`
utterance, the same sentence as the `slt` x-vector every published SpeechT5 example uses, and `slt` is
also the built-in voice. `--from <x.npy> --name --license` converts a caller's own x-vector.

## Consequences

* A voice file made here loads into any SpeechT5 export whose checkpoint uses that extractor at that
  width, fine-tunes included, and into no other architecture (`loom.voice.architecture`).
* The stamp says which extractor, and a SpeechT5 config does not record one. A fine-tune trained on a
  different 512-d extractor would still declare this stamp and accept the wrong vectors. The export
  cannot know better. Such a checkpoint should be exported with its own extractor named, which is a
  config field when one first appears.
* The card gate now loads every staged `voices/*.gguf` and synthesises one
  (`test_every_staged_voice_file_fits_and_speaks`). A re-export that changes any model's stamp shows up
  there, rather than on the Hub.
