---
type: adr
status: accepted
date: 2026-10-10
tags: [exporter, licensing, gpl, sanotts, tts, model-zoo]
supersedes: []
---

# ADR-070: A GPL Upstream Is a Reference, Not a Dependency

Serves [Epic-03](../epics/epic-03-model-coverage.md). It extends
[ADR-012](adr-012-permissive-phonemizer.md), which kept GPL source out of the engine, to the exporter
and the Hub.

## Context

sanoTTS (`ampixa/sanoTTS`, 2026-10-10) is GPL-3.0, both its training code and its published weights.
It ships **no torch checkpoint**. Each voice is a raw fp16 blob plus a manifest (the piperlite line),
or a C runtime's blobs plus a generated offset header (the nano line). The tensor names are the
`state_dict` of torch modules defined in upstream's GPL training scripts. For the nano line's decoder,
`TinyVocosStudent`, no torch module was ever published: it is in no commit of the repository. What
upstream does publish is a numpy runtime in its pip package, which it gates against its own PyTorch
reference.

The zoo rule is that a model enters through a PyTorch checkpoint
(the user's rule since Kitten TTS was dropped, 2026-10-02). The repositories here are MIT.

## Options

1. **Vendor upstream's modules into loom-exporter.** This puts GPL source in an MIT repository, the
   conflict ADR-012 rejected for espeak-ng.
2. **Re-implement every module clean-room.** This is more work, and it is arguably still derived from
   reading the GPL code. It also loses the strongest oracle available, upstream's own module.
3. **Import upstream's modules from a pinned clone at export time, never vendored.** Write an own module
   only where upstream published none, to the definition upstream does publish.
4. **Drop sanoTTS.**

## Decision

Option 3 (the user, 2026-10-10):

* The exporter imports upstream's training scripts from a git clone on `sys.path`
  (`LOOM_SANOTTS_REPO`, default `/home/flavio/Dev/sanoTTS`). It is the F5-TTS pattern, and how
  upstream's own golden exporters import them. The clone is **pinned**: the leaf refuses a checkout
  whose HEAD is not the recorded commit, because the modules are the architecture.
* The nano decoder is the export's own `_TinyVocos`, written to upstream's numpy runtime. This is the
  Silero precedent: published weights plus a plain-Python definition (the user's call). Its oracle is
  that runtime, imported from the same clone.
* Published GGUFs carry upstream's licence on their card, `license: gpl-3.0`, the zoo's first. The
  engine, the exporter and loom-py contain no GPL code.

## Consequences

* A sanoTTS export needs the clone. CI cannot have it, so CI tests what does not need it: the package
  readers, the framing, the gather lowering. The real-package comparisons are the gate half.
* Moving the pin is a deliberate act: re-verify against the new commit, then change `SANOTTS_COMMIT`.
* A second GPL model would follow the same pattern. It does not change what a loom user installs.
