---
type: adr
status: accepted
date: 2026-09-29
tags: [hosts, loom-cli, codecs, family-10, family-11, moss-tts, file-format]
supersedes: []
---

# ADR-055: A Codes File States Its Width

## Context

`loom_cli`'s `text2codes` path runs an AR codec LM and its codec in one process. Because the codes are
the return value rather than an implementation detail ([ADR-022](adr-022-dia-and-its-codec-stay-two-files.md)),
it can also write them to disk with `--codes-out`, and a codec decodes them later with
`--prompt @file`.

The first version wrote text, one frame per line. The codec branch read it through `read_number_spec`,
which folds newlines into spaces, so the rows were lost on the way in. For three of the four pairs that
costs nothing, because the LM and the codec are the same width. For MOSS-TTS it broke the round trip.
The LM emits 12 codebooks and MOSS-Audio-Tokenizer decodes 32, padded with `codec.absent_code`
([ADR-050](adr-050-a-codec-declares-its-absent-id-and-its-channels.md)). 36 frames of 12 are 432
numbers, and at the codec's width 432 is not a whole number of frames, so the decode was refused.

The deeper problem is that a flat list of numbers cannot be checked at all. 36×12 and 27×16 are both
432 numbers. A width that happened to divide would decode the wrong frame count, and nothing would
say so.

## Options

* **Keep text and have the codec branch read it line by line.** Rejected. The width would be
  implied by line breaks, so a one-frame file or a file whose newlines were lost would still be
  ambiguous. The file would also not say that it holds codes, or which model produced them.
* **A GGUF that states its shape.** Chosen. It is the same move as a voice file
  ([ADR-045](adr-045-a-voice-is-a-file-of-driver-inputs-stamped-with-its-weights.md)), which is a small
  GGUF of driver inputs, and every host already links a GGUF reader.

## Decision

A **codes file** is a GGUF with `general.architecture = "loom-codes"`. Its one tensor, `codes`, is I32,
row-major `[n_frames, n_codebooks]`. Its metadata is `loom.codes.n_codebooks`, `loom.codes.n_frames` and,
when known, `loom.codes.source` (the architecture of the LM that produced the codes).

* **The reader refuses what it cannot decode, by name.** It refuses a GGUF that is not a codes file, a
  tensor that disagrees with its own declared width, and rows the codec cannot take. That last check is
  the same rule the in-process pair applies: rows no wider than the codec, and narrower ones only when
  the codec declares `codec.absent_code`. The rule is written once, as `pairing_error` / `widen_rows` in
  `tools/loom_cli/main.cpp`.
* **The code lives with the CLI for now** (`tools/loom_cli/codes_file.{h,cpp}`, beside `wav_file`).
  loom-py hands codes over as Python rows and has no file for them. If it ever needs one, the format
  moves into the engine, as `voice_file` did, rather than being written a second time.
* **A plain `--prompt "<codes>"` list still works** at exactly the codec's width. It is what a caller
  types by hand.

## Consequences

* **Verified**: a MOSS-TTS clone written with `--codes-out` decodes through the codec branch
  bit-identically to the in-process pair. A file written independently with gguf-py decodes
  bit-identically to the pair for Dia → DAC. A 10-wide file, an 8-wide file for a codec with no
  absent id, and a voice file are each refused with their reason.
* **No compatibility check on what the codes mean.** Nothing on the codec side declares a fingerprint
  to compare against (a MOSS voice's `loom.voice.compat` is checked by the LM, not the codec). A width
  that matches but belongs to a different codec will decode to noise.

## Related

* [ADR-022](adr-022-dia-and-its-codec-stay-two-files.md): the LM and its codec stay two files
* [ADR-050](adr-050-a-codec-declares-its-absent-id-and-its-channels.md): the absent id that pads a narrow row
* [ADR-045](adr-045-a-voice-is-a-file-of-driver-inputs-stamped-with-its-weights.md): the voice file this mirrors
* [Epic-06](../epics/epic-06-high-level-api-and-hosts.md): the hosts
