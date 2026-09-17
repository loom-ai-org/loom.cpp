---
type: adr
status: accepted
date: 2026-09-17
tags: [exporter, model-coverage, family-10, family-11, qwen3-tts]
supersedes: []
---

# ADR-038: The Codec's Encoder Ships Inside the Talker, Because It Has No Task of Its Own

## Context

Qwen3-TTS clones a voice in one of two nested modes. `x_vector_only_mode` needs a speaker embedding
and shipped with the model in September 2026. **ICL** additionally replays the reference clip itself:
its transcript on the text stream and its own codec frames on the codec stream, summed row for row. So
ICL needs the reference clip's codes, and producing them means running the speech tokenizer's
**encoder** — a `MimiModel` with the decode side nulled, 39.4 M parameters, 158 MB at F32.

[ADR-022](adr-022-dia-and-its-codec-stay-two-files.md) put this checkpoint's codec DECODER in its own
GGUF: one codec serves every size and variant of the talker, and the intermediate codes are worth
having on their own. The obvious symmetry would be to do the same for the encoder.

Three places were considered:

1. **Add encode to `qwen3-tts-tokenizer-12hz`**, the published decode-half GGUF.
2. **A third GGUF**, an encoder of its own.
3. **Inside the talker**, as more phases beside the speaker encoder.

## Decision

**Option 3.** The encoder is four phases of the talker's own export (`ref_encode`,
`rvq_project_semantic`, `rvq_project_acoustic`, `rvq_step`), driven by the talker's driver when a
caller asks for ICL with audio rather than with codes.

## Consequences

**What decided it is that the encode direction has no task.** `loom.task = "audio-codec"` is
`audio_codes -> audio`; the task registry says so in as many words — *"the encode direction is a
different pair and has no family"*. A GGUF is reached through a door named by its task
([ADR-013](adr-013-one-door-per-task.md)), so options 1 and 2 both require a new task, a new door in
the high-level API, and therefore engine C++ and a release — for a model whose only caller in this
tree is the prompt three phases away. Option 3 needs none of that: the talker already takes a
reference waveform for the x-vector, and this is the same waveform read a second way.

**It is the speaker encoder's own precedent.** `_SpeakerEncoderWrapper` is an 8.9 M ECAPA stack that
lives here for exactly this reason — its two halves have no other caller. The Mimi encoder is the same
argument at 39 M.

**ADR-022 is not weakened, because it was never about direction.** Its argument is that one DECODER
serves every talker size and variant and that codes are a useful intermediate. Neither is true of the
encoder: its output feeds this model's prompt and nothing else's, and a caller who wants codes on
their own has no door to ask through.

**The cost is 190 MB on a 3.94 GB file (4.8%), paid by everyone who downloads the talker** — including
callers who only ever use `x_vector_only_mode`. Measured from the written file rather than estimated:
155.5 MB of encoder, 33.6 MB for the sixteen codebooks concatenated into one table, and 1.0 MB for the
two quantizer input projections. (The ICL prompt's own 125.8 MB embedding table is NOT part of that —
it is the merged predictor table `frame_embed` already carried, and the writer stores it once.) That is
the real price of this decision and it is accepted rather than overlooked: the alternative charges a
new task and a release instead, and a file 5% larger is cheaper than a door nobody else walks through.

**If a second consumer ever appears** — another family wanting codes from audio, or a caller asking
for them directly — this becomes option 1 or 2 and gets its task then. The weights are the same
weights; what changes is which file carries them and which door reaches them. Declaring the task now,
on one caller, would be inventing an interface from a single example, which is what
[ADR-005](adr-005-export-config-and-task-registry.md) exists to avoid.

## Related

* [ADR-022](adr-022-dia-and-its-codec-stay-two-files.md) — why the decoder is its own file.
* [ADR-034](adr-034-a-chunked-decode-is-the-drivers-loop-not-a-longer-call.md) — the decode half's own
  driver loop.
* [Retro-050](../retros/retro-050-the-config-declared-a-window-the-reference-never-applied.md) — what
  the encoder cost to get right.
