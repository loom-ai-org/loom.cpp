---
type: adr
status: accepted
date: 2026-09-25
tags: [exporter, hosts, loom-py, voices, family-10, family-11, moss]
supersedes: []
---

# ADR-053: A Codec LM's Voice Is Its References' Codes, Stamped With the Codec

## Context

MOSS-TTS-Local clones a voice from one or more reference clips. What the model reads of a clip is its
**codes**: the processor runs MOSS-Audio-Tokenizer-v2's encoder on it and puts the frames in the
prompt, one row per frame, each reference as `<audio_start>` rows `<audio_end>` where the template
otherwise says `None`. Several references are one per speaker of a dialogue (`[S1] ... [S2] ...`).

Three facts shaped where the encoder runs and what a voice file holds:

* **The encoder is large.** About 1.06B parameters, 4.2 GB at F32. Putting it in the TTS GGUF, as
  [ADR-038](adr-038-the-codecs-encoder-ships-inside-the-talker.md) did for Qwen3-TTS, would add 25% to
  a 16.8 GB file that every caller downloads, where Qwen3-TTS paid 5%.
* **A code means what the codec's quantizer says it means.** The TTS weights read codes; they do not
  produce them. Any MOSS-TTS-Local checkpoint trained on this codec reads the same codes the same way.
* **A host passes a voice's inputs through without knowing what they are** ([ADR-045](adr-045-a-voice-is-a-file-of-driver-inputs-stamped-with-its-weights.md)).
  So a cast of several references cannot be several files merged by the host.

## Decision

The user chose voice files first (2026-09-25). Where that leaves the details:

* **The encoder runs once per voice, in Python**: `loom_exporter.moss_tts_voices` builds the
  reference's own processor with an F32 codec and calls its `encode_audios_from_path`, which covers
  channel handling, the 48 kHz resample, loudness normalisation and `batch_encode`. No encoder ships
  in any GGUF.
* **One file holds every reference.** It has two driver inputs: `reference_codes` (every reference's
  frames, frame-major, 12 per frame, in order) and `reference_frames` (one frame count per
  reference). The driver emits the reference's direct clone layout from them: no separator between
  references, the USER slot id (not the assistant's) in column 0.
* **`loom.voice.compat` is the codec's fingerprint**: a sha256 over its `quantizer.*` tensors. The TTS
  export finds the codec as the directory beside the checkpoint that its config's
  `audio_tokenizer_name_or_path` names, or takes `codec_dir`. It refuses to export without one,
  because a model with no fingerprint takes no voice files.
* **loom-py's `text2codes.infer` takes `voice=`**, resolved exactly as `text2speech`'s is.

## Consequences

* **Cloning costs no engine change and no size.** The driver is the only thing that moved, and a
  voice file is a few KB: 137 frames of JFK are 1,644 codes.
* **There is no clip-in door inside loom.** A caller who has a WAV and no Python cannot clone. The
  encoder can join later as phases of the TTS export, or as the codec GGUF's encode direction if a
  second consumer appears (ADR-038's own condition). Either would produce the same two inputs, so the
  driver would not change.
* **ADR-045's rule is kept, and its wording is widened.** "Stamped with the weights it fits" means
  the weights that give the voice its meaning. For a voice that is model state (Pocket-TTS) or
  front-end features (CosyVoice3), those are the model's weights. For a voice that is codes, they are
  the codec's.
* **A reference code outside `[0, codebook_size)` is refused by the driver.** The export folds the
  reference's mask into a zero row at the pad code (1024), so a stray 1024 would silently read as
  "absent" rather than fail.
* The reference's `mode="continuation"` (a prompt clip as the ASSISTANT's audio, continued) is a
  different prompt and is not covered.

## Related

* [ADR-045](adr-045-a-voice-is-a-file-of-driver-inputs-stamped-with-its-weights.md): the voice-file format
* [ADR-038](adr-038-the-codecs-encoder-ships-inside-the-talker.md): the other place a codec's encoder went
* [ADR-050](adr-050-a-codec-declares-its-absent-id-and-its-channels.md): the pad/absent id this refuses
* [Epic-03](../epics/epic-03-model-coverage.md) family 10
