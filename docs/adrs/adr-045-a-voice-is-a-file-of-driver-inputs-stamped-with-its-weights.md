---
type: adr
status: accepted
date: 2026-09-24
tags: [engine, hosts, loom-py, loom-cli, voices, tts, family-9, pocket-tts, hub]
supersedes: []
---

# ADR-045: A Voice Is a File of Driver Inputs, Stamped With the Weights It Fits

## Context

Pocket-TTS ships 26 predefined voices. One, `alba`, is built into the GGUF as a driver weight
([ADR-043](adr-043-a-voice-that-is-attention-state-is-seeded-not-run.md)), and its driver already takes
any other as the input `voice_kv`, per layer K then V. But there was no way to GET one there: the
reference's voice files are safetensors with model-specific key names
(`transformer.layers.<i>.self_attn/cache`) and a layout the driver does not read directly, and
`Text2Speech` had no `voice=`.

Two further facts shaped it:

* **A voice fits only the weights that made it.** It is the flow LM's own state after hearing the
  speaker. The reference says a predefined voice fed to other weights "typically never emits EOS",
  and the local snapshot already held two English releases (2026-04 and 2026-09) whose voices share
  file names.
* **The precedent was per-model and host-side.** Kokoro's card tells the reader to `torch.load` an
  upstream `.pt` pack and slice a row. That is fine for a card, but it would put this model's key names
  and layout into every host.

## Options

* **Ship all 26 inside the GGUF.** Rejected. That is ~160 MB on a 405 MB file, paid by everyone who
  wants one voice, and it still leaves no door for a user's own `export-voice` state.
* **Teach loom-py (and `loom_cli`) to read the reference's safetensors.** Rejected. That is
  per-architecture code in the hosts, which all three repos forbid.
* **Voice files that are driver inputs by name, converted by the exporter and read by the engine.**
  Chosen.

## Decision

A **voice file** is a small GGUF (`general.architecture = "loom-voice"`). Its tensors are driver
inputs, named by the input they become: Pocket-TTS's has one, `voice_kv`. Its metadata is
`loom.voice.architecture`, `loom.voice.compat`, `loom.voice.name`, `loom.voice.license` and
`loom.voice.origin`. The MODEL declares `loom.voice.compat` in its contract. For Pocket-TTS that is a
sha256 over its `flow_lm.*` tensors: Mimi is excluded because the release without voice cloning zeroes
the encoder, and a voice from either release is the same voice. The model also names its built-in
voice in `loom.tts.voices`.

* **The engine owns the format and the check**: `loom::load_voice(model, path)` returns the inputs, and
  refuses by name a file for another architecture or other weights (`SchemaError`, both fingerprints
  in the message), a non-F32 tensor, or a model that declares no fingerprint.
* **The exporter owns the conversion**: `loom_exporter.pocket_tts_voices` converts every predefined
  voice, or a user's own `export-voice` file (`--from`, with a required `--name` and `--license`).
* **The hosts own resolution**: `loom_cli --voice <file>`; in loom-py, `text2speech.infer(voice=...)`
  takes a name, a path or a `loom.Voice`. A name is `voices/<name>.gguf` beside the model file, then,
  for a model from `from_pretrained`, the same path in its repo (downloaded into the same snapshot).
  `model.voices` lists the built-in voice first, then the files. An explicit driver input still wins
  over the voice's.
* **A Hub repo's model is its ROOT `.gguf`**: `download()` no longer counts `voices/*.gguf` when it
  picks the one model a repo holds.

## Consequences

* **Licences travel per voice**, because they are the recordings' and not one licence. Per
  `kyutai/tts-voices`' README: VCTK and Alba MacKenna are CC-BY-4.0; voice donations, Voice-Zero,
  Unmute's own recordings and Common Voice are CC0; **`cosette` (Expresso) and `jean` (EARS) are
  CC-BY-NC-4.0**; `juergen` and `rafael` state none. The card lists all 26.
* **Verified**: `marius.gguf` through `load_voice` and the driver, teacher-forced against the reference
  run with that voice, at max |d| 8.0e-06 and rmse 1.9e-07 (gate arm). A 2026-04 voice on the 2026-09
  model is refused by name. `tests/ci/test_voice_file.cpp` runs a loaded voice through `loom.seed_kv`
  and checks the four refusals. loom-py's `tests/ci/test_voices.py` pins name resolution (local and
  Hub), the input merge and the refusals.
* **Nothing in the format is Pocket-TTS's.** Chatterbox's voice (five driver inputs) or a Kokoro style
  row fits it unchanged, once an export declares a fingerprint.

## Related

* [ADR-043](adr-043-a-voice-that-is-attention-state-is-seeded-not-run.md): what a Pocket-TTS voice is
* [ADR-013](adr-013-one-door-per-task.md): one door per task
* [Epic-06](../epics/epic-06-high-level-api-and-hosts.md): the hosts
* `loom-exporter/loom_exporter/pocket_tts_voices.py`, `include/loom/core/voice_file.h`
