---
type: adr
status: accepted
date: 2026-08-07
tags: [exporter, gguf-metadata, driver, api-surface]
---

# ADR-006: A Model's Constants Come Out of the Export, and Who Reads One Decides Where It Lives

## Context

Thirty numbers had to be passed into `infer` by the host — Matcha's `n_feats`/`mel_mean`/`mel_std`,
Supertonic's five, VITS's four, Kokoro's seven, StyleTTS2's eleven. They are properties of the model,
and the only host that ever supplied them was a test.

The complication is that a driver, by design, reads no GGUF metadata: the engine hands it topologies
and host math, not hparams ([ADR-002](adr-002-embedded-lua-drivers.md)).

## Options Considered

1. **Keep them as host arguments.** Every host re-derives them, and a wrong value is silent.
2. **Give the driver access to GGUF metadata.** Breaks the driver contract and makes every read a
   possible runtime `nil`.
3. **Split by reader.**

## Decision

**Which half a number belongs in is decided by who reads it.**

* A number the **driver** needs is an `ExportConstants` value, bound as an IR local — so every read
  goes through `driver_ir.validate` instead of being a runtime `nil`.
* A number the **host** needs — to size an input it must build before `infer` can be called at all —
  is a `loom.<key>` GGUF KV, declared by `LoomExportConfig.hparams()` and read back through
  `GgufModel::hparam_u32`/`hparam_f32`. Same namespace `loom::make_kv_cache` already reads its geometry
  from; no new engine code.

Exactly two are host-facing: `loom.style_dim` (a caller cannot build Kokoro's `ref_s` without knowing
how long each half is) and `loom.txt_len` (Supertonic). StyleTTS2 deliberately declares **nothing** —
it samples its style vector inside the driver, so a KV there would be one nobody reads.

## Consequences

* **Positive:** the host API shrinks to what a caller genuinely has to know, and a driver constant
  cannot be silently absent.
* **Positive:** it generalises — the same rule later decided what a file must declare for a host to
  dispatch on it, [ADR-013](adr-013-one-door-per-task.md).
* **Negative:** where a number *comes from* turned out to have four different answers (the restored
  module's own attributes, the config, a derivation, a literal), so adding a family means finding each
  one rather than reading a table.
* **Negative:** a declared KV nobody reads is now possible and is not detectable by a numeric gate —
  see [Retro-006](../retros/retro-006-kokoro-shipped-noise.md), where four declarations were wrong at
  once against a driver that matched PyTorch at cosine 0.996.

## Related

* Epic: [Epic-02: MIL Exporter and Compiler](../epics/epic-02-mil-exporter-and-compiler.md)
* Ledger record, verbatim:


P4.0.8's first follow-up. Thirty numbers that a host had to pass into `infer` — Matcha's `n_feats`/
`mel_mean`/`mel_std`, Supertonic's five, VITS's four, Kokoro's seven, StyleTTS2's eleven — are
properties of the model, and the only host that ever supplied them was a test. They now come out of
the export.

**The split, and it is the whole design.** A driver reads no GGUF metadata; `loom` gives it topologies
and host math, not hparams. So:

* a number the **driver** needs is an `ExportConstants` value (P4.0.17's answer, unchanged), bound as
  an IR local, so every read of it goes through `driver_ir.validate` instead of being a runtime `nil`;
* a number the **host** needs — to size an input it must build before `infer` can be called at all —
  is a `loom.<key>` GGUF KV, declared by the new `LoomExportConfig.hparams()` and read back with
  `GgufModel::hparam_u32`/`hparam_f32`. Same namespace `loom::make_kv_cache` already reads its five
  geometry facts from; no new engine code.

Which half a number belongs in is decided by who reads it. Exactly two are host-facing:
`loom.style_dim` (Kokoro — a caller cannot build `ref_s` without knowing how long each half is) and
`loom.txt_len` (Supertonic — every text-touching topology was traced at a fixed length, so any other
count is a model that cannot run). StyleTTS2 deliberately declares **nothing**: it samples its style
vector inside the driver, so a KV there would be one nobody reads.

**Where the numbers actually come from, which turned out to be four different answers:**

| source | examples |
|---|---|
| read off the restored **module** | Supertonic's `lat_dim`/`compression_factor`/`base_chunk_size` (the SpeechDecoder's own attributes); Kokoro's and StyleTTS2's `style_dim`/`d_model`/`hidden_per_dir` (`prosody_dims`, one derivation with two readers — `build_prosody_phases` traces against the same values) |
| read off a **config file** | Matcha's `n_feats` + the state dict's own `mel_mean`/`mel_std`; StyleTTS2's `sigma_data`, out of `config.yml`'s `model_params.diffusion.dist.sigma_data` |
| **baked into the trace**, so declared and cross-checked | Kokoro/StyleTTS2's `gen_istft_n_fft`/`gen_istft_hop`/`upsample_scale`; Supertonic's `T_TEXT`; VITS's `inter_channels` |
| genuinely **not in the checkpoint** | Supertonic's 44100 Hz (a `supertonic_tts.lightning` default, not shipped); piper's three synthesis scales; StyleTTS2's Karras `sigma_min`/`sigma_max`/`rho` |

**Two things this item did not predict.**

* **One family's numbers are knobs, not facts.** VITS's `noise_scale`/`noise_scale_w`/`length_scale`
  are piper's synthesis defaults and `length_scale` is *speaking rate*. Binding them hard would have
  made the driver strictly less capable than before, so they are bound as **defaults the caller may
  override** (`inputs.length_scale or LENGTH_SCALE`) — the model declares them, the host still
  decides. StyleTTS2's Karras parameters got the opposite treatment on the same test: its repo's own
  inference entry point exposes `diffusion_steps` and not those, so making them overridable would be
  inventing an interface rather than preserving one.
* **`sigma_data` forced a new file dependency, and correctly.** It is in neither checkpoint nor
  Kokoro's `config.json`; it is in StyleTTS2's own `config.yml`, which also says
  `estimate_sigma_data: True` — a statistic of the training data, not a constant of the architecture.
  So the export reads that file and raises naming it rather than substituting the LJSpeech value for a
  checkpoint that does not state it.

**A check fell out that had nothing to do with the follow-up.** `_STFT_N_FFT`/`_STFT_HOP`/
`_UPSAMPLE_SCALE` are baked into the traced `decoder_vocoder` graph *and* into the driver's host-side
`compute_wsum`, and nothing verified the checkpoint agreed. `check_istftnet_geometry()` now compares
them against `config.json`'s own istftnet section for both families and raises naming both sides.

**Gates.** Per family: the constants the export emits are compared value-for-value against the
literals `tts_driver_inputs.h` supplied, and the model is re-exported and run through its MIL Lua
driver test with nothing passed for them. Matcha reproduced its frozen waveform exactly
(`max_abs_diff` 0.0104421, rmse 0.000678027 — the numbers in its own header); Supertonic 2.0843e-06;
VITS's two calls (defaults vs. explicit) are **bit-identical**; Kokoro 22208/22208; StyleTTS2
22207/22207. Negative gates, each breaking one thing and watching a real export or test fail:
Matcha's `mel_mean` +1.0 → 0.860849 against a 0.02 bound; Supertonic's `base_chunk_size` halved →
0.197135 against 1e-2; VITS's `LENGTH_SCALE` 1.5 → 0.264348; Kokoro's `_STFT_HOP` 4 → the export
refuses before tracing, naming both sides; StyleTTS2's `config.yml` missing → the export refuses,
naming the file and the key.

**Honest about one limit:** Kokoro's and StyleTTS2's MIL tests have deliberately loose bounds (no
frozen oracle — see their headers), so a perturbed constant would not reliably trip them. For those
two the exact claim is the constant comparison, not a numeric probe, and the commits say so.

Full `ctest` **128/128** with all five MIL GGUFs wired in; exporter suite **480/480**. The four
non-TTS families carry `hparams` through `backend_kwargs()` too and declare nothing, so their exports
are byte-identical by construction — `test_export_hparams.py` walks the registry to check the channel
is actually there, because an override that quietly dropped it would disable the hook for one family
with no other symptom.

