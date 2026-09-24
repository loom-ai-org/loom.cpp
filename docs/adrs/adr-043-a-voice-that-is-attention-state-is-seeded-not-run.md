---
type: adr
status: accepted
date: 2026-09-24
tags: [engine, lua-bridge, kv-cache, tts, family-9, pocket-tts, voices]
supersedes: []
---

# ADR-043: A Voice That Is Attention State Is Seeded, Not Run

## Context

Pocket-TTS (family 9's fifth leaf) conditions its flow LM on a voice by PREFILLING it:
`[bos_before_voice | speaker_proj(mimi_encoder(audio))]` runs through the 6-layer transformer, and
generation continues from the resulting KV cache. The reference ships its 26 predefined voices as that
cache, not as audio or as the prefill's inputs: `embeddings/<name>.safetensors` holds
`transformer.layers.<i>.self_attn/cache` (`(2, 1, n, 16, 64)`, K then V, RoPE already applied to K)
and an `offset`. `pocket-tts export-voice` writes the same format for a user's own clip. The released
weights without voice cloning zero the Mimi encoder, so for most users the cache is the ONLY form a
voice exists in.

A loom driver could not use it. The only writer of a module's `KvCache` was an ATTENTION node, which
writes what it just projected.

## Options

* **Recover the prefill's inputs from the cache.** Impossible. K and V are linear in
  `LayerNorm(x)`, and the norm has discarded each row's mean and scale, which the residual stream
  still needs.
* **Re-derive the voice from its source audio at export time**, through the Mimi encoder, and ship
  the 126×1024 conditioning rows as an ordinary prefill input. Rejected. It needs the gated
  voice-cloning weights and the source clips, which are downloads with their own licences. It
  reproduces the shipped state only if the reference's audio pipeline is replayed exactly. It also
  cannot load a voice a user exported with `export-voice`.
* **Per-layer K/V graph inputs in the `lm` topology**, concatenated ahead of the cache inside the
  graph. Rejected. That is a per-model attention variant, and `fuse_loom_attention` would have to
  recognise a second shape. It would also cost the cached steps an extra input per layer, forever.
* **A bridge binding that writes rows into the cache.** Chosen.

## Decision

`loom.seed_kv(module, source [, n_rows]) -> n_rows` writes positions `[0, n_rows)` of `module`'s KV
cache. `source` is a weight name, read tensor to tensor with no Lua copy (a voice is ~1.5 M floats),
or a flat Lua array. The layout is the saved state's own: per layer, K `[n_rows, n_embd_k]` then V,
each row the heads flattened head-major, exactly as an ATTENTION node writes a row. `n_rows` is
derived from the size, and an explicit one must agree with it. Under it is
`KvCache::set_rows(layer, value, data, first_row, n_rows)`.

The driver seeds, then prefills the text at `n_past = n_rows`. Positions are the driver's already
(`position_ids`, `causal_mask(n, n_past)`), so a seeded prefix needs nothing else. A private-cache
stream ([ADR-023](adr-023-a-second-stream-is-declared-not-derived.md)) is seeded by its own module name.

It meets the binding criterion: it reads no model config (the cache geometry is the one the host
allocated), and the cache is engine-owned state that no graph input can reach.

## Consequences

* **The export ships the cache.** `pocket_tts_export.read_voice` flattens the file into the seed
  layout and refuses a padded state (cache length ≠ offset) or NaN. The built-in voice (`alba`, the
  reference's English default, 126 rows, 6.2 MB) travels as the driver weight `voice.kv`. A caller may
  pass any other saved state as `inputs.voice_kv`.
* **Refused rather than truncated:** a size that is not a whole number of positions, more positions
  than the cache holds, a mismatched `n_rows`, a non-F32 weight (a quantized state is not the state
  that was saved), and a module with no ATTENTION node.
* **Tested where a wrong layout cannot hide:** `tests/ci/test_seed_kv.cpp` seeds a bare two-layer
  ATTENTION module whose q/k/v are inputs. Each head of each layer attends to exactly one seeded row,
  and every V is distinct. A sabotage run with K and V swapped fails 3 of 9 checks.
* **Voice cloning from audio is not this decision's problem.** It is the Mimi encoder as one more
  phase feeding an ordinary prefill, and the prefill already exists.

## Related

* [Epic-03 §2](../epics/epic-03-model-coverage.md): family 9's fifth leaf
* [ADR-023](adr-023-a-second-stream-is-declared-not-derived.md): private KV caches
* [ADR-003](adr-003-per-model-complexity-in-the-exporter.md): what earns C++ here
* `loom-exporter/loom_exporter/pocket_tts_export.py`: the voice reader and the driver
