---
type: adr
status: accepted
date: 2026-08-07
tags: [mil, coremltools, tracing, exporter, ir]
---

# ADR-004: MIL Is the Single Export Path

## Context

Every supported model originally had a hand-written `tools/convert_*` script — roughly 14,000 lines
across ten directories, each one an independent re-derivation of how to walk a checkpoint and emit a
topology. Two paths to the same artifact meant two places for a fix to be needed and one of them to be
forgotten.

The question was what intermediate representation an offline compiler should use to ingest PyTorch.

## Options Considered

1. **Hand-written converters, forever.** Full control per model; no generality, and the cost is linear
   in models.
2. **`aten`→loom directly off TorchScript.** Tried as a POC; needed custom torch ops for things like
   relative-position attention.
3. **StableHLO.** Filed as a validation exercise rather than a fix; deliberately not started.
4. **CoreML's MIL, via `coremltools`.** A mature PyTorch ingestion frontend with device-agnostic
   optimization passes, separating a Python compile-time frontend from a fast C++ runtime — the same
   split loom wants.

## Decision

**MIL is the IR.** `coremltools` traces the *real* upstream module; loom's compiler lowers MIL to
topology JSON plus a generated Lua driver. The bespoke converters are retired.

Two structural choices came with it:

* **Per-op lowering is a declarative rule table** (`topology_ops.py`), keyed on
  `(mil_op_type, guard_predicate)`, not a 4,400-line dispatch function. An op no rule claims falls
  through to a generic `OP_MAP` path, and that fall-through is a deliberate route rather than an
  accident.
* **Decomposition is a strategy object**, not a mode string: `Flattened`, `Modular(spec)`,
  `MultiPhase`. The three forms need genuinely different data, and one config carrying every field with
  a string selecting which subset is live makes invalid states representable. See
  [Retro-016](../retros/retro-016-the-profile-field-was-not-inert.md).

**Every exporter change is gated on a byte-identity sweep** (`snapshot_gguf.py`, `diff -r`), or on
`compare_snapshots.py` when the change is *meant* to rewrite shape attributes.

## Consequences

* **Positive:** one path, one place to fix anything. Tracing the real upstream module means the oracle
  is the upstream model rather than a reimplementation of it.
* **Positive:** symbolic shapes are sympy objects rendered once at emission through a printer
  restricted to the engine's grammar, which raises on anything it cannot express rather than shipping
  unparseable text.
* **Negative:** the project inherits `coremltools`' tracing limitations and its dependency weight in
  the export environment.
* **Negative:** the modular blueprint's *generality* claim still rests on a single model. Tracked in
  [the backlog](../backlog/active-index.md#exporter--mil-compiler).

## Related

* Epic: [Epic-02: MIL Exporter and Compiler](../epics/epic-02-mil-exporter-and-compiler.md)
* Retro: [Retro-013: Retrofitting Eight Bespoke Converters](../retros/retro-013-retrofitting-eight-bespoke-converters.md)
* Ledger record, verbatim:

### The bespoke NeMo converters are gone — DONE (2026-08-07)


P4.0.17 step 3, and the end of `tools/convert_nemo/` as a converter directory: only `mel_common.py`,
`nemo_common.py` (both still imported by `convert_generic`/`convert_whisper`) and
`reference_forward_conformer.py` (a real numerical oracle for the MIL encoder) remain.

**A prerequisite surfaced that the plan had not named: the MIL artifact carried no tokenizer.** The
bespoke converters wrote the checkpoint's SentencePiece vocab into their GGUF; the MIL export did not,
so its artifact could not be detokenized. That — not the decode loop — was the last thing keeping the
old converters alive. `extract_nemo_tokenizer_dir` unpacks the archive's `<hash>_tokenizer.model` into
a temp dir and hands it to the exporter's existing `sentencepiece_proto` family, so there is no new
writer and no second vocab schema. Conformer now embeds 1024 tokens, Parakeet 8192, and
`loom_cli --model parakeet-tdt.gguf --wav samples/jfk.wav` prints *"And so, my fellow Americans, ask
not what your country can do for you, ask what you can do for your country."* from one file.

Two things had to move for that:

  * **`MultiPhase.export` never forwarded `backend_kwargs()` to the output exporter**, only to the
    per-phase ones — so a multi-phase family had no way to say anything about its own GGUF. It does now,
    which is what lets Parakeet carry a vocab at all.
  * **`tokenizer_common.py` moved to `loom_mil_compiler/spm_tokenizer_export.py`**, beside the other
    vocab writers. `exporter.py` had been importing it as `convert_nemo.tokenizer_common`, which only
    resolved when `tools/` happened to be on `sys.path` as a package root — it failed the moment the
    export actually tried to use it.

**`loom_cli --wav` is now model-agnostic.** It read the *bare* `model.graph_topology`, computed the
relative-position table host-side and called `loom::ctc_greedy_decode` — all three properties of the
bespoke artifact. It now registers whatever topologies the file declares, calls the driver the file
ships, and detokenizes with the vocab the file embeds. One path for Conformer-CTC, Parakeet-TDT and
Parakeet-RNN-T; `compute_pos_emb` went with it.

**Retired:** `convert_conformer_ctc.py`, `test_e2e_conformer_ctc` and
`test_e2e_conformer_ctc_dynamic_length`. The dynamic-length property they proved is not lost — it moved
to `test_e2e_conformer_ctc_lua_driver`, which already runs 10240, 32000 and 176000-sample inputs through
one artifact, which is the same claim on a wider spread. `test_vocab` is re-based onto the MIL GGUF and
still asserts the same 1024-token vocab, unk id and round trips.

`ctc_decode.{h,cpp}` STAYS, and deliberately: it is not a converter and not a per-model driver, it has
its own unit test, and it is the independent oracle `test_e2e_conformer_ctc_lua_driver` compares the
driver's Lua decode against. Deleting it would remove a check, not dead code.

