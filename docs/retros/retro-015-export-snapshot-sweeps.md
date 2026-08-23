---
type: retro
date: 2026-08-12
domain: exporter
tags: [verification, byte-identity, snapshot-diff, regression]
---

# Retro-015: What the Byte-Identity Sweeps Caught

## The Issue

Every exporter change risks silently moving an artifact that was not meant to move. The sweep is the
control: snapshot every model's export before and after, `diff -r`, and require that exactly the
intended models differ.

## Root Cause Analysis

The sweeps below are kept because of what they demonstrate rather than what they found. In each, the
*expected* set and the *observed* set matched — and in the second, three models differed in ways that
had to be read and understood rather than accepted, because a driver-script change alters the recorded
sha of the driver as well as the driver.

## Resolution & Lesson Learned

* **Actionable takeaway 1 — record the baseline from a `git worktree` at the merge-base, with its own
  `cwd` and `PYTHONPATH`.** Exporting from the working tree and calling it the baseline measures the
  branch twice. (See the standing sweep recipe; a sweep takes ~22 min to record and ~24 to compare.)
* **Actionable takeaway 2 — "byte-identical" is only meaningful if the sweep can fail.** Confirm the
  comparison goes red before trusting a green one; see
  [ADR-015](../adrs/adr-015-ci-and-gate-test-classes.md).
* **Actionable takeaway 3 — name the models expected to move, before running it.** A sweep that reports
  a diff you then rationalise is not a gate.

---

## Full record (verbatim from the ledger)

### Sweep after the Supertonic text door (P4.6/P4.6a/P4.6b) — 2026-08-12


**All seventeen models, one expected to move, and it did.** Baseline recorded from a `git worktree` at
`origin/main` (c4e7221) with its own `cwd` and `PYTHONPATH`; current from the branch tip (f10c309).
22m20s to record, 24m01s to compare.

| | |
|---|---|
| **moved (expected)** | supertonic |
| **byte-identical** | conformer-ctc, gigaam-rnnt, kokoro, matcha, lfm2-monolithic, lfm2-modular, qwen3, smollm2, gemma-3-270m-it, whisper, granite-speech, parakeet-tdt, parakeet-rnnt, styletts2, vits, qwen3-asr |

The prediction was written before the run: supertonic must differ, the other sixteen must not. Four of
the negatives are the ones that carry weight — **kokoro, styletts2 and both parakeets** use
`ComputedCall`/`HelperCall`, the machinery nearest `SubgraphCallComponent.variants`, and **matcha** is
the only other `FlowMatchingSpec` user, so it is what proves `estimator_variants` left the unbucketed
path alone.

**Supertonic's diff is only what it should be:** 4 topologies → 16, `loom.txt_len` 10 → 512, 646 → 704
tensors, the three tokenizer KVs added, and two `loom.default_style.*` tensors. The KV-key diff
contains no other addition or removal.

**Only one test skipped in the whole run** — a model with no recorded baseline *skips* rather than
passes, so every one of those sixteen was a real diff against a real baseline rather than a silent
no-op. The sweep's own `test_a_changed_export_is_detected` passed too.

**Three things the sweep needed before it could answer this, all of them real gaps:**

1. **It swept 9 of the 17 models.** supertonic, vits, styletts2, both parakeets, smollm2 and
   gemma-3-270m-it were absent — including the one model this change was *about*, so the run had no
   positive control at all. The list's own comment says it "should fail review when a family is added
   and not swept"; seven had been.
2. **It never deleted the exported GGUFs**, and pytest keeps `tmp_path` for the session, so its disk
   cost was the SUM over models (~30 GB, Granite alone 8.75 GB) instead of the largest one. On a
   machine that has run at 19 GB free that is the difference between a sweep and an out-of-disk.
3. **qwen3-asr was `pytest.mark.skip`**, unconditionally — so it skipped in the `transformers>=5.13`
   environment too, and was therefore swept by nothing, ever. It is `skipif` on the interpreter's own
   `transformers` version now (read from package metadata, not by importing it, since this runs at
   collection time for every invocation).

**qwen3-asr was swept without pytest and without touching the ovos venv**, which has no pytest
installed: the *export* ran under ovos, the *snapshot* and *diff* under piper, since those are pure
`gguf`+`numpy`. Both sides produced a 3136.8 MB artifact with an identical driver-script hash, 4
topologies, 638 tensors and 24 KVs, and `diff -r` was empty — then the same comparison was shown to
notice a one-line tamper, because an empty diff between two directories is also what you get when you
compare nothing to nothing.

It ran **after** the sixteen-model pass rather than beside it, on purpose: Granite-Speech peaks at
22.9 GB RSS on a 33 GB machine, and a concurrent second export is how an OOM gets mistaken for a
failed comparison.



### Sweep after the window routing, the marshalling fix and the tokenizer

12 models from a `git worktree` at `abd6b0a` against the working tree, snapshotted and `diff -r`'d.
**Nine byte-identical** — conformer-ctc, parakeet-tdt, parakeet-rnnt, kokoro, matcha, supertonic, vits,
styletts2, and lfm2-**modular** (unfused, so it has no cache and takes no reducing call).

**Three differ, all fused causal LMs, and in one file each**: qwen3, smollm2 and lfm2-monolithic each
show 11 changed driver lines — `run_subgraph`+`argmax_row` collapsing into one `run_subgraph_argmax` in
both `infer` and `infer_with_past` — plus the `kv.txt` line that records the driver script's own sha.
**`model_graph_topology_main_topology.json` and `tensors.txt` are identical for all three**, which is
the claim worth having: reducing engine-side is a driver change and touches no graph and no weight.

Gemma-3-270m-it is new coverage rather than a diff (it could not export at the baseline, whose
tokenizer support predates the SPM family): 1742 nodes, 18 `ATTENTION`, inputs
`[tokens, cache_position, attention_mask, attention_mask_sw512]`, `tokenizer.ggml.pre =
granite-embed-multi-311m` auto-detected.

