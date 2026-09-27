---
type: adr
status: accepted
date: 2026-09-17
tags: [exporter, memory, multi-phase, p5.0, model-coverage]
supersedes: []
---

# ADR-039: A Phase Boundary Can Be a Process Boundary, and It Has to Be a Respawn

## Context

Peak RSS during *conversion* — not the family template, not the driver — is what decides which
checkpoints this exporter can turn into a GGUF at all. `MultiPhase.export` made that peak a **sum**
over phases where it should be a **max**, and P5.0 is the three changes that take it apart.

Change 1 shipped with family 3's second leaf: the traced module, the wrapper and the converted MIL
program are the *same arrays*, so they have to be released together — dropping the torch half alone
moved the peak by 0.2 GB, measured. That took Granite-Speech from **30.4 GB to 22.9 GB**, OOM to clean
export. What it did not touch is the two things a phase leaves behind:

* **its weights**, carried as F32 numpy arrays from the moment that phase converts until the last one
  has, then dtype-cast and quantized in one pass inside `write_gguf`. For Granite-Speech that is
  8.75 GB of artifact held at full width for the whole export;
* **everything the conversion touched that Python cannot see it holding** — coremltools' and torch's
  own caches, and the arenas glibc keeps rather than returning to the OS. `gc.collect()` cannot reach
  any of it.

Voxtral-Mini-3B is the checkpoint that names the stakes: 4.68B parameters, ~14.4 GB of F32 weights in
its LM phase alone, against 28 GB of RAM and no swap. It is deferred for the machine rather than the
template ([Epic-03 §2](../epics/epic-03-model-coverage.md)), and both remaining changes are what it
would need first.

## Decision

**Pack each phase's weights at the phase, and — when asked — convert each phase in a child process,
spilling the packed bytes to disk for the parent to memory-map.**

### Packing moves to the phase

`LoomGGUFExporter.pack_weights()` is `write_gguf`'s weight preparation, extracted: prune-free, it
normalises dtypes, folds conv kernels so their quantization blocks align (P4.13) and quantizes what is
eligible, recording per name what it became in a `WeightPacking`. `convert_phase` calls it as soon as a
phase's topology exists; `write_gguf` calls it again and finds only the driver's own `loom.get_weight`
tensors unpacked. What a converted phase hands forward is therefore the tensor's **on-disk payload** —
a quarter of its F32 size at Q8_0 — rather than its F32 array.

**What makes a per-phase answer equal to the merged one.** Both quantization gates are questions about
*every* topology: `name in _collect_mul_mat_weight_names()`, and the fold's requirement that a name's
entire use is one convolution's first input. A phase exporter sees only its own topology. The answers
coincide because a multi-phase export leaves `flat_namespace` False, so every weight is written
`{func_name}.{weight}` and no other phase's graph can name this phase's tensors. That is a property of
the naming convention, not something any family declares, and a violation would produce a *plausible*
artifact rather than an error — a kernel folded for a consumer that wanted its declared shape. So
`_check_phase_weight_namespaces` checks it against the merged topologies rather than assuming it.

### Isolation is a respawn

`--isolate-phases` (or `$LOOM_PHASE_ISOLATION`) runs each phase as
`python -m loom_exporter.phase_isolation <config.pkl> <index> <spill dir>`. The child unpickles the
config, calls `phases()` itself, drops every phase but its own, converts and packs, and writes one
`manifest.json` + one concatenated `weights.bin`. The parent maps each tensor back with `np.memmap`,
so it accumulates N phases of weights without faulting them in: `GGUFWriter.add_tensor` holds the
array and `write_tensors_to_file` streams it with `tofile`, straight from the spill into the artifact.

The spill goes **beside the output**, not under `$TMPDIR`. /tmp is a tmpfs on a normal Linux box, and
spilling to a tmpfs puts the weights back in RAM under another name — which would not even fail
cleanly: the export would run, the peak would be the sum again, and the only symptom would be an OOM
at a different line. The spill is bounded by the artifact (it holds exactly the tensors the GGUF will
hold, in exactly the form it will hold them), so the directory with room for the output has room for it.

## Alternatives Considered

**Fork instead of respawn — rejected because it deadlocks, and that is measured.** A fork was the
obvious shape: the child inherits the config object, the loaded checkpoint and every wrapper by
copy-on-write, so nothing needs serializing and nothing is loaded twice. Probed on this machine
(torch 2.8.0+cpu, coremltools 9.0), forking and then calling `torch.jit.trace` in the child:

| parent before the fork | child |
|---|---|
| no forward pass at all | converts, exits 0 |
| `torch.set_num_threads(1)`, then 20 forwards | converts, exits 0 |
| 20 forwards at the default thread count | **hangs indefinitely** |

The classic OpenMP-after-fork hazard: the child inherits the thread pool's locks without its threads.
It is not avoidable by ordering here, because **`phases()` itself runs real forward passes** — the
speech-LM family probes its encoder's row geometry and its padding invariance before it returns a
single phase, and that probe is exactly what makes `audio_geometry` safe to declare. Paying for a
second checkpoint load is the price of not having that hang.

**Load only that phase's submodule — not attempted, and it is not what the peak needs.** The backlog
named it as part of this change. It needs a per-family partial-loading hook that does not exist, and
which ~15 families would each have to implement; and what it would buy is *time*, not peak — the
child's peak is the checkpoint plus one conversion either way. The N+1 loads are the honest cost of
this shape and the place a later change would go.

**Spill without isolating — rejected as the primary mechanism.** Writing each phase's weights to disk
and mapping them back bounds the accumulation with no child process at all, and it is in fact how the
parent side works. It does nothing about the allocator arenas and framework caches a conversion leaves
behind, which is the half only an exiting process reclaims.

**Making isolation the default — rejected.** It is a property of the *machine*, not the model: the same
checkpoint isolates on a 16 GB laptop and has no reason to on a 128 GB workstation. For a model that
already fits it buys nothing and costs a checkpoint load per phase, so it is the one field on a
`Decomposition` that is not a fact about the model, and it defaults to off.

**Falling back to in-process conversion when a worker fails — rejected.** A caller turned isolation on
because the export did not fit. Finishing it the way that did not fit, quietly, is worse than useless:
`convert_phase_isolated` raises and names the worker.

## Consequences

### What it measures

Peak RSS, `--quantize Q8_0`, this machine (4 cores, 33 GB, no swap), idle. *self* is
`/usr/bin/time -v`'s maximum for the export process; *tree* is a 0.1 s sampler summing VmRSS over the
whole descendant tree, which is the number the machine actually has to satisfy. They coincide where
there are no children, which is the sampler's own control.

| | A: before | B: + packing per phase | C: + `--isolate-phases` |
|---|---|---|---|
| **granite-speech-4.0.1b** (3.2 GB artifact) | 20.99 | 21.11 | **14.56** self / **15.85** tree |
| **kokoro** (143 MB artifact) | 2.67 | 2.74 | 2.56 self / **4.34 tree** |

**Packing per phase did not move the peak on either model, and that is worth stating plainly rather
than describing what it was supposed to do.** It does shrink what is carried — `tests/ci` asserts that
a phase hands the merge uint8 blocks and not F32 arrays — but the peak is
`max over phases (what was carried, plus that phase's own conversion)`, and on both models the phase
that sets the peak is one where little had been carried yet. Granite's order is encoder → embed →
**decoder** → lm_head and the decoder is the 40-layer LM: 16.1 GiB was resident when it started
converting, of which the encoder's packed weights had saved 0.3. It is the right shape for a model
whose big phases come first or whose phases are many and alike, and it is not what saves Granite.

**Isolation is what moves the number, and it moves the two halves differently.** The largest thing any
one process must hold went **20.99 → 14.68 GiB**, a 30% cut, and that is the quantity that decides
whether an export OOMs on a machine with headroom for one big allocation. The whole tree went 20.99 → 15.85,
and getting it there took one more line. Without it the tree peak was 19.68 — barely better than the
baseline — because the parent sat at 5.0 GiB beside every child. `release_heap_to_os()` is what that
5.0 turned out to be: 1.21 GiB of interpreter with torch and coremltools imported, and **3.66 GiB of
arenas glibc had not returned**, invisible to `gc.collect()` because the objects were already gone.
One `malloc_trim(0)` takes the same parent to 1.31 GiB, and it stayed at 1.3 for all four phases.

That call earns its place *only* under isolation. In a single process the pages it returns are
immediately reused by the next phase, so it lowers the current RSS and not the peak; under isolation
the parent's residency is charged against every child, in parallel, for the whole conversion.

**On a small model isolation costs memory rather than saving it** — kokoro's tree peak is 4.34 GiB
against a 2.67 baseline, because a 143 MB artifact's peak is the framework floor and isolation pays
that floor twice. This is the quantitative form of the argument for leaving it off by default.

### Everything else

* A multi-phase export carries packed bytes between phases instead of F32 arrays, on every path,
  isolated or not. At Q8_0 that is roughly a quarter of what it carried — which, per the table above,
  is not the same claim as a lower peak.
* An export that does not fit has a switch. An N-phase model costs N+1 checkpoint loads under it —
  Granite's four phases took the isolated arm to roughly twice the wall time of the baseline.
* Both changes are output-preserving and are gated as such. `tests/ci/test_phase_isolation.py` requires
  an isolated export to be byte-identical to a non-isolated one at F32 *and* at Q8_0 — with the fixture
  widened so that quantization actually reaches something, since a Q8_0 arm over an 8-wide model is an
  F32 arm wearing its name. The `tests/gate` sweep requires the whole zoo's snapshots not to move: nine
  models — kokoro, matcha, lfm2-modular, qwen3, whisper, parakeet-tdt, dac-44khz, dia-1.6b,
  flan-t5-small — recorded from a `git worktree` at the merge base and diffed clean, with the sweep's
  own sabotage test confirming the comparison could still fail. And the product, not just the fixture:
  **`whisper-small` exported both ways from fresh processes is `cmp`-clean at 969,918,400 bytes**, three
  phases including a KV-cached one and a topology rewrite.
* **One thing can make an isolated export differ, and it is older than this change.** `coremltools`'
  `Builder.name_count` is a process-global class counter, so an op a MIL pass builds without an
  explicit name is `transpose_3` in a fresh interpreter and `transpose_21` in a warmed one, and that
  name reaches the topology. A `loom-export` process converts one model and exits and a worker starts
  clean, so both sides always start from zero — which is why whisper matches and why the sweep has
  never seen it. A long pytest session is the exception, and it is what surfaced this at all: the
  in-process arm had a counter twenty ahead of its own children's. The test resets the counter per
  export to reproduce CLI conditions rather than paper over it, and the hazard is filed on its own
  terms in [the backlog](../backlog/active-index.md#exporter--mil-compiler).
* `merge_phase_weights` now merges memory-mapped arrays on the isolated path. Its collision check is
  untouched and still cannot fire, for the same reason `_check_phase_weight_namespaces` now states
  outright.
* Voxtral is still ~29 GB against 28. **These changes do not make that model exportable**; a bigger
  machine is the honest answer, and Epic-03 §2 keeps the measurements for one.

## Related

* [Epic-02 §2](../epics/epic-02-mil-exporter-and-compiler.md) — where the write step sits in the
  pipeline.
* [Epic-03 §2](../epics/epic-03-model-coverage.md) — Voxtral's measurements, and why the zoo is picked
  by peak memory rather than by the template.
* [ADR-015](adr-015-ci-and-gate-test-classes.md) — why the byte-identity gate is the acceptance test
  for an output-preserving change, and why it has to be able to fail.
* [Retro-015](../retros/retro-015-export-snapshot-sweeps.md) — recording a baseline from a worktree
  with its own `cwd` and `PYTHONPATH`.
