---
type: retro
date: 2026-08-12
domain: exporter
tags: [refactoring, grep-scope, naming, dead-code, near-miss]
---

# Retro-016: A Field Proposed for Deletion Was Not Dead Code

## The Issue

A backlog item claimed the exporter's `profile` field was inert and proposed deleting it. **The premise
was wrong**, and deleting it would have renamed weights in every exported GGUF.

## Root Cause Analysis

The evidence for "inert" was that `LoomGGUFExporter` reads `self.profile` in one place and only against
`None`. That grep covered `exporter.py` and `register.py` and **missed the real users**:
`topology_ops.py` reads `self.profile == "monolithic"` in **eight** places — because `self` inside a
topology rule *is* the exporter. Each of the eight gates whether a weight gets a `{func_name}.`
namespace prefix.

The deeper finding: `profile` was a real switch wearing the wrong name. It did not name a profile; it
named "flatten the weight namespace", which *correlates* with monolithic-ness without being it.

## Resolution & Lesson Learned

The field became a plain `bool` named `flat_namespace`, read at the same eight sites, and decomposition
became a strategy object (`Flattened` / `Modular` / `MultiPhase`) rather than a mode string — see
[ADR-004](../adrs/adr-004-mil-as-the-single-export-path.md).

* **Actionable takeaway 1 — a grep proves absence only over the files it covered.** When the receiver
  of an attribute is passed as `self` into another module's rule table, the obvious search scope is the
  wrong one. Enumerate readers, don't grep for them.
* **Actionable takeaway 2 — a name that correlates with its effect is a latent bug.** `profile ==
  "monolithic"` was true whenever the namespace should be flat, until it wasn't going to be.
* **Actionable takeaway 3 — keep a rename out of a restructure commit.** The two landed separately on
  purpose, so neither diff had to be read through the other.
* **Actionable takeaway 4 — the correction is the most useful thing in an entry.** This item is kept
  precisely because its original premise was wrong.

---

## Full record (verbatim from the ledger)


**The premise this item was written on was wrong, and the correction is the most useful thing in it.**
The original entry claimed `profile` was inert, on the evidence that `LoomGGUFExporter` reads
`self.profile` in one place (`exporter.py`'s bespoke-path dispatch) and only against `None`. That grep
covered `exporter.py` and `register.py` and missed the real users: **`topology_ops.py` reads
`self.profile == "monolithic"` in EIGHT places** (`self` inside a topology rule *is* the exporter), each
gating whether a weight gets a `{func_name}.` namespace prefix — `namespaced_name`,
`gelu_tanh_approx.one`, and six more. Deleting the field, as this item originally proposed, would have
renamed weights in every exported GGUF.

What survives the correction, verified rather than assumed:

1. **The monolithic/modular *dispatch* really is on `modular_layout`**, never on `profile`
   (`exporter.py`'s `export()`). Since P0.1 retired `profile="atomic"`, no dispatch anywhere
   distinguishes `"monolithic"` from `"modular"` by value.
2. **The eight naming reads are all shadowed today.** Every one is `func_name == "main_topo" or
   self.profile == "monolithic"`, and the monolithic path emits exactly one topology, always named
   `main_topo` (`exporter.py:1166`). So for every current caller the profile half never decides
   anything — but it is a live guard, not dead code: a multi-topology export whose exporter was handed
   `profile="monolithic"` would flatten the namespace, and the modular path's deliberate *omission* of
   `profile` is what keeps its per-submodule prefixes.
3. **So `profile` is a real switch wearing the wrong name.** It does not name a profile; it names
   "flatten the weight namespace", which correlates with monolithic-ness without being it.

**What was built.** `decomposition.py` — `Decomposition` with `Flattened`, `Modular(spec, dummy_seq_len)`
and `MultiPhase`, each owning the trace-and-assemble mechanics that used to live in a family's own
`export()`. `LoomExportConfig` gains a `decomposition` field and a single `export()` that delegates to
it; no family overrides `export()` any more. The two causal-LM classes collapse into one
`LMCausalModelExportConfig`, so exporting LFM2 both ways is one type with a field set differently
instead of two types — the concrete thing this item existed to fix. `profile` is gone from
`LoomExportConfig` and now appears only where it is meant: inside `Flattened`-shaped families'
`backend_kwargs()`, with a comment at each site saying it controls weight namespacing.

**Why a strategy object rather than a mode string.** The three forms need genuinely different data
(`Modular` a spec and a non-colliding dummy length; `Flattened` a trace length and quantize mode;
`MultiPhase` the phase list). One config carrying every field with a string selecting which subset is
live makes invalid states representable and pushes checking into `export()`.

**The field is universal; the choice is not.** Only causal-LM currently accepts either decomposition,
because only LFM2 exports both ways from one checkpoint — a caller decision, which is why both its
recognizers deliberately `detect()` the same directory. Kokoro cannot be exported flattened and Qwen3
has no phases: for those families the decomposition is a structural fact, declared once via a
`default_factory` rather than chosen per run. `decomposition.py`'s module docstring states this, so the
next family does not read the uniform field as a uniform menu.

**Follow-up, landed separately: `profile` → `flat_namespace`.** Kept out of the restructure commit so a
rename and a restructure wouldn't share one diff. The flag is now a plain `bool` named for its effect,
read in the same eight `topology_ops.py` rules as `func_name == "main_topo" or self.flat_namespace`, and
passed only by the two `Flattened`-shaped families' `backend_kwargs()`. Its second, unrelated meaning is
gone: the bespoke hand-built-Program path is now decided by `is_bespoke` alone (`len(functions) > 1 and
"main" in functions`), since no caller ever passed a profile to suppress it — both call sites in
`export()`/`_ensure_mil_passes_applied` were checked, and the only two tests that passed
`profile="monolithic"` use single-function programs where `is_bespoke` is already `False`. That is the
one behavioral difference in the rename: a hypothetical caller handing the exporter a multi-function
`main` Program *and* a flat-namespace request would now take the bespoke path. None exists.
`LOOM_PROFILE`, the env override on the old field, had no readers anywhere in the tree and is gone.

