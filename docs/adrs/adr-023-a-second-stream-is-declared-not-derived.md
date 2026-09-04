---
type: adr
status: accepted
date: 2026-09-03
domain: inference-engine
tags: [kv-cache, engine, family-10, guidance, topology-schema]
---

# ADR-023: A Second Decode Stream Is Declared, Not Derived

## Context

Classifier-free guidance runs one decoder **twice per step** — once on the conditional input, once on
the unconditional one — over two histories that must never see each other. Family 10 is the first
model here to need it, and this engine's KV cache is single-sequence by design
([Epic-01 §4](../epics/epic-01-inference-engine-core.md#4-standing-scope-limitations)), so the second
run cannot be a second batch row the way `transformers` does it. It has to be a second **module**.

`Session` allocated **one** KV cache and handed it to every module whose topology reported
`uses_kv_cache()`. Under that rule the two streams share one cache: at step *t* the second run writes
over the cell the first just wrote, then attends to a mixture of the two histories. Nothing raises —
the shapes are right, the graph is right, and what comes out is a plausible sequence of codes.

The obvious move is to make `uses_kv_cache()` decide this too, and it cannot.
[KV-CACHE.md decision 5](../KV-CACHE.md) makes *whether a cache is reachable* a derived fact, on the
grounds that `op_attention` is the sole door to one so a declaration "could only ever agree with it or
be wrong". That argument does not carry over: **which** cache is not a property of the graph. Two
modules running one topology are two streams of one generation when a driver runs them side by side,
and two phases of one stream when it runs them in sequence. Both are legitimate, both look identical
in the JSON, and only the driver knows which it is doing.

## Options

**A. One cache per KV-using module.** No declaration at all; every module gets its own. Correct for
every model shipped today, because each of them has exactly one KV-using topology — verified across
the exporter (`fuse_attention=True` appears once per family) and the ci fixtures.

**B. Declare it: `"kv_cache_scope": "private"` on the topology, default `"shared"`.**

**C. Give the driver a run-time way to ask.** `Session` registers modules before the script runs, so
this needs the script to be able to create modules, which is a much larger change to what a driver is.

## Decision

**B.** A is simpler and is right for the fleet as it stands, and that is exactly its problem: it
forecloses the *other* reading with no way to spell it. A model with separate prefill and decode
topologies over one stream is an ordinary thing to want to export, and under A it silently gets two
caches and re-reads none of its own prefill. Making the shared case the default and the private case
declared keeps every existing GGUF bit-identical and makes the new one say what it means.

The mechanism is one optional top-level key in the topology JSON, and one field on `ExportPhase`:
`extra_streams` names additional copies of a traced topology, each emitted with
`kv_cache_scope: "private"`. An alias is the same JSON under a second key — **no weights are
duplicated**, since its nodes name the tensors the phase already wrote.

## Consequences

* **`Session` owns a vector of private caches** beside the shared one, declared before the bridge so
  the lifetime rule that made `Session` a class rather than a function still holds.
* **Cross-attention K/V need aliasing too, for a different reason.** They carry no cache; what they
  carry is a *retained output* that the decode loop reads at every step for the whole generation. One
  module holds one, so the unconditional projection would overwrite the conditional one before the
  next step read it. `extra_streams` covers both because both are "a second independent run of this
  topology" — which is what made it worth a general field rather than a cache flag.
* **The engine's memory doubles for the cached phase under guidance**: Dia's cache is ~226 MB, so two
  is ~452 MB against a 6.4 GB F32 model. Guidance also doubles the decoder work per step, which is
  its real cost and is inherent rather t