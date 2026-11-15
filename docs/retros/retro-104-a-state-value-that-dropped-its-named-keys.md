---
type: retro
date: 2026-11-09
domain: engine
tags: [lua-bridge, stream-state, streaming, encodec, lstm, silent-failure]
---

# Retro-104: A State Value That Dropped Its Named Keys

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming), from
[ADR-104](../adrs/adr-104-a-recurrence-carries-across-pushes-and-noise-belongs-to-its-frame.md).

## Issue

EnCodec's first live stream gave back the right number of samples, and from frame 2 onward they were 0.3 away
from one call. Pushing every frame in one go still left the tail 0.1 off. Each piece agreed with its
one-pass counterpart when probed alone: `pre` both ways, an LSTM layer through a Lua array and through a retained
reference, a carried and an uncarried sweep, `post` both ways. So the fault was in what happened *between*
pushes.

## Root cause

The stream kept each LSTM layer's carry as `loom.set_state('carry_1', {h = {...}, c = {...}})`. `set_state` takes a
number or a flat array, and it read the table with `read_number_array`, which walks `1..#t`. A table with only
named keys has length 0, so it was stored as `{}` with no error. The next push read an empty carry, and the
sweep started from zeros as if the stream had just opened.

## Takeaway

**A binding that coerces a value must refuse whatever it cannot hold, not keep the part it can.** That
`set_state` read a table's array part was never wrong for the arrays every earlier driver stored, and that is
why nothing caught it: the first value with named keys read as empty, not as an error. `set_state` now refuses
a table with any key outside `1..#t`, naming the value (`test_speech_stream`'s
`test_a_named_key_is_refused_not_dropped`, which fails with the check removed). The EnCodec fragment keeps
`h_l` and `c_l` under names of their own.

**When every piece agrees alone, the fault is in the seam between calls.** The probes, each run alone with no
state between them, said where the fault was not. The difference between "one push" and "two pushes" said
where it was.

The other readers of a Lua table that go through `read_number_array` (graph inputs, `run_recurrent`'s
sequence) take arrays only by contract and have not been given the same check.
