---
type: retro
date: 2026-09-25
updated: 2026-09-26
domain: host-api
tags: [loom-py, memory, gate, model-cards, moss]
---

# Retro-061: A Model in a Reference Cycle Kept Its Weights After `del`

## The Issue

The model-card gate for MOSS-TTS was SIGKILLed (exit 137) on its second row, with no traceback and a
log that stopped mid-line. Each row loads the card's pair, the 16.8 GB LM and the 4.3 GB codec, and
this box has 33 GB of RAM.

## Root Cause

`loom.Model.__init__` stored one interface object per door on the model
(`setattr(self, "text2codes", Text2Codes(self))`), and each interface stores its model. So every
`Model` was a reference cycle, and CPython frees a cycle only when the cycle collector next runs,
not when the last name goes away. The first row's 21 GB pair was still resident when the second row
loaded its own. Measured directly: `weakref(model)` is still alive after `del model`, and dead only
after `gc.collect()`.

Nothing had shown it before because nothing loaded two models this large in a row. Dia's card holds
6.6 GB, where the leftover copy fits.

## The Fix

The interfaces are properties built on access (loom-py `loom/__init__.py`). An interface holds its
model and the model holds no interface, so `del model` frees the engine's weights at once, while a
door kept on its own (`door = model.text2codes`) still keeps its model alive.
`test_dropping_a_model_frees_it_without_the_cycle_collector` pins that with the collector disabled,
and fails against the old code. With the fix the card gate passes, 2 passed and 6 not applicable on
each card.

## Takeaway

**An object that owns gigabytes must not be in a reference cycle**, because for it "freed later" means
"freed after the next allocation fails". Check with a `weakref` and `gc.disable()`, not by watching
RSS. And a SIGKILL with a truncated log is memory until proven otherwise; the gate's own memory note
had recorded exactly that symptom before.

## It Came Back One Layer Out (2026-09-26)

MOSS-TTS's card gained a voice-cloning block, and the gate was OOM-killed again: exit 137, peak
27.3 GB, on the row after `test_the_card_runs`. The model had no cycle this time. The cloning block
needs the reader's own voice file, so that row now SKIPS, and pytest keeps a skip's exception on the
report. The traceback keeps the test's frame, and the frame keeps the namespace `run_card` returned,
with the 16.8 GB LM and its codec in it. Before the card changed, that row passed, so nothing held
the frame. The gate now empties every card namespace at teardown (an autouse fixture in
`tests/gate/test_model_cards.py`), and the same run peaks at 23.0 GB and passes.

**The takeaway widens:** a test that holds gigabytes must also release them on the paths where it
does NOT pass, because a skip or a failure keeps its frame alive for the rest of the session.

## Related

[ADR-050](../adrs/adr-050-a-codec-declares-its-absent-id-and-its-channels.md),
[Epic-06](../epics/epic-06-high-level-api-and-hosts.md).
