---
type: retro
date: 2026-09-09
domain: performance
tags: [armv6, benchmarking, measurement, raspberry-pi, tooling]
---

# Retro-037: `ps` Said 6%, `top` Said 50%, and a Day of Numbers Went With It

## The Issue

P7.1's second item — the im2col patch-batch budget on ARMv6 — was measured on the Pi Zero W across
four sessions. Every arm came back roughly **1.65x slower than the day before**, including a
configuration whose output was byte-identical to a run recorded at 100.0 s and which now took 165 s.

The numbers were internally consistent, reproduced across two independent runs, and completely wrong
as a description of the engine. Worse, the *previous* day's numbers — already committed to
[Epic-08 §6.8](../epics/epic-08-packaging-and-release.md), the hub, a patch comment and a commit
message — were contaminated by a weaker dose of the same thing: the shipped configuration is **80.9 s**
on a clean board, not the 100.0 s that was published, and the conv1d budget is worth **1.231x**, not
the 1.371x.

## Root Cause Analysis

`/usr/local/bin/bluetooth-km-switch`, a service unrelated to any of this, had entered a spin. It was
looked at early and dismissed on this evidence:

```
%CPU %MEM     ELAPSED STAT CMD
 5.7  0.2  4-07:59:21 Rsl  /usr/local/bin/bluetooth-km-switch
```

**`ps`'s `%CPU` is CPU time divided by process lifetime — an average since start, not a rate.** Over
four days of uptime a process that has been spinning for the last several hours reads as single
digits. `top`, whose `%CPU` is a delta between samples, showed what was actually happening:

```
  PID USER      PR  NI    VIRT    RES  S  %CPU  %MEM     TIME+ COMMAND
 6068 root      20   0   59388    648  R  49.8   0.1      6,54 bluetooth+
28227 pi        20   0  186716  96420  R  48.6  22.0   1:13.91 python
```

Half the core. On a single-core board that is not noise on top of a benchmark, it *is* the other half
of the benchmark — the model process was getting 48.6% of the CPU instead of ~95%.

**A wrong explanation was found first, and acted on.** `vmstat` showed `r` at 4-6 with zero idle, and
`ps` showed `wireplumber`, `pipewire` and `systemd --user` all with elapsed times of ten seconds —
which is real: every `ssh` login starts a user session and with it the audio stack. That is a genuine
effect and the polling protocol was rewritten around it (launch detached, wait locally, collect once).
It changed the numbers by nothing at all: 158.34 s held-connection against 157.30 s detached. The
mechanism was plausible, the evidence was consistent with it, and it was not the cause.

What finally isolated it was not a better guess. It was **an anchor**: the same wheel that had run at
100.0 s, reinstalled and re-run in the current session, at 165.59 / 164.64 s. A configuration whose
output hash is known cannot get 65% slower for any reason inside the binary, so the binary was
exonerated in one step and everything after that was about the board.

## Resolution & Lesson Learned

`systemctl stop bluetooth-km-switch` (a plain kill would have been undone by its `Restart=always`).
The board went from `r` 4-6 and 0% idle to 96.4% idle, and the anchor came back to 79.7 / 80.1 s.
Every measurement in P7.1 was then re-taken.

* **Actionable takeaway 1 — `ps %CPU` cannot answer "is this box busy right now".** It is an average
  over the process's whole life, so the longer the runaway has been up the more innocent it looks.
  Use `top -b -n 2` and read the SECOND sample, or `pidstat`. The check for an idle box is the
  `%Cpu(s)` idle line, not a scan of `ps` output.
* **Actionable takeaway 2 — put a known configuration in every session, and make it an arm.** Not a
  note of what it measured last time: an actual run, in this session, of something whose answer is
  already recorded. It costs one arm and it is the only thing that distinguishes "my change is slow"
  from "this machine is slow today". Both of this session's sweeps were self-consistent and useless
  without it.
* **Actionable takeaway 3 — a plausible mechanism that survives a check is still not a cause.** The
  ssh/pipewire finding was true, verifiable, and worth keeping ([[env-pi-zero-ssh-starts-pipewire]]) —
  and it explained none of the gap. Having found a mechanism, measure how much of the effect it
  accounts for before acting as though the question is closed.
* **Actionable takeaway 4 — absolute numbers on a shared board have a shelf life; ratios have a
  little more.** The published 137.1 -> 100.0 s was wrong in both terms, but the direction and the
  decision it justified survived. Quote ratios where the argument only needs a ratio, and re-anchor
  before quoting seconds.

## See Also

* [Retro-036](retro-036-one-switch-two-decisions.md) — the other measurement failure in the same item:
  one switch gating two decisions
* [Retro-012](retro-012-optimizations-that-were-measured-out.md) — the register of measured-out ideas,
  and why an entry names an incumbent
* [Epic-08 §6.8](../epics/epic-08-packaging-and-release.md) — the corrected ARMv6 numbers
