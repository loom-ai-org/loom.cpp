---
type: retro
date: 2026-10-28
domain: backends
tags: [vc4, qpu, emulator, vpm, mutex, concurrency, raspberry-pi-zero, hang]
---

# Retro-097: An Emulator That Runs the QPUs One at a Time

Serves [Epic-04](../epics/epic-04-backends-and-accelerators.md) (P4.45); the decision is
[ADR-089](../adrs/adr-089-the-qpus-take-convolutions-and-per-channel-int8.md). The trap before it is
[Retro-095](retro-095-an-emulator-that-allows-what-the-hardware-breaks.md).

## Issue

The 1-D convolution kernel (`qpu_conv1d.h`) passed its emulator test bit for bit at seven shapes, with
both sabotages failing. On the Pi Zero, one tile on one QPU was correct; the first full launch -- 12
QPUs, Citrinet's pointwise convolution -- finished on none of them. The next launch of a known-good
probe timed out in the firmware, and a soft reboot never came back: two power cycles.

## Root cause

The kernel wrote the VPM DMA stride setup once at program start, outside the VPM mutex, because it
never changes. The VPM and its DMA engine are shared by all 12 QPUs; every other setup write in every
kernel is under the mutex. One QPU's stride write landing inside another's setup-address-wait sequence
is a race only a concurrent run can have, and the emulator runs the QPUs one after another, so it had
no way to see it. Moving the write under the mutex fixed it (Citrinet 0/66 816 wrong on 12 QPUs).

Bisected, not guessed: block mode at one tile on one QPU passed, so the k loop, the TMU gather and the
uniform weights were right; 12 QPUs hung in both output modes, so the cause was what they share.

## Takeaway

**An emulator that serializes threads must refuse, not model, whatever could race.** The emulator now
rejects any VPM write, VPM/VDW setup or DMA store outside the mutex, a second acquire and a release
unheld -- and the original kernel fails it at the exact instruction. More generally: for every
resource a sequential model shares implicitly, ask what the hardware's concurrency adds, and encode the
discipline that makes it irrelevant as a hard rule the model enforces.

And for the board: launch anything new at one tile on one QPU first, then at full width, chained with
`&&` -- a `;` sent the next launch onto stuck QPUs.

## Record

* `test_conv1d_hw block 1 0` (one tile): 0/288 wrong. `block 12 7` (Citrinet, 12 QPUs): 0/12 done
  markers in 2 s, then `mailbox tag 196625: Connection timed out` for `probe_unpack`; `systemctl reboot`
  did not return (no ping after 8 min).
* Fixed: `block 1 7` 0/66 816 wrong 152.8 ms; `block 12 7` 16.4 ms; `rows 12 7` 18.5 ms.
* Emulator on the old kernel: `emu @12: VPM / VDW setup outside the VPM mutex`.
