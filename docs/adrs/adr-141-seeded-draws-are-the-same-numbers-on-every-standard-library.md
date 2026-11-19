---
type: adr
status: accepted
date: 2026-11-19
domain: engine-core
tags: [random, reproducibility, libc++, macos, webassembly, determinism]
supersedes: []
---

# ADR-141: Seeded Draws Are the Same Numbers on Every Standard Library

Serves [Epic-01](../epics/epic-01-inference-engine-core.md). Found by
[ADR-139](adr-139-the-engine-runs-in-a-browser-on-webgpu-or-the-cpu.md)'s web gate.

## Context

The bridge's seeded draws (`loom.seed_rng`, `loom.gaussian_array`, `loom.uniform_array`, the sampler's
uniform, and a stream's own generator in `StreamState::Rng`) were `std::mt19937` with
`std::normal_distribution<float>` and `std::uniform_real_distribution<float>`. The standard fixes the
engine's output, but not the distributions'. libstdc++ and libc++ both run Marsaglia's polar method,
and libstdc++ returns the pair's `y` first where libc++ returns `x`. For seed 0, Linux gives
2.21220636, 1.16307867, 0.774003804, ..., while macOS and emscripten give 1.16307867, 2.21220636,
0.483804613, .... Uniforms agree.

So every model that draws Gaussian noise through the bridge (Pocket-TTS's latents, the flow-matching
TTSes' noise) made different audio from the same seed on a Mac or in a browser than on Linux. The web
gate showed it as Pocket-TTS 2M stopping at 61 440 samples where Linux stopped at 69 120. No Linux gate
could catch it, and the macOS gates pin their noise. `aten_randn` (Paradee, tiny-voca) is Lua and was
never affected.

## Decision

`loom/core/portable_random.h`: `NormalFloat` and `UniformFloat`, libstdc++'s algorithms written out.
They are `generate_canonical<float, 24>` over one 32-bit draw, then the polar method with libstdc++'s
pair order and its float/double conversions. They replace the std distributions in the bridge, the
sampler and `StreamState::Rng`, with the same interface (`operator()(engine)`, `reset()`).

Linux's numbers do not move: 0 of 10⁶ mixed draws differ from libstdc++'s own. macOS and the browser now
get them. The one thing left to the platform is `logf` and `sqrtf`, whose last bit may differ between
libms. That is rounding, not another sequence.

## Options not taken

* **Pin the noise in every cross-platform comparison** (the macOS gates' approach). That tests around
  the defect. A user calling `infer(seed=7)` on a Mac still gets different audio than on Linux.
* **Another generator** (Philox, PCG). That would make Linux's numbers move too, and every reference
  that seeds rather than pins would move with them. Matching the platform where the references were
  recorded moves nothing.
* **libc++'s order.** It is as arbitrary as libstdc++'s, and every recorded seed is libstdc++'s.

## Consequences

* `tests/ci/test_lua_bridge` now holds fixed libstdc++ values, where it used to compare against a
  `std::normal_distribution` compiled into the test. That comparison would pass on any library
  and so could not see this. Swapping the pair order fails two of its checks.
* The emscripten build gives the Linux values bit for bit (checked with the same program under
  `em++`). After the change, the web gate's Pocket-TTS case has the native sample count.
* Seeded audio on macOS changes, once, to Linux's. Models exported with pinned-noise references are
  unaffected.
