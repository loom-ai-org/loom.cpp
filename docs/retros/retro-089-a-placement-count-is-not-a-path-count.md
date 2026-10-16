---
type: retro
date: 2026-10-16
domain: performance
tags: [armv6, pi-zero, int8, extra-buffer-type, routing, instrumentation, gate]
---

# Retro-089: A Placement Count Is Not a Path Count

## The Issue

ADR-079 P4.33's `ggml-0043` taught `CPU_ARMV6_I8` Citrinet-256-ls's pointwise convolutions. On the board,
the engine's log said "173 weights (9.6 MB) in CPU_ARMV6_I8", the CI test held the conv path to its
contract bit for bit, and `scripts/bench62.cpp` timed the op at 1.21-1.52x over F32 on Citrinet's shapes.
Two gates in a row then gave the card **0.95-0.96x of F32**, level with ggml's own Q8_0 path on the same
file. Two hypotheses were built and measured before the cause was found: the channel-strided quantize
(a TLB argument: 1.3% when fixed), and an unfused bias ADD (the profile had the same ADD time in both arms).

## Root Cause Analysis

**The int8 convolution never ran.** A timer inside the op (a temporary build, `GGML_CPU_ARMV6_I8_PROF`)
counted **0 calls** on Citrinet and 48 on Moonshine. `GgufModel::route_repacked_uses` (ADR-075) keeps a
consumer on a packed weight only if the op is `MUL_MAT` or `GET_ROWS` with an activation of at most three
dimensions. Those are the conditions `CPU_REPACK`'s forward asserts instead of checking, and they were
applied to every extra buffer type. A convolution's input is 4-D, so every pointwise conv was pointed at a
plain copy of its kernel and ran ggml's Q8_0 convolution. That is why the int8 arm matched the Q8_0 arm,
and why the outputs stayed right.

**The engine said so, 274 times per gate:** "loom: plain copy of '...' for CONV_2D". The checks of the logs
grepped for "CPU_ARMV6_I8" (the placement line) and never for "plain copy".

**Every check in the chain was about the wrong thing:** the placement count (where the weights live), the
CI test (the kernel, called directly), the bench (the op in isolation), and the timing (the outcome). None
of them asked which code ran inside the model.

With the routing fixed (`CPU_ARMV6_I8`'s `supports_op`, which checks the layouts its kernels need, is
the whole test for it), the same card ran jfk in 10.6 s against F32's 12.7.

## Takeaways

* **Count the calls on the path you are crediting.** A placement log says where a weight lives, not which
  kernel reads it. A counter inside the kernel, enabled by an environment variable, answers in one run.
  Two rounds of hypotheses about why a kernel was slow were spent on a kernel that was not running.
* **When an A/B arm matches the control, suspect that the arm did not switch.** Int8 at 0.99x of ggml's
  Q8_0 on the same weights was the signature: the same code, run twice.
* **Read every line the engine prints during a gate.** Diff the stderr of the two arms, not just grep for
  the line you expect.
* **A rule written for one buffer type's limits is not a rule for every buffer type.** The routing guards
  were correct for `CPU_REPACK`, and silently wrong for a type whose `supports_op` already checks more.
