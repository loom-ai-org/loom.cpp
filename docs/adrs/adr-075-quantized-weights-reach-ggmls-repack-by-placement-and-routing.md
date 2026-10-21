---
type: adr
status: accepted
date: 2026-10-11
domain: performance
tags: [cpu, quantization, q8_0, q4_0, cpu-repack, kleidiai, extra-buffer-type, loader, graph-builder]
---

# ADR-075: Quantized Weights Reach ggml's Repack by Placement at Load and Routing at Build

Serves [Epic-05](../epics/epic-05-edge-performance.md) and step 2 of the LiteRT/XNNPACK queue in the
[hub](../backlog/active-index.md#engine--performance). The code is `GgufModel::load`,
`GgufModel::route_repacked_uses` (`src/core/gguf_model.cpp`) and its one call in `GraphBuilder::build`.
Opt-in for now: `LOOM_CPU_REPACK=1` (ggml's `CPU_REPACK`) or `LOOM_CPU_REPACK=kleidiai`
(`CPU_KLEIDIAI`, in a `GGML_CPU_KLEIDIAI=ON` build). Whether it becomes the default is still open in the hub.

## Context

ggml's CPU backend has dedicated int8 GEMMs for Q4_0 and Q8_0: `CPU_REPACK` re-lays a matrix as
interleaved blocks at upload (Q4_0 for AVX2, NEON dotprod and i8mm; Q8_0 for dotprod and i8mm), and
KleidiAI packs it for Arm's micro-kernels. `scripts/bench_xnnpack.cpp` measured 1.6-2.3x at the GEMM on
the M1. loom used neither, because ggml chooses the path from **the weight's buffer type**, and loom put
every weight in the default buffer.

The difficulty is that a repacked tensor is a valid Q4_0 / Q8_0 tensor only to a direct `mul_mat`.
`GET_ROWS` on a tied embedding, ggml-0013's dequantize of a folded conv kernel, and any view (a
`RESHAPE` of a weight) all read the interleaved bytes as plain blocks. Nothing fails; they are just
wrong. A `mul_mat` with an F16 or 4-D activation trips an assert instead. Which consumers a weight has
is only known from the built graph, because primitives reshape and view weights before using them.

## Options

1. **Place by name or by an exporter tag.** Any convention misses a primitive that views a weight, and
   a tag costs a re-export of every model.
2. **Scan the topology JSON at load.** It sees the op names but not what each primitive lowers to (a
   `CONV_1D` becomes a `mul_mat` over a reshape, or a direct conv, depending on the path).
3. **Load plain, repack a twin at first use.** Correct, but every repacked weight is then held twice.
4. **Place every eligible weight at load, and at each graph build point the consumers the extra buffer
   cannot serve at a plain copy re-read from the file.** Chosen.

## Decision

At load, a 2-D Q4_0 / Q8_0 weight goes to the extra buffer if the CPU device's `supports_op` accepts
a probe `mul_mat` over it there. That is llama.cpp's test, and it answers per tensor and per CPU without
restating ggml's rules. CPU_REPACK must also set `extra`, because its `set_tensor` dereferences that
unchecked. Everything else is allocated in the plain buffer as before.

Before allocating a graph, `GraphBuilder::build` calls `route_repacked_uses`. That keeps a consumer on
the packed weight only if it reads it directly as `src[0]`, `supports_op` accepts it, and the
activation is at most 3-D with contiguous rows (the two things repack.cpp asserts rather than checks).
Every other consumer is pointed at a plain copy, and every view is re-pointed into it. The copy is
created once, the first time a consumer needs it.

## Consequences

* **Correct whatever the topology does.** `test_cpu_repack_routing` reads one weight through
  `mul_mat`, `GET_ROWS`, and a `RESHAPE` view: the last two are bit-exact against the switch-off arm.
  It was verified red by disabling each route in turn. The whole CI suite passes with the switch forced on.
  At model level (x86), forcing every consumer onto its plain copy left Qwen3's 64-token sequence
  bit-identical to the switch-off arm, so the routing itself is exact. A repacked `mul_mat` is within
  4e-7 relative of the plain one (summation order), and that alone moves greedy tokens: Qwen3-0.6B Q4_0
  parts at token 6. Every whisper transcript is unchanged.
* **The cost is a second copy of every MIXED-use weight.** A tied embedding is the case that matters:
  Qwen3-0.6B's 151936 x 1024 table is +165 MB at Q8_0 (+26%) and +87 MB at Q4_0, in both buffer
  types. whisper's decoder table is duplicated only under KleidiAI (+42 MB at Q8_0), because CPU_REPACK
  declines its 51865 rows. Keeping a `GET_ROWS`-read weight out of placement would remove the copy, at
  the price of running its `lm_head` `mul_mat` unrepacked.
* **Hybrid builds are excluded.** The scheduler plans its copies from each weight's buffer and would
  have to read a repacked one back, so the switch applies to a plain CPU primary only.
* **`read_tensor_prefix` refuses a repacked weight by name** rather than calling a `get_tensor` that
  the buffer does not have.

**Measured on the M1 Pro**, paired, one switch per pair (`paired_arms.py --env`), means in seconds:

| | off | `CPU_REPACK` | `CPU_KLEIDIAI` |
|---|---|---|---|
| whisper-small Q8_0, jfk, 1 thread | 4.010 | 2.988 (**1.34x**) | 2.627 (**1.53x**) |
| whisper-small Q8_0, 8 threads | 0.709 | 0.573 (1.24x) | 0.510 (1.39x) |
| whisper-small Q4_0, 1 thread | 4.466 | 2.893 (**1.54x**) | 3.043 (1.47x) |
| whisper-small Q4_0, 8 threads | 0.754 | 0.544 (1.39x) | 0.569 (1.32x) |
| Qwen3-0.6B Q8_0, 64-token decode, 1 thread | 1.616 | 1.202 (1.34x) | 1.940 (**0.83x**) |
| Qwen3-0.6B Q8_0, 8 threads | 0.601 | 0.594 (1.02x) | 0.589 (1.02x) |
| Qwen3-0.6B Q4_0, 1 thread | 1.650 | 1.060 (**1.56x**) | 1.100 (1.50x) |
| Qwen3-0.6B Q4_0, 8 threads | 0.457 | 0.431 (1.07x) | 0.451 (1.02x) |

On the Ryzen 3 3250U (2 threads, pinned), where only Q4_0 is repacked: whisper 1.28x and Qwen3 decode
1.18x. KleidiAI is the better engine for Q8_0 prefill and the worse one for Q8_0 decode at one thread.
At 8 threads, decode is bound by memory and neither moves it.
