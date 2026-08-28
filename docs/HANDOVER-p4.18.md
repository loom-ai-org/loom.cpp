---
type: handover
status: transient
date: 2026-08-25
domain: performance
tags: [p4.18, whisper, onnxruntime, handover]
---

# Handover: the three things left in P4.18

**This file is TRANSIENT and is not a fifth documentation tier.** It exists so a cold session can pick
up three specific items without re-deriving their context. Everything durable in it already lives in
[Epic-05 §2](epics/epic-05-edge-performance.md), [Retro-012](retros/retro-012-optimizations-that-were-measured-out.md)
and the [hub](backlog/active-index.md); the hub is still the list of what is open.
**Delete this file when the three items below are closed** — do not grow it, and do not cite it as a
source. If it disagrees with the epic, the epic is right.

---

## 0. Orientation, in five lines

* Work from `/home/flavio/Dev/loom`. Three repos side by side: **`loom.cpp`** (engine, this repo),
  **`loom-exporter`** (PyTorch → GGUF), **`loom-py`** (bindings).
* Read `loom.cpp/CLAUDE.md` and `docs/backlog/active-index.md` first — that is the standing rule, and
  the hub is where these items are tracked.
* Branch `perf/tinyblas-k-tail`, at `40ad9ef`. `ci` 70/70 and `gate` 82/82 green there.
* Gate fixtures: `export LOOM_FIXTURES=/home/flavio/Dev/loom-engine-artifacts/v5`.
* Build: `cmake -B build && cmake --build build -j"$(nproc)"`. Threads are `$LOOM_N_THREADS`
  (unset means ggml's default of 4, whatever the machine has).

### The state these items start from

whisper-small is the one task still behind onnxruntime. At one thread on the 285K the split is
**encoder 5.46 s vs 2.38 (2.29x), decode 1.10 s vs 0.97 (1.14x slower)**. Since that split was taken,
two patches landed: `ggml-0010` (vectorised exact-erf GELU, 9.9x on the op) and `ggml-0011` (the
`k % KN` contraction tail, 2.15x at whisper's `A@V` shape). Two things were closed as measured out:
`SOFT_MAX`/`ggml_v_expf`, and the "half of all clips" reading of `ggml-0011`.

**This is where the remaining encoder gap is believed to be** — dev box, one thread, 96 calls against
96. **The loom column is COMPOSED, not re-measured**: it is the epic's shape-split table with `A@V`
scaled by `ggml-0011`'s measured 2.15x and GELU by `ggml-0010`. Item A exists to replace it.

| encoder piece | loom (composed) | onnxruntime | gap | share of what is left |
|---|---|---|---|---|
| **`QK^T`** (12) | 2570 ms | 1193 | **1377 ms** | **58%** |
| `Softmax` (12) | 949 | 515 | 434 | 18% |
| Q/K/V/O + fc2 (60) | 4866 | 4434 | 432 | 18% |
| fc1 (12) | 2493 | 2235 | 258 | 11% |
| `A@V` (12), post-`ggml-0011` | ~1230 | 1185 | ~45 | 2% |
| GELU, post-`ggml-0010` | 77 | 206 | −129 | *loom ahead* |
| LayerNorm | ~90 | 176 | −86 | *loom ahead* |
| conv frontend | ~200 | 176 | 24 | 1% |
| **total** | **~12.5 s** | **~10.1 s** | **~2.4 s → 1.23x** | |

**Do items A, B and C in that order of confidence, not necessarily of time.** A and B are independent
and use different machines and languages, so they can run in parallel. **C is gated on A.**

---

## 1. Machines, and the traps that have each cost a measurement

| | |
|---|---|
| **dev box** | Ryzen 3 3250U, AVX2, **2 cores / 4 threads**, 4 MB L3. Thermally noisy. `taskset -c 0,2`, medians of seven. **Nothing under ~1.2x is resolvable here.** |
| **workstation** | `192.168.1.100`, Core Ultra 9 285K (Arrow Lake), 24 cores, **P-cores are CPU 0-7, E-cores 8-23**. No `cmake` on the default PATH — use `~/micromamba/envs/buildtools/bin`. **`/home` is 99% full (17 GB)**; clean up after yourself. Its clock is ~300 s behind, so `touch` files after an `rsync` or ninja loops on future mtimes. |
| **Pi** | `ssh pi@rpi4` (IPv6; the IPv4 moves). Cortex-A72, 4 cores. NEON has `KN = 4`, which is why whisper's `k % KN` cliff is invisible there. |

**Read [Retro-012](retros/retro-012-optimizations-that-were-measured-out.md) before opening anything.**
It is the register of measured-out ideas. The four rules that have each been paid for:

1. **Check the box is idle first.** `uptime`; `ps -eo pid,pcpu,comm --sort=-pcpu | head`. On the dev box
   the noise is usually a process the previous session backgrounded — one runaway `find` at 124% CPU
   produced two bench runs that contradicted each other by 1.4x. `taskset` does not save you.
2. **Pair the arms, do not interleave them.** Two arms back to back inside one round, the *ratio*
   recorded per round, reported as a median with p10/p90. `scripts/paired_arms.py` does this for two
   builds of one `.so`; `scripts/bench14.cpp` does it for two GEMM kernels.
   **A p10 that crosses 1.0 is "unresolved", not a number.**
3. **A floor arm must be built the same way as the arm it bounds** — a template parameter or an `#if`
   on the *real* function, never a simpler program that does the same job. Two verdicts were reversed
   by getting this wrong.
4. **Count the operations before measuring.** A two-of-sixteen removal cannot be a 1.4x. That is what
   closed the `ggml_v_expf` item after a working patch had already been written.

**And pairing does not cancel everything.** On the 285K, thread placement is drawn fresh *per process*,
so a paired test cannot cancel it: at 24 threads, 41 paired rounds still straddle 1.0 for an effect
that is a clean 1.22x at one thread. Multi-thread end-to-end numbers on that box need a different
estimator, not more rounds.

---

## 2. Item A — re-measure the encoder PER SHAPE (do this first)

**Why it is first, and it is not about the README.** The table in §0 has one measured column and one
composed one, and *every other decision here rests on it*. If `A@V` did not move in-model the way the
microbench predicted, "58% is `QK^T`" is wrong and item C is aimed at nothing.
The README's ASR column is a by-product: it was re-measured in `646c91c` and `ggml-0011` landed in
`aaa3172` immediately after, so **the published ratio predates a patch worth 1.06x (Ryzen) / 1.085x
(285K) on whisper.**

**What to run.** Both harnesses are in `scripts/`, and `scripts/bench_onnx_tasks.py`'s header block
says where the models and Python environments live on each of the three machines — read it, do not
guess.

* loom's per-shape side: `$LOOM_PROFILE=<path>` at **one thread** (`$LOOM_N_THREADS=1`; at four the
  per-node dispatch floor dominates and the attribution misleads). Add `$LOOM_PROFILE_NODES=1` to get
  the node-name table if any bucket is ambiguous — that is what it is for.
* onnxruntime's per-shape side: its own per-node profile (`so.enable_profiling`), aggregating `Node`
  events whose name ends `_kernel_time` by `args["op_name"]`, then **split by input shape**. This
  build's events carry no `run_index`, so split runs by order. **Profiling costs onnxruntime ~1.18x**,
  so use *shares* only and apportion them over an un-profiled wall time measured separately.
* end-to-end cells: `scripts/bench_{asr,vits,lm}_loom.cpp` against `bench_onnx_tasks.py`, as medians
  over **separate process launches** on the 285K.

**Three traps, each of which has already invalidated a run:**

* **Name the onnxruntime build.** conda-forge's build of the *same version* is **1.86x** faster than
  the PyPI wheel on VITS. The README's baseline is the PyPI `onnxruntime` 1.28.0. Quoting a number
  without the build is quoting nothing.
* **Do not use `optimum`** — ~2x of its own overhead on whisper.
* **Do not put the benchmark binaries under `/tmp`.** systemd-tmpfiles reaped them mid-sweep and the
  wrapper printed blank rows with exit code 0. A missing arm must be louder than a slow one.

**Acceptance.** The shape-split table in [Epic-05 §2](epics/epic-05-edge-performance.md) replaced by a
measured one, and the README's ASR column re-measured with the build named. **Then re-read the gap
shares before starting item C.**

---

## 3. Item B — the decoder's loop-invariant V transpose (the one guaranteed win)

**The finding.** whisper's decoder re-materialises the transpose of its cross-attention V **every
decode step, in every layer** — 12 nodes × 4.6 MB per token, **9.0% of the whole transcription and 47%
of the decode loop** at one thread. `xv_N` is a *graph input* that the cross-KV fix already made
constant for the utterance, so this is a loop-invariant transform of a constant.

The chain is nodes 42-44 of the decoder topology (`RESHAPE(xv_N) → PERMUTE(0,2,1,3) → PERMUTE(1,0,2,3)
→ CONT`), and **the `CONT` is genuinely required** — a double permute leaves `nb[0] != type_size`,
which `ggml_mul_mat` will not accept. The fix is not to delete it; it is to make it unnecessary.

**Where.** `loom-exporter/loom_exporter/whisper_export.py`, `_WhisperCrossKvWrapper.forward` (line
147) — currently `return tuple(proj(xa) for proj in self.projs)`. Emit the **V half** already
head-split and transposed so the decoder's chain becomes a no-op and the copy happens 12 times per
utterance instead of 324. Its own docstring already anticipates the cost. **This is exporter work, not
engine work** — per [ADR-003](adrs/adr-003-per-model-complexity-in-the-exporter.md), which is the
standing rule for this repo.

Note the K and V halves are **interleaved** (`k_0, v_0, k_1, v_1, …`) so the driver's index arithmetic
is `2*layer + 1`; only the odd entries change shape, and `cross_kv_input_names` must keep agreeing with
the driver.

**How to verify it worked** — this is what P4.19 was built for:

```sh
LOOM_N_THREADS=1 LOOM_PROFILE=/tmp/p.txt LOOM_PROFILE_NODES=1 \
  ./build/tools/loom_cli/loom_cli --model <whisper.gguf> --wav samples/jfk.wav
grep CONT /tmp/p.txt | grep 1500,64
```

Before the fix that shows **twelve `xv_N (reshaped) (permuted) (permuted) (cont)` buckets at 27 calls
each** (~2276 ms) plus one unnamed encoder bucket at 12 calls (~89 ms). After it, the twelve should be
gone or down to one call each.

**Gate.** `test_e2e_whisper_mil_export` compares the encoder against HuggingFace at
`max_abs_diff < 1e-3 * ref_absmax`. Current margin: `max/absmax` **4.34e-04** against the 1e-3 limit.
This change is a layout change and should not move it at all; if it does, something else happened.

**⚠ ASK THE USER BEFORE STARTING — this is a release-sequencing decision, not a technical one.**
The change needs a **whisper re-export and a fixture refresh** ([ADR-003](adrs/adr-003-per-model-complexity-in-the-exporter.md)),
which means a new Hub push. **rc6 is ready to tag except for three already-stale Hub models**
(whisper, vits, matcha). This makes a fourth. Either tag rc6 first and land this in rc7, or hold rc6
and do one combined Hub push. **Do not decide this unilaterally.**

---

## 4. Item C — `QK^T` at `k = 64`, with a hardware profiler (gated on item A)

**The size of it.** `k = 64` divides `KN = 8`, so `QK^T` *does* enter tinyBLAS — at **23.5 GFLOP/s
where the same core does 44 at k ≥ 256**. MLAS drops only 9% over that range, so it is a property of
ggml's kernel, not of the shape. It is ~1.1 s of the one-thread dev-box encoder and, per §0, **58% of
the remaining encoder gap** (confirm against item A's measured table before starting).

| `m=n=1500` | k=64 | k=128 | k=256 | k=512 | k=768 |
|---|---|---|---|---|---|
| GFLOP/s | **23.5** | 36.7 | 41.6 | 43.8 | 44.4 |

`scripts/bench13.cpp` section 2a is that sweep; `scripts/bench14.cpp` is the paired harness. Note this
sweep is **not** flattened by `ggml-0011` — every `k` in it already divided 8.

**Split it before spending anything.** Roughly 2.1x against a projection-shaped witness, of which:

* **~1.15x is explained**: tinyBLAS's `BM` row blocking, which whisper's `m = 1500` cannot reach
  (`1500 % 16 == 12`, so it gets `BM = 1`). Worth ~1.7% of a transcription. The fix shape is a
  **cascade** — 16-row prefix at `BM=4`, then 8, then 4, then the existing ≤3-row 1×1 tail — which
  needs a row offset threaded through `gemm`'s job partitioning. The naive version (a 16-row prefix
  with a 15-row `gemm_bloc<1,1>` tail) is a **regression**, because that tail is one `hsum` per element.
* **~1.8x has no identified mechanism**, and that is the part worth ~7% end to end.

**Four mechanisms are already falsified — do not re-test them:** a materialised dense transpose of the
permuted `src0` (worth 4%); rows-inner loop order for store locality (**mechanism falsified** — no
dependence on the size of C); a cheaper `hsum` epilogue (1.06x, under the noise floor); `ggml-0002`'s
aarch64 address hoist applied to x86 (neutral). **Do not write a fused attention kernel** — that is
separately ruled out, because onnxruntime does not fuse attention either and materialises the same
score matrix loom does.

**Start with counters, not a fifth guess.** `perf` is now installed on the workstation
(**version 6.12.105**, installed by the user 2026-08-25). Verified working; four things about it:

* **It is a hybrid PMU.** Events split across `cpu_core` and `cpu_atom`, so **pin to a P-core and
  prefix the events**: `taskset -c 0 perf stat -e cpu_core/cycles/,cpu_core/instructions/ …`.
  Un-prefixed events silently get counted on both and the shares are meaningless.
* **Basic counters work per-process without sudo** (`perf_event_paranoid = 2`) — verified for
  `cycles`, `instructions`, `branches`, `branch-misses`.
* **The `topdown-*` events do NOT work per-process**: `"Invalid event in per-thread mode, enable system
  wide with -a"`. System-wide needs `sudo perf stat -a` or `kernel.perf_event_paranoid <= 0` (it is
  currently **2**). **Ask the user** rather than changing a sysctl or running sudo on their machine.
* **There are no named microarchitectural events** for this core in this perf build — `perf list` has
  zero matches for `fp_arith*` or `uops_*`. If port-level counts are wanted, take the raw encodings
  from Intel's perfmon event JSON **for Lion Cove**; do not guess encodings from an older core.

**The first question is narrow, and it splits the remaining 1.8x into tractable or not:**
*is the FP port saturated?* Cheap version with counters that are already known to work: run the k=64
shape and the k=768 witness and compare **retired instructions against the theoretical FMA count**. If
the kernel retires only slightly more instructions per FMA at k=64 than at k=768, the time is real work
and the mechanism is somewhere nobody has looked. If it retires far more, it is per-tile overhead
amortised over only 8 `k`-iterations — which makes the `BM` cascade the actual fix rather than a
consolation, and makes unrolling or a `k`-specialised kernel worth pricing. `topdown` (front-end vs
back-end bound) answers the same question more directly if the user is willing to grant sudo.

**Do it on the workstation, not the dev box** — two cores cannot test threaded job partitioning, and
the dev box's noise floor is larger than most of the effects involved.

---

## 5. What NOT to spend time on

Each of these is measured, with the numbers in
[Retro-012](retros/retro-012-optimizations-that-were-measured-out.md):

* **A faster `exp` for `SOFT_MAX`.** Specialising `ggml_v_expf` to the domain is 1.00x on the Ryzen and
  1.13-1.16x on the 285K at identical accuracy; deleting the polynomial *entirely* is only 1.8x, and
  the exp is 7-26% of an op that is 5.9% of the encoder.
* **The `SOFT_MAX` pass fusion.** Real (1.19x Ryzen / 1.08x 285K on the op) and declined: ~1% end to
  end for a ggml patch carried forever. It is the best-understood item on the list and the least worth
  doing.
* **"`SOFT_MAX` is DRAM-bandwidth bound."** False on both machines — against a `memcpy` of the same
  bytes ggml's row body is 3.6x (Ryzen) and 7.8x (285K).
* **Fused attention.** Ruled out twice.
* **The dense GEMMs.** Within 10-12% of MLAS, against a witness already at 84-88% of the machine's
  real single-core peak. There is nothing there.

One thing that is **not** a gap but is often mistaken for one: whisper pads every clip to 30 s, so an
11-second file pays the full 1500-frame encoder on **both** engines. Shortening it is a
model-semantics change (the positional embedding is fixed at 1500).

## 6. One piece of context worth carrying

`ggml-0011`'s mechanism is the **attention head dimension**, not the clip length. Conformer-CTC's
`176 / 4 = 44` misses `KN = 8` on every utterance, which is why it gains 1.20-1.24x flat across six
clip lengths whose encoder residue alternates. **whisper is the worst case for this class of fix** —
head dim 64, already aligned — and its own 1.06x comes from `A@V` over its *fixed* 1500 frames. If the
goal is loom's ASR column rather than whisper specifically, that asymmetry is worth knowing before
choosing what to fund.
