---
type: retro
date: 2026-09-02
domain: performance
tags: [convolution, quantization, metal, gemm, benchmarking, scoping, p4.30c]
---

# Retro-028: Three Closing Arguments That Were Never Measured

## The Issue

P4.30c gathered six small remainders of P4.10–P4.29. Three of them were written with a number
attached and an instruction to close on it:

* **`op_conv_2d`** — *"Nothing in tree has a quantized 2-D convolution hot enough to notice… **If it
  does, close it unfixed.**"*
* **`ldc` alignment** — *"Worth **~4 ms of a 3.9 s transcription**… If the 4 ms holds, that ratio is
  the argument for closing it unfixed."*
* **The Metal `PAD` kernel** — *"Do not pick it up expecting a speedup… On Apple it is worth 1.8%."*

**Two of the three numbers were wrong, and the third had already been flagged as stale by the person
who wrote it.** One item became a shipped ggml patch, one became a shipped engine change, and only the
one whose number was re-derived from scratch closed the way it was expected to.

| item | the number it was filed with | the number it has | outcome |
|---|---|---|---|
| `op_conv_2d` | "nothing hot enough to notice" | **1.8x on two ops, 6% of a Q4_0 transcription** | shipped |
| Metal `PAD` | 1.8% (493.3 → 484.7 ms) | **11.0% (97.9 → 88.7 ms)** | shipped as `ggml-0016` |
| `ldc` alignment | ~4 ms of a 3.9 s transcription | **~49 ms of a 3.95 s transcription** | still closed unfixed |

## What Happened, Item By Item

### `op_conv_2d`: the evidence named in the item could not answer the question

The item said to decide on the profiles step 1 produced — Matcha, Kokoro and StyleTTS2. **Those three
models contain no genuinely 2-D convolution at all.** A census of every topology in all seventeen
shipped models finds 2-D convolutions in exactly four of them, all ASR encoders, and none of them was
in the step that was supposed to decide. The premise was not tested by the evidence it named; it was
restated by it.

Asking the right models found that qwen3-asr-0.6b's subsampling stem folds to `[4320, 480]` — block
aligned, so it quantizes — and that quantizing it costs **1.77x and 1.81x** on the two convolutions,
6% of a transcription. The fix was eleven lines and no ggml change: `ggml_conv_2d_direct_packed` had
taken a `kh` parameter since `ggml-0013` shipped, documented as *"1 for a 1-D convolution"*, and
nothing had ever passed it anything else.

### Metal `PAD`: the patch did not change, the denominator did

This one is the honest case, and it is here because the write-up did the right thing. §5.4 measured
1.8%, said so, and then wrote: *"re-measure it rather than quoting the percentage — the absolute
8.6 ms is the number that carried, and against the new baseline it would be 9%."* It was right on
every count. `ggml-0014` and `ggml-0015` cut the baseline from 493 ms to 98; the saving re-measured at
**9.2 ms**, within 7% of the 8.6 ms recorded a week earlier; and 9.2 of 97.9 is **11.0%**, which is a
patch worth shipping rather than a note worth keeping.

**A percentage measured against a baseline that is itself under repair has a shelf life. The absolute
does not.** Record both.

### `ldc` alignment: the ratio was right and the arithmetic around it was not

`15.0 ms against 10.9` is a per-CALL delta, and whisper-small issues twelve of that call. The
handover carried `15.0 - 10.9 = 4.1` straight into a sentence about the whole transcription. Re-measured
at HEAD the ratio holds exactly (1.31x at four threads) and the model-level ceiling is **~49 ms of
3.95 s, 1.25%** — twelve times what the closing argument assumed, and still not enough to buy an
allocator change. **The verdict survived; the argument for it did not.**

## What This Cost, And What It Bought

Nothing shipped on a wrong number — all three were still open, which is the system working. What it
cost is that **the pass had to re-derive the evidence for every item it was told it could close
cheaply**, and two of the three then turned into work. What it bought is two patches, a corrected
number, and three findings that only appeared because the ops were run rather than reasoned about:

* **A backend test that reports `not supported` is not a passing test.** `test-backend-ops -b MTL0`
  read 21/21 green on `PAD` while every case with a leading pad was being skipped. Allowing them
  exposed a real bug — the kernel indexed a possibly-permuted source as `T[]` — that failed at
  ERR 1.94 the first time the kernel was allowed to see the case.
* **A silent zero from a measurement harness is a dangling read until proven otherwise.**
  `scripts/tts_synth.cpp` bound a `const auto&` to a vector inside a by-value return. gcc left the
  storage readable for years of Linux runs; clang did not, and the same binary printed
  `samples=0 peak=0.0000` for all four TTS families on macOS, which reads exactly like four broken
  exports.
* **A follow-up nobody attempted can rot in place.** Writing the `OMP_WAIT_POLICY` report meant
  reading the code it proposed changing, and `ggml_backend_cpu_set_threadpool` — which Epic-05 §2 said
  was absent from the CPU backend's proc-address table and would need a tenth patch — is present at
  the pin, beside `ggml_threadpool_new`, and no patch of ours put it there.

## The Rule

**A closing argument is a measurement, not a scope note.** "Close it unfixed if X" is only a decision
if X was measured; if X is an assumption, the item is not scoped, it is deferred with a reason
attached. Three checks make the difference, and each of them caught something here:

1. **Name the evidence and check it can answer the question.** "Decide on step 1's profiles" was a
   real instruction pointing at models that structurally could not contain the thing being decided.
2. **Carry absolutes beside percentages, and re-derive a percentage whose denominator has moved.**
3. **Check the arithmetic between the op and the model.** A per-call delta is not a model delta, and
   the multiplier is the call count.

Related: [Retro-018](retro-018-a-table-of-ratios-nobody-could-re-derive.md) (ratios that could not be
re-derived), [Retro-027](retro-027-the-register-that-was-an-address.md) (candidate solutions in a
scoping note are hypotheses — measure them), [Retro-023](retro-023-a-bench-whose-graph-was-the-treatment.md)
(a harness that answers a different question than the one asked).
