<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-inline-dark.svg">
    <img src="assets/logo-inline.svg" alt="" width="52" align="middle">
  </picture>
  &nbsp;loom.cpp
</h1>

An inference engine built on [`ggml`](https://github.com/ggml-org/ggml) that hardcodes no model. A
model is a GGUF that carries its own **graph topologies** as JSON metadata and its own **driver
script** as embedded Lua, alongside the weights those describe; the engine parses them and builds the
compute graph at run time. Adding a model architecture is an export, not a C++ patch.

That is the whole architectural bet, and it is why this repo stays small: it targets edge devices, so
every per-model decision belongs in the exporter where it costs a Python change instead of a
specialized C++ driver. See [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md).

## What the engine offers a host, and what it leaves alone

The engine hardcodes no model, but it does own the loops every host would otherwise write. Those are
**per-task, not per-model** — one CTC decoder covers every CTC model — and they live here because the
copies hosts wrote had already drifted apart from each other:

| | |
|---|---|
| `loom::text::generate` | the causal-LM decode loop, both driver shapes, the file's own stop tokens and its own decode rule |
| `loom::ChatTemplate` | a conversation to the prompt text a checkpoint was tuned on, from role tags the exporter reduced its own Jinja to |
| `loom::audio::transcribe` | long-form ASR: windowing, segment splitting, and the seek to where the model closed its last segment |
| `loom::Session` | topologies registered and caches attached, in an order that cannot dangle |

Underneath them the low-level surface is unchanged and stays raw: `LoomLuaBridge::call` invokes the
driver with the driver's own arguments. The split is the same rule the whole tree is built on —
**in the file when it is a property of the checkpoint, in the engine when it is a property of the
task, in the host when it needs the host's ecosystem** — and
[`docs/HIGH-LEVEL-API.md`](docs/HIGH-LEVEL-API.md) is where it is argued.

A model says what it is, so a host never has to recognise one:

```cpp
const loom::ModelContract contract = loom::ModelContract::read(model);
contract.task;             // "automatic-speech-recognition"
contract.interface_name(); // "speech2text" -- the modality pair a host offers a door for
```

## The three repos

| | |
|---|---|
| [**loom.cpp**](https://github.com/loom-ai-org/loom.cpp) | this one — the runtime, its primitives and its Lua bridge |
| [**loom-exporter**](https://github.com/loom-ai-org/loom-exporter) | turns a PyTorch checkpoint into a GGUF this engine runs |
| [**loom-py**](https://github.com/loom-ai-org/loom-py) | Python bindings, with the engine as a submodule |

## Supported models

Seventeen, published at [huggingface.co/loom-ai-org](https://huggingface.co/loom-ai-org). Each is a
single GGUF carrying its own topologies, driver and — where the architecture has one — its vocabulary,
so the engine runs all of them without a line of per-model C++.

### Language models

| Model | Exported from |
|---|---|
| [`loom-ai-org/qwen3-0.6b-base-loom`](https://huggingface.co/loom-ai-org/qwen3-0.6b-base-loom) | [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) |
| [`loom-ai-org/lfm2-350m-monolithic-loom`](https://huggingface.co/loom-ai-org/lfm2-350m-monolithic-loom) | [`LiquidAI/LFM2-350M`](https://huggingface.co/LiquidAI/LFM2-350M) |
| [`loom-ai-org/lfm2-350m-modular-loom`](https://huggingface.co/loom-ai-org/lfm2-350m-modular-loom) | [`LiquidAI/LFM2-350M`](https://huggingface.co/LiquidAI/LFM2-350M) |
| [`loom-ai-org/smollm2-360m-instruct-loom`](https://huggingface.co/loom-ai-org/smollm2-360m-instruct-loom) | [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) |
| [`loom-ai-org/gemma-3-270m-it-loom`](https://huggingface.co/loom-ai-org/gemma-3-270m-it-loom) | [`google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it) |

The two LFM2 entries are the *same checkpoint exported two ways* — one fused graph against one topology
per layer — which is how the engine's two decomposition paths stay honest about producing the same
model.

### Speech recognition

| Model | Exported from |
|---|---|
| [`loom-ai-org/whisper-small-loom`](https://huggingface.co/loom-ai-org/whisper-small-loom) | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) |
| [`loom-ai-org/conformer-ctc-small-loom`](https://huggingface.co/loom-ai-org/conformer-ctc-small-loom) | [`nvidia/stt_en_conformer_ctc_small`](https://huggingface.co/nvidia/stt_en_conformer_ctc_small) |
| [`loom-ai-org/parakeet-tdt-0.6b-loom`](https://huggingface.co/loom-ai-org/parakeet-tdt-0.6b-loom) | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| [`loom-ai-org/parakeet-rnnt-0.6b-loom`](https://huggingface.co/loom-ai-org/parakeet-rnnt-0.6b-loom) | [`nvidia/parakeet-rnnt-0.6b`](https://huggingface.co/nvidia/parakeet-rnnt-0.6b) |
| [`loom-ai-org/gigaam-v3-rnnt-loom`](https://huggingface.co/loom-ai-org/gigaam-v3-rnnt-loom) | [`ai-sage/GigaAM-v3`](https://huggingface.co/ai-sage/GigaAM-v3) |
| [`loom-ai-org/qwen3-asr-0.6b-loom`](https://huggingface.co/loom-ai-org/qwen3-asr-0.6b-loom) | [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) |
| [`loom-ai-org/granite-speech-4.0-1b-loom`](https://huggingface.co/loom-ai-org/granite-speech-4.0-1b-loom) | [`ibm-granite/granite-4.0-1b-speech`](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) |

Each takes a raw waveform: the mel frontend is inside the graph, not in front of it.

### Speech synthesis

| Model | Exported from |
|---|---|
| [`loom-ai-org/kokoro-82m-loom`](https://huggingface.co/loom-ai-org/kokoro-82m-loom) | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) |
| [`loom-ai-org/matcha-tts-ljspeech-loom`](https://huggingface.co/loom-ai-org/matcha-tts-ljspeech-loom) | [Matcha-TTS (LJSpeech checkpoint)](https://github.com/shivammehta25/Matcha-TTS) |
| [`loom-ai-org/supertonic-2-loom`](https://huggingface.co/loom-ai-org/supertonic-2-loom) | [`Supertone/supertonic-2`](https://huggingface.co/Supertone/supertonic-2) |
| [`loom-ai-org/vits-piper-en-gb-miro-loom`](https://huggingface.co/loom-ai-org/vits-piper-en-gb-miro-loom) | [`OpenVoiceOS/pipertts_en-GB_miro`](https://huggingface.co/OpenVoiceOS/pipertts_en-GB_miro) |
| [`loom-ai-org/styletts2-ljspeech-loom`](https://huggingface.co/loom-ai-org/styletts2-ljspeech-loom) | [`yl4579/StyleTTS2-LJSpeech`](https://huggingface.co/yl4579/StyleTTS2-LJSpeech) |

**Only Supertonic takes text.** It encodes graphemes itself and its GGUF carries the codepoint table;
the other four consume *phoneme* ids that a phonemiser produces outside the engine, so their files
embed no vocabulary at all. That is a real limitation of those checkpoints, not a missing feature here.

## Performance against onnxruntime

The reference question for an edge runtime: **the same checkpoint, on the same machine, at the same
thread count — how does loom compare to onnxruntime?** Three tasks, because no single one is
representative: an all-convolutional vocoder, an encoder-decoder ASR model, and an autoregressive LM.

**`>1.00x` means loom is faster.** **Every TTS and LM cell was re-sampled per launch on 2026-09-02**
and those are the numbers below, on all three machines; the ASR column is from 2026-08-29 and is not
re-sampled here. **Both engines run back to back on the machine in the row** — one second apart on x86,
and on the Pi cooled to a fixed 60 C before each arm, which is that board's standing protocol. See
[what these numbers are not](#what-these-numbers-are-not).

The **default** row for each machine is the one at that machine's physical core count, which is what
the engine now uses when `$LOOM_N_THREADS` is unset.

| machine | arch | threads | onnxruntime 1.28.0 build | TTS<br>VITS (piper en-GB) | LM<br>Qwen3-0.6B | ASR<br>whisper-small |
|---|---|---|---|---|---|---|
| Intel Core Ultra 9 285K | x86-64 | 4 | conda-forge | **1.05x** | 1.02x | 0.64x |
| Intel Core Ultra 9 285K | x86-64 | **24 (default)** | conda-forge | 1.20x | 0.99x | **1.37x** |
| AMD Ryzen 3 3250U | x86-64 | **2 (default)** | PyPI wheel | 0.98x | **1.01x** | 0.80x |
| AMD Ryzen 3 3250U | x86-64 | 4 (all) | PyPI wheel | 1.00x | **1.02x** | 0.81x |
| Raspberry Pi 4B | aarch64 | **4 (default)** | PyPI wheel | 0.96x | **1.03x** | 0.57x |

**Half of the ten TTS and LM cells resolve, and those are the bolded ones.** In those two columns a
cell is bolded when the paired p10 and p90 over its launches both sit on one side of 1.00x; the rest
straddle it, which means the measurement did not separate the two engines — not that they tied at
exactly the printed number. **Read an unbolded TTS or LM cell as parity.** The five that resolve are
the 285K's four-thread TTS (p10 1.03, p90 1.17), both Ryzen LM cells (p10 1.00, p90 1.02-1.03), and
both Pi cells — its LM at p10 1.03 / p90 1.04 and its TTS at p10 0.95 / p90 0.98, the one cell in the
table that resolves as a **loss**. In the ASR column bold keeps its older meaning of simply `>1.00x`.

**The Pi is the best-behaved machine in this table, not the worst.** Cooled to 60 C before every arm it
holds a 1.02-1.05x spread per engine across launches, where the 285K's hybrid placement gives 1.45x —
so the board that needs the most protocol produces the tightest numbers once it has it, and both its
cells resolve. Its LM arms run `nrun=2` on both sides rather than the x86 rows' 3, because a generation
there is 47 s; the estimator is matched within the row, which is what the ratio depends on.

**Every TTS and LM cell is now a MEAN over separate process LAUNCHES** — per engine, 31 launches for
TTS on x86 and 15 on the Pi; 21 for the LM on the 285K, 15 on the Ryzen and 7 on the Pi, where a single
generation is 47 s. A mean rather than a median because at 24 threads both engines' VITS time splits
into two per-launch modes about 1.4x apart at roughly even odds, and a median over that is a coin flip
between them that moves the cell by the whole 1.4x. **This retires the Pi's old "report the fastest
run" special case**: cooling to a fixed temperature before every arm is what makes its launches
comparable to each other, so it can use the same estimator as the rest of the table. The ASR column is
still a median over launches (9 on the 285K, 7 on the Ryzen, 5 cooled pairs on the Pi), taken
2026-08-29.

**The re-sample confirmed the table rather than overturning it.** No TTS or LM cell on any of the three
machines moved by more than 0.04, the largest being the Ryzen's two-thread LM at 1.05x → 1.01x. Three
faults were fixed on the way there and none of them was worth much: `bench_onnx_tasks.py`'s `vits` task
had no warm-up while loom's did, `bench_vits_loom.cpp` took `ts[ts.size()/2]` rather than a median, and
the LM's equal-work check was asserted rather than printed. What *did* move cells is the second of two
back-to-back launches paying for the first — see
[Retro-025](docs/retros/retro-025-the-arm-that-ran-second-paid-for-the-first.md).

**All three machines were checked to be running byte-identical inputs on both sides**, and they are the
published artifacts rather than a working export: the VITS GGUF is `md5 949dd988…` and the
Qwen3-0.6B-Base GGUF is `md5 4882ce99…` (the files in `hf-models/`), with the onnx models matching
across all three boxes as well (`md5 395ed93d…` and `md5 44961f8a…`). The whisper GGUF was checked the
same way when the ASR column was taken (`md5 1deaac83…`). The Pi additionally reproduces the *same
audio digest* from the 62.8 MB post-P4.28 export and the 81.7 MB one before it — `fnv1a=aa320f8a…` from
both — which is what says the export that halved the file did not change the waveform.

**The Pi row was regressed by one of this repository's own ggml patches, and that is now fixed.**
`ggml-0011` — worth 2.15x at whisper's `A@V` shape on AVX2 — cost a Cortex-A72 **1.3-1.75x on every
f32 GEMM shape measured**, because it inlined a scalar tail loop into the register tile. NEON's
`KN = 4` divides every contraction these models have, so that tail never even ran; its presence alone
changed what GCC's allocator did with the tile. Dispatching it out of line, before the tile, restores
aarch64 without moving x86 at all:

| Raspberry Pi 4B, 4 threads | published 2026-08-24 | with the regression | **fixed** |
|---|---|---|---|
| TTS | 0.98x | 0.84x | **0.96x** |
| LM | 1.08x | 1.05x | **1.06x** |
| ASR | 0.58x | 0.45x | **0.57x** |

It was measured on x86 and shipped to every architecture, and **the Pi was not re-measured after it
landed** — which is the whole of why it survived four days.

**And it happened again with the next patch — now fixed, and the axis was not the ISA.** `ggml-0012` —
worth **2.75x** at whisper's `QK^T` on x86 — was checked on this board at that same shape and found
neutral, which it is. It was never checked at any other shape, and it cost the Pi **2.4%** on VITS,
most of it in the *convolution*, because this repository lowers convolution through the same `sgemm`.
**That is why the Pi's TTS cell moved 0.96x -> 0.93x between 2026-08-29 and 2026-08-30 with no change
to the harness.**

What the patch removes is a per-*output* cost and what it adds is per-*work*, so its benefit decays as
`1/k` while its cost does not: over a `k` sweep it is 2.04x at `k = 64` and 0.97x at `k = 2304` on the
285K, 1.01x and 0.90x on the Pi — monotone on both, crossing 1.0 at about the same place. **So the
predicate is `k`, not the ISA**, and `!defined(__aarch64__)` would have been wrong in both directions.
Gating the ragged prefix on `k <= 256` keeps every x86 win (whisper's `QK^T` improves further, 1.81x to
**1.94x**) and returns the Pi to parity with the unpatched tree. P4.26 in
[Epic-05](docs/epics/epic-05-edge-performance.md) has the sweeps;
[Retro-022](docs/retros/retro-022-a-benefit-and-a-cost-on-the-same-axis.md) is the lesson, which
generalises [Retro-019](docs/retros/retro-019-a-patch-measured-on-one-isa.md)'s "measure every ISA a
patch is enabled for" to **every regime it is enabled for**.

**The Pi's LM and ASR cells did not carry it**, which was the open question when this was found: the
LM's decode step has `ne1 = 1` and never reaches the blocked GEMM, and whisper's own `QK^T` is `k = 64`
— the regime the branch is *for* — and measures neutral on that board at the op level and unresolved
end to end (ASR median 1.007 over four paired rounds, p10 0.989, p90 1.016; LM median 0.995, p10 0.983,
p90 1.003 over six).

**The Pi's TTS cell is re-measured, and it reads 0.95x** — it was 0.93x with the regression and 0.96x
before it. Four rounds, both engines back to back on the board, **cooled to a fixed 60 C before every
arm** and the arm order alternated round to round, onnxruntime normalised to loom's 73 216 samples:

| round | loom | onnxruntime | ratio |
|---|---:|---:|---:|
| 1 (loom first) | 1.1154 s | 1.1172 s | 1.002 |
| 2 (onnx first) | 1.1118 s | 1.0567 s | 0.950 |
| 3 (loom first) | 1.1193 s | 1.0609 s | 0.948 |
| 4 (onnx first) | 1.1147 s | 1.0552 s | 0.947 |

**Round 1's onnxruntime arm is the session's first and pays the 63 MB model's cold page cache**; it is
kept in the table and in the median (0.949) rather than dropped, because dropping the round that
disagrees is how a number stops being reproducible. The other three span 0.5% and loom's four span
0.7%, which is what the cooling protocol is for.

**Re-sampled at 15 cooled rounds on 2026-09-02 the cell reads 0.96x**, which is the number in the
table: loom mean 1.1151 s over 15 launches (spread 1.05x), onnxruntime 1.0732 s (1.03x), paired p10
0.947 and p90 0.977, so it is one of the two cells in this table that resolve as a **loss** rather than
as parity. Four rounds had already put it within 0.01 of that, which is the useful part — on this board
the protocol, not the sample size, is what the number depends on.

**TTS and the LM are at parity on x86; ASR is still behind, and on the Pi loom wins the LM and loses
TTS.** Most x86 cells sit within a few percent of 1.00x, which on those machines is the noise floor
rather than a result — read them as parity, not as wins. The Pi is the exception in both directions:
its two cells are the tightest in the table and both resolve. The caveats below matter as much as the
numbers.

* **TTS is at parity everywhere**, 0.96x to 1.20x. That is what P4.14/P4.15 was for: a built-in F32
  GEMM micro-kernel, four convolution patches to the pinned `ggml`, and a duplicated text encoder
  removed from the export. Two cells resolve: the 285K's four-thread one at **1.05x**, and the Pi's at
  **0.96x**, which is the clearest *loss* in the table and the only place TTS is meaningfully behind.
  The 285K's 24-thread cell is the largest number in the column and also the one the launch lottery
  makes least certain (p10 0.94, p90 1.73 over 31 paired launches), so 1.20x is where the mean fell and
  not a win to quote. The Pi's cell read 0.84x for four days while `ggml-0011` was regressing it
  (above).
* **The LM is at parity**, a few percent either way, and it was the one task `ggml-0011`'s aarch64
  regression left alone — a decode step's `mul_mat` has `ne1 = 1` and never reaches the blocked GEMM,
  which is the control that made that diagnosis trustworthy. It only just became so — until 2026-08-23 the
  engine called every causal-LM driver's `infer` rather than its `infer_with_past`, so the host re-fed
  a growing prompt and each token recomputed the whole sequence. That was worth 2.83x. **Its one loss
  is the 24-thread cell, and that is the LM at a thread count that does not suit it**: a decode step's
  `mul_mat` has `ne1 = 1`, so this task peaks at 8 threads and plateaus after. It shows up as *spread* —
  loom ranges 22.0-28.6 tok/s over 21 launches there where onnxruntime holds 27.3-28.0. **Re-sampled
  per launch that cell is 0.99x rather than 0.96x**, which is parity and not a loss; what moved it was
  letting the machine settle between arms, worth 0.94x → 0.99x on its own. **Three LM cells resolve in
  loom's favour** — both Ryzen ones and the Pi's 1.03x — and they are the tightest measurements in this
  table (1.02-1.03x spread per engine). The LM is the one task where loom is ahead on the two machines
  a person would actually deploy on.
* **ASR is the one still behind — 1.24x on the Ryzen and 1.57x on the 285K at four threads, 2.2x on the
  Pi, against a 1.37x win at 24.** (The Pi is the outlier for a reason that is not the encoder: see
  `ggml-0011` above.) Whisper's
  exported driver used to hand the decoder the raw encoder output every step, so cross-attention K/V
  was re-projected over all 1500 encoder frames per token — `MUL_MAT 768x1500` ran 684 times for an
  11-second clip where the encoder itself needs 84, **57.7% of ASR runtime**, where onnxruntime
  computes those tensors once. Exporting the projected K/V as its own topology is worth **2.4-2.6x on
  every machine here** (on the Pi, 90.0 s to 37.4 s for the same clip).
* **Most of the remaining ASR gap is the ENCODER, and it is two kernels rather than a systemic
  deficit.** Split per shape at one thread with both engines' own profilers, interleaved round by round
  (2026-08-28, Ryzen; `scripts/whisper_encoder_split.py` re-derives it), loom's encoder is **1.20x**:

  | encoder piece | loom | onnxruntime | ratio | share of the gap |
  |---|---|---|---|---|
  | **`QK^T`** (12 calls) | 1957 ms | 1027 ms | **1.96x** | **51%** |
  | `Softmax` (12) | 780 | 491 | 1.60x | 16% |
  | `fc2` (12) | 2302 | 2016 | 1.14x | 16% |
  | `fc1` (12) | 2240 | 2034 | 1.09x | 11% |
  | Q/K/V/O (48) | 2138 | 1970 | 1.11x | 9% |
  | `A@V` (12) | 1101 | 1046 | **1.07x** | 3% |
  | GELU, LayerNorm + bias | 290 | 476 | **0.61x** | *loom ahead, −11%* |

  **The dense GEMMs are 1.09-1.14x each and 36% of the gap** — both true: per shape there is nothing in
  them to win, but they are 6.7 s of an 11.1 s encoder. `QK^T` alone is 6.0% of a whole transcription.
  *An earlier version of this table read the decoder as 1.50x FASTER and "nothing left there".* That was
  wrong: the 471 ms layout bucket under it was assigned to the encoder by eye, and 93% of it is the
  decode loop — which is why `$LOOM_PROFILE_NODES=1` now exists to attribute a bucket to a graph instead
  of to a guess. Three items on that list are now closed:
  * **GELU is done.** ggml's exact-erf GELU was a scalar `erff()` libm call per element with no SIMD
    path on any architecture; `cmake/patches/ggml-0010` replaces it with a rational approximation,
    **9.9x on the op in model** and 14.3-21.8x standalone. End to end on whisper that is **2.1% on the
    285K at four threads, 4.4% on the Ryzen at two, and 0.9% on the Pi** — a large kernel win that is a
    small share of a whole transcription, which is exactly what a 3.6%-of-runtime op can be.
  * **A GEMM cliff is done.** tinyBLAS rejected every matmul whose *contraction* was not a whole number
    of vectors (`k % KN`), handing the whole thing to ggml's generic kernel — and a contraction in
    attention is a sequence length or a head dimension, i.e. a number nothing rounds. `ggml-0011` splits
    it: **2.15x at whisper's `A@V` shape** — confirmed in the model, where `A@V` went 2.23x to **1.07x**
    against onnxruntime — and **1.19-1.24x end to end on Conformer-CTC**, whose head dimension
    (176/4 = 44) misses the vector width on every utterance. It also **regressed aarch64 by 1.3-1.75x
    for four days** by inlining its tail into the register tile; dispatching that out of line fixed it
    without moving x86 (see the note under the table).
  * **`SOFT_MAX` is measured out — but not for the reason first given.** Its five row passes really are
    three too many, and rewriting them is worth 1.08x on the machine that motivated it. The original
    reason, *"at 108 MB the op is DRAM-bandwidth bound"*, is false: against a `memcpy` of the same bytes
    ggml's row body is **3.6x on the Ryzen and 7.8x on the 285K**. It is closed on size instead —
    the exp is 7-26% of the op depending on the core, specialising `ggml_v_expf` to this domain is
    1.00-1.16x, and deleting the polynomial outright is only 1.8x
    ([Retro-012](docs/retros/retro-012-optimizations-that-were-measured-out.md)).
  * **The decoder's loop-invariant V transpose is done** (2026-08-29), and it is an EXPORT change, so
    the table above does not contain it. Whisper's decoder re-materialised the transpose of its
    cross-attention V every token in every layer — 12 nodes x 4.6 MB, 47% of the decode loop — of a
    tensor the cross-KV fix had already made constant for the utterance. The `cross_kv` phase now emits
    it transposed and the traced chain is deleted from the decoder topology: **1.106x end to end**, and
    the decoder's logits against HF are bit-identical. **The published GGUFs do not have it yet** — the
    ASR column above is the rc6 artifact, and the re-export lands in rc7.
  * What is left is `QK^T` at `k = 64` — **51% of the remaining encoder gap**, and closing it entirely
    would take the encoder from 1.20x to 1.10x.
* **Read the 285K's four-thread ASR cell as unstable.** onnxruntime's own time at that thread count
  ranges **1.08 s to 1.64 s across launches, a 1.5x spread**, because that part is 8 P-cores plus 16
  E-cores and a process is placed once and then stays there; loom's spread over the same launches is
  1.09x. The cell has read 0.71x, 0.59x and now 0.64x across three samplings in which the engines did
  not change. **The two onnxruntime builds are indistinguishable here** — nine paired launches of the
  conda-forge and PyPI 1.28.0 wheels straddle 1.0 at four threads and are 1.01x at 24 — so the 1.86x
  build difference that VITS shows does **not** carry to whisper.
* **loom used to stop scaling at 8 threads and go backwards; that is fixed, and the 24-thread row is
  the fix.** VITS on the 285K was 0.080 s at 8 threads and **0.191 s at 24 — the same as at one
  thread**. The cause was not in this repository: `ggml` defaults to OpenMP, so `ggml_barrier` was
  `#pragma omp barrier` — one after every non-empty graph node, 2520 of them in a synthesis — and
  libgomp's default wait policy **slept every thread on a futex at each one** (334,609 voluntary
  context switches per 5 syntheses, against 160 now). Building `ggml` against its own threadpool, whose
  spin is bounded, made the curve monotonic again and is worth **4.8x at 24 threads**
  ([Retro-017](docs/retros/retro-017-libgomp-slept-at-every-graph-node.md)).
* **The engine defaults to this machine's PHYSICAL CORE COUNT**, as of P4.30b. It used to default to
  `ggml`'s `GGML_DEFAULT_N_THREADS` — 4, whatever the machine has — which ran a 24-core workstation on
  four cores and said nothing about it; that is why two machines have two rows here. `$LOOM_N_THREADS`
  overrides it, and on Linux the count respects the process's CPU affinity, so `taskset` and a cgroup
  cpuset are honoured. Against the old default of 4 the 285K gains **1.98x on TTS, 2.41x on ASR and
  1.18x on the LM**, and the Ryzen gains **1.19x on TTS and 1.03x on the LM by going *down* to 2** —
  its two extra SMT siblings buy nothing on any task and cost on two, so "use every CPU" would have
  been right on one of these machines and a 1.19x TTS regression on the other. A Pi 4 does not move.
  **This does not move the published ratios much, because onnxruntime wants the same thing** — over the
  same 4 → 2 change on the Ryzen it gains 1.17x, 1.02x and 1.07x on the three tasks, so loom's ASR cell
  gets *worse* (0.80x → 0.67x at the time it was measured) purely because onnxruntime is the one that
  gains there. **The physical-core rule is a property of these CPUs, not of loom.**
  What the change costs is the case it was previously left at 4 for: every figure on this page is one
  inference at a time on an idle machine, and a host running several loom instances concurrently now
  has each of them claim every core. That host sets `$LOOM_N_THREADS`.

### What these numbers are not

**The onnxruntime build is named per row, because at the identical version it is worth up to 1.86x.**
The conda-forge build synthesises VITS in 0.065 s where the PyPI wheel takes 0.120 s — same machine,
same script — and against conda-forge the x86 TTS wins above become losses. The two 285K rows are
measured against conda-forge and the other three against the PyPI wheel, which is the channel `loom-py`
ships through. **That difference does not carry to whisper**: nine paired launches on the 285K put the
two builds within the thread-placement noise at four threads and at 1.01x at 24, so the ASR column is
comparable across rows even though its baselines are not the same package. The TTS column is not, and
the Pi and Ryzen TTS cells are the PyPI-wheel ones.

**Each pair is checked for equal work, not merely equal wall time.** TTS pins VITS's three scales so
both engines emit the same 73216 samples, and both harnesses print that count; ASR compares the
transcripts, which are identical; the LM runs the same prompt to the same token budget greedily on both
sides, and both emit the same tokens. **Model load is outside every timer on both sides.**

**Every cell is an average over separate process LAUNCHES, not over runs inside one.** The TTS and LM
cells are means — per engine, 31 TTS launches on x86 and 15 on the Pi; 21 LM launches on the 285K, 15
on the Ryzen and 7 on the Pi, where one generation is 47 s. The ASR cells are medians (9 per engine on
each 285K row, 7 on each Ryzen row, 5 cooled pairs on the Pi) and were taken 2026-08-29. **The Pi's TTS
and LM cells no longer report the fastest run.** That special case existed because thermal drift on
that board only ever makes a run slower; enforcing the cooldown before *every arm* rather than around
the session removes the drift instead of estimating around it, and the board then holds a 1.02-1.05x
spread per engine — tighter than either x86 machine.

**A median is used where the samples are unimodal and a mean where they are not, and the difference is
worth up to 1.4x.** At 24 threads on the 285K both engines' VITS time splits into two per-launch modes
about 1.4x apart at roughly even odds; a median over that lands on whichever mode drew the majority, so
the same eight minutes of benchmarking can report 1.19x or 1.66x with nothing changed. `paired_arms.py`
now prints both estimators and warns when an arm's launches split into two weighted clusters.

**Thread placement is chosen once per PROCESS on a hybrid part, it is worth up to 1.47x, and it is not
a lottery — it is WHICH CLUSTER.** The Core Ultra 9 285K is 8 P-cores (`/sys/devices/cpu_core/cpus` =
0-7) plus 16 E-cores (`cpu_atom` = 8-23), no SMT. A four-thread process lands on one cluster and then
stays there for its whole life, so every run inside it inherits that choice and a within-process median
cannot average it out. Pinned to a cluster, every number is reproducible to ~1%:

| VITS, 4 threads, 5 launches each | loom | onnxruntime | ratio |
|---|---:|---:|---:|
| P-cluster (`taskset -c 0-3`) | 0.0624-0.0628 s | 0.0646-0.0661 s | **1.034-1.053** |
| E-cluster (`taskset -c 8-11`) | 0.0916-0.0927 s | 0.1033-0.1075 s | **1.114-1.165** |

So the two clusters give *different ratios*, and a cell that does not say which one the process got is
under-specified. Pinning is how that table was measured and it is the right way to *explain* the
spread, but it is not how the cells above are taken: it constrains onnxruntime more than loom (pinned
to P-cores its whisper time is 1.76 s, worse than its lucky unpinned launches), so a pinned table would
not be like-for-like.

**One second of idle between the two arms on x86, and a cooldown to 60 C on the Pi, because otherwise
the second arm pays for the first.** Run
immediately after loom, onnxruntime landed in its slow VITS mode in 29 of 31 launches; run alone on the
same box in the same session, 10 of 20. A `sleep 1` between arms restored it to 21 of 31 and moved the
285K's 24-thread TTS cell from 1.41x to 1.20x. Alternating the arm order does not cancel this, because
both orders penalise whoever is second, and it is not symmetric between the engines — ggml's threadpool
spins where onnxruntime re-decides placement at session creation. The settle is worth ~1% on the
homogeneous Ryzen and up to 1.18x on the hybrid 285K, which is what says it is a placement effect.
[Retro-025](docs/retros/retro-025-the-arm-that-ran-second-paid-for-the-first.md) has the rest.

**Both sides use the same estimator**, which is not a detail: every harness here warms up and reports
a best-or-median over repeated runs in one process. `bench_lm_loom.cpp` used to time a single cold
generation instead, and comparing that against a warmed-up onnxruntime moved the LM column by 5-7% —
enough to flip its sign on all three machines. **`bench_asr_loom.cpp` had the same fault until
2026-08-29** and it was missed because the retro that fixed the LM harness asserted this one was
already warm rather than checking. Cold/warm on whisper is 1.25-1.7x at 24 threads on the 285K, 1.02x
at four, and *below* 1.0 on the Ryzen, where the box heats faster than the first run pays for itself —
a thread-count effect, which is why it landed on the one cell the table called an ASR win. The harness
now discards a warm-up run and prints it, and **`nrun` must be >= 3**: `times[size / 2]` on two samples
is the larger of the two, not a median.

**And the same two faults were still in the other half of the comparison until 2026-09-02.**
`bench_onnx_tasks.py`'s `vits` task had no warm-up while `bench_vits_loom.cpp` discards its first
synthesis, and `bench_vits_loom.cpp` carried the same `ts[size / 2]` index. Both are fixed. The audit
that fixed the ASR harness opened **loom's three harnesses** and asserted the onnxruntime side, which is
Retro-018's own lesson landing on its own blind spot; the check is one line and it has to name both
globs — `grep -n 'warm' scripts/bench_*_loom.cpp scripts/bench_onnx_tasks.py`. `bench_lm_loom.cpp` now
prints its first five tokens, so the LM's equal-work check is shown rather than asserted, the way VITS
prints a sample count and ASR prints a transcript.

**Reproducing it:** loom's side is `scripts/bench_{vits,lm,asr}_loom.cpp`, onnxruntime's is
`scripts/bench_onnx_tasks.py`. The latter drives `onnxruntime` directly rather than through `optimum`,
which added roughly 2x of its own overhead to whisper and mis-derives Qwen3's `head_dim`.

**The Pi throttles, and it will lie if allowed to.** It goes 55 -> 84 C during a single whisper run and
caps the ARM clock at 1580 MHz; two back-to-back measurements once came out 87.1 s and 115.8 s, 33%
apart. Its row is taken with a cooldown before every measurement and the two engines interleaved, so
both meet the same clock.

## Building

```sh
cmake -B build
cmake --build build -j"$(nproc)"
```

Dependencies (`ggml`, `nlohmann_json`, LuaJIT) are fetched by CMake; nothing else is needed to build
and run the hermetic suite. The fetched `ggml` is patched at configure time from `cmake/patches/` —
eleven diffs at present, fixing GCC's code generation for ggml's ARM F32 GEMM (1.6x), the matmuls it
declined to accept at all — by row count and by contraction length — a fused convolution that batched
its work too coarsely to stay in cache, a direct 1-D convolution for long activations with small
weights, the elementwise nodes a vocoder's resblock wraps around every convolution — its bias, its
leaky ReLU and its residual — none of which now costs a pass over memory of its own, an exact-erf GELU
that was a scalar libm call per element, and `conv_transpose_1d`'s single-threaded prologue and
dot-product-at-a-time compute; see `cmake/GgmlPatches.cmake` for the rules such a patch has to meet.
**`ggml-0011` is why `UPSTREAM.md` now requires a number from an x86 box AND one from the Pi before a
patch here is called done** — it was measured on one ISA, shipped to both, and regressed the other by
1.3-1.75x.

Two of them are heuristics tuned on measured hardware, so they carry a run-time escape:
`GGML_CPU_DISABLE_CONV_HEURISTICS=1` declines both, the way ggml's own `GGML_CPU_DISABLE_FUSION`
declines its fusions.

There is one build option of this repo's own, `-DLOOM_TINYBLAS=OFF`, which drops ggml's blocked GEMM
(`GGML_LLAMAFILE`) back out again. It exists to make GEMM measurements A/B-able and defaults **on**,
where it is worth ~2x on x86-64 and 1.6x on aarch64 at convolutional shapes ([Epic-05](docs/epics/epic-05-edge-performance.md)).

### Running on a GPU

A default build is CPU-only. Compiling a device backend in is a `ggml` option, passed straight through
— this repo adds no option of its own here, because there is nothing per-backend for it to decide:

```sh
cmake -B build -DGGML_VULKAN=ON     # or -DGGML_CUDA=ON, -DGGML_METAL=ON, -DGGML_SYCL=ON, ...
cmake --build build -j"$(nproc)"

build/tools/loom_cli/loom_cli --list-devices
build/tools/loom_cli/loom_cli --device gpu --model model.gguf --wav audio.wav
```

`--device` takes `auto` (the default, and what `$LOOM_DEVICE` sets), `cpu`, `gpu`, or a device name
such as `Vulkan0`. **`auto` prefers a device and falls back to the CPU; `gpu` is an error when there
is none**, because a caller who spelled it out is asking a question about the machine, and answering it
with a silent CPU run turns "no GPU here" into an unexplained performance number.

**The Vulkan build tools sort themselves out.** `glslc` and the Vulkan headers on a stable distribution
are likely too old for `ggml`'s Vulkan backend, and both fail in ways that name neither cause — so
`cmake/VulkanToolchain.cmake` probes for those two failures specifically and, when it finds them,
fetches pinned Vulkan-Headers and builds `glslc` from a pinned `shaderc` into the build directory. That
costs several minutes on the first configure of a machine that needed it, and nothing at all on one that
did not. `-DLOOM_VULKAN_FETCH_TOOLCHAIN=OFF` turns the diagnosis into an error naming what to install
instead, which is the right setting for an image that provides its own toolchain.

**A primitive can choose its own lowering.** Some ops are host callbacks through `ggml_map_custom` (a C
function pointer, so there is nothing for a GPU to dispatch) and others are real ggml ops a given backend
happens not to implement. Either way a primitive builds what the topology asked for, asks
`ggml_backend_supports_op`, and either keeps it or emits an equivalent — so the same GGUF lowers
differently per backend and the file on disk keeps saying what the model does. Every device run still
carries a CPU backend behind it for whatever is left. The CLI prints where each module actually ran;
`ctest -L gate -R device_parity` checks that a device gets the same answer as the CPU.

## Testing

Two classes of test, and a test's own directory is which class it is in.

```sh
ctest --test-dir build -L ci      # hermetic: builds its own fixtures, seconds, what CI runs
ctest --test-dir build -L gate    # real exported checkpoints; skips cleanly without them
```

`tests/ci/` needs nothing but this repo, a toolchain and `gguf`+`numpy` — every fixture it reads is
generated from `tests/fixtures/*.py` by a ctest step that runs first. `tests/gate/` compares against
real models: gigabytes that cannot live in git and hours that cannot run in CI, so each of those tests
exits 77 (Skipped) when its fixture is absent, and a developer with none of them still gets a green
suite that means *nothing hermetic broke*.

Point the gate suite at its fixtures with one variable:

```sh
export LOOM_FIXTURES=~/loom-fixtures
scripts/fixtures.py status    # what the gate tests want, and what you have
scripts/fixtures.py fetch     # from the published fixture repo
```

Every fixture also still honours the per-test variable it always had
(`LOOM_KOKORO_MIL_GGUF`, …), which wins over the root — pointing one test at one artifact you just
rebuilt is what you do while working on it.

## Documentation

The project's knowledge base is a four-tier hub-and-spoke set under [`docs/`](docs/), covering all
three repos.

| | |
|---|---|
| [`docs/backlog/active-index.md`](docs/backlog/active-index.md) | **the hub** — open work only, one line each, linked to its context |
| [`docs/epics/`](docs/epics/) | what each domain is and how it works (engine, exporter, models, backends, performance, host API, text front-ends, packaging) |
| [`docs/adrs/`](docs/adrs/) | why a technical choice was made, what was considered, and what it cost |
| [`docs/retros/`](docs/retros/) | what broke, the root cause, and the takeaway |

The specifications those tiers refer back to:

| | |
|---|---|
| [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md) | the data-driven design and why the engine hardcodes nothing |
| [`docs/KV-CACHE.md`](docs/KV-CACHE.md) | how a cached attention block reaches the engine from an export |
| [`docs/LOOM_PROCEDURAL_GENERALIZATION.md`](docs/LOOM_PROCEDURAL_GENERALIZATION.md) | the embedded-Lua orchestration blueprint |
| [`docs/HIGH-LEVEL-API.md`](docs/HIGH-LEVEL-API.md) | one door per task, what each layer owns, and what a file must declare for a host to dispatch |

[`BACKLOG.md`](BACKLOG.md) was the single-file ledger until it reached ~9,000 lines. It is now a
redirect carrying a map from every old section to its new home, so a code comment citing an item by
number (`P4.3e`, `P4.15b`) still resolves. Item numbers were not changed.

## Roadmap

**1. GPUs — done; NPUs, not yet.** The engine takes a device backend and a CPU fallback and schedules
across them ([Epic-04](docs/epics/epic-04-backends-and-accelerators.md)); see [Running on a GPU](#running-on-a-gpu) above. Measured on an AMD
Vega 3 iGPU against 4 CPU threads, one forward each:

| model | splits | GPU vs CPU |
|---|---|---|
| conformer-ctc-small | 1 | **2.56×** |
| lfm2-350m | 1 | **2.22×** |
| qwen3-0.6b | 1 | **2.82×** |
| matcha `encoder_mu` | 1 | **3.65×** |
| kokoro `decoder_vocoder` | 3 | **4.62×** |

What decides that number is how many times the scheduler has to cut the graph, and what forces a cut is
`ggml_map_custom` — a host callback, so there is nothing for a device to dispatch. Those splits used to
be 453, 181, 61, 107 and 5, which left Qwen3 at 0.95× and Matcha at 0.84× — *slower than the CPU*. None
of it was the engine: it was three patterns the exporter emitted as host callbacks because it had never
been taught to recognise them — an RMS norm (`POW`+`RSQRT`), a squaring (`POW`), and a hand-rolled
LayerNorm. **Across all thirteen exported models there are now exactly two `ggml_map_custom` nodes
left** — one `ATAN` each in Kokoro's and StyleTTS2's STFT phase, which has no ggml counterpart.
[Retro-009](docs/retros/retro-009-host-callback-count-was-the-wrong-lens.md) has the numbers, including a CPU measurement that came out wrong twice before
anything interleaved the runs.

**Counting host callbacks turned out to be the wrong lens**, which is worth knowing before optimizing
anything here: a graph splits just as readily on a *real* ggml op whose backend kernel is missing, and
the gaps do not line up between backends — CUDA has `PAD_REFLECT_1D` but no `POOL_1D`, Vulkan has
`POOL_2D` but neither, and the NPU backends have none of the three.

So **a primitive asks the backend what it can run** ([ADR-007](docs/adrs/adr-007-backend-capability-negotiation.md)) and emits either the native op
or an exactly-equivalent composition. That decision belongs in the engine rather than the export: one
GGUF may be run by any backend, so deciding it at export time compiles every artifact for the least
capable one. The same Kokoro file builds 1692 ggml nodes on a CPU and 1732 on Vulkan, and its topology
says `PAD_1D_REFLECT` either way.

`ATAN` had no exact composition anywhere — ggml has no inverse trigonometry in any backend — so it gets
the one **approximation** in the engine: range reduction, a degree-8 minimax polynomial and a branchless
reconstruction, measured at **1.81 ULP** and confined to backends that cannot dispatch the host callback,
so a CPU build still gets libm. **Eleven of the twelve models now run a whole module on the GPU with
nothing falling back at all.** The twelfth is Whisper, whose 400-wide reflect pad is cheaper to fall back
on than to compose — and which CUDA, Metal and SYCL run natively regardless.

One caveat on every speedup on this page: the GPU measured here reports `uma: 1`, so it shares memory
with the host and a split costs a synchronisation rather than a transfer. These numbers are a lower bound
on what a discrete card over PCIe would show.

Of the two decisions the earlier version of this item said were waiting on a GPU, one was answered and
one is still open. Retained inter-module outputs turn out not to be what a device charges for — measured
before the fusion above, LFM2's 20-module modular export cost 183 splits against the monolithic
export's 181, so decomposing a model into modules was never the expensive part. `FLASH_ATTENTION` is
still unbuilt: a GPU makes `ggml_flash_attn_ext`'s forced F16 K/V cast worth considering, but what
stands in the way is the gate suite's exact-fp32 comparisons, not the hardware.

**What is next is CUDA, then NPUs** — see [Epic-04](docs/epics/epic-04-backends-and-accelerators.md). Sixteen backend directories already
ship in the pinned ggml — CUDA, Metal, SYCL, OpenCL, HIP, OpenVINO, Hexagon (Qualcomm), CANN (Ascend)
among them — and because `loom::Device` resolves a spec against ggml's *device registry* rather than
against any backend name it knows, a CUDA build's `CUDA0` is already selectable by code that has never
heard of CUDA. Those cost a build matrix and a test run, not C++. CoreML (the Neural Engine, which
Metal is not) and RKNPU2 are out of tree and cost more, licence check included.

Compiling all of them in would end the leanness this engine is for, so the answer is `GGML_BACKEND_DL`:
each backend becomes a shared library ggml discovers at run time, one engine binary serves every
accelerator, and the deployment decides which files travel with it. That already works through
`loom::Device` unchanged. See [ADR-009](docs/adrs/adr-009-backends-as-dynamic-libraries.md) and the [backlog](docs/backlog/active-index.md#backends--accelerators) for what is still missing (a `Backends` that holds more than two, and
the custom-op fusion above, which on an NPU stops being an optimization and becomes a prerequisite).

**2. Builds for more platforms.** Linux x86-64 is what is built and tested today. Next: macOS on Intel,
macOS on Apple Silicon, and Linux on ARM — the last of which is the one that matters most for an engine
whose stated target is edge devices.

**3. More models — [Epic-03](docs/epics/epic-03-model-coverage.md)**, ordered by coverage per unit of effort: BERT token classifiers
(the smallest possible template, and the first non-audio task) → codec decoders → CNN+CTC and SANM
encoders → the remaining TTS families → text encoder-decoders → small classifiers → music. Each is an
export, so the measure of the design is that none of them should need engine work.

**4. The follow-ups the docs already name.** [`docs/backlog/active-index.md`](docs/backlog/active-index.md)
is the ledger and the authority; the ones worth knowing about from here are the `KvCache` memory redesign (deferred with its reasons),
KV-cache addressing policies beyond the contiguous append `ggml_set_rows` already permits, quantized KV
cache, a permissively-licensed phonemiser so the phoneme-input TTS models get a text door, and
generalizing the grapheme front-end out of C++ once a second such model exists.

## Licence

MIT — see [`LICENSE`](LICENSE).
