# Handover: family 10 (Dia) and proving the DAC composition

**Temporary. Delete it when family 10 lands** — the same way P4.18's handover was deleted when that
work closed. Nothing here is the source of truth: it is a *sequence*, plus pointers to the tiers that
are. If this document and the hub disagree, the hub is right.

Durable knowledge lives where the protocol puts it:

| | |
|---|---|
| open work | [`backlog/active-index.md`](backlog/active-index.md) — family 10, family 11's remaining leaves |
| the families and what they cost | [Epic-03 §2](epics/epic-03-model-coverage.md) |
| why codec tokens are their own modality | [ADR-020](adrs/adr-020-audio-codes-is-its-own-modality.md) |
| the tracing rule every family has needed | [ADR-019](adrs/adr-019-family-12-needs-no-attention-mask.md) |
| one door per task | [ADR-013](adrs/adr-013-one-door-per-task.md) |

---

## 1. Branches, PRs, and the submodule

One branch name in all three repos: **`feat/p5-families-11-and-12`**.

| repo | PR | head commit at handover |
|---|---|---|
| loom.cpp | [#21](https://github.com/loom-ai-org/loom.cpp/pull/21) | `5211d39` |
| loom-exporter | [#17](https://github.com/loom-ai-org/loom-exporter/pull/17) | `f1131ae` |
| loom-py | [#19](https://github.com/loom-ai-org/loom-py/pull/19) | `e04142a` |

**The branch is misnamed for what it now carries** (families 12 *and* 11, and family 10 will land on
top). Renaming it again would close the PRs a second time — GitHub's branch-rename API deletes the old
ref and does **not** retarget open PRs, which is how #20/#16/#18 died. Either live with the name or
open fresh PRs deliberately; do not use the rename API expecting it to move them.

**loom-py's submodule is one commit behind** (`vendor/loom.cpp` at `158086a`, loom.cpp at `5211d39`).
The gap is docs-only, so it builds — bump it with the next real change rather than on its own.

The bump, which is needed after every loom.cpp change and before every release:

```sh
git -C vendor/loom.cpp fetch origin feat/p5-families-11-and-12
git -C vendor/loom.cpp checkout <sha>
cmake --build build -j8          # proves it builds FROM the submodule, not from stray copies
git add vendor/loom.cpp && git commit      # same commit as the code that needs it
```

The bump and the binding change are **one commit**: `src/binding.cpp` includes headers that only exist
at the new sha, so a split commit does not build.

**Merge order is loom.cpp first.** loom-py records a submodule pointer at a branch commit; if
loom.cpp's PR is squashed, that sha stops existing on `main` and loom-py must be re-bumped to the
squashed one before its own PR merges.

---

## 2. Environment — the things that bite in the first ten minutes

* **Use `~/.venvs/piper`.** `python3` resolves to `~/.venvs/ovos`, which is the Qwen3-ASR-only
  environment and fails four tests for environment reasons alone. Never upgrade piper: NeMo pins
  `transformers~=4.53` and it currently holds 4.57.6, torch 2.8.0, coremltools 9.0.
* **`Dev/models` is a symlink to an external drive** — `/media/flavio/Samsung_T5`, which runs at
  **100%, single-digit GB free**. Two consequences: a download fails with `No space left on device`
  while `df /home` reports tens of GB free, and `du -sh Dev/models` returns **0** because du does not
  follow symlinks. Check `df -h /media/flavio/Samsung_T5` before fetching a checkpoint, and write
  exported GGUFs to `/home` (the scratch dir), never beside the model.
* **Always `TMPDIR=/home/flavio/.claude/tmp`.** The exporter's pytest suite writes tens of GB of real
  exports; `/tmp` is small and filling it kills unrelated things.
* Checkpoints: `models/dia-1.6b` is now the **transformers-native** `nari-labs/Dia-1.6B-0626`
  (`DiaForConditionalGeneration`, 1.61B, sharded) — it replaced nari-labs' original release in place.
  `models/dac-44khz` is the codec Dia decodes through.
* `parler_tts` **cannot** be installed into piper. It pins `transformers==4.46.1`, and `--no-deps`
  does not terminate: it hard-imports `dac.model`, whose package hard-imports `audiotools`, whose 22
  runtime deps include `protobuf<3.20` — which breaks coremltools. Verified and fully reverted.

---

## 3. What is done, and what "done" means here

**Family 12 — token classification.** Two structurally different encoders, both verified against
`transformers` **on the tensor**, 138 tokens: `bert-base-NER` max |Δ| 1.24e-05, `distilbert-NER`
5.72e-06, cosine 0.99999988, 138/138 argmax. Sabotage arms (same graph, different sentence's
reference) 11.94 and 9.54.

**Family 11 — DAC.** Verified on the **waveform** against reference DAC on real speech: max |Δ|
1.85e-04 at 2 s, 2.22e-04 at 5 s, cosine ~1.0, sabotage arm 1.14. Card built, in the model-card gate.
`CodecFamily` is an enum carrying loader/decode/geometry, with an `ENCODEC` member that **raises**
naming its two blockers rather than failing inside coremltools.

Test state at handover: loom.cpp `ctest -L ci` **78/78**, loom-exporter `pytest tests/ci` **629**,
loom-py `pytest tests/ci` **92**, model-card gate green on the staging tree.

---

## 4. Dia: what is known, and what is only assumed

Dia is family 10's composition target because **its own `audio_tokenizer_config.json` names
`descript/dac_44khz`** — the codec already exported and verified — so it costs the LM half only.
MusicGen was the earlier pick and was dropped because it would have dragged EnCodec in, which has two
blockers of its own (see the hub).

Shape, from `config.json`:

* encoder: 12 layers, hidden 1024, **byte-level vocab 256**, head_dim 128
* decoder: 18 layers, hidden 2048, **9 channels**, vocab 1028, head_dim 128, cross-attn to the encoder
* `delay_pattern: [0, 8, 9, 10, 11, 12, 13, 14, 15]` — **declared in the file**, so it is read rather
  than derived, and belongs in the driver by ADR-020's reasoning
* generation defaults: `temperature 1.8, top_k 50, top_p 0.9, do_sample true`, eos 1024, bos 1026,
  pad 1025, `max_length 3072`

**It does not convert without one patch, and the diagnosis is the reusable part.**
`modeling_dia.rotate_half` slices at `x.shape[-1] // 2`, which traces to **48 ×
`aten::Int(aten::floor_divide(...))`** in the encoder alone; coremltools' `_int` handler does
`int(x.val)` and dies with `TypeError: only 0-dimensional arrays can be converted to Python scalars`
at node `encoder/0/self_attention/128`.

It fails **at a static text length too**, and under both `sdpa` and `eager`. So — unlike every other
blocker on this thread — it is *not* a dynamic-axis artefact. Do not spend time bucketing the text
axis over it.

The fix, which makes the 12-layer encoder convert with a fully symbolic axis (`(1, is0, 1024)`) using
only ops already in the dialect:

```python
def rotate_half_static(x):
    half = x.shape[-1] // 2                 # static per module; see below for why not a config read
    if not isinstance(half, int):
        raise TypeError("last dim is not static; the per-module midpoint assumption does not hold")
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
modeling_dia.rotate_half = rotate_half_static
```

**Verified bit-identical** to the original at lengths 7, 32 and 128 — `max|patched - original| = 0` at
every one. To reproduce: run the encoder twice over the same random byte ids, once with
`modeling_dia.rotate_half` left alone and once with it replaced, and diff the hidden states.

Two notes on the *form*, because the first draft got both wrong and they are the kind of thing that
ships silently:

1. **The midpoint is derived per module, not from a config read.** An earlier version closed over
   `config.encoder_config.head_dim // 2`. Dia happens to use `head_dim` 128 for the encoder, the
   decoder's self-attention *and* its cross-attention (`cross_head_dim: 128`), so one constant is
   right in all three **by luck of this checkpoint** — a model whose halves differed would be silently
   wrong in one of them, and an encoder-only probe cannot catch that.
2. **It raises rather than falling back** if the last dim is ever not static. A `//` on a symbolic dim
   is how this whole failure started; reintroducing it silently would be worse than the original bug.

Nothing has been exported. There is no Dia GGUF, no family module, no driver — only the finding above,
and it lives in the hub.

---

## 5. The sequence

Each step names the check that closes it. **A step is not done because it ran.**

**0. Done** — the patch is verified bit-identical and in its per-module form (§4). Start at step 1.

**1. The encoder phase.** Export encoder-only through the real path (not a `ct.convert` probe) and
verify the hidden states against `transformers` on the tensor. Byte-level input, so the "tokenizer" is
a byte vocabulary — check whether `byte_vocab` covers it before writing anything.

**2. The decoder phase — the actual new work.** 18 layers, cross-attention to the encoder output,
KV-cached, **nine output heads** at vocab 1028. The precedent is Whisper: `multi_phase_export` plus
`PrefillDecodeLoop.bound`, which is the cross-attention decode loop family 2 already built. What is
genuinely new is the 9-wide head — every existing decode loop reduces **one** row to **one** token,
and this one must emit nine per step. Expect that to need a new epilogue component, the way each of
the last three families needed exactly one.

**3. The delay pattern, in the driver.** Read `loom.delay_pattern` from the file (declare it in
`contract()`/`hparams()`; it is a per-checkpoint fact). Emission offsets channel *k* by
`delay_pattern[k]`; decoding must realign before the codes reach DAC. It is index arithmetic over a
small array — Lua, not C++, by the ADR-013 §2 rule. **The engine must not learn what a delay pattern
is.**

**4. The composition.** Decide deliberately whether Dia ships as one GGUF with DAC merged (loom's
"the model is one file" property, ~6.6 GB) or two files chained by the host. Either way the
end-to-end check is the same and is the point of the whole exercise:

> text → Dia → 9 delayed code streams → realign → DAC → waveform,
> against `transformers` running the identical pipeline.

**Grade it with the ASR oracle, not correlation.** Kokoro matched PyTorch at cosine 0.996 and shipped
unintelligible ([Retro-006](retros/retro-006-kokoro-shipped-noise.md)); Dia produces speech, so
transcribe the output and compare the words. Fix the text in the model card so the expectation is a
constant, exactly as the TTS rows do with "hello world".

**5. Ship it.** Catalogue entry in `build_model_cards.py` (a `text-to-codes`-ish snippet — note that
task is currently *reserved* in `tasks.py` and family 10 is what claims it), a row in the export
sweep, and an arm in loom-py's model-card gate that asks the *is it right* question for this family.

---

## 6. Traps this thread has already paid for

* **An export that runs is not an export that works.** DAC's first working version produced correct
  audio and returned one frame's worth of it for **every** input, because the shape walk gave up on a
  rank-reducing slice and returned a literal `1`. Nothing raised. The only thing that catches this
  class is asserting on the **emitted shapes**, not on the call — and exporting the same checkpoint at
  **two trace lengths** and requiring an identical topology.
* **A fallback that returns a number for an unresolved dynamic dim is a silent-wrong-answer
  generator.** Returning `None` would have raised. If you touch `_infer_dynamic_dim_expr`, prefer
  giving up loudly.
* **Tensor oracle, not token oracle.** A wrong encoder still decodes a plausible transcript.
* **A gate that cannot fail proves nothing.** Every check added on this thread was sabotaged and
  confirmed red before being trusted. Do the same.
* **Verify a licence, do not assume it.** `descript/dac_44khz` publishes no `license:` tag at all; MIT
  came from the upstream project's LICENSE and its explicit statement about the *weights*.
* **`build_model_cards.py` snippets are Python**, so they may contain braces — `render_snippet`
  substitutes named placeholders rather than `str.format`, which used to crash on `{n_codebooks}`.
