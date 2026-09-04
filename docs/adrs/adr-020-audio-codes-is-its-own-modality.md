---
type: adr
status: accepted
date: 2026-09-03
tags: [model-contract, api-design, host-bindings, model-coverage]
---

# ADR-020: Codec Tokens Are Their Own Modality, Not `token_ids`

## Context

Family 11 — neural audio codec decoders (DAC, SNAC, WavTokenizer, EnCodec, Mimi) — takes discrete
codes in and returns a waveform. The task name has been reserved since P4.0.4:
`tasks.py` declares `audio-codec` with `reserved=True` and the note *"discrete-token-in/waveform-out,
which is a different contract from text-to-speech and is what family 10's AR codec-token models decode
through. Claimed by P5's family 11."*

Claiming it means declaring a modality pair, and the obvious pair is wrong.

`loom.input.kind = "token_ids"` reads correctly — codec tokens *are* integer ids — but
`ModelContract::interface_side` folds `token_ids` onto `"text"`, deliberately and for a good reason:
supplying ids rather than a string is how you talk to a model that cannot encode, **not a different
contract**, and a host that offered `Tokens2Speech` beside `Text2Speech` would be naming one door
twice ([ADR-013](adr-013-one-door-per-task.md)).

That reasoning does not extend here. A codec decoder declaring `("token_ids", "audio")` would resolve
to **`text2speech`**, so loom-py would offer `Text2Speech` on it and `model.capabilities` would report
a TTS model. A caller would then reasonably pass `text=`, which the door would try to encode through a
vocabulary the file does not have. This is the *"hosts guess what a model is"* defect the declared
contract exists to remove ([HIGH-LEVEL-API §1](../HIGH-LEVEL-API.md)), arriving from the other
direction: not a host guessing, but the file declaring something untrue.

The distinction is real and not cosmetic. `token_ids` and `phoneme_ids` fold onto `text` because
**text is what they encode** — a phoneme id is a way of writing a sound in a word, and a caller with
the string could get the ids. A codec token encodes *audio*: it is a compressed acoustic frame, there
is no string it came from, and no caller can produce one except by running an encoder or an AR LM.

## Options Considered

1. **`("token_ids", "audio")`.** Misroutes to `text2speech`, as above.
2. **Fold nothing: make `interface_side` return `"tokens"` for `token_ids`.** Fixes this case by
   breaking the one the fold exists for — every phoneme-input TTS model would stop being
   `text2speech`, which is four shipped models and their published cards.
3. **Reuse `Text2Speech.infer(tokens=...)` as the codec door.** It already accepts raw ids and would
   need no new interface. Rejected: it makes `capabilities` lie about what the model is, and the
   `text=` and `phonemes=` arms of that door are unreachable for a codec — a door where two of three
   arguments raise is a door in the wrong place.
4. **A new modality.**

## Decision

**`audio_codes` is a modality of its own**, and `interface_side` maps it to `"codes"`.

```
loom.task         = "audio-codec"
loom.input.kind   = "audio_codes"
loom.output.kind  = "audio"
                  -> interface_name() == "codes2speech"
```

`modality::AUDIO_CODES` joins the open string set in `model_contract.h`, and loom-py's taxonomy gains
`Codes2Speech`. The fold for `token_ids`/`phoneme_ids` is untouched, so no shipped model's interface
name moves.

**What the file must also declare**, because a caller cannot construct the input without it and
nothing else in the artifact states it:

```
loom.codec.n_codebooks    u32    how many code streams per frame
loom.codec.codebook_size  u32    the valid id range per stream
loom.codec.frame_rate     f32    codes per second, so a caller can size a clip
loom.sample_rate          u32    the existing key, read under its own name
```

These are `hparams()` by the split that method's docstring already draws: a number the HOST needs in
order to build an input or interpret an output belongs in the file, and a number the DRIVER needs is an
`ExportConstants` value. `n_codebooks` is the host's — it is the width of the matrix a caller passes.

**What is NOT declared here is the delay pattern.** An AR LM emits codebook *k* offset by *k* steps
(MusicGen's convention, which Parler and Dia inherit); undoing that is index arithmetic over a small
array, and it is a property of the **LM**, not of the codec — DAC knows nothing about it. It belongs
in the family-10 driver's Lua by the §2 rule, and putting it in the codec's contract would make every
codec carry a fact that only some of its callers have.

## Consequences

* **Positive: a codec cannot be mistaken for a TTS model**, by a host or by a person reading
  `model.capabilities`. The failure it prevents is silent — `Text2Speech.infer("hello")` on a codec
  would raise inside a tokenizer lookup, naming nothing useful.
* **Positive: family 10 gets a name for its own output.** An AR codec-token LM is `text2codes`, which
  is a real interface rather than a `_Planned` placeholder, and it composes: `text2codes` then
  `codes2speech`. Whether family 10 ships as one GGUF or two is a separate decision this does not
  foreclose.
* **Negative: one more entry in a set that is meant to be small.** The modality list is deliberately
  short, and every addition is a claim that a genuinely new kind of thing exists. The argument that it
  does is the one above — a codec token has no string behind it — and if that argument is wrong, this
  is a door named twice.
* **Negative: `interface_side`'s fold is now asymmetric** — two id modalities fold and one does not,
  which is exactly the kind of rule that gets "tidied" by someone who has not read this. The test in
  `test_model_contract.cpp` names all three cases so the fold cannot be made uniform without a red
  test.

## See Also

* [ADR-013](adr-013-one-door-per-task.md) — one door per task, declared by the file; the fold this
  amends and the reasoning it preserves
* [Epic-03](../epics/epic-03-model-coverage.md) — families 10 and 11 and why they are two halves of
  one pipeline
* [HIGH-LEVEL-API §3](../HIGH-LEVEL-API.md) — the tier-0 key list this adds to
