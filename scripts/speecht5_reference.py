#!/usr/bin/env python3
"""Regenerate the SpeechT5 oracle for family 9b's first leaf.

SpeechT5 (`microsoft/speecht5_tts`) is a text encoder and a KV-cached transformer decoder that emits
MEL FRAMES autoregressively, `reduction_factor` (2) per step, until a stop head says so; a
convolutional postnet refines the whole spectrogram once, and a separate HiFi-GAN
(`microsoft/speecht5_hifigan`) turns it into 16 kHz audio. The voice is a 512-d x-vector.

**The loop is stochastic at inference, on purpose.** The decoder prenet applies dropout (p = 0.5) in
eval mode too (`SpeechT5SpeechDecoderPrenet.forward`: "Dropout is always applied, even when
evaluating", Tacotron 2 §2.2), and the reference draws a fresh `torch.bernoulli` mask for EVERY row of
the output sequence at every step, of which only the last row reaches the decoder. So the only thing
that makes a run reproducible is that last row's two masks per step, and this script pins them:
`_consistent_dropout` is replaced by one that reads `masks[step, layer]` for the last row (the other
rows are computed and discarded by the reference, so what they hold does not matter).

Writes, all float32 `.npy` (ids too, since `tests/support/npy_fixture.h` reads one dtype):

* `tokens`       the text's ids, `</s>` appended (`SpeechT5Tokenizer`)
* `speaker`      `[512]`, the x-vector, NOT normalised (the model normalises it)
* `masks`        `[n_steps, 2, 256]` of {0, 1}, the pinned prenet dropout masks, one pair per step
* `encoder`      `[n_tokens, 768]`, the encoder's last hidden state
* `spectra`      `[n_steps, 2, 80]`, `feat_out` per step (the frames before the postnet)
* `stop_logits`  `[n_steps, 2]`, `prob_out` per step, before the sigmoid
* `mel`          `[2 * n_steps, 80]`, the spectrogram after the postnet
* `wave`         `[n_samples]`, HiFi-GAN's output

    ~/.venvs/piper/bin/python scripts/speecht5_reference.py \
        --model ~/Dev/models/speecht5-tts --out $LOOM_FIXTURES/speecht5_ref

`--f64` adds `*_f64.npy`: the whole loop re-run at float64 with the same tokens, voice and masks. Every
frame feeds the next step, so f32 rounding compounds, and the f64 arm says how far two correct f32
implementations can drift apart (`chatterbox_reference.py`'s lesson). The f64 run may stop at a
different step; `meta.json` records both.

The vocoder is read from `<model>/hifigan` and the voice from `<model>/xvectors/spkrec-xvect.zip`
(`Matthijs/cmu-arctic-xvectors`), sorted index 7306 by default -- the utterance every published
SpeechT5 example uses (`cmu_us_slt_arctic-wav-arctic_a0508`).
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

DEFAULT_TEXT = "Hello, world! This is a test of the speech synthesis system."


def save(out: Path, name: str, value) -> None:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    np.save(out / f"{name}.npy", np.ascontiguousarray(array, dtype=np.float32))


def read_xvector(model_dir: Path, index: int):
    with zipfile.ZipFile(model_dir / "xvectors" / "spkrec-xvect.zip") as z:
        names = sorted(n for n in z.namelist() if n.endswith(".npy"))
        return names[index], np.load(io.BytesIO(z.read(names[index]))).astype(np.float32)


def run(model, vocoder, ids, speaker, masks, dtype):
    """One pinned generation at `dtype`; returns every recorded tensor."""
    from transformers.models.speecht5 import modeling_speecht5 as m

    prenet_cls = m.SpeechT5SpeechDecoderPrenet
    original = prenet_cls._consistent_dropout
    calls = {"n": 0}

    def pinned(self, inputs_embeds, p):
        step, layer = divmod(calls["n"], 2)
        calls["n"] += 1
        if step >= masks.shape[0]:
            raise RuntimeError(f"the loop ran past the {masks.shape[0]} pinned steps")
        mask = torch.zeros_like(inputs_embeds[0])
        mask[-1] = torch.from_numpy(masks[step, layer]).to(inputs_embeds.dtype)
        all_masks = mask.unsqueeze(0).repeat(inputs_embeds.size(0), 1, 1)
        return torch.where(all_masks == 1, inputs_embeds, 0) * 1 / (1 - p)

    spectra, logits = [], []
    postnet = model.speech_decoder_postnet
    hooks = [postnet.feat_out.register_forward_hook(lambda mod, a, out: spectra.append(out.detach())),
             postnet.prob_out.register_forward_hook(lambda mod, a, out: logits.append(out.detach()))]
    encoder = {}

    def keep_encoder(mod, args, out):
        # Returns None: a forward hook's non-None return value REPLACES the module's output.
        encoder.setdefault("h", out.last_hidden_state.detach())

    hooks.append(model.speecht5.encoder.register_forward_hook(keep_encoder))
    prenet_cls._consistent_dropout = pinned
    try:
        model.to(dtype)
        vocoder.to(dtype)
        with torch.no_grad():
            mel = model.generate_speech(torch.tensor([ids]), torch.from_numpy(speaker).to(dtype)[None])
            wave = vocoder(mel)
    finally:
        prenet_cls._consistent_dropout = original
        for h in hooks:
            h.remove()
    n = len(spectra)
    return dict(encoder=encoder["h"][0], spectra=torch.cat(spectra).view(n, 2, -1),
                stop_logits=torch.cat(logits).view(n, 2), mel=mel, wave=wave, n_steps=n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the microsoft/speecht5_tts directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--speaker-index", type=int, default=7306)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--f64", action="store_true", help="also re-run the whole loop at float64")
    args = ap.parse_args()

    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Tokenizer

    model_dir, out = Path(args.model), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = SpeechT5Tokenizer.from_pretrained(model_dir)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_dir).eval()
    vocoder = SpeechT5HifiGan.from_pretrained(model_dir / "hifigan").eval()
    ids = tok(args.text)["input_ids"]
    speaker_name, speaker = read_xvector(model_dir, args.speaker_index)

    # Enough pinned steps for the longest run the reference allows: `maxlenratio` 20 over
    # `reduction_factor` 2 is ten steps per token.
    max_steps = len(ids) * 10 + 1
    rng = np.random.default_rng(args.seed)
    masks = (rng.random((max_steps, 2, model.config.speech_decoder_prenet_units)) < 0.5).astype(np.float32)

    ref = run(model, vocoder, ids, speaker, masks, torch.float32)
    n = ref["n_steps"]
    save(out, "tokens", np.array(ids, dtype=np.float32))
    save(out, "speaker", speaker)
    save(out, "masks", masks[:n])
    for name in ("encoder", "spectra", "stop_logits", "mel", "wave"):
        save(out, name, ref[name])
    meta = dict(text=args.text, speaker=speaker_name, seed=args.seed, n_tokens=len(ids), n_steps=n,
                n_frames=int(ref["mel"].shape[0]), n_samples=int(ref["wave"].shape[0]),
                sample_rate=vocoder.config.sampling_rate)
    print(f"f32: {len(ids)} tokens -> {n} steps -> {meta['n_frames']} frames -> {meta['n_samples']} samples")

    if args.f64:
        f64 = run(model, vocoder, ids, speaker, masks, torch.float64)
        for name in ("encoder", "spectra", "stop_logits", "mel", "wave"):
            save(out, f"{name}_f64", f64[name])
        meta["n_steps_f64"] = f64["n_steps"]
        k = min(n, f64["n_steps"])
        print(f"f64: {f64['n_steps']} steps; encoder max|d| "
              f"{(ref['encoder'].double() - f64['encoder']).abs().max():.3e}, spectra[:{k}] max|d| "
              f"{(ref['spectra'][:k].double() - f64['spectra'][:k]).abs().max():.3e}")
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
