#!/usr/bin/env python3
"""Regenerate the Soprano TTS oracle for family 9's eighth leaf.

Soprano (`ekwek/Soprano-1.1-80M`) is a 17-layer Qwen3 LM over 8000 audio ids. Each generation step's
FINAL-NORM hidden row -- not the id it draws -- is what a Vocos decoder (linear x4 upsample, ConvNeXt,
ISTFT head) turns into 2048 samples of 32 kHz audio. The text goes through the reference's English
normaliser and is split into sentences first; each sentence is one generation.

**This runs the reference's own `SopranoTTS.infer`**, transformers backend, f32 on CPU. Its sentences
are generated together as ONE left-padded batch, where loom generates them one at a time; the pad is
`[STOP]`, which every prompt already opens with, so the batch is the same arithmetic and the gate
holds the two to it. The reference samples (temperature 0.001, top-p 0.95, top-k 50), which picks the
argmax unless two ids are within ~1e-3 of each other; `torch.manual_seed` is set for reproducibility.

Writes, all float32 `.npy` (ids too, since `tests/support/npy_fixture.h` reads one dtype):

* `tokens`      every sentence's prompt ids, concatenated -- `[STOP][TEXT]{sentence}[START]` each
* `ids_<i>`     sentence i's drawn ids, `[STOP]` included when it stopped
* `hidden_<i>`  `[n_rows, 512]`, the rows the decoder read for sentence i
* `wave`        `[n_samples]`, `infer`'s waveform: every sentence's audio, joined

and `meta.json` with the text, the sentences and their row counts.

    ~/.venvs/piper/bin/python scripts/soprano_reference.py \\
        --model ~/Dev/models/soprano-1.1-80m --repo ~/Dev/soprano --out $LOOM_FIXTURES/soprano_ref
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

TEXT = ("Dr. Smith paid $15,000.50 on May 3, 2026 at 8:05, and nobody objected. "
        "The meeting itself ran long, far longer than anyone had planned! "
        "Was it worth it in the end? I really think so.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo", required=True, help="a checkout of github.com/ekwek1/soprano")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    from soprano.backends.transformers import TransformersModel
    from soprano.tts import SopranoTTS
    from soprano.vocos.decoder import SopranoDecoder

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    tts = SopranoTTS.__new__(SopranoTTS)
    tts.pipeline = TransformersModel(device="cpu", model_path=args.model)
    tts.device, tts.backend = "cpu", "transformers"
    tts.decoder = SopranoDecoder()
    tts.decoder.load_state_dict(torch.load(f"{args.model}/decoder.pth", map_location="cpu"))
    tts.decoder.eval()
    tts.decoder_batch_size, tts.RECEPTIVE_FIELD, tts.TOKEN_SIZE = 1, 4, 2048

    sentences = tts._preprocess_text([args.text])
    tok = tts.pipeline.tokenizer
    tokens = []
    for prompt, _, _ in sentences:
        tokens += tok(prompt)["input_ids"]
    np.save(out / "tokens.npy", np.asarray(tokens, dtype=np.float32))

    # The hidden rows and drawn ids, per sentence, from the backend `infer` itself calls; then the
    # waveform from `infer_batch`'s own decode-and-join, run on those rows.
    model = tts.pipeline.model
    captured = {}
    original = model.generate

    def generate(*a, **kw):
        result = original(*a, **kw)
        captured["sequences"] = result.sequences
        captured["n_prompt"] = kw["input_ids"].shape[1]
        return result

    model.generate = generate
    torch.manual_seed(args.seed)
    wave = tts.infer(args.text)
    model.generate = original
    np.save(out / "wave.npy", wave.numpy().astype(np.float32))

    torch.manual_seed(args.seed)
    responses = tts.pipeline.infer([p for p, _, _ in sentences], top_p=0.95, temperature=0.0,
                                   repetition_penalty=1.2)
    rows = []
    for i, r in enumerate(responses):
        hidden = r["hidden_state"].float()
        np.save(out / f"hidden_{i}.npy", hidden.numpy())
        ids = captured["sequences"][i, captured["n_prompt"]:].tolist()
        eos = model.config.eos_token_id
        ids = ids[: ids.index(eos) + 1] if eos in ids else ids
        np.save(out / f"ids_{i}.npy", np.asarray(ids, dtype=np.float32))
        rows.append(int(hidden.shape[0]))
    meta = {"text": args.text, "sentences": [p for p, _, _ in sentences], "rows": rows,
            "samples": int(wave.shape[0]), "seed": args.seed}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
