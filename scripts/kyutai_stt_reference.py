#!/usr/bin/env python3
"""Regenerate the Kyutai STT oracle: Kyutai's own `moshi` inference, `run_inference`'s STT path.

Kyutai STT (`kyutai/stt-1b-en_fr`, the moshi-format release) streams 24 kHz audio through the Mimi
encoder, 32 codes per 80 ms frame, into a 16-layer LM that emits one text id per frame. **This runs
`moshi`, not the transformers port**: the port re-encodes the first frame and windows its LM at 375
positions where this checkpoint's `context` is 750 (loom-exporter `kyutai_stt_export.py` has the
account). f32 on CPU, greedy, as the checkpoint's `lm_gen_config` (temperature 0) asks.

`run_inference` steps the FIRST frame's codes twice, because step 0 consumes the initial tokens
whatever it is handed; this does the same. Ids 0 (`<unk>`) and 3 (`<pad>`) are what it drops when it
prints; the transcript here is the remaining pieces joined.

Writes (float32 `.npy`, since `tests/support/npy_fixture.h` reads one dtype; ids and codes too):

* `pcm`          the 24 kHz signal moshi encoded, its trailing 1.5 s of silence included
* `codes`        `[32, n_frames]`, Mimi's codes, streamed one frame at a time
* `text_ids`     `[n_frames]`, the text id of steps 1..n (step 0's is consumed, never printed)
* `step_logits`  `[n_frames + 1, 8000]`, every step's text logits, step 0 included
* `meta.json`    the transcript, the frame count, the input file

    PYTHONPATH=~/Dev/moshi/moshi ~/.venvs/piper/bin/python scripts/kyutai_stt_reference.py \\
        --model ~/Dev/models/kyutai-stt-1b-en-fr --wav samples/jfk.wav --out $LOOM_FIXTURES/kyutai_stt_ref
"""
import argparse
import json
from math import gcd
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from moshi.models import loaders
    from moshi.models.lm import LMGen

    d = Path(args.model)
    cfg = json.loads((d / "config.json").read_text())
    info = loaders.CheckpointInfo.from_hf_repo(
        "kyutai/stt-1b-en_fr", moshi_weights=d / "model.safetensors", mimi_weights=d / cfg["mimi_name"],
        tokenizer=d / cfg["tokenizer_name"], config_path=d / "config.json")
    mimi = info.get_mimi(device="cpu")
    lm = info.get_moshi(device="cpu", dtype=torch.float32)
    sp = info.get_text_tokenizer()
    step_logits = []
    gen = LMGen(lm, use_sampling=False,
                on_text_logits_hook=lambda l: step_logits.append(l[0, 0, 0].float().numpy().copy()),
                **{k: v for k, v in info.lm_gen_config.items() if k != "use_sampling"})

    sr, x = wavfile.read(args.wav)
    x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 24000:
        g = gcd(sr, 24000)
        x = resample_poly(x, 24000 // g, sr // g).astype(np.float32)
    pcm = torch.from_numpy(x).view(1, 1, -1)
    stt = info.stt_config
    pad_left = int(stt.get("audio_silence_prefix_seconds", 0.0) * 24000)
    pad_right = int((stt.get("audio_delay_seconds", 0.0) + 1.0) * 24000)
    pcm = torch.nn.functional.pad(pcm, (pad_left, pad_right))
    frame = int(mimi.sample_rate / mimi.frame_rate)
    chunks = [c for c in pcm.split(frame, dim=2) if c.shape[-1] == frame]

    codes, text_ids = [], []
    mimi.streaming_forever(1)
    gen.streaming_forever(1)
    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            c = mimi.encode(chunk)
            codes.append(c[0, :, 0].numpy())
            if i == 0:
                gen.step(c)
            tokens = gen.step(c)
            if tokens is not None:
                text_ids.append(int(tokens[0, 0, 0]))
    text = "".join(sp.id_to_piece(t).replace("▁", " ") for t in text_ids if t not in (0, 3)).strip()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pcm.npy", pcm[0, 0].numpy().astype(np.float32))
    np.save(out / "codes.npy", np.stack(codes, 1).astype(np.float32))
    np.save(out / "text_ids.npy", np.asarray(text_ids, dtype=np.float32))
    np.save(out / "step_logits.npy", np.stack(step_logits).astype(np.float32))
    meta = {"text": text, "n_frames": len(chunks), "wav": Path(args.wav).name, "pad_left": pad_left,
            "pad_right": pad_right}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
