#!/usr/bin/env python3
"""Regenerate the LFM2.5-Audio speech-to-text oracle: liquid-audio's own `generate_sequential`.

LFM2.5-Audio-1.5B transcribes with the README's fixed system prompt, "Perform ASR.": `ChatState`
builds the system turn, the user turn around the audio rows (NeMo log-mel -> FastConformer -> adapter,
one row per 80 ms), and the assistant turn's opening; `generate_sequential` then draws text greedily
until `<|im_end|>`. Everything here is f32 on CPU -- `ChatState(dtype=float32)`, where the README's default
casts the mel to bf16.

liquid-audio needs Python >= 3.12 (`~/.venvs/liquid`, Python 3.13, CPU torch 2.8, transformers 4.57).

Writes, all float32 (ids too, since `tests/support/npy_fixture.h` reads one dtype): `text.npy` (the
prompt's text ids), `modality_flag.npy`, `mel.npy` `[128, frames]`, `in_emb.npy` (the prefill
embeddings, audio rows included), `step_logits.npy` (every drawn id's text logits), `ids.npy` (the
drawn ids, `<|im_end|>` last), `meta.json` (the transcript).

    ~/.venvs/liquid/bin/python scripts/lfm25_audio_reference.py \
        --model ~/Dev/models/lfm2.5-audio-1.5b --wav samples/jfk.wav --out $LOOM_FIXTURES/lfm25_audio_ref
"""
import json, os, sys
import numpy as np
import torch
import soundfile as sf
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality

from pathlib import Path
import argparse
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True)
ap.add_argument("--wav", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
M, wav, out = Path(args.model), args.wav, args.out
os.makedirs(out, exist_ok=True)
processor = LFM2AudioProcessor.from_pretrained(M, device="cpu").eval()
model = LFM2AudioModel.from_pretrained(M, dtype=torch.float32, device="cpu").eval()

chat = ChatState(processor, dtype=torch.float32)
chat.new_turn("system")
chat.add_text("Perform ASR.")
chat.end_turn()
chat.new_turn("user")
x, sr = sf.read(wav, dtype="float32")
if x.ndim > 1:
    x = x.mean(1)
chat.add_audio(torch.from_numpy(x).unsqueeze(0), sr)
chat.end_turn()
chat.new_turn("assistant")

captured = {}
orig_prefill = model._prefill
def prefill(**kw):
    e = orig_prefill(**kw)
    captured["in_emb"] = e.detach().clone()
    return e
model._prefill = prefill
logits = []
orig_sample = model._sample_text_token
def sample(text_logits, **kw):
    logits.append(text_logits.detach().float().clone())
    return orig_sample(text_logits, **kw)
model._sample_text_token = sample

ids = []
with torch.no_grad():
    for t in model.generate_sequential(**chat, max_new_tokens=512):
        if t.numel() == 1:
            ids.append(int(t))
text = processor.text.decode(torch.tensor(ids))
print("prompt", tuple(chat.text.shape), "flags", tuple(chat.modality_flag.shape), "mel", tuple(chat.audio_in.shape))
print("ids", len(ids), "text:", text)
np.save(f"{out}/text.npy", chat.text[0].numpy().astype(np.float32))
np.save(f"{out}/modality_flag.npy", chat.modality_flag[0].numpy().astype(np.float32))
np.save(f"{out}/mel.npy", chat.audio_in.float().numpy())
np.save(f"{out}/in_emb.npy", captured["in_emb"][0].float().numpy())
np.save(f"{out}/step_logits.npy", torch.stack(logits).numpy())
np.save(f"{out}/ids.npy", np.asarray(ids, dtype=np.float32))
np.save(f"{out}/wave16k.npy", x.astype(np.float32) if sr == 16000 else np.zeros(0, np.float32))
json.dump({"text": text, "sr": sr, "audio_in_flag": int(LFMModality.AUDIO_IN)}, open(f"{out}/meta.json", "w"), indent=1)
