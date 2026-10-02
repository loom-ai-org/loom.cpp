#!/usr/bin/env python3
"""Regenerate the LFM2.5-Audio text-to-speech oracle: liquid-audio's own `generate_sequential`.

A "Perform TTS. Use the ... voice." system prompt, the text as the user turn, and the assistant turn's
opening; the model emits `<|audio_start|>`, then one frame of 8 Mimi-codebook codes per step (its last
hidden row through a depthformer), until a frame opens with end-of-audio. The frames (end-of-audio
dropped, as the README does) go through the LFM2-based detokenizer to 24 kHz. f32 on CPU. GREEDY audio
by default (`--temperature` omitted): the README samples at 0.8 / top-k 64, and a sampled run cannot be
compared draw for draw across two random streams.

`processor.decode` builds its detokenizer with `.cuda()`, so this builds it the way it does -- the
config's `sliding_attention` layers loaded as `full_attention`, the window being the detokenizer's own
mask -- on the CPU.

Writes, all float32: `prompt.npy` (the whole prompt's ids), `voice_prompt.npy`, `text_ids.npy` (the
pieces a driver is handed), `codes.npy` `[8, T]`, `wave.npy`, `hidden.npy`, `depth_logits.npy`,
`text_logits.npy`, and `meta.json` (the steps, text and audio, in order) and `ref.wav`.

    ~/.venvs/liquid/bin/python scripts/lfm25_audio_tts_reference.py \
        --model ~/Dev/models/lfm2.5-audio-1.5b --out $LOOM_FIXTURES/lfm25_audio_tts_ref
"""
import json, os, sys
from pathlib import Path
import numpy as np
import torch
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
from liquid_audio.detokenizer import LFM2AudioDetokenizer
from transformers import Lfm2Config

import argparse
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--voice", default="Perform TTS. Use the US female voice.")
ap.add_argument("--text", default="Hello there. This is a test of the speech synthesis model, running on loom.")
ap.add_argument("--temperature", type=float, default=None, help="audio codes; omitted is greedy")
ap.add_argument("--top-k", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
M, out, VOICE, TEXT = Path(args.model), args.out, args.voice, args.text
temp, top_k, seed = args.temperature, args.top_k, args.seed
os.makedirs(out, exist_ok=True)
processor = LFM2AudioProcessor.from_pretrained(M, device="cpu").eval()
model = LFM2AudioModel.from_pretrained(M, dtype=torch.float32, device="cpu").eval()

chat = ChatState(processor, dtype=torch.float32)
chat.new_turn("system"); chat.add_text(VOICE); chat.end_turn()
chat.new_turn("user"); chat.add_text(TEXT); chat.end_turn()
chat.new_turn("assistant")

hiddens, depth_logits, text_logits = [], [], []
orig_frame = model._sample_audio_frame
def frame(embedding, **kw):
    hiddens.append(embedding.detach().clone())
    return orig_frame(embedding, **kw)
model._sample_audio_frame = frame
orig_text = model._sample_text_token
def text_tok(logits, **kw):
    text_logits.append(logits.detach().clone())
    return orig_text(logits, **kw)
model._sample_text_token = text_tok
orig_get = [e.get_logits for e in model.depth_embeddings]
for i, e in enumerate(model.depth_embeddings):
    def gl(x, _f=orig_get[i]):
        y = _f(x); depth_logits.append(y.detach().clone()); return y
    e.get_logits = gl

torch.manual_seed(seed)
seq, frames = [], []
with torch.no_grad():
    for t in model.generate_sequential(**chat, max_new_tokens=1024, audio_temperature=temp, audio_top_k=top_k):
        if t.numel() == 1:
            seq.append(("text", int(t)))
        else:
            seq.append(("audio", t.tolist())); frames.append(t.clone())
print("prompt ids", chat.text.shape[1], "steps", len(seq), "frames", len(frames),
      "text ids", [v for k, v in seq if k == "text"])

cfg = Lfm2Config.from_pretrained(str(M / "audio_detokenizer" / "config.json"))
# `LFM2AudioProcessor.audio_detokenizer`: the window is the detokenizer's own mask, so its layers load as full.
cfg.layer_types = ["full_attention" if t == "sliding_attention" else t for t in cfg.layer_types]
detok = LFM2AudioDetokenizer(cfg).eval()
from safetensors.torch import load_file
detok.load_state_dict(load_file(str(M / "audio_detokenizer" / "model.safetensors")))
codes = torch.stack(frames[:-1], 1).unsqueeze(0)          # drop the end-of-audio frame, as the README does
with torch.no_grad():
    wave = detok(codes)[0]
print("codes", tuple(codes.shape), "wave", tuple(wave.shape))
np.save(f"{out}/prompt.npy", chat.text[0].numpy().astype(np.float32))
tok = processor.text
np.save(f"{out}/voice_prompt.npy", np.asarray(tok.encode(VOICE, add_special_tokens=False), np.float32))
np.save(f"{out}/text_ids.npy", np.asarray(tok.encode(TEXT, add_special_tokens=False), np.float32))
np.save(f"{out}/codes.npy", codes[0].numpy().astype(np.float32))          # [8, T]
np.save(f"{out}/wave.npy", wave.numpy().astype(np.float32))
np.save(f"{out}/hidden.npy", torch.stack(hiddens).numpy())
np.save(f"{out}/depth_logits.npy", torch.stack(depth_logits).numpy())
np.save(f"{out}/text_logits.npy", torch.stack(text_logits).numpy())
json.dump({"voice": VOICE, "text": TEXT, "seq": seq, "temperature": temp, "top_k": top_k, "seed": seed},
          open(f"{out}/meta.json", "w"))
import soundfile as sf
sf.write(f"{out}/ref.wav", wave.numpy(), 24000)
