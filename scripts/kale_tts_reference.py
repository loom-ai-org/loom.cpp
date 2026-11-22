#!/usr/bin/env python3
"""kale-tts-1-tiny (loom-exporter `kale_tts_export.py`, ADR-142) by upstream's own code, as the oracle
`test_e2e_kale_tts` is held to.

    kale_tts_reference.py OUT_DIR "<text>" [--voice kenzie] [--seed 0] [--model ~/Dev/models/kale-tts-1-tiny]
        [--teacher ~/Dev/models/kale-tts-1-base] [--ettin ~/Dev/models/ettin-encoder-17m]
    kale_tts_reference.py OUT_DIR --frontend TEXTS.jsonl      ({"text", "lang"} lines)

The text goes through upstream's front end as its own runtime runs it (`js/kale-tts.mjs` `KaleTTS.inputs`, in
node), and each sentence's inputs through upstream's modules: Ettin (transformers' ModernBERT), K20Text
(`recipe/textside/textside.py`'s `_mods`, compiled out of that file), the teacher's predictor and text encoder
(kokoro's `KModel` with kale-tts-1-base's weights and the published duration and pitch heads), the release's
per-voice end clamp and unvoicing floor, and the decoder (`recipe/train/model.py` `Dec`) through
`recipe/train/onnx_export.py`'s `HeadX`, the head upstream's ONNX holds. Sentence i's noise is
`torch.randn(samples, generator=manual_seed(seed + i))`, which is what the driver draws (`aten_randn`).

Writes one case directory:

    case.json      the arguments, the voice's row, the sentences, per-sentence sample counts
    tokens.npy     the kale_tts tokenizer's packed records for the text (the driver's `tokens`)
    noise.npy      every sentence's noise draw, end to end
    audio.npy      the sentences' waveforms, each followed by upstream's 0.12 s gap (`KaleTTS.speak`)

With --frontend, writes OUT_DIR/frontend.jsonl instead: each text's packed records, sentences and IPA.
"""
import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("text", nargs="?")
ap.add_argument("--frontend")
ap.add_argument("--model", default=os.path.expanduser("~/Dev/models/kale-tts-1-tiny"))
ap.add_argument("--teacher", default=os.path.expanduser("~/Dev/models/kale-tts-1-base"))
ap.add_argument("--ettin", default=os.path.expanduser("~/Dev/models/ettin-encoder-17m"))
ap.add_argument("--voice", default="kenzie")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
MD = Path(a.model)

# ---------------------------------------------------------------------------------------- the front end (node)
JS = r"""
import { readFileSync } from "node:fs";
const [, , dir, inp] = process.argv;
const { KaleTTS } = await import(dir + "/js/kale-tts.mjs");
const tts = await KaleTTS.prepare(dir + "/");
for (const line of readFileSync(inp, "utf8").split("\n")) {
  if (!line.trim()) continue;
  const { text, voice } = JSON.parse(line);
  const v = await tts._loadVoice(voice);
  const sentences = v.fe.sentences(text), ipa = sentences.map((s) => v.fe.ipa(s));
  const tokens = [0], feeds = []; let n = 0;
  for await (const { inputs } of tts.inputs(text, { voice })) {
    const ids = Array.from(inputs.input_ids.data, Number).slice(1, -1), widx = Array.from(inputs.widx.data, Number).slice(1, -1);
    const [, W, S] = inputs.wavg.dims, wavg = inputs.wavg.data, sub = Array.from(inputs.ettin_ids.data, Number);
    const subWord = sub.map((_, k) => { let best = -1, bv = 0; for (let w = 0; w < W; w++) if (wavg[w * S + k] > bv) { bv = wavg[w * S + k]; best = w; } return best; });
    tokens.push(ids.length, ...ids, ...widx, W, S, ...sub, ...subWord); n++;
    const f = {};
    for (const [k, t] of Object.entries(inputs)) f[k] = { dims: t.dims, data: Array.from(t.data, Number) };
    feeds.push(f);
  }
  tokens[0] = n;
  process.stdout.write(JSON.stringify({ tokens, sentences, ipa, feeds }) + "\n");
}
"""


def front_end(rows):
    """[{text, voice}] -> upstream's front end on each: packed tokens, sentences, IPA and every record's feeds."""
    with tempfile.TemporaryDirectory() as d:
        script, inp = Path(d) / "fe.mjs", Path(d) / "in.jsonl"
        script.write_text(JS)
        inp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        out = subprocess.run(["node", str(script), str(MD), str(inp)], check=True, capture_output=True, text=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
if a.frontend:
    voice_of = {"en-GB": "amy"}
    rows = [json.loads(l) for l in open(a.frontend) if l.strip()]
    res = front_end([{"text": r["text"], "voice": voice_of.get(r.get("lang", ""), "kenzie")} for r in rows])
    with open(out / "frontend.jsonl", "w") as f:
        for r, x in zip(rows, res):
            f.write(json.dumps({"text": r["text"], "lang": r.get("lang", ""), "tokens": x["tokens"],
                                "sentences": x["sentences"], "ipa": x["ipa"]}, ensure_ascii=False) + "\n")
    print(f"{out / 'frontend.jsonl'}: {len(rows)} texts")
    sys.exit(0)

# ---------------------------------------------------------------------------------------- the modules
TRAIN = MD / "recipe" / "train"
sys.path[:0] = [str(TRAIN / "third_party" / "paradee"), str(TRAIN)]
import model as M            # noqa: E402,F401  (recipe/train/model.py: Dec)
import onnx_export as X      # noqa: E402       (strip, HeadX)
from huggingface_hub import hf_hub_download   # noqa: E402
from kokoro import KModel    # noqa: E402
from transformers import AutoModel            # noqa: E402

torch.set_num_threads(1)
load = lambda p: torch.load(p, map_location="cpu", weights_only=False)  # noqa: E731
km = KModel(repo_id="hexgrad/Kokoro-82M", config=hf_hub_download("hexgrad/Kokoro-82M", "config.json",
                                                                 revision="f3ff3571791e39611d31c381e3a41a3af07b4987"), model=False)
sd = load(Path(a.teacher) / "kale-tts-1-base.pth")
for key in ("predictor", "text_encoder"):   # kline.load_kmodel's rename
    ck = {}
    for k, v in sd[key].items():
        k = k[7:] if k.startswith("module.") else k
        ck[k.replace(".parametrizations.weight.original0", ".weight_g").replace(".parametrizations.weight.original1", ".weight_v")] = v
    res = getattr(km, key).load_state_dict(ck, strict=False)
    assert not [m for m in res.missing_keys if not m.endswith((".norm.weight", ".norm.bias"))] and not res.unexpected_keys
dur = load(MD / "pytorch/tiny/duration_head_fld.pt")
km.predictor.lstm.load_state_dict(dur["lstm"]); km.predictor.duration_proj.load_state_dict(dur["duration_proj"])
pitch = load(MD / "pytorch/tiny/pitch_heads_flp.pt")
for m in ("shared", "F0", "N", "F0_proj", "N_proj"):
    getattr(km.predictor, m).load_state_dict(pitch[m])
km.eval(); X.strip(km)
src = (MD / "recipe/textside/textside.py").read_text()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_mods")
scope = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "textside.py", "exec"), scope)
st = scope["_mods"]()[1]().eval()
st.load_state_dict(load(MD / "pytorch/tiny/ettin_text_side_fle.pt")["model"])
dec_state = load(MD / "pytorch/tiny/decoder_flkd.pt")["model"]
dec = M.Dec("A", 512, 16, 160, 4).eval()
dec.load_state_dict({k[4:]: v for k, v in dec_state.items() if k.startswith("dec.")})
X.strip(dec)
head = X.HeadX(dec.head)
et = AutoModel.from_pretrained(a.ettin, attn_implementation="eager").eval()

vj = json.loads((MD / "voices.json").read_text())
voice = next(v for v in vj["voices"] if v["name"].lower() == a.voice.lower() or v["id"] == a.voice)
vindex = vj["voices"].index(voice)


def clamp(F0, tail, thr, floor):
    """The release graph's end clamp with its per-voice inputs (textside.clamp_onnx after ettin_voice_clamp.py and
    add_voice_tail.py): the voice's floor in place of the sentence's."""
    f = F0[0]; v = f > thr; cnt = v.sum(); ar = torch.arange(f.shape[0])
    last = torch.where(v, ar, torch.zeros_like(ar)).max(); t = (ar >= last - tail) & (ar <= last)
    return torch.where(v & t & (cnt >= 5), torch.maximum(f, torch.tensor(floor)), f)[None]


@torch.no_grad()
def sentence(fd, seed):
    T_ = lambda k, dt=torch.long: torch.tensor(np.array(fd[k]["data"]).reshape(fd[k]["dims"]), dtype=dt)  # noqa: E731
    p = km.predictor
    ids = T_("input_ids"); style = T_("style", torch.float32); widx = T_("widx")
    hs = et(input_ids=T_("ettin_ids"), attention_mask=T_("ettin_mask"), output_hidden_states=True).hidden_states
    h = (hs[-1] + hs[-2]) / 2
    d_en = st(ids, torch.zeros_like(ids, dtype=torch.bool), T_("wavg", torch.float32) @ h, widx).transpose(-1, -2)
    s = style[:, 128:]; T = ids.shape[1]; sx = s[:, :, None].expand(-1, -1, T); x = torch.cat([d_en, sx], 1)
    for b in p.text_encoder.lstms:
        if isinstance(b, nn.LSTM): x = b(x.transpose(-1, -2))[0].transpose(-1, -2)
        else: x = torch.cat([b(x.transpose(-1, -2), s).transpose(-1, -2), sx], 1)
    d = x.transpose(-1, -2)
    n = torch.clamp(torch.round(torch.sigmoid(p.duration_proj(p.lstm(d)[0])).sum(-1)[0]), min=1).long()
    end = torch.cumsum(n, 0); start = end - n; f = torch.arange(end[-1])[None]
    aln = ((f >= start[:, None]) & (f < end[:, None])).float()[None]
    F0, N = p.F0Ntrain(d.transpose(-1, -2) @ aln, s)
    one = lambda k: float(fd[k]["data"][0])  # noqa: E731
    F0 = clamp(F0, one("voice_clamp_tail"), one("voice_clamp_thr"), one("voice_pitch_floor"))
    F0 = torch.where(F0 < one("voice_unvoice_hz"), torch.zeros_like(F0), F0)
    e = km.text_encoder; hh = e.embedding(ids).transpose(1, 2)
    for c in e.cnn: hh = c(hh)
    t_en = e.lstm(hh.transpose(1, 2))[0].transpose(-1, -2)
    xt = dec.trunk(t_en @ aln, F0, N, T_("voice_sound", torch.float32))
    noise = torch.randn(xt.shape[2] * 300, generator=torch.Generator().manual_seed(seed))
    return head(xt, F0, noise[None])[0].numpy(), noise.numpy()


fe = front_end([{"text": a.text, "voice": voice["id"]}])[0]
waves, noises = [], []
for i, fd in enumerate(fe["feeds"]):
    w, nz = sentence(fd, a.seed + i)
    waves.append(w.astype(np.float32)); noises.append(nz.astype(np.float32))
gap = np.zeros(int(round(0.12 * 24000)), np.float32)
audio = []
for w in waves:   # `speak`: every sentence, then the gap
    audio += [w, gap]
np.save(out / "tokens.npy", np.array(fe["tokens"], np.float32))
np.save(out / "noise.npy", np.concatenate(noises))
np.save(out / "audio.npy", np.concatenate(audio))
(out / "case.json").write_text(json.dumps({
    "text": a.text, "voice": voice["name"].lower(), "voice_index": vindex, "seed": a.seed, "speed": 1.0,
    "sentences": fe["sentences"], "samples": [int(w.size) for w in waves]}, indent=1, ensure_ascii=False))
print(f"{out}: {len(waves)} record(s), {sum(w.size for w in waves)} samples, voice {voice['name']} (row {vindex})")
