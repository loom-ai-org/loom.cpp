"""MOSS-TTS-Local GGUF against the reference's own `generate` (ADR-051/052): greedy and pinned-sampled
codes, then free runs saved for the codec + ASR half.

    python scripts/moss_tts_oracle.py <checkpoint> <moss-tts.gguf> <out_dir>

Needs RAM for both (the F32 reference is 18 GB and the GGUF 16.8 GB), so it ran on the workstation.
The free runs' codes decode through the codec with `loom.Model(...).codes2speech.infer(codes)` (12-wide
rows are filled with the codec's absent id), and `scripts/asr_oracle.py` transcribes the result.

Recorded 2026-09-25: greedy 38/38 and 40/40 frames identical; pinned-sampled at the README's settings
34/34 and 60/60, stop frames included; Whisper 9/9 words in English and French at seeds 1 and 2.

**The pinned sampler.** The reference's `_sample_next_token` is replaced by one that builds the same
filtered distribution (temperature, then its own `_filter_logits`) and draws by walking it in
descending-probability order with a pinned uniform, which is the walk `loom.sample_row` does with
`uniform`. torch's `multinomial` walks index order, which maps the same number to a different id
(ADR-047).
"""

import sys, time, json
import numpy as np
import torch

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "loom-py"))
sys.path.insert(0, str(ROOT / "loom-exporter"))
import loom
from loom_exporter import moss_tts_export as mt
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

DIR, GGUF, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(16)
model, config = mt.load_reference(DIR)
tok = AutoTokenizer.from_pretrained(DIR)
Proc = get_class_from_dynamic_module("processing_moss_tts.MossTTSLocalProcessor", DIR)
proc = Proc(tokenizer=tok, audio_tokenizer=None, model_config=config)
t = time.time(); m = loom.Model.from_file(GGUF); print(f"loom load {time.time() - t:.1f}s", flush=True)
TEXTS = [("The quick brown fox jumps over the lazy dog.", "English", 3),
         ("Bonjour, je voudrais essayer une voix française naturelle.", "French", 9)]
report = {}


def ref_generate(text, lang, frames, **kw):
    batch = proc([[proc.build_user_message(text=text, language=lang)]], mode="generation")
    with torch.no_grad():
        out = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             max_new_tokens=frames, **kw)
    start_len, gen = out[0]
    return [r[1:].tolist() for r in gen[start_len + 1:] if int(r[0]) == config.audio_assistant_slot_token_id]


def compare(name, got, ref):
    same = sum(a == b for a, b in zip(got, ref))
    first = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), None)
    line = (f"{name}: loom {len(got)} frames, ref {len(ref)}; identical {same}/{max(len(got), len(ref))}"
            f"{'' if first is None else f'; first diff at frame {first}'}")
    print(line, flush=True)
    report[name] = dict(loom=len(got), ref=len(ref), identical=same, first_diff=first)


# 1. greedy, 40 frames
for text, lang, idx in TEXTS:
    ids = tok.encode(text, add_special_tokens=False)
    assert m.tokenize(text) == ids, "loom's tokenizer differs from the checkpoint's"
    ref = ref_generate(text, lang, 40, do_sample=False)
    t = time.time()
    got = m.text2codes.infer(tokens=ids, max_new_tokens=40, temperature=0.0, text_temperature=0.0,
                             language=float(idx))
    print(f"  loom greedy {len(got)} frames in {time.time() - t:.1f}s", flush=True)
    compare(f"greedy {lang}", got, ref)

# 2. sampled with pinned draws, the README's settings, 60 frames
state = {"u": None, "i": 0}


def pinned(self, logits, do_sample, temperature, top_k, top_p, previous_token_ids=None,
           repetition_penalty=1.0):
    scores = self._apply_repetition_penalty(logits.float(), previous_token_ids, repetition_penalty)
    scores = self._filter_logits(scores / float(temperature), top_k=top_k, top_p=top_p)
    probs = torch.softmax(scores, dim=-1)[0].double()
    order = torch.argsort(probs, descending=True, stable=True)
    kept = probs[order][probs[order] > 0]
    u = float(state["u"][state["i"]]); state["i"] += 1
    target, running = u * float(kept.sum()), 0.0
    for j, p in enumerate(kept.tolist()):
        running += p
        if running >= target:
            return order[j].view(1)
    return order[len(kept) - 1].view(1)


original = model._sample_next_token
model._sample_next_token = pinned.__get__(model)
rng = np.random.default_rng(11)
for text, lang, idx in TEXTS:
    u = rng.random(61 * (config.n_vq + 1)).astype(np.float32)
    state.update(u=u, i=0)
    ref = ref_generate(text, lang, 60, do_sample=True, audio_temperature=mt.AUDIO_TEMPERATURE,
                       audio_top_p=mt.AUDIO_TOP_P, audio_top_k=mt.AUDIO_TOP_K,
                       audio_repetition_penalty=1.0)
    got = m.text2codes.infer(tokens=tok.encode(text, add_special_tokens=False), max_new_tokens=60,
                             language=float(idx), draws=[float(x) for x in u])
    compare(f"pinned {lang}", got, ref)
model._sample_next_token = original

# 3. free runs to EOS at the checkpoint's own settings, for the codec + ASR oracle downstream
for text, lang, idx in TEXTS:
    for seed in (1, 2):
        t = time.time()
        got = m.text2codes.infer(text, language=float(idx), seed=seed)
        dt = time.time() - t
        np.save(f"{OUT}/loom_{lang}_seed{seed}.npy", np.array(got, dtype=np.int64))
        print(f"free {lang} seed {seed}: {len(got)} frames ({len(got) / 12.5:.2f} s) in {dt:.1f}s",
              flush=True)
        report[f"free {lang} {seed}"] = dict(frames=len(got), seconds=dt)
json.dump(report, open(f"{OUT}/verify_report.json", "w"), indent=1)
print("VERIFY-DONE")
