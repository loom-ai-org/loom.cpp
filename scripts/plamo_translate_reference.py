#!/usr/bin/env python3
"""PLaMo 3 translate's own greedy translation, as the oracle `test_e2e_plamo` is held to.

    plamo_translate_reference.py OUT_DIR --text "..." --target-lang en [--source-lang ja] [--context "..."]
        [--glossary "src=tgt" ...] [--form polite|plain] [--history "src|||tgt" ...] [--max-new 128]

Writes one case directory:

    case.json          the arguments, the rendered prompt string, the decoded translation
    prompt_ids.npy     the chat template's ids (`apply_chat_template(..., add_generation_prompt=True)`)
    output_ids.npy     the greedy continuation, through the stop token
    logits_first.npy   the prefill's last-row logits (a tensor check before any token is drawn)
    segments.json      every string the PFML3 builder tokenizes, with its ids: the door's input pieces

Run in an environment with transformers 4.57.x and numba (the tokenizer's `njit`); the checkpoint's
own remote code is loaded with `trust_remote_code=True`. f32 on the CPU.
"""
import argparse
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--model", default=os.path.expanduser("~/Dev/models/plamo-3-translate-0.8b"))
ap.add_argument("--text", required=True)
ap.add_argument("--target-lang", required=True)
ap.add_argument("--source-lang")
ap.add_argument("--context")
ap.add_argument("--glossary", action="append", default=[])
ap.add_argument("--form")
ap.add_argument("--history", action="append", default=[])
ap.add_argument("--max-new", type=int, default=128)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.float32).eval()

messages = []
for pair in a.history:
    src, tgt = pair.split("|||")
    messages += [{"role": "user", "content": src}, {"role": "assistant", "content": tgt}]
messages.append({"role": "user", "content": a.text})
kwargs = dict(target_lang=a.target_lang)
for key in ("source_lang", "context", "form"):
    if getattr(a, key) is not None:
        kwargs[key] = getattr(a, key)
if a.glossary:
    kwargs["glossary"] = [dict(zip(("source", "target"), g.split("=", 1))) for g in a.glossary]

prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kwargs)
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt", **kwargs)
prompt_ids = ids["input_ids"]
with torch.no_grad():
    logits = model(input_ids=prompt_ids).logits[0, -1].numpy()
    out = model.generate(**ids, do_sample=False, max_new_tokens=a.max_new)
new = out[0, prompt_ids.shape[1]:].numpy()
np.save(os.path.join(a.out, "prompt_ids.npy"), prompt_ids[0].numpy().astype(np.float32))
np.save(os.path.join(a.out, "output_ids.npy"), new.astype(np.float32))
np.save(os.path.join(a.out, "logits_first.npy"), logits.astype(np.float32))

pieces = [a.text, a.target_lang] + [m["content"] for m in messages[:-1]]
pieces += [x for x in (a.source_lang, a.context, a.form) if x]
pieces += [s for g in a.glossary for s in g.split("=", 1)]
segments = {p: tok(p, add_special_tokens=False).input_ids for p in pieces}
json.dump(segments, open(os.path.join(a.out, "segments.json"), "w"), ensure_ascii=False, indent=0)
case = dict(vars(a), prompt=prompt, translation=tok.decode(new, skip_special_tokens=True))
json.dump(case, open(os.path.join(a.out, "case.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(case, ensure_ascii=False)[:400])
