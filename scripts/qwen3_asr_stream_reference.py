#!/usr/bin/env python3
"""Regenerate the oracle for Qwen3-ASR's live stream (ADR-121).

Upstream's `Qwen3ASRModel.streaming_transcribe` (qwen-asr 0.0.6) runs on vLLM only. Its loop, replayed here
line for line with transformers' `generate` (greedy, f32, eager attention) in vLLM's place: every 2 s ALL the
audio so far is re-fed; from update 2 on, the previous raw answer (`language English<asr_text>...`) is
re-encoded, its last 5 ids dropped (more while the decoded prefix holds U+FFFD), decoded, and appended to the
prompt STRING; `finish_streaming_transcribe` re-feeds the tail the same way, keeping at least one id.

The driver does the same round trip through the file's vocabulary (`loom.tokenize` / `loom.detokenize`), from
the answer it holds as ids. This script runs that path too and REFUSES a clip where its text differs from
upstream's at any update: the fixture then holds the ids the driver must produce, and the text says they are
upstream's. Upstream's `finish` lacks the U+FFFD back-off its per-chunk path has; both paths here have it,
as the driver does (ADR-121).

Written per clip, float32 `.npy` (`tests/support/npy_fixture.h` reads no other dtype):

* `<clip>/samples.npy`: the audio each update decoded, in samples;
* `<clip>/hyp_<k>.npy`: the transcript ids after update k (after `<asr_text>`, the stop id not included);
* `<clip>/text_<k>.txt`: upstream's transcript text after it.

`--update-seconds` replays another cadence (upstream's `chunk_size_sec`). `--max-seconds` stops a clip early: past what a file's decoder cache holds, its stream is refused.

    ~/.venvs/ovos/bin/python scripts/qwen3_asr_stream_reference.py \\
        --model ~/Dev/models/qwen3-asr-0.6b-hf --out $LOOM_FIXTURES/qwen3_asr_stream_ref clips/*.npy

(transformers >= 5.13: the venv that exports Qwen3-ASR.) For oruk-loop-153M, its teacher's mode with the
language its export forces: `--model ~/Dev/models/oruk-loop-153M --language English
--tokenizer <its kept tokenizer>`. The gate is `tests/gate/test_e2e_qwen3_asr_stream.cpp`.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

SR = 16000
# `init_streaming_state`'s defaults.
CHUNK_S, UNFIXED_CHUNKS, UNFIXED_TOKENS = 2.0, 2, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=256, help="the driver's default budget")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="stop the updates here; 0 runs the whole clip")
    ap.add_argument("--update-seconds", type=float, default=CHUNK_S,
                    help="another cadence; written to `<clip>@<seconds>/`, which the gate opens the stream with")
    ap.add_argument("--language", help="upstream's `language=`: forced, appended to the prompt as "
                    "`language X<asr_text>`; oruk-loop's export forces English, so its fixture needs it")
    ap.add_argument("--tokenizer", help="the tokenizer the round trip re-encodes with AND the prompt's prefix is "
                    "encoded with, when the file's is not the checkpoint's: oruk-loop's export keeps 36,974 ids, and "
                    "its full tokenizer re-encodes some text into ids it dropped. Its kept tokenizer "
                    "(`oruk_loop_export._write_kept_tokenizer`) is the file's own, so oruk-loop's reference is its "
                    "teacher's loop over that")
    ap.add_argument("clips", nargs="+")
    args = ap.parse_args()

    kept_ids = None
    if "oruk_loop" in json.loads((Path(args.model) / "config.json").read_text()):
        # oruk-loop-153M: its own looped modeling code (`reference_forward_qwen3_asr_mil.py` loads it the same
        # way), and its export numbers ids by their index in the kept vocabulary.
        from transformers import AutoModel
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True, device="cpu",
                                          dtype=torch.float32, merge=False).eval()
        model.set_attn_implementation("eager")
        kept_ids = model.model.language_model.token_table.keep.numpy()
    else:
        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.float32, attn_implementation="eager").eval()
    proc = AutoProcessor.from_pretrained(args.model)
    tok = proc.tokenizer
    # `_build_text_prompt(context="", force_language=args.language)`.
    msgs = [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
    prompt_raw = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if args.language:
        prompt_raw += f"language {args.language}<asr_text>"
    marker = tok.convert_tokens_to_ids("<asr_text>")
    stops = set(model.generation_config.eos_token_id)

    def generate(audio, prefix_text="", prefix_ids=()):
        inputs = proc(text=[prompt_raw + prefix_text], audio=[audio], return_tensors="pt", padding=True)
        if prefix_ids:
            ids = torch.cat([inputs["input_ids"], torch.tensor([list(prefix_ids)], dtype=torch.long)], 1)
            inputs["input_ids"], inputs["attention_mask"] = ids, torch.ones_like(ids)
        inputs["input_features"] = inputs["input_features"].to(torch.float32)
        n = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
        gen = (out.sequences if hasattr(out, "sequences") else out)[0][n:].tolist()
        return [t for t in gen if t not in stops]

    if args.tokenizer:
        from transformers import AutoTokenizer
        kept_tok = AutoTokenizer.from_pretrained(args.tokenizer)

        def encode(text):
            return [int(kept_ids[i]) for i in kept_tok.encode(text)]
    else:
        encode = tok.encode

    def text_of(ids):
        # vLLM's detokenizer and the transformers backend's `batch_decode`: specials skipped, no clean-up.
        return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def upstream_prefix(raw_text, final):
        """`streaming_transcribe`'s prefix: re-encode, drop the last 5 ids, decode, back off while the text holds
        U+FFFD. `final` is `finish_streaming_transcribe`'s, at least one id kept, WITH the back-off its own code
        lacks (it put a literal U+FFFD into a Korean clip's prompt and transcript; the driver fixes that, so the
        reference does too -- the one place the two leave upstream, on purpose)."""
        cur = encode(raw_text)
        end = len(cur) - UNFIXED_TOKENS
        if final:
            end = max(1, end)
        end = min(end, len(cur))
        while end > 0:
            prefix = tok.decode(cur[:end])
            if "\ufffd" not in prefix:
                return prefix
            end -= 1
        return ""

    def transcript(text):
        return text.split("<asr_text>", 1)[1] if "<asr_text>" in text else text

    for clip in args.clips:
        name = Path(clip).stem
        wave = np.load(clip).astype(np.float32).reshape(-1)
        chunk = int(round(args.update_seconds * SR))
        ends = list(range(chunk, wave.shape[0] + 1, chunk))
        if wave.shape[0] % chunk:
            ends.append(wave.shape[0])
        if args.max_seconds > 0:
            ends = [e for e in ends if e <= args.max_seconds * SR]
        out = args.out / (name if args.update_seconds == CHUNK_S else f"{name}@{args.update_seconds:g}")
        out.mkdir(parents=True, exist_ok=True)
        raw_text, raw_ids = "", []
        for k, end in enumerate(ends):
            final = end == wave.shape[0] and end % chunk != 0
            audio = wave[:end]
            prefix_text, prefix_ids = "", []
            if k >= UNFIXED_CHUNKS:
                prefix_text = upstream_prefix(raw_text, final)
                # The driver's own path, from its own ids: the same rule over `decode(raw_ids)`, the prompt
                # holding the prefix text's ids.
                prefix_ids = encode(upstream_prefix(text_of(raw_ids), final))
            if args.tokenizer:
                # The prompt's prefix in the file's ids: oruk-loop's full tokenizer spells " Qtai" with " Qt", an
                # id its export dropped, so a prompt STRING is one the file cannot hold (long60 update 9 then
                # read "catatized" where the file's own ids read "cutatai's").
                gen = generate(audio, prefix_ids=encode(prefix_text) if prefix_text else ())
            else:
                gen = generate(audio, prefix_text=prefix_text)
            raw_text = prefix_text + text_of(gen)
            raw_ids = list(prefix_ids) + generate(audio, prefix_ids=prefix_ids)
            hyp = raw_ids[raw_ids.index(marker) + 1:] if marker in raw_ids else raw_ids
            want = transcript(raw_text)
            if text_of(hyp) != want:
                raise SystemExit(f"{name} update {k}: the driver's path gives {text_of(hyp)!r}, upstream's "
                                 f"{want!r} -- a clip this fixture cannot hold")
            if kept_ids is not None:
                hyp = [int(np.searchsorted(kept_ids, t)) for t in hyp]
            np.save(out / f"hyp_{k}.npy", np.asarray(hyp, np.float32))
            (out / f"text_{k}.txt").write_text(want)
        np.save(out / "samples.npy", np.asarray(ends, np.float32))
        print(f"{out.name}: {len(ends)} updates  {transcript(raw_text)!r}")


if __name__ == "__main__":
    main()
