#!/usr/bin/env python3
"""Regenerate the oracle for CosyVoice3's live stream (ADR-112).

Upstream's `CosyVoice3Model.tts(stream=True)` does not carry the flow's state from hop to hop; it re-runs
it. Each hop of speech tokens (25, plus whatever puts the prompt and the first hop on a 25-token
boundary, then 50, then 100) it runs `flow.inference(streaming=True)` over the voice's prompt and the
tokens so far, plus the flow's 3 look-ahead tokens, keeps the mel past what it had, runs CausalHiFT over
ALL of the mel with `finalize=False` and hands out the samples it had not. This replays that loop over
the LM's tokens. The timing does not change the slices, so the LM runs first, and its draws are pinned
the way `cosyvoice3_reference.py` pins them (that script's helpers are imported, not copied).

It also runs the two one passes the stream is compared with: `streaming=True` (the DiT's block-causal
chunk mask, what `infer` runs with `streaming = 1`) and `streaming=False` (full attention, `infer`'s
default). Written out, all float32 `.npy` (`tests/support/npy_fixture.h` reads no other dtype):

* `text_ids`, `draws` (two uniforms per LM step), `noise` (the flow's construction-time noise table,
  frame-major), `nsf_noise` (the sine source's, harmonic-major): every draw, pinned.
* `mel_streaming`, `mel_stream_loop`, `mel_full`: the generated frames' mel, frame-major, from the
  streaming one pass, upstream's stream loop and the full-attention one pass.

    ~/.venvs/piper/bin/python scripts/cosyvoice3_stream_reference.py \\
        --model ~/Dev/models/fun-cosyvoice3-0.5b-2512 --out $LOOM_FIXTURES/cosyvoice3_stream_ref

The gate is `tests/gate/test_e2e_cosyvoice3_stream.cpp`.

**`--f64` writes nothing; it prints how far upstream's loop is from its streaming one pass with the flow
at float64.** That distance is 0 (each hop ends on a chunk boundary, the position conv and the chunk
mask are causal, the noise is one table sliced from the start), so the loop IS that one pass, and the
4.7e-3 the f32 run shows is the ten guided Euler steps amplifying the re-run's different rounding. The
vocoder stays f32 there: CausalHiFT's sine phase turns any mel difference into O(1) sample differences,
which is why the gate compares mels and checks the vocoder on its own.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cosyvoice3_reference as R  # noqa: E402

# The sentence the gate was built on: long enough for four hops (223 tokens kept, 446 frames).
DEFAULT_TEXT = ("The history of the city of Rome begins on the Palatine hill, where shepherds built their "
                "first huts beside the river, long before the first temples rose above the forum.")
# `CosyVoice3Model.silent_tokens`, set in its __init__ (the FSQ silence and breath codes).
SILENT_TOKENS = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]


class _F32Vocoder:
    """CausalHiFT kept at f32 under `--f64`: the question there is the flow's mel, and the vocoder's
    construction-time tables are f32."""

    def __init__(self, hift):
        self.hift = hift

    def inference(self, speech_feat, finalize):
        wave, source = self.hift.inference(speech_feat=speech_feat.float(), finalize=finalize)
        return wave.double(), source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the Fun-CosyVoice3-0.5B-2512 directory")
    ap.add_argument("--src", default="/home/flavio/Dev/CosyVoice")
    ap.add_argument("--matcha", default="/home/flavio/Dev/Matcha-TTS")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--f64", action="store_true", help="run the flow at float64 and only print the spread")
    args = ap.parse_args()

    R.import_reference(args.src, args.matcha)
    R.patch_step_mask()
    sampler = R.PinnedSampler(args.seed)
    sampler.install()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    llm, flow, hift = R.build_reference(args.model)
    fe = R.build_frontend(args.model)
    mi = fe.frontend_zero_shot(args.text, R.PROMPT_TEXT, f"{args.src}/asset/zero_shot_prompt.wav", 24000, "")
    with torch.inference_mode():
        tokens, _, _ = R.run_lm(llm, mi, out)
    kept = R.silence_filter(tokens, set(SILENT_TOKENS))
    prompt_tokens = mi["flow_prompt_speech_token"]
    if args.f64:
        flow = flow.double()
        flow.decoder.rand_noise = flow.decoder.rand_noise.double()
        hift = _F32Vocoder(hift)
        mi["prompt_speech_feat"] = mi["prompt_speech_feat"].double()
        mi["flow_embedding"] = mi["flow_embedding"].double()
    prompt_feat, embedding = mi["prompt_speech_feat"], mi["flow_embedding"]
    print(f"{len(tokens)} tokens drawn, {len(kept)} kept; prompt {prompt_tokens.shape[1]} tokens, "
          f"{prompt_feat.shape[1]} frames")

    def flow_run(token_ids, streaming, finalize):
        t = torch.tensor([token_ids], dtype=torch.int32)
        mel, _ = flow.inference(
            token=t, token_len=torch.tensor([t.shape[1]], dtype=torch.int32),
            prompt_token=prompt_tokens, prompt_token_len=torch.tensor([prompt_tokens.shape[1]], dtype=torch.int32),
            prompt_feat=prompt_feat, prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32),
            embedding=embedding, streaming=streaming, finalize=finalize)
        return mel                                                            # (1, 80, generated frames)

    with torch.inference_mode():
        mel_full = flow_run(kept, False, True)
        mel_streaming = flow_run(kept, True, True)
        wave_streaming, _ = hift.inference(speech_feat=mel_streaming, finalize=True)

        # `CosyVoice3Model.tts(stream=True)` with the LM already done: the same slices, whatever the timing.
        hop, max_hop = 25, 100
        look, ratio = flow.pre_lookahead_len, flow.token_mel_ratio
        pad = int(np.ceil(prompt_tokens.shape[1] / hop) * hop - prompt_tokens.shape[1])
        offset, mels, pieces = 0, [], []
        emitted = 0

        def token2wav(token_ids, offset, finalize):
            nonlocal emitted
            mels.append(flow_run(token_ids, True, finalize)[:, :, offset * ratio:])
            wave, _ = hift.inference(speech_feat=torch.cat(mels, dim=2), finalize=finalize)
            wave = wave[:, emitted:]
            emitted += wave.shape[1]
            return wave

        while True:
            this_hop = hop + pad if offset == 0 else hop
            if len(kept) - offset < this_hop + look:
                break
            pieces.append(token2wav(kept[:offset + this_hop + look], offset, False))
            offset += this_hop
            hop = min(max_hop, hop * 2)
        pieces.append(token2wav(kept, offset, True))
    mel_loop = torch.cat(mels, dim=2)

    def dist(a, b):
        n = min(a.shape[-1], b.shape[-1])
        return (a[..., :n] - b[..., :n]).abs().max().item()

    print(f"{len(pieces)} hops, pieces of {[p.shape[1] for p in pieces]} samples (first hop's pad {pad})")
    print(f"mel: stream loop vs streaming one pass {dist(mel_loop, mel_streaming):.3g}, streaming vs full "
          f"attention {dist(mel_streaming, mel_full):.3g}")
    print(f"wave: stream loop vs streaming one pass {dist(torch.cat(pieces, dim=1), wave_streaming):.3g}")
    if args.f64:
        return

    n_frames = 2 * (prompt_tokens.shape[1] + len(kept))
    R.save(out, "text_ids", mi["text"][0])
    R.save(out, "draws", np.asarray(sampler.draws))                           # (steps, 2)
    R.save(out, "noise", flow.decoder.rand_noise[0, :, :n_frames].T)          # (frames, 80)
    R.save(out, "nsf_noise", hift.m_source.l_sin_gen.sine_waves[0, :wave_streaming.shape[1]].T)  # (9, n)
    R.save(out, "mel_streaming", mel_streaming[0].T)                          # (generated frames, 80)
    R.save(out, "mel_stream_loop", mel_loop[0].T)
    R.save(out, "mel_full", mel_full[0].T)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
