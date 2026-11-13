#!/usr/bin/env python3
"""Regenerate the oracle for Chatterbox's live stream (ADR-117).

S3Gen is CosyVoice2's flow and HiFT and keeps CosyVoice2's streaming hook, `flow_inference(finalize=
False)`: the encoder runs over every token so far and the mel of the last `pre_lookahead_len` (3) tokens is
dropped, those tokens being the look-ahead conv's context. Chatterbox ships no loop around it (the
`S3GenStreamer` its docstring names is not in the release), so the loop is CosyVoice2Model's
`tts(stream=True)` over Chatterbox's modules: each hop of speech tokens (25, then 50, then 100 from there
on; CosyVoice2's first-hop pad onto its chunk boundary is dropped, since S3Gen has no chunk mask) runs the
flow over the voice's prompt and the tokens so far plus the 3 look-ahead tokens and keeps the mel past
what it had; the last hop runs `finalize=True` over all of them. The timing does not change the slices,
so T3 runs first, guided greedy as `chatterbox_reference.py` runs it (that script's helpers are imported,
not copied).

**The hook does not run as shipped**: `CausalMaskedDiffWithXvec.inference` trims `h` for `finalize=False`
but builds the decoder's mask from the untrimmed lengths, and `solve_euler` fails on the mismatch (364
against 370 frames). The mask is all ones for one unpadded utterance, so this trims it to `mu`'s length,
which is what CosyVoice2's own flow does (it builds the mask from `h`).

Both runs draw the flow's noise from ONE table sliced from the start (`CausalConditionalCFM` draws a fresh
`randn_like(mu)` per call), which is what the driver does with its per-frame draw. Written out, all float32
`.npy` (`tests/support/npy_fixture.h` reads no other dtype):

* `text_ids` (the text's ids without the start/stop ids the driver adds), `noise` (the table, frame-major,
  for the whole grid), `nsf_phase` (9, 1) and `nsf_noise` (9, n): the draws, pinned.
* `mel_one`, `mel_stream_loop`: the generated frames' mel, frame-major, from the one pass and the loop.
* `mel_other_draw`: the one pass under another noise table -- another take of the same tokens, the
  spread a stream's distance from the one pass is read against.

    ~/.venvs/piper/bin/python scripts/chatterbox_stream_reference.py \\
        --model ~/Dev/models/chatterbox --out $LOOM_FIXTURES/chatterbox_stream_ref

The gate is `tests/gate/test_e2e_chatterbox_stream.cpp`. `--f64` writes nothing; it prints how far the f32
loop and one pass are from their f64 runs, which is the rounding a gate tolerance has to allow.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chatterbox_reference as R  # noqa: E402

# Long enough for four hops (200 speech tokens, 400 frames, 8.0 s).
DEFAULT_TEXT = ("The history of the city of Rome begins on the Palatine hill, where shepherds built their "
                "first huts beside the river, long before the first temples rose above the forum.")
HOP, MAX_HOP, SCALE = 25, 100, 2


def install_pinned_noise(table: torch.Tensor) -> None:
    """`CausalConditionalCFM.forward` with its `randn_like(mu)` replaced by the table's first frames, and
    its mask trimmed to `mu` (see the top)."""
    from chatterbox.models.s3gen import flow_matching

    real_forward = flow_matching.CausalConditionalCFM.forward

    def forward(module, mu, *args, **kwargs):
        if "mask" in kwargs:
            kwargs["mask"] = kwargs["mask"][..., :mu.shape[2]]
        real_randn_like = torch.randn_like
        torch.randn_like = lambda x, *a, **k: table[:, :, :x.shape[2]].to(x.dtype)
        try:
            return real_forward(module, mu, *args, **kwargs)
        finally:
            torch.randn_like = real_randn_like

    flow_matching.CausalConditionalCFM.forward = torch.inference_mode()(forward)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the ResembleAI/chatterbox directory")
    ap.add_argument("--src", default="/home/flavio/Dev/chatterbox/src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--f64", action="store_true", help="also run S3Gen at float64 and only print the spread")
    args = ap.parse_args()

    R._stub_reference_imports()
    sys.path.insert(0, args.src)
    from chatterbox.tts import ChatterboxTTS, punc_norm

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    table = torch.randn(1, 80, 4096)
    other = torch.randn(1, 80, 4096)
    noise = table.clone()
    install_pinned_noise(noise)
    tts = ChatterboxTTS.from_local(args.model, "cpu")
    hp = tts.t3.hp
    text_tokens = tts.tokenizer.text_to_tokens(punc_norm(args.text))
    framed = F.pad(F.pad(text_tokens, (1, 0), value=hp.start_text_token), (0, 1), value=hp.stop_text_token)
    # `guided_greedy` writes its own T3 probes; the one-pass oracle keeps them, this one does not.
    with torch.inference_mode(), tempfile.TemporaryDirectory() as probes:
        predicted, _ = R.guided_greedy(tts.t3, tts.conds.t3, framed.long(), 0.5, 1.2, 1000, Path(probes))
    kept = [t for t in predicted if t < 6561]
    s3gen, gen = tts.s3gen, dict(tts.conds.gen)
    n_prompt = gen["prompt_token"].shape[1]
    print(f"T3: {len(predicted)} ids, {len(kept)} kept; prompt {n_prompt} tokens")

    def run(s3, ref):
        def flow(ids, finalize):
            return s3.flow_inference(torch.tensor([ids]), ref_dict=dict(ref), finalize=finalize)

        with torch.inference_mode():
            one = flow(kept, True)
            hop, offset, mels = HOP, 0, []
            while len(kept) - offset >= hop + s3.flow.pre_lookahead_len:
                mels.append(flow(kept[:offset + hop + s3.flow.pre_lookahead_len], False)[:, :, 2 * offset:])
                offset += hop
                hop = min(MAX_HOP, hop * SCALE)
            mels.append(flow(kept, True)[:, :, 2 * offset:])
        return one, torch.cat(mels, dim=2), [m.shape[2] for m in mels]

    mel_one, mel_loop, hops = run(s3gen, gen)
    noise.copy_(other)
    with torch.inference_mode():
        mel_other = s3gen.flow_inference(torch.tensor([kept]), ref_dict=dict(gen), finalize=True)
    noise.copy_(table)

    def dist(a, b):
        return (a - b).abs().max().item(), (a - b).abs().mean().item()

    print(f"{len(hops)} hops of {hops} frames; mel frames {mel_loop.shape[2]} (one pass {mel_one.shape[2]})")
    print("loop vs one pass: max|d| %.3g mean %.3g" % dist(mel_loop, mel_one))
    print("another draw vs one pass: max|d| %.3g mean %.3g" % dist(mel_other, mel_one))
    if args.f64:
        s3_64 = s3gen.double()
        from chatterbox.models.s3gen import decoder as s3gen_decoder
        s3gen_decoder.mask_to_bias = lambda mask, dtype: (1.0 - mask.to(dtype)) * -1.0e+10
        gen64 = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in gen.items()}
        one64, loop64, _ = run(s3_64, gen64)
        print("f32 vs f64: one pass max|d| %.3g, loop %.3g" % (dist(mel_one.double(), one64)[0],
                                                              dist(mel_loop.double(), loop64)[0]))
        return

    n_frames = 2 * (n_prompt + len(kept))
    n_samples = 480 * (n_frames - gen["prompt_feat"].shape[1])
    gen_rng = torch.Generator().manual_seed(args.seed + 1)
    phase = (2 * torch.rand(9, 1, generator=gen_rng) - 1) * np.pi
    phase[0] = 0
    R.save(out, "text_ids", text_tokens[0])
    R.save(out, "noise", table[0, :, :n_frames].T)                              # (frames, 80)
    R.save(out, "nsf_phase", phase)
    R.save(out, "nsf_noise", torch.randn(9, n_samples, generator=gen_rng))      # (9, n)
    R.save(out, "speech_tokens", kept)
    R.save(out, "mel_one", mel_one[0].T)                                        # (generated frames, 80)
    R.save(out, "mel_stream_loop", mel_loop[0].T)
    R.save(out, "mel_other_draw", mel_other[0].T)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
