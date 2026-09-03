#!/usr/bin/env python3
"""Regenerate the `transformers` oracle that `tests/gate/test_e2e_dia_mil_export.cpp` compares against.

Run it in the exporter environment, which is the one that has `transformers`:

    ~/.venvs/piper/bin/python scripts/dia_reference_codes.py \
        --model ~/Dev/models/dia-1.6b --frames 32

It prints the expected frame count and the codes as a C array, ready to paste into the test's
`kExpectedCodes`. It writes nothing into the repo -- the numbers live in the test source, which is
where a reader can see what is being asserted.

**Greedy, with classifier-free guidance OFF, and both of those are load-bearing.** The driver under
test decodes that way today, and an oracle produced by a different algorithm grades the sampler rather
than the export. When the driver learns to sample and to run CFG, this script has to learn the same
thing on the same commit, or the gate silently starts measuring the difference between two samplers.
`rotate_half` is deliberately left unpatched here: the reference must come from the unmodified model,
or the export's own patch is grading itself.

**`--frames` is a count of AUDIO FRAMES, and the translation to what HF wants is the subtle part.**
This sentence does not make the model emit EOS on its own inside any budget a gate test can afford, so
both sides stop because they were told to. `transformers` has no frame-count knob: it takes
`max_new_tokens` in ROWS, and `DiaEOSDelayPatternLogitsProcessor` turns that into a forced EOS at row
`max_length - max(delay_pattern) - 1`, which yields that many minus one audio frames. So N frames
means `max_new_tokens = N + max(delay_pattern) + 1`, which is what `--frames` converts to below. The
loom driver's own `max_new_tokens` already counts frames, so it takes N directly -- a better host
contract, since rows are an artefact of the delay pattern rather than anything a caller asked for.
"""
import argparse
import sys

import torch
from transformers import DiaForConditionalGeneration, DiaProcessor


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the Dia checkpoint directory")
    ap.add_argument("--text", default="[S1] Hello world.",
                    help="the sentence to capture; must match the test's kPromptIds")
    ap.add_argument("--frames", type=int, default=32, help="how many AUDIO frames to capture")
    args = ap.parse_args()

    processor = DiaProcessor.from_pretrained(args.model)
    model = DiaForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager").eval()

    delay = list(model.config.delay_pattern)
    n_channels = model.config.decoder_config.num_channels
    bos, pad = model.config.bos_token_id, model.config.pad_token_id
    # See the module docstring: HF counts rows and forces an EOS `max(delay) + 1` short of the budget.
    max_new_tokens = args.frames + max(delay) + 1

    encoded = processor(text=[args.text])
    with torch.no_grad():
        out = model.generate(**encoded, do_sample=False, temperature=1.0,
                             guidance_scale=None, max_new_tokens=max_new_tokens)

    seq = out[0]
    # The delay revert, exactly as `DiaProcessor.batch_decode` does it before handing codes to DAC --
    # `out[t][k] = in[t + delay[k]][k]` -- with the same window: from the BOS scaffold's end to the row
    # channel 0 said EOS on. Written out rather than called, because `batch_decode` goes on to run the
    # codec and this script has no need of one.
    start = int((seq[:, 0] == bos).sum())
    end = int(seq.shape[0] - (seq[:, 0] == pad).sum() - 1)
    frames = [[int(seq[t + delay[k], k]) for k in range(n_channels)] for t in range(start, end)]

    if len(frames) != args.frames:
        print(f"WARNING: asked for {args.frames} frames and got {len(frames)} -- the model emitted EOS "
              f"on its own before the ceiling. Use this count in the test, not the one you asked for.",
              file=sys.stderr)

    print(f"// {args.text!r}, greedy, CFG off, {len(frames)} frames x {n_channels} channels")
    print(f"const std::vector<double> kPromptIds = {{"
          f"{', '.join(str(i) for i in encoded['input_ids'][0].tolist())}}};")
    print(f"constexpr int kExpectedFrames = {len(frames)};")
    print(f"constexpr int kChannels = {n_channels};")
    print("const int32_t kExpectedCodes[] = {")
    flat = [c for frame in frames for c in frame]
    for i in range(0, len(flat), 12):
        print("    " + ", ".join(str(v) for v in flat[i:i + 12]) + ",")
    print("};")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
