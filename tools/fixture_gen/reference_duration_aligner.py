#!/usr/bin/env python3
"""Independent numpy computation of Kokoro/StyleTTS2's duration-based frame-expansion formula (real
source: kokoro/model.py's KModel.forward_with_tokens):
    duration = sigmoid(duration_logits).sum(-1) / speed
    pred_dur = round(duration).clamp(min=1)
    indices = repeat_interleave(arange(T), pred_dur)
    pred_aln_trg[indices, arange(sum(pred_dur))] = 1
    expanded = seq.T @ pred_aln_trg  (equivalently: row t of seq repeated pred_dur[t] times)
writing duration_logits/seq/expected_pred_dur/expected_expanded as raw binaries for
test_duration_aligner.cpp to compare loom::predict_durations/loom::expand_by_duration against. Uses
np.round (round-half-to-even, matching torch.round) rather than a round-half-away-from-zero function,
matching the real formula exactly.

Usage: python3 reference_duration_aligner.py <out_dir>
Requires: pip install numpy
"""
import sys
from pathlib import Path

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("duration_aligner_ref")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    T, max_dur, channels, speed = 6, 50, 5, 1.0

    # Biased toward larger logits so most sigmoid sums round to a handful of frames per token (a
    # too-small mean would make every token round down to the same 1-frame floor, which wouldn't
    # exercise the repeat-count-varies-per-token path this test actually cares about).
    duration_logits = rng.normal(loc=-1.0, scale=2.5, size=(T, max_dur)).astype(np.float32)
    seq = rng.normal(scale=0.5, size=(T, channels)).astype(np.float32)

    duration = sigmoid(duration_logits.astype(np.float64)).sum(axis=-1) / speed
    pred_dur = np.clip(np.round(duration), 1, None).astype(np.uint32)

    expanded = np.repeat(seq, pred_dur.astype(np.int64), axis=0)

    duration_logits.tofile(out_dir / "duration_logits.bin")
    seq.tofile(out_dir / "seq.bin")
    pred_dur.astype(np.int32).tofile(out_dir / "expected_pred_dur.bin")
    expanded.tofile(out_dir / "expected_expanded.bin")
    print(f"T={T}, max_dur={max_dur}, channels={channels}, pred_dur={pred_dur}, "
          f"T_frames={int(pred_dur.sum())}")


if __name__ == "__main__":
    main()
