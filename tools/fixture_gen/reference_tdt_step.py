#!/usr/bin/env python3
"""Independent numpy computation of the tiny synthetic TDT decode (see tdt_step_common.py), writing the
encoder output plus the expected token/frame-index sequence for test_tdt_decoder.cpp to compare loom's new
TdtDecoder C++ driver against.

Usage: python3 reference_tdt_step.py <out_dir>
Requires: pip install numpy
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tdt_step_common as common


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tdt_step_ref")
    out_dir.mkdir(parents=True, exist_ok=True)

    w = common.generate_weights()
    encoder_output = common.generate_encoder_output()
    tokens, frame_indices = common.reference_greedy_decode(encoder_output, w)

    encoder_output.tofile(out_dir / "encoder_output.bin")
    (out_dir / "expected.json").write_text(json.dumps({"tokens": tokens, "frame_indices": frame_indices}))
    print(f"wrote fixture to {out_dir}: tokens={tokens}, frame_indices={frame_indices}")


if __name__ == "__main__":
    main()
