#!/usr/bin/env python3
"""Independent numpy computation of the tiny synthetic LSTM-step fixture (see lstm_step_common.py),
writing x/h_prev/c_prev/expected_h_new/expected_c_new as raw f32 binaries for test_lstm_step.cpp to
compare loom-engine's composite-topology output against.

Usage: python3 reference_lstm_step.py <out_dir>
Requires: pip install numpy
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_step_common as common


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("lstm_step_ref")
    out_dir.mkdir(parents=True, exist_ok=True)

    w = common.generate_weights()
    inputs = common.generate_inputs()
    h_new, c_new = common.reference_step(inputs["x"], inputs["h_prev"], inputs["c_prev"], w)

    inputs["x"].tofile(out_dir / "x.bin")
    inputs["h_prev"].tofile(out_dir / "h_prev.bin")
    inputs["c_prev"].tofile(out_dir / "c_prev.bin")
    h_new.tofile(out_dir / "expected_h_new.bin")
    c_new.tofile(out_dir / "expected_c_new.bin")
    print(f"wrote fixture to {out_dir}: h_new={h_new}, c_new={c_new}")


if __name__ == "__main__":
    main()
