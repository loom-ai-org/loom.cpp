#!/usr/bin/env python3
"""
Generates the GGUF fixture `tests/test_e2e_lstm_recurrent.cpp` loads: traces a real `torch.nn.LSTM`
(bidirectional), builds its per-timestep cell topologies via `recurrent.build_lstm_cell_topologies`, and
writes everything the C++ test needs into one GGUF -- the h_fwd/c_fwd/h_bwd/c_bwd topologies + weights,
plus the test's own input sequence and PyTorch-computed reference output as KV metadata (JSON arrays), so
the C++ test binary (which has no PyTorch available to it) can compare `loom.run_recurrent`'s real output
against a genuine `nn.LSTM` forward pass without needing Python at test-run time.

Usage:
  ~/.venvs/piper/bin/python3 export_lstm_test_fixture.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import loom_mil_compiler  # noqa: F401 -- registers the "loom" backend + applies torch-frontend patches
import coremltools as ct
from loom_mil_compiler.recurrent import build_lstm_cell_topologies

HIDDEN_DIM = 6
INPUT_DIM = 4
SEQ_LEN = 9
OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / "lstm_recurrent_test.gguf"


def main():
    torch.manual_seed(1234)
    lstm = torch.nn.LSTM(input_size=INPUT_DIM, hidden_size=HIDDEN_DIM, num_layers=1,
                          batch_first=False, bidirectional=True).eval()
    x = torch.randn(SEQ_LEN, 1, INPUT_DIM)

    class M(torch.nn.Module):
        def __init__(self, lstm):
            super().__init__()
            self.lstm = lstm

        def forward(self, x):
            out, _ = self.lstm(x)
            return out

    m = M(lstm).eval()
    with torch.no_grad():
        ref_out = m(x).squeeze(1).numpy().astype(np.float32)  # (SEQ_LEN, 2*HIDDEN_DIM)

    traced = torch.jit.trace(m, (x,))
    prog = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x.shape)], convert_to="milinternal")
    func = prog.functions["main"]
    op = next(o for o in func.operations if o.op_type == "lstm")

    result = build_lstm_cell_topologies(op, weight_namespace="lstm.")
    assert result["bidirectional"]
    assert result["hidden_dim"] == HIDDEN_DIM
    assert result["input_dim"] == INPUT_DIM

    from gguf import GGUFWriter
    w = GGUFWriter(str(OUTPUT_PATH), "loom-lstm-recurrent-test")
    for name, topo in (
        ("h_fwd", result["forward"]["h"]), ("c_fwd", result["forward"]["c"]),
        ("h_bwd", result["backward"]["h"]), ("c_bwd", result["backward"]["c"]),
    ):
        w.add_string(f"model.graph_topology.{name}", json.dumps(topo))
    for name, arr in result["weights"].items():
        w.add_tensor(name, arr.astype(np.float32))

    sequence_flat = x.squeeze(1).numpy().astype(np.float32).reshape(-1).tolist()
    reference_flat = ref_out.reshape(-1).tolist()
    w.add_string("test.input_sequence", json.dumps(sequence_flat))
    w.add_string("test.reference_output", json.dumps(reference_flat))
    w.add_uint32("test.seq_len", SEQ_LEN)
    w.add_uint32("test.input_dim", INPUT_DIM)
    w.add_uint32("test.hidden_dim", HIDDEN_DIM)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
