#!/usr/bin/env python3
"""Generates a tiny synthetic SentencePiece-Unigram ("t5" tag) GGUF fixture for
test_vocab_ugm_bos_eos.cpp -- just enough pieces/scores to hand-trace a Viterbi segmentation, plus
bos_token_id/eos_token_id/add_bos_token/add_eos_token KVs set, to test the ALBERT/XLNet-style gap
Vocab::encode closed (see EXPORT-BACKLOG.md item 4). No precompiled_charsmap -- Vocab::load/normalize()
both work fine without one (falls back to identity normalization + add_space_prefix).

Requires: pip install gguf
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

TOPOLOGY_JSON = '{"version": 1, "nodes": []}'

# SentencePiece's own piece Type enum (see tools/convert_nemo/tokenizer_common.py's own doc comment).
NORMAL, UNKNOWN, CONTROL = 1, 2, 3

# (piece, score, type) -- "▁hi" is a much better (less negative) score than any alternative
# segmentation ("▁h" + "i"), so the Viterbi search should pick it as a single token.
PIECES = [
    ("<unk>", 0.0, UNKNOWN),   # 0
    ("<s>", 0.0, CONTROL),     # 1 -- BOS, never matchable in normal segmentation (CONTROL type)
    ("</s>", 0.0, CONTROL),    # 2 -- EOS, ditto
    ("▁hi", -1.0, NORMAL),  # 3 -- "▁hi"
    ("▁h", -3.0, NORMAL),   # 4 -- "▁h"
    ("i", -3.0, NORMAL),          # 5
]


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("vocab_ugm_bos_eos_test.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokens = [p for p, _, _ in PIECES]
    scores = [s for _, s, _ in PIECES]
    types = [t for _, _, t in PIECES]

    w = GGUFWriter(str(out_path), "loom-vocab-ugm-bos-eos-fixture")
    w.add_string("loom.architecture", "vocab_ugm_bos_eos_test")
    w.add_string("model.graph_topology", TOPOLOGY_JSON)

    w.add_tokenizer_model("t5")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(types)
    w.add_unk_token_id(0)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_add_bos_token(True)
    w.add_add_eos_token(True)

    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
