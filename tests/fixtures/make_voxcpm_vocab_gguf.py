#!/usr/bin/env python3
"""A tiny `tokenizer.ggml.model = "voxcpm2"` file, for tests/ci/test_voxcpm_vocab.cpp.

A rank-merged character BPE with all 256 byte-fallback pieces, a handful of pieces and merges, two added
tokens and a one-entry split table -- small enough that every id in the test is derivable by hand. The
real keys come from the reference (`loom-exporter/loom_exporter/voxcpm2_tokenizer_export.py`).

Requires: pip install gguf numpy
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

NORMAL, CONTROL, BYTE = 1, 3, 6
P = "tokenizer.ggml.voxcpm2."

# ids: 0 <unk>, 1 <s>, 2 <|audio_start|>, 3..258 the bytes, then these in order (259 ..).
PIECES = ["▁", "H", "i", "Hi", "▁Hi", "a", "b", "ab", "ba", "你", "好", "你好",
          "▁你好"]
# Rank order is the test: `a b` outranks `b a`, so "aba" is `ab a`, and the leftmost of two equal
# ranks goes first.
MERGES = ["H i", "▁ Hi", "a b", "b a", "你 好", "▁ 你好"]


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("voxcpm_vocab_test.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = ["<unk>", "<s>", "<|audio_start|>"] + [f"<0x{b:02X}>" for b in range(256)] + PIECES
    types = [CONTROL, CONTROL, CONTROL] + [BYTE] * 256 + [NORMAL] * len(PIECES)
    ids = {t: i for i, t in enumerate(tokens)}
    w = GGUFWriter(str(out_path), "loom-voxcpm-vocab-fixture")
    w.add_string("loom.architecture", "voxcpm_vocab_test")
    w.add_string("model.graph_topology", '{"version": 1, "nodes": []}')
    w.add_tokenizer_model("voxcpm2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(MERGES)
    w.add_unk_token_id(0)
    w.add_array(P + "added_tokens", ["<unk>", "<s>", "<|audio_start|>"])
    w.add_string(P + "prepend", "▁")
    w.add_string(P + "space", "▁")
    # Both spellings of "你好" split, as the reference's wrapper strips the `▁` before looking.
    w.add_array(P + "split_from", [ids["你好"], ids["▁你好"]])
    w.add_array(P + "split_offsets", [0, 2, 4])
    w.add_array(P + "split_to", [ids["你"], ids["好"], ids["你"], ids["好"]])
    # One tensor, because a file with none cannot allocate the weight buffer `GgufModel::load` wants.
    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
