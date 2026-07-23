#!/usr/bin/env python3
"""Generates a small ByT5-family GGUF fixture for test_byte_vocab.cpp. Uses a REAL (small) extra_ids
count of 4 (rather than the real default 125) purely to keep the fixture/test small -- the scheme itself
(pad=0/eos=1/unk=2, byte b -> id b+3, sentinels sequential right after the byte range) is not
parameterized by extra_ids count in any way that would make a smaller count unrepresentative.

Requires: pip install gguf
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

TOPOLOGY_JSON = '{"version": 1, "nodes": []}'
EXTRA_IDS = 4


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("byte_vocab_test.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vocab_size = 3 + 256 + EXTRA_IDS
    tokens = [""] * vocab_size
    tokens[0] = "<pad>"
    tokens[1] = "</s>"
    tokens[2] = "<unk>"
    for i in range(EXTRA_IDS):
        tokens[3 + 256 + i] = f"<extra_id_{i}>"

    w = GGUFWriter(str(out_path), "loom-byte-vocab-fixture")
    w.add_string("loom.architecture", "byte_vocab_test")
    w.add_string("model.graph_topology", TOPOLOGY_JSON)

    w.add_tokenizer_model("byt5")
    w.add_token_list(tokens)
    w.add_pad_token_id(0)
    w.add_eos_token_id(1)
    w.add_unk_token_id(2)

    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
