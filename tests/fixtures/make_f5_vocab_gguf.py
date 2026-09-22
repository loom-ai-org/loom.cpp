#!/usr/bin/env python3
"""Generates a small F5-TTS character-vocabulary GGUF fixture for test_f5_vocab.cpp.

The rows are `F5TTS_v1_Base`'s own `vocab.txt` layout in miniature: the SPACE at row 0 -- which is
also the unknown fallback, because the reference's `vocab_char_map.get(c, 0)` falls back to row 0 and
that row happens to be a space -- then punctuation, then letters, then two multi-character PINYIN
syllables of the kind that make up half the real table.

The pinyin rows are the point of the fixture rather than decoration. They are what a longest-match
scan would wrongly consume out of ASCII text, and `F5Vocab` indexes only the single-CODEPOINT rows
precisely so it cannot: `"an1"` is in the table below and `"an1"` typed as English must still encode
as three characters.

Requires: pip install gguf
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

TOPOLOGY_JSON = '{"version": 1, "nodes": []}'
# ids:            0    1    2    3    4    5    6    7    8    9   10   11   12   13     14      15
TOKENS = [" ", "!", ",", ".", "'", '"', "a", "e", "n", "o", "t", "z", "1", "é", "an1", "zhong1"]
UNK_ID = 0
FILLER_OFFSET = 1

TOKEN_TYPE_NORMAL = 1


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("f5_vocab_test.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = GGUFWriter(str(out_path), "loom-f5-vocab-fixture")
    w.add_string("loom.architecture", "f5_vocab_test")
    w.add_string("model.graph_topology", TOPOLOGY_JSON)

    w.add_tokenizer_model("f5")
    w.add_token_list(TOKENS)
    w.add_token_types([TOKEN_TYPE_NORMAL] * len(TOKENS))
    w.add_unk_token_id(UNK_ID)
    w.add_uint32("tokenizer.ggml.f5.filler_offset", FILLER_OFFSET)

    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
