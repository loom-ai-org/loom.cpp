#!/usr/bin/env python3
"""Generates a small Chatterbox text-front-end GGUF fixture for test_chatterbox_vocab.cpp.

Chatterbox's real layout in miniature: `[STOP]`/`[UNK]`/`[SPACE]` at rows 0-2, an event tag as an
added token, a handful of characters, and three merges whose RANKS are what the test's "then" case
depends on (`t h` before `th e`, and no `the n`). The `punc_norm` rules are a subset of the real ones,
in the real order, and the case table is Python's own `str.upper()` over the letters the tests use --
including `ß`, whose uppercase is two codepoints.

Requires: pip install gguf
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

TOPOLOGY_JSON = '{"version": 1, "nodes": []}'
# ids:     0         1        2          3            4    5    6    7    8    9    10   11   12   13
TOKENS = ["[STOP]", "[UNK]", "[SPACE]", "[laughter]", ".", ",", "'", "H", "i", "t", "h", "e", "a", "n",
          # 14    15     16    17   18   19   20
          "th", "the", "an", "-", "I", "S", "s"]
ADDED = ["[STOP]", "[UNK]", "[SPACE]", "[laughter]"]
MERGES = ["t h", "th e", "a n"]
WORD_CHARS = ["H", "i", "t", "h", "e", "a", "n", "I", "S", "s"]
REPLACEMENTS = [("...", ", "), (";", ", "), (" ,", ",")]
ENDERS = [".", "!", "?", "-", ","]
UPPER = [(c, c.upper()) for c in "ihteansß"]


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("chatterbox_vocab_test.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out_path), "loom-chatterbox-vocab-fixture")
    w.add_string("loom.architecture", "chatterbox_vocab_test")
    w.add_string("model.graph_topology", TOPOLOGY_JSON)
    w.add_tokenizer_model("chatterbox")
    w.add_token_list(TOKENS)
    w.add_token_types([3 if t in ADDED else 1 for t in TOKENS])
    w.add_token_merges(MERGES)
    w.add_unk_token_id(1)
    p = "tokenizer.ggml.chatterbox."
    w.add_array(p + "added_tokens", ADDED)
    w.add_array(p + "word_chars", WORD_CHARS)
    w.add_array(p + "replace_from", [f for f, _ in REPLACEMENTS])
    w.add_array(p + "replace_to", [t for _, t in REPLACEMENTS])
    w.add_array(p + "sentence_enders", ENDERS)
    w.add_array(p + "upper_from", [f for f, _ in UPPER])
    w.add_array(p + "upper_to", [t for _, t in UPPER])
    w.add_array(p + "decode_drop", ["[STOP]", "[UNK]"])
    w.add_string(p + "space_token", "[SPACE]")
    w.add_string(p + "terminal", ".")
    w.add_string(p + "empty_text", "Hi.")
    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
