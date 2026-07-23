#!/usr/bin/env python3
"""Generates a small, deterministic WordPiece GGUF fixture for test_wordpiece_vocab.cpp -- NOT a real
BERT vocab, just enough pieces to hand-trace exact expected token ids for: a fully-covered word, a word
needing a "##"-continuation split, punctuation isolation, [UNK] fallback, accent-stripping, and
CLS/SEP auto-wrap.

Tokens are written already `phantom()`-transformed (see wordpiece_tokenizer_export.py's own doc comment
for why: control tokens unchanged, "##continuation" pieces have the marker stripped bare, everything
else gets "▁" prepended) -- this is what a real export would write, and what
loom::WordPieceVocab::load reads directly with no further transformation.

Requires: pip install gguf
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

TOPOLOGY_JSON = '{"version": 1, "nodes": []}'

# id: piece (already phantom-transformed)
TOKENS = [
    "[PAD]",     # 0
    "[UNK]",     # 1
    "[CLS]",     # 2
    "[SEP]",     # 3
    "[MASK]",    # 4
    "▁hello",  # 5
    "▁world",  # 6
    "▁un",     # 7 -- whole-word-start piece "un"
    "happy",        # 8 -- continuation piece, was "##happy"
    "▁,",      # 9 -- punctuation is always word-start (never a continuation), so still ▁-prefixed
    "▁cafe",   # 10 -- accent-stripped form of "café"
]


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("wordpiece_vocab_test.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = GGUFWriter(str(out_path), "loom-wordpiece-vocab-fixture")
    w.add_string("loom.architecture", "wordpiece_vocab_test")
    w.add_string("model.graph_topology", TOPOLOGY_JSON)

    w.add_tokenizer_model("bert")
    w.add_token_list(TOKENS)
    w.add_unk_token_id(1)
    w.add_bos_token_id(2)   # CLS reuses the generic BOS KV
    w.add_sep_token_id(3)
    w.add_pad_token_id(0)
    w.add_mask_token_id(4)
    w.add_add_bos_token(True)
    w.add_bool("tokenizer.ggml.add_sep_token", True)
    w.add_bool("tokenizer.ggml.normalizer.lowercase", True)
    w.add_bool("tokenizer.ggml.normalizer.strip_accents", True)

    # GgufModel::load() allocates a backend buffer sized to the meta context's tensors -- a vocab-only
    # fixture with zero tensors hits a ggml_backend edge case, so (like every other fixture in this repo)
    # a small placeholder weight is included even though this test never reads it.
    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
