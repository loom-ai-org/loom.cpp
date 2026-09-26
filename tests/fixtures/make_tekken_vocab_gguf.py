#!/usr/bin/env python3
"""Generates a tiny Tekken-shaped GGUF fixture for test_tekken_vocab.cpp: `tokenizer.ggml.pre` =
"tekken", four markers (the fourth spelled "Wo", like rank 261, so text must reach the RANK), the 256 byte ranks and nine hand-picked ranks, and NO merges -- the layout
`loom_exporter.tekken_tokenizer_export` writes for a real `tekken.json`, small enough to trace by hand.

Ids are `rank + 4` (the four markers come first). The ranks are chosen so each rule the `tekken` shape
adds changes an id the test can name:

  256 "oW"    LOWEST, so "HelloWorld" merged as one chunk would pair o+W before anything else; the
              case-change split is what keeps "Hello" whole
  257 "ll", 258 "He", 259 "Hell", 260 "Hello", 261 "Wo"
  262 "<s"    what tiktoken's regex makes of a typed "<s>": the letter alternative takes "<" as its
              optional prefix and stops at ">"
  263 "12"    tiktoken's `\\p{N}` takes one digit per chunk, so this never applies
  264 "é"     U+00E9's two bytes: an NFC pass would turn "e + U+0301" into it; tiktoken does not

Requires: pip install gguf
"""
import argparse
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

TOPOLOGY_JSON = '{"version": 1, "nodes": []}'
MARKERS = ["<unk>", "<s>", "</s>", "Wo"]
RANKS = [b"oW", b"ll", b"He", b"Hell", b"Hello", b"Wo", b"<s", b"12", "é".encode()]


def bytes_to_unicode() -> dict:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path", nargs="?", default="tekken_vocab_test.gguf")
    args = parser.parse_args()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = bytes_to_unicode()
    ranks = [bytes([b]) for b in range(256)] + RANKS
    tokens = MARKERS + ["".join(m[b] for b in r) for r in ranks]
    types = [3] * len(MARKERS) + [1] * len(ranks)

    w = GGUFWriter(str(out_path), "loom-tekken-vocab-fixture")
    w.add_string("loom.architecture", "tekken_vocab_test")
    w.add_string("model.graph_topology", TOPOLOGY_JSON)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("tekken")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_add_bos_token(False)
    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
