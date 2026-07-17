#!/usr/bin/env python3
"""Generates a small, deterministic byte-level-BPE GGUF fixture for test_bpe_vocab.cpp -- NOT a real
tokenizer, just enough vocab + merges to hand-trace exact expected token ids for a few plain-ASCII cases
(see the C++ test for the traced-by-hand expectations), plus non-ASCII/NFC round-trip cases that don't
need hand-traced ids.

Base vocab (ids 0..255) is the GPT2 byte-level mapping applied to every possible byte value, using the
exact same algorithm the real Qwen2/Qwen3 tokenizer.json (and loom::BpeVocab's own byte_encoder()) use --
reproduced here in Python so the fixture's vocab lines up byte-for-byte with what the C++ side computes at
runtime, without duplicating this project's C++ table by hand.

Requires: pip install gguf
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter


def bytes_to_unicode() -> dict[int, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, cs))


# Kept in sync with the C++ test's expectation of the raw topology string, same convention as every other
# fixture generator in this repo (a BPE vocab test has no graph to run, but GgufModel::load requires it).
TOPOLOGY_JSON = '{"version": 1, "nodes": []}'


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bpe_vocab_test.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    byte_to_cp = bytes_to_unicode()
    base_tokens = [chr(byte_to_cp[b]) for b in range(256)]

    # Learned in this exact order (rank = list index) so "hello" fully reduces to one token:
    #   ['h','e','l','l','o'] --"l l"--> ['h','e','ll','o'] --"h e"--> ['he','ll','o']
    #                         --"he ll"--> ['hell','o']    --"hell o"--> ['hello']
    merges = ["l l", "h e", "he ll", "hell o"]
    extra_tokens = ["ll", "he", "hell", "hello"]

    tokens = base_tokens + extra_tokens

    w = GGUFWriter(str(out_path), "loom-bpe-vocab-fixture")
    w.add_string("loom.architecture", "bpe_vocab_test")
    w.add_string("model.graph_topology", TOPOLOGY_JSON)

    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    w.add_token_merges(merges)
    w.add_bos_token_id(0)
    w.add_eos_token_id(1)

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
