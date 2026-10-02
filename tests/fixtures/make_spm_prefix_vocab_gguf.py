#!/usr/bin/env python3
"""Generates two tiny GGUF fixtures for test_spm_prefix_vocab.cpp.

`spm_prefix_vocab_test.gguf`: the SentencePiece-style BPE shape (`tokenizer.ggml.pre` =
"spm-byte-fallback") WITH `tokenizer.ggml.add_space_prefix` -- HF's `Prepend("\\u2581")` normalizer and
`Strip(" ", 1, 0)` decoder, Moonshine Streaming's tokenizer.json. Ids:

  0 <unk>  1 <s>  2 </s>  (CONTROL)
  3..258   <0x00>..<0xFF> (BYTE)
  259 "\\u2581"  260 "a"  261 "b"  262 "\\u2581a"  263 "ab"  264 "\\u2581ab"

merges, by rank: "\\u2581 a", "a b", "\\u2581a b" -- so "ab" with its prefix reduces to one piece, and
without one would stop at "ab".

`bpe_prefix_refused_test.gguf`: the byte-level qwen2 shape carrying the same key, which must not load.

Requires: pip install gguf
"""
import argparse
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

TOPOLOGY_JSON = '{"version": 1, "nodes": []}'
SP = "▁"


def write(path: Path, pre: str, tokens, types, merges) -> None:
    w = GGUFWriter(str(path), "loom-spm-prefix-vocab-fixture")
    w.add_string("loom.architecture", "spm_prefix_vocab_test")
    w.add_string("model.graph_topology", TOPOLOGY_JSON)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre(pre)
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_add_bos_token(False)
    w.add_add_space_prefix(True)
    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", nargs="?", default=".")
    out_dir = Path(parser.parse_args().out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)] + \
             [SP, "a", "b", SP + "a", "ab", SP + "ab"]
    types = [3, 3, 3] + [6] * 256 + [1] * 6
    merges = [f"{SP} a", "a b", f"{SP}a b"]
    write(out_dir / "spm_prefix_vocab_test.gguf", "spm-byte-fallback", tokens, types, merges)
    write(out_dir / "bpe_prefix_refused_test.gguf", "qwen2", tokens, types, merges)


if __name__ == "__main__":
    main()
