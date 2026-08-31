#!/usr/bin/env python3
"""Generates a small byte-level-BPE GGUF carrying ADDED tokens and a decomposed chat template, for
test_chat_template.cpp (P4.23).

Deliberately hermetic and hand-traceable, the same way make_bpe_vocab_gguf.py is: the base vocab is the
GPT2 byte-level mapping over all 256 bytes, plus four merges that reduce "hello" to one token, plus three
ADDED tokens whose ids are the point of the fixture --

    <|im_start|>  id 260, CONTROL       a marker the model consumes; ChatML's turn opener
    <|im_end|>    id 261, CONTROL       ... and its closer, which is also this fixture's chat eos
    "\\n\\n"        id 262, USER_DEFINED  an added token that is NOT special

The third is not decoration. HF's `AddedVocabulary` splits the raw input on EVERY added token, special
or not -- Gemma 3 declares 6408 non-special ones, all whitespace runs -- so a pre-pass that only knew
about markers would tokenize `"a\\n\\nb"` differently from the reference tokenizer on ordinary prose.

`--no-token-type` writes the identical file WITHOUT `tokenizer.ggml.token_type`, which is what every
GGUF exported before P4.23 looks like. The C++ test loads both and requires them to tokenize
DIFFERENTLY: that is the backward-compatibility claim stated as a test rather than as a comment.

Requires: pip install gguf
"""
import argparse
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

# gguf's TokenType; see loom_exporter/bpe_tokenizer_export.py, which writes the same values.
TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_USER_DEFINED = 4

TOPOLOGY_JSON = '{"version": 1, "nodes": []}'


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path", nargs="?", default="chat_template_test.gguf")
    parser.add_argument("--no-token-type", action="store_true",
                        help="omit tokenizer.ggml.token_type, i.e. a pre-P4.23 file")
    args = parser.parse_args()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    byte_to_cp = bytes_to_unicode()
    base_tokens = [chr(byte_to_cp[b]) for b in range(256)]

    # Same merge chain make_bpe_vocab_gguf.py uses, so "hello" is one token and the ids in the C++ test
    # stay hand-traceable.
    merges = ["l l", "h e", "he ll", "hell o"]
    # ids 256..259
    extra_tokens = ["ll", "he", "hell", "hello"]
    # ids 260..262. An added token's piece is its RAW content, not the byte-level spelling the rest of a
    # gpt2-schema vocabulary uses -- that is what `write_bpe_vocab` writes for a real checkpoint
    # (`tokens[t["id"]] = t["content"]`), and the newline run is here because it is the case where the
    # difference is visible: a literal "\n" has no entry in the byte DECODER, so decoding it through that
    # map would drop it.
    added_tokens = ["<|im_start|>", "<|im_end|>", "\n\n"]

    tokens = base_tokens + extra_tokens + added_tokens
    token_types = [TOKEN_TYPE_NORMAL] * len(tokens)
    token_types[260] = TOKEN_TYPE_CONTROL
    token_types[261] = TOKEN_TYPE_CONTROL
    token_types[262] = TOKEN_TYPE_USER_DEFINED

    w = GGUFWriter(str(out_path), "loom-chat-template-fixture")
    w.add_string("loom.architecture", "chat_template_test")
    w.add_string("model.graph_topology", TOPOLOGY_JSON)

    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    if not args.no_token_type:
        w.add_token_types(token_types)
    w.add_token_merges(merges)
    w.add_bos_token_id(0)
    # Two stop tokens, the shape P4.23 exists for: a base end-of-text and the chat turn's own end. The
    # scalar KV stays the first of them, so a reader that predates the array behaves as it always did.
    w.add_eos_token_id(0)
    w.add_array("tokenizer.ggml.eos_token_ids", [0, 261])

    # A ChatML decomposition, the shape loom-exporter's chat_template_export.py derives from SmolLM2 and
    # LFM2. No prologue: ChatML opens straight onto the first turn.
    w.add_array("tokenizer.chat_template.roles", ["user", "assistant", "system"])
    w.add_array("tokenizer.chat_template.prefixes",
                ["<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_start|>system\n"])
    w.add_array("tokenizer.chat_template.suffixes", ["<|im_end|>\n", "<|im_end|>\n", "<|im_end|>\n"])
    w.add_string("tokenizer.chat_template.prologue", "")
    w.add_string("tokenizer.chat_template.system_prologue", "")
    w.add_string("tokenizer.chat_template.generation_prefix", "<|im_start|>assistant\n")
    w.add_bool("tokenizer.chat_template.trim_content", False)

    # GgufModel::load() sizes a backend buffer from the meta context, and a tensor-free file hits a
    # ggml_backend edge case -- same placeholder every other vocab fixture here carries.
    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
