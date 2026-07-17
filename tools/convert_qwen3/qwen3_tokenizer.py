"""Extracts byte-level-BPE vocab data from a real tokenizer.json and writes it into a GGUF file using
llama.cpp's own "tokenizer.ggml.*" schema for a "gpt2"-style vocab (confirmed directly against the
installed `gguf` package's GGUFWriter methods: add_tokenizer_model/add_token_list/add_token_merges/
add_bos_token_id/add_eos_token_id -- the same schema loom::BpeVocab (include/loom/core/bpe_vocab.h)
reads back).

Confirmed directly against Qwen3-0.6B-Base's real tokenizer.json (not assumed): `model.vocab` is a
{piece: id} dict with ids contiguous 0..151642 (151643 entries), `model.merges` is a list of "tok_a tok_b"
strings (already the exact format loom::BpeVocab::load expects, no reformatting needed), and
`added_tokens` (22 entries, e.g. "<|endoftext|>", "<|im_start|>") occupy the next contiguous ids
(151643..151664) with no overlap against `model.vocab`. The checkpoint's own vocab_size (151936, from
config.json) is larger than vocab+added_tokens combined (151665) -- the remaining ids are unused/reserved
embedding-matrix rows with no token text (a common padding-for-tensor-alignment convention); this module
pads the GGUF token list out to vocab_size with empty-string placeholders so BpeVocab::id_to_piece never
throws for the full valid id range, even though a well-trained model should never actually predict one.

Requires: pip install gguf
"""
from gguf import GGUFWriter


def write_bpe_vocab(writer: GGUFWriter, tokenizer_json: dict, vocab_size: int, bos_token_id: int, eos_token_id: int) -> None:
    vocab: dict[str, int] = tokenizer_json["model"]["vocab"]
    merges: list[str] = tokenizer_json["model"]["merges"]
    added_tokens: list[dict] = tokenizer_json.get("added_tokens", [])

    max_id = max([*vocab.values(), *(t["id"] for t in added_tokens)], default=-1)
    token_count = max(max_id + 1, vocab_size)

    tokens = [""] * token_count
    for piece, idx in vocab.items():
        tokens[idx] = piece
    for t in added_tokens:
        tokens[t["id"]] = t["content"]

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(bos_token_id)
    writer.add_eos_token_id(eos_token_id)
