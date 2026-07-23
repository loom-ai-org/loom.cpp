"""Extracts WordPiece vocab data from a real HF BERT-family tokenizer directory (`tokenizer.json` +
`tokenizer_config.json`) and writes it into a GGUF file, `tokenizer.ggml.model`="bert" (a new tag this
project introduces -- see `include/loom/core/wordpiece_vocab.h`'s own doc comment for why: llama.cpp
tags this vocab type via its own internal enum rather than a `tokenizer.ggml.model` string, but "bert"
follows the SAME per-family-tag convention every other type in this schema already uses).

Mirrors `bpe_tokenizer_export.py`'s own approach: reads `tokenizer.json`/`tokenizer_config.json` directly
(no `tokenizers`/`AutoTokenizer` dependency), and applies the exact same `phantom()` transform llama.cpp's
own `conversion/bert.py` uses (confirmed directly against that source) so `WordPieceVocab::encode`'s
plain trie-lookup + "##"-free concatenation reconstructs the reference tokenizer's spacing with no extra
bookkeeping needed on the C++ side: control tokens (CLS/SEP/PAD/UNK/MASK) are left as-is, "##continuation"
tokens have the "##" stripped (bare, no marker), and every other ("starting a new word") token gets the
SentencePiece phantom-space marker U+2581 ("▁") prepended instead.

CLS/SEP reuse the generic BOS/SEP GGUF KVs (gguf-py's own `add_bos_token_id`/`add_sep_token_id`) rather
than inventing dedicated CLS/SEP-only KVs -- matches llama.cpp's own WPM convention. `add_sep_token` has
no dedicated gguf-py writer method in the currently pinned gguf-py version, so it's written directly via
`add_bool` under the same `tokenizer.ggml.add_sep_token` key `WordPieceVocab::load` reads. The normalizer
lowercase/strip_accents flags are likewise project-specific KVs (not part of llama.cpp's own schema),
written the same way.

Requires: pip install gguf
"""
import json
from pathlib import Path

from gguf import GGUFWriter


def write_wordpiece_vocab(writer: GGUFWriter, tokenizer_dir: str) -> None:
    tok_dir = Path(tokenizer_dir)
    tokenizer_json = json.loads((tok_dir / "tokenizer.json").read_text())
    config_path = tok_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    model = tokenizer_json["model"]
    if model["type"] != "WordPiece":
        raise ValueError(f"write_wordpiece_vocab: expected a WordPiece tokenizer.json model, got {model['type']!r}")

    vocab: dict[str, int] = model["vocab"]
    added_tokens: list[dict] = tokenizer_json.get("added_tokens", [])
    max_id = max([*vocab.values(), *(t["id"] for t in added_tokens)], default=-1)
    raw_tokens = [""] * (max_id + 1)
    is_special = [False] * (max_id + 1)
    for piece, idx in vocab.items():
        raw_tokens[idx] = piece
    for t in added_tokens:
        raw_tokens[t["id"]] = t["content"]
        if t.get("special", False):
            is_special[t["id"]] = True

    # llama.cpp conversion/bert.py's own `phantom()` transform, ported verbatim: special/control tokens
    # pass through unchanged; "##continuation" pieces lose the "##" marker (bare, glues onto the
    # preceding piece at decode time); every other piece gets "▁" prepended.
    tokens = []
    for piece, special in zip(raw_tokens, is_special):
        if special:
            tokens.append(piece)
        elif piece.startswith("##"):
            tokens.append(piece[2:])
        else:
            tokens.append("▁" + piece)

    def _token_id(key: str) -> int:
        value = config.get(key)
        if value is None:
            return -1
        piece = value["content"] if isinstance(value, dict) else value
        for t in added_tokens:
            if t["content"] == piece:
                return t["id"]
        return vocab.get(piece, -1)

    writer.add_tokenizer_model("bert")
    writer.add_token_list(tokens)

    unk_id = _token_id("unk_token")
    writer.add_unk_token_id(max(unk_id, 0))

    cls_id = _token_id("cls_token")
    sep_id = _token_id("sep_token")
    pad_id = _token_id("pad_token")
    mask_id = _token_id("mask_token")
    if cls_id >= 0:
        writer.add_bos_token_id(cls_id) # CLS reuses the generic BOS KV, see module docstring
        writer.add_add_bos_token(True)
    if sep_id >= 0:
        writer.add_sep_token_id(sep_id)
        writer.add_bool("tokenizer.ggml.add_sep_token", True)
    if pad_id >= 0:
        writer.add_pad_token_id(pad_id)
    if mask_id >= 0:
        writer.add_mask_token_id(mask_id)

    normalizer = tokenizer_json.get("normalizer") or {}
    lowercase = normalizer.get("lowercase", config.get("do_lower_case", True))
    strip_accents = normalizer.get("strip_accents")
    if strip_accents is None:
        strip_accents = bool(lowercase) # HF's own BertNormalizer default: strip_accents follows lowercase
    writer.add_bool("tokenizer.ggml.normalizer.lowercase", bool(lowercase))
    writer.add_bool("tokenizer.ggml.normalizer.strip_accents", bool(strip_accents))
