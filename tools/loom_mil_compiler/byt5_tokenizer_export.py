"""Extracts vocab data for a real HF ByT5-family tokenizer directory (`tokenizer_config.json` only -- ByT5
has no `tokenizer.json`/`tokenizer.model` at all, see tokenizer_detect.py's own doc comment for how this
family gets detected in the first place) and writes it into a GGUF file, `tokenizer.ggml.model`="byt5".

ByT5 tokenizes raw UTF-8 bytes directly with a fixed, computable scheme (see
include/loom/core/byte_vocab.h's own doc comment) -- confirmed directly against a real
`transformers.ByT5Tokenizer` instance's actual saved `tokenizer_config.json`, not assumed from its
docstring (which describes a different, non-matching sentinel-ordering scheme). Concretely:
`added_tokens_decoder` (a real, on-disk id -> {"content": ...} mapping) has entries "0"/"1"/"2" for
pad/eos/unk (fixed positions, hardcoded by every real `ByT5Tokenizer.__init__`, not a per-checkpoint
choice) and one entry per T5-style span-corruption sentinel ("<extra_id_N>") at ids 259, 260, ...
(sequential, right after the 256-entry byte range) -- NOT reversed/counted-from-the-end as the upstream
docstring claims. The top-level `extra_ids` config field is unreliable (`ByT5Tokenizer.__init__` always
passes `extra_ids=0` to its base class, regardless of the real sentinel count) -- the real count is
derived from `added_tokens_decoder` directly instead.

Byte-range piece text (ids 3..258) is deliberately left as empty placeholders in the written token list --
`loom::ByteVocab` computes those arithmetically at load time rather than storing them (see that class's
own doc comment for why storing them as normal GGUF token strings would silently corrupt any byte >=
0x80).

Requires: pip install gguf
"""
import json
from pathlib import Path

from gguf import GGUFWriter

_PAD_ID, _EOS_ID, _UNK_ID = 0, 1, 2
_BYTE_OFFSET = 3
_BYTE_RANGE_SIZE = 256


def write_byt5_vocab(writer: GGUFWriter, tokenizer_dir: str) -> None:
    tok_dir = Path(tokenizer_dir)
    config = json.loads((tok_dir / "tokenizer_config.json").read_text())
    if config.get("tokenizer_class") != "ByT5Tokenizer":
        raise ValueError(f"write_byt5_vocab: expected tokenizer_class=='ByT5Tokenizer', got "
                          f"{config.get('tokenizer_class')!r}")

    decoder: dict[str, object] = config["added_tokens_decoder"]

    def _content(id_str: str) -> str:
        value = decoder[id_str]
        return value["content"] if isinstance(value, dict) else value

    pad_str = _content(str(_PAD_ID))
    eos_str = _content(str(_EOS_ID))
    unk_str = _content(str(_UNK_ID))

    # Every added_tokens_decoder entry at an id >= the byte range's end is a sentinel -- ByT5 never adds
    # any other kind of token beyond pad/eos/unk (ids 0-2) and sentinels.
    sentinel_ids = sorted(int(k) for k in decoder if int(k) >= _BYTE_OFFSET + _BYTE_RANGE_SIZE)
    if sentinel_ids and sentinel_ids != list(range(sentinel_ids[0], sentinel_ids[0] + len(sentinel_ids))):
        raise NotImplementedError(
            f"write_byt5_vocab: sentinel ids {sentinel_ids} are not contiguous -- unexpected ByT5 "
            "tokenizer layout, not supported")

    vocab_size = _BYTE_OFFSET + _BYTE_RANGE_SIZE + len(sentinel_ids)
    tokens = [""] * vocab_size
    tokens[_PAD_ID] = pad_str
    tokens[_EOS_ID] = eos_str
    tokens[_UNK_ID] = unk_str
    for sid in sentinel_ids:
        tokens[sid] = _content(str(sid))

    writer.add_tokenizer_model("byt5")
    writer.add_token_list(tokens)
    writer.add_pad_token_id(_PAD_ID)
    writer.add_eos_token_id(_EOS_ID)
    writer.add_unk_token_id(_UNK_ID)
