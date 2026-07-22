"""Extracts byte-level-BPE vocab data from a real HF tokenizer directory (`tokenizer.json` +
`tokenizer_config.json`) and writes it into a GGUF file using llama.cpp's own "tokenizer.ggml.*" schema
for a "gpt2"-style vocab -- the same schema `tools/convert_qwen3/qwen3_tokenizer.py`'s `write_bpe_vocab`
writes and `loom::BpeVocab` (include/loom/core/bpe_vocab.h) reads back, generalized to also handle:

- tokenizer.json schema variants where `model.merges` is a list of `[a, b]` pairs (LFM2's own
  tokenizer.json) rather than pre-joined "a b" strings (Qwen3's own tokenizer.json) -- both normalized to
  the "a b" format `loom::BpeVocab::load` expects.
- an explicit `pre_type` ("qwen2" default, or "llama3" for LFM2's grouped-up-to-3-digit pretokenizer
  regex variant, `\\p{N}{1,3}` -- see bpe_vocab.h's own doc comment) written as `tokenizer.ggml.pre`,
  dispatched on by `loom::BpeVocab` at load time. Not auto-detected from the regex string: per
  EXPORT-BACKLOG.md item 4's own plan, tokenizer family/variant selection is a bounded, one-time choice
  made by each model's own export script, not a generic regex-sniffing framework.
- `tokenizer_config.json`'s `add_bos_token`, needed because LFM2 (unlike Qwen3) prepends a BOS token to
  every encoded sequence per its own `TemplateProcessing` post-processor.

Requires: pip install gguf
"""
import json
from pathlib import Path

from gguf import GGUFWriter


def write_bpe_vocab(writer: GGUFWriter, tokenizer_dir: str, pre_type: str = "qwen2") -> None:
    tok_dir = Path(tokenizer_dir)
    tokenizer_json = json.loads((tok_dir / "tokenizer.json").read_text())
    config_path = tok_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    vocab: dict[str, int] = tokenizer_json["model"]["vocab"]
    raw_merges: list = tokenizer_json["model"]["merges"]
    # Normalize both tokenizer.json merges schemas ("a b" strings, or [a, b] pair lists) to "a b" strings.
    merges = [m if isinstance(m, str) else " ".join(m) for m in raw_merges]
    added_tokens: list[dict] = tokenizer_json.get("added_tokens", [])

    max_id = max([*vocab.values(), *(t["id"] for t in added_tokens)], default=-1)
    tokens = [""] * (max_id + 1)
    for piece, idx in vocab.items():
        tokens[idx] = piece
    for t in added_tokens:
        tokens[t["id"]] = t["content"]

    def _token_id(value) -> int:
        # tokenizer_config.json's bos_token/eos_token are either a bare string or an AddedToken-style dict.
        if value is None:
            return -1
        piece = value["content"] if isinstance(value, dict) else value
        for t in added_tokens:
            if t["content"] == piece:
                return t["id"]
        return vocab.get(piece, -1)

    bos_token_id = _token_id(config.get("bos_token"))
    eos_token_id = _token_id(config.get("eos_token"))

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre(pre_type)
    writer.add_token_list(tokens)
    writer.add_token_merges(merges)
    if bos_token_id >= 0:
        writer.add_bos_token_id(bos_token_id)
    if eos_token_id >= 0:
        writer.add_eos_token_id(eos_token_id)
    writer.add_add_bos_token(bool(config.get("add_bos_token", False)))
