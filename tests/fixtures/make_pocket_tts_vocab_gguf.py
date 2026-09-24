#!/usr/bin/env python3
"""A tiny `tokenizer.ggml.model = "pocket_tts"` file, for tests/ci/test_pocket_tts_vocab.cpp.

A SentencePiece Unigram table with all 256 byte-fallback pieces and a handful of words, and the
Pocket-TTS text-path keys with a chunk budget of FOUR tokens, so every branch of the chunking is
reachable in a sentence a person can trace by hand. The real keys come from the reference
(`loom-exporter/loom_exporter/pocket_tts_tokenizer_export.py`); the terminal, weak and closer sets
here are the reference's own, and the case and digit tables are cut down to what the test uses.

Requires: pip install gguf numpy
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

NORMAL, UNKNOWN, CONTROL, BYTE = 1, 2, 3, 6
P = "tokenizer.ggml.pocket_tts."

WORDS = ["▁", "▁Hi", "▁hi", "▁the", "▁The", ".", ",", "!", "▁A", "▁b",
         "▁SSa", "▁1", "2", "▁One"]


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pocket_tts_vocab_test.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pieces = [("<unk>", 0.0, UNKNOWN), ("<s>", 0.0, CONTROL), ("</s>", 0.0, CONTROL)]
    pieces += [(f"<0x{b:02X}>", 0.0, BYTE) for b in range(256)]            # ids 3 .. 258
    pieces += [(w, -1.0, NORMAL) for w in WORDS]                             # ids 259 ..
    w = GGUFWriter(str(out_path), "loom-pocket-tts-vocab-fixture")
    w.add_string("loom.architecture", "pocket_tts_vocab_test")
    w.add_string("model.graph_topology", '{"version": 1, "nodes": []}')
    w.add_tokenizer_model("pocket_tts")
    w.add_token_list([p for p, _, _ in pieces])
    w.add_token_scores([s for _, s, _ in pieces])
    w.add_token_types([t for _, _, t in pieces])
    w.add_unk_token_id(0)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_add_space_prefix(True)
    w.add_remove_extra_whitespaces(False)
    w.add_bool("tokenizer.ggml.byte_fallback", True)
    w.add_array(P + "replace_from", ["\n", "\r", "  "])
    w.add_array(P + "replace_to", [" ", " ", " "])
    w.add_array(P + "terminal", list(".!?…"))
    w.add_array(P + "weak", list(",;:-–—"))
    w.add_array(P + "closers", list("\"')]»”’"))
    w.add_array(P + "full_stop", ["."])
    w.add_array(P + "upper_from", ["h", "t", "ß", "o"])
    w.add_array(P + "upper_to", ["H", "T", "SS", "O"])
    w.add_array(P + "digits", list("0123456789"))
    ids = {p: i for i, (p, _, _) in enumerate(pieces)}
    w.add_array(P + "sentence_end_ids", [ids["."], ids["!"]])
    w.add_array(P + "clause_end_ids", [ids[","]])
    w.add_int32(P + "max_tokens_per_chunk", 4)
    w.add_int32(P + "chunk_header_short", 1)
    w.add_int32(P + "chunk_header_long", 2)
    w.add_int32(P + "short_chunk_max_words", 4)
    w.add_bool(P + "capitalize_first_letter", True)
    w.add_bool(P + "append_terminal_punctuation", True)
    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
