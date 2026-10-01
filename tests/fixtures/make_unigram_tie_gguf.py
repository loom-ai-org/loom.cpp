"""Two tiny Unigram vocabularies on which an exact TIE is broken differently by SentencePiece's
arithmetic and by doubles, for tests/ci/test_unigram_scoring.cpp (ADR-060).

`g`+`gg` and `gg`+`g` score the same in real arithmetic. SentencePiece stores path scores as float and
compares piece candidates in double, and on `yyxyxggg` with these scores that picks `gg`, `g`; doubles
throughout pick `g`, `gg`. Found by searching random scores against the real `sentencepiece` library
(0.2.1), which encodes this text as `▁ y y x y x gg g`. One file declares
`tokenizer.ggml.unigram_scoring = "sentencepiece"`, the other declares nothing.

Requires: pip install gguf numpy
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

NORMAL, UNKNOWN, CONTROL = 1, 2, 3
PIECES = [("<pad>", 0.0, CONTROL), ("</s>", 0.0, CONTROL), ("<unk>", 0.0, UNKNOWN),
          ("▁", -1.3730874061584473, NORMAL), ("x", -10.241003036499023, NORMAL),
          ("y", -8.812788009643555, NORMAL), ("g", -9.635525703430176, NORMAL),
          ("gg", -7.877923965454102, NORMAL)]


def write(path: Path, sentencepiece_scoring: bool) -> None:
    w = GGUFWriter(str(path), "loom-unigram-tie-fixture")
    w.add_string("loom.architecture", "unigram_tie_test")
    w.add_string("model.graph_topology", '{"version": 1, "nodes": []}')
    w.add_tokenizer_model("t5")
    w.add_token_list([p for p, _, _ in PIECES])
    w.add_token_scores([float(np.float32(s)) for _, s, _ in PIECES])
    w.add_token_types([t for _, _, t in PIECES])
    w.add_unk_token_id(2)
    w.add_eos_token_id(1)
    w.add_add_space_prefix(True)
    w.add_remove_extra_whitespaces(True)
    if sentencepiece_scoring:
        w.add_string("tokenizer.ggml.unigram_scoring", "sentencepiece")
    # A file with no tensors has no weight buffer to allocate, and GgufModel::load refuses it.
    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    write(out_dir / "unigram_tie_sentencepiece.gguf", True)
    write(out_dir / "unigram_tie_doubles.gguf", False)
    print(f"wrote 2 files to {out_dir}")


if __name__ == "__main__":
    main()
