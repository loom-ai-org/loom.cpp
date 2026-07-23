"""Converts SupertonicTTS v2's real `TextVectorizer` vocabulary (models/modules/text_vectorizer.py) into a
standalone GGUF `loom::SupertonicTextVectorizer` (include/loom/core/supertonic_text_vectorizer.h) loads.

Unlike every other convert_supertonic_*.py script, this one converts no model weights at all -- the real
`TextVectorizer` has no learned parameters, just a static `assets/onnx/unicode_indexer.json` asset (a flat
65536-entry array: `indexer[codepoint]` = vocab id, or -1 if unsupported) plus a fixed, hand-written
preprocessing pipeline (ported natively in supertonic_text_vectorizer.cpp; the JSON asset is the only
per-checkpoint DATA this needs). `tokenizer.ggml.model`="supertonic" follows the same per-family-tag
convention every other vocab schema in this project uses, though this isn't a generic HF tokenizer family
(it doesn't go through tools/loom_mil_compiler/'s tokenizer_detect.py auto-detection at all).

Usage: python3 convert_supertonic_text_vectorizer.py <supertonic-tts repo root> <out_path.gguf>
"""
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter


def convert(repo_root: str, out_path: str) -> None:
    indexer_path = Path(repo_root) / "assets" / "onnx" / "unicode_indexer.json"
    table: list[int] = json.loads(indexer_path.read_text())

    w = GGUFWriter(out_path, "loom-supertonic-text-vectorizer")
    w.add_string("loom.architecture", "supertonic_text_vectorizer")
    # No model graph -- a tokenizer-only GGUF, same convention as every other vocab-only fixture/export
    # in this project (GgufModel::load still requires the "model.graph_topology" KV to be present).
    w.add_string("model.graph_topology", '{"version": 1, "nodes": []}')

    w.add_tokenizer_model("supertonic")
    w.add_array("tokenizer.ggml.supertonic.codepoint_to_id", table)

    # GgufModel::load() allocates a backend buffer sized to the meta context's tensors -- a vocab-only
    # GGUF with zero tensors hits a ggml_backend edge case, so (like every other vocab-only fixture in
    # this project) a small placeholder weight is included even though nothing ever reads it.
    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}: {sum(1 for v in table if v >= 0)} mapped codepoints "
          f"(vocab ids 0..{max(table)})")


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
