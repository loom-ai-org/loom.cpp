#!/usr/bin/env python3
"""Generates a small CTC-character-vocabulary GGUF fixture for test_ctc_vocab.cpp.

The table is `facebook/hubert-large-ls960-ft`'s real one truncated to the letters the test spells, and
it keeps that checkpoint's own layout exactly: `<pad>` at 0 -- which is the BLANK, not the last class --
`<s>`/`</s>`/`<unk>` next, then `|` as the word delimiter, then the letters.

The one thing deliberately NOT copied is the multilingual checkpoint's arrangement, where the blank is
spelled `<s>` and a separate unused `<pad>` sits beside it. That is covered by the export-side test
(`loom-exporter/tests/ci/test_ctc_asr_export.py`), because it is a question about which id the WRITER
picks; by the time it reaches here it is one number in one KV.

Requires: pip install gguf
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

TOPOLOGY_JSON = '{"version": 1, "nodes": []}'
TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "|", "E", "T", "A", "O", "N", "H", "L", "W"]
BLANK_ID = 0
UNK_ID = 3
WORD_DELIMITER_ID = 4

TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_CONTROL = 3


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ctc_vocab_test.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = GGUFWriter(str(out_path), "loom-ctc-vocab-fixture")
    w.add_string("loom.architecture", "ctc_vocab_test")
    w.add_string("model.graph_topology", TOPOLOGY_JSON)

    w.add_tokenizer_model("ctc")
    w.add_token_list(TOKENS)
    w.add_token_types([TOKEN_TYPE_CONTROL if i in (BLANK_ID, UNK_ID) else TOKEN_TYPE_NORMAL
                       for i in range(len(TOKENS))])
    w.add_pad_token_id(BLANK_ID)
    w.add_unk_token_id(UNK_ID)
    w.add_uint32("tokenizer.ggml.word_delimiter_id", WORD_DELIMITER_ID)

    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
