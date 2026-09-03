#!/usr/bin/env python3
"""Generates three small "phonemes"-family GGUF fixtures for test_phoneme_vocab.cpp.

Deliberately NOT a real checkpoint's table -- piper's is 159 entries and Kokoro/StyleTTS2/Matcha carry
178 each, and none of them makes an expected id readable at a glance. What this covers is everything
around the table, which is where the behaviour that is easy to get wrong lives: longest-match over
multi-codepoint IPA symbols, the per-checkpoint assembly, the -1 sentinel, and the skip-don't-raise rule
for a symbol outside the inventory. A twelve-entry synthetic table makes each of those checkable against
a number written in this file rather than against a recorded oracle.

Three files, because the properties that matter are properties of DIFFERENT declared assemblies:

  phoneme_vocab_test.gguf       piper's shape -- bos/eos/blank all present and `interleave_blank`, the
                                [BOS, p1, blank, ..., pn, blank, EOS] build. Also holds the two
                                multi-codepoint symbols and one `<unusedN>` id gap.
  phoneme_vocab_bare.gguf       StyleTTS2's shape -- bos only, `eos_id`/`blank_id` both the -1 SENTINEL
                                and no interleave. A vocabulary that treated -1 as an id would append it
                                here, and it reaches the engine as an out-of-range GET_ROWS rather than
                                as an error. Also carries a duplicate symbol, for the first-id-wins rule.
  phoneme_vocab_no_tokens.gguf  the tag with no `tokenizer.ggml.tokens` -- malformed rather than merely
                                new, and the one input `PhonemeVocab::load` is supposed to REFUSE.

Requires: pip install gguf
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

TOPOLOGY_JSON = '{"version": 1, "nodes": []}'

# Id-indexed, exactly as the exporter writes it. Ids 0-2 are the assembly, so they carry the names a real
# table gives them; id 10 is a GAP, filled with the same `<unusedN>` the exporter uses -- a name no
# phonemizer emits, rather than the empty string, which would make every gap collide on one lookup key.
TOKENS = [
    "<blank>",    # 0
    "<bos>",      # 1
    "<eos>",      # 2
    " ",          # 3
    "a",          # 4
    "ɪ",          # 5
    "aɪ",         # 6  two codepoints, one symbol -- the diphthong a shortest-first scan would split
    "t",          # 7
    "ʃ",          # 8
    "t͡ʃ",         # 9  three codepoints, five bytes, and it BEGINS with id 7's symbol
    "<unused10>", # 10 an id gap, named the way the exporter names one
    "h",          # 11
]


def _write(out_path: Path, tokens, *, bos: int, eos: int, blank: int, interleave: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out_path), "loom-phoneme-vocab-fixture")
    w.add_string("loom.architecture", "phoneme_vocab_test")
    w.add_string("model.graph_topology", TOPOLOGY_JSON)

    w.add_tokenizer_model("phonemes")
    if tokens is not None:
        w.add_array("tokenizer.ggml.tokens", tokens)
        w.add_int32("tokenizer.ggml.phoneme.bos_id", bos)
        w.add_int32("tokenizer.ggml.phoneme.eos_id", eos)
        w.add_int32("tokenizer.ggml.phoneme.blank_id", blank)
        w.add_bool("tokenizer.ggml.phoneme.interleave_blank", interleave)

    # GgufModel::load() sizes a backend buffer from the meta context's tensors; a GGUF with zero tensors
    # hits a ggml_backend edge case, so a placeholder nothing reads is included -- same convention as
    # every other vocab-only fixture here.
    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> None:
    args = sys.argv[1:]
    piper_path = Path(args[0]) if args else Path("phoneme_vocab_test.gguf")
    bare_path = (Path(args[1]) if len(args) > 1
                 else piper_path.with_name("phoneme_vocab_bare.gguf"))
    no_tokens_path = (Path(args[2]) if len(args) > 2
                      else piper_path.with_name("phoneme_vocab_no_tokens.gguf"))

    _write(piper_path, TOKENS, bos=1, eos=2, blank=0, interleave=True)

    # Same symbols, one appended duplicate: "a" again at id 12, where 4 already holds it. The export
    # writes one id per symbol, so this can only come of a table built from a non-injective map -- and
    # which id then wins has to be decided by the loader rather than by array order.
    _write(bare_path, TOKENS + ["a"], bos=0, eos=-1, blank=-1, interleave=False)

    _write(no_tokens_path, None, bos=0, eos=-1, blank=-1, interleave=False)


if __name__ == "__main__":
    main()
