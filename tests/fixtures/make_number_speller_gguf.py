"""A tiny Unigram vocabulary of single characters carrying a number speller, for
tests/ci/test_number_speller.cpp (ADR-059).

The speller's keys are what loom-exporter's `number_normalizer_export` writes from transformers'
`EnglishNumberNormalizer`: its English word tables and currency names, the pattern's symbol chain, and
Python's own `\\d` / `\\w` classes as codepoint tables. The tables are computed here from this
interpreter's `unicodedata` exactly as the exporter computes them; the words are restated.

Requires: pip install gguf numpy
"""
import sys
import unicodedata
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

NORMAL, UNKNOWN, CONTROL = 1, 2, 3
P = "tokenizer.ggml.numbers."

ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TEENS = ["", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SCALES = ["", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion", "sextillion",
          "septillion", "octillion", "nonillion", "decillion"]
CHAIN = ["-", "$", "€", "£", "¢", "¥", "₹", "₽", "฿", "₺", "₴", "₣", "₡", "₱", "₪", "₮", "₩", "₦", "₫", "﷼"]
CURRENCY = [("$", " dollars"), ("€", " euros"), ("£", " pounds"), ("¢", " cents"), ("¥", " japanese yen"),
            ("﷼", " saudi riyal"), ("₹", " indian rupees"), ("₽", " russian rubles"), ("฿", " thai baht"),
            ("₺", " turkish liras"), ("₴", " ukrainian hryvnia"), ("₣", " swiss francs"),
            ("₡", " costa rican colon"), ("₱", " philippine peso"), ("₪", " israeli shekels"),
            ("₮", " mongolian tögrög"), ("₩", " south korean won"), ("₦", " nigerian naira"),
            ("₫", " vietnamese Đồng")]


def digit_zeros():
    return [cp for cp in range(sys.maxunicode + 1)
            if chr(cp).isdecimal() and unicodedata.decimal(chr(cp)) == 0]


def word_ranges():
    ranges, start = [], None
    for cp in range(sys.maxunicode + 2):
        inside = cp <= sys.maxunicode and (chr(cp).isalnum() or cp == 0x5F)
        if inside and start is None:
            start = cp
        elif not inside and start is not None:
            ranges += [start, cp - 1]
            start = None
    return ranges


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("number_speller_test.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pieces = [("<s>", 0.0, CONTROL), ("<pad>", 0.0, CONTROL), ("</s>", 0.0, CONTROL), ("<unk>", 0.0, UNKNOWN)]
    pieces += [(c, -1.0 - i * 0.01, NORMAL) for i, c in enumerate("▁abcdefghijklmnopqrstuvwxyz.,")]
    w = GGUFWriter(str(out_path), "loom-number-speller-fixture")
    w.add_string("loom.architecture", "number_speller_test")
    w.add_string("model.graph_topology", '{"version": 1, "nodes": []}')
    w.add_tokenizer_model("t5")
    w.add_token_list([p for p, _, _ in pieces])
    w.add_token_scores([s for _, s, _ in pieces])
    w.add_token_types([t for _, _, t in pieces])
    w.add_unk_token_id(3)
    w.add_eos_token_id(2)
    w.add_add_space_prefix(True)
    w.add_remove_extra_whitespaces(True)
    w.add_string(P + "scheme", "english_number_normalizer")
    w.add_array(P + "ones", ONES)
    w.add_array(P + "teens", TEENS)
    w.add_array(P + "tens", TENS)
    w.add_array(P + "scales", SCALES)
    w.add_array(P + "symbol_chain", CHAIN)
    w.add_array(P + "currency_symbols", [s for s, _ in CURRENCY])
    w.add_array(P + "currency_names", [n for _, n in CURRENCY])
    w.add_array(P + "digit_zeros", digit_zeros())
    w.add_array(P + "word_ranges", word_ranges())
    # A file with no tensors has no weight buffer to allocate, and GgufModel::load refuses it.
    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
