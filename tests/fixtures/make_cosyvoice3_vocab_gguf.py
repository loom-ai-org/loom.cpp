#!/usr/bin/env python3
"""A tiny `tokenizer.ggml.model = "cosyvoice3"` file, for tests/ci/test_cosyvoice3_vocab.cpp.

A byte-level BPE with NO merges -- ids 0-255 are the GPT-2 byte pieces, so a text's id count is its
UTF-8 byte count and every chunk budget below is countable by hand -- plus three added tokens
(`<|endoftext|>` 256, the chunk header; `<|endofprompt|>` 257; `[breath]` 258) and three merges: `xy` (259), an NBSP's two bytes (260) and two NBSPs (261). The rule tables are the
exporter's (`loom_exporter.cosyvoice3_tokenizer_export`), restated here small: the real ones are generated
from the reference, inflect and `regex`, none of which a CI fixture may need. The budgets are shrunk to
12/6/4 so a four-sentence text splits.

Requires: pip install gguf
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

P = "tokenizer.ggml.cosyvoice3."


def bytes_to_unicode() -> dict[int, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, cs))


def ranges(chars: str) -> list[int]:
    out: list[int] = []
    for cp in sorted(set(map(ord, chars))):
        if out and out[-1] == cp - 1:
            out[-1] = cp
        else:
            out += [cp, cp]
    return out


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cosyvoice3_vocab_test.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    byte_to_cp = bytes_to_unicode()
    tokens = [chr(byte_to_cp[b]) for b in range(256)] + ["<|endoftext|>", "<|endofprompt|>", "[breath]", "xy",
                                                         "\u00c2\u0142", "\u00c2\u0142\u00c2\u0142"]
    types = [1] * 256 + [3, 3, 3, 1, 1, 1]

    w = GGUFWriter(str(out_path), "loom-cosyvoice3-vocab-fixture")
    w.add_string("loom.architecture", "cosyvoice3_vocab_test")
    w.add_string("model.graph_topology", '{"version": 1, "nodes": []}')
    w.add_tokenizer_model("cosyvoice3")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    # `xy` (259), and an NBSP (bytes C2 A0, byte-level "Âł", 260) and a pair of them (261): whether two
    # NBSPs are one pre-token is what the pair's merge makes visible in the ids.
    w.add_token_merges(["x y", "\u00c2 \u0142", "\u00c2\u0142 \u00c2\u0142"])

    w.add_string(P + "markup_open", "<|")
    w.add_string(P + "markup_close", "|>")
    w.add_array(P + "zh_ranges", [0x4E00, 0x9FFF])
    w.add_array(P + "zh_pre_from", ["\n"])
    w.add_array(P + "zh_pre_to", [""])
    zh = [("²", "平方"), ("³", "立方"), (".", "。"), (" - ", "，"), ("（", ""), ("）", ""), ("【", ""), ("】", ""),
          ("`", ""), ("`", ""), ("——", " ")]
    w.add_array(P + "zh_replace_from", [a for a, _ in zh])
    w.add_array(P + "zh_replace_to", [b for _, b in zh])
    w.add_array(P + "zh_trailing", list("，,、"))
    w.add_string(P + "zh_trailing_to", "。")
    w.add_array(P + "zh_enders", ["。", "？", "！", "；", "：", "、", ".", "?", "!", ";"])
    w.add_array(P + "en_enders", [".", "?", "!", ";", ":"])
    w.add_array(P + "closers", ['"', "”"])
    w.add_string(P + "zh_terminal", "。")
    w.add_string(P + "en_terminal", ".")
    w.add_int32(P + "token_max_n", 12)
    w.add_int32(P + "token_min_n", 6)
    w.add_int32(P + "merge_len", 4)
    # `str.isdigit()`: ASCII and Arabic-Indic digits (Nd), and superscripts (No -- digits to `isdigit`,
    # not to inflect's `\d`).
    arabic = "".join(chr(0x660 + i) for i in range(10))
    w.add_array(P + "digits", list("0123456789" + arabic + "¹²³"))
    w.add_array(P + "decimal_from", list("0123456789" + arabic))
    w.add_array(P + "decimal_value", list(range(10)) * 2)
    w.add_array(P + "num_units", ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])
    w.add_array(P + "num_teens", ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                                  "seventeen", "eighteen", "nineteen"])
    w.add_array(P + "num_tens", ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
                                 "ninety"])
    w.add_array(P + "num_scales", [" ", " thousand", " million", " billion", " trillion", " quadrillion",
                                   " quintillion", " sextillion", " septillion", " octillion", " nonillion",
                                   " decillion"])
    w.add_string(P + "num_hundred", " hundred")
    w.add_string(P + "num_and", "and")
    w.add_string(P + "num_zero", "zero")
    w.add_string(P + "num_one", "one")
    w.add_array(P + "punct_ranges", ranges("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~。，、！？；：“”（）【】—…"))
    w.add_int32(P + "chunk_header", 256)

    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
