#!/usr/bin/env python3
"""A small `tokenizer.ggml.model = "soprano"` file, for tests/ci/test_soprano_vocab.cpp.

The rule tables are the exporter's (`loom_exporter.soprano_tokenizer_export`), restated VERBATIM: they
are the reference's own pattern strings, and a CI fixture may not import the reference, inflect or
unidecode to read them. Two things are shrunk. `unidecode`'s table is the handful of codepoints the
test's texts use (the real one is 41,379). The vocabulary is a character table -- `[UNK]`, `[TEXT]`,
`[START]`, `[STOP]`, then one piece per character the cleaned text can hold -- plus two merges, `t h`
and `th e`, so a test can see a merge happen.

Requires: pip install gguf numpy
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

P = "tokenizer.ggml.soprano."

RULES = [
    ("num_prefix_re", "#\\d", 0), ("num_suffix_re", "\\b\\d+(K|M|B|T)\\b", 1),
    ("num_letter_split_re", "(\\d[a-z]|[a-z]\\d)", 1), ("comma_number_re", "(\\d[\\d\\,]+\\d)", 0),
    ("date_re", "(^|[^/])(\\d\\d?[/-]\\d\\d?[/-]\\d\\d(?:\\d\\d)?)($|[^/])", 0),
    ("phone_number_re", "(\\(?\\d{3}\\)?[-.\\s]\\d{3}[-.\\s]?\\d{4})", 0),
    ("time_re", "(\\d\\d?:\\d\\d(?::\\d\\d)?)", 0), ("pounds_re", "£([\\d\\,]*\\d+)", 0),
    ("dollars_re", "\\$([\\d\\.\\,]*\\d+)", 0), ("decimal_number_re", "(\\d+(?:\\.\\d+)+)", 0),
    ("multiply_re", "(\\d\\s?\\*\\s?\\d)", 0), ("divide_re", "(\\d\\s?/\\s?\\d)", 0),
    ("add_re", "(\\d\\s?\\+\\s?\\d)", 0), ("subtract_re", "(\\d?\\s?-\\s?\\d)", 0),
    ("fraction_re", "(\\d+(?:/\\d+)+)", 0), ("ordinal_re", "\\d+(st|nd|rd|th)", 0), ("number_re", "\\d+", 0),
    ("link_header_re", "(https?://)", 0), ("dash_re", "(. - .)", 0), ("dot_re", "([A-Z]\\.[A-Z])", 1),
    ("parentheses_re", "[\\(\\[\\{].*[\\)\\]\\}](.|$)", 0), ("camelcase_re", "\\b([A-Z][a-z]*)+\\b", 0),
    ("date_split", "[./-]", 0), ("phone_non_digit", "\\D", 0), ("paren_open", "[\\(\\[\\{]", 0),
    ("paren_close_inner", "[\\)\\]\\}][^$.!?,]", 0), ("paren_close", "[\\)\\]\\}]", 0),
    ("camel_part", "[A-Z][a-z]*", 0), ("unknown_chars", "[^A-Za-z !\\$%&'\\*\\+,-./0123456789<>\\?_]", 0),
    ("unknown_symbols", "[<>/_+]", 0), ("whitespace", "\\s+", 0), ("space_before_punct", " [.\\?!,]", 0),
    ("ellipsis", "\\.\\.\\.+", 0), ("commas", ",+", 0), ("periods", "[\\.,]*\\.[\\.,]*", 0),
    ("exclamations", "[\\.,!]*![\\.,!]*", 0), ("questions", "[\\.,!\\?]*\\?[\\.,!\\?]*", 0),
    ("ellipsis_back", "\\[ELLIPSIS\\]", 0), ("triple_letters", "(\\w)\\1{2,}", 0),
    ("split_newlines", "\\n\\n+", 0), ("split_whitespace", "\\s+", 0), ("split_quotes", "[“”]", 0),
    ("split_empty", "^[\\s\\.,;:!?]*$", 0), ("inflect_non_digit", "\\D", 0),
    ("inflect_whitespaces_comma", "\\s+,", 0), ("inflect_comma_word", ", (\\S+)\\s+\\Z", 0),
    ("inflect_whitespaces", "\\s+", 0),
]
TEMPLATES = [
    ("phone_non_digit", ""), ("paren_open", ", "), ("paren_close_inner", ", "), ("paren_close", ""),
    ("unknown_chars", ""), ("unknown_symbols", ""), ("whitespace", " "), ("ellipsis", "[ELLIPSIS]"),
    ("commas", ","), ("periods", "."), ("exclamations", "!"), ("questions", "?"), ("ellipsis_back", "..."),
    ("split_newlines", "\n"), ("split_whitespace", " "), ("split_quotes", '"'), ("pounds_re", "\\1 pounds"),
]
ABBREVIATIONS = [(f"\\b{a}\\.", 1, r) for a, r in [
    ("mrs", "misess"), ("ms", "miss"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"), ("co", "company"),
    ("jr", "junior"), ("maj", "major"), ("gen", "general"), ("drs", "doctors"), ("rev", "reverend"),
    ("lt", "lieutenant"), ("hon", "honorable"), ("sgt", "sergeant"), ("capt", "captain"), ("esq", "esquire"),
    ("ltd", "limited"), ("col", "colonel"), ("ft", "fort")]] + [(f"\\b{a}\\b", 0, r) for a, r in [
    ("Hz", "hertz"), ("kHz", "kilohertz"), ("KBs", "kilobytes"), ("KB", "kilobyte"), ("MBs", "megabytes"),
    ("MB", "megabyte"), ("GBs", "gigabytes"), ("GB", "gigabyte"), ("TBs", "terabytes"), ("TB", "terabyte"),
    ("APIs", "a p i's"), ("API", "a p i"), ("CLIs", "c l i's"), ("CLI", "c l i"), ("CPUs", "c p u's"),
    ("CPU", "c p u"), ("GPUs", "g p u's"), ("GPU", "g p u"), ("Ave", "avenue"), ("etc", "et cetera"),
    ("Mon", "monday"), ("Tues", "tuesday"), ("Wed", "wednesday"), ("Thurs", "thursday"), ("Fri", "friday"),
    ("Sat", "saturday"), ("Jan", "january"), ("Feb", "february"), ("Mar", "march"), ("Apr", "april"),
    ("Aug", "august"), ("Sept", "september"), ("Oct", "october"), ("Nov", "november"), ("Dec", "december"),
    ("and/or", "and or")]]
SPECIAL = [("@", " at "), ("&", " and "), ("%", " percent "), (":", "."), (";", ","), ("\\+", " plus "),
           ("\\\\", " backslash "), ("~", " about "), ("(^| )<3", " heart "), ("<=", " less than or equal to "),
           (">=", " greater than or equal to "), ("<", " less than "), (">", " greater than "),
           ("=", " equals "), ("/", " slash "), ("_", " "), ("\\*", " ")]
UNIDECODE = {"£": "PS", "é": "e", "ï": "i", "“": '"', "”": '"', "—": "--"}


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("soprano_vocab_test.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chars = " !$%&'*,-.0123456789?abcdefghijklmnopqrstuvwxyz"
    tokens = ["[UNK]", "[TEXT]", "[START]", "[STOP]"] + list(chars) + ["th", "the"]
    types = [3, 3, 3, 3] + [1] * (len(tokens) - 4)

    w = GGUFWriter(str(out_path), "soprano")
    w.add_string("loom.architecture", "soprano_vocab_test")
    w.add_string("model.graph_topology", '{"version": 1, "nodes": []}')
    w.add_tokenizer_model("soprano")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(["t h", "th e"])
    w.add_unk_token_id(0)
    w.add_int32(P + "stop_id", 3)
    w.add_int32(P + "text_id", 1)
    w.add_int32(P + "start_id", 2)
    w.add_array(P + "rule_names", [n for n, _, _ in RULES])
    w.add_array(P + "rule_patterns", [p for _, p, _ in RULES])
    w.add_array(P + "rule_icase", [ic for _, _, ic in RULES])
    w.add_array(P + "template_names", [n for n, _ in TEMPLATES])
    w.add_array(P + "template_values", [v for _, v in TEMPLATES])
    for key, rows in (("preunicode", [("—", 0, " - ")]), ("abbreviations", ABBREVIATIONS),
                      ("special", [(p, 0, r) for p, r in SPECIAL])):
        w.add_array(P + key + "_patterns", [p for p, _, _ in rows])
        w.add_array(P + key + "_icase", [ic for _, ic, _ in rows])
        w.add_array(P + key + "_repl", [r for _, _, r in rows])
    w.add_array(P + "inflect_unit", ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])
    w.add_array(P + "inflect_teen", ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                                     "seventeen", "eighteen", "nineteen"])
    w.add_array(P + "inflect_ten", ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
                                    "ninety"])
    w.add_array(P + "inflect_mill", [" ", " thousand", " million", " billion", " trillion", " quadrillion",
                                     " quintillion", " sextillion", " septillion", " octillion", " nonillion",
                                     " decillion"])
    w.add_array(P + "inflect_ordinal_from", ["ty", "one", "two", "three", "five", "eight", "nine", "twelve"])
    w.add_array(P + "inflect_ordinal_to", ["tieth", "first", "second", "third", "fifth", "eighth", "ninth",
                                           "twelfth"])
    w.add_array(P + "inflect_nth_suffixes", ["nd", "rd", "st", "th"])
    w.add_int32(P + "desired_length", 1)
    w.add_int32(P + "max_length", 300)
    w.add_int32(P + "min_length", 30)
    items = sorted(UNIDECODE.items(), key=lambda kv: ord(kv[0]))
    offsets = [0]
    for _, v in items:
        offsets.append(offsets[-1] + len(v))
    w.add_array(P + "unidecode_cps", [ord(k) for k, _ in items])
    w.add_array(P + "unidecode_offsets", offsets)
    w.add_string(P + "unidecode_text", "".join(v for _, v in items))
    # `GgufModel::load` allocates a weights buffer, which a file with no tensors cannot have.
    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
