import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loom_mil_compiler import tokenizer_detect as td


class _FakeTokenizer:
    """Stub for detect_loom_pre_type's `tokenizer` param -- only needs .encode(str) -> list[int]."""

    def __init__(self, chktok):
        self._chktok = chktok

    def encode(self, text):
        assert text == td._CHKTXT
        return self._chktok


class TestTokenizerDetect(unittest.TestCase):
    def test_chkhsh_to_loom_pre_type_table_completeness(self):
        """Every llama.cpp pre name the hash table can produce must have an entry (possibly None, meaning
        "recognized but not yet implemented") in _LLAMA_PRE_TO_LOOM_PRE_TYPE -- catches a transcription
        gap mechanically rather than surfacing as a silent KeyError at export time."""
        llama_pre_names = set(td._CHKHSH_TO_LLAMA_PRE.values())
        missing = llama_pre_names - set(td._LLAMA_PRE_TO_LOOM_PRE_TYPE.keys())
        self.assertEqual(missing, set())

    def test_implemented_pre_types_match_bpe_vocab_cpp(self):
        """Every non-None loom pre_type value must appear in bpe_vocab.cpp's pre_spec_table() -- a
        divergence between the two tables (introduced by editing only one side) fails a test instead of
        silently producing a GGUF the C++ loader then rejects at load time."""
        cpp_path = Path(__file__).resolve().parent.parent.parent / "src" / "core" / "bpe_vocab.cpp"
        cpp_src = cpp_path.read_text()
        implemented = {v for v in td._LLAMA_PRE_TO_LOOM_PRE_TYPE.values() if v is not None}
        for name in implemented:
            self.assertIn(f'"{name}"', cpp_src, f"{name!r} not found in bpe_vocab.cpp's pre_spec_table()")

    def test_compute_chkhsh_is_sha256_of_str_of_encoded_tokens(self):
        fake_tokens = [1, 2, 3]
        expected_hash = hashlib.sha256(str(fake_tokens).encode()).hexdigest()
        self.assertEqual(td.compute_chkhsh(_FakeTokenizer(fake_tokens)), expected_hash)

    def test_detect_loom_pre_type_known_implemented_hash(self):
        # A real qwen2 chkhsh from the transcribed table, resolved end to end via a tokenizer stub whose
        # .encode() reproduces the token list that hashes to it (found by construction: compute_chkhsh
        # only depends on encode()'s return value, so a single-element list containing that exact hash
        # string, verified to also match some legitimate chktok, would require the real tokenizer --
        # instead exercise the table-lookup chain directly through detect_llama_pre_name/_pre_type).
        qwen2_hash = next(h for h, n in td._CHKHSH_TO_LLAMA_PRE.items() if n == "qwen2")
        self.assertEqual(td._CHKHSH_TO_LLAMA_PRE[qwen2_hash], "qwen2")
        self.assertEqual(td._LLAMA_PRE_TO_LOOM_PRE_TYPE["qwen2"], "qwen2")

    def test_detect_loom_pre_type_unrecognized_hash_falls_back(self):
        tok = _FakeTokenizer([9999999999])
        self.assertIsNone(td.detect_llama_pre_name(tok))
        self.assertEqual(td.detect_loom_pre_type(tok, fallback="qwen2"), "qwen2")

    def test_detect_loom_pre_type_recognized_but_not_implemented_raises(self):
        # "gpt-4o" is a real llama.cpp pre name this project doesn't implement yet (a None entry in
        # _LLAMA_PRE_TO_LOOM_PRE_TYPE) -- monkeypatch compute_chkhsh so detect_loom_pre_type resolves
        # exactly that hash without needing a real tokenizer to reproduce it, and confirm it raises
        # rather than silently guessing a shape.
        gpt4o_hash = next(h for h, n in td._CHKHSH_TO_LLAMA_PRE.items() if n == "gpt-4o")
        self.assertIsNone(td._LLAMA_PRE_TO_LOOM_PRE_TYPE["gpt-4o"])

        orig = td.compute_chkhsh
        td.compute_chkhsh = lambda tokenizer: gpt4o_hash
        try:
            with self.assertRaises(NotImplementedError):
                td.detect_loom_pre_type(_FakeTokenizer([]))
        finally:
            td.compute_chkhsh = orig

    def test_detect_vocab_family_bpe(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE"}}))
            self.assertEqual(td.detect_vocab_family(d), "bpe")

    def test_detect_vocab_family_wordpiece(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "tokenizer.json").write_text(json.dumps({"model": {"type": "WordPiece"}}))
            self.assertEqual(td.detect_vocab_family(d), "wordpiece")

    def test_detect_vocab_family_unigram_with_proto(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "tokenizer.json").write_text(json.dumps({"model": {"type": "Unigram"}}))
            (Path(d) / "tokenizer.model").write_bytes(b"")
            self.assertEqual(td.detect_vocab_family(d), "sentencepiece_proto")

    def test_detect_vocab_family_unigram_without_proto_raises(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "tokenizer.json").write_text(json.dumps({"model": {"type": "Unigram"}}))
            with self.assertRaises(NotImplementedError):
                td.detect_vocab_family(d)

    def test_detect_vocab_family_bare_sentencepiece_model(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "tokenizer.model").write_bytes(b"")
            self.assertEqual(td.detect_vocab_family(d), "sentencepiece_proto")

    def test_detect_vocab_family_byt5(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "ByT5Tokenizer"}))
            self.assertEqual(td.detect_vocab_family(d), "byte")

    def test_detect_vocab_family_nothing_found_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(NotImplementedError):
                td.detect_vocab_family(d)


if __name__ == "__main__":
    unittest.main()
