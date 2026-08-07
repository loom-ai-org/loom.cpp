"""`LoomExportConfig.hparams()` and the `loom.*` KVs it becomes (P4.0.8's first follow-up).

Two things are worth a test here and they are different in kind. The first is the writer: what a
number declared as an hparam turns into in the real file, checked by writing one and reading it back
rather than by re-running the branch under test, which would pass whatever the branch did. The second
is the wiring, and it is the one that would rot silently -- every family's `backend_kwargs()` builds
its own dict instead of updating `super()`'s, so a hook every family has can be honoured by some of
them and not others with nothing about the export looking wrong. That is the same failure
`test_spec_protocol`'s standing-rule scan exists for one level up: a declaration nobody reads.
"""
import tempfile
import unittest
from pathlib import Path


def _write_and_read_back(hparams):
    """Writes a minimal GGUF carrying `hparams` and returns `{key: value}` for its `loom.*` KVs."""
    from gguf import GGUFReader

    from .exporter import LoomGGUFExporter

    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "hparams.gguf")
        exporter = LoomGGUFExporter(None, output_path=out, architecture="test", hparams=hparams)
        exporter.write_gguf("-- driver")
        reader = GGUFReader(out)
        # `loom.architecture` is written unconditionally and is a string, not an hparam.
        return {field.name: field.parts[field.data[0]][0]
                for field in reader.fields.values()
                if field.name.startswith("loom.") and field.name != "loom.architecture"}


class TestHparamsAreWritten(unittest.TestCase):
    def test_int_is_u32_and_float_is_f32(self):
        out = _write_and_read_back({"style_dim": 128, "sample_rate": 44100.0})
        self.assertEqual(int(out["loom.style_dim"]), 128)
        self.assertAlmostEqual(float(out["loom.sample_rate"]), 44100.0, places=1)

    def test_declaring_none_writes_none(self):
        """The absence is a real property, not a nicety: `hparam_u32` raises naming the missing key, so
        a family that declares nothing produces a file that says so rather than one carrying zeros."""
        self.assertEqual(_write_and_read_back({}), {})

    def test_non_scalar_raises_naming_the_key(self):
        with self.assertRaises(TypeError) as ctx:
            _write_and_read_back({"durations": [1, 2, 3]})
        self.assertIn("durations", str(ctx.exception))
        self.assertIn("hparam_u32", str(ctx.exception))

    def test_bool_is_rejected_rather_than_written_as_1(self):
        """`isinstance(True, int)` is true in Python, so a bool would sail through as u32 1. It is
        rejected instead: a flag is not a hyperparameter, and silently widening one is how a KV
        namespace stops meaning anything."""
        with self.assertRaises(TypeError) as ctx:
            _write_and_read_back({"fused": True})
        self.assertIn("fused", str(ctx.exception))


class TestEveryFamilyCarriesHparamsThrough(unittest.TestCase):
    """The wiring half. Walks the registry the way `component_registry.usage()` does and asserts that
    every registered family's `backend_kwargs()` -- the only channel between a config and the writer --
    still carries `hparams`. An override that forgets it disables the hook for that family alone, with
    no other symptom."""

    def test_backend_kwargs_carries_hparams(self):
        from .registry import default_registry

        checked = []
        for task, entry in sorted(default_registry()._entries.items()):
            for recognizer in entry.recognizers:
                config = recognizer.build_config(Path("<unused>"), "<unused>.gguf")
                kwargs = config.backend_kwargs()
                self.assertIn(
                    "hparams", kwargs,
                    f"{type(config).__name__}.backend_kwargs() (recognizer {recognizer.name!r}, task "
                    f"{task!r}) does not carry hparams=self.hparams(), so nothing this family declares "
                    f"there can reach the GGUF.")
                self.assertIsInstance(kwargs["hparams"], dict)
                checked.append(recognizer.name)
        # A scan that silently found nothing to scan passes just as loudly as one that checked
        # everything, which is the whole reason this line is here.
        self.assertGreater(len(checked), 5, f"only walked {checked}")


if __name__ == "__main__":
    unittest.main()
