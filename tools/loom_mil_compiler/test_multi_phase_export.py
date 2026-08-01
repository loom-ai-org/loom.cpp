"""Checks `ExportPhase`'s axis declarations (BACKLOG.md P4.0.5, `EXPORT-PREPARATION.md` stage B.5).

The division of labour is the point, and it is not obvious from either side alone:

* `LoomGGUFExporter._validate_input_axes` and `_resolve_declared_axes` (P4.0.2) stay exactly where they
  are and keep their messages. They operate on the *traced program* -- they ask whether two genuinely
  independent dynamic axes would collapse onto one symbol, which is only answerable once coremltools has
  assigned real MIL symbols. No spec can see that.
* What lives here is the half answerable from the declaration alone, and that half was not being asked
  at all. A typo'd axis name is a perfectly good dict key: `_sub_symbol` substitutes it happily and the
  phase emits shape expressions over a symbol nothing else in the model uses. Wrong, not malformed, and
  no downstream gate looks at it.

Run: ~/.venvs/piper/bin/python3 -m pytest tools/loom_mil_compiler/test_multi_phase_export.py
"""
import sys
import unittest
from pathlib import Path

import coremltools as ct
import numpy as np
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loom_mil_compiler.multi_phase_export import (
    BaseMultiPhaseModelExportConfig, ExportPhase, TTSFlowMatchingModelExportConfig,
)
from loom_mil_compiler.nemo_asr_export import ASRNemoEncoderExportConfig
from loom_mil_compiler.spec_protocol import (
    LinkError, check_links, dangling_coverage, undeclared_fields,
)


def _phase(**overrides):
    """Kokoro's `decoder_vocoder` shape, cut down: the one real phase in the tree that declares
    non-default axes, and the reason `declared_axes` exists at all (its f0_curve/n_curve/noise_in/wsum
    leaves have no data-flow path back to `asr`, so their lengths are not derivable from the graph)."""
    kwargs = dict(
        name="decoder_vocoder",
        wrapper=nn.Identity(),
        dummy_inputs=(),
        mil_inputs=[
            ct.TensorType(name="asr", shape=(1, 4), dtype=np.float32),
            ct.TensorType(name="f0_curve", shape=(1, 8), dtype=np.float32),
            ct.TensorType(name="wsum", shape=(2404,), dtype=np.float32),
        ],
        root_axis="n_enc_frames",
        declared_axes={"f0_curve": {1: "2*n_enc_frames"}, "wsum": {0: "600*n_enc_frames+20"}},
    )
    kwargs.update(overrides)
    return ExportPhase(**kwargs)


def _message(phase):
    try:
        check_links(phase)
    except LinkError as exc:
        return str(exc)
    raise AssertionError("expected a LinkError")


class TestAxisDeclarations(unittest.TestCase):
    def test_the_real_kokoro_declaration_passes(self):
        self.assertEqual(check_links(_phase()), [])

    def test_the_default_phase_declares_nothing_and_passes(self):
        plain = _phase(root_axis="n_tokens", declared_axes=None)
        self.assertEqual(check_links(plain), [])

    def test_a_typod_root_axis_is_rejected(self):
        msg = _message(_phase(root_axis="n_enc_frame"))
        self.assertIn("root_axis = 'n_enc_frame'", msg)
        self.assertIn("['batch', 'n_codes', 'n_enc_frames', 'n_latent', 'n_samples', 'n_tokens']", msg)

    def test_a_typod_symbol_inside_a_declared_expression_is_rejected(self):
        msg = _message(_phase(declared_axes={"f0_curve": {1: "2*n_frames"}}))
        self.assertIn("declared_axes['f0_curve'][1] = '2*n_frames'", msg)
        self.assertIn("['n_frames']", msg)

    def test_a_declaration_for_an_input_the_phase_does_not_declare_is_rejected(self):
        """Reaches the same class of error `_resolve_declared_axes` raises, but before the trace rather
        than after it -- the phase already knows its own input names, so this costs nothing."""
        msg = _message(_phase(declared_axes={"f0_cruve": {1: "2*n_enc_frames"}}))
        self.assertEqual(
            msg,
            "ExportPhase 'decoder_vocoder' declares axes for input(s) ['f0_cruve'], which it does not "
            "declare in mil_inputs. Its inputs are ['asr', 'f0_curve', 'wsum'].",
        )

    def test_the_axis_links_need_no_context_so_they_cannot_defer(self):
        """This is what lets the phase checks run before the first trace rather than after the last."""
        self.assertEqual(check_links(_phase()), [])


class TestStandingRuleOnTheMultiPhaseFamily(unittest.TestCase):
    def test_every_field_of_every_multi_phase_class_is_declared(self):
        for cls in (ExportPhase, BaseMultiPhaseModelExportConfig, TTSFlowMatchingModelExportConfig):
            self.assertEqual(undeclared_fields(cls), [], cls.__name__)
            self.assertEqual(dangling_coverage(cls), [], cls.__name__)


class TestASRRootAxisIsNowADeclaration(unittest.TestCase):
    """`backend_kwargs()` used to return the literal `"n_samples"`. Making it a field changes no value
    and no export -- it makes the claim checkable, which is the whole of B.5's "what moves is the
    declaration side"."""

    def _config(self, **kw):
        from loom_mil_compiler.nemo_asr_export import EncoderOutput

        return ASRNemoEncoderExportConfig(
            checkpoint="/nonexistent.nemo", output=EncoderOutput.CTC_LOG_PROBS,
            architecture="test-arch", output_path="test.gguf", **kw,
        )

    def test_the_default_is_the_axis_the_family_always_used(self):
        config = self._config()
        self.assertEqual(config.root_axis, "n_samples")
        self.assertEqual(config.backend_kwargs(), {"flat_namespace": True, "root_axis": "n_samples"})

    def test_an_axis_outside_the_vocabulary_is_rejected(self):
        with self.assertRaises(LinkError) as cm:
            check_links(self._config(root_axis="n_sample"), strict=False)
        self.assertIn("root_axis = 'n_sample'", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
