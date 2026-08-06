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
        self.assertIn("['batch', 'n_codes', 'n_enc_frames', 'n_kv', 'n_latent', 'n_samples', 'n_tokens']", msg)

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
        # `driver_builder` rides along because this is the CTC family, whose orchestration is not
        # implied by its decomposition (BACKLOG.md P4.0.17); `ctc_blank_id` does not, because nothing
        # has traced the checkpoint here and only it knows the class count.
        self.assertEqual(config.backend_kwargs(),
                         {"flat_namespace": True, "root_axis": "n_samples",
                          "driver_builder": "CtcGreedy"})

    def test_an_axis_outside_the_vocabulary_is_rejected(self):
        with self.assertRaises(LinkError) as cm:
            check_links(self._config(root_axis="n_sample"), strict=False)
        self.assertIn("root_axis = 'n_sample'", str(cm.exception))


if __name__ == "__main__":
    unittest.main()


class TestRecurrentPhase(unittest.TestCase):
    """`recurrent.build_lstm_cell_topologies` was verified against a real traced nn.LSTM when it was
    written and then had **no caller** for the whole of P4.0 -- `generate_graph_topology` raised on an
    `lstm` op and named it as the fix. `RecurrentPhase` is that wiring, so what these check is the
    wiring: the names it creates, the guards on what it will accept, and that a bidirectional module
    yields four cells rather than two."""

    def _lstm_phase(self, input_dim=6, hidden=4, bidirectional=True, name="enc_lstm"):
        import torch

        from loom_mil_compiler.multi_phase_export import RecurrentPhase

        torch.manual_seed(0)
        lstm = torch.nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=bidirectional)
        return RecurrentPhase(name=name, module=lstm, input_dim=input_dim)

    def test_a_bidirectional_module_yields_the_two_cells_run_bi_lstm_drives(self):
        topologies, weights = self._lstm_phase().topologies()
        self.assertEqual(sorted(topologies), ["enc_lstm_bwd", "enc_lstm_fwd"])
        # ONE topology per direction, each declaring both `h_new` and `c_new` -- it was four before,
        # two of which recomputed the other two's node list to read the other half of the same step.
        self.assertEqual([t["outputs"] for t in topologies.values()],
                         [["h_new", "c_new"], ["h_new", "c_new"]])
        # Exactly the names `loom_lua`'s run_bi_lstm composes, so a driver calling
        # run_bi_lstm("enc_lstm", ...) needs no change at all.
        self.assertEqual(sorted(weights), [
            "enc_lstm.bwd.bias", "enc_lstm.bwd.weight_hh", "enc_lstm.bwd.weight_ih",
            "enc_lstm.fwd.bias", "enc_lstm.fwd.weight_hh", "enc_lstm.fwd.weight_ih",
        ])

    def test_a_unidirectional_module_yields_one(self):
        topologies, _ = self._lstm_phase(bidirectional=False).topologies()
        self.assertEqual(sorted(topologies), ["enc_lstm_fwd"])

    def test_input_dim_defaults_to_the_module_s_own(self):
        """So an nn.LSTM never restates it, and a wrong value is impossible rather than checked."""
        phase = self._lstm_phase(input_dim=6)
        phase.input_dim = None
        topologies, _ = phase.topologies()
        self.assertEqual(len(topologies), 2)

    def test_a_wrong_explicit_input_dim_is_torch_s_error_not_ours(self):
        """The trace runs at the declared width, so torch raises first -- naming both numbers, which is
        better than anything this module would write. A post-trace comparison was written here and then
        deleted for being unreachable."""
        phase = self._lstm_phase(input_dim=6)
        phase.input_dim = 7
        with self.assertRaises(RuntimeError) as raised:
            phase.topologies()
        self.assertIn("input.size(-1) must be equal to input_size", str(raised.exception))

    def test_a_module_with_two_lstms_is_rejected_rather_than_half_exported(self):
        import torch

        from loom_mil_compiler.multi_phase_export import RecurrentPhase

        class _Two(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.LSTM(6, 4, batch_first=True)
                self.b = torch.nn.LSTM(4, 4, batch_first=True)

            def forward(self, x):
                out, _ = self.a(x)
                out, _ = self.b(out)
                return out

        with self.assertRaises(ValueError) as raised:
            RecurrentPhase(name="two", module=_Two(), input_dim=6).topologies()
        self.assertIn("traced to 2 'lstm' ops, expected exactly one", str(raised.exception))

    def test_the_cell_is_the_mil_formulation_not_the_state_dict_one(self):
        """One pre-summed `bias`, not `bias_ih`/`bias_hh`. This is why replacing a bespoke BiLSTM
        topology with a generated one is a NUMERIC gate and not a structural one -- the two are
        different arrangements of the same arithmetic."""
        _, weights = self._lstm_phase().topologies()
        self.assertIn("enc_lstm.fwd.bias", weights)
        self.assertNotIn("enc_lstm.fwd.bias_ih", weights)
