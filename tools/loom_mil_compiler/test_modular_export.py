"""Checks `ModularExportSpec`'s validation (BACKLOG.md P4.0.5, `EXPORT-PREPARATION.md` stage B.4).

Deliberately runs without a checkpoint, coremltools or a trace: what is under test is whether a spec's
claims about a module's structure are checked against the module, and a small `nn.Module` reproduces
every shape of mismatch faithfully. The real LFM2 modular export is covered by
`test_causal_lm_export.py`, which needs the checkpoint and several minutes.

The property that matters most here is *when* the failure happens. Before the retrofit, `get_by_path`
raised a bare `AttributeError` from wherever its traversal reached -- which for `suffix_attrs` was after
the prefix and aux submodules had already been traced. A misspelled attribute name therefore cost a
full trace to discover, and reported only the missing attribute, not which declaration named it.
"""
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loom_mil_compiler.modular_export import ModularExportSpec, export_modular
from loom_mil_compiler.spec_protocol import (
    LinkError, dangling_coverage, undeclared_fields,
)
from loom_mil_compiler.spec_protocol import check_links


class _Layer(nn.Module):
    def forward(self, hidden_states, position_embeddings=None, past_key_values=None):
        return hidden_states


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([_Layer(), _Layer()])
        self.embedding_norm = nn.LayerNorm(4)
        self.pos_emb = nn.Identity()


class _FakeCausalLM(nn.Module):
    """The shape `ModularExportSpec` describes: a prefix embedding, a repeated `nn.ModuleList` with more
    than one child, suffix stages, and a once-computed auxiliary submodule."""

    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.lm_head = nn.Linear(4, 8)


def _spec(**overrides):
    kwargs = dict(
        prefix_attr="model.embed_tokens",
        repeated_attr="model.layers",
        suffix_attrs=["model.embedding_norm", "lm_head"],
        aux_attr="model.pos_emb",
        aux_kwarg="position_embeddings",
    )
    kwargs.update(overrides)
    return ModularExportSpec(**kwargs)


def _message(spec):
    model = _FakeCausalLM()
    try:
        check_links(spec, model=model)
    except LinkError as exc:
        return str(exc)
    raise AssertionError("expected a LinkError")


class TestModularExportSpecLinks(unittest.TestCase):
    def test_the_real_lfm2_declaration_shape_passes(self):
        self.assertEqual(check_links(_spec(), model=_FakeCausalLM()), [])

    def test_a_wrong_prefix_path_names_the_field_the_path_and_the_type(self):
        msg = _message(_spec(prefix_attr="model.embed_token"))
        self.assertIn("prefix_attr = 'model.embed_token'", msg)
        # The type it walked into is the part that says whether this is a typo or the wrong nesting
        # level -- a bare AttributeError gave the attribute name and nothing else.
        self.assertIn("_Inner (at model) has no attribute 'embed_token'", msg)

    def test_a_wrong_suffix_path_names_which_entry(self):
        msg = _message(_spec(suffix_attrs=["model.embedding_norm", "lm_heads"]))
        self.assertIn("suffix_attrs[1] = 'lm_heads'", msg)

    def test_the_repeated_block_message_is_preserved_verbatim(self):
        """This one had a good message already and it survives byte-for-byte: `find_repeated_blocks`
        re-derives the qualifying blocks independently, so listing them is what turns a wrong name into
        a one-line fix."""
        self.assertEqual(
            _message(_spec(repeated_attr="model.layer")),
            "'model.layer' is not a qualifying repeated block (an nn.ModuleList/Sequential with more "
            "than one child); discovered repeated blocks: ['model.layers']",
        )

    def test_a_non_repeated_attribute_is_rejected_even_though_the_path_resolves(self):
        """`model.embedding_norm` exists, so a plain path check would pass it. The declaration is about
        structure, not existence."""
        self.assertIn("is not a qualifying repeated block",
                      _message(_spec(repeated_attr="model.embedding_norm")))

    def test_a_misspelled_aux_kwarg_names_the_parameters_that_do_exist(self):
        """Previously unchecked entirely: a wrong name here produces a layout whose aux inputs match no
        repeated-block parameter, surfacing much later inside `apply_modular_export` with nothing
        pointing back at the spec."""
        msg = _message(_spec(aux_kwarg="position_embedding"))
        self.assertIn("aux_kwarg='position_embedding'", msg)
        self.assertIn("model.layers's blocks take no such parameter", msg)
        self.assertIn("'position_embeddings'", msg)

    def test_a_spec_with_no_aux_submodule_is_neither_checked_nor_deferred(self):
        """Both aux fields are optional, and WhenSet is what keeps leaving them unset from tripping the
        deferral report."""
        self.assertEqual(
            check_links(_spec(aux_attr=None, aux_kwarg=None), model=_FakeCausalLM()), [])

    def test_every_field_is_declared(self):
        self.assertEqual(undeclared_fields(ModularExportSpec), [])
        self.assertEqual(dangling_coverage(ModularExportSpec), [])


class TestFailureHappensBeforeAnythingIsTraced(unittest.TestCase):
    """The behavioural upgrade, stated as a test rather than as a claim in a docstring."""

    def test_export_modular_rejects_a_bad_spec_before_touching_its_inputs(self):
        # `dummy_inputs` is deliberately empty and `seq_len` meaningless: if the check did not run
        # first, this would fail somewhere inside the capture/trace machinery with a different error.
        with self.assertRaises(LinkError) as cm:
            export_modular(_FakeCausalLM(), _spec(suffix_attrs=["nope"]), {}, seq_len=37)
        self.assertIn("suffix_attrs[0] = 'nope'", str(cm.exception))

    def test_a_valid_spec_gets_past_the_check_and_fails_later_on_its_own_terms(self):
        """The negative control: proves the test above is measuring the check and not merely that
        `export_modular` rejects everything it is handed here."""
        with self.assertRaises(Exception) as cm:
            export_modular(_FakeCausalLM(), _spec(), {}, seq_len=37)
        self.assertNotIsInstance(cm.exception, LinkError)


if __name__ == "__main__":
    unittest.main()
