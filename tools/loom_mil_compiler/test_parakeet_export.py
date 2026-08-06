"""Checks `parakeet_export.ASRParakeetExportConfig` (BACKLOG.md P4.0.17 step 2).

These run with no checkpoint: the two wrappers take modules, so a `nn.Embedding`/`RNNTJoint`-shaped
stand-in exercises the same code the real export traces. What needs the real 0.6B checkpoint -- that the
traced phases reproduce its numbers -- is an e2e gate, not a unit test.
"""
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loom_mil_compiler.parakeet_export import _EmbedWrapper, _JointWrapper


class _FakeJoint(nn.Module):
    """`RNNTJoint`'s three pieces and nothing else -- `enc`, `pred`, `joint_net` -- which is exactly the
    surface `_JointWrapper` touches."""

    def __init__(self, n_embd=8, pred_hidden=6, n_out=11):
        super().__init__()
        self.enc = nn.Linear(n_embd, pred_hidden)
        self.pred = nn.Linear(pred_hidden, pred_hidden)
        self.joint_net = nn.Sequential(nn.ReLU(), nn.Dropout(0.0), nn.Linear(pred_hidden, n_out))
        self.num_classes_with_blank = n_out


class TestJointHeadSplit(unittest.TestCase):
    """The joint's 8198 outputs are 8193 token classes and 5 durations concatenated. Splitting them at
    export time is what lets the driver argmax the tokens engine-side and marshal only the five
    durations -- so the split has to be in the right place, and in the right order."""

    def test_the_two_heads_are_the_two_halves_of_the_concatenated_output(self):
        joint = _FakeJoint(n_out=11).eval()
        f, g = torch.randn(8), torch.randn(6)
        with torch.no_grad():
            whole = joint.joint_net(joint.enc(f) + joint.pred(g))
            tokens, durations = _JointWrapper(joint, n_durations=5)(f, g)
        self.assertEqual(tokens.shape, (6,))
        self.assertEqual(durations.shape, (5,))
        # The token head is the FRONT of the vector and the durations the tail, which is the order the
        # driver's `argmax_row(joint, 0)` / `get_output(joint, 2)` pair depends on.
        torch.testing.assert_close(tokens, whole[:6])
        torch.testing.assert_close(durations, whole[6:])

    def test_plain_rnnt_has_no_duration_head_at_all(self):
        """Not a zero-width second output -- no second output. `TdtDecoderConfig::durations` empty draws
        the same distinction, and it is what selects every-blank-advances-one-frame in the driver."""
        joint = _FakeJoint(n_out=7).eval()
        out = _JointWrapper(joint, n_durations=0)(torch.randn(8), torch.randn(6))
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (7,))


class TestEmbedWrapper(unittest.TestCase):
    def test_it_maps_a_one_element_label_to_the_cell_s_input_vector(self):
        """The shapes are the contract with two other phases: `[1]` is what the driver has (a Lua array
        of one number) and `[pred_hidden]` is what `RecurrentPhase`'s cell takes as `layer_input`."""
        embed = nn.Embedding(13, 6)
        out = _EmbedWrapper(embed)(torch.tensor([4], dtype=torch.int64))
        self.assertEqual(out.shape, (6,))
        torch.testing.assert_close(out, embed.weight[4])


class TestTheDurationSetIsCrossCheckedAgainstTheJoint(unittest.TestCase):
    """The check that caught a real error while this config was being written.

    `joint.num_classes_with_blank` reads like the token count and is not: for a TDT joint NeMo sets it
    to `num_classes + 1 + num_extra_outputs`, so it already counts the durations (8198, not 8193).
    Deriving the blank from it put it five classes too high and would have split the head in the wrong
    place -- token logits running into the duration ones, which no shape check would catch because the
    widths still add up. The token count comes off the EMBEDDING instead, and the joint's own width is
    compared against tokens + durations.
    """

    def test_a_duration_set_that_disagrees_with_the_joint_is_rejected(self):
        from loom_mil_compiler.parakeet_export import ASRParakeetExportConfig

        class _Model:
            class decoder:
                class prediction:
                    embed = nn.Embedding(9, 6)
                    class dec_rnn:
                        lstm = nn.LSTM(6, 6, num_layers=2)
            joint = _FakeJoint(n_out=11)   # 9 tokens + 2 durations, not 5

        cfg = ASRParakeetExportConfig(checkpoint="/unused.nemo", durations=(0, 1, 2, 3, 4))
        cfg.load_model = lambda: _Model()
        with self.assertRaises(ValueError) as raised:
            cfg.phases()
        message = str(raised.exception)
        self.assertIn("the joint emits 11 values", message)
        self.assertIn("9 token classes and 5 duration(s)", message)


if __name__ == "__main__":
    unittest.main()
