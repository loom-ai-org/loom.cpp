"""
Confirms EXPORT-IMPROVEMENT-BACKLOG.md item 4's STFT/complex-dialect finding actually holds end-to-end:
`torch.stft` decomposes via coremltools' own `common::lower_complex_dialect_ops` pass into ops this
exporter now fully covers (the new "pad"/"conv_transpose" handling in exporter.py), and the new
`tools/loom_mil_compiler/istft.py` module lets the inverse transform -- which coremltools' torch frontend
has no handler for at all -- flow through the same standard pipeline instead of needing a bespoke
hand-derived path (the way Kokoro's own STFT/ISTFT does today, outside this compiler entirely).
"""
import unittest

import torch
import coremltools as ct

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import loom_mil_compiler  # noqa: F401 -- registers the "loom" backend + applies torch-frontend patches
from loom_mil_compiler.exporter import LoomGGUFExporter
from loom_mil_compiler.istft import ISTFT


class RoundTripModule(torch.nn.Module):
    """torch.stft -> ISTFT (this module's own, since torch.istft itself can't be traced) round trip,
    mirroring the shape a real vocoder's STFT-domain processing would take."""

    def __init__(self, n_fft=400, hop_length=160):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, center=True)

    def forward(self, x):
        spec = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft,
            window=self.window, return_complex=True, center=True,
        )
        return self.istft(spec.real, spec.imag)


class TestStftExport(unittest.TestCase):
    def test_stft_istft_round_trip_exports_without_bespoke_ops(self):
        m = RoundTripModule().eval()
        x = torch.randn(1, 16000)

        traced = torch.jit.trace(m, (x,))
        prog = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x.shape)], convert_to="milinternal")

        exporter = LoomGGUFExporter(prog, output_path="test_stft_output.gguf", architecture="stft_test")
        try:
            path = exporter.export()
            self.assertTrue(Path(path).exists())

            topo = next(iter(exporter.topologies.values()))
            ops_used = {node["op"] for node in topo["nodes"]}
            # Every op the STFT->ISTFT decomposition needs must be a real Loom primitive -- if any of
            # these were still missing, .export() itself would have raised NotImplementedError above.
            self.assertIn("PAD_1D_REFLECT", ops_used)  # STFT's center-framing reflect-pad
            self.assertIn("CONV_1D", ops_used)  # STFT's DFT-as-convolution
            self.assertIn("CONV_TRANSPOSE_1D", ops_used)  # ISTFT's synthesis + wsum normalization
        finally:
            Path("test_stft_output.gguf").unlink(missing_ok=True)

    def test_istft_matches_real_torch_istft_numerically(self):
        """The exported topology's own correctness rests on ISTFT's math matching torch.istft -- verified
        directly here (not just via export success) on random, non-self-consistent magnitude/phase, the
        same rigor kokoro_stft_common.py's own docstring requires of the equivalent ggml-graph reduction."""
        torch.manual_seed(0)
        n_fft, hop, n_frames, batch = 400, 160, 47, 2
        n_freq = n_fft // 2 + 1

        real = torch.randn(batch, n_freq, n_frames)
        imag = torch.randn(batch, n_freq, n_frames)
        window = torch.hann_window(n_fft, periodic=True)

        ref = torch.istft(
            torch.complex(real, imag), n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=window, center=True,
        )
        got = ISTFT(n_fft=n_fft, hop_length=hop, center=True).eval()(real, imag)

        self.assertEqual(ref.shape, got.shape)
        self.assertLess((ref - got).abs().max().item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
