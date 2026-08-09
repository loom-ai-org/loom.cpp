"""Checks `gigaam_export.py` (BACKLOG.md P4.2) -- the parts that do not need the real checkpoint.

Almost all of GigaAM's export is `transducer_export.BaseTransducerExportConfig`, which
`test_parakeet_export.py` already covers. What is genuinely this module's is the *loader*, and inside
the loader the one thing that rewrites the model rather than merely reading it: the mel frontend.
That is what these tests are about, plus the layout `transducer_parts()` claims.

The mel tests use a real `torchaudio.transforms.MelSpectrogram` built here rather than one read off a
checkpoint -- the equivalence claim is about torchaudio's arithmetic, so torchaudio is the authority and
a 450 MB download adds nothing to it. That the CHECKPOINT's own module also matches is checked at export
time, on every export, by `_replace_mel_frontend`.

Run: ~/.venvs/piper/bin/python3 -m pytest tools/loom_mil_compiler/test_gigaam_export.py
"""
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loom_mil_compiler.gigaam_export import (  # noqa: E402
    GIGAAM_ENCODER_INPUT_NAMES,
    ASRGigaAMExportConfig,
    _TraceableMelSpectrogram,
    _replace_mel_frontend,
)

SR = 16000


def _mel(**overrides):
    """GigaAM v3's own frontend geometry, read off its `config.json`: 64 mels, a 320-sample window and
    hop 160, `center=False`."""
    kwargs = dict(sample_rate=SR, n_mels=64, win_length=320, hop_length=160, n_fft=320, center=False)
    kwargs.update(overrides)
    return torchaudio.transforms.MelSpectrogram(**kwargs).eval()


def _speechlike(seconds=2.0):
    """A deterministic sweep -- no RNG, so nothing this test does can perturb a trace, and every mel bin
    gets real energy at some point (white noise would too, but a sweep makes a per-bin failure obvious)."""
    t = torch.arange(int(seconds * SR), dtype=torch.float32) / SR
    return torch.sin(2.0 * torch.pi * (150.0 + 2000.0 * t / seconds) * t).unsqueeze(0)


class TestTheRewrittenFrontendIsTheSameArithmetic:
    """The replacement exists because coremltools cannot lower the `complex_shape` op torchaudio's own
    batch pack/unpack emits -- not because anything about the mel is different. So the property that
    matters is that it is not merely close."""

    def test_it_reproduces_torchaudios_own_output_exactly(self):
        mel = _mel()
        probe = _speechlike()
        with torch.no_grad():
            want, got = mel(probe), _TraceableMelSpectrogram(mel)(probe)
        assert want.shape == got.shape
        # Bit-for-bit: same ops, same order, same buffers, with only the reshape that exists to support
        # a batch axis this export does not have removed.
        assert torch.equal(want, got)

    def test_it_carries_the_checkpoints_own_filterbank_and_window_rather_than_rebuilding_them(self):
        mel = _mel()
        rewritten = _TraceableMelSpectrogram(mel)
        assert torch.equal(rewritten.window, mel.spectrogram.window)
        assert torch.equal(rewritten.fb, mel.mel_scale.fb)

    def test_a_centred_spectrogram_is_reproduced_too(self):
        """`center` is read off the module, not assumed: GigaAM's is False and Whisper's own frontend
        (a different family, same shape of rewrite) is True."""
        mel = _mel(center=True)
        probe = _speechlike()
        with torch.no_grad():
            assert torch.equal(mel(probe), _TraceableMelSpectrogram(mel)(probe))

    @pytest.mark.parametrize("field, value", [("pad", 4), ("normalized", True)])
    def test_an_option_this_rewrite_does_not_reproduce_is_rejected_not_ignored(self, field, value):
        """Silently dropping a pad or a normalization would show up as a slightly-wrong transcript and
        nothing else, which is the one failure mode worth spending a raise on."""
        mel = _mel()
        setattr(mel.spectrogram, field, value)
        with pytest.raises(ValueError, match="reproduces torchaudio's spectrogram"):
            _TraceableMelSpectrogram(mel)


class TestTheSwapIsCheckedAgainstTheModelItReplaces:
    def _model(self, mel):
        model = types.SimpleNamespace()
        model.preprocessor = types.SimpleNamespace(featurizer=nn.Sequential(mel, nn.Identity()))
        return model

    def test_the_swap_happens_and_the_original_is_gone(self):
        model = self._model(_mel())
        _replace_mel_frontend(model, SR)
        assert isinstance(model.preprocessor.featurizer[0], _TraceableMelSpectrogram)

    def test_a_frontend_that_disagrees_fails_the_export(self):
        """The check is the reason this module can claim the frontend is unchanged. Rig the replacement
        to be wrong and the export must stop, rather than produce a GGUF whose mel is subtly its own."""
        model = self._model(_mel())

        class _Wrong(_TraceableMelSpectrogram):
            def forward(self, waveform):
                return super().forward(waveform) * 1.5

        import loom_mil_compiler.gigaam_export as module

        original = module._TraceableMelSpectrogram
        module._TraceableMelSpectrogram = _Wrong
        try:
            with pytest.raises(ValueError, match="differs from this checkpoint's own MelSpectrogram"):
                _replace_mel_frontend(model, SR)
        finally:
            module._TraceableMelSpectrogram = original

    def test_a_frontend_of_the_wrong_shape_is_reported_as_a_shape_not_a_tolerance(self):
        model = self._model(_mel())

        class _Shorter(_TraceableMelSpectrogram):
            def forward(self, waveform):
                return super().forward(waveform)[..., :-1]

        import loom_mil_compiler.gigaam_export as module

        original = module._TraceableMelSpectrogram
        module._TraceableMelSpectrogram = _Shorter
        try:
            with pytest.raises(ValueError, match="MelSpectrogram produced"):
                _replace_mel_frontend(model, SR)
        finally:
            module._TraceableMelSpectrogram = original


class TestTheLayoutThisCheckpointDeclares:
    """`transducer_parts` is the whole of what the family template does not know. A fake with GigaAM's
    own attribute names is exactly the surface it touches."""

    def _model(self):
        head = types.SimpleNamespace(
            decoder=types.SimpleNamespace(embed=nn.Embedding(1025, 320),
                                          lstm=nn.LSTM(320, 320, num_layers=1)),
            joint=types.SimpleNamespace(enc=nn.Linear(768, 320), pred=nn.Linear(320, 320),
                                        joint_net=nn.Sequential(nn.ReLU(), nn.Linear(320, 1025))),
        )
        return types.SimpleNamespace(head=head)

    def test_it_reads_both_halves_out_of_the_rnnt_head(self):
        model = self._model()
        parts = ASRGigaAMExportConfig(checkpoint="/unused").transducer_parts(model)
        assert parts.embed is model.head.decoder.embed
        assert parts.lstm is model.head.decoder.lstm
        assert parts.joint is model.head.joint

    def test_a_plain_rnnt_declares_no_durations_and_no_second_joint_width(self):
        """Both are real absences rather than unread fields: `modeling_gigaam.RNNTJoint` has no
        `num_classes_with_blank` and the config has no `model_defaults` at all, so the token count comes
        off the embedding and the joint's width is checked against that alone."""
        parts = ASRGigaAMExportConfig(checkpoint="/unused").transducer_parts(self._model())
        assert parts.durations == ()
        assert parts.declared_joint_width is None


class TestTheLoaderIsTheOnlyThingThisFamilyChanges:
    def test_the_encoder_wrapper_names_this_checkpoints_own_forward_arguments(self):
        """NeMo's `forward(input_signal=, input_signal_length=)` and GigaAM's
        `forward(features=, feature_lengths=)` take the same two tensors. That difference is the whole
        of what the encoder half of the family needed to become loader-independent."""
        from loom_mil_compiler.nemo_asr_export import ENCODER_INPUT_NAMES

        wrapper = ASRGigaAMExportConfig(checkpoint="/unused").encoder_wrapper(nn.Identity())
        assert wrapper.input_names == GIGAAM_ENCODER_INPUT_NAMES
        assert wrapper.input_names != ENCODER_INPUT_NAMES

    def test_it_shares_the_transducer_familys_driver_rather_than_carrying_one(self):
        from loom_mil_compiler.parakeet_export import ASRParakeetExportConfig

        gigaam = ASRGigaAMExportConfig(checkpoint="/unused")
        parakeet = ASRParakeetExportConfig(checkpoint="/unused.nemo")
        assert gigaam.driver_script_path == parakeet.driver_script_path
        assert gigaam.driver_script_path.name == "transducer_driver"

    def test_the_tokenizer_travels_with_the_directory_when_there_is_one(self, tmp_path):
        """Unlike the `.nemo` archives -- whose SentencePiece model is content-hashed inside a tarball
        and has to be extracted -- GigaAM's sits in the directory under the name `_write_tokenizer`
        already looks for."""
        (tmp_path / "tokenizer.model").write_bytes(b"not really a proto")
        kwargs = ASRGigaAMExportConfig(checkpoint=str(tmp_path)).backend_kwargs()
        assert kwargs["tokenizer_dir"] == str(tmp_path)
        assert kwargs["tokenizer_family"] == "sentencepiece_proto"

    def test_a_directory_with_no_tokenizer_says_nothing_rather_than_raising(self, tmp_path):
        """`component_registry.usage()` builds every registered config against a path that does not
        exist, so "no tokenizer here" has to be an answer."""
        kwargs = ASRGigaAMExportConfig(checkpoint=str(tmp_path / "nope")).backend_kwargs()
        assert "tokenizer_dir" not in kwargs
