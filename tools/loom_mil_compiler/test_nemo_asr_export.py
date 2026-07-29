"""Covers `nemo_asr_export.py`'s validation -- the half of the family template that isn't codegen.

Deliberately runs without NeMo, coremltools or any checkpoint: the thing under test is whether a spec's
claim about a model is checked against the model, and a fake module reproduces every shape of mismatch
faithfully (wrong forward arity, wrong channel count, missing preprocessor/encoder, missing sample
rate). The real checkpoints are covered by the three end-to-end reference tests instead
(`test_e2e_conformer_ctc_mil_export` and the two Parakeet equivalents).

Run: ~/.venvs/piper/bin/python3 -m pytest tools/loom_mil_compiler/test_nemo_asr_export.py
"""
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loom_mil_compiler.nemo_asr_export import (  # noqa: E402
    EncoderOutput,
    NeMoASREncoderSpec,
    _NeMoASREncoderWrapper,
)


class _FakeDecoder:
    def __init__(self, num_classes_with_blank):
        self.num_classes_with_blank = num_classes_with_blank


def _cfg(sample_rate=16000, d_model=1024):
    return types.SimpleNamespace(
        preprocessor=types.SimpleNamespace(sample_rate=sample_rate),
        encoder=types.SimpleNamespace(d_model=d_model),
    )


class _FakeASRModel(nn.Module):
    """Stands in for a restored NeMo model: the same `forward(input_signal=, input_signal_length=)`
    signature, returning a tuple of the family's own arity and shapes."""

    def __init__(self, arity=3, channels=1025, transposed=False, cfg=None, decoder=None,
                 has_preprocessor=True, has_encoder=True):
        super().__init__()
        self.arity = arity
        self.channels = channels
        self.transposed = transposed  # emit (B, C, T) like NeMo's own encoder output
        self.cfg = _cfg() if cfg is None else cfg
        self.decoder = _FakeDecoder(1025) if decoder is None else decoder
        if has_preprocessor:
            self.preprocessor = nn.Identity()
        if has_encoder:
            self.encoder = nn.Identity()

    def forward(self, input_signal, input_signal_length):
        n_frames = 7
        main = (torch.zeros(1, self.channels, n_frames) if self.transposed
                else torch.zeros(1, n_frames, self.channels))
        rest = tuple(torch.zeros(1) for _ in range(self.arity - 1))
        return (main,) + rest


def _spec(output=EncoderOutput.CTC_LOG_PROBS, **kw):
    return NeMoASREncoderSpec(checkpoint="/nonexistent.nemo", output=output,
                              architecture="test-arch", output_path="test.gguf", **kw)


def _run(model, output):
    return _NeMoASREncoderWrapper(model, output)(torch.zeros(1, 16000), torch.tensor([16000]))


# -- the happy paths ---------------------------------------------------------------------------------

def test_ctc_returns_log_probs_untransposed():
    out = _run(_FakeASRModel(arity=3, channels=1025), EncoderOutput.CTC_LOG_PROBS)
    assert tuple(out.shape) == (1, 7, 1025)


def test_encoder_output_is_transposed_to_bt_d():
    model = _FakeASRModel(arity=2, channels=1024, transposed=True)
    out = _run(model, EncoderOutput.ENCODER_BT_D)
    # NeMo's own (B, D, T) becomes this project's ne[0]=feature/ne[1]=time (B, T, D).
    assert tuple(out.shape) == (1, 7, 1024)


# -- the mismatches the template exists to catch -----------------------------------------------------

def test_ctc_spec_on_an_rnnt_checkpoint_raises_on_arity():
    """The headline case: an RNNT/TDT checkpoint returns 2 values, not 3. Without the check this
    exports an encoder activation under the name `log_probs`."""
    model = _FakeASRModel(arity=2, channels=1024, transposed=True)
    with pytest.raises(ValueError, match=r"returns 3 values, but .* returned 2"):
        _run(model, EncoderOutput.CTC_LOG_PROBS)


def test_encoder_spec_on_a_ctc_checkpoint_raises_on_arity():
    with pytest.raises(ValueError, match=r"returns 2 values, but .* returned 3"):
        _run(_FakeASRModel(arity=3, channels=1025), EncoderOutput.ENCODER_BT_D)


def test_channel_mismatch_against_the_checkpoints_own_config_raises():
    """Right arity, wrong model: the config says 1025 CTC classes and the tensor has 999."""
    model = _FakeASRModel(arity=3, channels=999, decoder=_FakeDecoder(1025))
    with pytest.raises(ValueError, match=r"axis -1 must be 1025"):
        _run(model, EncoderOutput.CTC_LOG_PROBS)


def test_the_encoder_channel_check_reads_nemos_own_b_d_t_layout():
    """The check runs on the model's OWN output, before the transpose -- so it must look at axis 1, not
    the last axis. A (B, T, D) reading would pass whenever T happened to equal d_model and fail
    otherwise, i.e. it would be a check on the audio length."""
    model = _FakeASRModel(arity=2, channels=1024, transposed=True, cfg=_cfg(d_model=1024))
    assert tuple(_run(model, EncoderOutput.ENCODER_BT_D).shape) == (1, 7, 1024)


def test_encoder_channel_mismatch_names_d_model():
    model = _FakeASRModel(arity=2, channels=512, transposed=True, cfg=_cfg(d_model=1024))
    with pytest.raises(ValueError, match=r"cfg.encoder.d_model"):
        _run(model, EncoderOutput.ENCODER_BT_D)


def test_rank_mismatch_raises():
    class _Rank2(_FakeASRModel):
        def forward(self, input_signal, input_signal_length):
            return (torch.zeros(7, 1025), torch.zeros(1), torch.zeros(1))

    with pytest.raises(ValueError, match=r"rank-3"):
        _run(_Rank2(), EncoderOutput.CTC_LOG_PROBS)


# -- pre-trace structural validation -----------------------------------------------------------------

def test_validate_against_model_returns_the_checkpoints_sample_rate():
    assert _spec().validate_against_model(_FakeASRModel(cfg=_cfg(sample_rate=22050))) == 22050


def test_missing_encoder_raises_before_tracing():
    model = _FakeASRModel(has_encoder=False)
    with pytest.raises(ValueError, match=r"no \['encoder'\]"):
        _spec().validate_against_model(model)


def test_missing_sample_rate_raises_before_tracing():
    model = _FakeASRModel(cfg=types.SimpleNamespace(preprocessor=types.SimpleNamespace(),
                                                    encoder=types.SimpleNamespace(d_model=1024)))
    with pytest.raises(ValueError, match=r"no preprocessor.sample_rate"):
        _spec().validate_against_model(model)


def test_ctc_spec_without_a_decoder_raises_before_tracing():
    """`expected_channels` is read during the pre-trace validation too, so a CTC spec pointed at a
    model with no CTC decoder fails before the (slow) trace, not during it."""
    model = _FakeASRModel(arity=2, channels=1024, transposed=True)
    del model.decoder
    with pytest.raises(AttributeError):
        _spec().validate_against_model(model)


# -- the real specs ----------------------------------------------------------------------------------

def test_the_three_real_export_scripts_declare_distinct_specs():
    """Guards the one mistake a copy-pasted spec makes: two models writing the same output file, or
    claiming the same architecture string."""
    import importlib.util

    repo = Path(__file__).resolve().parents[2]
    specs = []
    for name in ("export_conformer_ctc_mil", "export_parakeet_tdt_mil", "export_parakeet_rnnt_mil"):
        spec_file = repo / f"{name}.py"
        module_spec = importlib.util.spec_from_file_location(name, spec_file)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        specs.append(module.SPEC)

    assert len({s.architecture for s in specs}) == 3
    assert len({s.output_path for s in specs}) == 3
    assert len({s.checkpoint for s in specs}) == 3
    assert [s.output for s in specs] == [EncoderOutput.CTC_LOG_PROBS,
                                          EncoderOutput.ENCODER_BT_D,
                                          EncoderOutput.ENCODER_BT_D]
