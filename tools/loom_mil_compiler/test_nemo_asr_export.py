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
    ASRNemoEncoderExportConfig,
    ASREncoderWrapper,
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
    return ASRNemoEncoderExportConfig(checkpoint="/nonexistent.nemo", output=output,
                              architecture="test-arch", output_path="test.gguf", **kw)


def _run(model, output):
    return ASREncoderWrapper(model, output)(torch.zeros(1, 16000), torch.tensor([16000]))


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

def test_the_three_registered_recognizers_declare_distinct_specs():
    """Guards the one mistake a copy-pasted spec makes: two models writing the same output file, or
    claiming the same architecture string. Was `test_the_three_real_export_scripts_declare_distinct_specs`,
    dynamically loading export_conformer_ctc_mil.py/export_parakeet_{tdt,rnnt}_mil.py -- those scripts
    are gone (BACKLOG.md P3.2: replaced by registry entries), so this now builds each recognizer's
    ASRNemoEncoderExportConfig directly via the registry instead, against the same three real checkpoint
    paths those scripts used to hardcode."""
    from loom_mil_compiler.registry import default_registry

    checkpoints = {
        "conformer-ctc": "/home/flavio/Dev/models/conformer-ctc-small/stt_en_conformer_ctc_small.nemo",
        "parakeet-tdt": "/home/flavio/Dev/models/parakeet_tdt_model/parakeet-tdt-0.6b-v3.nemo",
        "parakeet-rnnt": "/home/flavio/Dev/models/parakeet_rnnt_model/parakeet-rnnt-0.6b.nemo",
    }
    registry = default_registry()
    specs = [
        registry.get("automatic-speech-recognition", name).build_config(Path(checkpoint), f"{name}.gguf")
        for name, checkpoint in checkpoints.items()
    ]

    assert len({s.architecture for s in specs}) == 3
    assert len({s.output_path for s in specs}) == 3
    assert len({s.checkpoint for s in specs}) == 3
    assert [s.output for s in specs] == [EncoderOutput.CTC_LOG_PROBS,
                                          EncoderOutput.ENCODER_BT_D,
                                          EncoderOutput.ENCODER_BT_D]


# -- the spec protocol retrofit (BACKLOG.md P4.0.5, `EXPORT-PREPARATION.md` stage B.3) ---------------
#
# These are the richest messages in the tree -- they name the checkpoint's own d_model AND the config
# field it was read from -- which is why B.3 is the real test of §2's acceptance criterion. The
# assertions above use `match=` and would still pass against a generic "validation failed"; these pin
# the whole string, so a protocol change that degrades a message fails here rather than being noticed
# by the next person to hit the error.

def _message(model, output):
    with pytest.raises(ValueError) as exc:
        _run(model, output)
    return str(exc.value)


def test_arity_message_is_preserved_verbatim():
    model = _FakeASRModel(arity=2, channels=1024, transposed=True)
    assert _message(model, EncoderOutput.CTC_LOG_PROBS) == (
        "ASRNemoEncoderExportConfig declares output=CTC_LOG_PROBS, whose family's forward() returns "
        "3 values, but _FakeASRModel.forward() returned 2. "
        "This checkpoint is not the family the spec claims."
    )


def test_rank_message_is_preserved_verbatim():
    class _Rank2(_FakeASRModel):
        def forward(self, input_signal, input_signal_length):
            return (torch.zeros(7, 1025), torch.zeros(1), torch.zeros(1))

    assert _message(_Rank2(), EncoderOutput.CTC_LOG_PROBS) == (
        "ASRNemoEncoderExportConfig declares output=CTC_LOG_PROBS, expecting a rank-3 tensor from "
        "_Rank2.forward(), but got rank 2 ((7, 1025))."
    )


def test_channel_message_is_preserved_verbatim_and_names_its_config_source():
    model = _FakeASRModel(arity=2, channels=512, transposed=True, cfg=_cfg(d_model=1024))
    assert _message(model, EncoderOutput.ENCODER_BT_D) == (
        "ASRNemoEncoderExportConfig declares output=ENCODER_BT_D, whose axis 1 must be 1024 "
        "(the checkpoint's own cfg.encoder.d_model), but _FakeASRModel produced (1, 512, 7). "
        "This checkpoint is not the family the spec claims."
    )


def test_ctc_channel_message_names_the_other_config_source():
    """The two members read different config fields, and the message says which -- the thing that
    tells a reader whether to look at the tokenizer's vocab size or the encoder's width."""
    model = _FakeASRModel(arity=3, channels=999, decoder=_FakeDecoder(1025))
    assert _message(model, EncoderOutput.CTC_LOG_PROBS) == (
        "ASRNemoEncoderExportConfig declares output=CTC_LOG_PROBS, whose axis -1 must be 1025 "
        "(the checkpoint's own decoder.num_classes_with_blank), but _FakeASRModel produced "
        "(1, 7, 999). This checkpoint is not the family the spec claims."
    )


def test_a_bare_tensor_return_is_caught_by_arity_before_anything_indexes_it():
    """Ordering is load-bearing: the rank and channel links index `outputs[0]`, which means something
    entirely different for a bare tensor than for a tuple. Arity runs first, as it did by hand."""
    class _BareTensor(_FakeASRModel):
        def forward(self, input_signal, input_signal_length):
            return torch.zeros(1, 7, 1025)

    assert "returned 1" in _message(_BareTensor(), EncoderOutput.CTC_LOG_PROBS)


def test_every_config_field_is_declared_including_the_inherited_ones():
    """The standing rule, on the family whose config carries the most inherited surface. `architecture`,
    `output_path` and `decomposition` come from LoomExportConfig and are declared there once -- if MRO
    merging broke, this is what would catch it."""
    from loom_mil_compiler.spec_protocol import dangling_coverage, declared_raw, undeclared_fields

    assert undeclared_fields(ASRNemoEncoderExportConfig) == []
    assert dangling_coverage(ASRNemoEncoderExportConfig) == []
    declared = declared_raw(ASRNemoEncoderExportConfig)
    assert "decomposition" in declared, "inherited from LoomExportConfig, not restated here"


def test_the_encoder_output_links_run_where_the_outputs_exist_and_nowhere_else():
    """EncoderOutput is a NestedSpec on the config: the checker deliberately does not walk into it,
    because its links need the traced forward's return value. Checking it from the config's own site
    must therefore report the missing context rather than quietly passing."""
    from loom_mil_compiler.spec_protocol import LinkChecker, LinkError

    checker = LinkChecker()
    checker.check(EncoderOutput.CTC_LOG_PROBS)
    checker.provide(model=_FakeASRModel())
    with pytest.raises(LinkError) as exc:
        checker.finish()
    assert "were never checked" in str(exc.value)
    assert "needs ['outputs']" in str(exc.value)
