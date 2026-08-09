"""GigaAM v3 -- the second loader for the Transducer family, and the point of BACKLOG.md P4.2.

GigaAM v3 is a Conformer-based Russian(+EN) foundation model (~240M parameters) released by ai-sage as
an HF directory carrying its own `modeling_gigaam.py`. Its graph is `EXPORT-ROADMAP.md` R5's family 1
and its head is an ordinary RNN-T, so the interesting part of exporting it is *not* the graph -- it is
that nothing about it is loaded the way the three `.nemo` checkpoints are:

    ASRModel.restore_from("parakeet-tdt-0.6b-v3.nemo")     vs
    AutoModel.from_pretrained("<dir>", trust_remote_code=True).model

R3 predicted that adding this would force the loader out of the family template ("the template's real
contract is *give me an nn.Module and tell me what its forward returns*"), and it did: this module is
that loader, the two forward argument names the remote code happens to use, where GigaAM puts its three
post-encoder modules, and two workarounds the remote code needs before it will trace. Every other fact
about the export -- the four phases, the blank id, the joint split, the decode loop, the driver -- comes
from `transducer_export.BaseTransducerExportConfig` and `nemo_asr_export`'s family-1 encoder half,
unchanged and un-parameterized. **That reuse is the deliverable**; the code below is what was left over.

## The two things the remote code needs before it traces

Both were found by tracing rather than predicted, and both are cross-checked rather than trusted:

1. **The rotary positional table is built lazily, on the first forward.** `ConformerEncoder.forward`
   begins `if not hasattr(self.pos_enc, "pe"): self.pos_enc.extend_pe(...)`, and `pe` is a
   `persistent=False` buffer, so a freshly loaded checkpoint has none. Built during the trace it would
   become graph ops; built under `torch.inference_mode()` it becomes an *inference tensor* and the trace
   then dies in autograd with "Inference tensors cannot be saved for backward". So `load_model` builds
   it eagerly, outside inference mode, at the checkpoint's own `pos_emb_max_len`.

2. **torchaudio's `MelSpectrogram` cannot be converted, for a reason that is not about mel at all.**
   `torchaudio.functional.spectrogram` reshapes the STFT result back to the input's batch shape, and
   reading `spec_f.shape` on a *complex* tensor emits a `complex_shape` op that coremltools' own
   `lower_complex_dialect_ops` pass cannot lower (`'complex_shape' object has no attribute 'data'`).
   `_TraceableMelSpectrogram` is the same arithmetic without that reshape -- and the same shape as
   `whisper_export.WhisperMelFrontend`, which exists for the neighbouring reason. It is not asserted to
   be equivalent: `load_model` runs both on real audio and compares, so the export fails rather than
   producing a subtly different frontend.

## What this module deliberately does not claim

GigaAM v3 ships five variants (`ssl`, `ctc`, `rnnt`, `e2e_ctc`, `e2e_rnnt`); only `e2e_rnnt` is on this
machine, and the recognizer says so -- it requires `model_class == "rnnt"` rather than claiming every
`model_type == "gigaam"` directory. A CTC variant is family 1's *`Flattened`* shape
(`ASRNemoEncoderExportConfig` with `CTC_LOG_PROBS`) plus this loader, which is a small addition and an
untested one; it fails detection with the candidate list rather than being exported by a path nothing
here has ever run.
"""
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .nemo_asr_export import ASREncoderWrapper
from .transducer_export import BaseTransducerExportConfig, TransducerParts

# GigaAM's own `forward(features, feature_lengths)`. The two tensors are the same waveform and sample
# count NeMo calls `input_signal`/`input_signal_length` -- the names are the whole difference, which is
# why `ASREncoderWrapper` takes them as a parameter.
GIGAAM_ENCODER_INPUT_NAMES = ("features", "feature_lengths")

# Seconds of real-shaped audio the mel equivalence check below runs on. Long enough to cover several
# hundred frames (a one-frame clip would agree trivially) and short enough to cost nothing.
MEL_CHECK_SECONDS = 5.0


class _TraceableMelSpectrogram(nn.Module):
    """`torchaudio.transforms.MelSpectrogram`, rewritten so the complex STFT never has its shape taken.

    Every parameter is read off the real module rather than restated: the window and the mel filterbank
    are its own buffers, and the FFT geometry is its own attributes. The arithmetic is
    `torchaudio.functional.spectrogram` followed by `MelScale.forward`, minus the pack/unpack reshape
    that only exists to support a batch axis this export does not have (the input is already `(1, n)`).

    Three of the real module's options are not reproduced, and are rejected rather than ignored --
    silently dropping a normalization or a pad is exactly the class of difference that would show up as
    a slightly-wrong transcript and nothing else.
    """

    def __init__(self, mel: nn.Module):
        super().__init__()
        spec = mel.spectrogram
        unsupported = {
            "pad": (int(spec.pad), 0),
            "normalized": (bool(spec.normalized), False),
            "onesided": (bool(spec.onesided), True),
        }
        wrong = {name: got for name, (got, want) in unsupported.items() if got != want}
        if wrong:
            raise ValueError(
                f"_TraceableMelSpectrogram reproduces torchaudio's spectrogram for "
                f"{ {name: want for name, (_, want) in unsupported.items()} }, but this checkpoint's "
                f"declares {wrong}. Reproducing those is real arithmetic, not a flag -- add it here "
                f"rather than letting the frontend differ."
            )
        self.n_fft = int(spec.n_fft)
        self.hop_length = int(spec.hop_length)
        self.win_length = int(spec.win_length)
        self.center = bool(spec.center)
        self.power = float(spec.power)
        self.register_buffer("window", spec.window.detach().clone())
        # `(n_freq, n_mels)`, the orientation `MelScale.forward` matmuls against.
        self.register_buffer("fb", mel.mel_scale.fb.detach().clone())

    def forward(self, waveform):
        spec = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window=self.window, center=self.center,
                          return_complex=True)
        power = spec.abs() ** self.power
        return torch.matmul(power.transpose(-1, -2), self.fb).transpose(-1, -2)


def _replace_mel_frontend(model, sample_rate: int, tolerance: float = 1e-5) -> None:
    """Swaps the preprocessor's `MelSpectrogram` for `_TraceableMelSpectrogram`, after checking on real
    audio that the two agree.

    The check runs BEFORE the swap and on a deterministic signal (a linear chirp -- no RNG, so it cannot
    perturb anything a traced constant depends on). The tolerance is generous relative to what is
    actually observed: the two agree exactly, bit for bit, because they are the same ops in the same
    order on the same buffers.
    """
    featurizer = model.preprocessor.featurizer
    original = featurizer[0]
    replacement = _TraceableMelSpectrogram(original).eval()

    n = int(MEL_CHECK_SECONDS * sample_rate)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    probe = torch.sin(2.0 * torch.pi * (200.0 + 1500.0 * t / MEL_CHECK_SECONDS) * t).unsqueeze(0)
    with torch.no_grad():
        want, got = original(probe), replacement(probe)
    if want.shape != got.shape:
        raise ValueError(
            f"_TraceableMelSpectrogram produced {tuple(got.shape)} where torchaudio's own "
            f"MelSpectrogram produced {tuple(want.shape)} for {MEL_CHECK_SECONDS} s of audio."
        )
    deviation = (want - got).abs().max().item()
    if deviation > tolerance * max(want.abs().max().item(), 1.0):
        raise ValueError(
            f"_TraceableMelSpectrogram differs from this checkpoint's own MelSpectrogram by "
            f"{deviation} (max |mel| = {want.abs().max().item()}). The replacement exists to be "
            f"numerically the same frontend, so a difference here is a defect in it, not a tolerance "
            f"to widen."
        )
    featurizer[0] = replacement


@dataclass
class ASRGigaAMExportConfig(BaseTransducerExportConfig):
    """GigaAM v3's RNN-T variants: an HF directory loaded through its own `trust_remote_code` modeling
    file, exported by the same template the `.nemo` transducers use."""

    architecture: str = "gigaam"
    output_path: str = "gigaam.gguf"

    def prepare_environment(self) -> None:
        # transformers' hf-hub version gate, the same stub `causal_lm_export`, `whisper_export` and
        # `prepare_nemo_environment` install. Nothing else: the checkpoint is a plain directory, so
        # none of NeMo's untar-into-TMPDIR handling applies here.
        mock_dep = types.ModuleType("dependency_versions_check")
        mock_dep.dep_version_check = lambda *args, **kwargs: None
        sys.modules.setdefault("transformers.dependency_versions_check", mock_dep)

    def load_model(self):
        """`AutoModel.from_pretrained(..., trust_remote_code=True)`, unwrapped to the real model, with
        the two trace preparations this module's docstring explains.

        The unwrap is not incidental. `GigaAMModel` is a thin `PreTrainedModel` shell whose `forward`
        forwards to `self.model`; the family template needs the `GigaAMASR` inside it, because that is
        what carries `preprocessor`, `encoder`, `head` and the `cfg` every structural check reads.
        """
        from transformers import AutoModel

        print(f"Loading GigaAM model from {self.checkpoint}...")
        model = AutoModel.from_pretrained(self.checkpoint, trust_remote_code=True).eval().model
        # Eagerly, and NOT under inference_mode -- see the module docstring's point 1.
        encoder = model.encoder
        encoder.pos_enc.extend_pe(encoder.pos_emb_max_len, torch.device("cpu"))
        _replace_mel_frontend(model, int(model.cfg.preprocessor.sample_rate))
        return model

    def encoder_wrapper(self, model):
        return ASREncoderWrapper(model, self.output, input_names=GIGAAM_ENCODER_INPUT_NAMES)

    def transducer_parts(self, model) -> TransducerParts:
        """GigaAM's layout: `RNNTHead` holds both halves, and the prediction LSTM is directly on the
        decoder rather than behind a `dec_rnn` wrapper.

        No durations and no declared joint width: `modeling_gigaam.RNNTJoint` is a plain RNN-T joint
        with no `num_classes_with_blank` and the config has no `model_defaults` at all, so the token
        count comes off the embedding and the joint's width is checked against that alone.
        """
        head = model.head
        return TransducerParts(embed=head.decoder.embed, lstm=head.decoder.lstm, joint=head.joint)

    def tokenizer_dir(self):
        """The checkpoint's SentencePiece model sits in the directory under exactly the name
        `LoomGGUFExporter._write_tokenizer`'s `sentencepiece_proto` family looks for, so unlike the
        `.nemo` archives there is no adapter to write here -- just the directory, when it has one."""
        if (Path(self.checkpoint) / "tokenizer.model").is_file():
            return self.checkpoint
        return None


def _gigaam_model_cfg(path: Path) -> Optional[dict]:
    """The `cfg.model.cfg` block of an HF directory declaring `model_type == "gigaam"`, or None for
    anything else. Never raises: `detect()` runs against unidentified paths by construction."""
    config_path = path / "config.json"
    if not path.is_dir() or not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(config, dict) or config.get("model_type") != "gigaam":
        return None
    inner = ((config.get("cfg") or {}).get("model") or {}).get("cfg")
    return inner if isinstance(inner, dict) else None


def _is_gigaam_rnnt(path: Path) -> bool:
    """A real structural check (BACKLOG.md P3.2): a GigaAM directory whose head is a transducer.

    `model_class` is the checkpoint's own word for which head it carries, and it is the discriminator
    that matters here -- `rnnt` and `e2e_rnnt` share this export and the `ctc` variants do not (see the
    module docstring). Reading it also keeps this recognizer from colliding with a future CTC one.
    """
    cfg = _gigaam_model_cfg(path)
    return cfg is not None and cfg.get("model_class") == "rnnt"


def _build_gigaam_rnnt(path: Path, output_path: str) -> ASRGigaAMExportConfig:
    return ASRGigaAMExportConfig(checkpoint=str(path), output_path=output_path)


def register(registry) -> None:
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=ASRGigaAMExportConfig,
        recognizers=[ModelRecognizer(name="gigaam-rnnt", detect=_is_gigaam_rnnt,
                                     build_config=_build_gigaam_rnnt)],
    ))
