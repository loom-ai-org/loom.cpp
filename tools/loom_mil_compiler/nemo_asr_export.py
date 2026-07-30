"""Generalizes the "NeMo ASR encoder" export family -- the third family template, alongside
`modular_export.py`'s `ModularExportSpec` (decoder-LLMs) and `iterative_export.py`'s
`IterativeRefinementSpec` (Euler-CFM samplers).

`export_conformer_ctc_mil.py` (93 lines), `export_parakeet_tdt_mil.py` (97) and
`export_parakeet_rnnt_mil.py` (91) were three near-identical scripts: mock out transformers' version
gate, restore a `.nemo` checkpoint, wrap the model so its multi-value `forward` returns exactly one
tensor, trace it on 1 s of dummy audio, convert with a `RangeDim` sample axis and FLOAT32 precision,
and hand the MIL program to `LoomGGUFBackend`. BACKLOG.md listed five things that differ between them.
Building this template found only **three** of the five to be real:

* `checkpoint`, `architecture` and `output_path` genuinely vary -- they are the whole spec.
* the **restore class does not vary**. `EncDecCTCModel.restore_from` and `ASRModel.restore_from` return
  the *identical* concrete class (`EncDecCTCModelBPE`) for the Conformer-CTC checkpoint, with identical
  state-dict keys -- NeMo's `restore_from` dispatches on the config's own `target`, so naming a subclass
  never selected anything. Checked directly against the real checkpoint rather than assumed; the field
  is gone, and `ASRModel` is used for all three.
* the **wrapper's return value is derived from the checkpoint, not declared free-form**. `EncoderOutput`
  names the *reason* the two wrappers differ (a CTC model's decoder is one 1x1 conv, cheap to keep in
  the graph; an RNNT/TDT decoder is an autoregressive LSTM + joint that stays host-side, so only the
  encoder is traced), and the claim is cross-checked against the real model -- see
  `EncoderOutput.validate`.

Everything else is family-wide and re-derived here rather than declared:

* **the sample rate** comes from `model.cfg.preprocessor.sample_rate`, and the dummy trace length
  (1 s) and the `RangeDim` bounds (0.1 s .. 20 s, matching NeMo's own min/max_duration convention) are
  computed from it, instead of three copies of the literals 16000 / 1600 / 320000.
* **`compute_precision=FLOAT32` is not optional and is not a per-model choice.** `ct.convert`'s own
  default (`compute_precision=None`) applies an FP16 cast to every constant weight EVEN for
  `convert_to="milinternal"` (`_need_fp16_cast_pass(None, "milinternal")` returns True in coremltools'
  own `_converters_entry.py`). That was root-caused as the real cause of the ~4-30x magnitude divergence
  in Conformer-CTC's second (176-channel) CONV_2D subsampling stage -- that stage sums over 176*3*3=1584
  taps, so per-weight fp16 rounding (~1e-2 relative) accumulates into visible output error; confirmed by
  reading the exported GGUF's own stored weight bytes and finding they exactly equalled
  `real_weight.astype(np.float16).astype(np.float32)`, bit for bit. See BACKLOG.md.
* **the `TMPDIR` trap.** NeMo's `restore_from()` untars a multi-GB checkpoint into `TMPDIR`, and `/` has
  ~2 GB free on this machine -- left at its default, the export dies with `OSError: No space left on
  device` partway through. Handled once here, for every model in the family.

The point, as with the other two templates, is **not** line count (item 4 of the improvement thread
measured that going *up*). It is that the spec's claims are checked against the real model at export
time and raise naming the exact mismatch: pointing an `EncoderOutput.CTC_LOG_PROBS` spec at an RNNT
checkpoint raises before the encoder is traced at all, instead of producing a GGUF whose "log_probs"
output is silently an encoder activation.
"""
import os
import sys
import tempfile
import types
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

# Dummy trace length, and the dynamic sample-axis bounds, all in SECONDS -- turned into sample counts
# against the checkpoint's own declared sample rate. 0.1 s .. 20 s matches NeMo's own
# min_duration/max_duration convention for these models.
TRACE_SECONDS = 1.0
MIN_SECONDS = 0.1
MAX_SECONDS = 20.0

# NeMo's restore_from() untars the (multi-GB) .nemo into a tempfile.mkdtemp() dir; the default /tmp is
# too small on this machine (see the module docstring and the env_disk_space_tmp note in BACKLOG.md).
DEFAULT_NEMO_TMPDIR = "/home/flavio/.claude/tmp/nemo_extract"


def prepare_nemo_environment(tmpdir: str = DEFAULT_NEMO_TMPDIR):
    """Everything that must happen BEFORE `import nemo.collections.asr`, in one place.

    Two unrelated workarounds, both previously copy-pasted at the top of all three export scripts:

    1. `transformers.dependency_versions_check` is stubbed out to bypass the library's hf-hub version
       bounds check -- needed because `nemo.collections.asr` eagerly imports transformers transitively
       through torchmetrics. Same stub as `export_hf_causal_lm.py` installs.
    2. `TMPDIR` is routed to a filesystem with room for an untarred checkpoint. Both the environment
       variable *and* `tempfile.tempdir` are set: `tempfile` memoizes its default directory the first
       time anything in the process asks for one, so setting only the env var is a race against
       whichever import got there first.
    """
    mock_dep = types.ModuleType("dependency_versions_check")
    mock_dep.dep_version_check = lambda *args, **kwargs: None
    sys.modules.setdefault("transformers.dependency_versions_check", mock_dep)

    os.makedirs(tmpdir, exist_ok=True)
    os.environ.setdefault("TMPDIR", tmpdir)
    # Unconditional, matching the docstring above: TensorFlow (imported transitively by coremltools,
    # before this function ever runs) already calls tempfile.gettempdir() during its own import,
    # memoizing tempfile.tempdir to the OS default ("/tmp") -- confirmed directly (`tempfile.tempdir`
    # is already "/tmp" by the time this line runs, for every one of this family's exports, not just
    # sometimes). A `None`-guarded assignment therefore never actually overrides it, silently losing
    # the exact race this function exists to win. Conformer-CTC-small's own checkpoint happens to be
    # small enough to unpack within `/tmp`'s ~2 GB regardless, which is why this went unnoticed; both
    # Parakeet checkpoints (~2.4 GB each) do not fit and fail with ENOSPC.
    tempfile.tempdir = os.environ["TMPDIR"]


class EncoderOutput(Enum):
    """Which single tensor the traced wrapper reduces NeMo's multi-value `forward` down to.

    Not a free-form "return this expression" field: each member names a *model family*, carries the
    reason that family's graph boundary falls where it does, and knows how to check that the checkpoint
    it was pointed at really is that family (`validate`).
    """

    # EncDecCTCModel(BPE).forward -> (log_probs, encoded_len, greedy_predictions). The CTC decoder is a
    # single 1x1 convolution, so it stays IN the graph; CTC greedy decode + detokenization are host-side
    # (loom::ctc_greedy_decode + loom::Vocab), the same "host logic, not a graph primitive" split every
    # other model here uses.
    CTC_LOG_PROBS = "ctc_log_probs"

    # EncDecRNNTBPEModel.forward -> (encoded, encoded_len), with `encoded` in NeMo's own (B, D, T)
    # convention; transposed here to (B, T, D) to match this project's ne[0]=feature/ne[1]=time GGUF
    # convention. The RNNT/TDT prediction network (2-layer LSTM) + joint + greedy search are NOT traced:
    # they stay the existing hand-derived small topologies (build_lstm_topology/build_joint_topology),
    # driven autoregressively by the C++ TdtDecoder -- the same "encoder graph vs. host-side
    # autoregressive loop" split as every other ASR/LLM model in this project.
    ENCODER_BT_D = "encoder_bt_d"

    @property
    def forward_arity(self) -> int:
        """How many values the real NeMo `forward` returns for this family."""
        return 3 if self is EncoderOutput.CTC_LOG_PROBS else 2

    @property
    def channel_axis(self) -> int:
        """Which axis of the model's OWN (pre-transpose) output carries the channel count. CTC
        log_probs already arrive as (B, T, C); NeMo's encoder emits (B, D, T)."""
        return -1 if self is EncoderOutput.CTC_LOG_PROBS else 1

    @property
    def channel_source(self) -> str:
        """Where the expected channel count is read from, for the error message to name."""
        return ("decoder.num_classes_with_blank" if self is EncoderOutput.CTC_LOG_PROBS
                else "cfg.encoder.d_model")

    def expected_channels(self, model) -> int:
        """The size of the traced output's own channel axis, read off the checkpoint's config rather
        than hardcoded -- the one number that proves the spec named the right family."""
        if self is EncoderOutput.CTC_LOG_PROBS:
            return int(model.decoder.num_classes_with_blank)
        return int(model.cfg.encoder.d_model)

    def select(self, outputs):
        """The wrapper's actual return value, given the real `forward` outputs.

        Deliberately written as a single expression with no intermediate local: `torch.jit.trace` names
        the traced value after the Python name it is bound to, so binding one here renames this
        topology's declared output (confirmed against the golden snapshot -- naming the local `selected`
        turned Parakeet's output from `var_4640` into `selected`, an otherwise byte-identical export).
        Validation therefore inspects the model's own outputs, never this result."""
        if self is EncoderOutput.CTC_LOG_PROBS:
            return outputs[0]
        return outputs[0].transpose(1, 2)

    def validate(self, model, outputs):
        """Cross-checks this claim against what the real model just returned, raising ValueError naming
        the mismatch. Runs inside the wrapper's `forward` -- i.e. exactly once, during tracing -- so it
        costs no extra forward pass and cannot perturb the trace (an out-of-band validation pass would
        consume RNG that a traced constant could depend on)."""
        if not isinstance(outputs, (tuple, list)) or len(outputs) != self.forward_arity:
            n = len(outputs) if isinstance(outputs, (tuple, list)) else 1
            raise ValueError(
                f"NeMoASREncoderSpec declares output={self.name}, whose family's forward() returns "
                f"{self.forward_arity} values, but {type(model).__name__}.forward() returned {n}. "
                f"This checkpoint is not the family the spec claims."
            )
        main = outputs[0]
        if main.dim() != 3:
            raise ValueError(
                f"NeMoASREncoderSpec declares output={self.name}, expecting a rank-3 tensor from "
                f"{type(model).__name__}.forward(), but got rank {main.dim()} ({tuple(main.shape)})."
            )
        expected = self.expected_channels(model)
        if int(main.shape[self.channel_axis]) != expected:
            raise ValueError(
                f"NeMoASREncoderSpec declares output={self.name}, whose axis {self.channel_axis} must "
                f"be {expected} (the checkpoint's own {self.channel_source}), but "
                f"{type(model).__name__} produced {tuple(main.shape)}. This checkpoint is not the "
                f"family the spec claims."
            )


@dataclass
class NeMoASREncoderSpec:
    """Everything that genuinely differs between the three NeMo ASR encoder exports.

    Every other parameter of the export -- restore class, sample rate, trace length, dynamic-axis
    bounds, compute precision, profile -- is either re-derived from the checkpoint or family-wide, and
    lives in this module rather than in three copies.
    """

    # Path to the .nemo checkpoint.
    checkpoint: str
    # Which tensor the traced wrapper returns; see EncoderOutput.
    output: EncoderOutput
    # GGUF `general.architecture` value, and the output .gguf path.
    architecture: str
    output_path: str
    # Only "monolithic" has ever been used for this family (there is no modular boundary to declare:
    # the whole preprocessor+encoder(+CTC decoder) chain is one graph). Declared rather than hardcoded
    # for symmetry with the other family templates' own `profile` field.
    profile: str = "monolithic"

    def validate_against_model(self, model) -> int:
        """Structural checks that don't need a forward pass, run before the (slow) trace. Returns the
        checkpoint's own sample rate, which the trace length and RangeDim bounds are derived from."""
        missing = [attr for attr in ("preprocessor", "encoder") if not hasattr(model, attr)]
        if missing:
            raise ValueError(
                f"NeMoASREncoderSpec({self.architecture!r}) restored {type(model).__name__} from "
                f"{self.checkpoint!r}, which has no {missing} -- this template targets NeMo ASR models "
                f"whose forward is preprocessor -> encoder (-> decoder)."
            )
        sample_rate = getattr(getattr(model.cfg, "preprocessor", None), "sample_rate", None)
        if sample_rate is None:
            raise ValueError(
                f"NeMoASREncoderSpec({self.architecture!r}): {type(model).__name__}'s config declares no "
                f"preprocessor.sample_rate, so the trace length and dynamic-axis bounds cannot be "
                f"derived from the checkpoint."
            )
        # Reading this also validates the spec's own family claim as far as it can be validated without
        # running the model; the rest is checked inside the wrapper during tracing (EncoderOutput.validate).
        self.output.expected_channels(model)
        return int(sample_rate)


class _NeMoASREncoderWrapper(nn.Module):
    """Reduces a NeMo ASR model's multi-value `forward` to the one tensor `spec.output` names, and
    validates that claim against what the model actually returned while doing it."""

    def __init__(self, model, output: EncoderOutput):
        super().__init__()
        self.model = model
        self.output = output

    def forward(self, waveform, length):
        outputs = self.model(input_signal=waveform, input_signal_length=length)
        self.output.validate(self.model, outputs)
        return self.output.select(outputs)


def load_model(spec: NeMoASREncoderSpec):
    """Restores `spec.checkpoint` through the base `ASRModel`, which dispatches on the checkpoint's own
    config `target` -- verified to return the identical concrete class (and identical state-dict keys)
    as naming the subclass directly, which is why the spec has no restore-class field."""
    import nemo.collections.asr as nemo_asr

    print(f"Loading NeMo model from {spec.checkpoint}...")
    model = nemo_asr.models.ASRModel.restore_from(spec.checkpoint, map_location="cpu")
    model.eval()
    return model


def trace_and_convert(spec: NeMoASREncoderSpec, model, sample_rate: int):
    """Traces the wrapper on `TRACE_SECONDS` of dummy audio and converts to a MIL program with a
    dynamic sample axis. Kept separate from `export_nemo_asr_encoder` so a caller can inspect or
    post-process the MIL program before it reaches the backend."""
    import coremltools as ct

    n_samples = int(TRACE_SECONDS * sample_rate)
    dummy_waveform = torch.randn(1, n_samples, dtype=torch.float32)
    dummy_length = torch.tensor([n_samples], dtype=torch.int64)

    print(f"Tracing the complete PyTorch graph (dummy n_samples={n_samples})...")
    traced = torch.jit.trace(_NeMoASREncoderWrapper(model, spec.output), (dummy_waveform, dummy_length))

    print(f"Compiling to GGUF ({spec.profile} profile)...")
    seq_dim = ct.RangeDim(int(MIN_SECONDS * sample_rate), int(MAX_SECONDS * sample_rate))
    return ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="waveform", shape=(1, seq_dim), dtype=np.float32),
            ct.TensorType(name="length", shape=(1,), dtype=np.int32),
        ],
        convert_to="milinternal",
        # Load-bearing, and NOT a per-model choice -- see this module's own docstring for the
        # CONV_2D magnitude-divergence root cause this one line prevents.
        compute_precision=ct.precision.FLOAT32,
    )


def export_nemo_asr_encoder(spec: NeMoASREncoderSpec):
    """The whole export, from `.nemo` on disk to `.gguf` on disk."""
    from .register import LoomGGUFBackend

    prepare_nemo_environment()
    model = load_model(spec)
    sample_rate = spec.validate_against_model(model)
    mil_prog = trace_and_convert(spec, model, sample_rate)

    LoomGGUFBackend()(
        mil_prog,
        output_path=spec.output_path,
        architecture=spec.architecture,
        profile=spec.profile,
    )
    print(f"SUCCESS! Monolithic model exported cleanly to: {spec.output_path}")
    return spec.output_path
