"""Generalizes the "NeMo ASR encoder" export family -- the third family template, alongside
`modular_export.py`'s `ModularExportSpec` (decoder-LLMs) and `flow_matching_export.py`'s
`FlowMatchingSpec` (Euler-CFM samplers).

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
import tarfile
import tempfile
import types
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml

import numpy as np
import torch
import torch.nn as nn

from .decomposition import Decomposition, Flattened
from .export_config import LoomExportConfig
from .spec_protocol import (
    MODEL, OUTPUTS, Axis, ConfigDerived, NestedSpec, Unchecked, check_links,
)

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

    Its three checks are `ConfigDerived` links (P4.0.5) -- see `__links__` at the bottom of the class.
    They look different from each other but are one shape: read a number the declared family implies,
    measure the same number on what the model actually did, compare.
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
        consume RNG that a traced constant could depend on).

        The three checks are `spec_protocol` links as of P4.0.5 (`EXPORT-PREPARATION.md` stage B.3); see
        `__links__` below. This method stays exactly where it was and keeps its signature, because the
        *timing* is the load-bearing part: the traced forward's real return value exists for one instant
        and only here."""
        check_links(self, model=model, outputs=outputs)

    # Order matters and is the order the hand-written validator ran in: arity first, because a model
    # returning a bare tensor rather than a tuple makes `outputs[0]` mean something entirely different,
    # and rank before channels, because indexing `channel_axis` on a rank-2 tensor reports the wrong
    # number rather than no number.
    __links__ = {
        "forward_arity": ConfigDerived(
            claim=lambda spec, ctx: spec.forward_arity,
            measured=lambda spec, ctx: (
                len(ctx.outputs) if isinstance(ctx.outputs, (tuple, list)) else 1
            ),
            message=(
                "ASRNemoEncoderExportConfig declares output={spec.name}, whose family's forward() "
                "returns {claimed} values, but {model_type}.forward() returned {actual}. "
                "This checkpoint is not the family the spec claims."
            ),
            needs=(MODEL, OUTPUTS),
        ),
        "output_rank": ConfigDerived(
            claim=lambda spec, ctx: 3,
            measured=lambda spec, ctx: ctx.outputs[0].dim(),
            detail=lambda spec, ctx: tuple(ctx.outputs[0].shape),
            message=(
                "ASRNemoEncoderExportConfig declares output={spec.name}, expecting a rank-{claimed} "
                "tensor from {model_type}.forward(), but got rank {actual} ({detail})."
            ),
            needs=(MODEL, OUTPUTS),
        ),
        # The one number that proves the spec named the right family, read off the checkpoint's own
        # config rather than hardcoded -- which is why the message names the config field it came from.
        "output_channels": ConfigDerived(
            claim=lambda spec, ctx: spec.expected_channels(ctx.model),
            measured=lambda spec, ctx: int(ctx.outputs[0].shape[spec.channel_axis]),
            detail=lambda spec, ctx: tuple(ctx.outputs[0].shape),
            message=(
                "ASRNemoEncoderExportConfig declares output={spec.name}, whose axis {spec.channel_axis} "
                "must be {claimed} (the checkpoint's own {spec.channel_source}), but {model_type} "
                "produced {detail}. This checkpoint is not the family the spec claims."
            ),
            needs=(MODEL, OUTPUTS),
        ),
    }


@dataclass(kw_only=True)
class ASRNemoEncoderExportConfig(LoomExportConfig):
    """Everything that genuinely differs between the three NeMo ASR encoder exports (Conformer-CTC,
    Parakeet-TDT, Parakeet-RNNT) -- the ASR family's `LoomExportConfig` (BACKLOG.md's naming convention:
    `ASR` domain, `NemoEncoder` function; renamed from `NeMoASREncoderSpec`, same shape).

    Every other parameter of the export -- restore class, sample rate, trace length, dynamic-axis
    bounds, compute precision -- is either re-derived from the checkpoint or family-wide, and lives in
    this module rather than in three copies. `decomposition` is defaulted rather than accepted: there is
    no modular boundary to declare here (the whole preprocessor+encoder(+CTC decoder) chain is one
    graph), so unlike the causal-LM family this one has no choice to offer -- see decomposition.py on
    structural vs. chosen decompositions.
    """

    # Path to the .nemo checkpoint.
    checkpoint: str
    # Which tensor the traced wrapper returns; see EncoderOutput.
    output: EncoderOutput
    decomposition: Decomposition = field(default_factory=Flattened)
    # EXPORT-ROADMAP.md R1: "waveform"'s own axis is raw audio samples, never a token count --
    # family-wide for all three models this template covers (axes.py's N_SAMPLES). A field rather than
    # the literal `backend_kwargs()` used to return, so it is a *declaration* the Axis link can check
    # (P4.0.5, stage B.5); the value and the resulting export are unchanged.
    root_axis: str = "n_samples"
    # The traced output's own channel count, captured during `validate_against_model` (which reads it
    # off the checkpoint anyway) so `backend_kwargs` can derive the CTC blank id from it. Not an init
    # argument: it is read from the model, never declared by a caller.
    num_classes: Optional[int] = field(default=None, init=False, repr=False)

    __links__ = {
        "root_axis": Axis(),
        "output": NestedSpec(
            where="EncoderOutput.validate, called from _NeMoASREncoderWrapper.forward -- the traced "
                  "forward's real return value exists for one instant during tracing and nowhere else, "
                  "so the checker cannot reach it from here"
        ),
    }
    __unchecked__ = {
        "num_classes": Unchecked(
            "READ off the restored checkpoint (`EncoderOutput.expected_channels`) during "
            "validate_against_model, not declared -- so there is nothing for a checkpoint to disagree "
            "with. It is a field only because `backend_kwargs()` is handed no model and runs after "
            "the trace."
        ),
        "checkpoint": Unchecked(
            "path to the .nemo archive. Already established by the recognizer's own detect(), which "
            "probes the archive's config rather than trusting the extension, and load_model raises on "
            "anything it cannot restore. A 'this path exists' link would check the weaker property and "
            "read as if it checked the stronger one."
        ),
    }

    def validate_against_model(self, model) -> int:
        """Structural checks that don't need a forward pass, run before the (slow) trace. Returns the
        checkpoint's own sample rate, which the trace length and RangeDim bounds are derived from."""
        missing = [attr for attr in ("preprocessor", "encoder") if not hasattr(model, attr)]
        if missing:
            raise ValueError(
                f"ASRNemoEncoderExportConfig({self.architecture!r}) restored {type(model).__name__} from "
                f"{self.checkpoint!r}, which has no {missing} -- this template targets NeMo ASR models "
                f"whose forward is preprocessor -> encoder (-> decoder)."
            )
        sample_rate = getattr(getattr(model.cfg, "preprocessor", None), "sample_rate", None)
        if sample_rate is None:
            raise ValueError(
                f"ASRNemoEncoderExportConfig({self.architecture!r}): {type(model).__name__}'s config "
                f"declares no preprocessor.sample_rate, so the trace length and dynamic-axis bounds "
                f"cannot be derived from the checkpoint."
            )
        # Reading this also validates the spec's own family claim as far as it can be validated without
        # running the model; the rest is checked inside the wrapper during tracing (EncoderOutput.validate).
        # Kept rather than discarded: for a CTC head it IS `decoder.num_classes_with_blank`, which is
        # where the driver's blank id comes from (BACKLOG.md P4.0.17).
        self.num_classes = int(self.output.expected_channels(model))
        return int(sample_rate)

    def prepare_environment(self) -> None:
        prepare_nemo_environment()

    def load_model(self):
        return load_model(self)

    def build_trace(self, model):
        """`Flattened`'s hook. The structural validation runs here rather than in `load_model` because
        it also yields the sample rate the trace length and RangeDim bounds are derived from."""
        sample_rate = self.validate_against_model(model)
        return build_trace(self, model, sample_rate)

    def synthesized_builder_key(self) -> str:
        """Which `driver_components.SYNTHESIZED_BUILDERS` entry assembles this config's driver.

        **The hook exists because this family is the counterexample to the usual rule** (BACKLOG.md
        P4.0.17). `Decomposition.driver_builder` holds that "the orchestration shape a driver has is a
        property of how the model was decomposed", and for a CTC encoder it is not: this is a
        `Flattened` export exactly like Qwen3, and what differs is entirely what the host does with the
        one output -- reduce every frame and collapse, rather than reduce one row. So the family says
        so, and both readers of the answer -- `LoomGGUFExporter` (through `backend_kwargs`) and
        `component_registry.usage()` (directly) -- take it from here rather than each inferring it.

        The RNNT encoders keep the default: they emit an encoder tensor whose consumer is still the C++
        TDT loop, which is step 2 of that item and not this one.
        """
        if self.output is EncoderOutput.CTC_LOG_PROBS:
            return "CtcGreedy"
        return type(self.decomposition).__name__

    def backend_kwargs(self) -> dict:
        kwargs = dict(flat_namespace=True, root_axis=self.root_axis,
                      driver_builder=self.synthesized_builder_key())
        # A CTC head's blank is its LAST class -- NeMo's own convention, and the same index
        # `loom_cli`'s C++ path passes to `ctc_greedy_decode` (`num_classes - 1`).
        #
        # Omitted rather than raised when the class count was never read, because this method is also
        # called by callers that never trace (`component_registry.usage()` builds every registered
        # config to attribute components to models). The export path cannot slip through: the exporter
        # raises when it is asked for the CTC builder without a blank id, which is the moment the
        # number is actually needed.
        if self.output is EncoderOutput.CTC_LOG_PROBS and self.num_classes is not None:
            kwargs["ctc_blank_id"] = self.num_classes - 1
        return kwargs


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


def load_model(spec: ASRNemoEncoderExportConfig):
    """Restores `spec.checkpoint` through the base `ASRModel`, which dispatches on the checkpoint's own
    config `target` -- verified to return the identical concrete class (and identical state-dict keys)
    as naming the subclass directly, which is why the spec has no restore-class field."""
    import nemo.collections.asr as nemo_asr

    print(f"Loading NeMo model from {spec.checkpoint}...")
    model = nemo_asr.models.ASRModel.restore_from(spec.checkpoint, map_location="cpu")
    model.eval()
    return model


def build_trace(spec: ASRNemoEncoderExportConfig, model, sample_rate: int):
    """The wrapper, `TRACE_SECONDS` of dummy audio, and the MIL input declarations with a dynamic sample
    axis -- `Flattened` does the trace and the `ct.convert` itself (including the load-bearing
    FLOAT32 compute precision this module's own docstring explains)."""
    import coremltools as ct

    n_samples = int(TRACE_SECONDS * sample_rate)
    dummy_inputs = (
        torch.randn(1, n_samples, dtype=torch.float32),
        torch.tensor([n_samples], dtype=torch.int64),
    )
    print(f"Tracing the complete PyTorch graph (dummy n_samples={n_samples})...")

    seq_dim = ct.RangeDim(int(MIN_SECONDS * sample_rate), int(MAX_SECONDS * sample_rate))
    mil_inputs = [
        ct.TensorType(name="waveform", shape=(1, seq_dim), dtype=np.float32),
        ct.TensorType(name="length", shape=(1,), dtype=np.int32),
    ]
    return _NeMoASREncoderWrapper(model, spec.output), dummy_inputs, mil_inputs


def export_nemo_asr_encoder(spec: ASRNemoEncoderExportConfig):
    """The whole export, from `.nemo` on disk to `.gguf` on disk. Thin shim over
    `ASRNemoEncoderExportConfig.export()`, kept as a module-level function for existing callers."""
    return spec.export()


def _read_nemo_model_config(path: Path) -> dict:
    """Reads a `.nemo` archive's embedded `model_config.yaml` WITHOUT restoring the checkpoint (no
    `ASRModel.restore_from`, no untar-to-tempdir) -- cheap enough to call once per recognizer during
    detection."""
    with tarfile.open(path) as t:
        f = t.extractfile("./model_config.yaml")
        return yaml.safe_load(f.read())


def _is_nemo_archive(path: Path) -> bool:
    return path.is_file() and path.suffix == ".nemo"


def _is_conformer_ctc(path: Path) -> bool:
    """Real structural check (BACKLOG.md P3.2): `target` is unambiguous for this family --
    `EncDecCTCModelBPE`, confirmed against the real checkpoint (see this module's own docstring: the
    restore class doesn't vary, `ASRModel.restore_from` dispatches on this same field)."""
    if not _is_nemo_archive(path):
        return False
    cfg = _read_nemo_model_config(path)
    return str(cfg.get("target", "")).endswith("EncDecCTCModelBPE")


def _is_parakeet_tdt(path: Path) -> bool:
    """Parakeet-TDT and Parakeet-RNNT both restore through `EncDecRNNTBPEModel` -- `target` alone can't
    tell them apart (confirmed against both real checkpoints). The real secondary discriminator, found by
    reading both checkpoints' own `model_config.yaml`: TDT's `model_defaults` declares `tdt_durations`
    (and `num_tdt_durations`); plain RNNT's `model_defaults` has neither key at all."""
    if not _is_nemo_archive(path):
        return False
    cfg = _read_nemo_model_config(path)
    if not str(cfg.get("target", "")).endswith("EncDecRNNTBPEModel"):
        return False
    return "tdt_durations" in (cfg.get("model_defaults") or {})


def _is_parakeet_rnnt(path: Path) -> bool:
    if not _is_nemo_archive(path):
        return False
    cfg = _read_nemo_model_config(path)
    if not str(cfg.get("target", "")).endswith("EncDecRNNTBPEModel"):
        return False
    return "tdt_durations" not in (cfg.get("model_defaults") or {})


def _build_conformer_ctc(path: Path, output_path: str):
    return ASRNemoEncoderExportConfig(
        checkpoint=str(path), output=EncoderOutput.CTC_LOG_PROBS,
        architecture="conformer-ctc", output_path=output_path,
    )


def _build_parakeet_tdt(path: Path, output_path: str):
    return ASRNemoEncoderExportConfig(
        checkpoint=str(path), output=EncoderOutput.ENCODER_BT_D,
        architecture="parakeet-tdt-encoder", output_path=output_path,
    )


def _build_parakeet_rnnt(path: Path, output_path: str):
    return ASRNemoEncoderExportConfig(
        checkpoint=str(path), output=EncoderOutput.ENCODER_BT_D,
        architecture="parakeet-rnnt-encoder", output_path=output_path,
    )


def register(registry) -> None:
    """Registers this family's `TaskRegistryEntry` (BACKLOG.md P3.2)."""
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="automatic-speech-recognition",
        config_class=ASRNemoEncoderExportConfig,
        recognizers=[
            ModelRecognizer(name="conformer-ctc", detect=_is_conformer_ctc, build_config=_build_conformer_ctc),
            ModelRecognizer(name="parakeet-tdt", detect=_is_parakeet_tdt, build_config=_build_parakeet_tdt),
            ModelRecognizer(name="parakeet-rnnt", detect=_is_parakeet_rnnt, build_config=_build_parakeet_rnnt),
        ],
    ))
