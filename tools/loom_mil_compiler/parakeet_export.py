"""Parakeet TDT/RNN-T as a `MultiPhase` export -- encoder, embedding, prediction LSTM and joint, all
traced, in one GGUF (BACKLOG.md P4.0.17 step 2).

**What this replaces and why it is a different decomposition.** The ASR family's other export
(`nemo_asr_export.ASRNemoEncoderExportConfig`) is `Flattened`: it traces the encoder and nothing else,
because a Transducer's prediction network is an `nn.LSTM` and ggml has no LSTM op. Everything after the
encoder therefore lived in hand-derived topologies written by `tools/convert_nemo/convert_parakeet_tdt.py`
into five separate GGUFs, driven by `src/core/tdt_decoder.cpp`. That left the MIL export of these two
checkpoints *unreachable*: nothing but its own test could run it, since the artifact held only half a
model and carried a driver (the causal-LM one) that raises when called.

Four phases, and the shapes below are read off `parakeet-tdt-0.6b-v3` rather than assumed:

  * `encoder`   -- the same trace the `Flattened` config does, moved into a named phase. Its dynamic
                   axis is `n_samples` (raw audio, never a token count), the one phase here that has
                   a dynamic axis at all.
  * `embed`     -- `nn.Embedding(8193, 640)`. The driver hands it the last emitted label and gets the
                   prediction stack's `layer_input`. Its own phase rather than a node inside the cell
                   topology (which is what the bespoke converter did) because the cell comes from
                   `RecurrentPhase`, whose input is a plain `[input_dim]` vector -- and because an
                   embedding lookup is not part of a recurrence.
  * `pred_lstm` -- `decoder.prediction.dec_rnn.lstm`, `nn.LSTM(640, 640, num_layers=2)`. ONE
                   `RecurrentPhase`: a stack traces to one `lstm` op per layer, so this emits
                   `pred_lstm_l0_fwd` and `pred_lstm_l1_fwd`.
  * `joint`     -- `enc` Linear(1024->640) + `pred` Linear(640->640), summed, ReLU, Linear(640->8198).

**The joint declares its two heads separately, and that is a driver decision made at export time.**
8198 is 8193 token classes plus the 5 TDT durations, concatenated by the checkpoint into one output.
Split into two declared outputs, the driver reduces the token half with `loom.argmax_row('joint', 0)` --
engine-side, nothing marshalled -- and reads only the five duration logits with `loom.get_output`, which
is genuinely host-side. Emitting the concatenated vector instead would have meant marshalling 8198
floats per frame to read five of them and argmax the rest. Plain RNN-T has no duration head, so the
second output simply does not exist and the driver's every-blank-advances-one-frame branch is the one
that runs (`TdtDecoderConfig::durations` empty says the same thing on the C++ side).
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .decomposition import Decomposition, MultiPhase
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase, RecurrentPhase
from .nemo_asr_export import (
    EncoderOutput, _NeMoASREncoderWrapper, build_trace, load_model, prepare_nemo_environment,
)
from .spec_protocol import Unchecked


class _EmbedWrapper(nn.Module):
    """`last_label -> [pred_hidden]`, the prediction stack's input for one step.

    Takes the label as a length-1 i32 tensor and squeezes the batch axis away, so the traced topology
    declares exactly the `[1]`-shaped input the driver has (a Lua array of one number) and the
    `[pred_hidden]`-shaped output the cell topology takes as `layer_input`.
    """

    def __init__(self, embed: nn.Embedding):
        super().__init__()
        self.embed = embed

    def forward(self, last_label):
        return self.embed(last_label).squeeze(0)


class _JointWrapper(nn.Module):
    """`(encoder_frame, decoder_out) -> (token_logits, duration_logits)`.

    NeMo's `RNNTJoint` computes `joint_net(relu(enc(f) + pred(g)))` into one concatenated vector; this
    returns its two halves separately (see the module docstring for why). `n_durations == 0` -- plain
    RNN-T -- returns the whole thing as the single token head, since there is no duration head to split
    off and a zero-width second output is not a thing a topology can declare.
    """

    def __init__(self, joint: nn.Module, n_durations: int):
        super().__init__()
        self.joint = joint
        self.n_durations = n_durations

    def forward(self, encoder_frame, decoder_out):
        combined = self.joint.joint_net(self.joint.enc(encoder_frame) + self.joint.pred(decoder_out))
        if self.n_durations == 0:
            return combined
        return combined[: -self.n_durations], combined[-self.n_durations :]


@dataclass
class ASRParakeetExportConfig(BaseMultiPhaseModelExportConfig):
    """A Transducer ASR checkpoint (Parakeet TDT or plain RNN-T) as four traced phases."""

    checkpoint: str = ""
    # Which tensor the traced encoder wrapper returns. Always the encoder's own (B, T, D) hidden states
    # for a Transducer -- the CTC alternative is a different family with a different driver entirely.
    output: EncoderOutput = EncoderOutput.ENCODER_BT_D
    architecture: str = "parakeet_tdt"
    output_path: str = "parakeet.gguf"
    # Bounds the inner per-frame symbol loop -- the same defensive cap `TdtDecoderConfig` carried, and
    # not part of the TDT algorithm itself.
    max_symbols_per_step: int = 10
    # The traced encoder's own dynamic axis: raw audio samples, never a token count (EXPORT-ROADMAP R1).
    root_axis: str = "n_samples"
    driver_script_path: Path = Path(__file__).resolve().parent / "parakeet_driver"
    decomposition: Decomposition = field(default_factory=lambda: MultiPhase())

    # Filled during `phases()`, which is the only moment the restored checkpoint is in hand. See the
    # `__unchecked__` note: these are READ off the model, never declared.
    # TDT's own duration set, EMPTY for plain RNN-T (no duration head at all) -- the distinction
    # `TdtDecoderConfig::durations` drew, deciding both the joint's output count and the driver's
    # branch. Derived, not declared: read off the RESTORED model in `phases()` rather than by
    # re-opening the archive in `build_config`, so a config can be built (and its components counted by
    # `component_registry.usage`) without a checkpoint on disk.
    durations: tuple = field(default=(), init=False, repr=False)
    blank_id: Optional[int] = field(default=None, init=False, repr=False)
    pred_hidden: Optional[int] = field(default=None, init=False, repr=False)
    num_pred_layers: Optional[int] = field(default=None, init=False, repr=False)

    __unchecked__ = {
        "output": Unchecked(
            "fixed at ENCODER_BT_D by this family: a Transducer's encoder emits hidden states, and the "
            "CTC alternative is a different config with a different driver. EncoderOutput.validate "
            "still cross-checks it against what the real model returns, inside the traced wrapper's "
            "forward -- the only moment those outputs exist."
        ),
        "checkpoint": Unchecked(
            "path to the .nemo archive, already established by the recognizer's own detect(), which "
            "probes the archive's config rather than trusting the extension. load_model raises on "
            "anything it cannot restore."
        ),
        "architecture": Unchecked("the GGUF's own architecture string; it names this export, and there "
                                  "is no second authority to compare it against"),
        "output_path": Unchecked("where to write. A caller's choice, not a claim about the model."),
        "durations": Unchecked(
            "READ off the restored model's `cfg.model_defaults.tdt_durations` in phases(), not declared "
            "-- and cross-checked there against the traced joint's own width, which is the only place "
            "both numbers exist at once."
        ),
        "max_symbols_per_step": Unchecked(
            "a defensive bound on the inner loop, not a property of the checkpoint -- nothing in the "
            "model could disagree with it"
        ),
        "root_axis": Unchecked("checked by the encoder ExportPhase's own Axis link, which is where the "
                               "value is actually used"),
        "driver_script_path": Unchecked("the hand-written TDT loop, adopted whole by "
                                        "MultiPhaseDriverBuilder and cross-checked against the traced "
                                        "topologies by its own parsed call sites"),
        "decomposition": Unchecked("MultiPhase by construction -- this config exists to be one"),
        "blank_id": Unchecked("READ off the restored checkpoint (`joint.num_classes_with_blank - 1`, "
                              "NeMo's own convention) during phases(); a field only because the driver "
                              "and the GGUF hparams need it after the trace"),
        "pred_hidden": Unchecked("same -- the prediction LSTM's own hidden_size"),
        "num_pred_layers": Unchecked("same -- the prediction LSTM's own num_layers"),
    }

    def prepare_environment(self) -> None:
        prepare_nemo_environment()

    def load_model(self):
        return load_model(self)

    def phases(self) -> List[ExportPhase]:
        model = self.load_model()
        # The checkpoint is the authority on its own duration set: `model_defaults.tdt_durations` for
        # TDT, absent entirely for plain RNN-T.
        defaults = getattr(model.cfg, "model_defaults", None) or {}
        self.durations = tuple(defaults.get("tdt_durations") or ())

        pred = model.decoder.prediction
        lstm = pred.dec_rnn.lstm
        self.pred_hidden = int(lstm.hidden_size)
        self.num_pred_layers = int(lstm.num_layers)
        n_durations = len(self.durations)

        # NeMo's blank is the last token class, the same index `TdtDecoderConfig::blank_id` carries.
        #
        # Read off the EMBEDDING, not off `joint.num_classes_with_blank` -- which for a TDT joint is
        # `num_classes + 1 + num_extra_outputs` and therefore already counts the durations (8198 here,
        # not 8193). Using it would have put the blank five classes too high and split the head in the
        # wrong place; the cross-check below is what caught that.
        n_token_classes = int(pred.embed.num_embeddings)
        self.blank_id = n_token_classes - 1

        # The joint's real output width against what the durations claim: the checkpoint states both
        # (`joint_net[-1].out_features` and `model_defaults.tdt_durations`) and only here are both in
        # hand. A mismatch means the recognizer read the wrong duration set, which would otherwise
        # surface as a silently mis-split head -- token logits running into the duration ones.
        out_features = int(model.joint.joint_net[-1].out_features)
        # `joint.num_classes_with_blank` is the joint's OWN total (tokens + blank + durations), so it is
        # the same number as `out_features` and cannot serve as the token count -- asserted rather than
        # commented, since believing otherwise is exactly the mistake this block exists to stop.
        assert int(model.joint.num_classes_with_blank) == out_features
        if out_features != n_token_classes + n_durations:
            raise ValueError(
                f"{type(self).__name__}({self.architecture!r}): the joint emits {out_features} values, "
                f"but the checkpoint declares {n_token_classes} token classes and {n_durations} "
                f"duration(s) ({list(self.durations)}), which sum to "
                f"{n_token_classes + n_durations}. The duration set and the joint disagree."
            )

        sample_rate = self.validate_encoder(model)
        wrapper, dummy_inputs, mil_inputs = build_trace(self, model, sample_rate)
        n_embd = int(model.cfg.encoder.d_model)

        import coremltools as ct

        return [
            ExportPhase(name="encoder", wrapper=wrapper, dummy_inputs=dummy_inputs,
                        mil_inputs=mil_inputs, root_axis=self.root_axis),
            ExportPhase(
                name="embed", wrapper=_EmbedWrapper(pred.embed).eval(),
                dummy_inputs=(torch.tensor([0], dtype=torch.int64),),
                mil_inputs=[ct.TensorType(name="last_label", shape=(1,), dtype=np.int32)],
            ),
            RecurrentPhase(name="pred_lstm", module=lstm),
            ExportPhase(
                name="joint", wrapper=_JointWrapper(model.joint, n_durations).eval(),
                dummy_inputs=(torch.randn(n_embd), torch.randn(self.pred_hidden)),
                mil_inputs=[
                    ct.TensorType(name="encoder_frame", shape=(n_embd,), dtype=np.float32),
                    ct.TensorType(name="decoder_out", shape=(self.pred_hidden,), dtype=np.float32),
                ],
            ),
        ]

    def validate_encoder(self, model) -> int:
        """The encoder-side structural checks and the checkpoint's own sample rate, shared verbatim with
        the `Flattened` config -- the encoder phase here IS that trace."""
        from .nemo_asr_export import ASRNemoEncoderExportConfig

        probe = ASRNemoEncoderExportConfig(
            checkpoint=self.checkpoint, output=self.output,
            architecture=self.architecture, output_path=self.output_path, root_axis=self.root_axis,
        )
        return probe.validate_against_model(model)

    def export_architecture(self) -> str:
        return self.architecture

    def backend_kwargs(self) -> dict:
        kwargs = dict(flat_namespace=True, root_axis=self.root_axis)
        from .nemo_asr_export import extract_nemo_tokenizer_dir

        tokenizer_dir = extract_nemo_tokenizer_dir(self.checkpoint)
        if tokenizer_dir is not None:
            kwargs["tokenizer_dir"] = tokenizer_dir
            kwargs["tokenizer_family"] = "sentencepiece_proto"
        return kwargs

    def driver_components(self) -> List:
        """Parakeet's driver, as components.

        Three of the four topologies are called from the hand-written decode fragment rather than as
        `SubgraphCallComponent`s, and that is the honest split rather than a shortcut: they are called
        from inside a double loop whose trip counts depend on what the joint just emitted, so their call
        sites are *statements inside control flow*, which is precisely what a `LuaFragment` is for. The
        encoder is not -- it runs exactly once, before any of it -- so it stays IR and gets its output
        arity checked. The fragment's own `loom.run_subgraph` call sites are parsed and declared against
        the traced topologies regardless (`parse_run_subgraph_calls`), and the computed per-layer name is
        covered by the `drives` declaration below.
        """
        from .driver_components import (
            ComputedCall, DriverReturn, ExportConstants, LuaFragment, SubgraphCallComponent,
        )
        from .driver_ir import FieldAccess, Len, Var

        fragment = self.driver_script_path
        return [
            LuaFragment(fragment / "00_header.lua", top_level=True),
            ExportConstants(values={
                "BLANK_ID": self.blank_id,
                "N_DURATIONS": len(self.durations),
                "DURATIONS": list(self.durations),
                "PRED_HIDDEN": self.pred_hidden,
                "N_PRED_LAYERS": self.num_pred_layers,
                "MAX_SYMBOLS_PER_STEP": self.max_symbols_per_step,
            }),
            # `_waveform` first: the encoder's own root axis is its length, so it has to be a local
            # before the call that reads `#_waveform`.
            LuaFragment(fragment / "01_inputs.lua", defines=("_waveform",)),
            SubgraphCallComponent(
                topology="encoder", outputs=("_enc",), extra_outputs=("_enc_shape",),
                inputs={"waveform": Var("_waveform"),
                        "length": FieldAccess("inputs", "length")},
                length=Len("_waveform"),
            ),
            LuaFragment(
                fragment / "02_decode.lua",
                reads=("_enc", "_enc_shape", "BLANK_ID", "N_DURATIONS", "DURATIONS", "PRED_HIDDEN",
                       "N_PRED_LAYERS", "MAX_SYMBOLS_PER_STEP"),
                defines=("tokens",),
                drives=(
                    # The per-layer cell name is computed (`'pred_lstm_l' .. (l - 1) .. '_fwd'`), so the
                    # parser cannot resolve it and the declaration says which topologies it reaches.
                    ComputedCall(topologies=tuple(f"pred_lstm_l{i}_fwd"
                                                   for i in range(self.num_pred_layers or 0)),
                                 inputs=("layer_input", "h_prev", "c_prev"),
                                 written="'pred_lstm_l' .. (l - 1) .. '_fwd'"),
                ),
            ),
            LuaFragment(fragment / "03_return.lua", reads=("tokens",)),
        ]
