"""The Transducer ASR family: a Conformer encoder plus an RNN-T/TDT head, as four traced phases in one
GGUF with a driver that decodes (BACKLOG.md P4.0.17 step 2, generalized past NeMo by P4.2).

**What this is, and what the two leaves under it are.** `EXPORT-ROADMAP.md` R5's family 1 is "Conformer
encoder + CTC / TDT / RNN-T head". The CTC half is `nemo_asr_export.ASRNemoEncoderExportConfig`, a
`Flattened` export whose one graph runs end to end. The transducer half cannot be: a prediction network
is an `nn.LSTM`, ggml has no LSTM op, and the decode is a double loop whose trip counts depend on what
the joint just emitted. So it is `MultiPhase` -- and this module is that shape, with the two things that
genuinely vary between checkpoints pushed onto the leaves:

  * **how the checkpoint is loaded.** `ASRParakeetExportConfig` restores a `.nemo` archive through
    `nemo.collections.asr.models.ASRModel.restore_from`; `ASRGigaAMExportConfig` loads an HF directory
    through `AutoModel.from_pretrained(..., trust_remote_code=True)`. Neither fact reaches this module.
  * **where the checkpoint keeps its transducer.** NeMo puts the prediction network at
    `model.decoder.prediction` and the joint at `model.joint`; GigaAM puts both under `model.head`.
    That is `transducer_parts()`, one method per leaf, returning a `TransducerParts`.

Everything else -- the four phases, their shapes, the blank-id derivation, the joint/duration
cross-check, the driver's component list -- is family-wide and lives here once. That split is what P4.2
existed to establish (`EXPORT-ROADMAP.md` R3: "the template's real contract is *give me an nn.Module and
tell me what its forward returns*, and which library restored it belongs in a per-model loader entry"),
and it is established by GigaAM reusing this file rather than by any claim inside it.

Four phases, and the shapes below are read off `parakeet-tdt-0.6b-v3` (GigaAM v3's are in
`gigaam_export.py`) rather than assumed:

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
                   `pred_lstm_l0_fwd` and `pred_lstm_l1_fwd`. `number_layers=True`, because the depth
                   is a per-checkpoint number (GigaAM's prediction network is one layer) and the
                   driver's loop addresses the cells by index either way.
  * `joint`     -- `enc` Linear(1024->640) + `pred` Linear(640->640), summed, ReLU, Linear(640->8198).

**The joint declares its two heads separately, and that is a driver decision made at export time.**
8198 is 8193 token classes plus the 5 TDT durations, concatenated by the checkpoint into one output.
Split into two declared outputs, the driver reduces the token half with `loom.argmax_row('joint', 0)` --
engine-side, nothing marshalled -- and reads only the five duration logits with `loom.get_output`, which
is genuinely host-side. Emitting the concatenated vector instead would have meant marshalling 8198
floats per frame to read five of them and argmax the rest. Plain RNN-T has no duration head, so the
second output simply does not exist and the driver's every-blank-advances-one-frame branch is the one
that runs (`TdtDecoderConfig::durations` empty said the same thing on the C++ side).

**Neither wrapper applies the joint's log-softmax, and both checkpoints have one.** NeMo's
`RNNTJoint.joint()` optionally log-softmaxes its result and GigaAM's always does; `_JointWrapper` calls
`joint_net` directly and skips it in both cases. A log-softmax is monotone, so the argmax that selects
the emitted token and the argmax over the duration head are both unchanged -- and these logits are never
compared against a threshold or summed into a score, because greedy decoding is the only thing this
family's driver does. Parakeet's e2e test agreeing with NeMo token-for-token is what checks it.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .decomposition import Decomposition, MultiPhase
from .multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase, RecurrentPhase
from .nemo_asr_export import ASREncoderWrapper, EncoderOutput, build_trace
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

    Both families' joints compute `joint_net(relu(enc(f) + pred(g)))` into one concatenated vector --
    NeMo's `RNNTJoint` and GigaAM's `modeling_gigaam.RNNTJoint` are the same three modules under the
    same three names; this returns its two halves separately (see the module docstring for why).
    `n_durations == 0` -- plain RNN-T -- returns the whole thing as the single token head, since there
    is no duration head to split off and a zero-width second output is not a thing a topology can
    declare.
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
class TransducerParts:
    """Where a restored checkpoint keeps the three modules after its encoder, plus the two numbers only
    the checkpoint can state about them. One per leaf, built in `transducer_parts()`.

    This is a *reading* of a real model, not a declaration about one -- which is why every field is
    `Unchecked` here and the checking happens in `phases()`, the one place where the numbers read off
    the config and the widths of the traced modules are both in hand.
    """

    # `nn.Embedding(n_token_classes, pred_hidden)`. Also the authority on the token-class count, and
    # therefore on the blank id -- see `phases()` for why the joint's own width is not.
    embed: nn.Embedding
    # The prediction network, `nn.LSTM(pred_hidden, pred_hidden, num_layers=N)`. One phase whatever N is.
    lstm: nn.LSTM
    # The joint: any module exposing `enc`, `pred` and `joint_net`, which both families' do.
    joint: nn.Module
    # TDT's own duration set, EMPTY for plain RNN-T (no duration head at all) -- the distinction
    # `TdtDecoderConfig::durations` drew, deciding both the joint's output count and the driver's branch.
    durations: Tuple[int, ...] = ()
    # The checkpoint's OWN statement of the joint's total output width, where it makes one (NeMo's
    # `joint.num_classes_with_blank`). `None` for a checkpoint that states no such number, which GigaAM
    # v3 does not. Cross-checked against the real `joint_net` width in `phases()`.
    declared_joint_width: Optional[int] = None

    __unchecked__ = {
        "embed": Unchecked(
            "the real `nn.Embedding` off the restored model. It IS the module being traced -- there is "
            "no second authority to compare it against, which is what the trace and the family's e2e "
            "test are for"
        ),
        "lstm": Unchecked("same -- the real prediction network, traced by RecurrentPhase, which raises "
                          "if it is not an nn.LSTM at all"),
        "joint": Unchecked("same -- the real joint. Its three submodules are read by `_JointWrapper`, "
                           "and a checkpoint missing one raises an AttributeError naming it"),
        "durations": Unchecked(
            "READ off the restored model (`cfg.model_defaults.tdt_durations` for NeMo, absent for a "
            "plain RNN-T), not declared -- and cross-checked in `phases()` against the traced joint's "
            "own width, which is the only place both numbers exist at once"
        ),
        "declared_joint_width": Unchecked(
            "the SECOND reading of a number the checkpoint states twice, supplied so `phases()` can "
            "compare the two. It is the input to a check, not a claim -- and `None` (the checkpoint "
            "states it once) is a legitimate answer rather than a missing one"
        ),
    }


@dataclass
class BaseTransducerExportConfig(BaseMultiPhaseModelExportConfig):
    """A Transducer ASR checkpoint (RNN-T or TDT) as four traced phases.

    Subclasses supply `load_model()`, `transducer_parts(model)` and -- when their checkpoint's forward
    names its inputs differently from NeMo's -- `encoder_wrapper(model)`. Nothing else.
    """

    checkpoint: str = ""
    # Which tensor the traced encoder wrapper returns. Always the encoder's own (B, T, D) hidden states
    # for a Transducer -- the CTC alternative is a different family with a different driver entirely.
    output: EncoderOutput = EncoderOutput.ENCODER_BT_D
    architecture: str = "transducer"
    output_path: str = "transducer.gguf"
    # Bounds the inner per-frame symbol loop -- the same defensive cap `TdtDecoderConfig` carried, and
    # not part of the TDT algorithm itself.
    max_symbols_per_step: int = 10
    # The traced encoder's own dynamic axis: raw audio samples, never a token count (EXPORT-ROADMAP R1).
    root_axis: str = "n_samples"
    driver_script_path: Path = Path(__file__).resolve().parent / "transducer_driver"
    decomposition: Decomposition = field(default_factory=lambda: MultiPhase())

    # Filled during `phases()`, which is the only moment the restored checkpoint is in hand. See the
    # `__unchecked__` note: these are READ off the model, never declared.
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
            "the path this family's loader is pointed at -- a `.nemo` archive for Parakeet, an HF "
            "directory for GigaAM -- already established by the recognizer's own detect(), which probes "
            "the checkpoint's own config rather than trusting the path shape. load_model raises on "
            "anything it cannot restore."
        ),
        "architecture": Unchecked("the GGUF's own architecture string; it names this export, and there "
                                  "is no second authority to compare it against"),
        "output_path": Unchecked("where to write. A caller's choice, not a claim about the model."),
        "durations": Unchecked(
            "READ off the restored model by `transducer_parts()`, not declared -- and cross-checked in "
            "phases() against the traced joint's own width, which is the only place both numbers exist "
            "at once."
        ),
        "max_symbols_per_step": Unchecked(
            "a defensive bound on the inner loop, not a property of the checkpoint -- nothing in the "
            "model could disagree with it"
        ),
        "root_axis": Unchecked("checked by the encoder ExportPhase's own Axis link, which is where the "
                               "value is actually used"),
        "driver_script_path": Unchecked("the hand-written transducer decode loop, adopted whole by "
                                        "MultiPhaseDriverBuilder and cross-checked against the traced "
                                        "topologies by its own parsed call sites"),
        "decomposition": Unchecked("MultiPhase by construction -- this config exists to be one"),
        "blank_id": Unchecked("READ off the restored checkpoint (the prediction embedding's row count "
                              "minus one, both families' own convention) during phases(); a field only "
                              "because the driver needs it after the trace"),
        "pred_hidden": Unchecked("same -- the prediction LSTM's own hidden_size"),
        "num_pred_layers": Unchecked("same -- the prediction LSTM's own num_layers"),
    }

    # -- what a leaf supplies: a loader, a layout, a vocabulary, and (rarely) two argument names -------

    def transducer_parts(self, model) -> TransducerParts:
        """Where THIS checkpoint keeps its embedding, prediction LSTM and joint, and what it says about
        its duration set and its joint's width. The one method that knows a layout."""
        raise NotImplementedError

    def encoder_wrapper(self, model):
        """The module traced as the encoder phase. Defaults to NeMo's forward argument names, which is
        what `ASREncoderWrapper` defaults to; see `gigaam_export` for the override."""
        return ASREncoderWrapper(model, self.output)

    def tokenizer_dir(self) -> Optional[str]:
        """A directory holding this checkpoint's `tokenizer.model`, or `None` for one that carries no
        vocabulary -- which is what makes the exported GGUF detokenizable on its own.

        A hook because the two leaves keep the same SentencePiece proto in different places, not because
        they want different tokenizers: GigaAM's sits in the model directory under exactly the name
        `LoomGGUFExporter._write_tokenizer` looks for, and a `.nemo` archive's is content-hashed inside
        a tarball and has to be extracted first.

        `None` is a normal answer, not a failure: `component_registry.usage()` builds every registered
        config against a path that does not exist.
        """
        raise NotImplementedError

    # -- everything else is family-wide --------------------------------------------------------------

    def phases(self) -> List[ExportPhase]:
        model = self.load_model()
        parts = self.transducer_parts(model)

        self.durations = tuple(parts.durations)
        self.pred_hidden = int(parts.lstm.hidden_size)
        self.num_pred_layers = int(parts.lstm.num_layers)
        n_durations = len(self.durations)

        # The blank is the last token class, the same index `TdtDecoderConfig::blank_id` carried and the
        # same one `modeling_gigaam.RNNTDecoder` computes for itself.
        #
        # Read off the EMBEDDING, not off the joint's declared width -- which for a NeMo TDT joint is
        # `num_classes + 1 + num_extra_outputs` and therefore already counts the durations (8198 here,
        # not 8193). Using it would have put the blank five classes too high and split the head in the
        # wrong place; the cross-check below is what caught that.
        n_token_classes = int(parts.embed.num_embeddings)
        self.blank_id = n_token_classes - 1

        # The joint's real output width against what the durations claim: the checkpoint states both and
        # only here are both in hand. A mismatch means the leaf read the wrong duration set, which would
        # otherwise surface as a silently mis-split head -- token logits running into the duration ones.
        out_features = int(parts.joint.joint_net[-1].out_features)
        # Where the checkpoint states its joint's total a second time, the two readings are compared
        # rather than one of them being trusted. This is the assertion that stops the mistake above from
        # being made again from the other direction: a NeMo joint's `num_classes_with_blank` is the
        # joint's OWN total (tokens + blank + durations), so it equals `out_features` and cannot serve
        # as the token count.
        if parts.declared_joint_width is not None and parts.declared_joint_width != out_features:
            raise ValueError(
                f"{type(self).__name__}({self.architecture!r}): the checkpoint declares a joint width "
                f"of {parts.declared_joint_width}, but its joint_net emits {out_features}. Those are "
                f"two readings of one number and they disagree."
            )
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
                name="embed", wrapper=_EmbedWrapper(parts.embed).eval(),
                dummy_inputs=(torch.tensor([0], dtype=torch.int64),),
                mil_inputs=[ct.TensorType(name="last_label", shape=(1,), dtype=np.int32)],
            ),
            # Numbered even at depth 1: the driver's loop addresses the cells by index, so the naming
            # must not depend on how deep this checkpoint's prediction network happens to be.
            RecurrentPhase(name="pred_lstm", module=parts.lstm, number_layers=True),
            ExportPhase(
                name="joint", wrapper=_JointWrapper(parts.joint, n_durations).eval(),
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
        kwargs = dict(flat_namespace=True, root_axis=self.root_axis, hparams=self.hparams())
        # The checkpoint's own SentencePiece vocab travels with the model, so the artifact is
        # detokenizable on its own -- the one capability the bespoke NeMo converters had that the first
        # MIL exports did not (P4.0.17 step 3).
        directory = self.tokenizer_dir()
        if directory is not None:
            kwargs["tokenizer_dir"] = directory
            kwargs["tokenizer_family"] = "sentencepiece_proto"
        return kwargs

    def driver_components(self) -> List:
        """The transducer decode loop, as components -- identical for every checkpoint in this family.

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
            ComputedCall, ExportConstants, LuaFragment, SubgraphCallComponent,
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
