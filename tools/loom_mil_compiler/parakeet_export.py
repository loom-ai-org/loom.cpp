"""Parakeet TDT/RNN-T -- the NeMo leaf of the Transducer family (BACKLOG.md P4.0.17 step 2).

**What this replaces and why it is a different decomposition.** The ASR family's other export
(`nemo_asr_export.ASRNemoEncoderExportConfig`) is `Flattened`: it traces the encoder and nothing else,
because a Transducer's prediction network is an `nn.LSTM` and ggml has no LSTM op. Everything after the
encoder therefore lived in hand-derived topologies written by `tools/convert_nemo/convert_parakeet_tdt.py`
into five separate GGUFs, driven by `src/core/tdt_decoder.cpp`. That left the MIL export of these two
checkpoints *unreachable*: nothing but its own test could run it, since the artifact held only half a
model and carried a driver (the causal-LM one) that raises when called.

**Everything that was general moved out** (P4.2). The four phases, their shapes, the blank-id
derivation, the joint/duration cross-check and the whole component list are
`transducer_export.BaseTransducerExportConfig`; what stays here is what is actually Parakeet's --
`ASRModel.restore_from`, where NeMo keeps the three post-encoder modules, and the two `.nemo`
recognizers. GigaAM v3 is the second leaf and shares the rest verbatim, which is the check on whether
that split was drawn in the right place.
"""
from dataclasses import dataclass

from .nemo_asr_export import extract_nemo_tokenizer_dir, load_model, prepare_nemo_environment
from .transducer_export import BaseTransducerExportConfig, TransducerParts


@dataclass
class ASRParakeetExportConfig(BaseTransducerExportConfig):
    """Parakeet TDT or plain RNN-T: a `.nemo` archive restored through NeMo's own loader."""

    architecture: str = "parakeet_tdt"
    output_path: str = "parakeet.gguf"

    def prepare_environment(self) -> None:
        prepare_nemo_environment()

    def load_model(self):
        return load_model(self)

    def transducer_parts(self, model) -> TransducerParts:
        """NeMo's layout: the prediction network under `decoder.prediction`, the joint beside it.

        `model_defaults.tdt_durations` is the checkpoint's own authority on its duration set -- present
        for TDT, absent entirely for plain RNN-T. Read off the RESTORED model rather than by re-opening
        the archive in `build_config`, so a config can be built (and its components counted by
        `component_registry.usage`) without a checkpoint on disk.
        """
        defaults = getattr(model.cfg, "model_defaults", None) or {}
        pred = model.decoder.prediction
        return TransducerParts(
            embed=pred.embed,
            lstm=pred.dec_rnn.lstm,
            joint=model.joint,
            durations=tuple(defaults.get("tdt_durations") or ()),
            # NeMo states the joint's total width a second time. It is NOT the token count -- see
            # `BaseTransducerExportConfig.phases`, where the two readings are compared.
            declared_joint_width=int(model.joint.num_classes_with_blank),
        )

    def tokenizer_dir(self):
        """A `.nemo` archive keeps its SentencePiece proto under a content-hashed name inside the
        tarball, so this one has to unpack it; `extract_nemo_tokenizer_dir` is that adapter."""
        return extract_nemo_tokenizer_dir(self.checkpoint)
