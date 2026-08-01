"""The canonical task vocabulary (BACKLOG.md P4.0.4, `EXPORT-PREPARATION.md` §3/§6 stage A).

A *task* is the I/O contract a family exports against -- `optimum`'s own axis, and the reason
`ORTModelForCTC.forward` is ~40 lines and covers every CTC model: the task fixes the contract, not the
architecture. This module is the closed list of names that axis may take, and the base
`LoomExportConfig` subclass each one builds.

**Why a checked list rather than whatever string a family passes.** Before P4.0.4 the registered names
were `causal-lm`, `nemo-asr-encoder`, `tts-multi-phase` and `tts-flow-matching`: two named a
decomposition and one named a loader library. `TaskRegistry` accepted them because it accepted any
string, so nothing could tell a task from a typo, and the task's config class was defined by whichever
family happened to import first. Two consequences that a list fixes:

* `audio-codec` can be *reserved* -- declared here, with no family registered against it until family 11
  exists -- which is only a meaningful statement if the vocabulary is real and checked.
* The base config class is declared here, once, rather than being an accident of import order. Families
  legitimately register different subclasses under one task (Kokoro/VITS/StyleTTS2 register
  `BaseMultiPhaseModelExportConfig` while Matcha/Supertonic register its `TTSFlowMatchingModelExportConfig`
  subclass), so the check `TaskRegistry.register()` runs is `issubclass(entry.config_class, base)`.

**Base classes are resolved lazily, by import path.** Every family module imports `registry`, which
imports this module; naming the classes as strings and importing them on demand is what keeps that from
being a cycle. `base_config_class()` is only ever called from `register()`, by which point the family
module that owns the class is fully imported.
"""
import importlib
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TaskSpec:
    """One canonical task: its name, what export shape it covers, and the base config class every
    family registered under it must build."""

    name: str
    # What the task covers, in the terms a family author needs to decide whether theirs fits.
    summary: str
    # `module:QualName` of the base `LoomExportConfig` subclass, relative to this package. `None` on a
    # reserved task -- no family exists yet, so the only check possible is `LoomExportConfig` itself.
    base_config: Optional[str]
    # True while the name is declared but unclaimed: registering against it is not an error (the family
    # that claims it will), but the base class is not pinned down yet.
    reserved: bool = False

    def base_config_class(self) -> type:
        """The class `register()` checks `config_class` against. Imported on demand; see the module
        docstring for why this is not a module-level import."""
        if self.base_config is None:
            from .export_config import LoomExportConfig
            return LoomExportConfig
        module_name, _, qual_name = self.base_config.partition(":")
        module = importlib.import_module(f".{module_name}", package=__package__)
        return getattr(module, qual_name)


TASKS: Dict[str, TaskSpec] = {
    spec.name: spec
    for spec in (
        TaskSpec(
            name="text-generation",
            summary=(
                "Autoregressive text-in/text-out LMs. One traced graph (flattened) or a submodule chain "
                "(modular); the driver runs the token loop. Qwen3, LFM2."
            ),
            base_config="causal_lm_export:LMCausalModelExportConfig",
        ),
        TaskSpec(
            name="automatic-speech-recognition",
            summary=(
                "Audio-in/text-out. Today the encoder half only -- one traced encoder graph plus the "
                "decoding the driver or the host performs (CTC, TDT, RNNT). Conformer-CTC, Parakeet-TDT, "
                "Parakeet-RNNT."
            ),
            # NeMo-shaped today because all three registered models are NeMo checkpoints. P4.2 (GigaAM)
            # is the item that moves the loader onto the recognizer and widens this to a loader-agnostic
            # ASR encoder config; this line is where that change gets recorded.
            base_config="nemo_asr_export:ASRNemoEncoderExportConfig",
        ),
        TaskSpec(
            name="text-to-speech",
            summary=(
                "Text-in/waveform-out, N traced phases merged into one GGUF plus a driver that "
                "orchestrates them. Covers every sampler shape -- feed-forward, flow-matching/CFM, "
                "diffusion -- since P4.0.3 made that a decomposition/component choice rather than a "
                "task. Kokoro, VITS, Matcha, Supertonic, StyleTTS2."
            ),
            base_config="multi_phase_export:BaseMultiPhaseModelExportConfig",
        ),
        TaskSpec(
            name="audio-codec",
            summary=(
                "RESERVED, no family yet. Neural audio codec encode/decode (EnCodec, DAC, SNAC, Mimi) -- "
                "discrete-token-in/waveform-out, which is a different contract from text-to-speech and "
                "is what family 10's AR codec-token models decode through. Claimed by P5's family 11."
            ),
            base_config=None,
            reserved=True,
        ),
    )
}

# Deliberately temporary: the four task strings the families register today, mapped onto the canonical
# name each becomes. This exists only so that declaring the vocabulary (A.1) and renaming the strings
# (A.2) stay two commits -- `register()` accepts these spellings and the recognizers keep resolving
# exactly as before, but nothing rewrites them, so no behaviour changes here. **A.2 deletes this table**
# along with the old spellings; there are no backwards-compatible aliases in the end state, because the
# task name is a CLI argument, not a stored artifact, and two spellings for one thing is precisely what
# P4.0.3 spent a commit removing.
_PRE_P404_SPELLINGS: Dict[str, str] = {
    "causal-lm": "text-generation",
    "nemo-asr-encoder": "automatic-speech-recognition",
    "tts-multi-phase": "text-to-speech",
    "tts-flow-matching": "text-to-speech",
}


def known_tasks() -> List[str]:
    """The canonical names, for error messages and `--help`."""
    return sorted(TASKS)


def task_spec(name: str) -> TaskSpec:
    """The `TaskSpec` for `name`, or `ValueError` naming the whole vocabulary. Unknown-task errors are
    the main thing this module buys a caller, so they list what *is* known rather than only what is
    not."""
    spec = TASKS.get(name)
    if spec is not None:
        return spec
    canonical = _PRE_P404_SPELLINGS.get(name)
    if canonical is not None:
        return TASKS[canonical]
    raise ValueError(f"unknown task {name!r}; known tasks: {known_tasks()}")
