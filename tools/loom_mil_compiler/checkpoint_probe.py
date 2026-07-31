"""Cheap, code-free structural probes a `ModelRecognizer.detect()` can run against a checkpoint
(BACKLOG.md P4.0.1).

Two shapes cover every family registered so far:

* **A sidecar JSON config** (`read_json`) -- Kokoro's `config.json`, and the same mechanism
  `causal_lm_export._hf_model_type` already uses for HF-style `model_type`.
* **A torch checkpoint's own pickle metadata** (`probe_torch_checkpoint`) -- every `torch.save` output
  since torch 1.6 is a zip archive whose `data.pkl` holds the object graph's structure (dict keys,
  class references) with the tensor payloads stored separately as `data/N` entries.

`probe_torch_checkpoint` walks that pickle with `pickletools.genops`, which reads opcodes and **never
unpickles** -- no `torch.load`, no `weights_only=` question, no checkpoint code executed, and no tensor
data read (only the `data.pkl` member is decompressed). That is a hard requirement for detection, which
by definition runs against paths that have not been identified yet: `detect()` is called with whatever
the user passed on the command line, against every registered recognizer in turn.

What the probe returns is deliberately raw -- the set of `module.Class` references and the set of
strings the pickle contains -- rather than a decoded object. Recognizers phrase their own structural
claim on top of it (a Lightning checkpoint declares `pytorch-lightning_version` and `state_dict`; a
fully-pickled `nn.Module` names its own real class), so the discriminating knowledge stays in the family
module that owns it, next to the `build_config` whose requirements it mirrors.
"""
import json
import pickletools
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

# Opcodes that push a string; `_STRING_OPS` feeds both the returned `strings` set and STACK_GLOBAL's
# two-operand lookahead (protocol 4 emits the module and class as two separate string pushes rather
# than one GLOBAL argument).
_STRING_OPS = frozenset({
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
    "SHORT_BINSTRING", "BINSTRING", "STRING",
})


@dataclass(frozen=True)
class TorchCheckpointProbe:
    """What `data.pkl` structurally refers to, without unpickling it."""

    # Every class the pickle references, as "module.Class" (e.g. "collections.OrderedDict",
    # "supertonic_tts.models.modules.text_to_latent_encoding.encoders.TTLTextEncoder").
    globals: Set[str]
    # Every string the pickle contains -- dict keys at every level, so both a top-level
    # "pytorch-lightning_version" and a nested state-dict key like "model_g.enc_p.emb.weight".
    strings: Set[str]


def read_json(path: Path) -> Optional[dict]:
    """`path` parsed as a JSON object, or None if it doesn't exist, isn't readable, isn't valid JSON,
    or isn't an object at the top level. Never raises -- a recognizer's job is to answer yes/no about
    an arbitrary path, not to diagnose it."""
    try:
        if not path.is_file():
            return None
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def probe_torch_checkpoint(path: Path) -> Optional[TorchCheckpointProbe]:
    """Structure of the torch checkpoint at `path`, or None if it isn't one (not a zip archive, or a
    zip with no `data.pkl` -- which is what a `.nemo` tar, a raw pickle file, or an unrelated file all
    look like here). Never raises, and never unpickles: a truncated or hostile pickle stops the walk
    and yields whatever was read up to that point."""
    try:
        if not path.is_file():
            return None
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.rsplit("/", 1)[-1] == "data.pkl"]
            if not members:
                return None
            raw = archive.read(members[0])
    except (OSError, zipfile.BadZipFile):
        return None

    found_globals: Set[str] = set()
    strings: Set[str] = set()
    recent: list = []  # last two string pushes, for STACK_GLOBAL
    try:
        for op, arg, _pos in pickletools.genops(raw):
            if op.name == "GLOBAL":
                # Protocol <= 3: one argument, "module class" separated by a newline that genops has
                # already turned into a space.
                found_globals.add(str(arg).replace(" ", ".", 1))
            elif op.name == "STACK_GLOBAL":
                if len(recent) == 2:
                    found_globals.add(f"{recent[0]}.{recent[1]}")
            elif op.name in _STRING_OPS and isinstance(arg, str):
                strings.add(arg)
                recent.append(arg)
                if len(recent) > 2:
                    del recent[0]
    except Exception:
        # A malformed pickle is a "no" for every recognizer that needs a specific key, not a crash --
        # and partial results are still sound, since every probe consumer asks whether something IS
        # present, never whether something is absent from a complete read.
        pass
    return TorchCheckpointProbe(globals=found_globals, strings=strings)
