"""`TaskRegistry` (BACKLOG.md P3.2/P4.0.4): maps a *task* -- the I/O contract a family exports against,
e.g. `"text-generation"`, `"automatic-speech-recognition"` -- to the set of concrete models that fit it,
each declared as a `ModelRecognizer`: a real structural check against a checkpoint (`detect`), and how to
build that model's own `LoomExportConfig` instance (`build_config`).

The task names are not free-form: `tasks.py` holds the canonical vocabulary and each task's base config
class, and `register()` validates against it (P4.0.4).

**Registry key is the task, not the model.** An earlier draft of this registry keyed per model
(`"qwen3"`, `"kokoro"`, ...), which conflates two axes `optimum` deliberately keeps separate: `task`
names the export shape shared by every model that fits it; *which* model you're pointing at is resolved
from the checkpoint itself wherever it's self-describing, exactly like `optimum`'s own `TasksManager`
resolves `model_type` separately from `task`. See BACKLOG.md's P3.2 entry for the full reasoning.

Each family module (`causal_lm_export.py`, `nemo_asr_export.py`, ...) owns its own recognizers and
registers them via a module-level `register(registry)` function; `default_registry()` below is just the
aggregator every `main_export()` call uses.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import tasks
from .export_config import LoomExportConfig


@dataclass
class ModelRecognizer:
    """One concrete model within a task: a real structural check against a checkpoint/model directory
    (`detect`), and how to build that model's own `LoomExportConfig` instance from a path + output path
    (`build_config`)."""

    name: str
    detect: Callable[[Path], bool]
    build_config: Callable[[Path, str], LoomExportConfig]


@dataclass
class TaskRegistryEntry:
    """One family's contribution to a task -- the export shape shared by every recognizer registered
    under it.

    `task` must be a name from `tasks.py`'s canonical vocabulary. `config_class` is *this family's*
    config class, which `register()` checks is a subclass of the base that vocabulary declares for the
    task; the task's authoritative base class lives in `tasks.py`, not here, so it does not depend on
    which family registered first."""

    task: str
    config_class: type
    recognizers: List[ModelRecognizer] = field(default_factory=list)


class TaskRegistry:
    def __init__(self):
        self._entries: Dict[str, TaskRegistryEntry] = {}

    def register(self, entry: TaskRegistryEntry) -> None:
        """Registers `entry`'s recognizers under its task -- creates the task if new, or extends an
        existing one's recognizer list if not. Multiple family modules legitimately share one task
        (e.g. Kokoro, StyleTTS2 and VITS are all `BaseMultiPhaseModelExportConfig` instances under
        `"text-to-speech"`, each contributing its own recognizer), so a second `register()` call for an
        already-known task is expected, not an error.

        Two things are checked, both against `tasks.py` (BACKLOG.md P4.0.4):

        1. **The task name is in the canonical vocabulary.** Before P4.0.4 any string registered, so a
           typo silently created a task nothing would ever detect against.
        2. **`config_class` is a subclass of the base class the vocabulary declares for that task** --
           not identical to whatever the first family to import happened to pass.
           `TTSFlowMatchingModelExportConfig` is a *subclass* of `BaseMultiPhaseModelExportConfig`, so
           identity would reject Matcha and Supertonic sharing one TTS task with Kokoro/VITS/StyleTTS2
           -- and which of the two the task got pinned to used to depend on import order."""
        spec = tasks.task_spec(entry.task)
        base = spec.base_config_class()
        if not (isinstance(entry.config_class, type) and issubclass(entry.config_class, base)):
            raise ValueError(
                f"task {entry.task!r} builds {base.__name__} instances, got config_class "
                f"{entry.config_class!r} -- a family registered under a task whose export shape it "
                f"does not build"
            )
        existing = self._entries.get(entry.task)
        if existing is None:
            self._entries[entry.task] = entry
            return
        existing.recognizers.extend(entry.recognizers)

    def _entry(self, task: str) -> TaskRegistryEntry:
        """The registered entry for `task`, or a `ValueError` that distinguishes the two ways a task can
        be absent: a name outside the vocabulary (a typo, or a pre-P4.0.4 spelling), versus a canonical
        name that is *declared but unclaimed* -- `audio-codec` until family 11 exists. Both are errors;
        conflating them sends the caller looking for a typo that isn't there."""
        entry = self._entries.get(task)
        if entry is not None:
            return entry
        spec = tasks.task_spec(task)  # raises naming the vocabulary if the name isn't canonical at all
        raise ValueError(
            f"task {spec.name!r} is declared but no family is registered against it yet; "
            f"registered tasks: {sorted(self._entries)}"
        )

    def _candidates(self, task: Optional[str]):
        if task is not None:
            entries = [self._entry(task)]
        else:
            entries = list(self._entries.values())
        return [(entry, rec) for entry in entries for rec in entry.recognizers]

    def detect(self, model_path: Path, task: Optional[str] = None) -> ModelRecognizer:
        """Tries every recognizer (all tasks, or only `task`'s if given) against `model_path` and
        returns the one real structural match. Raises naming every candidate tried on no match or more
        than one match -- an ambiguous or unrecognized checkpoint fails loudly with the exact candidates
        considered, not a guess."""
        candidates = self._candidates(task)
        matches = [(entry, rec) for entry, rec in candidates if rec.detect(model_path)]
        if not matches:
            tried = [f"{entry.task}/{rec.name}" for entry, rec in candidates]
            raise ValueError(
                f"no registered recognizer matched {str(model_path)!r} (tried: {tried}); "
                f"pass --task/--model explicitly"
            )
        if len(matches) > 1:
            names = [f"{entry.task}/{rec.name}" for entry, rec in matches]
            raise ValueError(
                f"{str(model_path)!r} matched more than one recognizer: {names}; "
                f"pass --task/--model to disambiguate"
            )
        _entry, rec = matches[0]
        return rec

    def get(self, task: str, model: str) -> ModelRecognizer:
        """Explicit override, naming both axes."""
        entry = self._entry(task)
        for rec in entry.recognizers:
            if rec.name == model:
                return rec
        raise ValueError(
            f"unknown model {model!r} for task {task!r}; registered: {[r.name for r in entry.recognizers]}"
        )


def default_registry() -> TaskRegistry:
    """The registry every `loom-export`/`main_export()` call uses -- one task per family, populated by
    each family module's own `register(registry)`."""
    registry = TaskRegistry()
    from . import causal_lm_export
    from . import nemo_asr_export
    from . import kokoro_export
    from . import matcha_export
    from . import styletts2_export
    from . import supertonic_export
    from . import vits_export

    causal_lm_export.register(registry)
    nemo_asr_export.register(registry)
    kokoro_export.register(registry)
    matcha_export.register(registry)
    styletts2_export.register(registry)
    supertonic_export.register(registry)
    vits_export.register(registry)
    return registry
