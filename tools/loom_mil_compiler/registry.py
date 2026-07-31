"""`TaskRegistry` (BACKLOG.md P3.2): maps a *task* -- an export shape / `LoomExportConfig` family, e.g.
`"causal-lm"`, `"nemo-asr-encoder"` (mirrors `optimum`'s own task vocabulary, `"text-generation"`/
`"automatic-speech-recognition"`) -- to the set of concrete models that fit it, each declared as a
`ModelRecognizer`: a real structural check against a checkpoint (`detect`), and how to build that
model's own `LoomExportConfig` instance (`build_config`).

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
    """One task -- the export shape shared by every recognizer registered under it."""

    task: str
    config_class: type
    recognizers: List[ModelRecognizer] = field(default_factory=list)


class TaskRegistry:
    def __init__(self):
        self._entries: Dict[str, TaskRegistryEntry] = {}

    def register(self, entry: TaskRegistryEntry) -> None:
        """Registers `entry`'s recognizers under its task -- creates the task if new, or extends an
        existing one's recognizer list if not. Multiple family modules legitimately share one task
        (e.g. `"tts-multi-phase"`: Kokoro, StyleTTS2, and VITS are all `BaseMultiPhaseModelExportConfig`
        instances, each contributing its own recognizer), so a second `register()` call for an
        already-known task is expected, not an error -- unless it disagrees about which config class
        the task builds, which would mean two families are colliding on one task name by mistake."""
        existing = self._entries.get(entry.task)
        if existing is None:
            self._entries[entry.task] = entry
            return
        if existing.config_class is not entry.config_class:
            raise ValueError(
                f"task {entry.task!r} already registered with config_class {existing.config_class!r}, "
                f"got {entry.config_class!r} -- two families disagree on what this task builds"
            )
        existing.recognizers.extend(entry.recognizers)

    def _candidates(self, task: Optional[str]):
        if task is not None:
            if task not in self._entries:
                raise ValueError(f"unknown task {task!r}; registered tasks: {sorted(self._entries)}")
            entries = [self._entries[task]]
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
        if task not in self._entries:
            raise ValueError(f"unknown task {task!r}; registered tasks: {sorted(self._entries)}")
        entry = self._entries[task]
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
