"""`ModelPatcher` (BACKLOG.md P3.3 / EXPORT-ROADMAP.md R4): the named hook a family module uses for the
import-order workarounds it needs before its real third-party package can be imported at all --
generalizes `nemo_asr_export.py`'s own `prepare_nemo_environment()` into a documented contract every
TTS family module (`kokoro_export.py`, `matcha_export.py`, ...) follows the same way.

Class-level monkeypatches (e.g. replacing a real module's own `.forward` with a trace-friendly version)
are NOT part of this hook: they need the real class already imported, so they stay module-level code
immediately after that family's own imports -- same place they've always lived, just now documented as
belonging to this same "environment preparation" concern rather than being unexplained top-of-file
side effects. `prepare_environment()` covers only the part that must run strictly BEFORE any import of
the family's own package (version-check stubs, numpy/coremltools compatibility patches).
"""


class ModelPatcher:
    """Base for a family's "make the real package importable and traceable" preparation. Subclasses
    override `prepare_environment()`; it's a no-op by default for families that need no such stub
    (calling it unconditionally, even when empty, is what lets `BaseMultiPhaseModelExportConfig.export()`
    treat every family uniformly)."""

    @staticmethod
    def prepare_environment() -> None:
        pass
