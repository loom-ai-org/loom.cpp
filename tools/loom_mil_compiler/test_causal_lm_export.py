"""Regression check for BACKLOG.md P3.1's causal-LM family (`causal_lm_export.py`).

`LMMonolithicCausalModelExportConfig` is exercised for real every time `export_hf_causal_lm.py`'s own
`export_causal_lm()` runs (it's a thin shim over the class now) -- `export_qwen3_mil.py` and
`export_lfm2_monolithic.py` both already go through it, so re-running either and snapshot-diffing
against a pre-P3.1 baseline is real coverage on its own (see BACKLOG.md's P3.1 gate).

`LMModularCausalModelExportConfig`, on the other hand, has no real caller yet: `export_lfm2_modular.py`
still calls `modular_export.export_modular` directly and is NOT migrated this pass (confirmed scope --
LFM2 stays regression-checked, not migrated; real migration is a later pass). So THIS test is the only
place that exercises it against a real checkpoint: it runs `export_lfm2_modular.py`'s own `main()`
unmodified, builds an equivalent `LMModularCausalModelExportConfig` by hand with the exact same
parameters, and snapshot-diffs the two resulting GGUFs -- proof the new class genuinely reproduces the
shape `export_lfm2_modular.py` hand-rolls, not just that it looks plausible.

Both tests need the real checkpoints and take real trace time; skipped (not failed) when the checkpoint
directory isn't present, matching this project's existing real-model-test skip convention.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import loom_mil_compiler  # noqa: E402  registers the "loom" backend + torch-frontend patches
from loom_mil_compiler.causal_lm_export import (  # noqa: E402
    LMModularCausalModelExportConfig,
    LMMonolithicCausalModelExportConfig,
)
from loom_mil_compiler.modular_export import ModularExportSpec  # noqa: E402
from loom_mil_compiler.snapshot_gguf import snapshot  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
LFM2_DIR = Path("/home/flavio/Dev/models/lfm2-350m")
QWEN3_DIR = Path("/home/flavio/Dev/models/qwen3-0.6b-base")


def _run_script_main(name: str):
    """Executes a repo-root export_*.py's own main() in-process -- same dynamic-load pattern
    test_nemo_asr_export.py already uses to read a script's declared SPEC, but calling main() here
    instead, since we need the real GGUF it produces, not just its metadata."""
    spec_file = REPO_ROOT / f"{name}.py"
    module_spec = importlib.util.spec_from_file_location(name, spec_file)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.main()


def _snapshot_dir(gguf_path: Path, out_dir: Path) -> Path:
    snapshot(gguf_path, out_dir)
    return out_dir / gguf_path.stem


def _assert_snapshots_match(old_dir: Path, new_dir: Path):
    old_files = {p.name: p.read_bytes() for p in old_dir.iterdir()}
    new_files = {p.name: p.read_bytes() for p in new_dir.iterdir()}
    assert set(old_files) == set(new_files), (set(old_files), set(new_files))
    mismatched = [name for name in old_files if old_files[name] != new_files[name]]
    assert not mismatched, f"snapshot files differ: {mismatched}"


@pytest.mark.skipif(not LFM2_DIR.exists(), reason="LFM2 checkpoint not available locally")
def test_modular_causal_model_export_config_matches_export_lfm2_modular(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_script_main("export_lfm2_modular")
    old_gguf = tmp_path / "lfm2_350m_modular.gguf"
    assert old_gguf.exists()

    # Exact same ModularExportSpec export_lfm2_modular.py's own main() constructs.
    modular_spec = ModularExportSpec(
        prefix_attr="model.embed_tokens",
        repeated_attr="model.layers",
        suffix_attrs=["model.embedding_norm", "lm_head"],
        aux_attr="model.pos_emb",
        aux_kwarg="position_embeddings",
    )
    new_gguf = tmp_path / "lfm2_350m_modular_new.gguf"
    LMModularCausalModelExportConfig(
        architecture="lfm2",
        output_path=str(new_gguf),
        model_dir=str(LFM2_DIR),
        modular_spec=modular_spec,
        tokenizer_dir=str(LFM2_DIR),
        tokenizer_pre="llama3",
    ).export()

    snap_dir = tmp_path / "snap"
    old_snap = _snapshot_dir(old_gguf, snap_dir)
    new_snap = _snapshot_dir(new_gguf, snap_dir)
    _assert_snapshots_match(old_snap, new_snap)


@pytest.mark.skipif(not QWEN3_DIR.exists(), reason="Qwen3 checkpoint not available locally")
def test_monolithic_causal_model_export_config_matches_export_qwen3_mil(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_script_main("export_qwen3_mil")
    old_gguf = tmp_path / "qwen3_0.6b_mil_monolithic.gguf"
    assert old_gguf.exists()

    new_gguf = tmp_path / "qwen3_0.6b_mil_monolithic_new.gguf"
    LMMonolithicCausalModelExportConfig(
        architecture="qwen3",
        output_path=str(new_gguf),
        profile="monolithic",
        model_dir=str(QWEN3_DIR),
    ).export()

    snap_dir = tmp_path / "snap"
    old_snap = _snapshot_dir(old_gguf, snap_dir)
    new_snap = _snapshot_dir(new_gguf, snap_dir)
    _assert_snapshots_match(old_snap, new_snap)
