"""Regression check for BACKLOG.md P3.1/P3.2's causal-LM family (`causal_lm_export.py`).

Every model in this family (Qwen3 monolithic, LFM2 monolithic, LFM2 modular) is now reachable ONLY
through the `text-generation` task's registry entries (`export_qwen3_mil.py`/`export_lfm2_modular.py`/
`export_lfm2_monolithic.py` are all deleted -- BACKLOG.md's P3.2 and the later LFM2 migration). With no
independent "old script" left to diff against, what's still worth guarding per model is
`registry.py`'s own `_build_*` factory silently drifting from constructing the same `LoomExportConfig`
class directly with the same parameters (e.g. someone edits one call site and not the other) -- so each
test below builds a model both ways and snapshot-diffs the two resulting GGUFs.

All three tests need the real checkpoints and take real trace time; skipped (not failed) when the
checkpoint directory isn't present, matching this project's existing real-model-test skip convention.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import loom_mil_compiler  # noqa: E402  registers the "loom" backend + torch-frontend patches
from loom_mil_compiler.decomposition import Flattened, Modular  # noqa: E402
from loom_mil_compiler.causal_lm_export import (  # noqa: E402
    LMCausalModelExportConfig,
)
from loom_mil_compiler.modular_export import ModularExportSpec  # noqa: E402
from loom_mil_compiler.registry import default_registry  # noqa: E402
from loom_mil_compiler.snapshot_gguf import snapshot  # noqa: E402

LFM2_DIR = Path("/home/flavio/Dev/models/lfm2-350m")
QWEN3_DIR = Path("/home/flavio/Dev/models/qwen3-0.6b-base")


def _snapshot_dir(gguf_path: Path, out_dir: Path) -> Path:
    snapshot(gguf_path, out_dir)
    return out_dir / gguf_path.stem


def _assert_snapshots_match(a_dir: Path, b_dir: Path):
    a_files = {p.name: p.read_bytes() for p in a_dir.iterdir()}
    b_files = {p.name: p.read_bytes() for p in b_dir.iterdir()}
    assert set(a_files) == set(b_files), (set(a_files), set(b_files))
    mismatched = [name for name in a_files if a_files[name] != b_files[name]]
    assert not mismatched, f"snapshot files differ: {mismatched}"


def _assert_registry_matches_direct(tmp_path, task, model, model_path, build_direct_config):
    """`build_direct_config` takes the output path and returns a freshly constructed `LoomExportConfig`
    -- kept lazy (a callable, not a pre-built instance) so each side of the comparison gets its own
    fresh, correctly-pathed config rather than mutating one instance's `output_path` after the fact."""
    registry = default_registry()
    via_registry_gguf = tmp_path / f"{model}_via_registry.gguf"
    registry.get(task, model).build_config(model_path, str(via_registry_gguf)).export()

    direct_gguf = tmp_path / f"{model}_direct.gguf"
    build_direct_config(str(direct_gguf)).export()

    snap_dir = tmp_path / "snap"
    via_registry_snap = _snapshot_dir(via_registry_gguf, snap_dir)
    direct_snap = _snapshot_dir(direct_gguf, snap_dir)
    _assert_snapshots_match(via_registry_snap, direct_snap)


@pytest.mark.skipif(not QWEN3_DIR.exists(), reason="Qwen3 checkpoint not available locally")
def test_qwen3_registry_entry_matches_direct_construction(tmp_path):
    _assert_registry_matches_direct(
        tmp_path, "text-generation", "qwen3", QWEN3_DIR,
        lambda output_path: LMCausalModelExportConfig(
            architecture="qwen3", output_path=output_path, decomposition=Flattened(),
            model_dir=str(QWEN3_DIR),
        ),
    )


@pytest.mark.skipif(not LFM2_DIR.exists(), reason="LFM2 checkpoint not available locally")
def test_lfm2_monolithic_registry_entry_matches_direct_construction(tmp_path):
    _assert_registry_matches_direct(
        tmp_path, "text-generation", "lfm2-monolithic", LFM2_DIR,
        lambda output_path: LMCausalModelExportConfig(
            architecture="lfm2", output_path=output_path, decomposition=Flattened(),
            model_dir=str(LFM2_DIR), tokenizer_pre="llama3",
        ),
    )


@pytest.mark.skipif(not LFM2_DIR.exists(), reason="LFM2 checkpoint not available locally")
def test_lfm2_modular_registry_entry_matches_direct_construction(tmp_path):
    _assert_registry_matches_direct(
        tmp_path, "text-generation", "lfm2-modular", LFM2_DIR,
        lambda output_path: LMCausalModelExportConfig(
            architecture="lfm2", output_path=output_path, model_dir=str(LFM2_DIR),
            decomposition=Modular(spec=ModularExportSpec(
                prefix_attr="model.embed_tokens",
                repeated_attr="model.layers",
                suffix_attrs=["model.embedding_norm", "lm_head"],
                aux_attr="model.pos_emb",
                aux_kwarg="position_embeddings",
            )),
            tokenizer_dir=str(LFM2_DIR), tokenizer_pre="llama3",
        ),
    )
