"""Unit tests for `registry.py`'s `TaskRegistry` and the causal-LM/NeMo-ASR family recognizers
(BACKLOG.md P3.2) -- pure structural detection, no real checkpoints, no torch/coremltools tracing.
Synthetic fixtures (a fake HF dir with a `config.json`, fake `.nemo` archives with a synthetic
`model_config.yaml`) reproduce every shape `detect()` has to tell apart, the same "fake module,
real check" philosophy `test_nemo_asr_export.py` already uses for `EncoderOutput.validate`.
"""
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loom_mil_compiler.registry import ModelRecognizer, TaskRegistry, TaskRegistryEntry  # noqa: E402
from loom_mil_compiler.causal_lm_export import _is_qwen3  # noqa: E402
from loom_mil_compiler.nemo_asr_export import (  # noqa: E402
    _is_conformer_ctc,
    _is_parakeet_rnnt,
    _is_parakeet_tdt,
)


def _make_hf_dir(tmp_path: Path, model_type: str) -> Path:
    d = tmp_path / model_type
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    return d


def _make_nemo_archive(tmp_path: Path, name: str, config: dict) -> Path:
    yaml_path = tmp_path / f"{name}_config.yaml"
    import yaml
    yaml_path.write_text(yaml.safe_dump(config))
    nemo_path = tmp_path / f"{name}.nemo"
    with tarfile.open(nemo_path, "w") as t:
        t.add(yaml_path, arcname="./model_config.yaml")
    return nemo_path


CTC_CONFIG = {"target": "nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE"}
TDT_CONFIG = {
    "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
    "model_defaults": {"enc_hidden": 1024, "tdt_durations": [0, 1, 2, 3, 4], "num_tdt_durations": 5},
}
RNNT_CONFIG = {
    "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
    "model_defaults": {"enc_hidden": 1024},
}


# -- causal-LM recognizer -----------------------------------------------------------------------------

def test_is_qwen3_matches_a_real_shaped_config(tmp_path):
    assert _is_qwen3(_make_hf_dir(tmp_path, "qwen3"))


def test_is_qwen3_rejects_a_different_model_type(tmp_path):
    assert not _is_qwen3(_make_hf_dir(tmp_path, "lfm2"))


def test_is_qwen3_rejects_a_directory_with_no_config(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert not _is_qwen3(d)


def test_is_qwen3_rejects_a_plain_file(tmp_path):
    f = tmp_path / "not_a_dir"
    f.write_text("x")
    assert not _is_qwen3(f)


# -- NeMo ASR encoder recognizers ---------------------------------------------------------------------

def test_is_conformer_ctc_matches_ctc_target(tmp_path):
    path = _make_nemo_archive(tmp_path, "ctc", CTC_CONFIG)
    assert _is_conformer_ctc(path)
    assert not _is_parakeet_tdt(path)
    assert not _is_parakeet_rnnt(path)


def test_tdt_and_rnnt_share_a_target_but_are_told_apart_by_model_defaults(tmp_path):
    """The headline case this recognizer pair exists for: both restore through the identical
    EncDecRNNTBPEModel class, so `target` alone can't disambiguate them (confirmed against the real
    checkpoints) -- `model_defaults.tdt_durations` is the real discriminator."""
    tdt_path = _make_nemo_archive(tmp_path, "tdt", TDT_CONFIG)
    rnnt_path = _make_nemo_archive(tmp_path, "rnnt", RNNT_CONFIG)

    assert _is_parakeet_tdt(tdt_path)
    assert not _is_parakeet_rnnt(tdt_path)
    assert not _is_conformer_ctc(tdt_path)

    assert _is_parakeet_rnnt(rnnt_path)
    assert not _is_parakeet_tdt(rnnt_path)
    assert not _is_conformer_ctc(rnnt_path)


def test_nemo_recognizers_reject_non_nemo_paths(tmp_path):
    d = tmp_path / "some_dir"
    d.mkdir()
    assert not _is_conformer_ctc(d)
    assert not _is_parakeet_tdt(d)
    assert not _is_parakeet_rnnt(d)


# -- TaskRegistry itself, independent of any real family -----------------------------------------------

def _toy_registry() -> TaskRegistry:
    registry = TaskRegistry()
    registry.register(TaskRegistryEntry(
        task="toy-task",
        config_class=object,
        recognizers=[
            ModelRecognizer(name="a", detect=lambda p: p.name == "a", build_config=lambda p, o: ("a", o)),
            ModelRecognizer(name="b", detect=lambda p: p.name == "b", build_config=lambda p, o: ("b", o)),
        ],
    ))
    return registry


def test_registering_the_same_task_twice_raises():
    registry = _toy_registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(TaskRegistryEntry(task="toy-task", config_class=object, recognizers=[]))


def test_detect_finds_the_one_real_match(tmp_path):
    registry = _toy_registry()
    rec = registry.detect(tmp_path / "a")
    assert rec.name == "a"


def test_detect_raises_naming_candidates_on_no_match(tmp_path):
    registry = _toy_registry()
    with pytest.raises(ValueError, match=r"toy-task/a.*toy-task/b|no registered recognizer matched"):
        registry.detect(tmp_path / "neither")


def test_detect_raises_on_ambiguous_match(tmp_path):
    registry = _toy_registry()
    registry._entries["toy-task"].recognizers.append(
        ModelRecognizer(name="a2", detect=lambda p: p.name == "a", build_config=lambda p, o: ("a2", o))
    )
    with pytest.raises(ValueError, match="matched more than one recognizer"):
        registry.detect(tmp_path / "a")


def test_detect_can_be_restricted_to_one_task(tmp_path):
    registry = _toy_registry()
    with pytest.raises(ValueError, match="unknown task"):
        registry.detect(tmp_path / "a", task="not-a-task")


def test_get_returns_the_named_recognizer():
    registry = _toy_registry()
    rec = registry.get("toy-task", "b")
    assert rec.name == "b"


def test_get_raises_on_unknown_model():
    registry = _toy_registry()
    with pytest.raises(ValueError, match="unknown model"):
        registry.get("toy-task", "nonexistent")


def test_get_raises_on_unknown_task():
    registry = _toy_registry()
    with pytest.raises(ValueError, match="unknown task"):
        registry.get("nonexistent-task", "a")


# -- the real default registry, structurally -------------------------------------------------------------

def test_default_registry_registers_the_two_p32_tasks():
    from loom_mil_compiler.registry import default_registry
    from loom_mil_compiler.causal_lm_export import LMCausalModelExportConfig
    from loom_mil_compiler.nemo_asr_export import ASRNemoEncoderExportConfig

    registry = default_registry()
    causal_lm = registry.get("causal-lm", "qwen3")
    assert causal_lm.name == "qwen3"
    for model in ("conformer-ctc", "parakeet-tdt", "parakeet-rnnt"):
        rec = registry.get("nemo-asr-encoder", model)
        assert rec.name == model
    # Sanity: config_class stored per task matches the family's own base class.
    assert registry._entries["causal-lm"].config_class is LMCausalModelExportConfig
    assert registry._entries["nemo-asr-encoder"].config_class is ASRNemoEncoderExportConfig


def test_default_registry_detects_a_synthetic_qwen3_dir(tmp_path):
    from loom_mil_compiler.registry import default_registry

    registry = default_registry()
    rec = registry.detect(_make_hf_dir(tmp_path, "qwen3"))
    assert rec.name == "qwen3"


def test_default_registry_detects_synthetic_nemo_archives(tmp_path):
    from loom_mil_compiler.registry import default_registry

    registry = default_registry()
    assert registry.detect(_make_nemo_archive(tmp_path, "ctc", CTC_CONFIG)).name == "conformer-ctc"
    assert registry.detect(_make_nemo_archive(tmp_path, "tdt", TDT_CONFIG)).name == "parakeet-tdt"
    assert registry.detect(_make_nemo_archive(tmp_path, "rnnt", RNNT_CONFIG)).name == "parakeet-rnnt"
