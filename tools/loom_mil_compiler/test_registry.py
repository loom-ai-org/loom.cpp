"""Unit tests for `registry.py`'s `TaskRegistry` and every family's recognizers (BACKLOG.md P3.2, P4.0.1)
-- pure structural detection, no real checkpoints, no torch/coremltools tracing. Synthetic fixtures (a
fake HF dir with a `config.json`, fake `.nemo` archives with a synthetic `model_config.yaml`, fake torch
zip archives holding a real pickle of the shape each TTS checkpoint has) reproduce every shape `detect()`
has to tell apart, the same "fake module, real check" philosophy `test_nemo_asr_export.py` already uses
for `EncoderOutput.validate`.

The TTS fixtures are deliberately built from `pickle.dumps` of plain dicts rather than `torch.save` of
real modules: what `checkpoint_probe` reads is the pickle's own opcode stream, so a plain pickle in a zip
named `data.pkl` is the same thing to it as a 300MB checkpoint, and the fixture stays readable as a
statement of what each family's checkpoint structurally IS.
"""
import json
import pickle
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loom_mil_compiler.registry import ModelRecognizer, TaskRegistry, TaskRegistryEntry  # noqa: E402
from loom_mil_compiler.export_config import LoomExportConfig  # noqa: E402
from loom_mil_compiler.tasks import known_tasks, task_spec  # noqa: E402
from loom_mil_compiler.checkpoint_probe import probe_torch_checkpoint, read_json  # noqa: E402
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


def _make_torch_zip(path: Path, payload) -> Path:
    """A minimal stand-in for a `torch.save` output: a zip archive whose only member is `data.pkl`.
    `payload` is either an object to pickle or raw pickle bytes (for shapes a plain `pickle.dumps`
    can't produce, e.g. a GLOBAL naming a class whose package isn't installed here)."""
    raw = payload if isinstance(payload, bytes) else pickle.dumps(payload, protocol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", raw)
    return path


# Each family's real checkpoint structure, reduced to the keys `detect()` actually claims -- see the
# corresponding `_is_*` docstring for what was probed off the real file to arrive at each one.
def _lightning(state_dict: dict, version: str = "2.0.8") -> dict:
    return {"epoch": 9, "global_step": 1000, "pytorch-lightning_version": version, "state_dict": state_dict}


MATCHA_CKPT = _lightning({"mel_mean": 0.0, "mel_std": 1.0, "encoder.emb.weight": 0.0})
VITS_CKPT = _lightning({"model_g.enc_p.emb.weight": 0.0, "model_d.convs.0.weight": 0.0}, version="1.9.5")
# Kokoro and StyleTTS2 both: component name -> state dict, no version marker, no config. StyleTTS2 adds
# the `net` wrapper and the training-time components Kokoro's inference release strips.
KOKORO_CKPT = {"bert": {"module.embeddings.word_embeddings.weight": 0.0}, "bert_encoder": {},
               "predictor": {}, "decoder": {}, "text_encoder": {}}
STYLETTS2_CKPT = {"net": {**KOKORO_CKPT, "diffusion": {}, "mpd": {}, "msd": {}, "wd": {},
                          "predictor_encoder": {}, "style_encoder": {}}}
# A fully-pickled `nn.Module`: the pickle names the class it would reconstruct. Written as raw opcodes
# (protocol 2 GLOBAL) since the real `supertonic_tts` package isn't importable in a unit test.
SUPERTONIC_PT_BYTES = (
    b"\x80\x02csupertonic_tts.models.modules.text_to_latent_encoding.encoders\nTTLTextEncoder\n."
)


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


# -- checkpoint_probe, the shared primitive under every TTS recognizer (BACKLOG.md P4.0.1) -------------

def test_probe_reads_globals_and_strings_without_unpickling(tmp_path):
    path = _make_torch_zip(tmp_path / "m.pt", SUPERTONIC_PT_BYTES)
    probe = probe_torch_checkpoint(path)
    assert probe is not None
    assert any(ref.startswith("supertonic_tts.") for ref in probe.globals)
    assert probe.globals == {
        "supertonic_tts.models.modules.text_to_latent_encoding.encoders.TTLTextEncoder"
    }


def test_probe_handles_protocol_4_stack_global(tmp_path):
    """Protocol <= 3 emits one GLOBAL with both names; protocol 4 pushes module and class separately and
    joins them with STACK_GLOBAL. Both must produce the same `module.Class` string, since which protocol
    a checkpoint was written with is not something a recognizer should have to know."""
    import collections

    path = _make_torch_zip(tmp_path / "p4.pt", pickle.dumps(collections.OrderedDict, protocol=4))
    probe = probe_torch_checkpoint(path)
    assert probe is not None and "collections.OrderedDict" in probe.globals


def test_probe_collects_nested_dict_keys_as_strings(tmp_path):
    path = _make_torch_zip(tmp_path / "l.ckpt", MATCHA_CKPT)
    probe = probe_torch_checkpoint(path)
    # A top-level key and a key one level down inside `state_dict` are both visible -- the recognizers
    # rely on exactly that (`pytorch-lightning_version` is top level, `mel_mean` is not).
    assert {"pytorch-lightning_version", "state_dict", "mel_mean"}.issubset(probe.strings)


@pytest.mark.parametrize("make", [
    lambda p: p / "does_not_exist.pt",
    lambda p: (p / "a_dir").mkdir() or (p / "a_dir"),
    lambda p: _write(p / "plain.txt", b"not a zip at all"),
    lambda p: _empty_zip(p / "no_data_pkl.zip"),
])
def test_probe_returns_none_for_anything_that_is_not_a_torch_checkpoint(tmp_path, make):
    """`detect()` runs against whatever the user typed, so the probe answers None rather than raising
    for a missing path, a directory, a non-zip file, and a zip with no `data.pkl` (what a `.nemo` tar
    and Matcha's own non-zip `generator_v1` HiFi-GAN checkpoint both look like here)."""
    assert probe_torch_checkpoint(make(tmp_path)) is None


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def _empty_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/version", "3")
    return path


def test_probe_survives_a_truncated_pickle(tmp_path):
    """A partial read is still sound: every consumer asks whether something IS present."""
    path = _make_torch_zip(tmp_path / "cut.pt", SUPERTONIC_PT_BYTES[:-10])
    probe = probe_torch_checkpoint(path)
    assert probe is not None


def test_read_json_returns_none_rather_than_raising(tmp_path):
    assert read_json(tmp_path / "missing.json") is None
    assert read_json(_write(tmp_path / "bad.json", b"{not json")) is None
    assert read_json(_write(tmp_path / "list.json", b"[1, 2]")) is None
    assert read_json(_write(tmp_path / "ok.json", b'{"a": 1}')) == {"a": 1}


# -- TTS family recognizers (BACKLOG.md P4.0.1) --------------------------------------------------------

def _make_kokoro_dir(tmp_path: Path) -> Path:
    d = tmp_path / "kokoro_model"
    d.mkdir()
    _make_torch_zip(d / "kokoro-v1_0.pth", KOKORO_CKPT)
    (d / "config.json").write_text(json.dumps({
        "istftnet": {"upsample_rates": [10, 6]}, "plbert": {"num_hidden_layers": 12},
        "n_token": 178, "style_dim": 128, "vocab": {"a": 1}, "n_mels": 80,
    }))
    return d


def _make_matcha_dir(tmp_path: Path) -> Path:
    d = tmp_path / "matcha_ckpt"
    d.mkdir()
    _make_torch_zip(d / "matcha_ljspeech.ckpt", MATCHA_CKPT)
    (d / "generator_v1").write_bytes(b"\x80\x02}q\x00.")  # real HiFi-GAN file: a raw pickle, not a zip
    return d


def _make_supertonic_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pt"
    d.mkdir()
    for name in ("duration_predictor.pt", "text_encoder.pt", "vector_estimator.pt", "vocoder.pt"):
        _make_torch_zip(d / name, SUPERTONIC_PT_BYTES)
    return d


def _tts_detectors():
    from loom_mil_compiler.kokoro_export import _is_kokoro
    from loom_mil_compiler.matcha_export import _is_matcha
    from loom_mil_compiler.styletts2_export import _is_styletts2
    from loom_mil_compiler.supertonic_export import _is_supertonic
    from loom_mil_compiler.vits_export import _is_vits

    return {"kokoro": _is_kokoro, "matcha": _is_matcha, "styletts2": _is_styletts2,
            "supertonic": _is_supertonic, "vits": _is_vits}


def _all_tts_fixtures(tmp_path: Path) -> dict:
    return {
        "kokoro": _make_kokoro_dir(tmp_path),
        "matcha": _make_matcha_dir(tmp_path),
        "supertonic": _make_supertonic_dir(tmp_path),
        "vits": _make_torch_zip(tmp_path / "piper.ckpt", VITS_CKPT),
        "styletts2": _make_torch_zip(tmp_path / "epoch_2nd_00100.pth", STYLETTS2_CKPT),
    }


def test_every_tts_recognizer_matches_its_own_fixture_and_no_other(tmp_path):
    """The whole point of P4.0.1: five families, five checkpoints, one match each. Two near-collisions
    are the reason this is a full cross-product rather than five independent asserts -- Matcha/VITS are
    both Lightning `.ckpt`s, and Kokoro/StyleTTS2 are both component-dict `.pth`s from the same model
    lineage."""
    detectors = _tts_detectors()
    fixtures = _all_tts_fixtures(tmp_path)
    for family, detect in detectors.items():
        matched = {name for name, path in fixtures.items() if detect(path)}
        assert matched == {family}, f"_is_{family} matched {matched or 'nothing'}"


def test_matcha_and_vits_are_told_apart_by_state_dict_keys_not_by_the_lightning_marker(tmp_path):
    """Recorded because P4.0.1 originally proposed the Lightning signature itself as Matcha's
    discriminator; probing the real checkpoints showed piper-VITS declares the identical marker."""
    detectors = _tts_detectors()
    both = {"pytorch-lightning_version", "state_dict"}
    for name, payload in (("matcha", MATCHA_CKPT), ("vits", VITS_CKPT)):
        path = _make_torch_zip(tmp_path / f"{name}.ckpt", payload)
        assert both.issubset(probe_torch_checkpoint(path).strings)
    # Same marker, opposite answers.
    vits_ckpt = _make_torch_zip(tmp_path / "v.ckpt", VITS_CKPT)
    assert detectors["vits"](vits_ckpt) and not detectors["matcha"](vits_ckpt)


def test_kokoro_needs_its_checkpoint_beside_its_config(tmp_path):
    """The config alone is not enough: StyleTTS2 loads this same `config.json` for the shared iSTFTNet
    decoder, so a directory holding only it is not a Kokoro export target."""
    detect = _tts_detectors()["kokoro"]
    d = _make_kokoro_dir(tmp_path)
    assert detect(d)

    (d / "kokoro-v1_0.pth").unlink()
    assert not detect(d)


def test_matcha_needs_both_checkpoints(tmp_path):
    detect = _tts_detectors()["matcha"]
    d = _make_matcha_dir(tmp_path)
    assert detect(d)

    (d / "generator_v1").unlink()
    assert not detect(d), "the HiFi-GAN vocoder half is required by phases(), so it is required here"


def test_supertonic_needs_all_four_checkpoints(tmp_path):
    detect = _tts_detectors()["supertonic"]
    d = _make_supertonic_dir(tmp_path)
    assert detect(d)

    (d / "vector_estimator.pt").unlink()
    assert not detect(d)


def test_supertonic_rejects_a_directory_of_plain_state_dicts(tmp_path):
    """Filenames alone would accept this; the `supertonic_tts.`-rooted class reference is what makes the
    check structural."""
    detect = _tts_detectors()["supertonic"]
    d = tmp_path / "impostor"
    d.mkdir()
    for name in ("duration_predictor.pt", "text_encoder.pt", "vector_estimator.pt", "vocoder.pt"):
        _make_torch_zip(d / name, {"weight": 0.0})
    assert not detect(d)


def test_tts_recognizers_reject_the_other_families_paths(tmp_path):
    """No TTS recognizer may claim an HF causal-LM directory or a `.nemo` archive -- `TaskRegistry.detect`
    runs every recognizer against every path, so a false positive here breaks an unrelated family."""
    others = [_make_hf_dir(tmp_path, "qwen3"), _make_nemo_archive(tmp_path, "ctc", CTC_CONFIG)]
    for family, detect in _tts_detectors().items():
        for path in others:
            assert not detect(path), f"_is_{family} claimed {path.name}"


# -- TaskRegistry itself, independent of any real family -----------------------------------------------

# A toy family, registered under the one canonical task no real family claims yet (`audio-codec`, whose
# base is therefore just `LoomExportConfig`). This keeps these tests independent of any real family while
# still going through P4.0.4's vocabulary and config-class checks, which is what a real caller does.
TOY_TASK = "audio-codec"


@dataclass(kw_only=True)
class _ToyConfig(LoomExportConfig):
    pass


def _toy_recognizer(name: str) -> ModelRecognizer:
    return ModelRecognizer(name=name, detect=lambda p: p.name == name, build_config=lambda p, o: (name, o))


def _toy_registry() -> TaskRegistry:
    registry = TaskRegistry()
    registry.register(TaskRegistryEntry(
        task=TOY_TASK,
        config_class=_ToyConfig,
        recognizers=[_toy_recognizer("a"), _toy_recognizer("b")],
    ))
    return registry


def test_registering_an_unknown_task_raises_naming_the_vocabulary():
    """P4.0.4: task names are a closed, checked list. Before it, any string registered -- so a typo
    silently created a task nothing would ever detect against."""
    registry = TaskRegistry()
    with pytest.raises(ValueError, match="unknown task 'txt-generation'") as excinfo:
        registry.register(TaskRegistryEntry(task="txt-generation", config_class=_ToyConfig))
    for name in known_tasks():
        assert name in str(excinfo.value), f"the error must name the whole vocabulary, missing {name}"


def test_registering_a_subclass_of_the_tasks_base_config_is_allowed():
    """The check is `issubclass`, not identity -- `TTSFlowMatchingModelExportConfig` really is a subclass
    of the `text-to-speech` base, so Matcha/Supertonic share one task with Kokoro/VITS/StyleTTS2."""
    class _ToySubConfig(_ToyConfig):
        pass

    registry = _toy_registry()
    registry.register(TaskRegistryEntry(
        task=TOY_TASK, config_class=_ToySubConfig, recognizers=[_toy_recognizer("c")],
    ))
    assert registry.get(TOY_TASK, "c").name == "c"


def test_registering_an_unrelated_config_class_raises():
    """A family registered under a task whose export shape it does not build is a real bug, and the one
    the identity check used to catch."""
    registry = _toy_registry()
    with pytest.raises(ValueError, match="does not build"):
        registry.register(TaskRegistryEntry(task=TOY_TASK, config_class=str))


def test_the_real_tts_flow_matching_config_registers_under_the_tts_task():
    """The concrete case the relaxed check exists for, against the real classes rather than toys: today
    these are two tasks that A.2 merges, and both halves must pass the same base-class check."""
    from loom_mil_compiler.multi_phase_export import (
        BaseMultiPhaseModelExportConfig,
        TTSFlowMatchingModelExportConfig,
    )

    registry = TaskRegistry()
    for config_class in (BaseMultiPhaseModelExportConfig, TTSFlowMatchingModelExportConfig):
        registry.register(TaskRegistryEntry(task="text-to-speech", config_class=config_class))
    assert issubclass(TTSFlowMatchingModelExportConfig, task_spec("text-to-speech").base_config_class())


def test_registering_the_same_task_twice_extends_recognizers():
    registry = _toy_registry()
    registry.register(TaskRegistryEntry(
        task=TOY_TASK, config_class=_ToyConfig, recognizers=[_toy_recognizer("c")],
    ))
    assert {r.name for r in registry._entries[TOY_TASK].recognizers} == {"a", "b", "c"}
    assert registry.get(TOY_TASK, "c").name == "c"


def test_detect_finds_the_one_real_match(tmp_path):
    registry = _toy_registry()
    rec = registry.detect(tmp_path / "a")
    assert rec.name == "a"


def test_detect_raises_naming_candidates_on_no_match(tmp_path):
    registry = _toy_registry()
    with pytest.raises(ValueError, match=rf"{TOY_TASK}/a.*{TOY_TASK}/b|no registered recognizer matched"):
        registry.detect(tmp_path / "neither")


def test_detect_raises_on_ambiguous_match(tmp_path):
    registry = _toy_registry()
    registry._entries[TOY_TASK].recognizers.append(
        ModelRecognizer(name="a2", detect=lambda p: p.name == "a", build_config=lambda p, o: ("a2", o))
    )
    with pytest.raises(ValueError, match="matched more than one recognizer"):
        registry.detect(tmp_path / "a")


def test_detect_can_be_restricted_to_one_task(tmp_path):
    registry = _toy_registry()
    with pytest.raises(ValueError, match="unknown task"):
        registry.detect(tmp_path / "a", task="not-a-task")


# -- the task vocabulary itself (P4.0.4) ---------------------------------------------------------------

def test_the_vocabulary_is_the_four_canonical_names():
    assert known_tasks() == [
        "audio-codec", "automatic-speech-recognition", "text-generation", "text-to-speech",
    ]


def test_every_declared_base_config_resolves_and_is_a_loom_export_config():
    """The base classes are named as import-path strings to avoid an import cycle, so nothing else would
    catch a typo or a rename in one of them."""
    for name in known_tasks():
        base = task_spec(name).base_config_class()
        assert isinstance(base, type) and issubclass(base, LoomExportConfig), name


def test_audio_codec_is_reserved_and_unclaimed():
    """Decision 3: the name is declared now so family 11 does not invent a competing one, but no family
    registers against it until it exists."""
    from loom_mil_compiler.registry import default_registry

    assert task_spec("audio-codec").reserved
    assert task_spec("audio-codec").base_config_class() is LoomExportConfig
    assert "audio-codec" not in default_registry()._entries


def test_a_declared_but_unclaimed_task_says_so_rather_than_unknown():
    """`--task audio-codec` is a valid argparse choice but has no family, and that is a different error
    from a typo -- conflating them sends the caller looking for a misspelling that isn't there."""
    from loom_mil_compiler.registry import default_registry

    registry = default_registry()
    with pytest.raises(ValueError, match="declared but no family is registered against it yet"):
        registry.get("audio-codec", "whatever")
    with pytest.raises(ValueError, match="declared but no family is registered against it yet"):
        registry.detect(Path("/nonexistent"), task="audio-codec")


def test_only_audio_codec_is_reserved():
    assert [n for n in known_tasks() if task_spec(n).reserved] == ["audio-codec"]


def test_get_returns_the_named_recognizer():
    registry = _toy_registry()
    rec = registry.get(TOY_TASK, "b")
    assert rec.name == "b"


def test_get_raises_on_unknown_model():
    registry = _toy_registry()
    with pytest.raises(ValueError, match="unknown model"):
        registry.get(TOY_TASK, "nonexistent")


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
    causal_lm = registry.get("text-generation", "qwen3")
    assert causal_lm.name == "qwen3"
    for model in ("conformer-ctc", "parakeet-tdt", "parakeet-rnnt"):
        rec = registry.get("automatic-speech-recognition", model)
        assert rec.name == model
    # Sanity: config_class stored per task matches the family's own base class.
    assert registry._entries["text-generation"].config_class is LMCausalModelExportConfig
    assert registry._entries["automatic-speech-recognition"].config_class is ASRNemoEncoderExportConfig


def test_every_registered_task_is_canonical():
    """The whole point of P4.0.4's vocabulary: after A.2 there is no spelling in the registry that
    `tasks.py` does not declare, and no family left on a name that describes a decomposition or a
    loader library."""
    from loom_mil_compiler.registry import default_registry

    registry = default_registry()
    assert set(registry._entries) <= set(known_tasks())
    assert sorted(registry._entries) == [
        "automatic-speech-recognition", "text-generation", "text-to-speech",
    ]


def test_the_five_tts_families_now_share_one_task():
    """`tts-multi-phase` + `tts-flow-matching` were one task whose members differ by a field, ever since
    P4.0.3 made decomposition a field. Flow-matching models register their `TTSFlowMatchingModelExportConfig`
    subclass under the same `text-to-speech` task as the plain multi-phase ones."""
    from loom_mil_compiler.registry import default_registry

    registry = default_registry()
    names = {rec.name for rec in registry._entries["text-to-speech"].recognizers}
    assert names == {"kokoro", "styletts2", "vits", "matcha", "supertonic"}
    for model in names:
        assert registry.get("text-to-speech", model).name == model


def test_the_pre_p404_task_spellings_are_gone():
    """No backwards-compatible aliases (A.2): the old names must fail like any other typo, naming the
    vocabulary that replaced them."""
    from loom_mil_compiler.registry import default_registry

    registry = default_registry()
    for old in ("causal-lm", "nemo-asr-encoder", "tts-multi-phase", "tts-flow-matching"):
        with pytest.raises(ValueError, match="unknown task"):
            registry.get(old, "kokoro")
        with pytest.raises(ValueError, match="unknown task"):
            task_spec(old)


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


def test_default_registry_detects_every_tts_family_end_to_end(tmp_path):
    """P4.0.1's acceptance shape: `loom-export <path> -o out.gguf` with no `--task`/`--model` resolves
    all five TTS families through the same registry as the causal-LM and NeMo ones -- against the whole
    registry, not just each family's own recognizer, so an ambiguity introduced by any other family
    fails here."""
    from loom_mil_compiler.registry import default_registry

    registry = default_registry()
    for family, path in _all_tts_fixtures(tmp_path).items():
        assert registry.detect(path).name == family
