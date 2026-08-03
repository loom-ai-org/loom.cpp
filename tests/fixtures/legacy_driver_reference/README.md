# Frozen reference waveforms of the retired per-model C++ drivers

These five `.npy` files are the numeric ground truth that `loom::VitsDriver`, `loom::MatchaDriver`,
`loom::SupertonicDriver`, `loom::KokoroDriver` and `loom::StyleTTS2Driver` used to supply *live*, by
being constructed and run inside the test that compared against them. P4.0.8 (stage E of
`EXPORT-PREPARATION.md`) retires those drivers, and BACKEND.md's R6 rule allows deleting one only in the
commit that re-points the last test consuming it — so this directory is what the tests were re-pointed
*onto*.

**Read this before treating a mismatch as a bug in the Lua driver.** These files are not a
reference implementation. They are one recorded output of a program that no longer exists.

## Provenance

Each file was produced by that model's own oracle test — the same binary, at the same inputs, in the
commit immediately before its driver was deleted — via the `LOOM_DUMP_REF_NPY` hook those tests carried
for exactly this purpose:

```
LOOM_VITS_DIR=<vits_stats/logw/flow_vocoder ggufs> \
  LOOM_DUMP_REF_NPY=.../vits_driver_waveform.npy         build/tests/test_e2e_vits_driver
LOOM_MATCHA_DIR=<four bespoke matcha ggufs> \
  LOOM_DUMP_REF_NPY=.../matcha_driver_waveform.npy       build/tests/test_e2e_matcha_driver
LOOM_KOKORO_ALL_DIR=<convert_kokoro_all.py output> \
  LOOM_DUMP_REF_NPY=.../kokoro_driver_waveform.npy       build/tests/test_e2e_kokoro_driver
LOOM_STYLETTS2_DIR=<convert_styletts2_* output> \
  LOOM_DUMP_REF_NPY=.../styletts2_driver_waveform.npy    build/tests/test_e2e_styletts2_driver
LOOM_SUPERTONIC_ALL_DIR=<convert_supertonic_all.py output> \
  LOOM_SUPERTONIC_VOICE_STYLE_JSON=<.../voice_styles/F1.json> \
  LOOM_DUMP_REF_NPY=.../supertonic_driver_waveform_F1.npy build/tests/test_e2e_supertonic_driver
```

| fixture | samples | driver | inputs |
|---|---|---|---|
| `vits_driver_waveform.npy` | 49664 | `VitsDriver` | the 62-token BOS/blank/EOS sequence in `test_e2e_vits_lua_driver.cpp`, seed 42 |
| `matcha_driver_waveform.npy` | 10240 | `MatchaDriver` | tokens `{5,42,7,88,13,100,3,61}`, n_steps 10, seed 42 |
| `kokoro_driver_waveform.npy` | 22200 | `KokoroDriver` | ids `{0,50,62,24,83,16,44,71,9,0}`, the synthetic `ref_s`, speed 1.0, seed 42 |
| `styletts2_driver_waveform.npy` | 22200 | `StyleTTS2Driver` | ids `{0,50,62,24,83,16,44,71,9}`, 5 diffusion steps, seed 42 |
| `supertonic_driver_waveform_F1.npy` | 70656 | `SupertonicDriver` | ids `{12,45,67,23,89,34,56,78,90,15}`, voice style **F1**, n_steps 10, seed 42 |

All five are 1-D float32, little-endian — `numpy.load` opens them directly.

**Supertonic is keyed by voice style** because the style vectors are an *input*, not a constant: a
different `voice_styles/*.json` produces a different waveform. `test_e2e_supertonic_{,mil_}lua_driver`
derive the fixture name from the JSON's basename and **skip** if no fixture exists for that style,
rather than compare against the wrong reference. Only `F1` is covered, and no more can be added — see
below.

## These cannot be regenerated

The generator was the driver. Deleting `src/core/*_driver.cpp` deleted the only code that can produce
these numbers, and that was the point of the stage. What this costs, stated plainly so it is not
rediscovered:

* a new voice style, a new token sequence or a different seed **cannot** get a fixture; a test needing
  one has to fall back on the per-phase reference tests, which is where the real numerical confidence
  lives anyway (`test_e2e_*_mil_{text_encoder,decoder,vocoder,albert,diffusion}_reference.cpp`, each
  against a real-PyTorch fixture from its `reference_forward_*.py`);
* if one of these files is lost, the check it backs is gone, not merely stale.

That trade was made knowingly: five hand-written C++ drivers, ~7k lines, whose whole job the exported
Lua driver now does, against five 40–280 KB files.

## The tolerance each fixture is compared at, and why

The bounds are the ones the live comparisons already used; re-pointing changed no tolerance. The
numbers below are what the re-pointed tests actually observe, and each matched the live-oracle run to
every digit printed:

| test | bound | observed |
|---|---|---|
| `test_e2e_vits_lua_driver` | `1e-3` | `3.27e-07` |
| `test_e2e_matcha_lua_driver` | `1e-3` | `1.39e-05` |
| `test_e2e_kokoro_lua_driver` | `1e-3` | `1.91e-06` |
| `test_e2e_styletts2_lua_driver` | `5e-3` | `3.83e-03` |
| `test_e2e_supertonic_lua_driver` | `1e-3` | `7.45e-07` |
| `test_e2e_matcha_mil_lua_driver` | `2e-2` | `1.04e-02` (rmse `6.78e-04`) |
| `test_e2e_supertonic_mil_lua_driver` | `1e-3` | `2.08e-06` |

The first five are the *same* op sequence run through the Lua interpreter instead of C++ control flow,
so they should and do match to near bit-exactness. The two MIL rows compare two independently derived
computation graphs of the same architecture, which is why Matcha's bound is looser — see that test's own
comment for the amplification reasoning.
