---
type: retro
date: 2026-08-06
domain: exporter
tags: [lstm, topology, redundant-compute, multi-output]
---

# Retro-002: Every LSTM Computed Its Gate Stack Twice

## The Issue

Found while scoping the Parakeet Lua migration, and it turned out not to be a Parakeet problem at all —
every LSTM in every model was doing twice the work it needed to.

## Root Cause Analysis

An LSTM cell step produces `h_new` and `c_new` from one gate stack. `GraphTopology` allowed a single
declared output, so the established precedent was to emit the **identical node list twice** and vary
only which output it declared — and every caller then ran both. The gate matmuls, the four gate VIEWs
and the six elementwise ops were all computed a second time purely to read the other result.

## Resolution & Lesson Learned

Fixed by the multi-output topology capability (see [P2 in
Epic-02](../epics/epic-02-mil-exporter-and-compiler.md)).

**Actionable takeaway:** a constraint that looks deliberate can be a cost nobody priced. The
one-output-per-topology convention was real and defensible, but nothing had measured what working
around it cost — and the workaround had become a copied precedent.

---

## Full record (verbatim from the ledger)


Found while scoping P4.0.17 step 2, and it turned out not to be a parakeet problem at all.

A cell step produces `h_new` and `c_new` from one gate stack. `GraphTopology` allowed a single declared
output when the pattern was established, so the precedent
(`convert_kokoro_duration_predictor.py::build_lstm_cell_topology`) was to emit the **identical node
list twice** and vary only which output it declared — and every caller then ran both. The gate matmuls,
the four gate VIEWs and the six elementwise ops were computed a second time purely to read the other
half of the same result. P2 added multi-output topologies; nothing went back to collect this.

It is not marginal, and it is not parakeet's: **Kokoro and StyleTTS2 each drive six BiLSTMs over a whole
sequence**, forward and backward, one cell call per timestep per direction. All of it was doubled.

`recurrent.py::_lstm_cell_topology` now declares `["h_new", "c_new"]`, and the output ORDER is the
contract every consumer reads by. Retrofitted across all of them:

* `RecurrentPhase` registers `<phase>_fwd`/`<phase>_bwd` instead of four `_h_*`/`_c_*` names.
* `loom_lua`'s `run_bi_lstm` captures both from one call per timestep per direction; its
  `DrivenTopologies` declaration follows, and `lua_library.drives_mismatches` checks the two agree.
* `loom.run_recurrent` takes ONE module name instead of `(h_module, c_module)`, and reads both outputs
  off one compute. It also gained an error for a topology that declares fewer than two, since the
  ordering is now load-bearing.
* The `export_lstm_test_fixture` GGUF and `test_e2e_lstm_recurrent`'s script.

**Not touched, deliberately:** the bespoke converters (`convert_parakeet_tdt.py`,
`convert_kokoro_duration_predictor.py`, `tools/fixture_gen/tdt_step_common.py`) and the hand-written
`kokoro_driver.lua`/`styletts2_driver.lua`, which carry their own copies of the four-topology
convention. They feed the legacy C++ path (`BiLstmStepper`, `TdtDecoder`) whose GGUFs must stay
byte-identical for their own tests, and they retire wholesale rather than being modernised.

**Gate.** `test_e2e_lstm_recurrent` still matches a real `torch.nn.LSTM(bidirectional=True)` to
**5e-8**. Kokoro and StyleTTS2 re-exported and re-run through their MIL Lua drivers: **22207/22207
checks each**, every waveform sample unchanged. Their topology counts fall exactly as predicted —
Kokoro 39 → 27, StyleTTS2 41 → 29, which is 6 BiLSTMs x 2 fewer apiece — and the driver still names
every one of them (`TestPeeledDriverCoverage`). ctest 146/146, exporter suite 466/466.

