# Backlog — moved

**This file is no longer the ledger.** It reached ~9,000 lines, which made it unmanageable as a working
document and expensive to load for any task that only needed one corner of it.

The ledger is now a four-tier hub-and-spoke knowledge base under [`docs/`](docs/):

| tier | where | what it holds |
|---|---|---|
| **Hub** | [`docs/backlog/active-index.md`](docs/backlog/active-index.md) | **open work only**, one line each, linked to its context |
| **Epics** | [`docs/epics/`](docs/epics/) | what each domain is and how it works |
| **ADRs** | [`docs/adrs/`](docs/adrs/) | why a technical choice was made, and what it cost |
| **Retros** | [`docs/retros/`](docs/retros/) | what broke, the root cause, and the takeaway |

Closed work is **not tracked**. When an item is done, its decision goes to an ADR, its lesson to a
retro, its architecture into the relevant epic — and the item leaves the index. Per-commit execution
detail for work that closed before 2026-08-10 is in [`docs/archive/`](docs/archive/), unmaintained.

The process for keeping it that way is in [`CLAUDE.md`](CLAUDE.md).

---

## Where everything went

Chronological, in the order it appeared in the old file, so a reference to an old section resolves.
Code comments cite items by their `P`-number, and those numbers are unchanged.

| old section | now in |
|---|---|
| Roadmap: Qwen3-ASR / Qwen3-TTS | [Backlog → Models](docs/backlog/active-index.md#models) |
| F5-TTS | [Backlog → Models](docs/backlog/active-index.md#models) |
| Task #79 — permissively-licensed phonemizer | [ADR-012](docs/adrs/adr-012-permissive-phonemizer.md) · [Epic-07](docs/epics/epic-07-text-frontends-and-tokenizers.md) |
| Supertonic's fixed text length | [Retro-005](docs/retros/retro-005-supertonic-fixed-text-length.md) |
| Grapheme text front-ends | [Epic-07](docs/epics/epic-07-text-frontends-and-tokenizers.md) |
| Causal LMs wrong when `n_tokens == n_head` | [Retro-001](docs/retros/retro-001-layout-healing-heuristics.md) |
| Who depends on the layout-"healing" heuristics? | [Backlog → Engine correctness](docs/backlog/active-index.md#engine--correctness) |
| Exporter orientation pointers (BACKEND / EXPORT-ROADMAP / EXPORT-PREPARATION) | [Epic-02](docs/epics/epic-02-mil-exporter-and-compiler.md) |
| Implementation sequence for the roadmap | [Epic-02 §5](docs/epics/epic-02-mil-exporter-and-compiler.md) |
| P0 — clear the ground | [Archive: exporter](docs/archive/ledger-2026-07-to-08-exporter.md) · [ADR-004](docs/adrs/adr-004-mil-as-the-single-export-path.md) |
| P1 — exporter internals | [Archive: exporter](docs/archive/ledger-2026-07-to-08-exporter.md) |
| P2 — multi-output topologies | [Archive: exporter](docs/archive/ledger-2026-07-to-08-exporter.md) · [Retro-002](docs/retros/retro-002-lstm-gate-stack-computed-twice.md) |
| P3 — the API skeleton | [ADR-005](docs/adrs/adr-005-export-config-and-task-registry.md) |
| LFM2 migrated onto the causal-LM registry | [ADR-005](docs/adrs/adr-005-export-config-and-task-registry.md) |
| What P3 deliberately did not build | [Backlog → Exporter](docs/backlog/active-index.md#exporter--mil-compiler) |
| P4.0 — the sixteen items | [Archive: exporter](docs/archive/ledger-2026-07-to-08-exporter.md) · [Epic-02 §5](docs/epics/epic-02-mil-exporter-and-compiler.md) |
| P4.0.17 — the NeMo ASR driver builder | [Archive: exporter](docs/archive/ledger-2026-07-to-08-exporter.md) · [ADR-003](docs/adrs/adr-003-per-model-complexity-in-the-exporter.md) |
| TTS driver constants moved to the export side | [ADR-006](docs/adrs/adr-006-model-constants-belong-to-the-export.md) |
| The stranded pre-MIL components | [ADR-003](docs/adrs/adr-003-per-model-complexity-in-the-exporter.md) |
| The bespoke NeMo converters are gone | [ADR-004](docs/adrs/adr-004-mil-as-the-single-export-path.md) |
| Parakeet decodes in Lua | [ADR-003](docs/adrs/adr-003-per-model-complexity-in-the-exporter.md) |
| Parakeet's four traced phases | [Archive: exporter](docs/archive/ledger-2026-07-to-08-exporter.md) |
| `RecurrentPhase` handles a stacked LSTM | [Archive: exporter](docs/archive/ledger-2026-07-to-08-exporter.md) |
| Every LSTM computed its gate stack twice | [Retro-002](docs/retros/retro-002-lstm-gate-stack-computed-twice.md) |
| The TDT decoder recomputed its prediction network | [Retro-003](docs/retros/retro-003-tdt-decoder-recomputed-its-prediction-network.md) |
| Conformer-CTC gains a Lua entry point | [Archive: exporter](docs/archive/ledger-2026-07-to-08-exporter.md) |
| Driver logits marshalling caps prefill | [Retro-004](docs/retros/retro-004-luajit-array-limit-caps-prefill.md) |
| SentencePiece-style byte-fallback BPE | [Epic-07](docs/epics/epic-07-text-frontends-and-tokenizers.md) |
| The two export sweeps | [Retro-015](docs/retros/retro-015-export-snapshot-sweeps.md) |
| `decomposition` vs what `profile` actually does | [Retro-016](docs/retros/retro-016-the-profile-field-was-not-inert.md) |
| P4 — flagship coverage (P4.1–P4.6) | [Archive: model coverage](docs/archive/ledger-2026-08-model-coverage.md) · [Epic-03](docs/epics/epic-03-model-coverage.md) |
| P4.10 — macOS wheels | [Epic-08 §4](docs/epics/epic-08-packaging-and-release.md) |
| P4.11 — Metal | [Epic-04 §5](docs/epics/epic-04-backends-and-accelerators.md) |
| P4.12 — Kokoro shipped speaking noise | [Retro-006](docs/retros/retro-006-kokoro-shipped-noise.md) |
| P5 — breadth | [Epic-03 §3](docs/epics/epic-03-model-coverage.md) · [Backlog → Models](docs/backlog/active-index.md#models) |
| P4.7 — the engine runs on a GPU | [Epic-04 §4](docs/epics/epic-04-backends-and-accelerators.md) |
| P4.7a–P4.7d — the host-callback thread | [Retro-009](docs/retros/retro-009-host-callback-count-was-the-wrong-lens.md) |
| P4.7e — a primitive that asks the backend | [ADR-007](docs/adrs/adr-007-backend-capability-negotiation.md) |
| P4.7f — `atan` without a host callback | [ADR-008](docs/adrs/adr-008-atan-approximation.md) |
| P4.8 / P4.8a — more backends without ending leanness | [ADR-009](docs/adrs/adr-009-backends-as-dynamic-libraries.md) |
| P4.8b / P4.8e — device selection | [ADR-010](docs/adrs/adr-010-device-selection-by-kind.md) |
| P4.8c / P4.8d — CUDA and the NPU probes | [Epic-04 §4](docs/epics/epic-04-backends-and-accelerators.md) |
| P4.8f — the gate was green for the wrong reason | [Retro-008](docs/retros/retro-008-a-gate-that-was-green-for-the-wrong-reason.md) |
| P4.8g / P4.8h / P4.8i — the CUDA wheel | [Epic-08 §5](docs/epics/epic-08-packaging-and-release.md) |
| P4.8j — `"gpu"` chose the iGPU | [Retro-007](docs/retros/retro-007-gpu-chose-the-integrated-gpu.md) |
| P4.9 — the ggml pin bump | [Epic-01 §5](docs/epics/epic-01-inference-engine-core.md) |
| P4.5 — one repo becomes three | [ADR-011](docs/adrs/adr-011-three-repositories.md) |
| P6 — cleanup | [Backlog → Exporter](docs/backlog/active-index.md#exporter--mil-compiler) |
| Third family template: NeMo ASR encoders | [Epic-02 §2](docs/epics/epic-02-mil-exporter-and-compiler.md) |
| Open follow-ups from the exporter-improvement thread | [Backlog → Exporter](docs/backlog/active-index.md#exporter--mil-compiler) |
| MIL primitive review | [Backlog → Engine correctness](docs/backlog/active-index.md#engine--correctness) |
| Known gap: `matmul` `transpose_x` | [Backlog → Exporter](docs/backlog/active-index.md#exporter--mil-compiler) |
| Known bug: LFM2 zero-RoPE NaN | [Backlog → Engine correctness](docs/backlog/active-index.md#engine--correctness) |
| Retrofit the bespoke `tools/convert_*` scripts | [Retro-013](docs/retros/retro-013-retrofitting-eight-bespoke-converters.md) |
| Modular-export blueprint | [Backlog → Exporter](docs/backlog/active-index.md#exporter--mil-compiler) |
| P4.14 — `$LOOM_PROFILE` | [Epic-05 §4](docs/epics/epic-05-edge-performance.md) |
| P4.14 — the onnxruntime baseline correction | [Retro-010](docs/retros/retro-010-an-unpinned-competitor-baseline.md) |
| P4.15 / P4.15b / P4.15e — the shipped kernels | [Epic-05 §4](docs/epics/epic-05-edge-performance.md) · [ADR-014](docs/adrs/adr-014-patch-ggml-rather-than-write-kernels.md) |
| P4.15 / P4.15b — the hunt, the traps, the 92% | [Retro-011](docs/retros/retro-011-chasing-the-gemm-and-convolution-gap.md) |
| P4.15c, chain tiling, im2col, "what not to re-propose" | [Retro-012](docs/retros/retro-012-optimizations-that-were-measured-out.md) |
| P4.15d / P4.15f — the text encoder, twice then once | [Retro-014](docs/retros/retro-014-the-text-encoder-was-in-the-graph-twice.md) |
| P4.16 — the convolution gap, re-measured and closed | [Epic-05 §5](docs/epics/epic-05-edge-performance.md) · [Retro-012](docs/retros/retro-012-optimizations-that-were-measured-out.md) |
| P4.25 — threading the unary gate, measured out | [Epic-05 §5](docs/epics/epic-05-edge-performance.md) · [Retro-012](docs/retros/retro-012-optimizations-that-were-measured-out.md) |
| P4.13 — 2-D conv kernels for Q4_0 | [Epic-05 §5](docs/epics/epic-05-edge-performance.md) · [ADR-017](docs/adrs/adr-017-no-k-quants.md) |
| P5.0 — the high-level API | [ADR-013](docs/adrs/adr-013-one-door-per-task.md) · [Epic-06](docs/epics/epic-06-high-level-api-and-hosts.md) |
| Performance optimizations designed but not implemented | [Backlog → Engine performance](docs/backlog/active-index.md#engine--performance) |
| Scope limitations | [Epic-01 §4](docs/epics/epic-01-inference-engine-core.md) |
| Minor cleanups | [Backlog → Minor cleanups](docs/backlog/active-index.md#minor-cleanups) |

The full pre-refactor text is in git history: `git log --follow -- BACKLOG.md`.
