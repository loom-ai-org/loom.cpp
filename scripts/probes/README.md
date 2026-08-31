# Measurement-only ggml patches

**These are not in `cmake/patches/` and must not be.** Everything there is applied on every configure
and ships in every wheel; everything here is applied by hand, for one measurement, and taken out again.
They exist because a negative result nobody can re-run is a rumour, and because both of the items that
produced them turned on being able to A/B **inside one process** — one binary, one allocation, one page
cache, one branch flipped — which building two trees cannot do.

To use one:

```sh
cp scripts/probes/<name>.patch cmake/patches/ggml-9999-probe.patch
cmake -S . -B build && cmake --build build -j"$(nproc)"
# ... measure ...
rm cmake/patches/ggml-9999-probe.patch && cmake -S . -B build   # resets the checkout and re-patches
```

The `ggml-9999-` prefix matters: patches are applied in sorted order and a probe has to land after the
`ggml-00xx` series it modifies. Two probes can be applied at once (`9998` and `9999`) — P4.26's final
Raspberry Pi runs used both.

| patch | what it adds | used by |
|---|---|---|
| `ggml-p426-sgemm-policy-probe.patch` | `GGML_SGEMM_POLICY` selects `matmul_aligned`'s row-block predicate at run time (`0` pre-`ggml-0012`, `1` `0012` as first shipped, `6` as shipped now, plus candidates), and `GGML_SGEMM_CENSUS=1` prints an `(m, n, k, nth)` histogram of every `sgemm` call at exit. `loom_sgemm_set_policy()` is exported so a bench can flip arms mid-process. | `scripts/bench19.cpp`, Epic-05 P4.26 |
| `ggml-p427-graph-plan-probe.patch` | `LOOM_PLAN_PROBE=1` prints every graph ggml clamps to fewer threads than were asked for (`2` prints every plan), and `LOOM_UNARY_SERIAL=1` puts every row of an `apply_unary_op` on thread 0 — which prices threading an op by **removing** it. | `scripts/bench20.cpp`, Epic-05 P4.27 |

Each patch is a diff against the pinned ggml **with `cmake/patches/ggml-00xx` already applied**, so it
fails loudly rather than silently no-opping if one of those changes. When that happens, regenerate it
rather than forcing it: `git apply` in the ggml checkout, edit, `diff -u` against a copy of the file
taken before the edit.
