# Debug tools

Diagnostics, not tests. **Nothing here is wired into CMake** — each file is compiled by hand when a
question comes up, which is the same arrangement `compare_logits.cpp` has always had. That is
deliberate: none of them is a gate, and adding four more targets to a build that already produces 129
test binaries would cost every developer time to serve an occasional question.

| | asks |
|---|---|
| `compare_logits.cpp` | elementwise logit difference between two exports of one checkpoint |
| `probe_tiers.cpp` | per device: ggml type, `buft_is_host`, `caps.host_buffer` |
| `probe_selection.cpp` | registry order, and what `auto`/`gpu`/`npu`/`cpu` resolve to |
| `probe_chain.cpp` | primary, assist count, and the full `schedule_order()` for a given spec |

The three probes are from the P4.8b device-ranking work (BACKLOG.md), and each caught something a
test would not have:

* **`probe_tiers`** showed `ggml_backend_dev_props::caps.host_buffer` is the INVERSE of what it sounds
  like — it means "can hand out pinned staging buffers", and measures true for Vulkan, false for BLAS.
  The question the ranking actually asks is `ggml_backend_buft_is_host` on the default buffer type.
* **`probe_selection`** caught `Device::open("gpu")` resolving to BLAS in a `GGML_BACKEND_DL` build and
  to Vulkan0 in a linked one, from the same source tree — the defect P4.8b exists to fix.
* **`probe_chain`** proved the host-memory assist is genuinely in the scheduler order while split
  counts stayed flat. Without it, "adding BLAS to the chain changed nothing" would have been
  indistinguishable from "the assist was never attached".

## Building one

They link the engine directly, so paths depend on which build tree you point at. From the repo root:

```sh
BUILD=build-dl3            # or build, or a CUDA build
GB=$PWD/$BUILD/_deps/ggml-build/src
g++ -std=c++17 -O0 tools/debug/probe_chain.cpp \
    -I include -I $BUILD/_deps/ggml-src/include -I $BUILD/_deps/nlohmann_json-src/include \
    -L $BUILD -L $GB -lloom_engine -lggml-base \
    -Wl,-rpath,$PWD/$BUILD -Wl,-rpath,$GB -o /tmp/probe_chain
```

`probe_tiers.cpp` needs only ggml — drop the engine and json includes, link `-lggml-base -lggml`.

**In a `GGML_BACKEND_DL` build, run the binary from the directory holding the backend `.so` files**
(`$BUILD/bin`). ggml's search covers the executable's directory and the current directory, and a
hand-compiled probe in `/tmp` is in neither, so the registry otherwise comes up empty — which looks
exactly like a machine with no devices.

## The first question to point them at

`probe_tiers`, on a box with a real NPU. **The rank-1 tier rests on an untested assumption**: that a
discrete NPU registers as `GGML_BACKEND_DEVICE_TYPE_ACCEL` with a non-host buffer type. Only BLAS
(ACCEL, host memory) and Vulkan (IGPU, device memory) have ever been measured. If an NPU reports
host memory it belongs in rank 2 and the hierarchy needs revisiting; if it is not ACCEL at all,
`Device::open("npu")` never resolves and that spec is dead code.

Then `probe_chain gpu` and `probe_chain npu` on a machine with both — the first place where the
rank-0/rank-1 distinction, and the tie-break within a rank, are observable at all.
