"""Converts VITS (piper) into a SINGLE self-contained loom-engine GGUF file (`vits.gguf`): the three
topologies `loom::VitsDriver::synthesize()` uses (`stats`, `logw`, `flow_vocoder`) plus the embedded Lua
orchestration script (`model.driver_script`) -- the same one-GGUF-per-model convention already landed
for Whisper/SupertonicTTS/Matcha-TTS (see BACKLOG.md's dated entries).

Reuses `build_text_sdp_topologies`/`build_flow_vocoder_topology` UNCHANGED, imported directly from the
existing `convert_vits.py` -- this script just calls them and merges the resulting weight dicts.
`convert_vits.py`'s own three-file output is untouched.

Real weight-name collision, expected and handled (not a bug): `stats`/`logw` both independently build
TextEncoder from scratch (same reasoning as Matcha's own mu/logw split) -- both register the SAME
`enc_p.*` weights (including the raw relative-position tables) under identical names with identical
values. The merge below is content-aware: identical-name+identical-value is a silent dedup, identical-
name+DIFFERENT-value is a hard error.

Usage: python3 convert_vits_lua_all.py <model.ckpt> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

from convert_vits import HP, build_text_sdp_topologies, build_flow_vocoder_topology
from vits_common import load_piper_checkpoint


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <model.ckpt> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    full_sd = load_piper_checkpoint(ckpt_path)
    sd = {k[len("model_g."):]: v for k, v in full_sd.items() if k.startswith("model_g.")}

    merged_weights = {}
    merged_int32 = set()

    def merge(name, weights, int32_names):
        for k, v in weights.items():
            if k in merged_weights:
                assert np.array_equal(merged_weights[k], v), \
                    f"real weight name collision merging '{name}': '{k}' has DIFFERENT values across modules"
                continue
            merged_weights[k] = v
        merged_int32.update(int32_names)

    stats_topo, logw_topo, stats_weights, stats_int32, logw_weights, logw_int32 = build_text_sdp_topologies(sd)
    merge("stats", stats_weights, stats_int32)
    merge("logw", logw_weights, logw_int32)

    flow_vocoder_topo, flow_vocoder_weights, flow_vocoder_int32 = build_flow_vocoder_topology(sd)
    merge("flow_vocoder", flow_vocoder_weights, flow_vocoder_int32)

    driver_script_path = Path(__file__).parent / "vits_driver.lua"

    w = GGUFWriter(str(out_dir / "vits.gguf"), "loom-vits")
    w.add_string("model.graph_topology.stats", json.dumps(stats_topo))
    w.add_string("model.graph_topology.logw", json.dumps(logw_topo))
    w.add_string("model.graph_topology.flow_vocoder", json.dumps(flow_vocoder_topo))
    w.add_string("model.driver_script", driver_script_path.read_text())
    for key, value in HP.items():
        if isinstance(value, float):
            w.add_float32(f"loom.{key}", value)
        elif isinstance(value, int):
            w.add_uint32(f"loom.{key}", value)
    for name, arr in merged_weights.items():
        if name in merged_int32:
            w.add_tensor(name, arr.astype(np.int32))
        else:
            w.add_tensor(name, arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_dir / 'vits.gguf'}, {len(merged_weights)} weights")


if __name__ == "__main__":
    main()
