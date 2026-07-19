"""Converts OpenAI Whisper into a SINGLE self-contained loom-engine GGUF file (`whisper.gguf`): both the
encoder and decoder topologies (named `model.graph_topology.encoder`/`model.graph_topology.decoder`) plus
the embedded Lua orchestration script (`model.driver_script`, `whisper_driver.lua`) -- the actual
end-state `LOOM_PROCEDURAL_GENERALIZATION.md`/`LOOM_MIL_CONVERSION.md` are aiming for (one GGUF = one
deployable model artifact), superseding the earlier two-separate-files convention for the DRIVER-level
use case (see BACKLOG.md's dated entry).

Reuses `build_encoder`/`build_decoder` from `convert_whisper_encoder.py`/`convert_whisper_decoder.py`
UNCHANGED (both already return a topology-builder-agnostic `(TopologyBuilder, ...)` pair) -- this script
just builds each into its own `TopologyBuilder`, merges their weight dicts, and writes one file. Real
weight names already come from the checkpoint's own module-qualified naming (`encoder.*`/`decoder.*`/
`mel.*`, confirmed via direct inspection of both scripts) -- zero collisions merging the two dicts.

`convert_whisper_encoder.py`/`convert_whisper_decoder.py`'s own two-separate-file output is UNCHANGED and
still used by the per-module isolation tests (test_e2e_whisper_{encoder,decoder}_reference.cpp), which
test each topology against a Python reference independent of the other.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

from convert_whisper_encoder import TopologyBuilder, build_encoder
from convert_whisper_decoder import build_decoder


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <model.pt> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    dims = checkpoint["dims"]
    sd = checkpoint["model_state_dict"]

    from whisper_common import mel_hparams

    # --- encoder ---
    enc_tb = TopologyBuilder()
    enc_x, _n_frames = build_encoder(enc_tb, sd, dims)
    hp = mel_hparams(dims["n_mels"])
    n_samples_padded = hp["n_samples"] + 2 * hp["reflect_pad"]
    encoder_inputs = [
        {"name": "waveform", "dtype": "f32", "shape": [str(n_samples_padded), "1", "1"]},
        {"name": "enc_attn_mask", "dtype": "f32", "shape": [str(dims["n_audio_ctx"]), str(dims["n_audio_ctx"])]},
    ]
    encoder_topo = enc_tb.topology(encoder_inputs, enc_x)

    # --- decoder ---
    dec_tb = TopologyBuilder()
    logits = build_decoder(dec_tb, sd, dims)
    n_state = dims["n_text_state"]
    n_audio_ctx = dims["n_audio_ctx"]
    decoder_inputs = [
        {"name": "tokens", "dtype": "i32", "shape": ["$n_tokens"]},
        {"name": "positions", "dtype": "i32", "shape": ["$n_tokens"]},
        {"name": "kq_mask", "dtype": "f32", "shape": ["$n_kv", "$n_tokens"]},
        {"name": "xa", "dtype": "f32", "shape": [str(n_state), str(n_audio_ctx)]},
        {"name": "xa_mask", "dtype": "f32", "shape": [str(n_audio_ctx), "$n_tokens"]},
    ]
    decoder_topo = dec_tb.topology(decoder_inputs, logits)

    # Merge weights -- confirmed no key collisions (real checkpoint's own "encoder."/"decoder."/"mel."
    # module-qualified naming, see this module's own docstring).
    merged_weights = dict(enc_tb.weights)
    for name, arr in dec_tb.weights.items():
        assert name not in merged_weights, f"weight name collision merging encoder+decoder: {name}"
        merged_weights[name] = arr

    driver_script_path = Path(__file__).parent / "whisper_driver.lua"

    writer = GGUFWriter(str(out_dir / "whisper.gguf"), "loom-whisper")
    writer.add_string("model.graph_topology.encoder", json.dumps(encoder_topo))
    writer.add_string("model.graph_topology.decoder", json.dumps(decoder_topo))
    writer.add_string("model.driver_script", driver_script_path.read_text())
    # hparams needed by loom::WhisperDriver / LoomLuaBridge's KvCache sizing.
    writer.add_uint32("loom.n_layer", dims["n_text_layer"])
    writer.add_uint32("loom.n_head_kv", dims["n_text_head"])
    writer.add_uint32("loom.n_embd_head_k", n_state // dims["n_text_head"])
    writer.add_uint32("loom.n_embd_head_v", n_state // dims["n_text_head"])
    for name, arr in merged_weights.items():
        writer.add_tensor(name, arr.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {out_dir / 'whisper.gguf'}, {len(merged_weights)} weights")


if __name__ == "__main__":
    main()
