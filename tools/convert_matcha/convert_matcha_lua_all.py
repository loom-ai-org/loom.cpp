"""Converts Matcha-TTS into a SINGLE self-contained loom-engine GGUF file (`matcha.gguf`): the four
topologies `loom::MatchaDriver::synthesize()` uses (TextEncoder mu/logw, Decoder U-Net, HiFi-GAN v1
vocoder) plus the embedded Lua orchestration script (`model.driver_script`) -- the same one-GGUF-per-model
convention already landed for Whisper and SupertonicTTS (see BACKLOG.md's dated entries).

Reuses every `build_*_topology`/`build_decoder`/`build_vocoder` function UNCHANGED, imported directly
from the existing per-module scripts (`convert_matcha_text_encoder.py`/`convert_matcha_decoder.py`/
`convert_matcha_vocoder.py`) -- this script just calls them and merges the resulting weight dicts. Those
scripts' own four-file output is untouched and still used by every existing per-module test.

Usage: python3 convert_matcha_lua_all.py <matcha_ljspeech.ckpt> <generator_v1> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

from matcha_common import load_matcha_checkpoint, load_hifigan_checkpoint
from convert_matcha_text_encoder import build_mu_topology, build_logw_topology, HP as TEXT_ENCODER_HP
from convert_matcha_decoder import build_decoder, HP as DECODER_HP
from convert_matcha_vocoder import build_vocoder, HP as VOCODER_HP


def main():
    if len(sys.argv) < 4:
        print(f"usage: {sys.argv[0]} <matcha_ljspeech.ckpt> <generator_v1> <out_dir>", file=sys.stderr)
        sys.exit(1)
    matcha_ckpt_path, hifigan_ckpt_path, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    matcha_sd = load_matcha_checkpoint(matcha_ckpt_path)
    hifigan_sd = load_hifigan_checkpoint(hifigan_ckpt_path)

    merged_weights = {}
    merged_int32 = set()
    topologies = {}

    def merge(name, weights, int32_names):
        # Matcha's mu/logw topologies both build their own independent TopologyBuilder over the SAME
        # shared TextEncoder body (build_text_encoder_body) -- real, expected duplicate registration of
        # the IDENTICAL weight data under the same name (same precedent as VITS's own stats/logw file
        # split), not a genuine collision. Only a same-name DIFFERENT-value pair is a real error.
        for k, v in weights.items():
            if k in merged_weights:
                assert np.array_equal(merged_weights[k], v), \
                    f"real weight name collision merging '{name}': '{k}' has DIFFERENT values across modules"
                continue
            merged_weights[k] = v
        merged_int32.update(int32_names)

    mu_topo, mu_weights, mu_int32 = build_mu_topology(matcha_sd)
    topologies["encoder_mu"] = mu_topo
    merge("encoder_mu", mu_weights, mu_int32)

    logw_topo, logw_weights, logw_int32 = build_logw_topology(matcha_sd)
    topologies["encoder_logw"] = logw_topo
    merge("encoder_logw", logw_weights, logw_int32)

    dec_topo, dec_weights, dec_int32 = build_decoder(matcha_sd, DECODER_HP)
    topologies["decoder"] = dec_topo
    merge("decoder", dec_weights, dec_int32)

    voc_topo, voc_weights, voc_int32 = build_vocoder(hifigan_sd, VOCODER_HP)
    topologies["vocoder"] = voc_topo
    merge("vocoder", voc_weights, voc_int32)

    driver_script_path = Path(__file__).parent / "matcha_driver.lua"

    w = GGUFWriter(str(out_dir / "matcha.gguf"), "loom-matcha")
    for name, topo in topologies.items():
        w.add_string(f"model.graph_topology.{name}", json.dumps(topo))
    w.add_string("model.driver_script", driver_script_path.read_text())
    for name, arr in merged_weights.items():
        if name in merged_int32:
            w.add_tensor(name, arr.astype(np.int32))
        else:
            w.add_tensor(name, arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_dir / 'matcha.gguf'}, {len(merged_weights)} weights")


if __name__ == "__main__":
    main()
