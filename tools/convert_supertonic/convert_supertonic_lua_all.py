"""Converts SupertonicTTS v2 into a SINGLE self-contained loom-engine GGUF file (`supertonic.gguf`):
the four topologies `loom::SupertonicDriver::synthesize()` actually uses (DurationPredictor, TTLTextEncoder,
VectorFieldEstimator, SpeechDecoder -- NOT the two style encoders, which the driver never calls: real
`SpeechGenerator.predict()` always takes PRECOMPUTED style embeddings, matching this project's own
established "skip the reference-audio style encoder" scope decision) plus the embedded Lua orchestration
script (`model.driver_script`) -- the same one-GGUF-per-model convention already landed for Whisper (see
BACKLOG.md's dated entries).

Reuses every `build_*` helper AND every hyperparameter constant from `convert_supertonic_all.py`
UNCHANGED (imported directly, not duplicated) -- this script just re-runs the SAME four build calls into
their own `TopologyBuilder`s (instead of `convert_supertonic_all.py`'s own "one file per topology"
loop) and merges the resulting weight dicts into one file. `convert_supertonic_all.py`'s own six-file
output is untouched and still used by every existing per-module test (test_e2e_supertonic_dp.cpp,
test_e2e_supertonic_ttl_text.cpp, etc.).

Usage: python3 convert_supertonic_lua_all.py <supertonic-tts repo root> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

from supertonic_common import (TopologyBuilder, build_dp_text_encoder, build_speech_decoder,
                                build_ttl_text_encoder, build_vector_field_estimator)
from convert_supertonic_all import (DEC_HP, DP_DIM, DP_INTERM, DP_MLP_STL_DIM, DP_N_ATTN, DP_N_CN,
                                     DP_N_HEADS, DP_WINDOW, T_TEXT, TTL_N_ATTN, TTL_N_CN, TTL_N_HEADS,
                                     TTL_N_STYLE, TTL_STL_DIM, TTL_TEXT_DIM, TTL_TEXT_INTERM, TTL_WINDOW,
                                     VFE_HP, build_rel_pos_tables)


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <supertonic-tts root> <out_dir>", file=sys.stderr)
        sys.exit(1)
    repo_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    dp = torch.load(repo_root / "assets/pt/duration_predictor.pt", weights_only=False, map_location="cpu")
    te = torch.load(repo_root / "assets/pt/text_encoder.pt", weights_only=False, map_location="cpu")
    ve = torch.load(repo_root / "assets/pt/vector_estimator.pt", weights_only=False, map_location="cpu")
    dec = torch.load(repo_root / "assets/pt/vocoder.pt", weights_only=False, map_location="cpu")

    merged_weights = {}
    topologies = {}

    def merge(name, weights):
        for k, v in weights.items():
            assert k not in merged_weights, f"weight name collision merging '{name}': {k}"
            merged_weights[k] = v

    # --- DurationPredictor ---
    dp_sd = dict(dp.sentence_encoder.state_dict())
    for k, v in dp.state_dict().items():
        if k.startswith("layers.") or k.startswith("activation."):
            dp_sd[k] = v
    tables = build_rel_pos_tables(dp_sd, DP_N_ATTN, DP_WINDOW, T_TEXT + 1)
    tb = TopologyBuilder()
    utt_emb = build_dp_text_encoder(tb, dp_sd, DP_DIM, DP_INTERM, DP_N_CN, DP_N_ATTN, DP_N_HEADS,
                                     DP_WINDOW, tables, str(T_TEXT), T_TEXT, "utt_emb")
    x = tb.node("CONCAT", [utt_emb, "stl_emb"], {"dim": 0}, "mlp_in")
    w0 = tb.weight("dp.layers0.weight", dp_sd["layers.0.weight"].numpy().astype(np.float32))
    b0 = tb.weight("dp.layers0.bias", dp_sd["layers.0.bias"].numpy().astype(np.float32))
    h = tb.node("ADD", [tb.node("MUL_MAT", [w0, x], None, "l0_mm"), b0], None, "l0")
    prelu_w = tb.weight("dp.prelu_weight", dp_sd["activation.weight"].numpy().astype(np.float32))
    relu_pos = tb.node("RELU", [h], None, "prelu_pos")
    relu_neg = tb.node("RELU", [tb.node("SCALE", [h], {"s": -1.0}, "prelu_negh")], None, "prelu_relu_neg")
    h = tb.node("SUB", [relu_pos, tb.node("MUL", [relu_neg, prelu_w], None, "prelu_scaled")], None, "prelu_out")
    w1 = tb.weight("dp.layers1.weight", dp_sd["layers.1.weight"].numpy().astype(np.float32))
    b1 = tb.weight("dp.layers1.bias", dp_sd["layers.1.bias"].numpy().astype(np.float32))
    h = tb.node("ADD", [tb.node("MUL_MAT", [w1, h], None, "l1_mm"), b1], None, "l1")
    duration = tb.node("EXP", [h], None, "duration")
    dp_inputs = [
        {"name": "txt_ids", "dtype": "i32", "shape": [str(T_TEXT)]},
        {"name": "stl_emb", "dtype": "f32", "shape": [str(DP_MLP_STL_DIM)]},
    ]
    topologies["dp"] = tb.topology(dp_inputs, duration)
    merge("dp", tb.weights)

    # --- TTLTextEncoder ---
    te_sd_full = te.state_dict()
    te_pre_sd = {k[len("text_encoder."):]: v for k, v in te_sd_full.items() if k.startswith("text_encoder.")}
    te_pre_sd.update({k: v for k, v in te_sd_full.items() if k.startswith("speech_prompted_text_encoder.")})
    tables = build_rel_pos_tables(te_pre_sd, TTL_N_ATTN, TTL_WINDOW, T_TEXT)
    tb = TopologyBuilder()
    txt_emb = build_ttl_text_encoder(tb, te_pre_sd, "stl_emb_ttl_cb", TTL_TEXT_DIM, TTL_TEXT_INTERM,
                                      TTL_N_CN, TTL_N_ATTN, TTL_N_HEADS, TTL_WINDOW, tables, TTL_N_STYLE,
                                      str(T_TEXT), T_TEXT, "txt_emb")
    ttl_text_inputs = [
        {"name": "txt_ids", "dtype": "i32", "shape": [str(T_TEXT)]},
        {"name": "stl_emb_ttl_cb", "dtype": "f32", "shape": [str(TTL_STL_DIM), str(TTL_N_STYLE)]},
    ]
    topologies["ttl_text"] = tb.topology(ttl_text_inputs, txt_emb)
    merge("ttl_text", tb.weights)

    # --- VectorFieldEstimator ---
    ve_sd = ve.state_dict()
    tb = TopologyBuilder()
    v = build_vector_field_estimator(tb, ve_sd, "z_t", "txt_emb_cb", "stl_emb_ttl_cb", "t", "lat_frac",
                                      "txt_frac", VFE_HP, "$n_tokens", str(T_TEXT), "v")
    vfe_inputs = [
        {"name": "z_t", "dtype": "f32", "shape": ["$n_tokens", str(VFE_HP["latent_dim"])]},
        {"name": "txt_emb_cb", "dtype": "f32", "shape": [str(VFE_HP["txt_dim"]), str(T_TEXT)]},
        {"name": "stl_emb_ttl_cb", "dtype": "f32", "shape": [str(TTL_STL_DIM), str(TTL_N_STYLE)]},
        {"name": "t", "dtype": "f32", "shape": ["1"]},
        {"name": "lat_frac", "dtype": "f32", "shape": ["$n_tokens"]},
        {"name": "txt_frac", "dtype": "f32", "shape": [str(T_TEXT)]},
    ]
    topologies["vfe"] = tb.topology(vfe_inputs, v)
    merge("vfe", tb.weights)

    # --- SpeechDecoder ---
    dec_sd = dec.state_dict()
    tb = TopologyBuilder()
    wav = build_speech_decoder(tb, dec_sd, "latent", DEC_HP, "$n_tokens", "wav")
    decoder_inputs = [{"name": "latent", "dtype": "f32", "shape": ["$n_tokens", "144"]}]
    topologies["decoder"] = tb.topology(decoder_inputs, wav)
    merge("decoder", tb.weights)

    driver_script_path = Path(__file__).parent / "supertonic_driver.lua"

    w = GGUFWriter(str(out_dir / "supertonic.gguf"), "loom-supertonic")
    for name, topo in topologies.items():
        w.add_string(f"model.graph_topology.{name}", json.dumps(topo))
    w.add_string("model.driver_script", driver_script_path.read_text())
    for name, arr in merged_weights.items():
        w.add_tensor(name, arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_dir / 'supertonic.gguf'}, {len(merged_weights)} weights (T_TEXT={T_TEXT} fixed)")


if __name__ == "__main__":
    main()
