"""Master conversion script: produces the full real-weight GGUF set for `loom::SupertonicDriver` from
the real SupertonicTTS v2 checkpoint (`.pt` files under `assets/pt/`).

**Known scope limitation, confirmed real (not an oversight)**: `loom::GraphBuilder::build(n_tokens,
n_past)` resolves EVERY declared graph input's shape via a SINGLE dynamic-length symbol ("$n_tokens") --
there is no mechanism for a topology to declare a SECOND independently-sized dynamic input. SupertonicTTS's
`VectorFieldEstimator` genuinely needs TWO independently-varying lengths in one graph (`T_lat`, the
CFM-iterated latent-frame count, bound to "$n_tokens"; and `T_text`, the input utterance's own phoneme
count, needed by `VFTextCrossAttention`'s cross-attention -- fixed for the WHOLE Euler loop of a given
utterance, but different across DIFFERENT utterances). This is the first model in this whole project
needing two such lengths in one graph. For THIS milestone, `T_TEXT` is a FIXED CONSTANT (baked at
conversion time, matching this project's own established "one representative input length, not full
dynamic-shape generality" driver-smoke-test precedent -- see Kokoro's/StyleTTS2's own driver tests, which
also use one fixed demo token sequence, not a arbitrary-length tokenizer end-to-end). A real production
driver would need EITHER a new engine mechanism for multi-symbol graphs OR per-utterance topology-JSON
templating (a placeholder token substituted via plain string replace before `GraphTopology::parse`,
avoiding any change to `GraphBuilder`'s core symbol resolution) -- tracked as a follow-on, not solved here.

`emb_rel_k`/`emb_rel_v` (DP/TTL text encoders' own Shaw et al. relative-position tables) are likewise
baked as fixed constants for `T_TEXT` (not declared per-call inputs, unlike VITS's own convention) --
consistent with the same fixed-T_text scope for this milestone.

Usage: python3 convert_supertonic_all.py <supertonic-tts repo root> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch

from supertonic_common import (TopologyBuilder, to_f32, write_gguf, build_dp_text_encoder,
                                build_style_encoder, build_ttl_text_encoder, build_vector_field_estimator,
                                build_speech_decoder, cross_ta_to_cb)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "convert_piper_vits"))
from vits_common import get_relative_embeddings  # noqa: E402

T_TEXT = 10  # FIXED for this milestone -- see module docstring

DP_DIM, DP_INTERM, DP_N_CN, DP_N_ATTN, DP_N_HEADS, DP_WINDOW = 64, 256, 6, 2, 2, 4
DP_STL_INPUT_DIM, DP_STL_EMBED_DIM, DP_STL_INTERM, DP_STL_N_CN = 144, 64, 256, 4
DP_STL_N_STYLE, DP_STL_DIM = 8, 16
DP_MLP_STL_DIM, DP_MLP_UTT_DIM, DP_MLP_HIDDEN = 128, 64, 128

TTL_STYLE_EMBED_DIM, TTL_STYLE_INTERM, TTL_STYLE_N_CN = 256, 1024, 6
TTL_N_STYLE, TTL_STL_DIM = 50, 256
TTL_TEXT_DIM, TTL_TEXT_INTERM, TTL_N_CN, TTL_N_ATTN, TTL_N_HEADS, TTL_WINDOW = 256, 1024, 6, 4, 4, 4

VFE_HP = {
    "latent_dim": 144, "hidden_dim": 512, "interm_dim": 1024, "txt_dim": 256,
    "n_groups": 4, "n_cn_layers": 4, "time_emb_dim": 64,
    "n_text_heads": 4, "text_head_dim": 64, "n_style_heads": 2, "style_head_dim": 128,
    "n_style": 50, "stl_dim": 256,
}

DEC_HP = {"lat_channels": 24, "n_codebooks": 6, "hidden_dim": 512, "interm_dim": 2048,
          "cn_dilations": (1, 2, 4, 1, 2, 4, 1, 1, 1, 1)}

CROP_LEN = 50


def build_rel_pos_tables(sd_attn_layers, n_layers, window_size, seq_len):
    tables = []
    for i in range(n_layers):
        p = f"attn_layers.{i}"
        ek_raw = to_f32(sd_attn_layers[f"{p}.emb_rel_k"]).squeeze(0)
        ev_raw = to_f32(sd_attn_layers[f"{p}.emb_rel_v"]).squeeze(0)
        tables.append({
            "k": get_relative_embeddings(ek_raw, window_size, seq_len),
            "v": get_relative_embeddings(ev_raw, window_size, seq_len),
        })
    return tables


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <supertonic-tts root> <out_dir>", file=sys.stderr)
        sys.exit(1)
    repo_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    dp = torch.load(repo_root / "assets/pt/duration_predictor.pt", weights_only=False, map_location="cpu")
    dp_se = torch.load(repo_root / "assets/pt/dp-style-encoder.pt", weights_only=False, map_location="cpu")
    ttl_se = torch.load(repo_root / "assets/pt/ttl-style-encoder.pt", weights_only=False, map_location="cpu")
    te = torch.load(repo_root / "assets/pt/text_encoder.pt", weights_only=False, map_location="cpu")
    ve = torch.load(repo_root / "assets/pt/vector_estimator.pt", weights_only=False, map_location="cpu")
    dec = torch.load(repo_root / "assets/pt/vocoder.pt", weights_only=False, map_location="cpu")

    # --- DurationPredictor: DPTextEncoder + MLP head (stl_emb fed as a declared input, real
    #     `SpeechGenerator.predict()` convention -- it always calls `dur_predictor.predict(..., stl_emb=
    #     precomputed)`, never recomputing style from a lat_crop at predict() time). ---
    dp_sd = dict(dp.sentence_encoder.state_dict())
    for k, v in dp.state_dict().items():
        if k.startswith("layers.") or k.startswith("activation."):
            dp_sd[k] = v
    tables = build_rel_pos_tables(dp_sd, DP_N_ATTN, DP_WINDOW, T_TEXT + 1)
    tb = TopologyBuilder()
    utt_emb = build_dp_text_encoder(tb, dp_sd, DP_DIM, DP_INTERM, DP_N_CN, DP_N_ATTN, DP_N_HEADS,
                                     DP_WINDOW, tables, str(T_TEXT), T_TEXT, "utt_emb")
    x = tb.node("CONCAT", [utt_emb, "stl_emb"], {"dim": 0}, "mlp_in")
    w0 = tb.weight("dp.layers0.weight", to_f32(dp_sd["layers.0.weight"]))
    b0 = tb.weight("dp.layers0.bias", to_f32(dp_sd["layers.0.bias"]))
    h = tb.node("ADD", [tb.node("MUL_MAT", [w0, x], None, "l0_mm"), b0], None, "l0")
    prelu_w = tb.weight("dp.prelu_weight", to_f32(dp_sd["activation.weight"]))
    relu_pos = tb.node("RELU", [h], None, "prelu_pos")
    relu_neg = tb.node("RELU", [tb.node("SCALE", [h], {"s": -1.0}, "prelu_negh")], None, "prelu_relu_neg")
    h = tb.node("SUB", [relu_pos, tb.node("MUL", [relu_neg, prelu_w], None, "prelu_scaled")], None, "prelu_out")
    w1 = tb.weight("dp.layers1.weight", to_f32(dp_sd["layers.1.weight"]))
    b1 = tb.weight("dp.layers1.bias", to_f32(dp_sd["layers.1.bias"]))
    h = tb.node("ADD", [tb.node("MUL_MAT", [w1, h], None, "l1_mm"), b1], None, "l1")
    duration = tb.node("EXP", [h], None, "duration")
    inputs = [
        {"name": "txt_ids", "dtype": "i32", "shape": [str(T_TEXT)]},
        {"name": "stl_emb", "dtype": "f32", "shape": [str(DP_MLP_STL_DIM)]},
    ]
    write_gguf(out_dir / "supertonic_dp.gguf", tb.topology(inputs, duration), tb.weights, "loom-supertonic-dp")

    # --- DP/TTL style encoders (both operate at the FIXED crop_len=50, a real model constant, not a
    #     dynamic-per-utterance length -- no scope limitation here). ---
    dp_se_sd = dp_se.state_dict()
    tb = TopologyBuilder()
    stl_dp = build_style_encoder(tb, "lat_crop", dp_se_sd, "dp_se", DP_STL_INPUT_DIM, DP_STL_EMBED_DIM,
                                  DP_STL_INTERM, DP_STL_N_STYLE, DP_STL_DIM, DP_STL_N_CN, str(CROP_LEN),
                                  "stl_dp")
    inputs = [{"name": "lat_crop", "dtype": "f32", "shape": [str(CROP_LEN), str(DP_STL_INPUT_DIM)]}]
    write_gguf(out_dir / "supertonic_dp_style.gguf", tb.topology(inputs, stl_dp), tb.weights,
               "loom-supertonic-dp-style")

    ttl_se_sd = ttl_se.state_dict()
    tb = TopologyBuilder()
    stl_ttl = build_style_encoder(tb, "lat_crop", ttl_se_sd, "ttl_se", DP_STL_INPUT_DIM, TTL_STYLE_EMBED_DIM,
                                   TTL_STYLE_INTERM, TTL_N_STYLE, TTL_STL_DIM, TTL_STYLE_N_CN, str(CROP_LEN),
                                   "stl_ttl")
    inputs = [{"name": "lat_crop", "dtype": "f32", "shape": [str(CROP_LEN), str(DP_STL_INPUT_DIM)]}]
    write_gguf(out_dir / "supertonic_ttl_style.gguf", tb.topology(inputs, stl_ttl), tb.weights,
               "loom-supertonic-ttl-style")

    # --- TTLTextEncoder (stl_emb fed as a declared input, TTL style precomputed the same way). ---
    te_sd_full = te.state_dict()
    te_pre_sd = {k[len("text_encoder."):]: v for k, v in te_sd_full.items() if k.startswith("text_encoder.")}
    te_pre_sd.update({k: v for k, v in te_sd_full.items() if k.startswith("speech_prompted_text_encoder.")})
    tables = build_rel_pos_tables(te_pre_sd, TTL_N_ATTN, TTL_WINDOW, T_TEXT)
    tb = TopologyBuilder()
    txt_emb = build_ttl_text_encoder(tb, te_pre_sd, "stl_emb_ttl_cb", TTL_TEXT_DIM, TTL_TEXT_INTERM,
                                      TTL_N_CN, TTL_N_ATTN, TTL_N_HEADS, TTL_WINDOW, tables, TTL_N_STYLE,
                                      str(T_TEXT), T_TEXT, "txt_emb")
    inputs = [
        {"name": "txt_ids", "dtype": "i32", "shape": [str(T_TEXT)]},
        {"name": "stl_emb_ttl_cb", "dtype": "f32", "shape": [str(TTL_STL_DIM), str(TTL_N_STYLE)]},
    ]
    write_gguf(out_dir / "supertonic_ttl_text.gguf", tb.topology(inputs, txt_emb), tb.weights,
               "loom-supertonic-ttl-text")

    # --- VectorFieldEstimator: T_lat="$n_tokens" (dynamic, resolved per synthesize() call once duration
    #     is known), T_text=T_TEXT (fixed, see module docstring). txt_emb_cb fed as a declared input
    #     (already Layout B, produced by the TTLTextEncoder call above then crossed by the driver). ---
    ve_sd = ve.state_dict()
    tb = TopologyBuilder()
    v = build_vector_field_estimator(tb, ve_sd, "z_t", "txt_emb_cb", "stl_emb_ttl_cb", "t", "lat_frac",
                                      "txt_frac", VFE_HP, "$n_tokens", str(T_TEXT), "v")
    inputs = [
        {"name": "z_t", "dtype": "f32", "shape": ["$n_tokens", str(VFE_HP["latent_dim"])]},
        {"name": "txt_emb_cb", "dtype": "f32", "shape": [str(VFE_HP["txt_dim"]), str(T_TEXT)]},
        {"name": "stl_emb_ttl_cb", "dtype": "f32", "shape": [str(TTL_STL_DIM), str(TTL_N_STYLE)]},
        {"name": "t", "dtype": "f32", "shape": ["1"]},
        {"name": "lat_frac", "dtype": "f32", "shape": ["$n_tokens"]},
        {"name": "txt_frac", "dtype": "f32", "shape": [str(T_TEXT)]},
    ]
    write_gguf(out_dir / "supertonic_vfe.gguf", tb.topology(inputs, v), tb.weights, "loom-supertonic-vfe")

    # --- SpeechDecoder: T_lat="$n_tokens" (dynamic, same value used for the CFM loop above). ---
    dec_sd = dec.state_dict()
    tb = TopologyBuilder()
    wav = build_speech_decoder(tb, dec_sd, "latent", DEC_HP, "$n_tokens", "wav")
    inputs = [{"name": "latent", "dtype": "f32", "shape": ["$n_tokens", "144"]}]
    write_gguf(out_dir / "supertonic_decoder.gguf", tb.topology(inputs, wav), tb.weights,
               "loom-supertonic-decoder")

    print(f"\nAll SupertonicTTS v2 GGUF files written to {out_dir} (T_TEXT={T_TEXT} fixed)")


if __name__ == "__main__":
    main()
