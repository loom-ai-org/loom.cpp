"""Converts TTLStyleEncoder + TTLTextEncoder (both computed in ONE graph, `stl_emb` feeding directly
into `TTLTextEncoder` -- matching the real `TextToLatentWrapper.forward`/`predict`'s own call order:
`stl_emb = style_encoder(crop); txt_emb = text_encoder(txt_ids, stl_emb=stl_emb, txt_msk=txt_msk)`).

`emb_rel_k`/`emb_rel_v` tables are windowed for the SPECIFIC T this standalone test targets (see
`add_multihead_relative_attention`'s own docstring for the real per-call driver's own deferred approach).

Usage: python3 convert_supertonic_ttl_text.py <supertonic-tts repo root> <out_dir>
"""
import sys
from pathlib import Path

import torch

from supertonic_common import (TopologyBuilder, to_f32, write_gguf, build_style_encoder,
                                build_ttl_text_encoder)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "convert_piper_vits"))
from vits_common import get_relative_embeddings  # noqa: E402

LAT_DIM = 144
STYLE_EMBED_DIM = 256
STYLE_INTERM_DIM = 1024  # confirmed against the real ttl-style-encoder.pt's own pwconv1 (256->1024)
STYLE_N_CN_LAYERS = 6
STYLE_STL_DIM = 256
STYLE_N_STYLE = 50

TEXT_DIM = 256
TEXT_INTERM_DIM = 1024
TEXT_N_CN_LAYERS = 6
TEXT_N_ATTN_LAYERS = 4
TEXT_N_HEADS = 4
TEXT_WINDOW_SIZE = 4

CROP_LEN = 50


def build_all(tb, se_sd, te_sd, T):
    stl_emb_cb = build_style_encoder(tb, "lat_crop", se_sd, "ttl_se", LAT_DIM, STYLE_EMBED_DIM,
                                      STYLE_INTERM_DIM, STYLE_N_STYLE, STYLE_STL_DIM, STYLE_N_CN_LAYERS,
                                      str(CROP_LEN), "stl_emb")  # [256, 50] Layout B

    tables = []
    for i in range(TEXT_N_ATTN_LAYERS):
        p = f"text_encoder.attn_layers.{i}"
        ek_raw = to_f32(te_sd[f"{p}.emb_rel_k"]).squeeze(0)
        ev_raw = to_f32(te_sd[f"{p}.emb_rel_v"]).squeeze(0)
        tables.append({
            "k": get_relative_embeddings(ek_raw, TEXT_WINDOW_SIZE, T),
            "v": get_relative_embeddings(ev_raw, TEXT_WINDOW_SIZE, T),
        })

    # `build_ttl_text_pre_encoder` reads UNPREFIXED keys from its own `sd` dict (the real checkpoint
    # nests them under "text_encoder."); `build_speech_prompted_text_encoder` is called with
    # `sd_prefix="speech_prompted_text_encoder"`, so ITS keys should stay exactly as they already are in
    # `te_sd` -- only the "text_encoder.*" keys need stripping.
    merged_sd = {k[len("text_encoder."):]: v for k, v in te_sd.items() if k.startswith("text_encoder.")}
    merged_sd.update({k: v for k, v in te_sd.items() if k.startswith("speech_prompted_text_encoder.")})

    txt_emb_ta = build_ttl_text_encoder(tb, merged_sd, stl_emb_cb, TEXT_DIM, TEXT_INTERM_DIM,
                                        TEXT_N_CN_LAYERS, TEXT_N_ATTN_LAYERS, TEXT_N_HEADS,
                                        TEXT_WINDOW_SIZE, tables, STYLE_N_STYLE, "$n_tokens", T, "txt_emb_ta")
    # Reference dumps txt_emb as (1,256,T) native channel-first -- byte-identical to Layout A [T,256],
    # which is exactly `build_ttl_text_encoder`'s own return convention. No further crossing needed.
    return txt_emb_ta, stl_emb_cb


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <supertonic-tts root> <out_dir>", file=sys.stderr)
        sys.exit(1)
    repo_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    ttl_se = torch.load(repo_root / "assets/pt/ttl-style-encoder.pt", weights_only=False, map_location="cpu")
    te = torch.load(repo_root / "assets/pt/text_encoder.pt", weights_only=False, map_location="cpu")
    se_sd = ttl_se.state_dict()
    te_sd = te.state_dict()

    T = 10
    tb = TopologyBuilder()
    txt_emb, stl_emb = build_all(tb, se_sd, te_sd, T)
    inputs = [
        {"name": "lat_crop", "dtype": "f32", "shape": [str(CROP_LEN), str(LAT_DIM)]},
        {"name": "txt_ids", "dtype": "i32", "shape": ["$n_tokens"]},
    ]
    topo = tb.topology(inputs, txt_emb)
    # NOTE: this topology declares a SINGLE output ("txt_emb"); `stl_emb` is verified via a SEPARATE
    # build below sharing the same weights, since GraphTopology supports exactly one declared output per
    # topology (established VITS-era convention -- multiple outputs need multiple GGUF files).
    write_gguf(out_dir / "supertonic_ttl_text.gguf", topo, tb.weights, "loom-supertonic-ttl-text")

    tb2 = TopologyBuilder()
    stl_only = build_style_encoder(tb2, "lat_crop", se_sd, "ttl_se", LAT_DIM, STYLE_EMBED_DIM,
                                    STYLE_INTERM_DIM, STYLE_N_STYLE, STYLE_STL_DIM, STYLE_N_CN_LAYERS,
                                    str(CROP_LEN), "stl_emb")
    inputs2 = [{"name": "lat_crop", "dtype": "f32", "shape": [str(CROP_LEN), str(LAT_DIM)]}]
    write_gguf(out_dir / "supertonic_ttl_style.gguf", tb2.topology(inputs2, stl_only), tb2.weights,
               "loom-supertonic-ttl-style")


if __name__ == "__main__":
    main()
