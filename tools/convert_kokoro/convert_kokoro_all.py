"""Master conversion script: produces the FULL set of GGUF files needed by loom::KokoroDriver from the
ONE real `kokoro-v1_0.pth` checkpoint. Every individual piece (CustomAlbert, bert_encoder, TextEncoder,
ProsodyPredictor's duration half, F0Ntrain, the STFT/SineGen pair, the Decoder core, the Generator) was
already built and numerically verified as its own standalone script earlier this milestone -- this just
orchestrates them against the real checkpoint in one place, adding the two pieces (SineGen's `l_linear`,
the Generator/Decoder-core's real weights) whose standalone scripts used SYNTHETIC weights for their own
structural verification (see BACKLOG.md) and now need the REAL ones wired in.

Usage: python3 convert_kokoro_all.py <kokoro-v1_0.pth> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

import convert_kokoro_albert
import convert_kokoro_bert_encoder
import convert_kokoro_decoder_core
import convert_kokoro_duration_predictor
import convert_kokoro_f0n
import convert_kokoro_generator
import convert_kokoro_sinegen
import convert_kokoro_stft
import convert_kokoro_text_encoder


def write_gguf(path, topology, weights, architecture):
    w = GGUFWriter(str(path), architecture)
    w.add_string("model.graph_topology", json.dumps(topology))
    for name, arr in weights.items():
        w.add_tensor(name, arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {path}, {len(weights)} weights")


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <kokoro-v1_0.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # --- Pieces whose own main() already does the full real-weight job against (ckpt_path, out_dir) ---
    for mod, name in (
        (convert_kokoro_albert, "CustomAlbert"),
        (convert_kokoro_bert_encoder, "bert_encoder"),
        (convert_kokoro_text_encoder, "TextEncoder"),
        (convert_kokoro_duration_predictor, "ProsodyPredictor duration half"),
        (convert_kokoro_f0n, "F0Ntrain (AdainResBlk1d family)"),
    ):
        sys.argv = [mod.__file__, ckpt_path, str(out_dir)]
        mod.main()
        print(f"--- {name} done ---")

    # --- STFT: no real checkpoint weights at all (pure constant DFT/window kernels) ---
    sys.argv = [convert_kokoro_stft.__file__, str(out_dir)]
    convert_kokoro_stft.main()
    print("--- STFT done ---")

    # --- SineGen: real l_linear weights (module.generator.m_source.l_linear.*), everything else constant ---
    gen_sd = sd_all["decoder"]
    l_linear_w = gen_sd["module.generator.m_source.l_linear.weight"].detach().cpu().numpy().astype(np.float32)
    l_linear_b = gen_sd["module.generator.m_source.l_linear.bias"].detach().cpu().numpy().astype(np.float32)
    sinegen_topo, sinegen_weights = convert_kokoro_sinegen.build_sinegen(convert_kokoro_sinegen.HP, l_linear_w, l_linear_b)
    write_gguf(out_dir / "kokoro_sinegen.gguf", sinegen_topo, sinegen_weights, "loom-kokoro-sinegen")
    print("--- SineGen done ---")

    # --- Decoder core: real weights, sd_all["decoder"]'s own keys are "module.encode.*"/"module.decode.*"/
    #     "module.F0_conv.*"/"module.N_conv.*"/"module.asr_res.*" ---
    decoder_hp = convert_kokoro_decoder_core.HP
    decoder_topo, decoder_weights = convert_kokoro_decoder_core.build_decoder_core(decoder_hp, gen_sd, "module")
    write_gguf(out_dir / "kokoro_decoder_core.gguf", decoder_topo, decoder_weights, "loom-kokoro-decoder-core")
    print("--- Decoder core done ---")

    # --- Generator: real weights, sd_all["decoder"]'s own keys are "module.generator.*" ---
    generator_hp = convert_kokoro_generator.HP
    generator_topo, generator_weights = convert_kokoro_generator.build_generator(generator_hp, gen_sd, "module.generator")
    write_gguf(out_dir / "kokoro_generator.gguf", generator_topo, generator_weights, "loom-kokoro-generator")
    print("--- Generator done ---")

    print(f"\nAll Kokoro GGUF files written to {out_dir}")


if __name__ == "__main__":
    main()
