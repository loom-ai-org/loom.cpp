"""Converts the pieces of StyleTTS2's real `yl4579/StyleTTS2-LJSpeech` checkpoint that are
ARCHITECTURALLY IDENTICAL to what Kokoro already forced us to build (see PLAN.md's "StyleTTS2 and Kokoro
are almost the same pipeline" section): `CustomAlbert`/PL-BERT, `bert_encoder`, `TextEncoder`,
`ProsodyPredictor`'s duration half (`DurationEncoder`/`lstm`/`duration_proj`), `F0Ntrain`
(`AdainResBlk1d` family), the STFT pair, and the Decoder core + Generator (istftnet, SAME
`upsample_rates`/`gen_istft_n_fft`/etc as Kokoro's own checkpoint, confirmed directly against
`config.yml`).

Rather than duplicating ~2000 lines of already-verified builder code, this imports
`tools/convert_kokoro/convert_kokoro_*.py` directly (added to `sys.path`) and calls their EXISTING
`build_*()`/`main()` functions unchanged against the NEW checkpoint -- every one of those builders was
already written taking `sd`/`sd_prefix` as plain parameters (or, for bert/text_encoder/duration_predictor,
hardcoding a `"module."` prefix that happens to ALSO be exactly this checkpoint's own real prefix,
confirmed directly against the state dict before assuming it, not guessed) -- so genuinely NO changes to
any Kokoro file were needed, only a different checkpoint path pointed at the same code. This is real
reuse earned by the fact the two checkpoints share byte-identical sub-architecture hyperparameters, not
a coincidental shortcut.

Deliberately NOT included here (see PLAN.md): the diffusion-based style sampler (StyleTTS2's own
genuinely new piece, no Kokoro equivalent -- built separately, see convert_styletts2_diffusion.py) and
`style_encoder`/`predictor_encoder` (real checkpoint pieces that exist but are never called by the real
`Demo/Inference_LJSpeech.ipynb`'s own `inference()` function -- deferred, same as Kokoro deferring its
phonemizer).

Usage: python3 convert_styletts2_reused.py <epoch_2nd_00100.pth> <out_dir>
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "convert_kokoro"))

import convert_kokoro_albert
import convert_kokoro_bert_encoder
import convert_kokoro_decoder_core
import convert_kokoro_duration_predictor
import convert_kokoro_f0n
import convert_kokoro_generator
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
        print(f"usage: {sys.argv[0]} <epoch_2nd_00100.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)["net"]
    # (Kokoro's own checkpoint is a bare state dict; StyleTTS2's is {"net": {...}} -- the ONE real
    # structural difference between the two files, confirmed directly against both checkpoints' own
    # torch.load() output before writing this line.) Every convert_kokoro_*.py script below does its own
    # `torch.load(ckpt_path)` internally and expects THAT bare-dict layout, so re-pack `sd_all` into a
    # temp file matching it -- zero changes needed to any already-verified/committed Kokoro file.
    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        torch.save(sd_all, tmp.name)

        for mod, name in (
            (convert_kokoro_albert, "CustomAlbert"),
            (convert_kokoro_bert_encoder, "bert_encoder"),
            (convert_kokoro_text_encoder, "TextEncoder"),
            (convert_kokoro_duration_predictor, "ProsodyPredictor duration half"),
            (convert_kokoro_f0n, "F0Ntrain (AdainResBlk1d family)"),
        ):
            sys.argv = [mod.__file__, tmp.name, str(out_dir)]
            mod.main()
            print(f"--- {name} done ---")

    sys.argv = [convert_kokoro_stft.__file__, str(out_dir)]
    convert_kokoro_stft.main()
    print("--- STFT done ---")

    gen_sd = sd_all["decoder"]

    decoder_hp = convert_kokoro_decoder_core.HP
    decoder_topo, decoder_weights = convert_kokoro_decoder_core.build_decoder_core(decoder_hp, gen_sd, "module")
    write_gguf(out_dir / "styletts2_decoder_core.gguf", decoder_topo, decoder_weights, "loom-styletts2-decoder-core")
    print("--- Decoder core done ---")

    generator_hp = convert_kokoro_generator.HP
    generator_topo, generator_weights = convert_kokoro_generator.build_generator(generator_hp, gen_sd, "module.generator")
    write_gguf(out_dir / "styletts2_generator.gguf", generator_topo, generator_weights, "loom-styletts2-generator")
    print("--- Generator done ---")

    l_linear_w = gen_sd["module.generator.m_source.l_linear.weight"].detach().cpu().numpy().astype(np.float32)
    l_linear_b = gen_sd["module.generator.m_source.l_linear.bias"].detach().cpu().numpy().astype(np.float32)
    import convert_kokoro_sinegen
    sinegen_topo, sinegen_weights = convert_kokoro_sinegen.build_sinegen(convert_kokoro_sinegen.HP, l_linear_w, l_linear_b)
    write_gguf(out_dir / "styletts2_sinegen.gguf", sinegen_topo, sinegen_weights, "loom-styletts2-sinegen")
    print("--- SineGen done ---")

    print(f"\nAll reused-from-Kokoro StyleTTS2 GGUF files written to {out_dir}")


if __name__ == "__main__":
    main()
