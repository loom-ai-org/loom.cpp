"""Converts SpeechDecoder (real source: speech_autoencoding/speech_autoencoder.py) -- the FINAL stage of
the whole SupertonicTTS v2 pipeline (compressed 144-channel latent -> raw waveform, no ISTFT/vocoder-GAN
stack at all). See `build_speech_decoder`'s own docstring in supertonic_common.py for the full real-math
derivation (codebook decompress, folded BatchNorm, causal ConvNeXt, direct-waveform-emission head).

Usage: python3 convert_supertonic_decoder.py <supertonic-tts repo root> <out_dir>
"""
import sys
from pathlib import Path

import torch

from supertonic_common import TopologyBuilder, write_gguf, build_speech_decoder

HP = {
    "lat_channels": 24,
    "n_codebooks": 6,
    "hidden_dim": 512,
    "interm_dim": 2048,
    "cn_dilations": (1, 2, 4, 1, 2, 4, 1, 1, 1, 1),
}


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <supertonic-tts root> <out_dir>", file=sys.stderr)
        sys.exit(1)
    repo_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    dec = torch.load(repo_root / "assets/pt/vocoder.pt", weights_only=False, map_location="cpu")
    sd = dec.state_dict()

    T = 4
    tb = TopologyBuilder()
    wav = build_speech_decoder(tb, sd, "latent", HP, str(T), "wav")
    inputs = [{"name": "latent", "dtype": "f32", "shape": [str(T), "144"]}]
    write_gguf(out_dir / "supertonic_decoder.gguf", tb.topology(inputs, wav), tb.weights,
               "loom-supertonic-decoder")


if __name__ == "__main__":
    main()
