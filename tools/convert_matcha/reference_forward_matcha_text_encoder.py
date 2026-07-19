"""Hand-independent ground truth for the TextEncoder conversion: loads the REAL
`matcha.models.components.text_encoder.TextEncoder` module directly (the real package is pip
installed editable in `/home/flavio/.venvs/matcha`, real checkpoint weights loadable verbatim) and
runs its real forward pass on a small token sequence -- same "use the real module as ground truth"
precedent as SupertonicTTS's own `.pt`-module-based reference scripts (both `supertonic-tts` and
`matcha` are real pip-installed packages, unlike VITS/Kokoro where the training code had its own
awkward dependencies and a hand-rolled reimplementation was preferred instead).

Usage: python3 reference_forward_matcha_text_encoder.py <matcha_ljspeech.ckpt> <out_dir>
Writes `<out_dir>/ref_text_encoder_{tokens,mu,logw}.npy`.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from matcha.models.components.text_encoder import TextEncoder


def main():
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    sd = ckpt["state_dict"]

    encoder_params = OmegaConf.create(dict(hp["encoder"]["encoder_params"]))
    dp_params = OmegaConf.create(dict(hp["encoder"]["duration_predictor_params"]))

    encoder = TextEncoder(
        hp["encoder"]["encoder_type"], encoder_params, dp_params,
        hp["n_vocab"], hp["n_spks"], hp["spk_emb_dim"],
    )
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    encoder.eval()

    # A small, arbitrary real-range token sequence (n_vocab=178) -- no padding (single utterance).
    tokens = np.array([5, 42, 7, 88, 13, 100, 3, 61], dtype=np.int64)
    x = torch.from_numpy(tokens).unsqueeze(0)
    x_lengths = torch.tensor([tokens.shape[0]])

    with torch.no_grad():
        mu, logw, x_mask = encoder(x, x_lengths, spks=None)

    np.save(out_dir / "ref_text_encoder_tokens.npy", tokens.astype(np.int32))
    np.save(out_dir / "ref_text_encoder_mu.npy", mu.squeeze(0).numpy().astype(np.float32))
    np.save(out_dir / "ref_text_encoder_logw.npy", logw.squeeze(0).squeeze(0).numpy().astype(np.float32))
    print("mu shape", mu.shape, "logw shape", logw.shape)
    print("mu[0,:5]", mu[0, 0, :5])
    print("logw[0,0,:5]", logw[0, 0, :5])


if __name__ == "__main__":
    main()
