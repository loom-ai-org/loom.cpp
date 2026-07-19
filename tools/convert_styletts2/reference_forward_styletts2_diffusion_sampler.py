"""Hand-rolled ground truth for the FULL style-diffusion sampling loop: `KarrasSchedule` +
`ADPM2Sampler` (real source: Modules/diffusion/sampler.py, already independently verified in
tools/fixture_gen/reference_style_diffusion_sampler.py's own toy-denoiser fixture) wrapped around
`KDiffusion`'s real `c_skip`/`c_out`/`c_in`/`c_noise` preconditioning (real source: same file's
`KDiffusion.get_scale_weights`/`denoise_fn`) driving the REAL `Transformer1d` network (reusing
`reference_forward_styletts2_diffusion.py`'s own `transformer1d_forward`, already verified against real
checkpoint weights in isolation). This is the first point in this project where the sampler loop and the
real network are combined -- everything downstream (`StyleTTS2Driver`) builds on this being correct.

Same "real weights + synthetic driving embedding" scope as `reference_forward_styletts2_diffusion.py`
(no real BERT forward needed to check the SAMPLING LOOP's own correctness). `embedding_scale=1.0`
throughout (see that script's own docstring for why the CFG branch is out of scope). Noise (both the
initial `noise` and every step's ancestral noise draw) is pre-generated here with a fixed seed and
dumped so `test_e2e_styletts2_diffusion_sampler.cpp` can replay the EXACT SAME sequence -- same
decoupling-RNG-from-math rationale as the toy-denoiser fixture.

Usage: python3 reference_forward_styletts2_diffusion_sampler.py <epoch_2nd_00100.pth> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_forward_styletts2_diffusion import HP, transformer1d_forward


def karras_schedule(num_steps, sigma_min, sigma_max, rho):
    rho_inv = 1.0 / rho
    steps = np.arange(num_steps, dtype=np.float64)
    denom = max(num_steps - 1, 1)
    sigmas = (sigma_max ** rho_inv + (steps / denom) * (sigma_min ** rho_inv - sigma_max ** rho_inv)) ** rho
    return np.concatenate([sigmas, [0.0]])


def kdiffusion_denoise(x, sigma, embedding, sd, prefix, sigma_data):
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data * (sigma_data ** 2 + sigma ** 2) ** -0.5
    c_in = (sigma ** 2 + sigma_data ** 2) ** -0.5
    c_noise = np.log(sigma) * 0.25

    x_t = torch.from_numpy(x.astype(np.float32))
    x_scaled = x_t * float(c_in)
    x_pred = transformer1d_forward(x_scaled, torch.tensor(float(c_noise)), embedding, sd, prefix, HP)
    x_pred_np = x_pred.detach().numpy().astype(np.float64)
    return c_skip * x + c_out * x_pred_np


def adpm2_step(x, fn, sigma, sigma_next, noise):
    sigma_up = np.sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
    sigma_down = np.sqrt(sigma_next ** 2 - sigma_up ** 2)
    sigma_mid = (sigma + sigma_down) / 2.0
    d = (x - fn(x, sigma)) / sigma
    x_mid = x + d * (sigma_mid - sigma)
    d_mid = (x_mid - fn(x_mid, sigma_mid)) / sigma_mid
    x_next = x + d_mid * (sigma_down - sigma)
    x_next = x_next + noise * sigma_up
    return x_next


def main() -> None:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <epoch_2nd_00100.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    net = sd_all["net"] if "net" in sd_all else sd_all
    sd = net["diffusion"]
    prefix = "module.unet"
    sigma_data = 0.45731624995853165

    torch.manual_seed(123)
    np.random.seed(123)
    T = 6
    embedding = torch.randn(T, HP["context_embedding_features"])

    num_steps = 5
    sigma_min, sigma_max, rho = 1e-4, 3.0, 9.0
    sigmas = karras_schedule(num_steps, sigma_min, sigma_max, rho)

    noise0 = np.random.normal(size=HP["channels"])
    step_noises = np.random.normal(size=(num_steps - 1, HP["channels"]))

    fn = lambda x, sigma: kdiffusion_denoise(x, sigma, embedding, sd, prefix, sigma_data)

    x = sigmas[0] * noise0
    for i in range(num_steps - 1):
        x = adpm2_step(x, fn, sigmas[i], sigmas[i + 1], step_noises[i])

    embedding.numpy().astype(np.float32).tofile(out_dir / "sampler_embedding.bin")
    sigmas.astype(np.float32).tofile(out_dir / "sampler_sigmas.bin")
    noise0.astype(np.float32).tofile(out_dir / "sampler_noise0.bin")
    step_noises.astype(np.float32).tofile(out_dir / "sampler_step_noises.bin")
    x.astype(np.float32).tofile(out_dir / "sampler_expected_x_final.bin")
    print(f"T={T}, num_steps={num_steps}, sigmas={sigmas}, x_final[:5]={x[:5]}")


if __name__ == "__main__":
    main()
