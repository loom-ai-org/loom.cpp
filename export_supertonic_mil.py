#!/usr/bin/env python3
"""Export the real SupertonicTTS v2 checkpoint (`assets/pt/*.pt`, full pickled `nn.Module`s -- `torch.save
(self, path)`, so `torch.load(..., weights_only=False)` returns an ALREADY-CONSTRUCTED real module
directly, no hyperparameter guessing/reconstruction needed, unlike Matcha's checkpoint format) through the
generic MIL exporter, tracing the REAL `supertonic_tts.models.modules.*` submodules directly -- not the
hand-built bespoke topology `tools/convert_supertonic/convert_supertonic_*.py` constructs op-by-op via its
own `TopologyBuilder` DSL (that bespoke conversion's own `supertonic_common.py` was invaluable here as an
independently-derived oracle for cross-checking every architectural quirk found while reading source, even
though this script doesn't reuse any of its op-building code).

Four topologies (same names/roles `loom::SupertonicDriver`/`supertonic_driver.lua` already established --
real `SpeechGenerator.predict()` never calls the two style encoders itself, always taking PRECOMPUTED style
embeddings, so those are out of scope here too):
  - dp:        real `DurationPredictor.forward(txt_ids, stl_emb, txt_msk)` -> scalar duration (seconds).
  - ttl_text:  real `TTLTextEncoder.forward(txt_ids, stl_emb, txt_msk)` -> txt_emb, ne=[T_TEXT,256].
  - vfe:       real `VectorFieldEstimator.compute_velocity(z_t, txt_emb, stl_emb, lat_msk, txt_msk, t)` --
               ONE Euler velocity evaluation (the `z += v*dt` update itself stays on the Lua/host side,
               matching supertonic_driver.lua's existing convention).
  - decoder:   real `SpeechDecoder.forward(latent)` -> flat waveform.

Text-length scope limitation (REAL, carried forward from the bespoke conversion, not a new one introduced
here, and NOT merely a `loom::GraphBuilder` restriction -- see below): `T_TEXT` is FIXED at trace/export
time for every topology that touches text (`T_TEXT_FIXED = 10`, matching `SupertonicConfig.txt_len_fixed`
exactly so this MIL export's own end-to-end driver test can compare directly against the EXISTING bespoke
`loom::SupertonicDriver` oracle). Two INDEPENDENT reasons force this, not one:
  (1) `vfe` needs TWO independently-sized sequences at once (the CFM-iterated latent-frame count `T_lat`,
      and the text length `T_TEXT`) -- `loom::GraphBuilder::build(n_tokens, n_past)` only ever resolves
      ONE dynamic-length symbol per topology, so `T_lat` gets "$n_tokens" and `T_TEXT` must be static.
  (2) `dp`/`ttl_text` independently CAN'T be traced with a dynamic `T_TEXT` at all, regardless of (1) --
      confirmed empirically: `MultiHeadRelativeAttention._get_relative_embeddings`'s relative-position-
      table windowing (`components.py`) pads by a length-DERIVED amount (`pad_len = max(length-(ws+1),
      0)`), which coremltools' own torch frontend explicitly refuses once `length` is genuinely dynamic
      (`NotImplementedError: Dynamic padding for n-dimensional tensors is not supported`) -- a real
      coremltools/MIL limitation, not a gap in this project's own exporter. This is the SAME underlying
      reason the bespoke conversion fixed `T_TEXT` in the first place (its own hand-built rel-pos-attention
      windowing needs a static T to build its lookup tables at all), not a coincidence.
  Net effect: ALL FOUR topologies (`dp`/`ttl_text`/`vfe`/`decoder`) are consistent in using this same fixed
  `T_TEXT_FIXED` wherever text length appears (`decoder` doesn't touch text at all, only `T_lat`).

Trace-friendliness patches needed (same category as every prior MIL export in this project):
  - `txt_msk`/`lat_msk` (always all-ones for this project's "single, unpadded utterance" convention, exactly
    like every other model) constructed via direct arithmetic on an already-real graph tensor (`txt_ids.
    unsqueeze(1).float()*0.0+1.0`, `z_t[:,:1,:]*0.0+1.0` -- NOT `torch.ones`/`torch.full`/`ones_like`),
    same "avoid a separate fill-shaped op" reasoning as every prior model's own mask construction. Unlike
    Matcha's Decoder (where every mask multiply was a provable no-op, so the mask was never even
    constructed), SupertonicTTS's masks are genuinely READ (softmax masking via `==0.0`/`masked_fill`,
    `.sum()` to recover fractional-RoPE sequence lengths) -- constructing a real all-ones tensor and letting
    those reads trace as real (structurally harmless, since the comparison is against a tensor of all 1s)
    EQUAL/SELECT/REDUCE_SUM ops is simpler and safer than trying to special-case every read site.
  - `nn.functional.pad(..., mode="replicate")` (every `ConvNextBlock` in this model pads this way before
    its depthwise conv) needed a NEW exporter capability, not a wrapper-level patch: see
    `tools/loom_mil_compiler/exporter.py`'s `pad` translation, `mode == "replicate"` branch -- ggml has no
    native replicate/edge-pad kernel (unlike PAD_1D/PAD_1D_REFLECT), so it's composed purely from
    already-existing primitives (VIEW the boundary column, REPEAT-broadcast it, CONCAT it on) rather than
    adding a new C++ op. Verified standalone (both symmetric and causal-only padding) via a small isolated
    trace/compare against real `F.pad` before ever touching the full model: 0.0 max abs diff.
  - Dilated depthwise conv (`ConvNextBlock`'s `big_convnext` groups, dilation up to 2**3=8) needed NO new
    engine work -- `CONV_1D_DW`'s existing `d0` (dilation) attribute already wraps `ggml_im2col`'s own
    dilation parameter correctly, confirmed by inspection (`src/ops/primitives_conv.cpp`), not newly added
    here.

Usage:
  ~/.venvs/piper/bin/python3 export_supertonic_mil.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
import loom_mil_compiler  # noqa: E402  registers the "loom" backend + torch-frontend patches
from loom_mil_compiler.exporter import LoomGGUFExporter  # noqa: E402

import coremltools as ct  # noqa: E402

SUPERTONIC_ROOT = Path("/home/flavio/Dev/supertonic-tts")
PT_DIR = SUPERTONIC_ROOT / "assets" / "pt"

T_TEXT_FIXED = 10  # see module docstring -- matches SupertonicConfig.txt_len_fixed exactly


def _ones_mask_from_ids(txt_ids):
    """(B,T) int -> (B,1,T) float, all-ones, derived via arithmetic on a real tensor (not torch.ones/
    ones_like/full) -- see module docstring."""
    return txt_ids.unsqueeze(1).to(torch.float32) * 0.0 + 1.0


def _ones_mask_from_float(x):
    """(B,C,T) float -> (B,1,T) float, all-ones, derived via arithmetic on `x` itself."""
    return x[:, :1, :] * 0.0 + 1.0


class DPWrapper(torch.nn.Module):
    """Real `DurationPredictor.forward` (assets/pt/duration_predictor.pt IS this class directly, not
    `DurationPredictorWrapper` -- confirmed via `tools/convert_supertonic/reference_forward_supertonic_dp.
    py`'s own usage, `dp(txt_ids, stl_emb, txt_msk)`). `stl_emb` (the DP style) is a precomputed input,
    matching `supertonic_driver.lua`'s own "dp" call (`stl_emb = inputs.style_dp`) -- `DPStyleEncoder` is
    out of scope here, same "basic synthesis from a precomputed style" precedent as every other model.
    """
    def __init__(self, dp):
        super().__init__()
        self.dp = dp

    def forward(self, txt_ids, stl_emb):
        txt_msk = _ones_mask_from_ids(txt_ids)
        duration = self.dp(txt_ids, stl_emb, txt_msk)  # (1,)
        return duration.reshape(-1)


class TTLTextWrapper(torch.nn.Module):
    """Real `TTLTextEncoder.forward` (assets/pt/text_encoder.pt). `stl_emb` (the TTL style, (1,50,256)) is
    precomputed, matching `supertonic_driver.lua`'s own "ttl_text" call -- `TTLStyleEncoder` out of scope.
    """
    def __init__(self, te):
        super().__init__()
        self.te = te

    def forward(self, txt_ids, stl_emb):
        txt_msk = _ones_mask_from_ids(txt_ids)
        txt_emb = self.te(txt_ids, stl_emb, txt_msk)  # (1, 256, T_TEXT_FIXED)
        return txt_emb.squeeze(0)  # (256, T_TEXT_FIXED) -> ggml ne=[T_TEXT_FIXED,256], T-fast


class VFEWrapper(torch.nn.Module):
    """Real `VectorFieldEstimator.compute_velocity` (assets/pt/vector_estimator.pt) -- ONE Euler velocity
    evaluation; the `z += v*dt` update itself is a Lua/host-side loop (`supertonic_driver_mil.lua`), same
    split as the bespoke `supertonic_driver.lua`. `txt_emb`'s own T axis is FIXED at trace time
    (T_TEXT_FIXED) -- see module docstring for why. `t`: (1,) float fractional step in [0,1).
    """
    def __init__(self, vfe):
        super().__init__()
        self.vfe = vfe

    def forward(self, z_t, txt_emb, stl_emb, t):
        lat_msk = _ones_mask_from_float(z_t)
        txt_msk = _ones_mask_from_float(txt_emb)
        v = self.vfe.compute_velocity(z_t, txt_emb, stl_emb, lat_msk, txt_msk, t)  # (1, 144, L)
        return v.squeeze(0)  # (144, L) -> ggml ne=[L,144], T-fast


class DecoderWrapper(torch.nn.Module):
    """Real `SpeechDecoder.forward` (assets/pt/vocoder.pt), unmodified -- no masking anywhere in this
    module at all (pure causal-conv stack + folded BatchNorm + PReLU head), so no trace-friendliness
    patch needed here."""
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, latent):
        wav = self.dec(latent)  # (1, T*6*512)
        return wav.reshape(-1)


def _build_topology(wrapper, dummy_args, mil_inputs, name):
    """Mirrors export_matcha_mil.py's own helper of the same name/reasoning: traces `wrapper`, converts to
    MIL, runs it through the exporter far enough to get back its topology JSON + weight dict WITHOUT
    writing a file (per-phase `func_name` namespacing avoids cross-phase weight-name collisions)."""
    traced = torch.jit.trace(wrapper, dummy_args)
    mil_prog = ct.convert(
        traced, inputs=mil_inputs, convert_to="milinternal", compute_precision=ct.precision.FLOAT32,
    )
    exporter = LoomGGUFExporter(mil_prog)
    main_func = mil_prog.functions["main"]
    topo = exporter.generate_graph_topology(main_func, name)
    print(f"  {name}: {len(topo['nodes'])} nodes, {len(exporter.weights)} weights")
    return topo, exporter.weights


def load_pt(name):
    mod = torch.load(PT_DIR / name, weights_only=False, map_location="cpu")
    mod.eval()
    return mod


def main():
    print(f"Loading SupertonicTTS checkpoints from {PT_DIR}...")
    dp = load_pt("duration_predictor.pt")
    te = load_pt("text_encoder.pt")
    vfe = load_pt("vector_estimator.pt")
    dec = load_pt("vocoder.pt")

    torch.manual_seed(0)
    dummy_txt_ids = torch.randint(1, 163, (1, T_TEXT_FIXED), dtype=torch.int64)
    dummy_dp_stl = torch.randn(1, 8, 16)
    dummy_ttl_stl = torch.randn(1, 50, 256)

    print("Tracing DurationPredictor...")
    dp_topo, dp_weights = _build_topology(
        DPWrapper(dp).eval(), (dummy_txt_ids, dummy_dp_stl),
        [ct.TensorType(name="txt_ids", shape=(1, T_TEXT_FIXED), dtype=np.int32),
         ct.TensorType(name="stl_emb", shape=(1, 8, 16), dtype=np.float32)], "dp",
    )

    print("Tracing TTLTextEncoder...")
    ttl_topo, ttl_weights = _build_topology(
        TTLTextWrapper(te).eval(), (dummy_txt_ids, dummy_ttl_stl),
        [ct.TensorType(name="txt_ids", shape=(1, T_TEXT_FIXED), dtype=np.int32),
         ct.TensorType(name="stl_emb", shape=(1, 50, 256), dtype=np.float32)], "ttl_text",
    )

    print("Tracing VectorFieldEstimator...")
    lat_seq_dim = ct.RangeDim(1, 512)
    dummy_L = 9
    dummy_z = torch.randn(1, 144, dummy_L)
    dummy_txt_emb = torch.randn(1, 256, T_TEXT_FIXED)
    dummy_t = torch.tensor([0.3])
    vfe_topo, vfe_weights = _build_topology(
        VFEWrapper(vfe).eval(), (dummy_z, dummy_txt_emb, dummy_ttl_stl, dummy_t),
        [ct.TensorType(name="z_t", shape=(1, 144, lat_seq_dim), dtype=np.float32),
         ct.TensorType(name="txt_emb", shape=(1, 256, T_TEXT_FIXED), dtype=np.float32),
         ct.TensorType(name="stl_emb", shape=(1, 50, 256), dtype=np.float32),
         ct.TensorType(name="t", shape=(1,), dtype=np.float32)], "vfe",
    )

    print("Tracing SpeechDecoder...")
    dec_seq_dim = ct.RangeDim(1, 512)
    dummy_latent = torch.randn(1, 144, 4)
    decoder_topo, decoder_weights = _build_topology(
        DecoderWrapper(dec).eval(), (dummy_latent,),
        [ct.TensorType(name="latent", shape=(1, 144, dec_seq_dim), dtype=np.float32)], "decoder",
    )

    merged_weights = {**dp_weights, **ttl_weights, **vfe_weights, **decoder_weights}
    assert len(merged_weights) == len(dp_weights) + len(ttl_weights) + len(vfe_weights) + len(decoder_weights)

    driver_script_path = Path(__file__).parent / "tools" / "convert_supertonic" / "supertonic_driver_mil.lua"

    out_exporter = LoomGGUFExporter(None, output_path="supertonic_mil.gguf", architecture="supertonic_mil")
    out_exporter.topologies = {"dp": dp_topo, "ttl_text": ttl_topo, "vfe": vfe_topo, "decoder": decoder_topo}
    out_exporter.weights = merged_weights
    out_exporter.write_gguf(driver_script_path.read_text())
    print("wrote supertonic_mil.gguf")


if __name__ == "__main__":
    main()
