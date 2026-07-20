"""Converts Kokoro-82M into a SINGLE self-contained loom-engine GGUF file (`kokoro.gguf`): the 43
topologies `loom::KokoroDriver::synthesize()` uses (CustomAlbert, bert_encoder, TextEncoder's own
CNN+BiLSTM, 3x DurationEncoder BiLSTM+AdaLayerNorm, the top BiLSTM, duration_proj, F0Ntrain's shared
BiLSTM + F0/N AdainResBlk1d stacks + projections, the Decoder core, SineGen, the forward STFT, and the
Generator) plus the embedded Lua orchestration script (`model.driver_script`) -- the same one-GGUF-
per-model convention already landed for Whisper/SupertonicTTS/Matcha-TTS/VITS (see BACKLOG.md's dated
entries). `kokoro_stft_inverse.gguf` is intentionally NOT included -- confirmed dead weight, loaded by
KokoroDriver's own constructor but never referenced anywhere in synthesize() (the sibling
StyleTTS2Driver has the identical dead-load pattern).

Reuses every build_* function from the existing per-module scripts (convert_kokoro_albert.py/
convert_kokoro_bert_encoder.py/convert_kokoro_text_encoder.py/convert_kokoro_duration_predictor.py/
convert_kokoro_f0n.py/convert_kokoro_decoder_core.py/convert_kokoro_sinegen.py/convert_kokoro_stft.py/
convert_kokoro_generator.py) UNCHANGED otherwise -- this script just calls them (with explicit distinct
weight namespaces for the six BiLSTM instances / three AdaLayerNorm instances / two 1x1-proj instances,
since Kokoro's per-module scripts historically wrote each such instance to its OWN isolated GGUF file
under identical generic weight names like "lstm.weight_ih", a real collision once merged into one file --
see BACKLOG.md's dated entry for the full story) and merges the resulting weight dicts. Those scripts'
own many-small-file output (via convert_kokoro_all.py) is untouched and still used by every existing
per-module/per-driver test.

Usage: python3 convert_kokoro_lua_all.py <kokoro-v1_0.pth> <out_dir>
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


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <kokoro-v1_0.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    merged_weights = {}
    topologies = {}

    def merge(name, weights):
        # Namespacing (see module docstring) should make every weight name globally unique already --
        # this content-aware check (dedup-on-match, hard-fail-on-mismatch) is the same defensive pattern
        # VITS's/Matcha's own convert_*_lua_all.py use, kept here as a real safety net rather than an
        # assumption.
        for k, v in weights.items():
            if k in merged_weights:
                assert np.array_equal(merged_weights[k], v), \
                    f"real weight name collision merging '{name}': '{k}' has DIFFERENT values across modules"
                continue
            merged_weights[k] = v

    def add(name, topo, weights):
        topologies[name] = topo
        merge(name, weights)

    # --- CustomAlbert ---
    albert_hp = convert_kokoro_albert.HP
    tb = convert_kokoro_albert.TopologyBuilder()
    out = convert_kokoro_albert.build_albert(tb, sd_all["bert"], albert_hp)
    albert_inputs = [
        {"name": "tokens", "dtype": "i32", "shape": ["$n_tokens"]},
        {"name": "positions", "dtype": "i32", "shape": ["$n_tokens"]},
        {"name": "attn_mask", "dtype": "f32", "shape": ["$n_tokens", "$n_tokens"]},
    ]
    add("albert", tb.topology(albert_inputs, out), tb.weights)

    # --- bert_encoder ---
    tb = convert_kokoro_bert_encoder.TopologyBuilder()
    out = convert_kokoro_bert_encoder.build_bert_encoder(tb, sd_all["bert_encoder"], "module")
    bert_encoder_inputs = [{"name": "x", "dtype": "f32", "shape": ["768", "$n_tokens"]}]
    add("bert_encoder", tb.topology(bert_encoder_inputs, out), tb.weights)

    # --- TextEncoder: CNN + its own BiLSTM (input_dim=channels=512, weight_namespace="text_encoder_lstm") ---
    te_sd = sd_all["text_encoder"]
    te_hp = convert_kokoro_text_encoder.HP
    topo, weights = convert_kokoro_text_encoder.build_cnn(te_sd, te_hp)
    add("text_encoder_cnn", topo, weights)
    for suffix, (topo, weights) in convert_kokoro_duration_predictor.build_bilstm(
            "module.lstm", te_sd, te_hp["hidden_per_dir"], te_hp["channels"],
            weight_namespace="text_encoder_lstm").items():
        add(f"text_encoder_lstm_{suffix}", topo, weights)

    # --- DurationEncoder: 3x (BiLSTM + AdaLayerNorm), predictor.lstm (top BiLSTM), duration_proj ---
    pred_sd = sd_all["predictor"]
    dp_hp = convert_kokoro_duration_predictor.HP
    duration_input_dim = dp_hp["d_model"] + dp_hp["style_dim"]
    for i, lstm_idx in enumerate((0, 2, 4)):
        for suffix, (topo, weights) in convert_kokoro_duration_predictor.build_bilstm(
                f"module.text_encoder.lstms.{lstm_idx}", pred_sd, dp_hp["hidden_per_dir"], duration_input_dim,
                weight_namespace=f"duration_lstm_{i}").items():
            add(f"duration_lstm_{i}_{suffix}", topo, weights)
    for i, lstm_idx in enumerate((1, 3, 5)):
        topo, weights = convert_kokoro_duration_predictor.build_adaln(pred_sd, lstm_idx, dp_hp,
                                                                        weight_prefix=f"duration_adaln_{i}")
        add(f"duration_adaln_{i}", topo, weights)
    for suffix, (topo, weights) in convert_kokoro_duration_predictor.build_bilstm(
            "module.lstm", pred_sd, dp_hp["hidden_per_dir"], duration_input_dim,
            weight_namespace="top_lstm").items():
        add(f"top_lstm_{suffix}", topo, weights)
    topo, weights = convert_kokoro_duration_predictor.build_duration_proj(pred_sd, dp_hp)
    add("duration_proj", topo, weights)

    # --- F0Ntrain: shared BiLSTM, F0/N AdainResBlk1d stacks, F0_proj/N_proj ---
    f0n_hp = convert_kokoro_f0n.HP
    for suffix, (topo, weights) in convert_kokoro_duration_predictor.build_bilstm(
            "module.shared", pred_sd, 256, 512 + f0n_hp["style_dim"], weight_namespace="f0n_shared_lstm").items():
        add(f"f0n_shared_lstm_{suffix}", topo, weights)
    block_dims = [(512, 512, False), (512, 256, True), (256, 256, False)]
    for i, (topo, weights) in convert_kokoro_f0n.build_stack(
            pred_sd, "f0n_f0", ["module.F0.0", "module.F0.1", "module.F0.2"], block_dims, f0n_hp).items():
        add(f"f0n_f0_block{i}", topo, weights)
    for i, (topo, weights) in convert_kokoro_f0n.build_stack(
            pred_sd, "f0n_n", ["module.N.0", "module.N.1", "module.N.2"], block_dims, f0n_hp).items():
        add(f"f0n_n_block{i}", topo, weights)
    topo, weights = convert_kokoro_f0n.build_proj1x1(pred_sd, "module.F0_proj", prefix="f0n_f0_proj")
    add("f0n_f0_proj", topo, weights)
    topo, weights = convert_kokoro_f0n.build_proj1x1(pred_sd, "module.N_proj", prefix="f0n_n_proj")
    add("f0n_n_proj", topo, weights)

    # --- Decoder core, SineGen, forward STFT, Generator (all real weights under sd_all["decoder"]) ---
    gen_sd = sd_all["decoder"]
    decoder_hp = convert_kokoro_decoder_core.HP
    topo, weights = convert_kokoro_decoder_core.build_decoder_core(decoder_hp, gen_sd, "module")
    add("decoder_core", topo, weights)

    sinegen_hp = convert_kokoro_sinegen.HP
    l_linear_w = gen_sd["module.generator.m_source.l_linear.weight"].detach().cpu().numpy().astype(np.float32)
    l_linear_b = gen_sd["module.generator.m_source.l_linear.bias"].detach().cpu().numpy().astype(np.float32)
    topo, weights = convert_kokoro_sinegen.build_sinegen(sinegen_hp, l_linear_w, l_linear_b)
    add("sinegen", topo, weights)

    generator_hp = convert_kokoro_generator.HP
    topo, weights = convert_kokoro_stft.build_forward(generator_hp["gen_istft_n_fft"], generator_hp["gen_istft_hop_size"])
    add("stft_forward", topo, weights)

    topo, weights = convert_kokoro_generator.build_generator(generator_hp, gen_sd, "module.generator")
    add("generator", topo, weights)

    driver_script_path = Path(__file__).parent / "kokoro_driver.lua"

    w = GGUFWriter(str(out_dir / "kokoro.gguf"), "loom-kokoro")
    for name, topo in topologies.items():
        w.add_string(f"model.graph_topology.{name}", json.dumps(topo))
    w.add_string("model.driver_script", driver_script_path.read_text())
    for name, arr in merged_weights.items():
        w.add_tensor(name, arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_dir / 'kokoro.gguf'}, {len(topologies)} topologies, {len(merged_weights)} weights")


if __name__ == "__main__":
    main()
