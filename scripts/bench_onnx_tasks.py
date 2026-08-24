# onnxruntime's side of the README's three-task comparison, driven DIRECTLY -- one subcommand per task.
# NOT part of the build; a standalone measurement, kept because "how does loom compare on this machine"
# has to stay re-runnable on the next machine.
#
#   python3 scripts/bench_onnx_tasks.py vits <miro_en-GB.onnx>        [threads] [nrun]
#   python3 scripts/bench_onnx_tasks.py lm   <qwen3/onnx/model.onnx>  [threads] [n_new] [nrun]
#   python3 scripts/bench_onnx_tasks.py asr  <whisper-small-dir> <wav> [threads] [nrun]
#
# Its loom counterparts are scripts/bench_{vits,lm,asr}_loom.cpp, and each pair is checked for EQUAL
# WORK rather than merely equal wall time: `vits` prints the sample count (both engines must say
# 73216), `asr` prints the transcript, and `lm` is equal by construction -- same prompt, same token
# budget, greedy on both sides.
#
# THREE THINGS THIS DOES NOT DO, each of which cost a previous measurement its meaning:
#  * it does not use `optimum`, which added roughly 2x of its own overhead to whisper and mis-derives
#    Qwen3's head_dim;
#  * it does not time model load -- every timer starts with the session already built, which is what
#    the loom harnesses time too;
#  * it does not leave VITS's duration predictor free. `scales` is pinned to [0, 1, 0] so both engines
#    synthesise ONE identical utterance; unpinned, they time two different ones (Retro-010).
#
# `bench_onnx.py` is the older, Pi-specific TTS harness that also carries the per-shape convolution
# profile P4.16's table is built from. This file is the portable three-task one.
#
# WHERE THE PIECES LIVE, because the assets are large and are not in any repo (2026-08-24):
#
#   dev box       onnx models  ~/Dev/loom/onnx/{qwen3-0.6b,whisper-small}, VITS at
#                              ~/Dev/piper/pipertts_en-GB_miro/miro_en-GB.onnx
#                 python       ~/.venvs/ovos/bin/python  (onnxruntime 1.28.0)
#   workstation   onnx models  ~/loom/onnx/{qwen3-0.6b,whisper-small}, VITS in ~/loom/fixtures/
#                 python       ~/micromamba/envs/onnxbench/bin/python3
#   rpi4          onnx models  ~/bench/onnx/{qwen3-0.6b,whisper-small,miro_en-GB.onnx}
#                 python       ~/test/bin/python3
#
# The whisper directory must be the FULL export -- `onnx/` plus the tokenizer and preprocessor JSON
# next to it -- because the ASR task builds features and detokenises with `transformers`.
#
# Pin the version you quote. `pip install onnxruntime` 1.28.0 is the README's baseline; conda-forge's
# build of the SAME version is 1.86x faster on VITS, which is large enough to change every conclusion
# ([Retro-012](../docs/retros/retro-012-optimizations-that-were-measured-out.md)).
import statistics, sys, time, wave
import numpy as np
import onnxruntime as ort

# The same sentence bench_vits_loom.cpp speaks, as piper phoneme ids from miro_en-GB.onnx.json.
VITS_IDS = [1,20,0,120,0,18,0,74,0,8,0,3,0,23,0,39,0,26,0,3,0,22,0,33,0,122,0,3,0,96,0,120,0,102,0,32,
            0,17,0,14,0,100,0,26,0,3,0,41,0,59,0,3,0,23,0,59,0,25,0,28,0,22,0,120,0,33,0,122,0,32,0,
            50,0,8,0,3,0,25,0,14,0,74,0,3,0,19,0,88,0,120,0,61,0,26,0,17,0,13,0,2]
LM_PROMPT = [785, 6722, 315, 9625, 374]     # "The capital of France is", Qwen3 BPE


def opts(threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return so


def empty_past(sess, prefix="past_key_values"):
    proto = {i.name: i for i in sess.get_inputs()}
    out = {}
    for n in proto:
        if not n.startswith(prefix):
            continue
        s = proto[n].shape                   # [batch, kv_heads, seq, head_dim], seq symbolic
        out[n] = np.zeros((1, int(s[1]), 0, int(s[3])), dtype=np.float32)
    return out


def bench_vits(model, threads, nrun):
    sess = ort.InferenceSession(model, opts(threads), providers=["CPUExecutionProvider"])
    x = np.array([VITS_IDS], dtype=np.int64)
    feed = {"input": x,
            "input_lengths": np.array([x.shape[1]], dtype=np.int64),
            "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32)}   # noise, length, noise_w -- PINNED
    ts, n = [], 0
    for _ in range(nrun):
        t0 = time.perf_counter()
        out = sess.run(None, feed)
        ts.append(time.perf_counter() - t0)
        n = int(np.asarray(out[0]).size)
    ts.sort()
    print(f"onnx   vits  samples={n}  median {statistics.median(ts):.4f} s  min {ts[0]:.4f} s  "
          f"(n={nrun}, intra_op={threads})")


def bench_lm(model, threads, n_new, nrun):
    sess = ort.InferenceSession(model, opts(threads), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    past_names = list(empty_past(sess))

    def generate():
        past, ids, n_past, toks = empty_past(sess), np.array([LM_PROMPT], dtype=np.int64), 0, []
        for step in range(n_new + 1):
            L = ids.shape[1]
            feed = dict(past)
            feed["input_ids"] = ids
            feed["attention_mask"] = np.ones((1, n_past + L), dtype=np.int64)
            feed["position_ids"] = np.arange(n_past, n_past + L, dtype=np.int64)[None, :]
            out = sess.run(None, feed)
            nxt = int(np.argmax(out[names.index("logits")][0, -1]))
            past = {p: out[names.index(p.replace("past_key_values", "present"))] for p in past_names}
            n_past += L
            if step < n_new:
                toks.append(nxt)
                ids = np.array([[nxt]], dtype=np.int64)
        return toks

    generate()                               # warm up: the first pass pays allocator and page-in costs
    ts = []
    for _ in range(nrun):
        t0 = time.perf_counter(); toks = generate(); ts.append(time.perf_counter() - t0)
    best = min(ts)
    print(f"onnx   lm    {best:.3f} s  ({n_new} tokens, {n_new / best:.2f} tok/s)  "
          f"intra_op={threads}  first5={toks[:5]}")


def bench_asr(directory, wav, threads, nrun):
    from transformers import WhisperFeatureExtractor, WhisperTokenizerFast
    with wave.open(wav) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    fe = WhisperFeatureExtractor.from_pretrained(directory)
    tok = WhisperTokenizerFast.from_pretrained(directory)
    feats = fe(pcm, sampling_rate=16000, return_tensors="np")["input_features"].astype(np.float32)

    so = opts(threads)
    enc = ort.InferenceSession(f"{directory}/onnx/encoder_model.onnx", so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(f"{directory}/onnx/decoder_model_merged.onnx", so, providers=["CPUExecutionProvider"])
    dec_out = [o.name for o in dec.get_outputs()]
    past_names = list(empty_past(dec))
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    start = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"])

    def run():
        h = enc.run(None, {"input_features": feats})[0]
        past, ids, pos, toks, first = empty_past(dec), np.array([start], dtype=np.int64), 0, [], True
        for _ in range(200):
            feed = dict(past)
            feed["input_ids"] = ids
            feed["encoder_hidden_states"] = h
            feed["cache_position"] = np.arange(pos, pos + ids.shape[1], dtype=np.int64)
            feed["use_cache_branch"] = np.array([not first], dtype=bool)
            out = dec.run(None, feed)
            nxt = int(np.argmax(out[dec_out.index("logits")][0, -1]))
            # The cross-attention (encoder) K/V are computed on the first pass and carried unchanged --
            # which is exactly the thing loom's whisper export used to recompute every step (P4.15/#15).
            past = {p: (out[dec_out.index(p.replace("past_key_values", "present"))]
                        if (first or ".decoder." in p) else past[p]) for p in past_names}
            pos += ids.shape[1]
            first = False
            if nxt == eot:
                break
            toks.append(nxt)
            ids = np.array([[nxt]], dtype=np.int64)
        return tok.decode(toks, skip_special_tokens=True)

    text = run()                             # warm up
    ts = []
    for _ in range(nrun):
        t0 = time.perf_counter(); text = run(); ts.append(time.perf_counter() - t0)
    ts.sort()
    print(f"onnx   asr   audio={len(pcm) / 16000:.2f}s  median {ts[len(ts) // 2]:.4f} s  "
          f"min {ts[0]:.4f} s  (n={nrun}, intra_op={threads})")
    print(f"  text: {text}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__ or "see header", file=sys.stderr); sys.exit(2)
    task = sys.argv[1]
    if task == "vits":
        bench_vits(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4,
                   int(sys.argv[4]) if len(sys.argv) > 4 else 9)
    elif task == "lm":
        bench_lm(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4,
                 int(sys.argv[4]) if len(sys.argv) > 4 else 24,
                 int(sys.argv[5]) if len(sys.argv) > 5 else 3)
    elif task == "asr":
        bench_asr(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 4,
                  int(sys.argv[5]) if len(sys.argv) > 5 else 3)
    else:
        print(f"unknown task {task!r}", file=sys.stderr); sys.exit(2)
