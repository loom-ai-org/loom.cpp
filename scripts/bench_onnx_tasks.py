# onnxruntime's side of the README's three-task comparison, driven DIRECTLY -- one subcommand per task.
# NOT part of the build; a standalone measurement, kept because "how does loom compare on this machine"
# has to stay re-runnable on the next machine.
#
#   python3 scripts/bench_onnx_tasks.py vits <miro_en-GB.onnx>        [threads] [nrun]
#   python3 scripts/bench_onnx_tasks.py lm   <qwen3/onnx/model.onnx>  [threads] [n_new] [nrun]
#   python3 scripts/bench_onnx_tasks.py asr  <whisper-small-dir> <wav> [threads] [nrun]
#   python3 scripts/bench_onnx_tasks.py moonshine <moonshine-tiny-dir> <wav> [threads] [nrun]
#   python3 scripts/bench_onnx_tasks.py ctc  <citrinet-dir> <clip-stem> [threads] [nrun]
#
# `$LOOM_ONNX_SUFFIX` picks a variant of the files inside an export directory: `_int8` reads
# onnx-community's `encoder_model_int8.onnx` / `decoder_model_merged_int8.onnx`. The int8 arm uses
# those PUBLISHED files rather than `quantize_dynamic` on the merged decoder, because quantize_dynamic
# leaves a merged decoder's `If` subgraphs in F32 (the output is the F32 size, 615 MB for whisper-small);
# onnx-community quantizes before merging. VITS and Citrinet have no published int8, so theirs is
# `quantize_dynamic(weight_type=QInt8)` with every other knob at its default.
#
# `ctc` reads NeMo's own ONNX export (`model.onnx`, features in, log-probs out) plus the clip's
# features as `<stem>.feats.npy` / `<stem>.len.npy` and `vocab.json`, all written by NeMo on the dev
# box. Like the whisper task, it leaves the front end outside onnxruntime's timer, where loom's
# `bench_asr_loom` times its in-graph STFT front end too -- a small bias in onnxruntime's favour.
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
# `bench_onnx.py` is the older, Pi-specific TTS harness. This file is the portable three-task one, and
# `prof_onnx_conv_shapes.py` is the per-shape convolution profile P4.16's table is built from.
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
# ONE TRAP WHEN DRIVING THIS FROM A SHELL WRAPPER: build the loom binaries somewhere durable, and make
# the wrapper FAIL when a measurement comes back empty. Putting them under /tmp cost a whole run here --
# systemd-tmpfiles reaped them mid-sweep, and the wrapper cheerfully printed rows with a blank loom
# column and a syntax error where the ratio should have been, exit code 0. A missing arm has to be
# louder than a slow one.
#
# Pin the version you quote. `pip install onnxruntime` 1.28.0 is the README's baseline; conda-forge's
# build of the SAME version is 1.86x faster on VITS, which is large enough to change every conclusion
# ([Retro-012](../docs/retros/retro-012-optimizations-that-were-measured-out.md)).
import json, os, statistics, sys, time, wave
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
    # ONE WARM-UP, DISCARDED, because the other arm has one. This was missing until 2026-09-02 and it
    # is the same fault Retro-018 is about, in the one task that had not been audited for it: the `lm`
    # and `asr` tasks below both warm up, and `bench_vits_loom.cpp:100` discards its first synthesis
    # because that call is what builds the graphs. Comparing a warmed loom against a cold-first
    # onnxruntime is not one measurement of two engines. It biases the median by 1/nrun of the
    # cold/warm gap, i.e. in loom's favour, which is the direction that flatters this repository.
    sess.run(None, feed)
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
    sfx = os.environ.get("LOOM_ONNX_SUFFIX", "")
    enc = ort.InferenceSession(f"{directory}/onnx/encoder_model{sfx}.onnx", so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(f"{directory}/onnx/decoder_model_merged{sfx}.onnx", so, providers=["CPUExecutionProvider"])
    dec_out = [o.name for o in dec.get_outputs()]
    # onnx-community's int8 merged decoder predates `cache_position` (positions come from the past length).
    has_cache_position = "cache_position" in {i.name for i in dec.get_inputs()}
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
            if has_cache_position:
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


def read_pcm(wav):
    with wave.open(wav) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def bench_moonshine(directory, wav, threads, nrun):
    # Moonshine takes the raw waveform, so unlike whisper there is no front end left outside the timer.
    from transformers import PreTrainedTokenizerFast
    pcm = read_pcm(wav)
    tok = PreTrainedTokenizerFast(tokenizer_file=f"{directory}/tokenizer.json")
    gen = json.load(open(f"{directory}/generation_config.json"))
    bos, eos = gen["decoder_start_token_id"], gen["eos_token_id"]
    so = opts(threads)
    sfx = os.environ.get("LOOM_ONNX_SUFFIX", "")
    enc = ort.InferenceSession(f"{directory}/onnx/encoder_model{sfx}.onnx", so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(f"{directory}/onnx/decoder_model_merged{sfx}.onnx", so, providers=["CPUExecutionProvider"])
    dec_out = [o.name for o in dec.get_outputs()]
    past_names = list(empty_past(dec))
    # transformers' Moonshine caps generation at 6.5 tokens per second of audio; the same cap here.
    max_new = int(np.ceil(len(pcm) / 16000 * 6.5))

    def run():
        h = enc.run(None, {"input_values": pcm[None, :]})[0]
        past, ids, toks, first = empty_past(dec), np.array([[bos]], dtype=np.int64), [], True
        for _ in range(max_new):
            feed = dict(past)
            feed["input_ids"] = ids
            feed["encoder_hidden_states"] = h
            feed["use_cache_branch"] = np.array([not first], dtype=bool)
            out = dec.run(None, feed)
            nxt = int(np.argmax(out[dec_out.index("logits")][0, -1]))
            past = {p: (out[dec_out.index(p.replace("past_key_values", "present"))]
                        if (first or ".decoder." in p) else past[p]) for p in past_names}
            first = False
            if nxt == eos:
                break
            toks.append(nxt)
            ids = np.array([[nxt]], dtype=np.int64)
        return tok.decode(toks, skip_special_tokens=True)

    text = run()                             # warm up
    ts = []
    for _ in range(nrun):
        t0 = time.perf_counter(); text = run(); ts.append(time.perf_counter() - t0)
    ts.sort()
    print(f"onnx   moonshine audio={len(pcm) / 16000:.2f}s  median {ts[len(ts) // 2]:.4f} s  "
          f"min {ts[0]:.4f} s  (n={nrun}, intra_op={threads})")
    print(f"  text: {text}")


def bench_ctc(directory, stem, threads, nrun):
    feats = np.load(f"{directory}/{stem}.feats.npy")
    flen = np.load(f"{directory}/{stem}.len.npy")
    vocab = json.load(open(f"{directory}/vocab.json"))
    sfx = os.environ.get("LOOM_ONNX_SUFFIX", "")
    sess = ort.InferenceSession(f"{directory}/model{sfx}.onnx", opts(threads), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    feed = {names[0]: feats, names[1]: flen}

    def run():
        logp = sess.run(None, feed)[0][0]
        ids, prev, keep = logp.argmax(-1), None, []
        for t in ids:                        # greedy CTC: collapse repeats, drop the blank (last id)
            if t != prev and t != len(vocab):
                keep.append(int(t))
            prev = t
        # Citrinet-256-ls is WordPiece (`##` continues a word; NeMo emits [CLS]/[SEP] as ordinary ids),
        # Citrinet-1024 SentencePiece (`▁` starts one): join either way, drop the two markers.
        pieces = [vocab[i] for i in keep if vocab[i] not in ("[CLS]", "[SEP]")]
        if any(p.startswith("##") for p in pieces):
            return "".join(p[2:] if p.startswith("##") else " " + p for p in pieces).strip()
        return "".join(pieces).replace("▁", " ").strip()

    text = run()                             # warm up
    ts = []
    for _ in range(nrun):
        t0 = time.perf_counter(); text = run(); ts.append(time.perf_counter() - t0)
    ts.sort()
    print(f"onnx   ctc   frames={feats.shape[-1]}  median {ts[len(ts) // 2]:.4f} s  "
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
    elif task in ("moonshine", "ctc"):
        (bench_moonshine if task == "moonshine" else bench_ctc)(
            sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 4,
            int(sys.argv[5]) if len(sys.argv) > 5 else 5)
    else:
        print(f"unknown task {task!r}", file=sys.stderr); sys.exit(2)
