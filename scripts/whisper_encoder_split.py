# The per-shape encoder split behind Epic-05's "Where the encoder gap is NOW" table: loom's
# $LOOM_PROFILE against onnxruntime's own per-node profile, bucket by bucket, for whisper-small.
# NOT part of the build. It exists because Retro-018's third lesson is that whichever side of a
# comparison has no script in the repository is the side that silently becomes unreproducible -- and
# the table it produces is what P4.18's remaining work is aimed by.
#
#   # one round of the onnxruntime arm (needs onnxruntime + transformers; see bench_onnx_tasks.py's
#   # header for which environment that is on each machine):
#   python3 scripts/whisper_encoder_split.py onnx <whisper-small-dir> <clip.wav> [threads] > r1.tsv
#
#   # the loom arm is the engine's own profiler, one thread, with node names:
#   LOOM_N_THREADS=1 LOOM_PROFILE=r1.txt LOOM_PROFILE_NODES=1 taskset -c 0 \
#       ./build/tools/loom_cli/loom_cli --model whisper_mil.gguf --wav <clip.wav>
#
#   # then pair N rounds of each (stdlib only):
#   python3 scripts/whisper_encoder_split.py pair r1.txt,r2.txt,... r1.tsv,r2.tsv,...
#
# RUN THE TWO ARMS INTERLEAVED, one round of each, and pair them. The dev box drifts 12% across five
# transcriptions and the drift is monotonic, so five loom runs followed by five onnxruntime runs
# attributes a thermal ramp to one engine. The `pair` output reports the per-round ratio's median and
# its lo/hi, which is what says whether a row is resolved (Retro-012).
#
# THREE THINGS THAT MAKE THE TWO SIDES COMPARABLE, each of which is easy to get wrong:
#
#  * THE GROUPINGS ARE NOT THE SAME SHAPE. loom's profiler buckets on the OUTPUT (`ne0, ne1`), so
#    `768 x 1500` holds Q/K/V/O, fc2 AND the 24 cross-KV projections; onnxruntime's events bucket on
#    the input, so `[1,1500,768]` holds Q/K/V/O and fc1. Keying onnxruntime on (input, OUTPUT) and
#    loom on the node index splits both into the same six matmul groups. Apportioning loom's bucket by
#    call count instead is wrong by 4x on fc2, and by FLOPs it assumes every shape runs at one rate.
#  * CROSS-KV IS NOT ENCODER WORK ON EITHER SIDE. loom computes it in its own topology (24 nodes,
#    ~1.09 s at one thread on the Ryzen); onnxruntime computes it on the decoder's first pass, so it is
#    not in `encoder_model.onnx` at all. It is reported separately, in neither column.
#  * THE MEL FRONTEND IS EXCLUDED for the same reason: onnxruntime's is `transformers` in numpy,
#    outside its timer, so loom's STFT nodes are not counted against it.
#
# And the onnxruntime numbers are SHARES apportioned over a separately measured un-profiled wall.
# Profiling costs it ~1.01x on whisper on x86 and ~1.18x on the Pi on VITS -- small either way, but the
# apportioning is what makes it not matter.
import ast
import json
import os
import statistics
import sys
import time
import wave
from collections import defaultdict

# loom node shapes that belong to the encoder or the cross-KV topology. Everything else in the profile
# is the mel frontend (3001x201, 3000x80) or the decoder (ne1 = 1).
ENC_NE = {'768,1500,1,1', '3072,1500,1,1', '1500,1500,12,1', '64,1500,12,1',
          '1500,64,12,1', '1500,1,768,1', '3000,1,768,1', '3000,768,1,1', '1500,768,1,1'}

# The cross-KV topology is 24 MUL_MAT / 24 ADD alternating at node_0..node_47, and the encoder's own
# nodes of these shapes start at node_51 (the positional-embedding ADD) -- its earlier nodes are the
# two convolutions, which carry different ne. Within the encoder the layer stride is 37 nodes and fc2
# sits at offset 31 of each, which is what separates it from the 48 projections it shares a shape with.
CROSS_KV_LAST = 47
ENC_LAYER_STRIDE = 37
ENC_LAYER_BASE = 55
FC2_OFFSET = 31

KEYS = ['proj', 'fc1', 'fc2', 'QK^T', 'A@V', 'softmax', 'layout', 'conv', 'norm+elemwise', 'gelu']


def loom_round(path):
    """Parse one $LOOM_PROFILE dump (written with $LOOM_PROFILE_NODES=1) into encoder buckets."""
    ms_by_key, calls_by_key, unclassified = defaultdict(float), defaultdict(int), []
    in_nodes = False
    for line in open(path):
        if line.split()[:4] == ['ms', 'calls', 'op', 'name']:
            in_nodes = True
            continue
        if not in_nodes:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            ms, calls, op = float(parts[0]), int(parts[1]), parts[2]
        except ValueError:
            continue
        ne, name = parts[-1], ' '.join(parts[3:-1])          # names contain spaces: "xv_0 (cont)"
        idx = int(name[5:]) if name.startswith('node_') and name[5:].isdigit() else -1
        key = None
        if ne in ENC_NE:
            if 0 <= idx <= CROSS_KV_LAST and ne == '768,1500,1,1':
                key = 'cross_kv'
            elif op == 'MUL_MAT' and ne == '768,1500,1,1':
                key = 'fc2' if (idx - ENC_LAYER_BASE) % ENC_LAYER_STRIDE == FC2_OFFSET else 'proj'
            elif op == 'MUL_MAT':
                key = {'3072,1500,1,1': 'fc1', '1500,1500,12,1': 'QK^T',
                       '64,1500,12,1': 'A@V'}.get(ne)
            elif op == 'SOFT_MAX':
                key = 'softmax'
            elif op == 'UNARY':
                key = 'gelu'
            elif op in ('NORM', 'ADD', 'MUL'):
                key = 'norm+elemwise'
            elif op == 'CONV_2D':
                key = 'conv'
            elif op == 'CONT':
                # 12 encoder permutes at one call each; the decoder's cross-attention V transpose runs
                # once per token and shares this shape (Epic-05, "the layout bucket was the decoder").
                key = 'layout' if calls <= 12 else 'decoder_xv'
            elif op in ('RESHAPE', 'PERMUTE', 'TRANSPOSE', 'VIEW', 'CPY'):
                key = 'layout'
        if key:
            ms_by_key[key] += ms
            calls_by_key[key] += calls
        elif ms > 1.0:
            unclassified.append((ms, calls, op, ne, name))
    return ms_by_key, calls_by_key, unclassified


def onnx_round(path):
    """Parse one TSV round written by the `onnx` subcommand below."""
    ms_by_key, calls_by_key, wall = defaultdict(float), defaultdict(int), None
    for line in open(path):
        field = line.rstrip('\n').split('\t')
        if field[0] == 'WALL':
            wall = float(field[1])
            continue
        _, op, in0, _in1, out0, calls, ms = field
        in0, out0 = ast.literal_eval(in0), ast.literal_eval(out0)
        if op == 'MatMul':
            key = ('proj' if in0 == [1, 1500, 768] and out0 == [1, 1500, 768] else
                   'fc1' if out0 == [1, 1500, 3072] else
                   'fc2' if in0 == [1, 1500, 3072] else
                   'QK^T' if out0 == [1, 12, 1500, 1500] else 'A@V')
        elif op in ('BiasGelu', 'Gelu'):
            key = 'gelu'
        elif op in ('LayerNormalization', 'SkipLayerNormalization', 'Add', 'Mul'):
            key = 'norm+elemwise'
        elif op in ('Transpose', 'Reshape'):
            key = 'layout'
        else:
            key = {'Softmax': 'softmax', 'Conv': 'conv'}.get(op, 'other')
        ms_by_key[key] += float(ms)
        calls_by_key[key] += int(calls)
    return ms_by_key, calls_by_key, wall


def pair(loom_paths, onnx_paths):
    rounds_l = [loom_round(p) for p in loom_paths]
    rounds_o = [onnx_round(p) for p in onnx_paths]
    print(f"{len(rounds_l)} rounds\n")
    print(f"{'bucket':<15}{'calls l/o':>12}{'loom ms':>10}{'onnx ms':>10}{'ratio':>8}{'lo':>7}{'hi':>7}")
    loom_total = onnx_total = 0.0
    rows = []
    for key in KEYS:
        loom_ms = [r[0][key] for r in rounds_l]
        onnx_ms = [r[0][key] for r in rounds_o]
        ratios = sorted(a / b for a, b in zip(loom_ms, onnx_ms) if b)
        l_med, o_med = statistics.median(loom_ms), statistics.median(onnx_ms)
        loom_total += l_med
        onnx_total += o_med
        rows.append((key, l_med - o_med))
        print(f"{key:<15}{f'{rounds_l[0][1][key]}/{rounds_o[0][1][key]}':>12}"
              f"{l_med:>10.1f}{o_med:>10.1f}"
              f"{statistics.median(ratios):>8.2f}{ratios[0]:>7.2f}{ratios[-1]:>7.2f}")
    gap = loom_total - onnx_total
    print(f"{'ENCODER TOTAL':<15}{'':>12}{loom_total:>10.1f}{onnx_total:>10.1f}"
          f"{loom_total / onnx_total:>8.2f}")
    print(f"\ngap {gap:.0f} ms, by share:")
    for key, delta in sorted(rows, key=lambda kv: -kv[1]):
        print(f"  {key:<15}{delta:>8.0f} ms{delta / gap * 100:>7.1f}%")
    print(f"\nonnx encoder wall (median of rounds) {statistics.median([r[2] for r in rounds_o]):.1f} ms")
    print(f"loom cross_kv, separate topology, in NEITHER column "
          f"{statistics.median([r[0]['cross_kv'] for r in rounds_l]):.1f} ms")
    print(f"loom decoder cross-attention V CONT   "
          f"{statistics.median([r[0]['decoder_xv'] for r in rounds_l]):.1f} ms")
    # A bucket that quietly stopped matching would otherwise look like a win, so show what was skipped.
    print("\nloom nodes NOT classified as encoder (>1 ms, first round) -- these should all be decoder:")
    for ms, calls, op, ne, name in sorted(rounds_l[0][2], reverse=True)[:8]:
        print(f"  {ms:>8.1f} x{calls:<5}{op:<12}{ne:<16}{name[:44]}")


def onnx(directory, wav, threads):
    """One round of the onnxruntime arm: an un-profiled wall and a profiled node table, as TSV."""
    import numpy as np
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor

    with wave.open(wav) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    feats = WhisperFeatureExtractor.from_pretrained(directory)(
        pcm, sampling_rate=16000, return_tensors="np")["input_features"].astype(np.float32)

    def session(profile):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.enable_profiling = profile
        if profile:
            so.profile_file_prefix = os.path.join(os.environ.get("TMPDIR", "."), "ortprof")
        return ort.InferenceSession(f"{directory}/onnx/encoder_model.onnx", so,
                                    providers=["CPUExecutionProvider"])

    feed = {"input_features": feats}
    plain, profiled = session(False), session(True)
    plain.run(None, feed)                                    # warm both, like bench_onnx_tasks.py
    profiled.run(None, feed)
    wall = min(_time(plain, feed) for _ in range(2))
    profiled.run(None, feed)
    events = json.load(open(profiled.end_profiling()))
    kernels = [e for e in events if e.get("cat") == "Node" and e["name"].endswith("_kernel_time")]
    warm = kernels[len(kernels) // 2:]                       # the second, warm profiled run
    total = sum(e["dur"] for e in warm)

    def shapes(event, which):
        return [v for t in event["args"].get(which, []) for _, v in t.items()]

    agg = {}
    for e in warm:
        i, o = shapes(e, "input_type_shape"), shapes(e, "output_type_shape")
        key = (e["args"]["op_name"], tuple(i[0]) if i else (),
               tuple(i[1]) if len(i) > 1 else (), tuple(o[0]) if o else ())
        entry = agg.setdefault(key, [0, 0])
        entry[0] += e["dur"]
        entry[1] += 1
    print(f"WALL\t{wall * 1e3:.1f}\t{total / 1e3:.1f}")
    for (op, i0, i1, o0), (us, calls) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"OP\t{op}\t{list(i0)}\t{list(i1)}\t{list(o0)}\t{calls}\t{us / total * wall * 1e3:.2f}")


def _time(sess, feed):
    t0 = time.perf_counter()
    sess.run(None, feed)
    return time.perf_counter() - t0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__ or "see header", file=sys.stderr)
        sys.exit(2)
    if sys.argv[1] == "onnx":
        onnx(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    elif sys.argv[1] == "pair":
        pair(sys.argv[2].split(','), sys.argv[3].split(','))
    else:
        print(f"unknown subcommand {sys.argv[1]!r}", file=sys.stderr)
        sys.exit(2)
