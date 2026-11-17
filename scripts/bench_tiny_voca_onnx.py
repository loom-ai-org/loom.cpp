# onnxruntime's side of the tiny-voca comparison (ADR-133): wall time for one utterance through
# upstream's own export, n4ze3m/tiny-voca's `onnx/model.onnx` (float32, opset 17), or an int8 file
# made from it with `quantize_dynamic`.
#
#   python bench_tiny_voca_onnx.py <model.onnx> <nrun> <ids,...> <stress,...> [threads]
#
# One utterance, as upstream's graph takes one: dizione's ids and stress for "On a quiet morning, the old
# fisherman pushed his boat into the grey water and waited for the tide." (82 tokens, voice F2, speed 1,
# no pitch changes). The loom arm, for the README's rows, takes the same ids with the stress folded in
# (id + 76 * stress, the GGUF's phoneme table):
#   bench_driver tiny-voca.gguf cpu 3 tokens=<id + 76 * stress, ...> seed=num:0
#
# EQUAL WORK: both print their sample count, which is the sum of the durations times 256, and they must
# match. The noise differs (each runtime draws its own), which costs nothing either side: one draw of
# white noise per sample.
import statistics, sys, time
import numpy as np
import onnxruntime as ort

path, nrun = sys.argv[1], int(sys.argv[2])
ids = [int(x) for x in sys.argv[3].split(",")]
stress = [int(x) for x in sys.argv[4].split(",")]
opts = ort.SessionOptions()
if len(sys.argv) > 5:
    opts.intra_op_num_threads = int(sys.argv[5])
session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
one = lambda v: np.array([v], dtype=np.float32)
feed = {"tokens": np.array([ids], dtype=np.int64), "stress": np.array([stress], dtype=np.int64),
        "voice": np.array([0], dtype=np.int64), "speed": one(1.0), "pitch_shift": one(0.0), "pitch_range": one(1.0)}
once = lambda: session.run(None, feed)[0]
n = once().shape[-1]                                # warm
ts = []
for _ in range(nrun):
    t0 = time.perf_counter(); a = once(); ts.append(time.perf_counter() - t0)
    assert a.shape[-1] == n
print(f"onnx   tiny-voca samples={n}  mean {1e3 * statistics.mean(ts):.1f} ms  best {1e3 * min(ts):.1f} ms  (n={nrun})")
