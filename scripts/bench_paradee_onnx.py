# onnxruntime's side of the Paradee comparison (ADR-135): wall time for one utterance through upstream's
# own export, sahilmahendrakar/Paradee-8M-v1.3's `onnx/paradee.onnx` (float32) or `onnx/paradee_int8.onnx`
# (its own int8 file, the one its README says to use).
#
#   python bench_paradee_onnx.py <model.onnx> <nrun> <ids,...> [threads] [voice]
#
# One utterance, as upstream's graph takes one: misaki's phonemes for "On a quiet morning, the old fisherman
# pushed his boat into the grey water and waited for the tide." as Kokoro-vocabulary ids wrapped in 0s
# (101 ids), voice 0 (af_heart), speed 1. The loom arm takes the same ids:
#   bench_driver paradee.gguf cpu 3 tokens=<ids> seed=num:0
#
# EQUAL WORK: both print their sample count, the sum of the durations times 600, and they must match. The
# noise differs (each runtime draws its own), which costs nothing either side: one draw per sample.
import statistics, sys, time
import numpy as np
import onnxruntime as ort

path, nrun = sys.argv[1], int(sys.argv[2])
ids = [int(x) for x in sys.argv[3].split(",")]
opts = ort.SessionOptions()
if len(sys.argv) > 4:
    opts.intra_op_num_threads = int(sys.argv[4])
voice = int(sys.argv[5]) if len(sys.argv) > 5 else 0
session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
feed = {"input_ids": np.array([ids], dtype=np.int64), "speed": np.array([1.0], dtype=np.float32)}
if any(i.name == "voice" for i in session.get_inputs()):
    feed["voice"] = np.array([voice], dtype=np.int64)
once = lambda: session.run(None, feed)[0]
n = once().shape[-1]                                # warm
ts = []
for _ in range(nrun):
    t0 = time.perf_counter(); a = once(); ts.append(time.perf_counter() - t0)
    assert a.shape[-1] == n
print(f"onnx   paradee samples={n}  mean {1e3 * statistics.mean(ts):.1f} ms  best {1e3 * min(ts):.1f} ms  (n={nrun})")
