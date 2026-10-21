# onnxruntime's side of the Pocket-TTS comparison: wall time for one synthesis through KevinAHM's
# community export (huggingface.co/KevinAHM/pocket-tts-onnx @ 58a6d00, bundle english_2026-04 --
# same 6-layer d1024 flow LM, LSD head and Mimi decoder as loom's english_2026-09 export) and its
# own `PocketTTSOnnx.generate`, which decodes on a second thread while the flow LM runs.
#
#   python bench_pocket_onnx.py <dir with pocket_tts_onnx.py + onnx/> <fp32|int8> <voice.safetensors>
#                               <frames> <nrun> "<text>"
#
# EQUAL WORK is pinned the way bench_vits_loom.cpp pins VITS's scales: the EOS head is disarmed
# (frames_after_eos huge), so the loop runs exactly <frames> frames, and bench_driver's loom arm gets
# max_frames=<frames> + eos_threshold=1e30. Both print their sample count; they must match.
#
# <voice.safetensors> is kyutai/pocket-tts's languages/english_2026-04/embeddings/alba.safetensors (a gated
# repo), the same 126-row voice the loom GGUF builds in. The loom arm, for the README's Pi 4 row:
#   bench_driver pocket-tts-f32.gguf cpu 3 tokens=2,1341,267,2888,1220,273,262,265,731,2130,295,778,1327,278,
#       376,2823,390,265,1746,906,824,269,934,278,284,265,427,1343,263 seed=num:1 max_frames=num:60 eos_threshold=num:1e30
# with the text "On a quiet morning, the old fisherman pushed his boat into the grey water and waited for the tide."
# here (the same 28 ids: the tokenizers are identical; loom's leading 2 is its chunk header). Needs onnxruntime,
# numpy, sentencepiece, safetensors and huggingface_hub (the export's module imports all four).
import statistics, sys, time
import numpy as np

root, prec, voice, frames, nrun, text = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
sys.path.insert(0, root)
from pocket_tts_onnx import PocketTTSOnnx  # noqa: E402

tts = PocketTTSOnnx(models_dir=f"{root}/onnx", language="english_2026-04", precision=prec,
                    temperature=0.3, lsd_steps=1)   # 0.3 = the checkpoint's English temperature, as loom
once = lambda: tts.generate(text, voice=voice, max_frames=frames, frames_after_eos=10**9)
np.random.seed(0)
n = len(once())                                     # warm: first run, voice state cached
ts = []
for _ in range(nrun):
    t0 = time.perf_counter(); a = once(); ts.append(time.perf_counter() - t0)
    assert len(a) == n
print(f"onnx   pocket [{prec}] samples={n}  mean {1e3 * statistics.mean(ts):.1f} ms  best {1e3 * min(ts):.1f} ms  (n={nrun})")
