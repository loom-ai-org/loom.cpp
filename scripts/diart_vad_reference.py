"""Ground truth for pyannote segmentation streamed as diart's voice activity detection streams it
(`test_e2e_diart_vad_stream.cpp`, ADR-129): diart's `VoiceActivityDetection` (github.com/juanmc2005/diart), run
here, not re-derived, the way its `Benchmark.run_single` runs a file -- `FileAudioSource` with
`config.get_file_padding`, the source's blocks one step long, `rearrange_audio_stream`'s scan cutting them into
`duration`-long chunks every `step`, one chunk per pipeline call (batch 1), each call's `Annotation` into a
`PredictionAccumulator`.

diart wires those pieces with RxPY, which the export venv does not have; the pieces themselves are plain
functions. So `rx` is stubbed with a recorder: `rx.pipe(ops.scan(f, seed), ops.filter(p), ...)` hands back the
operators it was given, and this script applies diart's own `accumulate`, filters and map in order. Nothing of
diart's arithmetic is restated here. `websocket_server` (a sink this never uses) is stubbed the same way, and
diart's `AudioLoader` (torchaudio's file reader) is replaced by one that hands back the clip's float32 samples.

The model is pyannote's own `Model.from_pretrained(checkpoint)` in diart's `PowersetAdapter` (the hard
powerset -> multilabel diart applies to segmentation-3.0), CPU, float32.

Per clip and setting, in `<out>/<clip>.<duration>-<step>-<latency>/`:
    meta.txt      -- `padding <left s> <right s>`, `shift <s>`, `chunks <n>`, `frames <per chunk>`
    steps.txt     -- per pipeline call: `step <i> <chunk start s> <region start s> <region end s> <rows>`
                     (the region and rows are what the call's aggregation returned, before the shift)
    scores.txt    -- the aggregated speech scores of every call, joined, one per line (float64, exact)
    windows.npy   -- each chunk's speech per frame (max over the multilabel speakers), (chunks, frames)
    segments.txt  -- the accumulated prediction after diart's 0.05 s patch: `<start> <end>` per speech turn

    python scripts/diart_vad_reference.py <diart src dir> <pyannote checkpoint dir> <out> \\
        [--duration 5] [--step 0.5] [--latency 0.5[,...]] [--tau 0.6] <clip.npy> [...]
"""
import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch


def _stub_modules():
    """`rx` as a recorder of the operators diart composes, and `websocket_server` as a name."""

    class _Anything(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return lambda *a, **k: ("unused", name)

    rx = _Anything("rx")
    ops = _Anything("rx.operators")
    core = types.ModuleType("rx.core")
    subject = types.ModuleType("rx.subject")
    rx.pipe = lambda *operators: operators
    ops.scan = lambda fn, seed: ("scan", fn, seed)
    ops.filter = lambda pred: ("filter", pred)
    ops.map = lambda fn: ("map", fn)

    class Observer:
        def __init__(self, *a, **k):
            pass

    class Observable:
        pass

    class Subject:
        def __init__(self, *a, **k):
            pass

    core.Observer, core.Observable, subject.Subject = Observer, Observable, Subject
    rx.operators, rx.core, rx.subject = ops, core, subject
    ws = types.ModuleType("websocket_server")
    ws.WebsocketServer = object
    sys.modules.update({"rx": rx, "rx.operators": ops, "rx.core": core, "rx.subject": subject,
                        "websocket_server": ws})


class _NpyLoader:
    """diart's `AudioLoader` interface over a float32 `.npy` clip already at the pipeline's rate."""

    def __init__(self, sample_rate, mono=True):
        self.sample_rate = sample_rate

    def load(self, filepath):
        return torch.from_numpy(np.load(filepath).astype(np.float32).reshape(1, -1))

    def get_duration(self, filepath):
        return np.load(filepath, mmap_mode="r").shape[-1] / self.sample_rate


class _Collect:
    def __init__(self):
        self.blocks = []

    def on_next(self, value):
        self.blocks.append(value)

    def on_error(self, error):
        raise error

    def on_completed(self):
        pass


def run(diart, model, clip, duration, step, latency, tau, out):
    from diart import operators as dops
    from diart import sources
    from diart.blocks import base
    from diart.blocks.vad import VoiceActivityDetection, VoiceActivityDetectionConfig
    from diart.models import SegmentationModel
    from diart.sinks import PredictionAccumulator

    base.AudioLoader = _NpyLoader
    sources.AudioLoader = _NpyLoader
    sample_rate = 16000
    config = VoiceActivityDetectionConfig(segmentation=SegmentationModel(lambda: model), duration=duration,
                                          step=step, latency=latency, tau_active=tau,
                                          device=torch.device("cpu"), sample_rate=sample_rate)
    pipeline = VoiceActivityDetection(config)

    # Benchmark.run_single, minus the progress bar and the rx plumbing.
    padding = config.get_file_padding(clip)
    source = sources.FileAudioSource(clip, sample_rate, padding, config.step)
    source.stream = _Collect()
    pipeline.set_timestamp_shift(-padding[0])
    source.read()
    operators = dops.rearrange_audio_stream(config.duration, config.step, source.sample_rate)

    # What each call's aggregation returned (the rows before Binarize), and each chunk's segmentation.
    calls, windows = [], []
    aggregate = pipeline.pred_aggregation

    class _Recorded:
        num_overlapping_windows = aggregate.num_overlapping_windows

        def __call__(self, buffers):
            result = aggregate(buffers)
            calls.append((buffers[-1].sliding_window.start, result))
            return result

    pipeline.pred_aggregation = _Recorded()
    segment = pipeline.segmentation

    def _recorded_segmentation(batch):
        result = segment(batch)
        windows.append(torch.max(result, dim=-1)[0][0].numpy().copy())
        return result

    pipeline.segmentation = _recorded_segmentation
    accumulator = PredictionAccumulator(source.uri)
    state = None
    for block in source.stream.blocks:
        value = block
        emitted = True
        for op in operators:
            if op[0] == "scan":
                state = op[1](op[2] if state is None else state, value)
                value = state
            elif op[0] == "filter":
                if not op[1](value):
                    emitted = False
                    break
            elif op[0] == "map":
                value = op[1](value)
            else:
                raise RuntimeError(f"diart composed an operator this script does not apply: {op}")
        if not emitted:
            continue
        for prediction in pipeline([value]):
            accumulator.on_next(prediction)
    accumulator.on_completed()

    out.mkdir(parents=True, exist_ok=True)
    frames = windows[0].shape[0] if windows else 0
    with open(out / "meta.txt", "w") as f:
        f.write(f"padding {padding[0]!r} {padding[1]!r}\nshift {-padding[0]!r}\nchunks {len(calls)}\n"
                f"frames {frames}\n")
    rows = []
    with open(out / "steps.txt", "w") as f:
        for i, (chunk_start, result) in enumerate(calls):
            sw = result.sliding_window
            n = result.data.shape[0]
            f.write(f"step {i} {float(chunk_start)!r} {float(sw.start)!r} {float(sw.start + n * sw.step)!r} {n}\n")
            rows.append(result.data[:, 0])
    with open(out / "scores.txt", "w") as f:
        for v in (np.concatenate(rows) if rows else []):
            f.write(f"{float(v)!r}\n")
    np.save(out / "windows.npy", np.stack(windows) if windows else np.zeros((0, 0), dtype=np.float32))
    prediction = accumulator.get_prediction()
    with open(out / "segments.txt", "w") as f:
        if prediction is not None:
            for seg in prediction.get_timeline().support():
                f.write(f"{float(seg.start)!r} {float(seg.end)!r}\n")
    return len(calls)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("diart", type=Path, help="diart's `src` directory")
    ap.add_argument("checkpoint", type=Path, help="pyannote-segmentation-3.0 directory")
    ap.add_argument("out", type=Path)
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--latency", default="0.5", help="comma-separated latencies, each in [step, duration]")
    ap.add_argument("--tau", type=float, default=0.6)
    args = ap.parse_args()

    _stub_modules()
    sys.path.insert(0, str(args.diart))
    from pyannote.audio import Model
    from diart.models import PowersetAdapter

    torch.set_grad_enabled(False)
    model = PowersetAdapter(Model.from_pretrained(str(args.checkpoint / "pytorch_model.bin"))).eval()
    for clip in args.clips:
        for latency in (float(x) for x in args.latency.split(",")):
            name = f"{clip.stem}.{args.duration:g}-{args.step:g}-{latency:g}"
            n = run(args.diart, model, str(clip), args.duration, args.step, latency, args.tau, args.out / name)
            print(f"{name}: {n} calls")


if __name__ == "__main__":
    main()
