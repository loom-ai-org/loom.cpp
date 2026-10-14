# ADR-078 gates 2 and 4 on the Pi Zero: one model, one inference, in whatever mode GGML_CPU_RUNFAST
# says (read by ggml at init, so one process per arm). Prints the time and a fingerprint of the output,
# and writes the raw output for the comparison on the dev box.
#   armv6_runfast_gate.py <kind> <gguf> <input|-> <out prefix>      (driven by armv6_runfast_gate.sh)
import array, hashlib, os, sys, time, wave
import loom

kind, gguf, inp, out = sys.argv[1:5]
mode = os.environ.get("GGML_CPU_RUNFAST", "default")

def f32(path):
    a = array.array("f"); a.frombytes(open(path, "rb").read()); return a.tolist()

def wav16(path):
    w = wave.open(path, "rb"); fr = w.readframes(w.getnframes())
    return [int.from_bytes(fr[i:i + 2], "little", signed=True) / 32768.0 for i in range(0, len(fr), 2)]

m = loom.Model.from_file(gguf)
text, floats = None, None
t = time.time()
if kind == "ctc_ids":            # Citrinet: the driver returns CTC ids
    a = f32(inp); floats = m.infer(waveform=a, length=[float(len(a))])
elif kind == "transcribe":       # Moonshine
    text = m.transcribe(f32(inp)).text
elif kind == "s2t":              # conformer-ctc (P7 set)
    text = m.speech2text.infer(wav16(inp)).text
elif kind == "tts_ids":          # sanoTTS / piperlite: phoneme ids in, waveform out
    floats = m.infer(tokens=[float(i) for i in f32(inp)])
elif kind == "vits":             # VITS (P7 set): tts_check.py's phrase and seed
    floats = m.text2speech.infer(phonemes="həlˈəʊ wˈɜːld, ðɪs ɪz ə ɹˈaspbəɹi pˈaɪ zˈiəɹəʊ", seed=1234,
                                 sample_rate=22050).samples
elif kind == "ner":
    text = m.text2class.infer("Barack Obama visited Berlin in March and met Angela Merkel at the Brandenburg Gate.").text
dt = time.time() - t
h = hashlib.sha256()
if floats is not None:
    raw = array.array("f", floats).tobytes(); h.update(raw); open(out + ".f32", "wb").write(raw)
if text is not None:
    h.update(text.encode()); open(out + ".txt", "w").write(text)
print(f"RESULT kind={kind} model={os.path.basename(gguf)} runfast={mode} infer={dt:.3f}s "
      f"n={len(floats) if floats is not None else len(text)} sha={h.hexdigest()[:16]} "
      f"text={text if text is not None else ''!r}", flush=True)
