#!/bin/bash
# ADR-078 gates 2-4 on the Pi Zero W: every ARMv6 model with and without RunFast (ggml-0031).
#
# Numerics and speed: one inference per launch (GGML_CPU_RUNFAST is read once at ggml init), both modes,
# two rounds in ABBA order, 20 s settle. Then the flag census (gate 3): each model once per mode under
# LOOM_PROFILE, which computes node by node, with scripts/armv6_fpflags.c preloaded. Then the tensor
# oracle: each ASR/NER model's raw output through the phase probe.
#
# NOT in qemu: qemu keeps the sign of a flushed zero, and it rounds a bounced vmla differently from the
# board in IEEE mode (Retro-086). Board layout, under $DIR (default ~/rfgate):
#   venv/            python3 -m venv --system-site-packages venv && venv/bin/pip install --no-deps <wheel>
#   armv6_runfast_gate.py
#   fpflags.so       gcc -O2 -fPIC -shared -I<ggml include> -o fpflags.so armv6_fpflags.c -ldl
#   probe            scripts/phase_probe.cpp, linked against the wheel's libloom_engine; the wheel is a
#                    GGML_BACKEND_DL build, so libggml-cpu.so must be symlinked beside it
#   models.txt       "<kind> <gguf> <input|-> <name>" per line, kinds as in armv6_runfast_gate.py
# Launch detached and do not poll (an ssh login starts the audio stack): setsid nohup ./this.sh &
# Compare on the dev box: scripts/armv6_runfast_compare.py <dir with gate.log, models.txt, out/>
#
# Reused for any on/off switch read once at init: VAR=<env name> (default GGML_CPU_RUNFAST; ADR-079 P4.32b
# used GGML_CPU_ARMV6_CONV). Arm 0 and arm 1 are that variable's values; the outputs keep the .rf0/.rf1
# names. FLAGS=0 skips the flag census, which is ADR-078's own question. PY0/PY1 give each arm its own
# interpreter (two venvs, two wheels: a wheel A/B, with VAR then unset by default); each arm's probe runs
# against that venv's libraries.
set -u
VAR="${VAR:-GGML_CPU_RUNFAST}"
DIR="${DIR:-$HOME/rfgate}"
cd "$DIR" || exit 1
mkdir -p out
sleep "${START:-120}"   # the launching ssh session is gone before the first arm
PY0="${PY0:-venv/bin/python}"
PY1="${PY1:-$PY0}"
PY=$PY0
libs() { local py; eval py=\$PY$1; echo "$(cd "$(dirname "$py")/.." && pwd)/lib/python3.11/site-packages/loom"; }
run() {  # mode kind gguf input name tag
  local py; eval py=\$PY$1
  env "$VAR=$1" $py armv6_runfast_gate.py $2 $3 $4 out/$5.rf$1.$6 >> gate.log 2>> gate.err
  sleep "${SETTLE:-20}"
}
for round in 1 2; do
  while read -r kind gguf inp name; do
    if [ $round = 1 ]; then a=0; b=1; else a=1; b=0; fi
    run $a "$kind" "$gguf" "$inp" "$name" r$round
    run $b "$kind" "$gguf" "$inp" "$name" r$round
  done < models.txt
done
echo GATE_DONE >> gate.log

[ "${FLAGS:-1}" = 1 ] && while read -r kind gguf inp name; do
  for m in 0 1; do
    echo "== $name runfast=$m" >> flags.log
    LOOM_PROFILE=/dev/null LD_PRELOAD="$DIR/fpflags.so" GGML_CPU_RUNFAST=$m \
      $PY armv6_runfast_gate.py "$kind" "$gguf" "$inp" out/$name.prof$m >> flags.log 2>&1
  done
done < models.txt

# The tensor oracle. Each body runs one topology alone and returns its first output.
CTC="loom.run_subgraph_and_retain('main_topology', {n_samples = #inputs.waveform, n_past = 0}, {waveform = inputs.waveform, length = {#inputs.waveform}}); local o = loom.get_output('main_topology', 1); return o"
ENC="loom.run_subgraph_and_retain('encoder', {n_samples = #inputs.waveform, n_past = 0}, {waveform = inputs.waveform}); local o = loom.get_output('encoder', 1); return o"
NER="loom.run_subgraph_and_retain('main_topology', {n_tokens = #inputs.tokens, n_past = 0}, {tokens = inputs.tokens, position_ids = loom.range(0, #inputs.tokens)}); local o = loom.get_output('main_topology', 1); return o"
while read -r kind gguf inp name; do
  case $kind in
    ctc_ids)    body=$CTC; arg="waveform=@$inp" ;;
    transcribe) body=$ENC; arg="waveform=@$inp" ;;
    s2t)        body=$CTC; $PY -c "
import array, sys, wave
w = wave.open(sys.argv[1], 'rb'); fr = w.readframes(w.getnframes())
open(sys.argv[2], 'wb').write(array.array('f', [int.from_bytes(fr[i:i + 2], 'little', signed=True) / 32768.0
                                                for i in range(0, len(fr), 2)]).tobytes())" "$inp" out/$name.in.f32
                arg="waveform=@out/$name.in.f32" ;;
    ner)        body=$NER; $PY -c "
import array, sys, loom
t = loom.Model.from_file(sys.argv[1]).tokenize('Barack Obama visited Berlin in March and met Angela Merkel at the Brandenburg Gate.')
open(sys.argv[2], 'wb').write(array.array('f', [float(x) for x in t]).tobytes())" "$gguf" out/$name.in.f32
                arg="tokens=@out/$name.in.f32" ;;
    *) continue ;;   # TTS: the waveform the gate run wrote is already the raw output
  esac
  for m in 0 1; do env "$VAR=$m" LD_LIBRARY_PATH="$(libs $m)" ./probe "$gguf" out/$name.t$m.f32 "$body" "$arg" >> probe.log 2>&1; done
done < models.txt
echo ALL_DONE >> gate.log
