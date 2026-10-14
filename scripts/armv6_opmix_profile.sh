#!/bin/bash
# ADR-077 gate 1's cross-model analysis, step 2: a one-thread $LOOM_PROFILE of every Pi Zero model, node
# by node (LOOM_PROFILE_NODES=1, so each row keeps its shape), in the engine's default mode (RunFast on,
# ggml-0031). One launch per model; the report is parsed on the dev box by scripts/armv6_opmix.py.
# Same board layout as armv6_runfast_gate.sh (venv/, armv6_runfast_gate.py, models.txt); launch detached:
#   (setsid nohup ./armv6_opmix_profile.sh > prof.out 2>&1 < /dev/null &)
set -u
DIR="${DIR:-$HOME/rfgate}"
cd "$DIR" || exit 1
mkdir -p prof
sleep "${START:-120}"
while read -r kind gguf inp name; do
  LOOM_N_THREADS=1 LOOM_PROFILE="prof/$name.prof" LOOM_PROFILE_NODES=1 \
    venv/bin/python armv6_runfast_gate.py "$kind" "$gguf" "$inp" "prof/$name" >> prof/run.log 2>> prof/run.err
  sleep "${SETTLE:-20}"
done < "${MODELS:-models.txt}"
echo PROF_DONE >> prof/run.log
