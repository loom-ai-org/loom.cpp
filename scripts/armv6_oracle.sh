#!/bin/bash
# Correctness gate for an ARMv6 wheel, WITHOUT the board.
#
# The `loom-armv6-build:bookworm` container runs linux/arm/v6, so a wheel installed inside it executes
# the REAL __ARM_FEATURE_SIMD32 kernels under qemu-user. Calibrated against the board on P7.1's gather
# wheel -- both give sha256 f4859f55... for the standard VITS phrase -- so qemu's softfloat and the
# ARM1176's VFP agree bit for bit. This is strictly better than checking an x86 build, where every
# kernel under test is #if'd out. It is NOT faster than the board (82.6 s against 69.0 for a
# synthesis); the point is that it does not compete for it.
#
# WHAT NEEDS WHAT:
#   conformer-ctc and distilbert-ner emit TEXT, so they are checked right here by string equality --
#     no oracle, no second model, no wav.
#   VITS emits AUDIO, so it needs scripts/asr_oracle.py, which is whisper-small from `transformers`,
#     has no ARM in it, and runs NATIVELY and much faster. Do not grade it with loom's own conformer:
#     a bug the two share would be invisible (Retro-006).
#
#   LOOM_ARMV6_ORACLE_DIR=<dir with tts_check.py asr_bench.py ner_bench.py jfk3.wav> \
#     scripts/armv6_oracle.sh <wheel> [<name>]
#   ~/.venvs/piper/bin/python scripts/asr_oracle.py <dir>/<name>.wav \
#     --expect "hello world this is a raspberry pi zero"
#
# Exits non-zero when a transcript differs. Sabotage it before trusting it: change one of the expected
# strings below and confirm the exit code goes to 1 (ADR-015).
set -uo pipefail
W="${1:?usage: armv6_oracle.sh <wheel> [name]}"
NAME="${2:-out}"
D="${LOOM_ARMV6_ORACLE_DIR:?set it to a directory holding tts_check.py, asr_bench.py, ner_bench.py and jfk3.wav}"
MODELS="${LOOM_ARMV6_ORACLE_MODELS:-/home/flavio/Dev/loom/hf-models}"

WANT_ASR='and so are my fellow americans'
WANT_NER='Barack/B-PER Obama/I-PER visited/O Berlin/B-LOC in/O March/O and/O met/O Angela/B-PER Me/I-PER rk/I-PER el/I-PER at/O the/O Brandenburg/B-LOC Gate/I-LOC ./O'

cp "$W" "$D/loom_py_rt-1.0.0rc8-cp311-cp311-linux_armv6l.whl"
out=$(docker run --rm --platform linux/arm/v6 -v "$D:/w" -v "$MODELS:/m:ro" -w /w \
  loom-armv6-build:bookworm bash -c "
    set -e
    /opt/wheel/bin/pip install -q --no-index --force-reinstall /w/loom_py_rt-1.0.0rc8-cp311-cp311-linux_armv6l.whl
    /opt/wheel/bin/python -u tts_check.py /m/vits-q4_0.gguf /w/$NAME.wav 2>&1 | grep -E '^infer|^samples|^sha256'
    /opt/wheel/bin/python -u asr_bench.py /m/conformer-ctc-q4_0.gguf /w/jfk3.wav 2>&1 | grep -E '^TEXT'
    /opt/wheel/bin/python -u ner_bench.py /m/distilbert-ner-q4_0.gguf 2>&1 | grep -E '^TEXT'
  ") || { echo "$out"; echo "FAIL: the container exited non-zero"; exit 1; }
echo "$out"

rc=0
got_asr=$(printf '%s\n' "$out" | sed -n "s/^TEXT: '\(.*\)'$/\1/p" | head -1)
got_ner=$(printf '%s\n' "$out" | sed -n 's/^TEXT: \(Barack.*\)$/\1/p' | head -1)
[ "$got_asr" = "$WANT_ASR" ] || { echo "FAIL conformer-ctc: got [$got_asr]"; rc=1; }
[ "$got_ner" = "$WANT_NER" ] || { echo "FAIL distilbert-ner: got [$got_ner]"; rc=1; }
printf '%s\n' "$out" | grep -q '^samples' || { echo "FAIL: VITS produced no audio"; rc=1; }
peak=$(printf '%s\n' "$out" | sed -n 's/.*peak=\([0-9.]*\).*/\1/p' | head -1)
awk -v p="${peak:-0}" 'BEGIN{exit !(p > 0.05 && p < 0.99)}' || { echo "FAIL: VITS peak $peak out of range"; rc=1; }

[ $rc -eq 0 ] && echo "OK: transcripts match, audio in range. Now grade the wav natively:" \
  && echo "  ~/.venvs/piper/bin/python scripts/asr_oracle.py $D/$NAME.wav --expect \"hello world this is a raspberry pi zero\""
exit $rc
