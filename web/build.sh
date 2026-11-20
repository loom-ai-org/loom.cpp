#!/usr/bin/env bash
# Builds loom for the browser into web/dist/ (ADR-139): `loom-cpu.{mjs,wasm}`, `loom-webgpu.{mjs,wasm}` and
# the JS API beside them.
#
#   source ~/emsdk/emsdk_env.sh
#   web/build.sh              # both flavours
#   web/build.sh cpu          # one
#
# Extra CMake arguments go in $LOOM_WEB_CMAKE_ARGS (e.g. -DFETCHCONTENT_SOURCE_DIR_GGML=... to reuse a
# checkout).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
command -v emcmake >/dev/null || { echo "web/build.sh: no emcmake on PATH; source emsdk_env.sh first" >&2; exit 1; }

# Every object agrees on these, or the link fails (wasm exceptions) or a longjmp crosses C++ frames
# wrongly (Lua's error path, which the bridge relies on). -msimd128 turns on ggml's WASM SIMD kernels.
flags="-fwasm-exceptions -sSUPPORT_LONGJMP=wasm -msimd128"
jobs=${LOOM_WEB_JOBS:-$(nproc)}
mkdir -p "$here/dist"
for flavour in "${@:-cpu webgpu}"; do
  for f in $flavour; do
    extra=()
    if [ "$f" = webgpu ]; then extra=(-DGGML_WEBGPU=ON -DGGML_WEBGPU_JSPI=ON); fi
    build="$root/build-web-$f"
    # shellcheck disable=SC2086
    emcmake cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release \
      -DLOOM_BUILD_TESTS=OFF -DLOOM_BUILD_TOOLS=OFF -DLOOM_TINYBLAS=OFF -DGGML_NATIVE=OFF \
      -DLOOM_WEB_FLAVOUR="$f" -DCMAKE_C_FLAGS="$flags" -DCMAKE_CXX_FLAGS="$flags" \
      "${extra[@]}" ${LOOM_WEB_CMAKE_ARGS:-}
    cmake --build "$build" --target loom_web -j"$jobs"
    cp "$build/web/loom-$f.mjs" "$build/web/loom-$f.wasm" "$here/dist/"
  done
done
cp "$here/js/loom.mjs" "$here/dist/"
ls -la "$here/dist"
