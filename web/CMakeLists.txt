# loom for the browser (ADR-139): `loom_web` links the engine and its embind binding into one ES module
# (`loom-<flavour>.mjs` + `.wasm`), which `js/loom.mjs` wraps. Built only under emscripten; `web/build.sh`
# configures both flavours:
#
#   webgpu  GGML_WEBGPU=ON, JSPI: every call that runs a graph returns a Promise while the GPU works.
#           Needs a browser with WebGPU and JSPI (Chrome/Edge 137+), or Node with --experimental-wasm-jspi
#           (and no GPU).
#   cpu     ggml's CPU backend alone, WASM SIMD, no JSPI: runs in any browser and in Node.
#
# One thread in both: a pthread build needs SharedArrayBuffer, so a cross-origin-isolated page (COOP/COEP
# headers), which a page cannot assume. `default_cpu_thread_count` returns 1 here.
if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "web/ builds under emscripten only (emcmake cmake ...)")
endif()

set(LOOM_WEB_FLAVOUR "cpu" CACHE STRING "Which web build: cpu or webgpu")

add_executable(loom_web src/loom_web.cpp)
target_link_libraries(loom_web PRIVATE loom_engine)
set_target_properties(loom_web PROPERTIES OUTPUT_NAME "loom-${LOOM_WEB_FLAVOUR}" SUFFIX ".mjs")

target_link_options(loom_web PRIVATE
    -lembind
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=createLoomModule
    -sENVIRONMENT=web,worker,node
    -sALLOW_MEMORY_GROWTH=1
    # wasm32's ceiling. A model's weights live in this heap on the CPU build; on the WebGPU build they
    # pass through it on the way to the GPU.
    -sMAXIMUM_MEMORY=4GB
    -sINITIAL_MEMORY=128MB
    # The bridge's Lua and the engine's recursion (the graph builder walks a topology) need more than
    # emscripten's 64 KB default.
    -sSTACK_SIZE=8MB
    -sFORCE_FILESYSTEM=1
    -sEXPORTED_RUNTIME_METHODS=FS
    -fwasm-exceptions
    -sSUPPORT_LONGJMP=wasm
)

if(LOOM_WEB_FLAVOUR STREQUAL "webgpu")
    if(NOT GGML_WEBGPU OR NOT GGML_WEBGPU_JSPI)
        message(FATAL_ERROR "LOOM_WEB_FLAVOUR=webgpu needs -DGGML_WEBGPU=ON -DGGML_WEBGPU_JSPI=ON")
    endif()
    target_compile_definitions(loom_web PRIVATE LOOM_WEB_JSPI=1)
    target_link_options(loom_web PRIVATE -sJSPI)
elseif(NOT LOOM_WEB_FLAVOUR STREQUAL "cpu")
    message(FATAL_ERROR "LOOM_WEB_FLAVOUR=${LOOM_WEB_FLAVOUR}: cpu or webgpu")
endif()
