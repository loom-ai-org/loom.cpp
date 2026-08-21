# Third-party dependencies, pulled via FetchContent so no submodules/vendored copies are needed.
include(FetchContent)

# ggml: the tensor library / graph runtime this engine is built on. The revision is pinned in its own
# file because an out-of-repo build -- an accelerator package shipping one libggml-<backend>.so -- has
# to build against exactly this one; see cmake/GgmlPin.cmake.
include(${CMAKE_CURRENT_LIST_DIR}/GgmlPin.cmake)
FetchContent_Declare(
    ggml
    GIT_REPOSITORY ${LOOM_GGML_REPOSITORY}
    GIT_TAG        ${LOOM_GGML_TAG}
)

# nlohmann/json: parses the JSON graph-topology definition embedded in GGUF metadata.
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")

# ggml defaults GGML_BUILD_TESTS/EXAMPLES to ON when built standalone; force them off since we only
# want the library here (keeps configure/build time down and avoids pulling ggml's own test deps).
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# tinyBLAS, ggml's blocked F32/F16 GEMM (`src/ggml-cpu/llamafile/sgemm.cpp`). A standalone ggml
# defaults it OFF -- it is llama.cpp that turns it on -- so this engine was shipping the generic
# one-output-element-per-call kernel. Measured at the eleven F32 GEMM shapes a VITS vocoder runs
# (BACKLOG.md P4.15), 4 threads:
#
#   x86-64, Ryzen 3 3250U (AVX2)      27.6 -> 54.4 GFLOP/s    1.97x   (median of 7, two cores, pinned)
#   aarch64, Cortex-A72               15.0 -> 15.6 GFLOP/s    1.04x -- nothing, until the two patches
#                                             -> 25.1          1.67x with them
#
# and it costs 111 KB of libggml-cpu (979 -> 1090 KB), which against the engine's size budget is the
# cheapest ratio in this repo. Two details in one line: the FORCE is what makes an EXISTING build tree
# pick this up -- ggml's own `option(GGML_LLAMAFILE ...)` would otherwise keep the OFF already in its
# cache and nothing would say so -- and the LOOM_ option is so that A/B-ing a GEMM change is a
# `-D` flag rather than an edit to this file.
option(LOOM_TINYBLAS "Build ggml's tinyBLAS (llamafile) GEMM path into the CPU backend" ON)
set(GGML_LLAMAFILE ${LOOM_TINYBLAS} CACHE BOOL "" FORCE)

# The pinned ggml, patched -- four diffs, each carrying its own measurement. Three are tinyBLAS: two
# aarch64-only fixes to GCC's code generation for its inner loop (a register tile GCC can actually
# allocate, and operand addresses it will strength-reduce -- together 15.6 -> 25.1 GFLOP/s, which takes
# ggml's own kernel PAST a hand-written one), and one architecture-neutral fix to which matmuls it
# accepts at all (it declined every `m % 4 != 0` matrix outright, handing thousands of rows back to the
# generic kernel over one or two leftovers). The fourth is ggml's fused convolution, which batched its
# im2col 16 MB at a time -- larger than any cache, so the patches it exists to keep local went to DRAM
# anyway -- and scattered its GEMM output one element at a time. See cmake/GgmlPatches.cmake for why a
# patch here rather than a fork, a vendored copy, or a change in this engine. Populated up front so that both branches below -- and any
# future one -- compile the patched sources, and re-checked on every configure so that an existing
# build tree cannot end up silently unpatched.
include(${CMAKE_CURRENT_LIST_DIR}/GgmlPatches.cmake)
FetchContent_GetProperties(ggml)
if(NOT ggml_POPULATED)
    FetchContent_Populate(ggml)
endif()
loom_patch_ggml(${ggml_SOURCE_DIR})

FetchContent_MakeAvailable(nlohmann_json)

# ggml's Vulkan backend needs two build-host tools, and a stable distribution's are likely too old for
# it -- in two ways that name neither cause. This makes `-DGGML_VULKAN=ON` work on such a machine by
# building what is missing, and does nothing at all on a machine that already has both (or on the
# CPU-only default). See cmake/VulkanToolchain.cmake for the two failures and the two probes.
#
# It runs BETWEEN populating ggml and adding it: it reads a feature-test shader out of ggml's sources,
# and everything it sets has to be in place before ggml's own `find_package(Vulkan COMPONENTS glslc)`.
if(GGML_VULKAN)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    include(${CMAKE_CURRENT_LIST_DIR}/VulkanToolchain.cmake)
    loom_setup_vulkan_toolchain()
endif()

# Added by hand rather than through FetchContent_MakeAvailable, which adds a dependency's subdirectory
# only on the configure that populates it -- so pre-populating ggml above (to patch it) would leave
# MakeAvailable a no-op and the ggml targets undefined.
add_subdirectory(${ggml_SOURCE_DIR} ${ggml_BINARY_DIR})

# LuaJIT: embedded Lua VM for the procedural-generalization orchestration layer (see
# LOOM_PROCEDURAL_GENERALIZATION.md / LOOM_MIL_CONVERSION.md) -- replaces bespoke per-model C++ drivers
# with a data-driven Lua script embedded in each model's GGUF. LuaJIT has no upstream CMakeLists (it's a
# Makefile-based build using its own "minilua"/DynASM bootstrapping), so unlike ggml/nlohmann_json above
# we FetchContent_Populate (source only, no add_subdirectory) and drive `make` ourselves via a custom
# command, then wrap the resulting static library as a normal IMPORTED target (`luajit::luajit`).
#
# Pinned to a COMMIT, not to `v2.1`. Unlike ggml's `v0.16.0` and nlohmann/json's `v3.11.3`, LuaJIT's
# "v2.1" is a rolling BRANCH that still receives commits, so fetching it made every configure pull
# whatever it pointed at that day -- reproducible builds locally (where the fetch is cached) and a
# moving target in CI, which fetches fresh each run. This commit is v2.1.ROLLING-363; bump it
# deliberately, and re-run the suite when you do.
FetchContent_Declare(
    luajit
    GIT_REPOSITORY https://github.com/LuaJIT/LuaJIT.git
    GIT_TAG        faaf663340347a78b22ed94c63c24fe090bd9784
)
FetchContent_GetProperties(luajit)
if(NOT luajit_POPULATED)
    FetchContent_Populate(luajit)
endif()

set(LUAJIT_SRC_DIR ${luajit_SOURCE_DIR}/src)
set(LUAJIT_LIBRARY ${LUAJIT_SRC_DIR}/libluajit.a)

include(ProcessorCount)
ProcessorCount(LUAJIT_NPROC)
if(LUAJIT_NPROC EQUAL 0)
    set(LUAJIT_NPROC 1)
endif()

# Plain `make` (not the GNU-make-only `$(MAKE)` recursive-invocation token) so this works regardless of
# the outer CMake generator (Ninja, etc.), at the cost of not sharing the outer build's job server --
# irrelevant here since LuaJIT itself builds in well under a minute.
# XCFLAGS=-fPIC: loom_engine is built as a shared library (libloom_engine.so), so every statically-linked
# .o inside it (including LuaJIT's own) must be position-independent -- without this, linking fails with
# "relocation R_X86_64_TPOFF32 ... can not be used when making a shared object".
add_custom_command(
    OUTPUT ${LUAJIT_LIBRARY}
    COMMAND make -C ${LUAJIT_SRC_DIR} BUILDMODE=static XCFLAGS=-fPIC -j${LUAJIT_NPROC}
    WORKING_DIRECTORY ${LUAJIT_SRC_DIR}
    COMMENT "Building LuaJIT (static, via its own Makefile)"
    VERBATIM
)
add_custom_target(luajit_build DEPENDS ${LUAJIT_LIBRARY})

add_library(luajit::luajit STATIC IMPORTED GLOBAL)
set_target_properties(luajit::luajit PROPERTIES
    IMPORTED_LOCATION ${LUAJIT_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${LUAJIT_SRC_DIR}
)
add_dependencies(luajit::luajit luajit_build)
# LuaJIT needs libm (math) and libdl (dynamic loading, its FFI/package.loadlib) on Linux.
target_link_libraries(luajit::luajit INTERFACE ${CMAKE_DL_LIBS} m)
