# Third-party dependencies, pulled via FetchContent so no submodules/vendored copies are needed.
include(FetchContent)

# ggml: the tensor library / graph runtime this engine is built on.
# Pinned to v0.16.0 (commit 524f974bb21a1013408f76d71c15732482c0c3fe) for reproducible builds.
FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggml-org/ggml.git
    GIT_TAG        v0.16.0
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

# ggml's Vulkan backend needs two build-host tools, and a stable distribution's are likely too old for
# it -- in two ways that name neither cause. This makes `-DGGML_VULKAN=ON` work on such a machine by
# building what is missing, and does nothing at all on a machine that already has both (or on the
# CPU-only default). See cmake/VulkanToolchain.cmake for the two failures and the two probes.
#
# Populated-then-added rather than MakeAvailable'd, because the toolchain setup has to run BETWEEN the
# two: it reads a feature-test shader out of ggml's sources, and everything it sets has to be in place
# before ggml's own `find_package(Vulkan COMPONENTS glslc)` runs.
if(GGML_VULKAN)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    include(${CMAKE_CURRENT_LIST_DIR}/VulkanToolchain.cmake)
    FetchContent_MakeAvailable(nlohmann_json)
    FetchContent_GetProperties(ggml)
    if(NOT ggml_POPULATED)
        FetchContent_Populate(ggml)
    endif()
    loom_setup_vulkan_toolchain()
    add_subdirectory(${ggml_SOURCE_DIR} ${ggml_BINARY_DIR})
else()
    FetchContent_MakeAvailable(ggml nlohmann_json)
endif()

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
