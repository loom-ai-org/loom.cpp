# LuaJIT under MSVC, built by CMake rather than by LuaJIT's own `src/msvcbuild.bat`.
#
# Included from Dependencies.cmake with LUAJIT_SRC_DIR set; defines `luajit::luajit` exactly as the
# Makefile path does, so nothing downstream knows which one ran.
#
# WHY NOT msvcbuild.bat, which is what LuaJIT documents for Windows. Two reasons, either sufficient:
#
#   * It needs a developer prompt. Its first line is `if not defined INCLUDE goto :FAIL`, and it calls
#     bare `cl`, `link` and `lib`. Under the Ninja generator from a vcvars shell those are on PATH; under
#     the Visual Studio generator -- which is what scikit-build-core and cibuildwheel use on Windows
#     unless told otherwise -- they are not, because MSBuild hands the compiler its paths as task
#     parameters rather than through the environment a custom command inherits.
#   * Its `static` mode compiles with no CRT flag, i.e. `/MT`, the STATIC runtime. Everything else in
#     this build is `/MD` (CMake's default), and the static library ends up inside loom_engine.dll, so
#     the DLL would carry two C runtimes -- two heaps, two sets of stdio handles -- and the linker says
#     so only as warning LNK4098.
#
# So this file is msvcbuild.bat's x64 path transcribed into targets: the same host tools, the same
# DynASM flags, the same buildvm outputs, compiled by whatever compiler and runtime CMake chose for the
# rest of the tree. The generated files land in LUAJIT_SRC_DIR, where the .bat (and the Makefile)
# put them, because buildvm and the library sources include them by bare name.
#
# x64 ONLY. The .bat also has an x86 path (vm_x86.dasc, /arch:SSE2) and an ARM64 one (vm_arm64.dasc,
# which needs a cross-built host toolchain), and nobody has asked for either.

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8 OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|x86_64)$")
    message(FATAL_ERROR
        "LuaJIT under MSVC is wired up for x64 only (target: ${CMAKE_SYSTEM_PROCESSOR}, "
        "${CMAKE_SIZEOF_VOID_P}-byte pointers). See cmake/LuaJitMsvc.cmake.")
endif()

set(LUAJIT_MSVC_DEFS _CRT_SECURE_NO_DEPRECATE)

# The two host tools get a directory of their own: a consumer that collects every DLL from
# CMAKE_RUNTIME_OUTPUT_DIRECTORY (loom-py does, to ship them in the wheel) would otherwise find two
# build-time executables beside them.
set(LUAJIT_HOST_DIR ${CMAKE_CURRENT_BINARY_DIR}/luajit-host)

# 1. minilua: a single-file Lua 5.1 that runs DynASM and genversion.lua at build time.
add_executable(luajit_minilua EXCLUDE_FROM_ALL ${LUAJIT_SRC_DIR}/host/minilua.c)
target_compile_definitions(luajit_minilua PRIVATE ${LUAJIT_MSVC_DEFS})
set_target_properties(luajit_minilua PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${LUAJIT_HOST_DIR})

# 2. luajit.h is GENERATED from luajit_rolling.h, with the rolling release number taken from the
# commit time -- the .bat runs `git show -s --format=%ct` into luajit_relver.txt. Done once here, at
# configure time, because the pin is a fixed commit and so is its answer. `.relver` is the fallback
# the .bat uses for a source tarball.
find_package(Git QUIET)
set(LUAJIT_RELVER "")
if(GIT_FOUND AND EXISTS ${LUAJIT_SRC_DIR}/../.git)
    execute_process(COMMAND ${GIT_EXECUTABLE} show -s --format=%ct
        WORKING_DIRECTORY ${LUAJIT_SRC_DIR}
        OUTPUT_VARIABLE LUAJIT_RELVER OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(LUAJIT_RELVER STREQUAL "" AND EXISTS ${LUAJIT_SRC_DIR}/../.relver)
    file(READ ${LUAJIT_SRC_DIR}/../.relver LUAJIT_RELVER)
endif()
file(WRITE ${LUAJIT_SRC_DIR}/luajit_relver.txt "${LUAJIT_RELVER}\n")

add_custom_command(
    OUTPUT ${LUAJIT_SRC_DIR}/luajit.h
    COMMAND luajit_minilua host/genversion.lua
    DEPENDS luajit_minilua ${LUAJIT_SRC_DIR}/luajit_rolling.h ${LUAJIT_SRC_DIR}/luajit_relver.txt
    WORKING_DIRECTORY ${LUAJIT_SRC_DIR}
    COMMENT "LuaJIT: generating luajit.h"
    VERBATIM
)

# 3. DynASM turns the x64 interpreter (vm_x64.dasc) into the encoder buildvm links. These are the
# .bat's x64 DASMFLAGS verbatim: GC64 is the default on x64, which is what `-D P64` selects.
add_custom_command(
    OUTPUT ${LUAJIT_SRC_DIR}/host/buildvm_arch.h
    COMMAND luajit_minilua ../dynasm/dynasm.lua -LN -D WIN -D JIT -D FFI -D ENDIAN_LE -D FPU -D P64
            -o host/buildvm_arch.h vm_x64.dasc
    DEPENDS luajit_minilua ${LUAJIT_SRC_DIR}/vm_x64.dasc
    WORKING_DIRECTORY ${LUAJIT_SRC_DIR}
    COMMENT "LuaJIT: DynASM vm_x64.dasc"
    VERBATIM
)

# 4. buildvm, and everything it emits: the interpreter itself as a COFF object (`peobj`, the Windows
# counterpart of the Makefile's `elfasm`), and the bytecode/fast-function/library/recorder/fold tables
# the library sources include.
file(GLOB LUAJIT_BUILDVM_SOURCES ${LUAJIT_SRC_DIR}/host/buildvm*.c)
add_executable(luajit_buildvm EXCLUDE_FROM_ALL
    ${LUAJIT_BUILDVM_SOURCES}
    ${LUAJIT_SRC_DIR}/host/buildvm_arch.h
    ${LUAJIT_SRC_DIR}/luajit.h
)
target_include_directories(luajit_buildvm PRIVATE ${LUAJIT_SRC_DIR} ${LUAJIT_SRC_DIR}/../dynasm)
target_compile_definitions(luajit_buildvm PRIVATE ${LUAJIT_MSVC_DEFS})
set_target_properties(luajit_buildvm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${LUAJIT_HOST_DIR})

set(LUAJIT_ALL_LIB
    lib_base.c lib_math.c lib_bit.c lib_string.c lib_table.c lib_io.c lib_os.c lib_package.c
    lib_debug.c lib_jit.c lib_ffi.c lib_buffer.c)
set(LUAJIT_GENERATED
    ${LUAJIT_SRC_DIR}/lj_vm.obj
    ${LUAJIT_SRC_DIR}/lj_bcdef.h
    ${LUAJIT_SRC_DIR}/lj_ffdef.h
    ${LUAJIT_SRC_DIR}/lj_libdef.h
    ${LUAJIT_SRC_DIR}/lj_recdef.h
    ${LUAJIT_SRC_DIR}/jit/vmdef.lua
    ${LUAJIT_SRC_DIR}/lj_folddef.h)
add_custom_command(
    OUTPUT ${LUAJIT_GENERATED}
    COMMAND luajit_buildvm -m peobj -o lj_vm.obj
    COMMAND luajit_buildvm -m bcdef -o lj_bcdef.h ${LUAJIT_ALL_LIB}
    COMMAND luajit_buildvm -m ffdef -o lj_ffdef.h ${LUAJIT_ALL_LIB}
    COMMAND luajit_buildvm -m libdef -o lj_libdef.h ${LUAJIT_ALL_LIB}
    COMMAND luajit_buildvm -m recdef -o lj_recdef.h ${LUAJIT_ALL_LIB}
    COMMAND luajit_buildvm -m vmdef -o jit/vmdef.lua ${LUAJIT_ALL_LIB}
    COMMAND luajit_buildvm -m folddef -o lj_folddef.h lj_opt_fold.c
    DEPENDS luajit_buildvm
    WORKING_DIRECTORY ${LUAJIT_SRC_DIR}
    COMMENT "LuaJIT: buildvm (interpreter object + generated tables)"
    VERBATIM
)

# 5. The library: every lj_*.c and lib_*.c, plus the interpreter object. `lj_*.c` does not match
# ljamalg.c, the single-file amalgamation that would compile everything a second time. The glob runs
# after any earlier build left lj_vm.obj in the tree, so .c only.
file(GLOB LUAJIT_LIB_SOURCES ${LUAJIT_SRC_DIR}/lj_*.c ${LUAJIT_SRC_DIR}/lib_*.c)
add_library(luajit_static STATIC EXCLUDE_FROM_ALL ${LUAJIT_LIB_SOURCES} ${LUAJIT_GENERATED})
set_source_files_properties(${LUAJIT_SRC_DIR}/lj_vm.obj PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
target_include_directories(luajit_static PUBLIC ${LUAJIT_SRC_DIR})
target_compile_definitions(luajit_static PRIVATE ${LUAJIT_MSVC_DEFS})

add_library(luajit::luajit ALIAS luajit_static)
