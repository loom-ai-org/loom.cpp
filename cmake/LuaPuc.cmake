# PUC Lua 5.1.5 + LuaBitOp 1.0.3: the driver VM where LuaJIT cannot go (`LOOM_LUA=puc`, ADR-139).
#
# LuaJIT's interpreter is hand-written assembly per architecture (DynASM), so it has no WebAssembly
# target; an emscripten build needs a VM written in C. Lua 5.1 is the dialect LuaJIT implements and the
# C API `lua_bridge.cpp` is written against (it already treats a Lua error as a longjmp), so the same
# bridge and the same drivers run on it. The one library LuaJIT has and 5.1 lacks is `bit`, which three
# driver families use (aten_randn's Mersenne Twister, Pocket-TTS, SpeechT5, Soprano): LuaBitOp is that
# library, by LuaJIT's author, with the same semantics (32-bit, signed results).
#
# Both are MIT, fetched by hash rather than vendored, as ggml and LuaJIT are.
include(FetchContent)

FetchContent_Declare(
    lua51
    URL      https://www.lua.org/ftp/lua-5.1.5.tar.gz
    URL_HASH SHA256=2640fc56a795f29d28ef15e13c34a47e223960b0240e8cb0a82d9b0738695333
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
    luabitop
    URL      https://bitop.luajit.org/download/LuaBitOp-1.0.3.tar.gz
    URL_HASH SHA256=d514a3d2cefa76c8d11c1b9ec740d5fae316a9c9764e1e12ddea21e4982fab4b
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
foreach(dep lua51 luabitop)
    FetchContent_GetProperties(${dep})
    if(NOT ${dep}_POPULATED)
        FetchContent_Populate(${dep})
    endif()
endforeach()

set(LUA51_SRC_DIR ${lua51_SOURCE_DIR}/src)
# The library half of src/: every file but the two programs (lua.c, luac.c) and luac's print.c.
set(LUA51_SOURCES
    lapi.c lcode.c ldebug.c ldo.c ldump.c lfunc.c lgc.c llex.c lmem.c lobject.c lopcodes.c lparser.c
    lstate.c lstring.c ltable.c ltm.c lundump.c lvm.c lzio.c
    lauxlib.c lbaselib.c ldblib.c liolib.c lmathlib.c loslib.c ltablib.c lstrlib.c loadlib.c linit.c)
list(TRANSFORM LUA51_SOURCES PREPEND ${LUA51_SRC_DIR}/)

add_library(loom_lua_puc STATIC ${LUA51_SOURCES} ${luabitop_SOURCE_DIR}/bit.c)
target_include_directories(loom_lua_puc PUBLIC ${LUA51_SRC_DIR})
set_target_properties(loom_lua_puc PROPERTIES POSITION_INDEPENDENT_CODE ON C_STANDARD 99)
# Seen by the bridge, which opens `bit` itself (LuaJIT has it built in).
target_compile_definitions(loom_lua_puc PUBLIC LOOM_LUA_PUC=1)
if(UNIX AND NOT EMSCRIPTEN)
    # mkstemp instead of tmpnam, and popen; what the Makefile's `posix` target sets.
    target_compile_definitions(loom_lua_puc PRIVATE LUA_USE_POSIX)
endif()
if(NOT MSVC)
    # Lua is C written for -O2 with no warnings flags; ours would only be noise in someone else's code.
    target_compile_options(loom_lua_puc PRIVATE -w)
    target_link_libraries(loom_lua_puc PUBLIC m)
endif()

add_library(loom::lua ALIAS loom_lua_puc)
message(STATUS "Lua: PUC Lua 5.1.5 + LuaBitOp 1.0.3 (LOOM_LUA=puc)")
