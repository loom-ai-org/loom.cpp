# P4.46: every symbol libggml-vc4.so needs from outside the C and C++ runtimes must come from
# libggml-base.so -- the only ggml library a backend module links. ggml's loader dlopens the module and
# DROPS it, with one line on stderr, when a symbol is missing: a call into ggml's registry (libggml)
# did exactly that on the Pi Zero, and "auto" ran every model on the CPU without saying why, while the
# board test -- which links libggml itself -- passed. Versioned imports (glibc, libstdc++, libgcc) and
# weak ones are the runtimes'; everything else must be defined by libggml-base.
#   cmake -DNM=nm -DMODULE=libggml-vc4.so -DBASE=libggml-base.so -P vc4_module_symbols.cmake
execute_process(COMMAND ${NM} -D --undefined-only ${MODULE} OUTPUT_VARIABLE undef RESULT_VARIABLE r1)
execute_process(COMMAND ${NM} -D --defined-only ${BASE} OUTPUT_VARIABLE defs RESULT_VARIABLE r2)
if(NOT r1 EQUAL 0 OR NOT r2 EQUAL 0)
    message(FATAL_ERROR "nm failed on ${MODULE} or ${BASE}")
endif()
string(REGEX MATCHALL "[A-Za-z] [^\n]+" def_lines "${defs}")
set(defined "")
foreach(l IN LISTS def_lines)
    string(REGEX REPLACE "^[A-Za-z] " "" sym "${l}")
    list(APPEND defined "${sym}")
endforeach()
string(REPLACE "\n" ";" undef_lines "${undef}")
set(missing "")
set(checked 0)
foreach(l IN LISTS undef_lines)
    if(NOT l MATCHES "^ +U ([^ @]+)$")   # versioned (name@VER) and weak (w / v) imports are the runtimes'
        continue()
    endif()
    set(sym "${CMAKE_MATCH_1}")
    math(EXPR checked "${checked} + 1")
    list(FIND defined "${sym}" at)
    if(at EQUAL -1)
        list(APPEND missing "${sym}")
    endif()
endforeach()
if(missing)
    message(FATAL_ERROR "libggml-vc4.so needs symbols libggml-base.so does not define (ggml's loader would drop it): ${missing}")
endif()
message(STATUS "libggml-vc4.so: ${checked} unversioned imports, all defined by libggml-base.so")
