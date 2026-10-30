# ggml-vc4: the Pi Zero / Pi 1-3 GPU (VideoCore IV, 12 QPUs) as a ggml ACCEL backend used WITH the
# CPU (P4.44, ADR-090). Built through ggml's own `ggml_add_backend_library`, so it is whatever every
# other ggml backend is in this build: a MODULE `libggml-vc4.so` beside the others under
# GGML_BACKEND_DL (what the wheels ship; a dependency of `ggml`, so an EXCLUDE_FROM_ALL consumer such
# as loom-py still builds it), a library linked into `ggml` otherwise. ggml's registry knows nothing
# of it in either case -- src/core/backend.cpp loads or registers it.
#
# It compiles anywhere Linux does and reports no device without /dev/vcio, which is how CI builds it
# on x86 (-DLOOM_GGML_VC4=ON). The kernels themselves are tested hermetically in the QPU emulator
# (tests/ci/test_vc4_*); only tests/gate/test_vc4_backend needs a board.

# ggml_add_backend_library writes a DL module to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}, read in THIS
# scope -- and ggml sets that variable only inside its own directory when nothing above did. Mirror
# its rule so the module lands in the same `bin/` as libggml-cpu.so rather than in this directory.
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

ggml_add_backend_library(ggml-vc4 ggml-vc4.cpp)
# ggml-backend-impl.h / ggml-impl.h, the private headers every in-tree backend includes. Safe for the
# same reason as profile.cpp's: cmake/GgmlPin.cmake pins ggml to one revision.
target_include_directories(ggml-vc4 PRIVATE ${ggml_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
