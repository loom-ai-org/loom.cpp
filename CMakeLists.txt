cmake_minimum_required(VERSION 3.19)
project(loom-engine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

option(LOOM_BUILD_TESTS "Build the ctest suite" ON)
option(LOOM_BUILD_TOOLS "Build the CLI/demo tools" ON)

include(cmake/Dependencies.cmake)

add_library(loom_engine
    src/core/gguf_model.cpp
    src/core/symbol_env.cpp
    src/core/graph_topology.cpp
    src/core/graph_builder.cpp
    src/core/kv_cache.cpp
    src/core/generation.cpp
    src/core/tdt_decoder.cpp
    src/core/whisper_driver.cpp
    src/core/kokoro_driver.cpp
    src/core/styletts2_driver.cpp
    src/core/bilstm_stepper.cpp
    src/core/duration_aligner.cpp
    src/core/relative_position.cpp
    src/core/style_diffusion_sampler.cpp
    src/core/cfm_euler_sampler.cpp
    src/core/supertonic_driver.cpp
    src/core/matcha_driver.cpp
    src/core/lua_bridge.cpp
    src/core/ode_stepper.cpp
    src/core/vocab.cpp
    src/core/unicode.cpp
    src/core/bpe_vocab.cpp
    src/core/wordpiece_vocab.cpp
    src/core/byte_vocab.cpp
    src/core/supertonic_text_vectorizer.cpp
    src/core/ctc_decode.cpp
    src/ops/primitive_registry.cpp
    src/ops/primitives_basic.cpp
    src/ops/primitives_mil.cpp
    src/ops/primitives_rope.cpp
    src/ops/primitives_attention.cpp
    src/ops/primitives_conv.cpp
    src/ops/primitives_spline.cpp
    src/ops/primitives_flow.cpp
    src/ops/primitives_recurrent.cpp
)
add_library(loom::engine ALIAS loom_engine)

target_include_directories(loom_engine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(loom_engine PUBLIC ggml nlohmann_json::nlohmann_json luajit::luajit)

if(LOOM_BUILD_TOOLS)
    add_subdirectory(tools/loom_cli)
endif()

if(LOOM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
