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
    src/core/backend.cpp
    src/core/gguf_model.cpp
    src/core/symbol_env.cpp
    src/core/graph_topology.cpp
    src/core/graph_builder.cpp
    src/core/conv_state_cache.cpp
    src/core/kv_cache.cpp
    src/core/output_store.cpp
    src/core/profile.cpp
    src/core/model_contract.cpp
    src/core/session.cpp
    src/core/transcribe.cpp
    src/core/text_generate.cpp
    src/core/generation.cpp
    src/core/bilstm_stepper.cpp
    src/core/duration_aligner.cpp
    src/core/relative_position.cpp
    src/core/lua_bridge.cpp
    src/core/vocab.cpp
    src/core/unicode.cpp
    src/core/bpe_vocab.cpp
    src/core/chat_template.cpp
    src/core/wordpiece_vocab.cpp
    src/core/byte_vocab.cpp
    src/core/phoneme_vocab.cpp
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

# `src/core/profile.cpp` includes ggml's PRIVATE `ggml-impl.h`, for the one declaration ggml does not
# export: `ggml_graph_view`, which returns a `ggml_cgraph` by value and so needs the complete type.
# PRIVATE and target-scoped on purpose -- exactly one translation unit may see ggml's internals, and a
# consumer of loom::engine sees none of them. Affordable only because cmake/GgmlPin.cmake pins ggml to
# an exact revision; see that file's own comment in profile.cpp.
target_include_directories(loom_engine PRIVATE $<BUILD_INTERFACE:${ggml_SOURCE_DIR}/src>)

if(LOOM_BUILD_TOOLS)
    add_subdirectory(tools/loom_cli)
endif()

if(LOOM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
