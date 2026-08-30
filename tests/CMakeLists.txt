# Two classes of test, and a test's own directory is which class it is in.
#
#   ci/    hermetic. Needs nothing but this repo, a C++ toolchain and `gguf`+`numpy`: every fixture it
#          reads is built from `fixtures/*.py` by a ctest step that runs first. 28 executables, the
#          whole suite in seconds. This is what GitHub Actions runs, and a failure here is a real
#          regression rather than a missing download.
#   gate/  needs a REAL model. 78 executables comparing against reference tensors that only exist once
#          a checkpoint has been exported (loom-exporter) and a reference forward has been run over it
#          -- gigabytes, hours, and per-checkpoint licences. Every one of them exits 77 when its
#          fixture is absent, which ctest reports as Skipped, so a developer with none of them still
#          gets a green suite that means "nothing hermetic broke".
#
# Both are also labelled, so `ctest -L ci` and `ctest -L gate` select them without knowing the names.
#
# Where a gate fixture comes from: `LOOM_FIXTURES` names one directory holding all of them, laid out
# as `scripts/fixtures.py` describes and verifies. The per-model environment variables each test
# documents still work and still win when set -- see `support/fixtures.h` -- because pointing one test
# at one file you just rebuilt is exactly what you do while working on it.

# Where a GGML_BACKEND_DL build keeps its backend .so files, for `support/cpu_backend.h`.
#
# In such a build the registry starts EMPTY -- the CPU is a plugin too -- and ggml's own search covers
# the executable's directory and the current directory, neither of which is right here: the test
# binaries land in `tests/` and ggml puts its backends in `bin/`. So every test is told the directory
# explicitly, through the engine's own `loom::add_backend_search_path`.
#
# A directory-scope definition rather than 129 per-target ones, and harmless in a linked build, where
# nothing consults it because the backends have already registered themselves.
add_compile_definitions(LOOM_TEST_BACKEND_DIR="${CMAKE_BINARY_DIR}/bin")

# Was a device backend compiled in? The device-parity gates skip when they find no device, which is
# right on a CPU-only build and a LIE on a build configured with one -- and a lie that renders as a
# green ctest run, since exit 77 is Skipped (BACKLOG.md P4.8f). With this defined they fail instead,
# because "Vulkan is compiled in and the registry has no Vulkan device" is a broken deployment, not an
# absent GPU.
#
# The list is the backends whose devices those gates can compare against, i.e. anything that registers
# as something other than the CPU. Add a backend here when adding one to the build.
if(GGML_VULKAN OR GGML_CUDA OR GGML_METAL OR GGML_SYCL OR GGML_HIP OR GGML_OPENCL OR GGML_MUSA OR GGML_CANN OR GGML_OPENVINO OR GGML_HEXAGON)
    add_compile_definitions(LOOM_TEST_EXPECTS_DEVICE=1)
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(LOOM_FIXTURE_DIR ${CMAKE_CURRENT_BINARY_DIR}/fixtures)

# `#include "test_util.h"` keeps working from either subdirectory: the shared headers moved to
# support/ and this is what spares 106 files an edit that would have said nothing.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/support)

# The frozen reference waveforms of the retired per-model C++ drivers (P4.0.8, E.3). Unlike
# LOOM_FIXTURE_DIR these are NOT generated at build time -- the code that produced them is gone -- so
# they live in the source tree and are read from there. See that directory's README.md.
set(LOOM_LEGACY_REF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/legacy_driver_reference)

# --- Phase 1: minimal ad hoc GGUF fixture, regenerated fresh before test_gguf_model_load runs ---
add_test(
    NAME gen_minimal_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_minimal_gguf.py
            ${LOOM_FIXTURE_DIR}/minimal.gguf
)
set_tests_properties(gen_minimal_gguf_fixture PROPERTIES LABELS ci)
set_tests_properties(gen_minimal_gguf_fixture PROPERTIES FIXTURES_SETUP minimal_gguf_fixture)

add_executable(test_gguf_model_load ci/test_gguf_model_load.cpp)
target_link_libraries(test_gguf_model_load PRIVATE loom::engine)
target_compile_definitions(test_gguf_model_load PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_gguf_model_load COMMAND test_gguf_model_load)
set_tests_properties(test_gguf_model_load PROPERTIES LABELS ci)
set_tests_properties(test_gguf_model_load PROPERTIES FIXTURES_REQUIRED minimal_gguf_fixture)

# --- What a file declares about itself, and what a file that declares nothing falls back to ---
add_test(
    NAME gen_contract_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_contract_gguf.py
            ${LOOM_FIXTURE_DIR}
)
set_tests_properties(gen_contract_gguf_fixture PROPERTIES LABELS ci)
set_tests_properties(gen_contract_gguf_fixture PROPERTIES FIXTURES_SETUP contract_gguf_fixture)

add_executable(test_model_contract ci/test_model_contract.cpp)
target_link_libraries(test_model_contract PRIVATE loom::engine)
target_compile_definitions(test_model_contract PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_model_contract COMMAND test_model_contract)
set_tests_properties(test_model_contract PROPERTIES LABELS ci)
set_tests_properties(test_model_contract PROPERTIES FIXTURES_REQUIRED contract_gguf_fixture)

add_executable(test_text_generate ci/test_text_generate.cpp)
target_link_libraries(test_text_generate PRIVATE loom::engine)
target_compile_definitions(test_text_generate PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_text_generate COMMAND test_text_generate)
set_tests_properties(test_text_generate PROPERTIES LABELS ci)
set_tests_properties(test_text_generate PROPERTIES FIXTURES_REQUIRED contract_gguf_fixture)

add_executable(test_transcribe_seek ci/test_transcribe_seek.cpp)
target_link_libraries(test_transcribe_seek PRIVATE loom::engine)
target_compile_definitions(test_transcribe_seek PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_transcribe_seek COMMAND test_transcribe_seek)
set_tests_properties(test_transcribe_seek PROPERTIES LABELS ci)
set_tests_properties(test_transcribe_seek PROPERTIES FIXTURES_REQUIRED contract_gguf_fixture)

add_executable(test_transcribe_no_selection ci/test_transcribe_no_selection.cpp)
target_link_libraries(test_transcribe_no_selection PRIVATE loom::engine)
target_compile_definitions(test_transcribe_no_selection PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_transcribe_no_selection COMMAND test_transcribe_no_selection)
set_tests_properties(test_transcribe_no_selection PROPERTIES LABELS ci)
set_tests_properties(test_transcribe_no_selection PROPERTIES FIXTURES_REQUIRED contract_gguf_fixture)

# --- Phase 2: primitive registry / topology parsing / graph builder ---

add_executable(test_primitive_registry ci/test_primitive_registry.cpp)
target_link_libraries(test_primitive_registry PRIVATE loom::engine)
add_test(NAME test_primitive_registry COMMAND test_primitive_registry)
set_tests_properties(test_primitive_registry PROPERTIES LABELS ci)

add_executable(test_e2e_sliding_window_attention gate/test_e2e_sliding_window_attention.cpp)
target_link_libraries(test_e2e_sliding_window_attention PRIVATE loom::engine)
add_test(NAME test_e2e_sliding_window_attention COMMAND test_e2e_sliding_window_attention)
set_tests_properties(test_e2e_sliding_window_attention PROPERTIES LABELS gate)
set_tests_properties(test_e2e_sliding_window_attention PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_e2e_spm_byte_fallback_tokenizer gate/test_e2e_spm_byte_fallback_tokenizer.cpp)
target_link_libraries(test_e2e_spm_byte_fallback_tokenizer PRIVATE loom::engine)
add_test(NAME test_e2e_spm_byte_fallback_tokenizer COMMAND test_e2e_spm_byte_fallback_tokenizer)
set_tests_properties(test_e2e_spm_byte_fallback_tokenizer PROPERTIES LABELS gate)
set_tests_properties(test_e2e_spm_byte_fallback_tokenizer PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_short_conv_state ci/test_short_conv_state.cpp)
target_link_libraries(test_short_conv_state PRIVATE loom::engine)
add_test(NAME test_short_conv_state COMMAND test_short_conv_state)
set_tests_properties(test_short_conv_state PROPERTIES LABELS ci)

add_executable(test_graph_topology_parse ci/test_graph_topology_parse.cpp)
target_link_libraries(test_graph_topology_parse PRIVATE loom::engine)
add_test(NAME test_graph_topology_parse COMMAND test_graph_topology_parse)
set_tests_properties(test_graph_topology_parse PROPERTIES LABELS ci)

# Regression: the "heal transposed layouts" heuristics must not fire on already-correct operands when
# two axes collide (n_tokens == n_head). Reuses the minimal fixture purely as a GgufModel to hand
# GraphBuilder -- the topologies it runs declare their own inputs and use no weights.
add_executable(test_broadcast_axis_collision ci/test_broadcast_axis_collision.cpp)
target_link_libraries(test_broadcast_axis_collision PRIVATE loom::engine)
target_compile_definitions(test_broadcast_axis_collision PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_broadcast_axis_collision COMMAND test_broadcast_axis_collision)
set_tests_properties(test_broadcast_axis_collision PROPERTIES LABELS ci)
set_tests_properties(test_broadcast_axis_collision PROPERTIES FIXTURES_REQUIRED minimal_gguf_fixture)

# Attention-free toy model GGUF shared by test_graph_builder_shapes and test_gallocr_reserve_reuse.
add_test(
    NAME gen_builder_test_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_builder_test_gguf.py
            ${LOOM_FIXTURE_DIR}/builder_test.gguf
)
set_tests_properties(gen_builder_test_gguf_fixture PROPERTIES LABELS ci)
set_tests_properties(gen_builder_test_gguf_fixture PROPERTIES FIXTURES_SETUP builder_test_gguf_fixture)

add_executable(test_graph_builder_shapes ci/test_graph_builder_shapes.cpp)
target_link_libraries(test_graph_builder_shapes PRIVATE loom::engine)
target_compile_definitions(test_graph_builder_shapes PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_graph_builder_shapes COMMAND test_graph_builder_shapes)
set_tests_properties(test_graph_builder_shapes PROPERTIES LABELS ci)
set_tests_properties(test_graph_builder_shapes PROPERTIES FIXTURES_REQUIRED builder_test_gguf_fixture)

add_executable(test_lua_bridge_has_function ci/test_lua_bridge_has_function.cpp)
target_link_libraries(test_lua_bridge_has_function PRIVATE loom::engine)
add_test(NAME test_lua_bridge_has_function COMMAND test_lua_bridge_has_function)
set_tests_properties(test_lua_bridge_has_function PROPERTIES LABELS ci)

add_executable(test_lua_bridge_run_subgraph ci/test_lua_bridge_run_subgraph.cpp)
target_link_libraries(test_lua_bridge_run_subgraph PRIVATE loom::engine)
target_compile_definitions(test_lua_bridge_run_subgraph PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_lua_bridge_run_subgraph COMMAND test_lua_bridge_run_subgraph)
set_tests_properties(test_lua_bridge_run_subgraph PROPERTIES LABELS ci)
set_tests_properties(test_lua_bridge_run_subgraph PROPERTIES FIXTURES_REQUIRED builder_test_gguf_fixture)

add_executable(test_lua_bridge_retained_outputs ci/test_lua_bridge_retained_outputs.cpp)
target_link_libraries(test_lua_bridge_retained_outputs PRIVATE loom::engine)
target_compile_definitions(test_lua_bridge_retained_outputs PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_lua_bridge_retained_outputs COMMAND test_lua_bridge_retained_outputs)
set_tests_properties(test_lua_bridge_retained_outputs PROPERTIES LABELS ci)
set_tests_properties(test_lua_bridge_retained_outputs PROPERTIES FIXTURES_REQUIRED builder_test_gguf_fixture)

# --- P4.24: loom.sample_row, on the same toy fixture -- 6 classes wide, which is what makes "every
#     class is eventually drawn" a statement a test can make.
add_executable(test_sample_row ci/test_sample_row.cpp)
target_link_libraries(test_sample_row PRIVATE loom::engine)
target_compile_definitions(test_sample_row PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_sample_row COMMAND test_sample_row)
set_tests_properties(test_sample_row PROPERTIES LABELS ci)
set_tests_properties(test_sample_row PROPERTIES FIXTURES_REQUIRED builder_test_gguf_fixture)

# Registered TWICE against one binary, differing only in ENVIRONMENT: `profile::enabled()` caches its
# answer on first call, so a single process can only ever observe one of the two routes through
# GraphBuilder::compute, and the branch being stuck in either direction is a real failure (one makes
# every shipped run pay per-node dispatch, the other makes the feature do nothing). The test asserts
# whichever direction its own environment selects.
add_executable(test_profile ci/test_profile.cpp)
target_link_libraries(test_profile PRIVATE loom::engine)
target_compile_definitions(test_profile PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_profile COMMAND test_profile)
set_tests_properties(test_profile PROPERTIES LABELS ci)
set_tests_properties(test_profile PROPERTIES FIXTURES_REQUIRED builder_test_gguf_fixture)
set_tests_properties(test_profile PROPERTIES ENVIRONMENT "LOOM_PROFILE=")

add_test(NAME test_profile_enabled COMMAND test_profile)
set_tests_properties(test_profile_enabled PROPERTIES LABELS ci)
set_tests_properties(test_profile_enabled PROPERTIES FIXTURES_REQUIRED builder_test_gguf_fixture)
set_tests_properties(test_profile_enabled PROPERTIES ENVIRONMENT "LOOM_PROFILE=1")

# A THIRD registration, for `$LOOM_PROFILE_NODES` -- cached on first call like the other two, so it
# needs its own process to be observed. The two above run with it unset and assert the finer table is
# empty; this one asserts it is populated and accounts for exactly the same node executions.
add_test(NAME test_profile_nodes COMMAND test_profile)
set_tests_properties(test_profile_nodes PROPERTIES LABELS ci)
set_tests_properties(test_profile_nodes PROPERTIES FIXTURES_REQUIRED builder_test_gguf_fixture)
set_tests_properties(test_profile_nodes PROPERTIES ENVIRONMENT "LOOM_PROFILE=1;LOOM_PROFILE_NODES=1")

# Registered TWICE, differing only in ENVIRONMENT, for the same reason test_profile is: ggml reads
# GGML_CPU_DISABLE_FUSION once, on first use, so one process can only observe one of the two paths.
# The two registrations also hand each other their outputs through LOOM_TEST_TMPDIR, so whichever runs
# second compares them byte for byte -- ordered by ctest's DEPENDS so that always happens.
add_executable(test_conv_bias_fusion ci/test_conv_bias_fusion.cpp)
target_link_libraries(test_conv_bias_fusion PRIVATE loom::engine)
add_test(NAME test_conv_bias_fusion_unfused COMMAND test_conv_bias_fusion)
set_tests_properties(test_conv_bias_fusion_unfused PROPERTIES LABELS ci)
set_tests_properties(test_conv_bias_fusion_unfused PROPERTIES ENVIRONMENT
    "GGML_CPU_DISABLE_FUSION=1;LOOM_TEST_TMPDIR=${LOOM_FIXTURE_DIR}")

add_test(NAME test_conv_bias_fusion COMMAND test_conv_bias_fusion)
set_tests_properties(test_conv_bias_fusion PROPERTIES LABELS ci)
set_tests_properties(test_conv_bias_fusion PROPERTIES ENVIRONMENT
    "GGML_CPU_DISABLE_FUSION=0;LOOM_TEST_TMPDIR=${LOOM_FIXTURE_DIR}")
set_tests_properties(test_conv_bias_fusion PROPERTIES DEPENDS test_conv_bias_fusion_unfused)

# Registered with the build's own answer for LOOM_TINYBLAS, so the test asserts the direction this
# build asked for rather than whatever it happens to have got -- `-DLOOM_TINYBLAS=OFF` has to keep
# meaning tinyBLAS is really absent, which is what makes a GEMM A/B trustworthy.
add_executable(test_tinyblas_gemm ci/test_tinyblas_gemm.cpp)
target_link_libraries(test_tinyblas_gemm PRIVATE loom::engine)
target_compile_definitions(test_tinyblas_gemm PRIVATE LOOM_EXPECT_TINYBLAS=$<BOOL:${LOOM_TINYBLAS}>)
add_test(NAME test_tinyblas_gemm COMMAND test_tinyblas_gemm)
set_tests_properties(test_tinyblas_gemm PROPERTIES LABELS ci)

add_executable(test_gallocr_shrink ci/test_gallocr_shrink.cpp)
target_link_libraries(test_gallocr_shrink PRIVATE loom::engine)
target_compile_definitions(test_gallocr_shrink PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_gallocr_shrink COMMAND test_gallocr_shrink)
set_tests_properties(test_gallocr_shrink PROPERTIES LABELS ci)
set_tests_properties(test_gallocr_shrink PROPERTIES FIXTURES_REQUIRED builder_test_gguf_fixture)

# The scheduler path GPU/NPU support runs on (BACKLOG.md P4.7). Uses the same toy model, driven through
# a two-backend `Backends` -- see the test's own header for why two CPUs is the right hermetic stand-in.
add_executable(test_scheduled_graph ci/test_scheduled_graph.cpp)
target_link_libraries(test_scheduled_graph PRIVATE loom::engine)
target_compile_definitions(test_scheduled_graph PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_scheduled_graph COMMAND test_scheduled_graph)
set_tests_properties(test_scheduled_graph PROPERTIES LABELS ci)
set_tests_properties(test_scheduled_graph PROPERTIES FIXTURES_REQUIRED builder_test_gguf_fixture)

# POOL_1D's lowering to a one-tall POOL_2D, and the one combination it must refuse (BACKLOG.md P4.7d).
add_executable(test_pool_1d_lowering ci/test_pool_1d_lowering.cpp)
target_link_libraries(test_pool_1d_lowering PRIVATE loom::engine)
add_test(NAME test_pool_1d_lowering COMMAND test_pool_1d_lowering)
set_tests_properties(test_pool_1d_lowering PROPERTIES LABELS ci)

# PAD_1D_REFLECT's composed fallback, against ggml's own primitive (BACKLOG.md P4.7e).
add_executable(test_pad_reflect_lowering ci/test_pad_reflect_lowering.cpp)
target_link_libraries(test_pad_reflect_lowering PRIVATE loom::engine)
add_test(NAME test_pad_reflect_lowering COMMAND test_pad_reflect_lowering)
set_tests_properties(test_pad_reflect_lowering PROPERTIES LABELS ci)

# ATAN's polynomial fallback, measured against libm (BACKLOG.md P4.7f).
add_executable(test_atan_lowering ci/test_atan_lowering.cpp)
target_link_libraries(test_atan_lowering PRIVATE loom::engine)
add_test(NAME test_atan_lowering COMMAND test_atan_lowering)
set_tests_properties(test_atan_lowering PROPERTIES LABELS ci)

# Device selection. Needs no fixture at all -- it asks the ggml backend registry what this BUILD has.
add_executable(test_device_selection ci/test_device_selection.cpp)
target_link_libraries(test_device_selection PRIVATE loom::engine)
add_test(NAME test_device_selection COMMAND test_device_selection)
set_tests_properties(test_device_selection PROPERTIES LABELS ci)

add_executable(test_gallocr_reserve_reuse ci/test_gallocr_reserve_reuse.cpp)
target_link_libraries(test_gallocr_reserve_reuse PRIVATE loom::engine)
target_compile_definitions(test_gallocr_reserve_reuse PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_gallocr_reserve_reuse COMMAND test_gallocr_reserve_reuse)
set_tests_properties(test_gallocr_reserve_reuse PROPERTIES LABELS ci)
set_tests_properties(test_gallocr_reserve_reuse PROPERTIES FIXTURES_REQUIRED builder_test_gguf_fixture)

# --- Phase 3: KV cache / ATTENTION primitive / autoregressive generation ---

add_executable(test_kv_cache ci/test_kv_cache.cpp)
target_link_libraries(test_kv_cache PRIVATE loom::engine)
add_test(NAME test_kv_cache COMMAND test_kv_cache)
set_tests_properties(test_kv_cache PROPERTIES LABELS ci)

# Tiny single-layer attention toy model shared by test_generation_smoke.
add_test(
    NAME gen_attention_test_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_attention_test_gguf.py
            ${LOOM_FIXTURE_DIR}/attention_test.gguf
)
set_tests_properties(gen_attention_test_gguf_fixture PROPERTIES LABELS ci)
set_tests_properties(gen_attention_test_gguf_fixture PROPERTIES FIXTURES_SETUP attention_test_gguf_fixture)

add_executable(test_generation_smoke ci/test_generation_smoke.cpp)
target_link_libraries(test_generation_smoke PRIVATE loom::engine)
target_compile_definitions(test_generation_smoke PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_generation_smoke COMMAND test_generation_smoke)
set_tests_properties(test_generation_smoke PROPERTIES LABELS ci)
set_tests_properties(test_generation_smoke PROPERTIES FIXTURES_REQUIRED attention_test_gguf_fixture)

# --- Phase 4: full toy-LLM end-to-end test, verified against an independent numpy reference ---

set(LOOM_E2E_REF_DIR ${LOOM_FIXTURE_DIR}/e2e_ref)

add_test(
    NAME gen_toy_llm_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_toy_llm_gguf.py
            ${LOOM_FIXTURE_DIR}/toy_llm.gguf
)
set_tests_properties(gen_toy_llm_gguf_fixture PROPERTIES LABELS ci)
set_tests_properties(gen_toy_llm_gguf_fixture PROPERTIES FIXTURES_SETUP toy_llm_gguf_fixture)

add_test(
    NAME gen_toy_llm_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/reference_forward.py
            ${LOOM_E2E_REF_DIR} --prompt 1 2 3 --n-new 4
)
set_tests_properties(gen_toy_llm_reference PROPERTIES LABELS ci)
set_tests_properties(gen_toy_llm_reference PROPERTIES FIXTURES_SETUP toy_llm_reference_fixture)

add_executable(test_e2e_toy_llm ci/test_e2e_toy_llm.cpp)
target_link_libraries(test_e2e_toy_llm PRIVATE loom::engine)
target_compile_definitions(test_e2e_toy_llm PRIVATE
    LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}"
    LOOM_TEST_REF_DIR="${LOOM_E2E_REF_DIR}"
)
add_test(NAME test_e2e_toy_llm COMMAND test_e2e_toy_llm)
set_tests_properties(test_e2e_toy_llm PROPERTIES LABELS ci)
set_tests_properties(test_e2e_toy_llm PROPERTIES FIXTURES_REQUIRED "toy_llm_gguf_fixture;toy_llm_reference_fixture")

# --- Generic-converter proof of concept: same weights/reference as test_e2e_toy_llm above, but the
#     topology JSON is auto-derived from a real torch.export() ATen graph (tools/convert_generic/), not
#     hand-written. Needs a torch environment to produce its GGUF, so -- like the real-model tests below
#     -- it's not wired into the default fixture-generation flow; it looks for LOOM_TOY_LLM_GENERIC_DIR
#     and skips cleanly (SKIP_RETURN_CODE) if that GGUF hasn't been prepared. Reuses toy_llm_reference_fixture
#     (same weights => same expected logits) rather than generating a second reference.
add_executable(test_e2e_toy_llm_generic gate/test_e2e_toy_llm_generic.cpp)
target_link_libraries(test_e2e_toy_llm_generic PRIVATE loom::engine)
target_compile_definitions(test_e2e_toy_llm_generic PRIVATE LOOM_TEST_REF_DIR="${LOOM_E2E_REF_DIR}")
add_test(NAME test_e2e_toy_llm_generic COMMAND test_e2e_toy_llm_generic)
set_tests_properties(test_e2e_toy_llm_generic PROPERTIES LABELS gate)
set_tests_properties(test_e2e_toy_llm_generic PROPERTIES
    SKIP_RETURN_CODE 77
    FIXTURES_REQUIRED toy_llm_reference_fixture)

# --- Milestone 2: toy vision encoder end-to-end test ---

set(LOOM_E2E_VISION_REF_DIR ${LOOM_FIXTURE_DIR}/e2e_vision_ref)

add_test(
    NAME gen_toy_vision_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_toy_vision_gguf.py
            ${LOOM_FIXTURE_DIR}/toy_vision.gguf
)
set_tests_properties(gen_toy_vision_gguf_fixture PROPERTIES LABELS ci)
set_tests_properties(gen_toy_vision_gguf_fixture PROPERTIES FIXTURES_SETUP toy_vision_gguf_fixture)

add_test(
    NAME gen_toy_vision_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/reference_forward_vision.py
            ${LOOM_E2E_VISION_REF_DIR}
)
set_tests_properties(gen_toy_vision_reference PROPERTIES LABELS ci)
set_tests_properties(gen_toy_vision_reference PROPERTIES FIXTURES_SETUP toy_vision_reference_fixture)

add_executable(test_e2e_toy_vision ci/test_e2e_toy_vision.cpp)
target_link_libraries(test_e2e_toy_vision PRIVATE loom::engine)
target_compile_definitions(test_e2e_toy_vision PRIVATE
    LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}"
    LOOM_TEST_REF_DIR="${LOOM_E2E_VISION_REF_DIR}"
)
add_test(NAME test_e2e_toy_vision COMMAND test_e2e_toy_vision)
set_tests_properties(test_e2e_toy_vision PROPERTIES LABELS ci)
set_tests_properties(test_e2e_toy_vision PROPERTIES FIXTURES_REQUIRED "toy_vision_gguf_fixture;toy_vision_reference_fixture")

# --- Milestone 2: toy ASR encoder end-to-end test ---

set(LOOM_E2E_ASR_REF_DIR ${LOOM_FIXTURE_DIR}/e2e_asr_ref)

add_test(
    NAME gen_toy_asr_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_toy_asr_gguf.py
            ${LOOM_FIXTURE_DIR}/toy_asr.gguf
)
set_tests_properties(gen_toy_asr_gguf_fixture PROPERTIES LABELS ci)
set_tests_properties(gen_toy_asr_gguf_fixture PROPERTIES FIXTURES_SETUP toy_asr_gguf_fixture)

add_test(
    NAME gen_toy_asr_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/reference_forward_asr.py
            ${LOOM_E2E_ASR_REF_DIR}
)
set_tests_properties(gen_toy_asr_reference PROPERTIES LABELS ci)
set_tests_properties(gen_toy_asr_reference PROPERTIES FIXTURES_SETUP toy_asr_reference_fixture)

add_executable(test_e2e_toy_asr ci/test_e2e_toy_asr.cpp)
target_link_libraries(test_e2e_toy_asr PRIVATE loom::engine)
target_compile_definitions(test_e2e_toy_asr PRIVATE
    LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}"
    LOOM_TEST_REF_DIR="${LOOM_E2E_ASR_REF_DIR}"
)
add_test(NAME test_e2e_toy_asr COMMAND test_e2e_toy_asr)
set_tests_properties(test_e2e_toy_asr PROPERTIES LABELS ci)
set_tests_properties(test_e2e_toy_asr PROPERTIES FIXTURES_REQUIRED "toy_asr_gguf_fixture;toy_asr_reference_fixture")

# --- Milestone 3's toy ODE-stepper end-to-end test lived here. Retired with `OdeStepper` itself
#     (P4.0.8 follow-up): the flow-matching integration loop it drove is a `FlowMatchingSampler`
#     component emitted into Matcha's and Supertonic's Lua drivers now, and the class had no consumer
#     left but this test. The graph-reuse discipline it demonstrated is not lost with it --
#     test_graph_reuse_safety.cpp pins that in plain ggml calls, independently of any driver. ---

# --- Milestone 3: toy VAE decoder end-to-end test ---

set(LOOM_E2E_VAE_REF_DIR ${LOOM_FIXTURE_DIR}/e2e_vae_ref)

add_test(
    NAME gen_toy_vae_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_toy_vae_gguf.py
            ${LOOM_FIXTURE_DIR}/toy_vae.gguf
)
set_tests_properties(gen_toy_vae_gguf_fixture PROPERTIES LABELS ci)
set_tests_properties(gen_toy_vae_gguf_fixture PROPERTIES FIXTURES_SETUP toy_vae_gguf_fixture)

add_test(
    NAME gen_toy_vae_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/reference_forward_vae.py
            ${LOOM_E2E_VAE_REF_DIR}
)
set_tests_properties(gen_toy_vae_reference PROPERTIES LABELS ci)
set_tests_properties(gen_toy_vae_reference PROPERTIES FIXTURES_SETUP toy_vae_reference_fixture)

add_executable(test_e2e_toy_vae ci/test_e2e_toy_vae.cpp)
target_link_libraries(test_e2e_toy_vae PRIVATE loom::engine)
target_compile_definitions(test_e2e_toy_vae PRIVATE
    LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}"
    LOOM_TEST_REF_DIR="${LOOM_E2E_VAE_REF_DIR}"
)
add_test(NAME test_e2e_toy_vae COMMAND test_e2e_toy_vae)
set_tests_properties(test_e2e_toy_vae PROPERTIES LABELS ci)
set_tests_properties(test_e2e_toy_vae PROPERTIES FIXTURES_REQUIRED "toy_vae_gguf_fixture;toy_vae_reference_fixture")

# --- Regression test pinning down BACKLOG.md's ggml graph-reuse finding (see
#     tests/test_graph_reuse_safety.cpp) -- the pattern OdeStepper relies on ---
add_executable(test_graph_reuse_safety ci/test_graph_reuse_safety.cpp)
target_link_libraries(test_graph_reuse_safety PRIVATE loom::engine)
add_test(NAME test_graph_reuse_safety COMMAND test_graph_reuse_safety)
set_tests_properties(test_graph_reuse_safety PROPERTIES LABELS ci)

# --- The engine-side half of the same finding: GraphBuilder retains and reuses its graph
#     (BACKLOG.md P4.0.13), so a reused graph must be bit-identical to a from-scratch rebuild ---
add_executable(test_graph_builder_reuse ci/test_graph_builder_reuse.cpp)
target_link_libraries(test_graph_builder_reuse PRIVATE loom::engine)
target_compile_definitions(test_graph_builder_reuse PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_graph_builder_reuse COMMAND test_graph_builder_reuse)
set_tests_properties(test_graph_builder_reuse PROPERTIES LABELS ci)
set_tests_properties(test_graph_builder_reuse PROPERTIES FIXTURES_REQUIRED toy_llm_gguf_fixture)

# --- Real-model test: NVIDIA stt_en_conformer_ctc_small, converted via tools/convert_nemo/. Unlike
#     every toy fixture above, this is NOT regenerated at ctest time (needs the real checkpoint + a
#     PyTorch environment) -- it looks for LOOM_CONFORMER_CTC_DIR (default /tmp/nemo_model) and skips
#     cleanly via ctest's SKIP_RETURN_CODE if that directory hasn't been prepared. See
#     tools/convert_nemo/convert_conformer_ctc.py and reference_forward_conformer.py for how to build it.

# --- Conformer-CTC-small exported via the generic MIL compiler (export_conformer_ctc_mil.py), verified
#     against the SAME reference_forward_conformer.py fixture as the bespoke-conversion test above ---
add_executable(test_e2e_conformer_ctc_mil_export gate/test_e2e_conformer_ctc_mil_export.cpp)
target_link_libraries(test_e2e_conformer_ctc_mil_export PRIVATE loom::engine)
add_test(NAME test_e2e_conformer_ctc_mil_export COMMAND test_e2e_conformer_ctc_mil_export)
set_tests_properties(test_e2e_conformer_ctc_mil_export PROPERTIES LABELS gate)
set_tests_properties(test_e2e_conformer_ctc_mil_export PROPERTIES SKIP_RETURN_CODE 77)

# --- CPU vs GPU on the same real encoder (BACKLOG.md P4.7). Skips on a missing fixture AND on a build
#     with no device backend compiled in, so it is green everywhere it cannot say anything. ---
add_executable(test_e2e_device_parity gate/test_e2e_device_parity.cpp)
target_link_libraries(test_e2e_device_parity PRIVATE loom::engine)
add_test(NAME test_e2e_device_parity COMMAND test_e2e_device_parity)
set_tests_properties(test_e2e_device_parity PROPERTIES LABELS gate)
set_tests_properties(test_e2e_device_parity PROPERTIES SKIP_RETURN_CODE 77)

# --- The other half of device parity: a KV-cached decode loop, where the cache, the reused graph and
#     the scheduler's split plan all have to hold together while n_past moves (BACKLOG.md P4.7). ---
add_executable(test_e2e_device_parity_kv gate/test_e2e_device_parity_kv.cpp)
target_link_libraries(test_e2e_device_parity_kv PRIVATE loom::engine)
add_test(NAME test_e2e_device_parity_kv COMMAND test_e2e_device_parity_kv)
set_tests_properties(test_e2e_device_parity_kv PROPERTIES LABELS gate)
set_tests_properties(test_e2e_device_parity_kv PROPERTIES SKIP_RETURN_CODE 77)

# --- The same MIL-exported GGUF reached the way every other family is reached: through the embedded
#     Lua driver's `infer`. Its greedy CTC decode must agree token-for-token with loom::ctc_greedy_decode,
#     the C++ implementation it replaces (BACKLOG.md P4.0.17). ---
add_executable(test_e2e_conformer_ctc_lua_driver gate/test_e2e_conformer_ctc_lua_driver.cpp)
target_link_libraries(test_e2e_conformer_ctc_lua_driver PRIVATE loom::engine)
add_test(NAME test_e2e_conformer_ctc_lua_driver COMMAND test_e2e_conformer_ctc_lua_driver)
set_tests_properties(test_e2e_conformer_ctc_lua_driver PROPERTIES LABELS gate)
set_tests_properties(test_e2e_conformer_ctc_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# Proves the same converted GGUF supports a second, different waveform length dynamically (see
# SPECIFICATION.md §4 and BACKLOG.md) -- no reconversion, just a different build() call.

# --- Gating-criterion proof of concept: generic converter (tools/convert_generic/) against the real
#     Conformer-CTC encoder -- a genuinely non-decoder-transformer shape. See BACKLOG.md. ---
add_executable(test_e2e_conformer_ctc_generic gate/test_e2e_conformer_ctc_generic.cpp)
target_link_libraries(test_e2e_conformer_ctc_generic PRIVATE loom::engine)
add_test(NAME test_e2e_conformer_ctc_generic COMMAND test_e2e_conformer_ctc_generic)
set_tests_properties(test_e2e_conformer_ctc_generic PROPERTIES LABELS gate)
set_tests_properties(test_e2e_conformer_ctc_generic PROPERTIES SKIP_RETURN_CODE 77)

# --- Tokenizer: Vocab (SentencePiece unigram, llama.cpp GGUF schema) against the real converted model ---
add_executable(test_vocab gate/test_vocab.cpp)
target_link_libraries(test_vocab PRIVATE loom::engine)
add_test(NAME test_vocab COMMAND test_vocab)
set_tests_properties(test_vocab PROPERTIES LABELS gate)
set_tests_properties(test_vocab PROPERTIES SKIP_RETURN_CODE 77)

# --- Vocab's new SentencePiece BPE path ("llama" tag), against the real Parakeet-TDT tokenizer ---
add_executable(test_vocab_spm_bpe gate/test_vocab_spm_bpe.cpp)
target_link_libraries(test_vocab_spm_bpe PRIVATE loom::engine)
add_test(NAME test_vocab_spm_bpe COMMAND test_vocab_spm_bpe)
set_tests_properties(test_vocab_spm_bpe PROPERTIES LABELS gate)
set_tests_properties(test_vocab_spm_bpe PROPERTIES SKIP_RETURN_CODE 77)

# --- EXPORT-BACKLOG.md item 4: Vocab's bos_token_id/eos_token_id/add_bos_token/add_eos_token support
#     (ALBERT/XLNet-style Unigram gap), synthetic fixture with a hand-traceable Viterbi segmentation ---
add_test(
    NAME gen_vocab_ugm_bos_eos_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_vocab_ugm_bos_eos_gguf.py
            ${LOOM_FIXTURE_DIR}/vocab_ugm_bos_eos_test.gguf
)
set_tests_properties(gen_vocab_ugm_bos_eos_gguf_fixture PROPERTIES LABELS ci)
set_tests_properties(gen_vocab_ugm_bos_eos_gguf_fixture PROPERTIES FIXTURES_SETUP vocab_ugm_bos_eos_gguf_fixture)

add_executable(test_vocab_ugm_bos_eos ci/test_vocab_ugm_bos_eos.cpp)
target_link_libraries(test_vocab_ugm_bos_eos PRIVATE loom::engine)
target_compile_definitions(test_vocab_ugm_bos_eos PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_vocab_ugm_bos_eos COMMAND test_vocab_ugm_bos_eos)
set_tests_properties(test_vocab_ugm_bos_eos PROPERTIES LABELS ci)
set_tests_properties(test_vocab_ugm_bos_eos PROPERTIES FIXTURES_REQUIRED vocab_ugm_bos_eos_gguf_fixture)

add_executable(test_ctc_decode ci/test_ctc_decode.cpp)
target_link_libraries(test_ctc_decode PRIVATE loom::engine)
add_test(NAME test_ctc_decode COMMAND test_ctc_decode)
set_tests_properties(test_ctc_decode PROPERTIES LABELS ci)

# --- Qwen3 roadmap item: Unicode subsystem (NFC normalize, \p{L}/\p{N} classification) backing BpeVocab ---
add_executable(test_unicode ci/test_unicode.cpp)
target_link_libraries(test_unicode PRIVATE loom::engine)
add_test(NAME test_unicode COMMAND test_unicode)
set_tests_properties(test_unicode PROPERTIES LABELS ci)

# --- Qwen3 roadmap item: BpeVocab (byte-level BPE, llama.cpp "gpt2" GGUF schema) against a small
#     synthetic fixture with hand-traceable merges (see tests/fixtures/make_bpe_vocab_gguf.py) ---
add_test(
    NAME gen_bpe_vocab_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_bpe_vocab_gguf.py
            ${LOOM_FIXTURE_DIR}/bpe_vocab_test.gguf
)
set_tests_properties(gen_bpe_vocab_gguf_fixture PROPERTIES LABELS ci)
set_tests_properties(gen_bpe_vocab_gguf_fixture PROPERTIES FIXTURES_SETUP bpe_vocab_gguf_fixture)

# Same tiny vocab/merges, different `tokenizer.ggml.pre` shapes -- exercises kGpt2Classic (item 4's
# BPE-pretokenizer-family generalization) and kWhitespacePunctExclude.
add_test(
    NAME gen_bpe_vocab_gpt2_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_bpe_vocab_gguf.py
            ${LOOM_FIXTURE_DIR}/bpe_vocab_gpt2_test.gguf --pre gpt-2
)
set_tests_properties(gen_bpe_vocab_gpt2_gguf_fixture PROPERTIES LABELS ci)
set_tests_properties(gen_bpe_vocab_gpt2_gguf_fixture PROPERTIES FIXTURES_SETUP bpe_vocab_gpt2_gguf_fixture)

add_test(
    NAME gen_bpe_vocab_poro_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_bpe_vocab_gguf.py
            ${LOOM_FIXTURE_DIR}/bpe_vocab_poro_test.gguf --pre poro-chat
)
set_tests_properties(gen_bpe_vocab_poro_gguf_fixture PROPERTIES LABELS ci)
set_tests_properties(gen_bpe_vocab_poro_gguf_fixture PROPERTIES FIXTURES_SETUP bpe_vocab_poro_gguf_fixture)

add_executable(test_bpe_vocab ci/test_bpe_vocab.cpp)
target_link_libraries(test_bpe_vocab PRIVATE loom::engine)
target_compile_definitions(test_bpe_vocab PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_bpe_vocab COMMAND test_bpe_vocab)
set_tests_properties(test_bpe_vocab PROPERTIES LABELS ci)
set_tests_properties(test_bpe_vocab PROPERTIES
    FIXTURES_REQUIRED "bpe_vocab_gguf_fixture;bpe_vocab_gpt2_gguf_fixture;bpe_vocab_poro_gguf_fixture")

# --- EXPORT-BACKLOG.md item 4: WordPieceVocab (BERT-family), synthetic fixture with hand-traceable
#     "##"-continuation split, punctuation isolation, [UNK] fallback, accents, CLS/SEP wrap ---
add_test(
    NAME gen_wordpiece_vocab_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_wordpiece_vocab_gguf.py
            ${LOOM_FIXTURE_DIR}/wordpiece_vocab_test.gguf
)
set_tests_properties(gen_wordpiece_vocab_gguf_fixture PROPERTIES LABELS ci)
set_tests_properties(gen_wordpiece_vocab_gguf_fixture PROPERTIES FIXTURES_SETUP wordpiece_vocab_gguf_fixture)

add_executable(test_wordpiece_vocab ci/test_wordpiece_vocab.cpp)
target_link_libraries(test_wordpiece_vocab PRIVATE loom::engine)
target_compile_definitions(test_wordpiece_vocab PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_wordpiece_vocab COMMAND test_wordpiece_vocab)
set_tests_properties(test_wordpiece_vocab PROPERTIES LABELS ci)
set_tests_properties(test_wordpiece_vocab PROPERTIES FIXTURES_REQUIRED wordpiece_vocab_gguf_fixture)

# --- ByT5-family byte-level tokenizer (ByteVocab), synthetic fixture with a small extra_ids count ---
add_test(
    NAME gen_byte_vocab_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_byte_vocab_gguf.py
            ${LOOM_FIXTURE_DIR}/byte_vocab_test.gguf
)
set_tests_properties(gen_byte_vocab_gguf_fixture PROPERTIES LABELS ci)
set_tests_properties(gen_byte_vocab_gguf_fixture PROPERTIES FIXTURES_SETUP byte_vocab_gguf_fixture)

add_executable(test_byte_vocab ci/test_byte_vocab.cpp)
target_link_libraries(test_byte_vocab PRIVATE loom::engine)
target_compile_definitions(test_byte_vocab PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_byte_vocab COMMAND test_byte_vocab)
set_tests_properties(test_byte_vocab PROPERTIES LABELS ci)
set_tests_properties(test_byte_vocab PROPERTIES FIXTURES_REQUIRED byte_vocab_gguf_fixture)

# --- SupertonicTTS grapheme front-end (SupertonicTextVectorizer), synthetic printable-ASCII fixtures.
#     The real-asset parity check is gate/test_supertonic_text_vectorizer.cpp; this covers what does NOT
#     need that checkout -- the inverse map, the derived vocab size, the default-language KV. It also
#     reads byte_vocab_test.gguf, to check the tag dispatch declines a file that isn't its own. ---
add_test(
    NAME gen_supertonic_vocab_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_supertonic_vocab_gguf.py
            ${LOOM_FIXTURE_DIR}/supertonic_vocab_test.gguf
            ${LOOM_FIXTURE_DIR}/supertonic_vocab_noninjective.gguf
)
set_tests_properties(gen_supertonic_vocab_gguf_fixture PROPERTIES LABELS ci)
set_tests_properties(gen_supertonic_vocab_gguf_fixture PROPERTIES FIXTURES_SETUP supertonic_vocab_gguf_fixture)

add_executable(test_supertonic_vocab ci/test_supertonic_vocab.cpp)
target_link_libraries(test_supertonic_vocab PRIVATE loom::engine)
target_compile_definitions(test_supertonic_vocab PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_supertonic_vocab COMMAND test_supertonic_vocab)
set_tests_properties(test_supertonic_vocab PROPERTIES LABELS ci)
set_tests_properties(test_supertonic_vocab PROPERTIES FIXTURES_REQUIRED "supertonic_vocab_gguf_fixture;byte_vocab_gguf_fixture")

# --- Qwen3 roadmap item: GQA (n_head_kv < n_head) regression test, verified against numpy before the
#     real Qwen3-0.6B-Base checkpoint (16 query / 8 KV heads) depends on ATTENTION's broadcast path ---
set(LOOM_E2E_GQA_REF_DIR ${LOOM_FIXTURE_DIR}/e2e_gqa_ref)

add_test(
    NAME gen_gqa_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_gqa_test_gguf.py
            ${LOOM_FIXTURE_DIR}/gqa_test.gguf
)
set_tests_properties(gen_gqa_gguf_fixture PROPERTIES LABELS ci)
set_tests_properties(gen_gqa_gguf_fixture PROPERTIES FIXTURES_SETUP gqa_gguf_fixture)

add_test(
    NAME gen_gqa_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/reference_forward_gqa.py
            ${LOOM_E2E_GQA_REF_DIR} --prompt 1 2 3 --n-new 4
)
set_tests_properties(gen_gqa_reference PROPERTIES LABELS ci)
set_tests_properties(gen_gqa_reference PROPERTIES FIXTURES_SETUP gqa_reference_fixture)

add_executable(test_e2e_gqa ci/test_e2e_gqa.cpp)
target_link_libraries(test_e2e_gqa PRIVATE loom::engine)
target_compile_definitions(test_e2e_gqa PRIVATE
    LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}"
    LOOM_TEST_REF_DIR="${LOOM_E2E_GQA_REF_DIR}"
)
add_test(NAME test_e2e_gqa COMMAND test_e2e_gqa)
set_tests_properties(test_e2e_gqa PROPERTIES LABELS ci)
set_tests_properties(test_e2e_gqa PROPERTIES FIXTURES_REQUIRED "gqa_gguf_fixture;gqa_reference_fixture")

# --- Real-model test: Qwen3-0.6B-Base, converted via tools/convert_qwen3/. Same "not regenerated at
#     ctest time, skip cleanly if the real checkpoint isn't prepared" pattern as test_e2e_conformer_ctc ---
add_executable(test_e2e_qwen3 gate/test_e2e_qwen3.cpp)
target_link_libraries(test_e2e_qwen3 PRIVATE loom::engine)
add_test(NAME test_e2e_qwen3 COMMAND test_e2e_qwen3)
set_tests_properties(test_e2e_qwen3 PROPERTIES LABELS gate)
set_tests_properties(test_e2e_qwen3 PROPERTIES SKIP_RETURN_CODE 77)

# --- Generic-converter proof of concept, round 2: real Qwen3-0.6B-Base through the same, unmodified
#     converter/op-mapping table the toy LLM used (tools/convert_generic/) -- see BACKLOG.md ---
add_executable(test_e2e_qwen3_generic gate/test_e2e_qwen3_generic.cpp)
target_link_libraries(test_e2e_qwen3_generic PRIVATE loom::engine)
add_test(NAME test_e2e_qwen3_generic COMMAND test_e2e_qwen3_generic)
set_tests_properties(test_e2e_qwen3_generic PROPERTIES LABELS gate)
set_tests_properties(test_e2e_qwen3_generic PROPERTIES SKIP_RETURN_CODE 77)

# --- Quantized-weight proof of concept: same real Qwen3-0.6B-Base checkpoint, matmul weights quantized
#     to Q8_0 by tools/quantize/quantize_gguf_q8_0.py -- see BACKLOG.md's "quantized weight support"
#     milestone. Same skip-cleanly-if-not-prepared pattern as the other real-model tests above.
add_executable(test_e2e_qwen3_q8_0 gate/test_e2e_qwen3_q8_0.cpp)
target_link_libraries(test_e2e_qwen3_q8_0 PRIVATE loom::engine)
add_test(NAME test_e2e_qwen3_q8_0 COMMAND test_e2e_qwen3_q8_0)
set_tests_properties(test_e2e_qwen3_q8_0 PROPERTIES LABELS gate)
set_tests_properties(test_e2e_qwen3_q8_0 PROPERTIES SKIP_RETURN_CODE 77)

# --- Parakeet-TDT roadmap, step 1: the LSTM-step composite-topology pattern (MUL_MAT/ADD/VIEW/SIGMOID/
#     TANH/MUL -- no monolithic LSTM primitive), verified bit-exact against an independent numpy reference
#     on a tiny synthetic fixture before it's trusted inside the real TdtDecoder driver. Fully synthetic,
#     procedurally generated at ctest time like the other toy fixtures -- not skip-if-missing. ---
set(LOOM_LSTM_STEP_REF_DIR ${LOOM_FIXTURE_DIR}/lstm_step_ref)

add_test(
    NAME gen_lstm_step_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_lstm_step_gguf.py
            ${LOOM_FIXTURE_DIR}/lstm_step_h.gguf ${LOOM_FIXTURE_DIR}/lstm_step_c.gguf
)
set_tests_properties(gen_lstm_step_gguf_fixture PROPERTIES LABELS ci)
set_tests_properties(gen_lstm_step_gguf_fixture PROPERTIES FIXTURES_SETUP lstm_step_gguf_fixture)

add_test(
    NAME gen_lstm_step_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/reference_lstm_step.py
            ${LOOM_LSTM_STEP_REF_DIR}
)
set_tests_properties(gen_lstm_step_reference PROPERTIES LABELS ci)
set_tests_properties(gen_lstm_step_reference PROPERTIES FIXTURES_SETUP lstm_step_reference_fixture)

add_executable(test_lstm_step ci/test_lstm_step.cpp)
target_link_libraries(test_lstm_step PRIVATE loom::engine)
target_compile_definitions(test_lstm_step PRIVATE
    LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}"
    LOOM_TEST_REF_DIR="${LOOM_LSTM_STEP_REF_DIR}"
)
add_test(NAME test_lstm_step COMMAND test_lstm_step)
set_tests_properties(test_lstm_step PROPERTIES LABELS ci)
set_tests_properties(test_lstm_step PROPERTIES FIXTURES_REQUIRED "lstm_step_gguf_fixture;lstm_step_reference_fixture")

# --- Kokoro/StyleTTS2 roadmap: the host-side duration-based frame-expansion (loom::predict_durations/
#     loom::expand_by_duration -- KModel.forward_with_tokens's sigmoid+sum/round/clamp/repeat_interleave
#     formula), verified against an independent numpy reference on a tiny synthetic fixture. Pure host
#     arithmetic, no GGUF/ggml graph involved at all -- fully synthetic, not skip-if-missing. ---
set(LOOM_DURATION_ALIGNER_REF_DIR ${LOOM_FIXTURE_DIR}/duration_aligner_ref)

add_test(
    NAME gen_duration_aligner_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/reference_duration_aligner.py
            ${LOOM_DURATION_ALIGNER_REF_DIR}
)
set_tests_properties(gen_duration_aligner_reference PROPERTIES LABELS ci)
set_tests_properties(gen_duration_aligner_reference PROPERTIES FIXTURES_SETUP duration_aligner_reference_fixture)

add_executable(test_duration_aligner ci/test_duration_aligner.cpp)
target_link_libraries(test_duration_aligner PRIVATE loom::engine)
target_compile_definitions(test_duration_aligner PRIVATE
    LOOM_TEST_REF_DIR="${LOOM_DURATION_ALIGNER_REF_DIR}"
)
add_test(NAME test_duration_aligner COMMAND test_duration_aligner)
set_tests_properties(test_duration_aligner PROPERTIES LABELS ci)
set_tests_properties(test_duration_aligner PROPERTIES FIXTURES_REQUIRED "duration_aligner_reference_fixture")

# --- Parakeet TDT and RNN-T, decoded by their own embedded Lua driver and compared against NeMo's own
#     `model.transcribe()` output for samples/jfk.wav (BACKLOG.md P4.0.17 step 2).
#
#     This one test replaces six: test_{tdt,rnnt}_decoder (loom::TdtDecoder against numpy references on
#     synthetic fixtures), test_e2e_parakeet_{tdt,rnnt} (the same decoder over six bespoke GGUFs) and
#     test_e2e_parakeet_{tdt,rnnt}_mil_export (the encoder-only MIL artifact, whose `main_topology` no
#     longer exists). None of them can be kept: the C++ decoder they exercise is gone, and the artifact
#     is now one GGUF whose driver decodes. The oracle is stronger for it -- NeMo itself on real speech,
#     rather than a hand-rolled reimplementation on a synthetic waveform that, for RNN-T, decoded to an
#     empty token list. ---
add_executable(test_e2e_parakeet_lua_driver gate/test_e2e_parakeet_lua_driver.cpp)
target_link_libraries(test_e2e_parakeet_lua_driver PRIVATE loom::engine)
add_test(NAME test_e2e_parakeet_lua_driver COMMAND test_e2e_parakeet_lua_driver)
set_tests_properties(test_e2e_parakeet_lua_driver PROPERTIES LABELS gate)
set_tests_properties(test_e2e_parakeet_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- The same transducer template, a different loader: GigaAM v3 (an HF `trust_remote_code` directory,
#     not a `.nemo` archive) through the identical embedded driver, against the checkpoint's own greedy
#     hypothesis for samples/jfk.wav (BACKLOG.md P4.2). Also the only check on the two things that
#     checkpoint needed before it would trace -- a rewritten mel frontend and a one-layer prediction
#     network whose cell the driver addresses by index. ---
add_executable(test_e2e_gigaam_lua_driver gate/test_e2e_gigaam_lua_driver.cpp)
target_link_libraries(test_e2e_gigaam_lua_driver PRIVATE loom::engine)
add_test(NAME test_e2e_gigaam_lua_driver COMMAND test_e2e_gigaam_lua_driver)
set_tests_properties(test_e2e_gigaam_lua_driver PROPERTIES LABELS gate)
set_tests_properties(test_e2e_gigaam_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro's CustomAlbert (PL-BERT) topology (tools/convert_kokoro/
#     convert_kokoro_albert.py) against a hand-rolled pure-PyTorch reference -- the first, isolated
#     verification step of Kokoro's full-continuous-build effort (see BACKLOG.md). ---
add_executable(test_e2e_kokoro_albert gate/test_e2e_kokoro_albert.cpp)
target_link_libraries(test_e2e_kokoro_albert PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_albert COMMAND test_e2e_kokoro_albert)
set_tests_properties(test_e2e_kokoro_albert PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_albert PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro's TextEncoder (CNN topology + the new loom::BiLstmStepper
#     host driver for its trailing bidirectional LSTM) against a hand-rolled pure-PyTorch reference. ---
add_executable(test_e2e_kokoro_text_encoder gate/test_e2e_kokoro_text_encoder.cpp)
target_link_libraries(test_e2e_kokoro_text_encoder PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_text_encoder COMMAND test_e2e_kokoro_text_encoder)
set_tests_properties(test_e2e_kokoro_text_encoder PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_text_encoder PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro's ProsodyPredictor duration-prediction half (DurationEncoder's
#     interleaved BiLSTM+AdaLayerNorm blocks -> top lstm -> duration_proj) against a hand-rolled
#     pure-PyTorch reference. ---
add_executable(test_e2e_kokoro_duration_predictor gate/test_e2e_kokoro_duration_predictor.cpp)
target_link_libraries(test_e2e_kokoro_duration_predictor PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_duration_predictor COMMAND test_e2e_kokoro_duration_predictor)
set_tests_properties(test_e2e_kokoro_duration_predictor PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_duration_predictor PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: a single AdainResBlk1d instance (predictor.F0.0, the simplest case --
#     no learned shortcut, no upsample) against a hand-rolled pure-PyTorch reference -- first isolated
#     verification step for Kokoro's F0Ntrain assembly. ---
add_executable(test_e2e_kokoro_f0_block0 gate/test_e2e_kokoro_f0_block0.cpp)
target_link_libraries(test_e2e_kokoro_f0_block0 PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_f0_block0 COMMAND test_e2e_kokoro_f0_block0)
set_tests_properties(test_e2e_kokoro_f0_block0 PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_f0_block0 PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: predictor.F0.1, the UPSAMPLING AdainResBlk1d instance (dim_in=512,
#     dim_out=256, learned conv1x1 shortcut + upsample=True) -- exercises the depthwise-ConvTranspose1d
#     "pool" composition and the learned shortcut together for the first time. ---
add_executable(test_e2e_kokoro_f0_block1 gate/test_e2e_kokoro_f0_block1.cpp)
target_link_libraries(test_e2e_kokoro_f0_block1 PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_f0_block1 COMMAND test_e2e_kokoro_f0_block1)
set_tests_properties(test_e2e_kokoro_f0_block1 PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_f0_block1 PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: the FULL ProsodyPredictor.F0Ntrain assembly (shared BiLSTM 640->512 +
#     two independent 3-block AdainResBlk1d stacks + F0_proj/N_proj) against a hand-rolled pure-PyTorch
#     reference -- ties together every Kokoro F0Ntrain piece verified so far. ---
add_executable(test_e2e_kokoro_f0n gate/test_e2e_kokoro_f0n.cpp)
target_link_libraries(test_e2e_kokoro_f0n PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_f0n COMMAND test_e2e_kokoro_f0n)
set_tests_properties(test_e2e_kokoro_f0n PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_f0n PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro Generator's forward/inverse STFT (istftnet.py's TorchSTFT),
#     against real torch.stft/torch.istft -- no real checkpoint weights involved (constant DFT/window
#     kernels only), but still gated on GGUF/reference files (needs real torch to generate). ---
add_executable(test_e2e_kokoro_stft gate/test_e2e_kokoro_stft.cpp)
target_link_libraries(test_e2e_kokoro_stft PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_stft COMMAND test_e2e_kokoro_stft)
set_tests_properties(test_e2e_kokoro_stft PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_stft PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro Generator's NSF harmonic source (istftnet.py's SineGen +
#     SourceModuleHnNSF), against a hand-rolled real-PyTorch reference. No real checkpoint weights
#     involved (l_linear is a fixed synthetic draw shared between conversion+reference scripts). ---
add_executable(test_e2e_kokoro_sinegen gate/test_e2e_kokoro_sinegen.cpp)
target_link_libraries(test_e2e_kokoro_sinegen PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_sinegen COMMAND test_e2e_kokoro_sinegen)
set_tests_properties(test_e2e_kokoro_sinegen PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_sinegen PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro Generator's AdaINResBlock1 (Snake-activation dilated resblock,
#     distinct from predictor.F0/N's AdainResBlk1d), against a hand-rolled PyTorch reference on a small
#     synthetic (checkpoint-independent) instance -- same precedent as VITS's test_hifigan_generator. ---
add_executable(test_e2e_kokoro_adainresblock1 gate/test_e2e_kokoro_adainresblock1.cpp)
target_link_libraries(test_e2e_kokoro_adainresblock1 PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_adainresblock1 COMMAND test_e2e_kokoro_adainresblock1)
set_tests_properties(test_e2e_kokoro_adainresblock1 PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_adainresblock1 PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro's full Generator "core" (upsample stack + noise path +
#     resblocks + conv_post + inverse STFT), against a hand-rolled PyTorch reference on a small synthetic
#     (real checkpoint shapes, random weights) instance. ---
add_executable(test_e2e_kokoro_generator gate/test_e2e_kokoro_generator.cpp)
target_link_libraries(test_e2e_kokoro_generator PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_generator COMMAND test_e2e_kokoro_generator)
set_tests_properties(test_e2e_kokoro_generator PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_generator PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro's Decoder "core" (F0_conv/N_conv + encode/decode AdainResBlk1d
#     stack, everything except the final Generator call), against a hand-rolled PyTorch reference on a
#     small synthetic (real checkpoint shapes, random weights) instance. ---
add_executable(test_e2e_kokoro_decoder_core gate/test_e2e_kokoro_decoder_core.cpp)
target_link_libraries(test_e2e_kokoro_decoder_core PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_decoder_core COMMAND test_e2e_kokoro_decoder_core)
set_tests_properties(test_e2e_kokoro_decoder_core PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_decoder_core PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro's bert_encoder (Linear(768,512) + transpose), against the real
#     checkpoint's own weights -- exercises the CustomAlbert-output (time-major) -> DurationEncoder-input
#     (channel-first) axis-convention crossing. ---
add_executable(test_e2e_kokoro_bert_encoder gate/test_e2e_kokoro_bert_encoder.cpp)
target_link_libraries(test_e2e_kokoro_bert_encoder PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_bert_encoder COMMAND test_e2e_kokoro_bert_encoder)
set_tests_properties(test_e2e_kokoro_bert_encoder PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_bert_encoder PROPERTIES SKIP_RETURN_CODE 77)

# --- Real-model structural smoke test: VITS (piper en-GB "miro"), converted via
#     tools/convert_piper_vits/convert_vits.py. Builds every declared topology with dummy inputs and
#     checks for finite output -- NOT yet a numerical-correctness check (that's still to come once the
#     host driver exists). Same skip-cleanly-if-not-prepared pattern as the other real-model tests. ---
add_executable(test_e2e_vits_smoke gate/test_e2e_vits_smoke.cpp)
target_link_libraries(test_e2e_vits_smoke PRIVATE loom::engine)
add_test(NAME test_e2e_vits_smoke COMMAND test_e2e_vits_smoke)
set_tests_properties(test_e2e_vits_smoke PROPERTIES LABELS gate)
set_tests_properties(test_e2e_vits_smoke PROPERTIES SKIP_RETURN_CODE 77)

# --- Same structural-smoke role as test_e2e_vits_smoke above, for the (in-progress, see BACKLOG.md)
#     MIL-traced Kokoro "decoder_vocoder" phase (Decoder core + SineGen + STFT + Generator, the single
#     riskiest chunk of the model). ---
add_executable(test_e2e_kokoro_mil_decoder_vocoder_smoke gate/test_e2e_kokoro_mil_decoder_vocoder_smoke.cpp)
target_link_libraries(test_e2e_kokoro_mil_decoder_vocoder_smoke PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_mil_decoder_vocoder_smoke COMMAND test_e2e_kokoro_mil_decoder_vocoder_smoke)
set_tests_properties(test_e2e_kokoro_mil_decoder_vocoder_smoke PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_mil_decoder_vocoder_smoke PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check for the same MIL-traced Kokoro "decoder_vocoder" topology, against a
#     real pure-PyTorch reference (reference_forward_kokoro_decoder_vocoder_mil.py) rather than the smoke
#     test's zero-filled/structural-only check above. ---
add_executable(test_e2e_kokoro_mil_decoder_vocoder_reference gate/test_e2e_kokoro_mil_decoder_vocoder_reference.cpp)
target_link_libraries(test_e2e_kokoro_mil_decoder_vocoder_reference PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_mil_decoder_vocoder_reference COMMAND test_e2e_kokoro_mil_decoder_vocoder_reference)
set_tests_properties(test_e2e_kokoro_mil_decoder_vocoder_reference PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_mil_decoder_vocoder_reference PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check for the MIL-traced Kokoro "albert_bert_encoder" combined topology
#     (CustomAlbert + bert_encoder, export_kokoro_mil.py's AlbertBertEncoderWrapper) against a
#     hand-rolled pure-PyTorch reference. ---
add_executable(test_e2e_kokoro_mil_albert_bert_encoder_reference gate/test_e2e_kokoro_mil_albert_bert_encoder_reference.cpp)
target_link_libraries(test_e2e_kokoro_mil_albert_bert_encoder_reference PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_mil_albert_bert_encoder_reference COMMAND test_e2e_kokoro_mil_albert_bert_encoder_reference)
set_tests_properties(test_e2e_kokoro_mil_albert_bert_encoder_reference PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_mil_albert_bert_encoder_reference PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: the deterministic `stats` (TextEncoder) topology against a real
#     PyTorch forward pass of piper's own models.TextEncoder, using REAL phonemized text (T=62, real
#     emb_rel_k/v pad branch exercised) -- see test_e2e_vits_stats_reference.cpp's own header comment. ---
add_executable(test_e2e_vits_stats_reference gate/test_e2e_vits_stats_reference.cpp)
target_link_libraries(test_e2e_vits_stats_reference PRIVATE loom::engine)
add_test(NAME test_e2e_vits_stats_reference COMMAND test_e2e_vits_stats_reference)
set_tests_properties(test_e2e_vits_stats_reference PROPERTIES LABELS gate)
set_tests_properties(test_e2e_vits_stats_reference PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: the flow+vocoder topology against a real PyTorch forward pass, with a
#     FIXED (non-random) z_p -- isolates the coupling-flow+vocoder wiring from any RNG-related
#     uncertainty in the full stochastic pipeline. See test_e2e_vits_flow_vocoder_reference.cpp. ---
add_executable(test_e2e_vits_flow_vocoder_reference gate/test_e2e_vits_flow_vocoder_reference.cpp)
target_link_libraries(test_e2e_vits_flow_vocoder_reference PRIVATE loom::engine)
add_test(NAME test_e2e_vits_flow_vocoder_reference COMMAND test_e2e_vits_flow_vocoder_reference)
set_tests_properties(test_e2e_vits_flow_vocoder_reference PROPERTIES LABELS gate)
set_tests_properties(test_e2e_vits_flow_vocoder_reference PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: the logw (TextEncoder+SDP reverse) topology against a real PyTorch
#     forward pass, with FIXED (externally-injected) SDP noise -- isolates the real spline-flow assembly
#     from RNG-related uncertainty. See test_e2e_vits_logw_reference.cpp. ---
add_executable(test_e2e_vits_logw_reference gate/test_e2e_vits_logw_reference.cpp)
target_link_libraries(test_e2e_vits_logw_reference PRIVATE loom::engine)
add_test(NAME test_e2e_vits_logw_reference COMMAND test_e2e_vits_logw_reference)
set_tests_properties(test_e2e_vits_logw_reference PROPERTIES LABELS gate)
set_tests_properties(test_e2e_vits_logw_reference PROPERTIES SKIP_RETURN_CODE 77)




# --- loom::karras_schedule/adpm2_sample had two tests here -- the discretization math against a numpy
#     reference with a toy denoiser, and the full loop around the real Transformer1d network. Both went
#     with `style_diffusion_sampler.{h,cpp}` (P4.0.8 follow-up): StyleTTS2's ADPM2 sampler is
#     styletts2_driver/02_style_diffusion.lua now, and the C++ class had no consumer but these two.
#
#     What that costs, stated rather than glossed: those were EXACT comparisons, and the Lua sampler
#     cannot have one. They replayed a fixture's ancestral-noise draws through an injectable
#     `GaussianSampleFn`; the driver draws its noise from `loom.gaussian_array`, which has no such seam,
#     so nothing bit-compares against a numpy reference. The Lua loop's own ground truth is the frozen
#     full-pipeline waveform in fixtures/legacy_driver_reference/ (test_e2e_styletts2_mil_lua_driver),
#     which the retired C++ driver produced THROUGH this sampler -- so the check moved rather than
#     vanished, at a looser bound and one level up. The denoiser network keeps its own tests below. ---

# --- Real-checkpoint test: StyleTTS2's style-diffusion denoiser network (Transformer1d) against a
#     hand-rolled PyTorch reference, real weights + synthetic driving embedding. See
#     tools/convert_styletts2/convert_styletts2_diffusion.py. Skips cleanly if env vars aren't set. ---
add_executable(test_e2e_styletts2_diffusion_net gate/test_e2e_styletts2_diffusion_net.cpp)
target_link_libraries(test_e2e_styletts2_diffusion_net PRIVATE loom::engine)
add_test(NAME test_e2e_styletts2_diffusion_net COMMAND test_e2e_styletts2_diffusion_net)
set_tests_properties(test_e2e_styletts2_diffusion_net PROPERTIES LABELS gate)
set_tests_properties(test_e2e_styletts2_diffusion_net PROPERTIES SKIP_RETURN_CODE 77)

# --- MIL-traced StyleTTS2 export (export_styletts2_mil.py, styletts2_mil.gguf) numerical checks: "albert"
#     (CustomAlbert alone) against tools/convert_kokoro's own already-verified albert_forward run against
#     StyleTTS2's own checkpoint weights; "decoder_vocoder" against a direct reuse of export_kokoro_mil
#     .py's own DecoderVocoderWrapper (same istftnet classes, StyleTTS2's own weights); "diffusion" (a real
#     trace of Modules/diffusion/modules.py's Transformer1d.run(), superseding
#     convert_styletts2_diffusion.py's own hand-derived topology) against the SAME reference fixtures
#     test_e2e_styletts2_diffusion_net.cpp already uses. Skips cleanly if env vars aren't set. ---
add_executable(test_e2e_styletts2_mil_albert_reference gate/test_e2e_styletts2_mil_albert_reference.cpp)
target_link_libraries(test_e2e_styletts2_mil_albert_reference PRIVATE loom::engine)
add_test(NAME test_e2e_styletts2_mil_albert_reference COMMAND test_e2e_styletts2_mil_albert_reference)
set_tests_properties(test_e2e_styletts2_mil_albert_reference PROPERTIES LABELS gate)
set_tests_properties(test_e2e_styletts2_mil_albert_reference PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_e2e_styletts2_mil_decoder_vocoder_reference gate/test_e2e_styletts2_mil_decoder_vocoder_reference.cpp)
target_link_libraries(test_e2e_styletts2_mil_decoder_vocoder_reference PRIVATE loom::engine)
add_test(NAME test_e2e_styletts2_mil_decoder_vocoder_reference COMMAND test_e2e_styletts2_mil_decoder_vocoder_reference)
set_tests_properties(test_e2e_styletts2_mil_decoder_vocoder_reference PROPERTIES LABELS gate)
set_tests_properties(test_e2e_styletts2_mil_decoder_vocoder_reference PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_e2e_styletts2_mil_diffusion_reference gate/test_e2e_styletts2_mil_diffusion_reference.cpp)
target_link_libraries(test_e2e_styletts2_mil_diffusion_reference PRIVATE loom::engine)
add_test(NAME test_e2e_styletts2_mil_diffusion_reference COMMAND test_e2e_styletts2_mil_diffusion_reference)
set_tests_properties(test_e2e_styletts2_mil_diffusion_reference PROPERTIES LABELS gate)
set_tests_properties(test_e2e_styletts2_mil_diffusion_reference PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: ConvNextBlock (replicate-pad composition), verified against the REAL
#     nn.Module's own forward pass (real package importable in this venv). See
#     tools/convert_supertonic/convert_supertonic_convnext.py. Skips cleanly if env vars aren't set. ---
add_executable(test_e2e_supertonic_convnext gate/test_e2e_supertonic_convnext.cpp)
target_link_libraries(test_e2e_supertonic_convnext PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_convnext COMMAND test_e2e_supertonic_convnext)
set_tests_properties(test_e2e_supertonic_convnext PROPERTIES LABELS gate)
set_tests_properties(test_e2e_supertonic_convnext PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: VFTimeEncoder (sinusoidal time embedding + Mish MLP), verified against
#     the real vector_estimator.pt's own time_encoder. See convert_supertonic_vftime.py. ---
add_executable(test_e2e_supertonic_vftime gate/test_e2e_supertonic_vftime.cpp)
target_link_libraries(test_e2e_supertonic_vftime PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_vftime COMMAND test_e2e_supertonic_vftime)
set_tests_properties(test_e2e_supertonic_vftime PROPERTIES LABELS gate)
set_tests_properties(test_e2e_supertonic_vftime PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: StyleEncoderCrossAttention (style-token-pooling cross-attention),
#     verified against the real dp-style-encoder.pt's own style_token_layer. See
#     convert_supertonic_style_attn.py. ---
add_executable(test_e2e_supertonic_style_attn gate/test_e2e_supertonic_style_attn.cpp)
target_link_libraries(test_e2e_supertonic_style_attn PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_style_attn COMMAND test_e2e_supertonic_style_attn)
set_tests_properties(test_e2e_supertonic_style_attn PROPERTIES LABELS gate)
set_tests_properties(test_e2e_supertonic_style_attn PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: confirms VITS's own REL_POS_ATTENTION_SHAW primitive family is directly
#     reusable for SupertonicTTS's MultiHeadRelativeAttention (same Shaw et al. mechanism, different
#     window_size/channels). See convert_supertonic_relpos_attn.py. ---
add_executable(test_e2e_supertonic_relpos_attn gate/test_e2e_supertonic_relpos_attn.cpp)
target_link_libraries(test_e2e_supertonic_relpos_attn PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_relpos_attn COMMAND test_e2e_supertonic_relpos_attn)
set_tests_properties(test_e2e_supertonic_relpos_attn PROPERTIES LABELS gate)
set_tests_properties(test_e2e_supertonic_relpos_attn PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: full DurationPredictor sub-model (DPTextEncoder + MLP head), the FIRST
#     complete coherent sub-model in this effort, verified against the real duration_predictor.pt. See
#     convert_supertonic_dp.py. ---
add_executable(test_e2e_supertonic_dp gate/test_e2e_supertonic_dp.cpp)
target_link_libraries(test_e2e_supertonic_dp PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_dp COMMAND test_e2e_supertonic_dp)
set_tests_properties(test_e2e_supertonic_dp PROPERTIES LABELS gate)
set_tests_properties(test_e2e_supertonic_dp PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: TTLStyleEncoder + TTLTextEncoder (exercises SpeechPromptedCrossAttention
#     for the first time), verified against the real ttl-style-encoder.pt/text_encoder.pt. See
#     convert_supertonic_ttl_text.py. ---
add_executable(test_e2e_supertonic_ttl_text gate/test_e2e_supertonic_ttl_text.cpp)
target_link_libraries(test_e2e_supertonic_ttl_text PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_ttl_text COMMAND test_e2e_supertonic_ttl_text)
set_tests_properties(test_e2e_supertonic_ttl_text PROPERTIES LABELS gate)
set_tests_properties(test_e2e_supertonic_ttl_text PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: VFTextCrossAttention (fractional RoPE) + VFStyleCrossAttention, verified
#     against the real vector_estimator.pt's own text_attn[0]/style_attn[0]. See
#     convert_supertonic_vf_attn.py. ---
add_executable(test_e2e_supertonic_vf_attn gate/test_e2e_supertonic_vf_attn.cpp)
target_link_libraries(test_e2e_supertonic_vf_attn PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_vf_attn COMMAND test_e2e_supertonic_vf_attn)
set_tests_properties(test_e2e_supertonic_vf_attn PROPERTIES LABELS gate)
set_tests_properties(test_e2e_supertonic_vf_attn PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: the FULL VectorFieldEstimator (biggest single assembly in this effort),
#     verified against the real vector_estimator.pt, ONE compute_velocity call. See
#     convert_supertonic_vfe.py. ---
add_executable(test_e2e_supertonic_vfe gate/test_e2e_supertonic_vfe.cpp)
target_link_libraries(test_e2e_supertonic_vfe PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_vfe COMMAND test_e2e_supertonic_vfe)
set_tests_properties(test_e2e_supertonic_vfe PROPERTIES LABELS gate)
set_tests_properties(test_e2e_supertonic_vfe PROPERTIES SKIP_RETURN_CODE 77)

# --- The FULL CFM Euler sampling loop (loom::cfm_euler_sample + the real VectorFieldEstimator) was
#     checked here against a Python port calling vector_estimator.pt's own .solve() in a loop. Retired
#     with `cfm_euler_sampler.{h,cpp}` (P4.0.8 follow-up): the loop is a `FlowMatchingSampler`
#     component emitted into supertonic_driver/ now, and the C++ function had no consumer but this.
#     Unlike StyleTTS2's ADPM2 the loop is deterministic given z0, so what replaces it is a genuine
#     equivalent rather than a looser stand-in: test_e2e_supertonic_mil_lua_driver runs the Lua sampler
#     over the real vfe topology and compares the whole pipeline against a frozen waveform at 1e-2. The
#     per-phase check of the estimator itself is unaffected -- test_e2e_supertonic_vfe above. ---

# --- SupertonicTTS v2 roadmap: SpeechDecoder (folded BatchNorm, causal ConvNeXt, direct waveform
#     emission, the FINAL pipeline stage), verified against the real vocoder.pt. See
#     convert_supertonic_decoder.py. ---
add_executable(test_e2e_supertonic_decoder gate/test_e2e_supertonic_decoder.cpp)
target_link_libraries(test_e2e_supertonic_decoder PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_decoder COMMAND test_e2e_supertonic_decoder)
set_tests_properties(test_e2e_supertonic_decoder PROPERTIES LABELS gate)
set_tests_properties(test_e2e_supertonic_decoder PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS's real TextVectorizer (unicode codepoint-map tokenizer), against the real
#     unicode_indexer.json asset -- see tools/convert_supertonic/convert_supertonic_text_vectorizer.py.
#     Skips cleanly if LOOM_SUPERTONIC_TEXT_VECTORIZER_GGUF isn't set. ---
add_executable(test_supertonic_text_vectorizer gate/test_supertonic_text_vectorizer.cpp)
target_link_libraries(test_supertonic_text_vectorizer PRIVATE loom::engine)
add_test(NAME test_supertonic_text_vectorizer COMMAND test_supertonic_text_vectorizer)
set_tests_properties(test_supertonic_text_vectorizer PROPERTIES LABELS gate)
set_tests_properties(test_supertonic_text_vectorizer PROPERTIES SKIP_RETURN_CODE 77)

# --- Matcha-TTS roadmap: TextEncoder (ConvReluNorm prenet, partial-rotary NeoX RoPE self-attention,
#     per-token DurationPredictor) against the real matcha.models.components.text_encoder.TextEncoder
#     module run directly. See tools/convert_matcha/convert_matcha_text_encoder.py. ---
add_executable(test_e2e_matcha_text_encoder gate/test_e2e_matcha_text_encoder.cpp)
target_link_libraries(test_e2e_matcha_text_encoder PRIVATE loom::engine)
add_test(NAME test_e2e_matcha_text_encoder COMMAND test_e2e_matcha_text_encoder)
set_tests_properties(test_e2e_matcha_text_encoder PROPERTIES LABELS gate)
set_tests_properties(test_e2e_matcha_text_encoder PROPERTIES SKIP_RETURN_CODE 77)

# --- Matcha-TTS roadmap: Decoder U-Net (the CFM estimator -- GROUP_NORM, one real downsample/upsample,
#     skip connections, SnakeBeta-FeedForward BasicTransformerBlocks) against the real
#     matcha.models.components.decoder.Decoder module run directly. See
#     tools/convert_matcha/convert_matcha_decoder.py. ---
add_executable(test_e2e_matcha_decoder gate/test_e2e_matcha_decoder.cpp)
target_link_libraries(test_e2e_matcha_decoder PRIVATE loom::engine)
add_test(NAME test_e2e_matcha_decoder COMMAND test_e2e_matcha_decoder)
set_tests_properties(test_e2e_matcha_decoder PROPERTIES LABELS gate)
set_tests_properties(test_e2e_matcha_decoder PROPERTIES SKIP_RETURN_CODE 77)

# --- Matcha-TTS roadmap: the real HiFi-GAN v1 vocoder (resblock="1", 4 upsample stages -- a different
#     topology shape from VITS's own resblock="2"/3-stage HiFi-GAN) against the real
#     matcha.hifigan.models.Generator module run directly. See
#     tools/convert_matcha/convert_matcha_vocoder.py. ---
add_executable(test_e2e_matcha_vocoder gate/test_e2e_matcha_vocoder.cpp)
target_link_libraries(test_e2e_matcha_vocoder PRIVATE loom::engine)
add_test(NAME test_e2e_matcha_vocoder COMMAND test_e2e_matcha_vocoder)
set_tests_properties(test_e2e_matcha_vocoder PROPERTIES LABELS gate)
set_tests_properties(test_e2e_matcha_vocoder PROPERTIES SKIP_RETURN_CODE 77)

# --- Procedural generalization: LoomLuaBridge's host-math bindings (loom.range/causal_mask/zero_mask/
#     argmax_row) and load_script()/call() plumbing, in isolation (no GGUF needed). See
#     LOOM_PROCEDURAL_GENERALIZATION.md / LOOM_MIL_CONVERSION.md and BACKLOG.md's dated entry. ---
add_executable(test_lua_bridge ci/test_lua_bridge.cpp)
target_link_libraries(test_lua_bridge PRIVATE loom::engine)
add_test(NAME test_lua_bridge COMMAND test_lua_bridge)
set_tests_properties(test_lua_bridge PROPERTIES LABELS ci)


# --- Procedural generalization: SupertonicTTS's deterministic Euler-ODE (CFM) sampling loop, ported to
#     Lua (tools/convert_supertonic/supertonic_driver.lua) and compared against the existing
#     loom::SupertonicDriver on the real checkpoint. See LOOM_PROCEDURAL_GENERALIZATION.md /
#     LOOM_MIL_CONVERSION.md and BACKLOG.md's dated entry. ---
add_executable(test_e2e_supertonic_lua_driver gate/test_e2e_supertonic_lua_driver.cpp)
target_link_libraries(test_e2e_supertonic_lua_driver PRIVATE loom::engine)
target_compile_definitions(test_e2e_supertonic_lua_driver PRIVATE LOOM_LEGACY_REF_DIR="${LOOM_LEGACY_REF_DIR}")
add_test(NAME test_e2e_supertonic_lua_driver COMMAND test_e2e_supertonic_lua_driver)
set_tests_properties(test_e2e_supertonic_lua_driver PROPERTIES LABELS gate)
set_tests_properties(test_e2e_supertonic_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness checks for the MIL-traced SupertonicTTS topologies (export_supertonic_mil.py,
#     part of supertonic_mil.gguf) against real-module reference fixtures. Skips cleanly if env vars
#     aren't set. ---
add_executable(test_e2e_supertonic_mil_dp gate/test_e2e_supertonic_mil_dp.cpp)
target_link_libraries(test_e2e_supertonic_mil_dp PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_mil_dp COMMAND test_e2e_supertonic_mil_dp)
set_tests_properties(test_e2e_supertonic_mil_dp PROPERTIES LABELS gate)
set_tests_properties(test_e2e_supertonic_mil_dp PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_e2e_supertonic_mil_ttl_text gate/test_e2e_supertonic_mil_ttl_text.cpp)
target_link_libraries(test_e2e_supertonic_mil_ttl_text PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_mil_ttl_text COMMAND test_e2e_supertonic_mil_ttl_text)
set_tests_properties(test_e2e_supertonic_mil_ttl_text PROPERTIES LABELS gate)
set_tests_properties(test_e2e_supertonic_mil_ttl_text PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_e2e_supertonic_mil_vfe gate/test_e2e_supertonic_mil_vfe.cpp)
target_link_libraries(test_e2e_supertonic_mil_vfe PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_mil_vfe COMMAND test_e2e_supertonic_mil_vfe)
set_tests_properties(test_e2e_supertonic_mil_vfe PROPERTIES LABELS gate)
set_tests_properties(test_e2e_supertonic_mil_vfe PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_e2e_supertonic_mil_decoder gate/test_e2e_supertonic_mil_decoder.cpp)
target_link_libraries(test_e2e_supertonic_mil_decoder PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_mil_decoder COMMAND test_e2e_supertonic_mil_decoder)
set_tests_properties(test_e2e_supertonic_mil_decoder PROPERTIES LABELS gate)
set_tests_properties(test_e2e_supertonic_mil_decoder PROPERTIES SKIP_RETURN_CODE 77)

# --- The same two text topologies, but at a REAL sentence's length against the real Python pipeline's
#     own unpadded answer for it (BACKLOG.md P4.6). The four checks above all run ten ids, which is the
#     empty string after the `<lang>` wrap and leaves the padding boundary at the very end of the axis. ---
add_executable(test_e2e_supertonic_mil_real_text gate/test_e2e_supertonic_mil_real_text.cpp)
target_link_libraries(test_e2e_supertonic_mil_real_text PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_mil_real_text COMMAND test_e2e_supertonic_mil_real_text)
set_tests_properties(test_e2e_supertonic_mil_real_text PROPERTIES LABELS gate)
set_tests_properties(test_e2e_supertonic_mil_real_text PROPERTIES SKIP_RETURN_CODE 77)

# --- Full end-to-end check: the MIL-traced topologies wired via supertonic_driver/, compared
#     against the existing bespoke loom::SupertonicDriver oracle. Skips cleanly if env vars aren't set. ---
add_executable(test_e2e_supertonic_mil_lua_driver gate/test_e2e_supertonic_mil_lua_driver.cpp)
target_link_libraries(test_e2e_supertonic_mil_lua_driver PRIVATE loom::engine)
target_compile_definitions(test_e2e_supertonic_mil_lua_driver PRIVATE LOOM_LEGACY_REF_DIR="${LOOM_LEGACY_REF_DIR}")
add_test(NAME test_e2e_supertonic_mil_lua_driver COMMAND test_e2e_supertonic_mil_lua_driver)
set_tests_properties(test_e2e_supertonic_mil_lua_driver PROPERTIES LABELS gate)
set_tests_properties(test_e2e_supertonic_mil_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Procedural generalization: Matcha-TTS's deterministic Euler-ODE (CFM) sampling loop + real
#     per-token duration expansion, ported to Lua (tools/convert_matcha/matcha_driver.lua) and compared
#     against the existing loom::MatchaDriver on the real checkpoint. See
#     LOOM_PROCEDURAL_GENERALIZATION.md / LOOM_MIL_CONVERSION.md and BACKLOG.md's dated entry. ---
add_executable(test_e2e_matcha_lua_driver gate/test_e2e_matcha_lua_driver.cpp)
target_link_libraries(test_e2e_matcha_lua_driver PRIVATE loom::engine)
target_compile_definitions(test_e2e_matcha_lua_driver PRIVATE LOOM_LEGACY_REF_DIR="${LOOM_LEGACY_REF_DIR}")
add_test(NAME test_e2e_matcha_lua_driver COMMAND test_e2e_matcha_lua_driver)
set_tests_properties(test_e2e_matcha_lua_driver PROPERTIES LABELS gate)
set_tests_properties(test_e2e_matcha_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness checks for the MIL-traced Matcha-TTS topologies (export_matcha_mil.py, part
#     of matcha_mil.gguf) against the SAME real-module reference fixtures the bespoke conversion's own
#     per-module tests already use -- see each test's own docstring for why those fixtures are valid
#     ground truth for the MIL wrappers too. ---
add_executable(test_e2e_matcha_mil_text_encoder gate/test_e2e_matcha_mil_text_encoder.cpp)
target_link_libraries(test_e2e_matcha_mil_text_encoder PRIVATE loom::engine)
add_test(NAME test_e2e_matcha_mil_text_encoder COMMAND test_e2e_matcha_mil_text_encoder)
set_tests_properties(test_e2e_matcha_mil_text_encoder PROPERTIES LABELS gate)
set_tests_properties(test_e2e_matcha_mil_text_encoder PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_e2e_matcha_mil_decoder gate/test_e2e_matcha_mil_decoder.cpp)
target_link_libraries(test_e2e_matcha_mil_decoder PRIVATE loom::engine)
add_test(NAME test_e2e_matcha_mil_decoder COMMAND test_e2e_matcha_mil_decoder)
set_tests_properties(test_e2e_matcha_mil_decoder PROPERTIES LABELS gate)
set_tests_properties(test_e2e_matcha_mil_decoder PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_e2e_matcha_mil_vocoder gate/test_e2e_matcha_mil_vocoder.cpp)
target_link_libraries(test_e2e_matcha_mil_vocoder PRIVATE loom::engine)
add_test(NAME test_e2e_matcha_mil_vocoder COMMAND test_e2e_matcha_mil_vocoder)
set_tests_properties(test_e2e_matcha_mil_vocoder PROPERTIES LABELS gate)
set_tests_properties(test_e2e_matcha_mil_vocoder PROPERTIES SKIP_RETURN_CODE 77)

# --- Procedural generalization: the SAME Euler-ODE (CFM) sampling loop + per-token duration expansion
#     as test_e2e_matcha_lua_driver above, but wired to the MIL-traced topologies
#     (tools/convert_matcha/matcha_driver/) instead of the bespoke hand-built ones -- compared
#     against the existing loom::MatchaDriver on the real checkpoint. ---
add_executable(test_e2e_matcha_mil_lua_driver gate/test_e2e_matcha_mil_lua_driver.cpp)
target_link_libraries(test_e2e_matcha_mil_lua_driver PRIVATE loom::engine)
target_compile_definitions(test_e2e_matcha_mil_lua_driver PRIVATE LOOM_LEGACY_REF_DIR="${LOOM_LEGACY_REF_DIR}")
add_test(NAME test_e2e_matcha_mil_lua_driver COMMAND test_e2e_matcha_mil_lua_driver)
set_tests_properties(test_e2e_matcha_mil_lua_driver PROPERTIES LABELS gate)
set_tests_properties(test_e2e_matcha_mil_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Procedural generalization: VITS's dynamic-length relative-position table cropping (loom.get_weight
#     + loom.pad_crop_relative_embeddings) + Gaussian-noise-interleaved duration expansion, ported to Lua
#     (tools/convert_piper_vits/vits_driver.lua) and compared against the existing loom::VitsDriver on
#     the real checkpoint. See LOOM_PROCEDURAL_GENERALIZATION.md / LOOM_MIL_CONVERSION.md and
#     BACKLOG.md's dated entry. ---
add_executable(test_e2e_vits_lua_driver gate/test_e2e_vits_lua_driver.cpp)
target_link_libraries(test_e2e_vits_lua_driver PRIVATE loom::engine)
target_compile_definitions(test_e2e_vits_lua_driver PRIVATE LOOM_LEGACY_REF_DIR="${LOOM_LEGACY_REF_DIR}")
add_test(NAME test_e2e_vits_lua_driver COMMAND test_e2e_vits_lua_driver)
set_tests_properties(test_e2e_vits_lua_driver PROPERTIES LABELS gate)
set_tests_properties(test_e2e_vits_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- MIL exporter: VITS retraced via export_vits_mil.py (real SynthesizerTrn submodules, not the
#     hand-built bespoke topology above) + tools/convert_piper_vits/vits_driver/. See
#     BACKLOG.md's VITS MIL-export entry. ---
add_executable(test_e2e_vits_mil_lua_driver gate/test_e2e_vits_mil_lua_driver.cpp)
target_link_libraries(test_e2e_vits_mil_lua_driver PRIVATE loom::engine)
add_test(NAME test_e2e_vits_mil_lua_driver COMMAND test_e2e_vits_mil_lua_driver)
set_tests_properties(test_e2e_vits_mil_lua_driver PROPERTIES LABELS gate)
set_tests_properties(test_e2e_vits_mil_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- MIL exporter: flow+vocoder topology vs. a REALISTIC-scale real-PyTorch reference (T=194, z_p up
#     to +-24) -- the fixture that caught a real bug in the bespoke topology (see the test's own
#     docstring and BACKLOG.md). ---
add_executable(test_e2e_vits_mil_flow_vocoder_reference gate/test_e2e_vits_mil_flow_vocoder_reference.cpp)
target_link_libraries(test_e2e_vits_mil_flow_vocoder_reference PRIVATE loom::engine)
add_test(NAME test_e2e_vits_mil_flow_vocoder_reference COMMAND test_e2e_vits_mil_flow_vocoder_reference)
set_tests_properties(test_e2e_vits_mil_flow_vocoder_reference PROPERTIES LABELS gate)
set_tests_properties(test_e2e_vits_mil_flow_vocoder_reference PROPERTIES SKIP_RETURN_CODE 77)

# --- Procedural generalization: Kokoro-82M, the largest driver ported so far (43 topologies, 6 BiLSTM
#     instances host-stepped via loom.run_subgraph Lua for-loops, the new loom.uniform_array binding for
#     SineGen's rand_ini draws), ported to Lua (tools/convert_kokoro/kokoro_driver.lua) and compared
#     against the existing loom::KokoroDriver on the real checkpoint. See
#     LOOM_PROCEDURAL_GENERALIZATION.md / LOOM_MIL_CONVERSION.md and BACKLOG.md's dated entry. ---
add_executable(test_e2e_kokoro_lua_driver gate/test_e2e_kokoro_lua_driver.cpp)
target_link_libraries(test_e2e_kokoro_lua_driver PRIVATE loom::engine)
target_compile_definitions(test_e2e_kokoro_lua_driver PRIVATE LOOM_LEGACY_REF_DIR="${LOOM_LEGACY_REF_DIR}")
add_test(NAME test_e2e_kokoro_lua_driver COMMAND test_e2e_kokoro_lua_driver)
set_tests_properties(test_e2e_kokoro_lua_driver PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Real-model test: the MIL-traced Kokoro export (export_kokoro_mil.py) end-to-end, via
#     kokoro_driver/ -- mixes the two MIL-traced topologies with the LSTM-bound bespoke ones
#     (kokoro_driver/'s own module docstring explains why those stay bespoke). ---
add_executable(test_e2e_kokoro_mil_lua_driver gate/test_e2e_kokoro_mil_lua_driver.cpp)
target_link_libraries(test_e2e_kokoro_mil_lua_driver PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_mil_lua_driver COMMAND test_e2e_kokoro_mil_lua_driver)
set_tests_properties(test_e2e_kokoro_mil_lua_driver PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_mil_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Drop-in equivalence: each MIL-traced topology against the bespoke one it replaces. The real
#     numerical gate for moving a topology out of kokoro.gguf and into kokoro_mil.gguf, since
#     test_e2e_kokoro_mil_lua_driver has no oracle waveform by design (see its own header). ---
add_executable(test_e2e_kokoro_mil_topology_equivalence gate/test_e2e_kokoro_mil_topology_equivalence.cpp)
target_link_libraries(test_e2e_kokoro_mil_topology_equivalence PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_mil_topology_equivalence COMMAND test_e2e_kokoro_mil_topology_equivalence)
set_tests_properties(test_e2e_kokoro_mil_topology_equivalence PROPERTIES LABELS gate)
set_tests_properties(test_e2e_kokoro_mil_topology_equivalence PROPERTIES SKIP_RETURN_CODE 77)

# --- Procedural generalization: StyleTTS2, the most complex driver ported so far (44 topologies, 6
#     BiLSTM instances, and the genuinely new style-diffusion sampler -- ADPM2 over the real
#     Transformer1d denoiser, ported to Lua as plain host arithmetic + loom.run_subgraph calls), compared
#     against the existing loom::StyleTTS2Driver on the real checkpoint. See
#     LOOM_PROCEDURAL_GENERALIZATION.md / LOOM_MIL_CONVERSION.md and BACKLOG.md's dated entry. ---
add_executable(test_e2e_styletts2_lua_driver gate/test_e2e_styletts2_lua_driver.cpp)
target_link_libraries(test_e2e_styletts2_lua_driver PRIVATE loom::engine)
target_compile_definitions(test_e2e_styletts2_lua_driver PRIVATE LOOM_LEGACY_REF_DIR="${LOOM_LEGACY_REF_DIR}")
add_test(NAME test_e2e_styletts2_lua_driver COMMAND test_e2e_styletts2_lua_driver)
set_tests_properties(test_e2e_styletts2_lua_driver PROPERTIES LABELS gate)
set_tests_properties(test_e2e_styletts2_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Real-model test: the MIL-traced StyleTTS2 export (export_styletts2_mil.py) end-to-end, via
#     styletts2_driver/ -- mirrors test_e2e_kokoro_mil_lua_driver.cpp's own "no oracle, orchestration
#     sanity" scope (see that test's own docstring, and this one's, for why a tight full-pipeline diff
#     against the bespoke C++ oracle isn't a meaningful target here). Skips cleanly if env vars aren't set. ---
add_executable(test_e2e_styletts2_mil_lua_driver gate/test_e2e_styletts2_mil_lua_driver.cpp)
target_link_libraries(test_e2e_styletts2_mil_lua_driver PRIVATE loom::engine)
add_test(NAME test_e2e_styletts2_mil_lua_driver COMMAND test_e2e_styletts2_mil_lua_driver)
set_tests_properties(test_e2e_styletts2_mil_lua_driver PROPERTIES LABELS gate)
set_tests_properties(test_e2e_styletts2_mil_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Procedural generalization: LFM2-350M fully dynamic multi-topology model compiled via coremltools
#     and Loom MIL compiler backend, executing dynamically via embedded LuaVM and GraphBuilder ---
add_executable(test_e2e_lfm2_lua_driver gate/test_e2e_lfm2_lua_driver.cpp)
target_link_libraries(test_e2e_lfm2_lua_driver PRIVATE loom::engine)
add_test(NAME test_e2e_lfm2_lua_driver COMMAND test_e2e_lfm2_lua_driver)
set_tests_properties(test_e2e_lfm2_lua_driver PROPERTIES LABELS gate)
set_tests_properties(test_e2e_lfm2_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- LFM2-350M exported via the generic MIL compiler (export_lfm2_monolithic.py), verified at genuinely
#     dynamic (non-fixed, unpadded) prompt lengths against real HF logits ---
add_executable(test_e2e_lfm2_mil_export gate/test_e2e_lfm2_mil_export.cpp)
target_link_libraries(test_e2e_lfm2_mil_export PRIVATE loom::engine)
add_test(NAME test_e2e_lfm2_mil_export COMMAND test_e2e_lfm2_mil_export)
set_tests_properties(test_e2e_lfm2_mil_export PROPERTIES LABELS gate)
set_tests_properties(test_e2e_lfm2_mil_export PROPERTIES SKIP_RETURN_CODE 77)

# --- The KV cache's own claim (KV-CACHE.md 3.4): a fused causal LM's `infer_with_past` generates the
#     same tokens as calling `infer` N times with a growing prompt. Generic over any fused causal-LM
#     GGUF -- set LOOM_CAUSAL_LM_KV_GGUF; skips if the file has no such entry. ---
add_executable(test_e2e_causal_lm_infer_with_past gate/test_e2e_causal_lm_infer_with_past.cpp)
target_link_libraries(test_e2e_causal_lm_infer_with_past PRIVATE loom::engine)
add_test(NAME test_e2e_causal_lm_infer_with_past COMMAND test_e2e_causal_lm_infer_with_past)
set_tests_properties(test_e2e_causal_lm_infer_with_past PROPERTIES LABELS gate)
set_tests_properties(test_e2e_causal_lm_infer_with_past PROPERTIES SKIP_RETURN_CODE 77)

# --- The marshalling ceiling's own claim (BACKLOG.md P4.0.14): a prefill whose logits tensor is larger
#     than LuaJIT's array part can hold now completes, because the logits never become a Lua table.
#     Generic over any causal-LM GGUF -- the prompt length comes from the file's own vocab size. Set
#     LOOM_PREFILL_CEILING_GGUF; skips if unset. ---
add_executable(test_e2e_prefill_past_marshalling_ceiling gate/test_e2e_prefill_past_marshalling_ceiling.cpp)
target_link_libraries(test_e2e_prefill_past_marshalling_ceiling PRIVATE loom::engine)
add_test(NAME test_e2e_prefill_past_marshalling_ceiling COMMAND test_e2e_prefill_past_marshalling_ceiling)
set_tests_properties(test_e2e_prefill_past_marshalling_ceiling PROPERTIES LABELS gate)
set_tests_properties(test_e2e_prefill_past_marshalling_ceiling PROPERTIES SKIP_RETURN_CODE 77)

# --- LSTM export via recurrent.py's build_lstm_cell_topologies + LoomLuaBridge::l_run_recurrent
#     (EXPORT-IMPROVEMENT-BACKLOG.md item 4), verified against a real bidirectional torch.nn.LSTM ---
add_executable(test_e2e_lstm_recurrent gate/test_e2e_lstm_recurrent.cpp)
target_link_libraries(test_e2e_lstm_recurrent PRIVATE loom::engine)
add_test(NAME test_e2e_lstm_recurrent COMMAND test_e2e_lstm_recurrent)
set_tests_properties(test_e2e_lstm_recurrent PROPERTIES LABELS gate)
set_tests_properties(test_e2e_lstm_recurrent PROPERTIES SKIP_RETURN_CODE 77)

# --- Same LFM2-350M monolithic export's new tokenizer.ggml.* KVs (EXPORT-BACKLOG.md item 4) --
#     loom::BpeVocab::encode/decode checked against the real HF AutoTokenizer on the same checkpoint ---
add_executable(test_e2e_lfm2_tokenizer gate/test_e2e_lfm2_tokenizer.cpp)
target_link_libraries(test_e2e_lfm2_tokenizer PRIVATE loom::engine)
add_test(NAME test_e2e_lfm2_tokenizer COMMAND test_e2e_lfm2_tokenizer)
set_tests_properties(test_e2e_lfm2_tokenizer PROPERTIES LABELS gate)
set_tests_properties(test_e2e_lfm2_tokenizer PROPERTIES SKIP_RETURN_CODE 77)

# --- Same LFM2-350M monolithic export, quantized to Q8_0 via the exporter's `quantize=` kwarg
#     (EXPORT-BACKLOG.md item 6) -- verifies real per-logit closeness against the F32 reference above ---
add_executable(test_e2e_lfm2_q8_0 gate/test_e2e_lfm2_q8_0.cpp)
target_link_libraries(test_e2e_lfm2_q8_0 PRIVATE loom::engine)
add_test(NAME test_e2e_lfm2_q8_0 COMMAND test_e2e_lfm2_q8_0)
set_tests_properties(test_e2e_lfm2_q8_0 PROPERTIES LABELS gate)
set_tests_properties(test_e2e_lfm2_q8_0 PROPERTIES SKIP_RETURN_CODE 77)

# --- End-to-end check: the MIL-compiler-exported Whisper (tools/loom_mil_compiler/whisper_export.py,
#     BACKLOG.md P4.1) against HF's own WhisperForConditionalGeneration -- the encoder half (which
#     includes the in-graph mel frontend) numerically, and the generated Lua driver's whole
#     encoder-then-cached-decode-loop as an exact token sequence. See test_e2e_whisper_mil_export.cpp. ---
add_executable(test_e2e_whisper_mil_export gate/test_e2e_whisper_mil_export.cpp)
target_link_libraries(test_e2e_whisper_mil_export PRIVATE loom::engine)
add_test(NAME test_e2e_whisper_mil_export COMMAND test_e2e_whisper_mil_export)
set_tests_properties(test_e2e_whisper_mil_export PROPERTIES LABELS gate)
set_tests_properties(test_e2e_whisper_mil_export PROPERTIES SKIP_RETURN_CODE 77)

# --- End-to-end check: the MIL-compiler-exported Qwen3-ASR (tools/loom_mil_compiler/speech_lm_export.py
#     + qwen3_asr_export.py, BACKLOG.md P4.3) against HF's own Qwen3ASRForConditionalGeneration -- the
#     encoder+projector half as a TENSOR comparison (a wrong encoder still decodes a plausible
#     transcript, see P4.2), and the generated driver's whole segmented-prompt-then-decode-loop as an
#     exact token sequence. Family 3's gate. See test_e2e_qwen3_asr_mil_export.cpp. ---
add_executable(test_e2e_qwen3_asr_mil_export gate/test_e2e_qwen3_asr_mil_export.cpp)
target_link_libraries(test_e2e_qwen3_asr_mil_export PRIVATE loom::engine)
add_test(NAME test_e2e_qwen3_asr_mil_export COMMAND test_e2e_qwen3_asr_mil_export)
set_tests_properties(test_e2e_qwen3_asr_mil_export PROPERTIES LABELS gate)
set_tests_properties(test_e2e_qwen3_asr_mil_export PROPERTIES SKIP_RETURN_CODE 77)

# --- End-to-end check: the MIL-compiler-exported Granite Speech
#     (tools/loom_mil_compiler/speech_lm_export.py + granite_speech_export.py, BACKLOG.md P4.3c) against
#     HF's own GraniteSpeechForConditionalGeneration. Family 3's SECOND leaf, and deliberately the same
#     two checks as the first: a conformer encoder and a Q-Former projector share nothing with
#     Qwen3-ASR's window-attention stack and linear projector, and the driver, the phase list and the
#     test are unchanged. See test_e2e_granite_speech_mil_export.cpp. ---
add_executable(test_e2e_granite_speech_mil_export gate/test_e2e_granite_speech_mil_export.cpp)
target_link_libraries(test_e2e_granite_speech_mil_export PRIVATE loom::engine)
add_test(NAME test_e2e_granite_speech_mil_export COMMAND test_e2e_granite_speech_mil_export)
set_tests_properties(test_e2e_granite_speech_mil_export PROPERTIES LABELS gate)
set_tests_properties(test_e2e_granite_speech_mil_export PROPERTIES SKIP_RETURN_CODE 77)

# --- The gate suite's own gate: how `fixture_env` decides whether a real-model test can run at all.
#     A ci test about gate tests, deliberately -- a wrong answer there makes 78 checks vanish into
#     Skipped without anything going red. See ci/test_fixture_resolution.cpp. ---
add_executable(test_fixture_resolution ci/test_fixture_resolution.cpp)
target_link_libraries(test_fixture_resolution PRIVATE loom::engine)
add_test(NAME test_fixture_resolution COMMAND test_fixture_resolution)
set_tests_properties(test_fixture_resolution PROPERTIES LABELS ci)
