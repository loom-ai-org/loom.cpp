find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(LOOM_FIXTURE_DIR ${CMAKE_CURRENT_BINARY_DIR}/fixtures)

# --- Phase 1: minimal ad hoc GGUF fixture, regenerated fresh before test_gguf_model_load runs ---
add_test(
    NAME gen_minimal_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_minimal_gguf.py
            ${LOOM_FIXTURE_DIR}/minimal.gguf
)
set_tests_properties(gen_minimal_gguf_fixture PROPERTIES FIXTURES_SETUP minimal_gguf_fixture)

add_executable(test_gguf_model_load test_gguf_model_load.cpp)
target_link_libraries(test_gguf_model_load PRIVATE loom::engine)
target_compile_definitions(test_gguf_model_load PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_gguf_model_load COMMAND test_gguf_model_load)
set_tests_properties(test_gguf_model_load PROPERTIES FIXTURES_REQUIRED minimal_gguf_fixture)

# --- Phase 2: primitive registry / topology parsing / graph builder ---

add_executable(test_primitive_registry test_primitive_registry.cpp)
target_link_libraries(test_primitive_registry PRIVATE loom::engine)
add_test(NAME test_primitive_registry COMMAND test_primitive_registry)

add_executable(test_graph_topology_parse test_graph_topology_parse.cpp)
target_link_libraries(test_graph_topology_parse PRIVATE loom::engine)
add_test(NAME test_graph_topology_parse COMMAND test_graph_topology_parse)

# Attention-free toy model GGUF shared by test_graph_builder_shapes and test_gallocr_reserve_reuse.
add_test(
    NAME gen_builder_test_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_builder_test_gguf.py
            ${LOOM_FIXTURE_DIR}/builder_test.gguf
)
set_tests_properties(gen_builder_test_gguf_fixture PROPERTIES FIXTURES_SETUP builder_test_gguf_fixture)

add_executable(test_graph_builder_shapes test_graph_builder_shapes.cpp)
target_link_libraries(test_graph_builder_shapes PRIVATE loom::engine)
target_compile_definitions(test_graph_builder_shapes PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_graph_builder_shapes COMMAND test_graph_builder_shapes)
set_tests_properties(test_graph_builder_shapes PROPERTIES FIXTURES_REQUIRED builder_test_gguf_fixture)

add_executable(test_gallocr_reserve_reuse test_gallocr_reserve_reuse.cpp)
target_link_libraries(test_gallocr_reserve_reuse PRIVATE loom::engine)
target_compile_definitions(test_gallocr_reserve_reuse PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_gallocr_reserve_reuse COMMAND test_gallocr_reserve_reuse)
set_tests_properties(test_gallocr_reserve_reuse PROPERTIES FIXTURES_REQUIRED builder_test_gguf_fixture)

# --- Phase 3: KV cache / ATTENTION primitive / autoregressive generation ---

add_executable(test_kv_cache test_kv_cache.cpp)
target_link_libraries(test_kv_cache PRIVATE loom::engine)
add_test(NAME test_kv_cache COMMAND test_kv_cache)

# Tiny single-layer attention toy model shared by test_generation_smoke.
add_test(
    NAME gen_attention_test_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_attention_test_gguf.py
            ${LOOM_FIXTURE_DIR}/attention_test.gguf
)
set_tests_properties(gen_attention_test_gguf_fixture PROPERTIES FIXTURES_SETUP attention_test_gguf_fixture)

add_executable(test_generation_smoke test_generation_smoke.cpp)
target_link_libraries(test_generation_smoke PRIVATE loom::engine)
target_compile_definitions(test_generation_smoke PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_generation_smoke COMMAND test_generation_smoke)
set_tests_properties(test_generation_smoke PROPERTIES FIXTURES_REQUIRED attention_test_gguf_fixture)

# --- Phase 4: full toy-LLM end-to-end test, verified against an independent numpy reference ---

set(LOOM_E2E_REF_DIR ${LOOM_FIXTURE_DIR}/e2e_ref)

add_test(
    NAME gen_toy_llm_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/make_toy_llm_gguf.py
            ${LOOM_FIXTURE_DIR}/toy_llm.gguf
)
set_tests_properties(gen_toy_llm_gguf_fixture PROPERTIES FIXTURES_SETUP toy_llm_gguf_fixture)

add_test(
    NAME gen_toy_llm_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/reference_forward.py
            ${LOOM_E2E_REF_DIR} --prompt 1 2 3 --n-new 4
)
set_tests_properties(gen_toy_llm_reference PROPERTIES FIXTURES_SETUP toy_llm_reference_fixture)

add_executable(test_e2e_toy_llm test_e2e_toy_llm.cpp)
target_link_libraries(test_e2e_toy_llm PRIVATE loom::engine)
target_compile_definitions(test_e2e_toy_llm PRIVATE
    LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}"
    LOOM_TEST_REF_DIR="${LOOM_E2E_REF_DIR}"
)
add_test(NAME test_e2e_toy_llm COMMAND test_e2e_toy_llm)
set_tests_properties(test_e2e_toy_llm PROPERTIES FIXTURES_REQUIRED "toy_llm_gguf_fixture;toy_llm_reference_fixture")

# --- Milestone 2: toy vision encoder end-to-end test ---

set(LOOM_E2E_VISION_REF_DIR ${LOOM_FIXTURE_DIR}/e2e_vision_ref)

add_test(
    NAME gen_toy_vision_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/make_toy_vision_gguf.py
            ${LOOM_FIXTURE_DIR}/toy_vision.gguf
)
set_tests_properties(gen_toy_vision_gguf_fixture PROPERTIES FIXTURES_SETUP toy_vision_gguf_fixture)

add_test(
    NAME gen_toy_vision_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/reference_forward_vision.py
            ${LOOM_E2E_VISION_REF_DIR}
)
set_tests_properties(gen_toy_vision_reference PROPERTIES FIXTURES_SETUP toy_vision_reference_fixture)

add_executable(test_e2e_toy_vision test_e2e_toy_vision.cpp)
target_link_libraries(test_e2e_toy_vision PRIVATE loom::engine)
target_compile_definitions(test_e2e_toy_vision PRIVATE
    LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}"
    LOOM_TEST_REF_DIR="${LOOM_E2E_VISION_REF_DIR}"
)
add_test(NAME test_e2e_toy_vision COMMAND test_e2e_toy_vision)
set_tests_properties(test_e2e_toy_vision PROPERTIES FIXTURES_REQUIRED "toy_vision_gguf_fixture;toy_vision_reference_fixture")

# --- Milestone 2: toy ASR encoder end-to-end test ---

set(LOOM_E2E_ASR_REF_DIR ${LOOM_FIXTURE_DIR}/e2e_asr_ref)

add_test(
    NAME gen_toy_asr_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/make_toy_asr_gguf.py
            ${LOOM_FIXTURE_DIR}/toy_asr.gguf
)
set_tests_properties(gen_toy_asr_gguf_fixture PROPERTIES FIXTURES_SETUP toy_asr_gguf_fixture)

add_test(
    NAME gen_toy_asr_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/reference_forward_asr.py
            ${LOOM_E2E_ASR_REF_DIR}
)
set_tests_properties(gen_toy_asr_reference PROPERTIES FIXTURES_SETUP toy_asr_reference_fixture)

add_executable(test_e2e_toy_asr test_e2e_toy_asr.cpp)
target_link_libraries(test_e2e_toy_asr PRIVATE loom::engine)
target_compile_definitions(test_e2e_toy_asr PRIVATE
    LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}"
    LOOM_TEST_REF_DIR="${LOOM_E2E_ASR_REF_DIR}"
)
add_test(NAME test_e2e_toy_asr COMMAND test_e2e_toy_asr)
set_tests_properties(test_e2e_toy_asr PROPERTIES FIXTURES_REQUIRED "toy_asr_gguf_fixture;toy_asr_reference_fixture")

# --- Milestone 3: toy ODE-stepper (flow-matching vector field) end-to-end test ---

set(LOOM_E2E_ODE_REF_DIR ${LOOM_FIXTURE_DIR}/e2e_ode_ref)

add_test(
    NAME gen_toy_ode_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/make_toy_ode_gguf.py
            ${LOOM_FIXTURE_DIR}/toy_ode.gguf
)
set_tests_properties(gen_toy_ode_gguf_fixture PROPERTIES FIXTURES_SETUP toy_ode_gguf_fixture)

add_test(
    NAME gen_toy_ode_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/reference_forward_ode.py
            ${LOOM_E2E_ODE_REF_DIR} --n-steps 8 --t-start 0.0 --t-end 1.0
)
set_tests_properties(gen_toy_ode_reference PROPERTIES FIXTURES_SETUP toy_ode_reference_fixture)

add_executable(test_e2e_toy_ode test_e2e_toy_ode.cpp)
target_link_libraries(test_e2e_toy_ode PRIVATE loom::engine)
target_compile_definitions(test_e2e_toy_ode PRIVATE
    LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}"
    LOOM_TEST_REF_DIR="${LOOM_E2E_ODE_REF_DIR}"
)
add_test(NAME test_e2e_toy_ode COMMAND test_e2e_toy_ode)
set_tests_properties(test_e2e_toy_ode PROPERTIES FIXTURES_REQUIRED "toy_ode_gguf_fixture;toy_ode_reference_fixture")

# --- Milestone 3: toy VAE decoder end-to-end test ---

set(LOOM_E2E_VAE_REF_DIR ${LOOM_FIXTURE_DIR}/e2e_vae_ref)

add_test(
    NAME gen_toy_vae_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/make_toy_vae_gguf.py
            ${LOOM_FIXTURE_DIR}/toy_vae.gguf
)
set_tests_properties(gen_toy_vae_gguf_fixture PROPERTIES FIXTURES_SETUP toy_vae_gguf_fixture)

add_test(
    NAME gen_toy_vae_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/reference_forward_vae.py
            ${LOOM_E2E_VAE_REF_DIR}
)
set_tests_properties(gen_toy_vae_reference PROPERTIES FIXTURES_SETUP toy_vae_reference_fixture)

add_executable(test_e2e_toy_vae test_e2e_toy_vae.cpp)
target_link_libraries(test_e2e_toy_vae PRIVATE loom::engine)
target_compile_definitions(test_e2e_toy_vae PRIVATE
    LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}"
    LOOM_TEST_REF_DIR="${LOOM_E2E_VAE_REF_DIR}"
)
add_test(NAME test_e2e_toy_vae COMMAND test_e2e_toy_vae)
set_tests_properties(test_e2e_toy_vae PROPERTIES FIXTURES_REQUIRED "toy_vae_gguf_fixture;toy_vae_reference_fixture")

# --- Regression test pinning down BACKLOG.md's ggml graph-reuse finding (see
#     tests/test_graph_reuse_safety.cpp) -- the pattern OdeStepper relies on ---
add_executable(test_graph_reuse_safety test_graph_reuse_safety.cpp)
target_link_libraries(test_graph_reuse_safety PRIVATE loom::engine)
add_test(NAME test_graph_reuse_safety COMMAND test_graph_reuse_safety)

# --- Real-model test: NVIDIA stt_en_conformer_ctc_small, converted via tools/convert_nemo/. Unlike
#     every toy fixture above, this is NOT regenerated at ctest time (needs the real checkpoint + a
#     PyTorch environment) -- it looks for LOOM_CONFORMER_CTC_DIR (default /tmp/nemo_model) and skips
#     cleanly via ctest's SKIP_RETURN_CODE if that directory hasn't been prepared. See
#     tools/convert_nemo/convert_conformer_ctc.py and reference_forward_conformer.py for how to build it.
add_executable(test_e2e_conformer_ctc test_e2e_conformer_ctc.cpp)
target_link_libraries(test_e2e_conformer_ctc PRIVATE loom::engine)
add_test(NAME test_e2e_conformer_ctc COMMAND test_e2e_conformer_ctc)
set_tests_properties(test_e2e_conformer_ctc PROPERTIES SKIP_RETURN_CODE 77)

# Proves the same converted GGUF supports a second, different waveform length dynamically (see
# SPECIFICATION.md §4 and BACKLOG.md) -- no reconversion, just a different build() call.
add_executable(test_e2e_conformer_ctc_dynamic_length test_e2e_conformer_ctc_dynamic_length.cpp)
target_link_libraries(test_e2e_conformer_ctc_dynamic_length PRIVATE loom::engine)
add_test(NAME test_e2e_conformer_ctc_dynamic_length COMMAND test_e2e_conformer_ctc_dynamic_length)
set_tests_properties(test_e2e_conformer_ctc_dynamic_length PROPERTIES SKIP_RETURN_CODE 77)

# --- Tokenizer: Vocab (SentencePiece unigram, llama.cpp GGUF schema) against the real converted model ---
add_executable(test_vocab test_vocab.cpp)
target_link_libraries(test_vocab PRIVATE loom::engine)
add_test(NAME test_vocab COMMAND test_vocab)
set_tests_properties(test_vocab PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_ctc_decode test_ctc_decode.cpp)
target_link_libraries(test_ctc_decode PRIVATE loom::engine)
add_test(NAME test_ctc_decode COMMAND test_ctc_decode)
