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

# --- Generic-converter proof of concept: same weights/reference as test_e2e_toy_llm above, but the
#     topology JSON is auto-derived from a real torch.export() ATen graph (tools/convert_generic/), not
#     hand-written. Needs a torch environment to produce its GGUF, so -- like the real-model tests below
#     -- it's not wired into the default fixture-generation flow; it looks for LOOM_TOY_LLM_GENERIC_DIR
#     and skips cleanly (SKIP_RETURN_CODE) if that GGUF hasn't been prepared. Reuses toy_llm_reference_fixture
#     (same weights => same expected logits) rather than generating a second reference.
add_executable(test_e2e_toy_llm_generic test_e2e_toy_llm_generic.cpp)
target_link_libraries(test_e2e_toy_llm_generic PRIVATE loom::engine)
target_compile_definitions(test_e2e_toy_llm_generic PRIVATE LOOM_TEST_REF_DIR="${LOOM_E2E_REF_DIR}")
add_test(NAME test_e2e_toy_llm_generic COMMAND test_e2e_toy_llm_generic)
set_tests_properties(test_e2e_toy_llm_generic PROPERTIES
    SKIP_RETURN_CODE 77
    FIXTURES_REQUIRED toy_llm_reference_fixture)

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

# --- Gating-criterion proof of concept: generic converter (tools/convert_generic/) against the real
#     Conformer-CTC encoder -- a genuinely non-decoder-transformer shape. See BACKLOG.md. ---
add_executable(test_e2e_conformer_ctc_generic test_e2e_conformer_ctc_generic.cpp)
target_link_libraries(test_e2e_conformer_ctc_generic PRIVATE loom::engine)
add_test(NAME test_e2e_conformer_ctc_generic COMMAND test_e2e_conformer_ctc_generic)
set_tests_properties(test_e2e_conformer_ctc_generic PROPERTIES SKIP_RETURN_CODE 77)

# --- Tokenizer: Vocab (SentencePiece unigram, llama.cpp GGUF schema) against the real converted model ---
add_executable(test_vocab test_vocab.cpp)
target_link_libraries(test_vocab PRIVATE loom::engine)
add_test(NAME test_vocab COMMAND test_vocab)
set_tests_properties(test_vocab PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_ctc_decode test_ctc_decode.cpp)
target_link_libraries(test_ctc_decode PRIVATE loom::engine)
add_test(NAME test_ctc_decode COMMAND test_ctc_decode)

# --- Qwen3 roadmap item: Unicode subsystem (NFC normalize, \p{L}/\p{N} classification) backing BpeVocab ---
add_executable(test_unicode test_unicode.cpp)
target_link_libraries(test_unicode PRIVATE loom::engine)
add_test(NAME test_unicode COMMAND test_unicode)

# --- Qwen3 roadmap item: BpeVocab (byte-level BPE, llama.cpp "gpt2" GGUF schema) against a small
#     synthetic fixture with hand-traceable merges (see tests/fixtures/make_bpe_vocab_gguf.py) ---
add_test(
    NAME gen_bpe_vocab_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_bpe_vocab_gguf.py
            ${LOOM_FIXTURE_DIR}/bpe_vocab_test.gguf
)
set_tests_properties(gen_bpe_vocab_gguf_fixture PROPERTIES FIXTURES_SETUP bpe_vocab_gguf_fixture)

add_executable(test_bpe_vocab test_bpe_vocab.cpp)
target_link_libraries(test_bpe_vocab PRIVATE loom::engine)
target_compile_definitions(test_bpe_vocab PRIVATE LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}")
add_test(NAME test_bpe_vocab COMMAND test_bpe_vocab)
set_tests_properties(test_bpe_vocab PROPERTIES FIXTURES_REQUIRED bpe_vocab_gguf_fixture)

# --- Qwen3 roadmap item: GQA (n_head_kv < n_head) regression test, verified against numpy before the
#     real Qwen3-0.6B-Base checkpoint (16 query / 8 KV heads) depends on ATTENTION's broadcast path ---
set(LOOM_E2E_GQA_REF_DIR ${LOOM_FIXTURE_DIR}/e2e_gqa_ref)

add_test(
    NAME gen_gqa_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/make_gqa_test_gguf.py
            ${LOOM_FIXTURE_DIR}/gqa_test.gguf
)
set_tests_properties(gen_gqa_gguf_fixture PROPERTIES FIXTURES_SETUP gqa_gguf_fixture)

add_test(
    NAME gen_gqa_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/reference_forward_gqa.py
            ${LOOM_E2E_GQA_REF_DIR} --prompt 1 2 3 --n-new 4
)
set_tests_properties(gen_gqa_reference PROPERTIES FIXTURES_SETUP gqa_reference_fixture)

add_executable(test_e2e_gqa test_e2e_gqa.cpp)
target_link_libraries(test_e2e_gqa PRIVATE loom::engine)
target_compile_definitions(test_e2e_gqa PRIVATE
    LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}"
    LOOM_TEST_REF_DIR="${LOOM_E2E_GQA_REF_DIR}"
)
add_test(NAME test_e2e_gqa COMMAND test_e2e_gqa)
set_tests_properties(test_e2e_gqa PROPERTIES FIXTURES_REQUIRED "gqa_gguf_fixture;gqa_reference_fixture")

# --- Real-model test: Qwen3-0.6B-Base, converted via tools/convert_qwen3/. Same "not regenerated at
#     ctest time, skip cleanly if the real checkpoint isn't prepared" pattern as test_e2e_conformer_ctc ---
add_executable(test_e2e_qwen3 test_e2e_qwen3.cpp)
target_link_libraries(test_e2e_qwen3 PRIVATE loom::engine)
add_test(NAME test_e2e_qwen3 COMMAND test_e2e_qwen3)
set_tests_properties(test_e2e_qwen3 PROPERTIES SKIP_RETURN_CODE 77)

# --- Generic-converter proof of concept, round 2: real Qwen3-0.6B-Base through the same, unmodified
#     converter/op-mapping table the toy LLM used (tools/convert_generic/) -- see BACKLOG.md ---
add_executable(test_e2e_qwen3_generic test_e2e_qwen3_generic.cpp)
target_link_libraries(test_e2e_qwen3_generic PRIVATE loom::engine)
add_test(NAME test_e2e_qwen3_generic COMMAND test_e2e_qwen3_generic)
set_tests_properties(test_e2e_qwen3_generic PROPERTIES SKIP_RETURN_CODE 77)

# --- Quantized-weight proof of concept: same real Qwen3-0.6B-Base checkpoint, matmul weights quantized
#     to Q8_0 by tools/quantize/quantize_gguf_q8_0.py -- see BACKLOG.md's "quantized weight support"
#     milestone. Same skip-cleanly-if-not-prepared pattern as the other real-model tests above.
add_executable(test_e2e_qwen3_q8_0 test_e2e_qwen3_q8_0.cpp)
target_link_libraries(test_e2e_qwen3_q8_0 PRIVATE loom::engine)
add_test(NAME test_e2e_qwen3_q8_0 COMMAND test_e2e_qwen3_q8_0)
set_tests_properties(test_e2e_qwen3_q8_0 PROPERTIES SKIP_RETURN_CODE 77)
