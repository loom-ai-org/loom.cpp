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

# --- Vocab's new SentencePiece BPE path ("llama" tag), against the real Parakeet-TDT tokenizer ---
add_executable(test_vocab_spm_bpe test_vocab_spm_bpe.cpp)
target_link_libraries(test_vocab_spm_bpe PRIVATE loom::engine)
add_test(NAME test_vocab_spm_bpe COMMAND test_vocab_spm_bpe)
set_tests_properties(test_vocab_spm_bpe PROPERTIES SKIP_RETURN_CODE 77)

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

# --- Parakeet-TDT roadmap, step 1: the LSTM-step composite-topology pattern (MUL_MAT/ADD/VIEW/SIGMOID/
#     TANH/MUL -- no monolithic LSTM primitive), verified bit-exact against an independent numpy reference
#     on a tiny synthetic fixture before it's trusted inside the real TdtDecoder driver. Fully synthetic,
#     procedurally generated at ctest time like the other toy fixtures -- not skip-if-missing. ---
set(LOOM_LSTM_STEP_REF_DIR ${LOOM_FIXTURE_DIR}/lstm_step_ref)

add_test(
    NAME gen_lstm_step_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/make_lstm_step_gguf.py
            ${LOOM_FIXTURE_DIR}/lstm_step_h.gguf ${LOOM_FIXTURE_DIR}/lstm_step_c.gguf
)
set_tests_properties(gen_lstm_step_gguf_fixture PROPERTIES FIXTURES_SETUP lstm_step_gguf_fixture)

add_test(
    NAME gen_lstm_step_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/reference_lstm_step.py
            ${LOOM_LSTM_STEP_REF_DIR}
)
set_tests_properties(gen_lstm_step_reference PROPERTIES FIXTURES_SETUP lstm_step_reference_fixture)

add_executable(test_lstm_step test_lstm_step.cpp)
target_link_libraries(test_lstm_step PRIVATE loom::engine)
target_compile_definitions(test_lstm_step PRIVATE
    LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}"
    LOOM_TEST_REF_DIR="${LOOM_LSTM_STEP_REF_DIR}"
)
add_test(NAME test_lstm_step COMMAND test_lstm_step)
set_tests_properties(test_lstm_step PROPERTIES FIXTURES_REQUIRED "lstm_step_gguf_fixture;lstm_step_reference_fixture")

# --- Kokoro/StyleTTS2 roadmap: the host-side duration-based frame-expansion (loom::predict_durations/
#     loom::expand_by_duration -- KModel.forward_with_tokens's sigmoid+sum/round/clamp/repeat_interleave
#     formula), verified against an independent numpy reference on a tiny synthetic fixture. Pure host
#     arithmetic, no GGUF/ggml graph involved at all -- fully synthetic, not skip-if-missing. ---
set(LOOM_DURATION_ALIGNER_REF_DIR ${LOOM_FIXTURE_DIR}/duration_aligner_ref)

add_test(
    NAME gen_duration_aligner_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/reference_duration_aligner.py
            ${LOOM_DURATION_ALIGNER_REF_DIR}
)
set_tests_properties(gen_duration_aligner_reference PROPERTIES FIXTURES_SETUP duration_aligner_reference_fixture)

add_executable(test_duration_aligner test_duration_aligner.cpp)
target_link_libraries(test_duration_aligner PRIVATE loom::engine)
target_compile_definitions(test_duration_aligner PRIVATE
    LOOM_TEST_REF_DIR="${LOOM_DURATION_ALIGNER_REF_DIR}"
)
add_test(NAME test_duration_aligner COMMAND test_duration_aligner)
set_tests_properties(test_duration_aligner PROPERTIES FIXTURES_REQUIRED "duration_aligner_reference_fixture")

# --- Parakeet-TDT roadmap, step 2: the full greedy-TDT decode loop (new TdtDecoder driver), verified
#     against an independent numpy reference on a tiny synthetic fixture -- see BACKLOG.md. ---
set(LOOM_TDT_STEP_REF_DIR ${LOOM_FIXTURE_DIR}/tdt_step_ref)

add_test(
    NAME gen_tdt_step_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/make_tdt_step_gguf.py
            ${LOOM_FIXTURE_DIR}/tdt_step_test
)
set_tests_properties(gen_tdt_step_gguf_fixture PROPERTIES FIXTURES_SETUP tdt_step_gguf_fixture)

add_test(
    NAME gen_tdt_step_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/reference_tdt_step.py
            ${LOOM_TDT_STEP_REF_DIR}
)
set_tests_properties(gen_tdt_step_reference PROPERTIES FIXTURES_SETUP tdt_step_reference_fixture)

add_executable(test_tdt_decoder test_tdt_decoder.cpp)
target_link_libraries(test_tdt_decoder PRIVATE loom::engine)
target_compile_definitions(test_tdt_decoder PRIVATE
    LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}/tdt_step_test"
    LOOM_TEST_REF_DIR="${LOOM_TDT_STEP_REF_DIR}"
)
add_test(NAME test_tdt_decoder COMMAND test_tdt_decoder)
set_tests_properties(test_tdt_decoder PROPERTIES FIXTURES_REQUIRED "tdt_step_gguf_fixture;tdt_step_reference_fixture")

# --- FastConformer RNN-T roadmap, step 1: loom::TdtDecoder's new plain-RNN-T mode (empty `durations`),
#     verified against an independent numpy reference of standard RNN-T greedy decoding on a tiny
#     synthetic fixture -- see BACKLOG.md. Same pattern as the TDT fixture above, deliberately
#     duplicated rather than shared (this project's usual per-fixture convention). ---
set(LOOM_RNNT_STEP_REF_DIR ${LOOM_FIXTURE_DIR}/rnnt_step_ref)

add_test(
    NAME gen_rnnt_step_gguf_fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/make_rnnt_step_gguf.py
            ${LOOM_FIXTURE_DIR}/rnnt_step_test
)
set_tests_properties(gen_rnnt_step_gguf_fixture PROPERTIES FIXTURES_SETUP rnnt_step_gguf_fixture)

add_test(
    NAME gen_rnnt_step_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/reference_rnnt_step.py
            ${LOOM_RNNT_STEP_REF_DIR}
)
set_tests_properties(gen_rnnt_step_reference PROPERTIES FIXTURES_SETUP rnnt_step_reference_fixture)

add_executable(test_rnnt_decoder test_rnnt_decoder.cpp)
target_link_libraries(test_rnnt_decoder PRIVATE loom::engine)
target_compile_definitions(test_rnnt_decoder PRIVATE
    LOOM_TEST_FIXTURE_DIR="${LOOM_FIXTURE_DIR}/rnnt_step_test"
    LOOM_TEST_REF_DIR="${LOOM_RNNT_STEP_REF_DIR}"
)
add_test(NAME test_rnnt_decoder COMMAND test_rnnt_decoder)
set_tests_properties(test_rnnt_decoder PROPERTIES FIXTURES_REQUIRED "rnnt_step_gguf_fixture;rnnt_step_reference_fixture")

# --- Real-model test: nvidia/parakeet-tdt-0.6b-v3, converted via tools/convert_nemo/convert_parakeet_tdt.py
#     -- the real FastConformer encoder (dw_striding subsampling, no bias, no xscale) + the new TdtDecoder
#     driver's greedy decode over it. Same "not regenerated at ctest time, skip cleanly if the real
#     checkpoint isn't prepared" pattern as the other real-model tests above. ---
add_executable(test_e2e_parakeet_tdt test_e2e_parakeet_tdt.cpp)
target_link_libraries(test_e2e_parakeet_tdt PRIVATE loom::engine)
add_test(NAME test_e2e_parakeet_tdt COMMAND test_e2e_parakeet_tdt)
set_tests_properties(test_e2e_parakeet_tdt PROPERTIES SKIP_RETURN_CODE 77)

# --- Real-model end-to-end test: nvidia/parakeet-rnnt-0.6b (converted by
#     tools/convert_nemo/convert_parakeet_rnnt.py) -- same FastConformer encoder architecture family as
#     parakeet-tdt (dw_striding subsampling) but biased/xscaled differently (confirmed real, see that
#     script's own docstring), decoded via TdtDecoder's new plain-RNN-T mode (empty `durations`). ---
add_executable(test_e2e_parakeet_rnnt test_e2e_parakeet_rnnt.cpp)
target_link_libraries(test_e2e_parakeet_rnnt PRIVATE loom::engine)
add_test(NAME test_e2e_parakeet_rnnt COMMAND test_e2e_parakeet_rnnt)
set_tests_properties(test_e2e_parakeet_rnnt PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro's CustomAlbert (PL-BERT) topology (tools/convert_kokoro/
#     convert_kokoro_albert.py) against a hand-rolled pure-PyTorch reference -- the first, isolated
#     verification step of Kokoro's full-continuous-build effort (see BACKLOG.md). ---
add_executable(test_e2e_kokoro_albert test_e2e_kokoro_albert.cpp)
target_link_libraries(test_e2e_kokoro_albert PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_albert COMMAND test_e2e_kokoro_albert)
set_tests_properties(test_e2e_kokoro_albert PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro's TextEncoder (CNN topology + the new loom::BiLstmStepper
#     host driver for its trailing bidirectional LSTM) against a hand-rolled pure-PyTorch reference. ---
add_executable(test_e2e_kokoro_text_encoder test_e2e_kokoro_text_encoder.cpp)
target_link_libraries(test_e2e_kokoro_text_encoder PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_text_encoder COMMAND test_e2e_kokoro_text_encoder)
set_tests_properties(test_e2e_kokoro_text_encoder PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro's ProsodyPredictor duration-prediction half (DurationEncoder's
#     interleaved BiLSTM+AdaLayerNorm blocks -> top lstm -> duration_proj) against a hand-rolled
#     pure-PyTorch reference. ---
add_executable(test_e2e_kokoro_duration_predictor test_e2e_kokoro_duration_predictor.cpp)
target_link_libraries(test_e2e_kokoro_duration_predictor PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_duration_predictor COMMAND test_e2e_kokoro_duration_predictor)
set_tests_properties(test_e2e_kokoro_duration_predictor PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: a single AdainResBlk1d instance (predictor.F0.0, the simplest case --
#     no learned shortcut, no upsample) against a hand-rolled pure-PyTorch reference -- first isolated
#     verification step for Kokoro's F0Ntrain assembly. ---
add_executable(test_e2e_kokoro_f0_block0 test_e2e_kokoro_f0_block0.cpp)
target_link_libraries(test_e2e_kokoro_f0_block0 PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_f0_block0 COMMAND test_e2e_kokoro_f0_block0)
set_tests_properties(test_e2e_kokoro_f0_block0 PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: predictor.F0.1, the UPSAMPLING AdainResBlk1d instance (dim_in=512,
#     dim_out=256, learned conv1x1 shortcut + upsample=True) -- exercises the depthwise-ConvTranspose1d
#     "pool" composition and the learned shortcut together for the first time. ---
add_executable(test_e2e_kokoro_f0_block1 test_e2e_kokoro_f0_block1.cpp)
target_link_libraries(test_e2e_kokoro_f0_block1 PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_f0_block1 COMMAND test_e2e_kokoro_f0_block1)
set_tests_properties(test_e2e_kokoro_f0_block1 PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: the FULL ProsodyPredictor.F0Ntrain assembly (shared BiLSTM 640->512 +
#     two independent 3-block AdainResBlk1d stacks + F0_proj/N_proj) against a hand-rolled pure-PyTorch
#     reference -- ties together every Kokoro F0Ntrain piece verified so far. ---
add_executable(test_e2e_kokoro_f0n test_e2e_kokoro_f0n.cpp)
target_link_libraries(test_e2e_kokoro_f0n PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_f0n COMMAND test_e2e_kokoro_f0n)
set_tests_properties(test_e2e_kokoro_f0n PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro Generator's forward/inverse STFT (istftnet.py's TorchSTFT),
#     against real torch.stft/torch.istft -- no real checkpoint weights involved (constant DFT/window
#     kernels only), but still gated on GGUF/reference files (needs real torch to generate). ---
add_executable(test_e2e_kokoro_stft test_e2e_kokoro_stft.cpp)
target_link_libraries(test_e2e_kokoro_stft PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_stft COMMAND test_e2e_kokoro_stft)
set_tests_properties(test_e2e_kokoro_stft PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro Generator's NSF harmonic source (istftnet.py's SineGen +
#     SourceModuleHnNSF), against a hand-rolled real-PyTorch reference. No real checkpoint weights
#     involved (l_linear is a fixed synthetic draw shared between conversion+reference scripts). ---
add_executable(test_e2e_kokoro_sinegen test_e2e_kokoro_sinegen.cpp)
target_link_libraries(test_e2e_kokoro_sinegen PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_sinegen COMMAND test_e2e_kokoro_sinegen)
set_tests_properties(test_e2e_kokoro_sinegen PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro Generator's AdaINResBlock1 (Snake-activation dilated resblock,
#     distinct from predictor.F0/N's AdainResBlk1d), against a hand-rolled PyTorch reference on a small
#     synthetic (checkpoint-independent) instance -- same precedent as VITS's test_hifigan_generator. ---
add_executable(test_e2e_kokoro_adainresblock1 test_e2e_kokoro_adainresblock1.cpp)
target_link_libraries(test_e2e_kokoro_adainresblock1 PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_adainresblock1 COMMAND test_e2e_kokoro_adainresblock1)
set_tests_properties(test_e2e_kokoro_adainresblock1 PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro's full Generator "core" (upsample stack + noise path +
#     resblocks + conv_post + inverse STFT), against a hand-rolled PyTorch reference on a small synthetic
#     (real checkpoint shapes, random weights) instance. ---
add_executable(test_e2e_kokoro_generator test_e2e_kokoro_generator.cpp)
target_link_libraries(test_e2e_kokoro_generator PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_generator COMMAND test_e2e_kokoro_generator)
set_tests_properties(test_e2e_kokoro_generator PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro's Decoder "core" (F0_conv/N_conv + encode/decode AdainResBlk1d
#     stack, everything except the final Generator call), against a hand-rolled PyTorch reference on a
#     small synthetic (real checkpoint shapes, random weights) instance. ---
add_executable(test_e2e_kokoro_decoder_core test_e2e_kokoro_decoder_core.cpp)
target_link_libraries(test_e2e_kokoro_decoder_core PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_decoder_core COMMAND test_e2e_kokoro_decoder_core)
set_tests_properties(test_e2e_kokoro_decoder_core PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Kokoro's bert_encoder (Linear(768,512) + transpose), against the real
#     checkpoint's own weights -- exercises the CustomAlbert-output (time-major) -> DurationEncoder-input
#     (channel-first) axis-convention crossing. ---
add_executable(test_e2e_kokoro_bert_encoder test_e2e_kokoro_bert_encoder.cpp)
target_link_libraries(test_e2e_kokoro_bert_encoder PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_bert_encoder COMMAND test_e2e_kokoro_bert_encoder)
set_tests_properties(test_e2e_kokoro_bert_encoder PROPERTIES SKIP_RETURN_CODE 77)

# --- End-to-end test of loom::KokoroDriver against the real checkpoint's full GGUF set -- exercises the
#     whole pipeline together for the first time. Finite/non-trivial output check only (same scope as
#     VITS's own test_e2e_vits_driver.cpp), not yet a full-pipeline numerical reference. ---
add_executable(test_e2e_kokoro_driver test_e2e_kokoro_driver.cpp)
target_link_libraries(test_e2e_kokoro_driver PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_driver COMMAND test_e2e_kokoro_driver)
set_tests_properties(test_e2e_kokoro_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Real-model structural smoke test: VITS (piper en-GB "miro"), converted via
#     tools/convert_piper_vits/convert_vits.py. Builds every declared topology with dummy inputs and
#     checks for finite output -- NOT yet a numerical-correctness check (that's still to come once the
#     host driver exists). Same skip-cleanly-if-not-prepared pattern as the other real-model tests. ---
add_executable(test_e2e_vits_smoke test_e2e_vits_smoke.cpp)
target_link_libraries(test_e2e_vits_smoke PRIVATE loom::engine)
add_test(NAME test_e2e_vits_smoke COMMAND test_e2e_vits_smoke)
set_tests_properties(test_e2e_vits_smoke PROPERTIES SKIP_RETURN_CODE 77)

# --- Real-model test: loom::VitsDriver end-to-end (TextEncoder -> StochasticDurationPredictor ->
#     host-side generate_path -> coupling flow -> HiFi-GAN vocoder), one level up from the structural
#     smoke test above -- exercises the full two-phase driver, not just each topology in isolation. ---
add_executable(test_e2e_vits_driver test_e2e_vits_driver.cpp)
target_link_libraries(test_e2e_vits_driver PRIVATE loom::engine)
add_test(NAME test_e2e_vits_driver COMMAND test_e2e_vits_driver)
set_tests_properties(test_e2e_vits_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: the deterministic `stats` (TextEncoder) topology against a real
#     PyTorch forward pass of piper's own models.TextEncoder, using REAL phonemized text (T=62, real
#     emb_rel_k/v pad branch exercised) -- see test_e2e_vits_stats_reference.cpp's own header comment. ---
add_executable(test_e2e_vits_stats_reference test_e2e_vits_stats_reference.cpp)
target_link_libraries(test_e2e_vits_stats_reference PRIVATE loom::engine)
add_test(NAME test_e2e_vits_stats_reference COMMAND test_e2e_vits_stats_reference)
set_tests_properties(test_e2e_vits_stats_reference PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: the flow+vocoder topology against a real PyTorch forward pass, with a
#     FIXED (non-random) z_p -- isolates the coupling-flow+vocoder wiring from any RNG-related
#     uncertainty in the full stochastic pipeline. See test_e2e_vits_flow_vocoder_reference.cpp. ---
add_executable(test_e2e_vits_flow_vocoder_reference test_e2e_vits_flow_vocoder_reference.cpp)
target_link_libraries(test_e2e_vits_flow_vocoder_reference PRIVATE loom::engine)
add_test(NAME test_e2e_vits_flow_vocoder_reference COMMAND test_e2e_vits_flow_vocoder_reference)
set_tests_properties(test_e2e_vits_flow_vocoder_reference PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: the logw (TextEncoder+SDP reverse) topology against a real PyTorch
#     forward pass, with FIXED (externally-injected) SDP noise -- isolates the real spline-flow assembly
#     from RNG-related uncertainty. See test_e2e_vits_logw_reference.cpp. ---
add_executable(test_e2e_vits_logw_reference test_e2e_vits_logw_reference.cpp)
target_link_libraries(test_e2e_vits_logw_reference PRIVATE loom::engine)
add_test(NAME test_e2e_vits_logw_reference COMMAND test_e2e_vits_logw_reference)
set_tests_properties(test_e2e_vits_logw_reference PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Whisper's AudioEncoder topology (tools/convert_whisper/
#     convert_whisper_encoder.py) against a real forward pass of OpenAI Whisper's own AudioEncoder --
#     fully deterministic (no sampling anywhere in the encoder), unlike VITS's noise-injection tests. ---
add_executable(test_e2e_whisper_encoder_reference test_e2e_whisper_encoder_reference.cpp)
target_link_libraries(test_e2e_whisper_encoder_reference PRIVATE loom::engine)
add_test(NAME test_e2e_whisper_encoder_reference COMMAND test_e2e_whisper_encoder_reference)
set_tests_properties(test_e2e_whisper_encoder_reference PROPERTIES SKIP_RETURN_CODE 77)

# --- Numerical-correctness check: Whisper's TextDecoder topology (tools/convert_whisper/
#     convert_whisper_decoder.py) against a real ONE-SHOT teacher-forced forward pass (kv_cache=None,
#     n_past=0/n_tokens=T covers the whole causal triangle) -- exercises causal self-attention w/
#     persistent KvCache, cross-attention against a fixed `xa`, and the tied output projection. ---
add_executable(test_e2e_whisper_decoder_reference test_e2e_whisper_decoder_reference.cpp)
target_link_libraries(test_e2e_whisper_decoder_reference PRIVATE loom::engine)
add_test(NAME test_e2e_whisper_decoder_reference COMMAND test_e2e_whisper_decoder_reference)
set_tests_properties(test_e2e_whisper_decoder_reference PROPERTIES SKIP_RETURN_CODE 77)

# --- End-to-end check: loom::WhisperDriver's full two-phase decode loop (encoder once, then
#     prefill+incremental decode with a persistent KvCache and a fixed cross-attention `xa`) against a
#     real plain-greedy-argmax reference sequence built directly from Whisper's own forward passes. ---
add_executable(test_e2e_whisper_driver test_e2e_whisper_driver.cpp)
target_link_libraries(test_e2e_whisper_driver PRIVATE loom::engine)
add_test(NAME test_e2e_whisper_driver COMMAND test_e2e_whisper_driver)
set_tests_properties(test_e2e_whisper_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- StyleTTS2 roadmap: loom::karras_schedule/adpm2_sample (the style-diffusion sampling loop's own
#     discretization math), verified against an independent numpy reference using a TOY affine denoiser
#     and REPLAYED (not freshly sampled) ancestral noise -- pure host arithmetic, no GGUF/ggml graph
#     involved, fully synthetic, not skip-if-missing. ---
set(LOOM_STYLE_DIFFUSION_SAMPLER_REF_DIR ${LOOM_FIXTURE_DIR}/style_diffusion_sampler_ref)

add_test(
    NAME gen_style_diffusion_sampler_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fixture_gen/reference_style_diffusion_sampler.py
            ${LOOM_STYLE_DIFFUSION_SAMPLER_REF_DIR}
)
set_tests_properties(gen_style_diffusion_sampler_reference PROPERTIES FIXTURES_SETUP style_diffusion_sampler_reference_fixture)

add_executable(test_style_diffusion_sampler test_style_diffusion_sampler.cpp)
target_link_libraries(test_style_diffusion_sampler PRIVATE loom::engine)
target_compile_definitions(test_style_diffusion_sampler PRIVATE
    LOOM_TEST_REF_DIR="${LOOM_STYLE_DIFFUSION_SAMPLER_REF_DIR}"
)
add_test(NAME test_style_diffusion_sampler COMMAND test_style_diffusion_sampler)
set_tests_properties(test_style_diffusion_sampler PROPERTIES FIXTURES_REQUIRED "style_diffusion_sampler_reference_fixture")

# --- Real-checkpoint test: StyleTTS2's style-diffusion denoiser network (Transformer1d) against a
#     hand-rolled PyTorch reference, real weights + synthetic driving embedding. See
#     tools/convert_styletts2/convert_styletts2_diffusion.py. Skips cleanly if env vars aren't set. ---
add_executable(test_e2e_styletts2_diffusion_net test_e2e_styletts2_diffusion_net.cpp)
target_link_libraries(test_e2e_styletts2_diffusion_net PRIVATE loom::engine)
add_test(NAME test_e2e_styletts2_diffusion_net COMMAND test_e2e_styletts2_diffusion_net)
set_tests_properties(test_e2e_styletts2_diffusion_net PROPERTIES SKIP_RETURN_CODE 77)

# --- Real-checkpoint test: the FULL style-diffusion sampling loop (KarrasSchedule + ADPM2Sampler +
#     KDiffusion preconditioning wrapping the real Transformer1d network) against a hand-rolled PyTorch
#     reference combining the same independently-verified pieces. See
#     tools/convert_styletts2/reference_forward_styletts2_diffusion_sampler.py. Skips cleanly if env vars
#     aren't set. ---
add_executable(test_e2e_styletts2_diffusion_sampler test_e2e_styletts2_diffusion_sampler.cpp)
target_link_libraries(test_e2e_styletts2_diffusion_sampler PRIVATE loom::engine)
add_test(NAME test_e2e_styletts2_diffusion_sampler COMMAND test_e2e_styletts2_diffusion_sampler)
set_tests_properties(test_e2e_styletts2_diffusion_sampler PROPERTIES SKIP_RETURN_CODE 77)

# --- End-to-end check: loom::StyleTTS2Driver's full synthesize() call (CustomAlbert -> style-diffusion
#     sampler -> bert_encoder -> DurationEncoder -> duration prediction/frame-expansion -> F0Ntrain ->
#     Decoder core -> SineGen -> forward STFT -> Generator core) against the real
#     yl4579/StyleTTS2-LJSpeech checkpoint -- finite/non-trivial output, same scope as
#     test_e2e_kokoro_driver.cpp. Skips cleanly if LOOM_STYLETTS2_DIR isn't set. ---
add_executable(test_e2e_styletts2_driver test_e2e_styletts2_driver.cpp)
target_link_libraries(test_e2e_styletts2_driver PRIVATE loom::engine)
add_test(NAME test_e2e_styletts2_driver COMMAND test_e2e_styletts2_driver)
set_tests_properties(test_e2e_styletts2_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: ConvNextBlock (replicate-pad composition), verified against the REAL
#     nn.Module's own forward pass (real package importable in this venv). See
#     tools/convert_supertonic/convert_supertonic_convnext.py. Skips cleanly if env vars aren't set. ---
add_executable(test_e2e_supertonic_convnext test_e2e_supertonic_convnext.cpp)
target_link_libraries(test_e2e_supertonic_convnext PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_convnext COMMAND test_e2e_supertonic_convnext)
set_tests_properties(test_e2e_supertonic_convnext PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: VFTimeEncoder (sinusoidal time embedding + Mish MLP), verified against
#     the real vector_estimator.pt's own time_encoder. See convert_supertonic_vftime.py. ---
add_executable(test_e2e_supertonic_vftime test_e2e_supertonic_vftime.cpp)
target_link_libraries(test_e2e_supertonic_vftime PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_vftime COMMAND test_e2e_supertonic_vftime)
set_tests_properties(test_e2e_supertonic_vftime PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: StyleEncoderCrossAttention (style-token-pooling cross-attention),
#     verified against the real dp-style-encoder.pt's own style_token_layer. See
#     convert_supertonic_style_attn.py. ---
add_executable(test_e2e_supertonic_style_attn test_e2e_supertonic_style_attn.cpp)
target_link_libraries(test_e2e_supertonic_style_attn PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_style_attn COMMAND test_e2e_supertonic_style_attn)
set_tests_properties(test_e2e_supertonic_style_attn PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: confirms VITS's own REL_POS_ATTENTION_SHAW primitive family is directly
#     reusable for SupertonicTTS's MultiHeadRelativeAttention (same Shaw et al. mechanism, different
#     window_size/channels). See convert_supertonic_relpos_attn.py. ---
add_executable(test_e2e_supertonic_relpos_attn test_e2e_supertonic_relpos_attn.cpp)
target_link_libraries(test_e2e_supertonic_relpos_attn PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_relpos_attn COMMAND test_e2e_supertonic_relpos_attn)
set_tests_properties(test_e2e_supertonic_relpos_attn PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: full DurationPredictor sub-model (DPTextEncoder + MLP head), the FIRST
#     complete coherent sub-model in this effort, verified against the real duration_predictor.pt. See
#     convert_supertonic_dp.py. ---
add_executable(test_e2e_supertonic_dp test_e2e_supertonic_dp.cpp)
target_link_libraries(test_e2e_supertonic_dp PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_dp COMMAND test_e2e_supertonic_dp)
set_tests_properties(test_e2e_supertonic_dp PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: TTLStyleEncoder + TTLTextEncoder (exercises SpeechPromptedCrossAttention
#     for the first time), verified against the real ttl-style-encoder.pt/text_encoder.pt. See
#     convert_supertonic_ttl_text.py. ---
add_executable(test_e2e_supertonic_ttl_text test_e2e_supertonic_ttl_text.cpp)
target_link_libraries(test_e2e_supertonic_ttl_text PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_ttl_text COMMAND test_e2e_supertonic_ttl_text)
set_tests_properties(test_e2e_supertonic_ttl_text PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: VFTextCrossAttention (fractional RoPE) + VFStyleCrossAttention, verified
#     against the real vector_estimator.pt's own text_attn[0]/style_attn[0]. See
#     convert_supertonic_vf_attn.py. ---
add_executable(test_e2e_supertonic_vf_attn test_e2e_supertonic_vf_attn.cpp)
target_link_libraries(test_e2e_supertonic_vf_attn PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_vf_attn COMMAND test_e2e_supertonic_vf_attn)
set_tests_properties(test_e2e_supertonic_vf_attn PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: the FULL VectorFieldEstimator (biggest single assembly in this effort),
#     verified against the real vector_estimator.pt, ONE compute_velocity call. See
#     convert_supertonic_vfe.py. ---
add_executable(test_e2e_supertonic_vfe test_e2e_supertonic_vfe.cpp)
target_link_libraries(test_e2e_supertonic_vfe PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_vfe COMMAND test_e2e_supertonic_vfe)
set_tests_properties(test_e2e_supertonic_vfe PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: the FULL CFM Euler sampling loop (loom::cfm_euler_sample + the real
#     VectorFieldEstimator), verified against a hand-rolled Python port calling the real
#     vector_estimator.pt's own .solve() in a loop. See
#     reference_forward_supertonic_cfm_sampler.py. ---
add_executable(test_e2e_supertonic_cfm_sampler test_e2e_supertonic_cfm_sampler.cpp)
target_link_libraries(test_e2e_supertonic_cfm_sampler PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_cfm_sampler COMMAND test_e2e_supertonic_cfm_sampler)
set_tests_properties(test_e2e_supertonic_cfm_sampler PROPERTIES SKIP_RETURN_CODE 77)

# --- SupertonicTTS v2 roadmap: SpeechDecoder (folded BatchNorm, causal ConvNeXt, direct waveform
#     emission, the FINAL pipeline stage), verified against the real vocoder.pt. See
#     convert_supertonic_decoder.py. ---
add_executable(test_e2e_supertonic_decoder test_e2e_supertonic_decoder.cpp)
target_link_libraries(test_e2e_supertonic_decoder PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_decoder COMMAND test_e2e_supertonic_decoder)
set_tests_properties(test_e2e_supertonic_decoder PROPERTIES SKIP_RETURN_CODE 77)

# --- End-to-end check: loom::SupertonicDriver's full synthesize() call (DurationPredictor ->
#     get_latent_mask -> TTLTextEncoder -> Euler CFM sampling loop -> SpeechDecoder) against the real
#     femelo/supertonic-tts checkpoint -- finite/non-trivial output, same scope as
#     test_e2e_kokoro_driver.cpp/test_e2e_styletts2_driver.cpp. Skips cleanly if env vars aren't set. ---
add_executable(test_e2e_supertonic_driver test_e2e_supertonic_driver.cpp)
target_link_libraries(test_e2e_supertonic_driver PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_driver COMMAND test_e2e_supertonic_driver)
set_tests_properties(test_e2e_supertonic_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Matcha-TTS roadmap: TextEncoder (ConvReluNorm prenet, partial-rotary NeoX RoPE self-attention,
#     per-token DurationPredictor) against the real matcha.models.components.text_encoder.TextEncoder
#     module run directly. See tools/convert_matcha/convert_matcha_text_encoder.py. ---
add_executable(test_e2e_matcha_text_encoder test_e2e_matcha_text_encoder.cpp)
target_link_libraries(test_e2e_matcha_text_encoder PRIVATE loom::engine)
add_test(NAME test_e2e_matcha_text_encoder COMMAND test_e2e_matcha_text_encoder)
set_tests_properties(test_e2e_matcha_text_encoder PROPERTIES SKIP_RETURN_CODE 77)

# --- Matcha-TTS roadmap: Decoder U-Net (the CFM estimator -- GROUP_NORM, one real downsample/upsample,
#     skip connections, SnakeBeta-FeedForward BasicTransformerBlocks) against the real
#     matcha.models.components.decoder.Decoder module run directly. See
#     tools/convert_matcha/convert_matcha_decoder.py. ---
add_executable(test_e2e_matcha_decoder test_e2e_matcha_decoder.cpp)
target_link_libraries(test_e2e_matcha_decoder PRIVATE loom::engine)
add_test(NAME test_e2e_matcha_decoder COMMAND test_e2e_matcha_decoder)
set_tests_properties(test_e2e_matcha_decoder PROPERTIES SKIP_RETURN_CODE 77)

# --- Matcha-TTS roadmap: the real HiFi-GAN v1 vocoder (resblock="1", 4 upsample stages -- a different
#     topology shape from VITS's own resblock="2"/3-stage HiFi-GAN) against the real
#     matcha.hifigan.models.Generator module run directly. See
#     tools/convert_matcha/convert_matcha_vocoder.py. ---
add_executable(test_e2e_matcha_vocoder test_e2e_matcha_vocoder.cpp)
target_link_libraries(test_e2e_matcha_vocoder PRIVATE loom::engine)
add_test(NAME test_e2e_matcha_vocoder COMMAND test_e2e_matcha_vocoder)
set_tests_properties(test_e2e_matcha_vocoder PROPERTIES SKIP_RETURN_CODE 77)

# --- End-to-end check: loom::MatchaDriver's full synthesize() call (TextEncoder -> duration expansion
#     -> Euler CFM sampling loop over the Decoder U-Net -> denormalize -> HiFi-GAN v1 vocoder) against
#     the real matcha_ljspeech.ckpt + generator_v1 checkpoints -- finite/non-trivial output, same scope
#     as test_e2e_supertonic_driver.cpp/test_e2e_kokoro_driver.cpp. Skips cleanly if env vars aren't set. ---
add_executable(test_e2e_matcha_driver test_e2e_matcha_driver.cpp)
target_link_libraries(test_e2e_matcha_driver PRIVATE loom::engine)
add_test(NAME test_e2e_matcha_driver COMMAND test_e2e_matcha_driver)
set_tests_properties(test_e2e_matcha_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Procedural generalization: LoomLuaBridge's host-math bindings (loom.range/causal_mask/zero_mask/
#     argmax_row) and load_script()/call() plumbing, in isolation (no GGUF needed). See
#     LOOM_PROCEDURAL_GENERALIZATION.md / LOOM_MIL_CONVERSION.md and BACKLOG.md's dated entry. ---
add_executable(test_lua_bridge test_lua_bridge.cpp)
target_link_libraries(test_lua_bridge PRIVATE loom::engine)
add_test(NAME test_lua_bridge COMMAND test_lua_bridge)

# --- Procedural generalization validation: runs the real Whisper checkpoint's transcribe() through BOTH
#     the existing hand-written loom::WhisperDriver AND a LoomLuaBridge running the hand-ported
#     tools/convert_whisper/whisper_driver.lua, asserting an EXACT generated-token-sequence match. See
#     LOOM_PROCEDURAL_GENERALIZATION.md / LOOM_MIL_CONVERSION.md and BACKLOG.md's dated entry. ---
add_executable(test_e2e_whisper_lua_driver test_e2e_whisper_lua_driver.cpp)
target_link_libraries(test_e2e_whisper_lua_driver PRIVATE loom::engine)
add_test(NAME test_e2e_whisper_lua_driver COMMAND test_e2e_whisper_lua_driver)
set_tests_properties(test_e2e_whisper_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Procedural generalization: SupertonicTTS's deterministic Euler-ODE (CFM) sampling loop, ported to
#     Lua (tools/convert_supertonic/supertonic_driver.lua) and compared against the existing
#     loom::SupertonicDriver on the real checkpoint. See LOOM_PROCEDURAL_GENERALIZATION.md /
#     LOOM_MIL_CONVERSION.md and BACKLOG.md's dated entry. ---
add_executable(test_e2e_supertonic_lua_driver test_e2e_supertonic_lua_driver.cpp)
target_link_libraries(test_e2e_supertonic_lua_driver PRIVATE loom::engine)
add_test(NAME test_e2e_supertonic_lua_driver COMMAND test_e2e_supertonic_lua_driver)
set_tests_properties(test_e2e_supertonic_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Procedural generalization: Matcha-TTS's deterministic Euler-ODE (CFM) sampling loop + real
#     per-token duration expansion, ported to Lua (tools/convert_matcha/matcha_driver.lua) and compared
#     against the existing loom::MatchaDriver on the real checkpoint. See
#     LOOM_PROCEDURAL_GENERALIZATION.md / LOOM_MIL_CONVERSION.md and BACKLOG.md's dated entry. ---
add_executable(test_e2e_matcha_lua_driver test_e2e_matcha_lua_driver.cpp)
target_link_libraries(test_e2e_matcha_lua_driver PRIVATE loom::engine)
add_test(NAME test_e2e_matcha_lua_driver COMMAND test_e2e_matcha_lua_driver)
set_tests_properties(test_e2e_matcha_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Procedural generalization: VITS's dynamic-length relative-position table cropping (loom.get_weight
#     + loom.pad_crop_relative_embeddings) + Gaussian-noise-interleaved duration expansion, ported to Lua
#     (tools/convert_piper_vits/vits_driver.lua) and compared against the existing loom::VitsDriver on
#     the real checkpoint. See LOOM_PROCEDURAL_GENERALIZATION.md / LOOM_MIL_CONVERSION.md and
#     BACKLOG.md's dated entry. ---
add_executable(test_e2e_vits_lua_driver test_e2e_vits_lua_driver.cpp)
target_link_libraries(test_e2e_vits_lua_driver PRIVATE loom::engine)
add_test(NAME test_e2e_vits_lua_driver COMMAND test_e2e_vits_lua_driver)
set_tests_properties(test_e2e_vits_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Procedural generalization: Kokoro-82M, the largest driver ported so far (43 topologies, 6 BiLSTM
#     instances host-stepped via loom.run_subgraph Lua for-loops, the new loom.uniform_array binding for
#     SineGen's rand_ini draws), ported to Lua (tools/convert_kokoro/kokoro_driver.lua) and compared
#     against the existing loom::KokoroDriver on the real checkpoint. See
#     LOOM_PROCEDURAL_GENERALIZATION.md / LOOM_MIL_CONVERSION.md and BACKLOG.md's dated entry. ---
add_executable(test_e2e_kokoro_lua_driver test_e2e_kokoro_lua_driver.cpp)
target_link_libraries(test_e2e_kokoro_lua_driver PRIVATE loom::engine)
add_test(NAME test_e2e_kokoro_lua_driver COMMAND test_e2e_kokoro_lua_driver)
set_tests_properties(test_e2e_kokoro_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Procedural generalization: StyleTTS2, the most complex driver ported so far (44 topologies, 6
#     BiLSTM instances, and the genuinely new style-diffusion sampler -- ADPM2 over the real
#     Transformer1d denoiser, ported to Lua as plain host arithmetic + loom.run_subgraph calls), compared
#     against the existing loom::StyleTTS2Driver on the real checkpoint. See
#     LOOM_PROCEDURAL_GENERALIZATION.md / LOOM_MIL_CONVERSION.md and BACKLOG.md's dated entry. ---
add_executable(test_e2e_styletts2_lua_driver test_e2e_styletts2_lua_driver.cpp)
target_link_libraries(test_e2e_styletts2_lua_driver PRIVATE loom::engine)
add_test(NAME test_e2e_styletts2_lua_driver COMMAND test_e2e_styletts2_lua_driver)
set_tests_properties(test_e2e_styletts2_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- Procedural generalization: LFM2-350M fully dynamic multi-topology model compiled via coremltools
#     and Loom MIL compiler backend, executing dynamically via embedded LuaVM and GraphBuilder ---
add_executable(test_e2e_lfm2_lua_driver test_e2e_lfm2_lua_driver.cpp)
target_link_libraries(test_e2e_lfm2_lua_driver PRIVATE loom::engine)
add_test(NAME test_e2e_lfm2_lua_driver COMMAND test_e2e_lfm2_lua_driver)
set_tests_properties(test_e2e_lfm2_lua_driver PROPERTIES SKIP_RETURN_CODE 77)

# --- LFM2-350M exported via the generic MIL compiler (export_lfm2_atomic.py/export_lfm2_monolithic.py),
#     verified at genuinely dynamic (non-fixed, unpadded) prompt lengths against real HF logits ---
add_executable(test_e2e_lfm2_mil_export test_e2e_lfm2_mil_export.cpp)
target_link_libraries(test_e2e_lfm2_mil_export PRIVATE loom::engine)
add_test(NAME test_e2e_lfm2_mil_export COMMAND test_e2e_lfm2_mil_export)
set_tests_properties(test_e2e_lfm2_mil_export PROPERTIES SKIP_RETURN_CODE 77)

# --- LSTM export via recurrent.py's build_lstm_cell_topologies + LoomLuaBridge::l_run_recurrent
#     (EXPORT-IMPROVEMENT-BACKLOG.md item 4), verified against a real bidirectional torch.nn.LSTM ---
add_executable(test_e2e_lstm_recurrent test_e2e_lstm_recurrent.cpp)
target_link_libraries(test_e2e_lstm_recurrent PRIVATE loom::engine)
add_test(NAME test_e2e_lstm_recurrent COMMAND test_e2e_lstm_recurrent)
set_tests_properties(test_e2e_lstm_recurrent PROPERTIES SKIP_RETURN_CODE 77)

# --- Same LFM2-350M monolithic export's new tokenizer.ggml.* KVs (EXPORT-BACKLOG.md item 4) --
#     loom::BpeVocab::encode/decode checked against the real HF AutoTokenizer on the same checkpoint ---
add_executable(test_e2e_lfm2_tokenizer test_e2e_lfm2_tokenizer.cpp)
target_link_libraries(test_e2e_lfm2_tokenizer PRIVATE loom::engine)
add_test(NAME test_e2e_lfm2_tokenizer COMMAND test_e2e_lfm2_tokenizer)
set_tests_properties(test_e2e_lfm2_tokenizer PROPERTIES SKIP_RETURN_CODE 77)

# --- Same LFM2-350M monolithic export, quantized to Q8_0 via the exporter's `quantize=` kwarg
#     (EXPORT-BACKLOG.md item 6) -- verifies real per-logit closeness against the F32 reference above ---
add_executable(test_e2e_lfm2_q8_0 test_e2e_lfm2_q8_0.cpp)
target_link_libraries(test_e2e_lfm2_q8_0 PRIVATE loom::engine)
add_test(NAME test_e2e_lfm2_q8_0 COMMAND test_e2e_lfm2_q8_0)
set_tests_properties(test_e2e_lfm2_q8_0 PROPERTIES SKIP_RETURN_CODE 77)
