# Third-party dependencies, pulled via FetchContent so no submodules/vendored copies are needed.
include(FetchContent)

# ggml: the tensor library / graph runtime this engine is built on.
# Pinned to v0.16.0 (commit 524f974bb21a1013408f76d71c15732482c0c3fe) for reproducible builds.
FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggml-org/ggml.git
    GIT_TAG        v0.16.0
)

# nlohmann/json: parses the JSON graph-topology definition embedded in GGUF metadata.
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")

# ggml defaults GGML_BUILD_TESTS/EXAMPLES to ON when built standalone; force them off since we only
# want the library here (keeps configure/build time down and avoids pulling ggml's own test deps).
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(ggml nlohmann_json)
