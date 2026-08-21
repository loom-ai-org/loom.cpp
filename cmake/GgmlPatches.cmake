# Source patches applied to the pinned ggml checkout at configure time.
#
# WHY THIS EXISTS AT ALL. ggml is a pinned upstream dependency (cmake/GgmlPin.cmake) and the standing
# rule is that per-model complexity belongs in the exporter and per-task C++ belongs in this engine --
# neither of which describes "the F32 GEMM micro-kernel is slow on ARM". That is a ggml-side concern,
# and the right home for a fix is upstream. But a fix upstream lands on upstream's schedule, and the
# thing it fixes is 71% of the loom-vs-onnxruntime gap on a Raspberry Pi (BACKLOG.md P4.14/P4.15), so
# each patch here is carried locally until the pin can be bumped past its upstream equivalent.
#
# The contract for anything added to `cmake/patches/`:
#   * it is a diff against the EXACT pin, so bumping `LOOM_GGML_TAG` makes it fail loudly at configure
#     time rather than silently no-op -- see the FATAL_ERROR below, which is the whole point;
#   * it carries its measurement in its own comment block, the way code in this repo does;
#   * it is submitted upstream, and deleted from here once the pin includes it.
#
# Applied on EVERY configure rather than through FetchContent's PATCH_COMMAND, which runs only on
# populate: an existing build tree (a developer's, a CI cache) is already populated, so a PATCH_COMMAND
# added today would never run there and the tree would build unpatched with nothing saying so.
function(loom_patch_ggml SOURCE_DIR)
    find_package(Git QUIET REQUIRED)
    file(GLOB patches "${CMAKE_CURRENT_LIST_DIR}/patches/ggml-*.patch")
    list(SORT patches)
    foreach(patch ${patches})
        get_filename_component(name "${patch}" NAME)
        # `--reverse --check` succeeds exactly when the patch is ALREADY applied, which is what makes
        # re-configuring an existing tree a no-op instead of an error.
        execute_process(
            COMMAND ${GIT_EXECUTABLE} apply --reverse --check --quiet "${patch}"
            WORKING_DIRECTORY "${SOURCE_DIR}"
            RESULT_VARIABLE already
            ERROR_QUIET OUTPUT_QUIET
        )
        if(already EQUAL 0)
            message(STATUS "ggml patch already applied: ${name}")
            continue()
        endif()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} apply "${patch}"
            WORKING_DIRECTORY "${SOURCE_DIR}"
            RESULT_VARIABLE applied
            ERROR_VARIABLE err
        )
        if(NOT applied EQUAL 0)
            message(FATAL_ERROR
                "failed to apply ${name} to the ggml checkout at ${SOURCE_DIR}:\n${err}\n"
                "This is what a ggml bump looks like from here. Either the change is now upstream -- "
                "in which case delete the patch -- or it still is not, in which case rebase it onto "
                "${LOOM_GGML_TAG} and re-run the measurement in its header before trusting it.")
        endif()
        message(STATUS "ggml patch applied: ${name}")
    endforeach()
endfunction()
