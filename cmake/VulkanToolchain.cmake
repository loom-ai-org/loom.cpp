# The two build-host tools ggml's Vulkan backend needs, and what to do when the machine's are too old.
#
# ---------------------------------------------------------------------------------------------------
# WHY THIS FILE EXISTS (BACKLOG.md P4.7)
# ---------------------------------------------------------------------------------------------------
# `-DGGML_VULKAN=ON` fails on a stable distribution, twice, in ways that name neither cause:
#
#   * **glslc.** ggml probes for `GL_KHR_cooperative_matrix` by running glslc on a feature-test shader
#     and grepping stderr for "extension not supported". Shaderc 2023.2 does not emit that string -- it
#     fails differently -- so the probe concludes the extension IS supported, generates the coopmat
#     shader variants, and the build dies in `conv2d_mm.comp` with `'coopmat': undeclared identifier`.
#     The `GGML_VULKAN_COOPMAT_GLSLC_SUPPORT=OFF` switch does not help: ggml's own CMake overwrites it
#     from the probe.
#   * **Vulkan-Headers.** `ggml-vulkan.cpp` uses `VkPhysicalDeviceCooperativeMatrixFeaturesKHR`
#     (Vulkan-Headers 1.3.264+) and `vk::LayerSettingEXT` (1.3.272+). Debian bookworm ships 1.3.239.
#     The headers are header-only and the Vulkan loader is ABI-stable, so newer headers against the
#     system `libvulkan.so.1` is a supported arrangement, not a hack.
#
# ---------------------------------------------------------------------------------------------------
# FETCHCONTENT, NOT A SUBMODULE
# ---------------------------------------------------------------------------------------------------
# Same answer as ggml, nlohmann_json and LuaJIT: a pinned `GIT_TAG` gives the "update it when we need
# to" property a submodule would, without a `--recursive` clone every consumer has to remember and
# without carrying a gitlink for something only one build configuration uses. `Dependencies.cmake`
# states that policy for the repo; this file follows it.
#
# ---------------------------------------------------------------------------------------------------
# AND ONLY WHEN THE MACHINE ACTUALLY NEEDS IT
# ---------------------------------------------------------------------------------------------------
# Both probes below test the FAILURE this file exists to prevent, not a version number, because a
# version number is a proxy that goes stale in both directions -- a backported distro package and a
# vendor SDK with its own numbering are both real. A machine that passes both probes builds against its
# own toolchain and fetches nothing; building shaderc from source takes minutes and nobody should pay
# that for a working glslc they already have.

include(FetchContent)
include(CheckCXXSourceCompiles)

# Turning this OFF makes an inadequate toolchain a configure ERROR naming what to install, rather than
# something this build quietly fixes. That is the right setting for a CI image that is supposed to
# provide its own toolchain, where a silent 10-minute source build is a worse outcome than a red X.
option(LOOM_VULKAN_FETCH_TOOLCHAIN
       "Build glslc / fetch Vulkan-Headers when the system ones are too old for ggml's Vulkan backend"
       ON)

# Pinned deliberately, and bumped deliberately. shaderc's own `utils/git-sync-deps` resolves glslang,
# SPIRV-Tools, SPIRV-Headers and abseil from the DEPS file at THIS revision, so one pin fixes the whole
# toolchain and its internal consistency is upstream's problem rather than ours.
set(LOOM_SHADERC_TAG "v2025.4" CACHE STRING "shaderc revision to build glslc from")
set(LOOM_VULKAN_HEADERS_TAG "v1.4.321" CACHE STRING "Vulkan-Headers revision")
# ggml's Vulkan CMakeLists does `find_package(SPIRV-Headers CONFIG REQUIRED)`, so this has to be
# resolvable even on a machine whose glslc and Vulkan headers are both fine.
set(LOOM_SPIRV_HEADERS_TAG "vulkan-sdk-1.4.321.0" CACHE STRING "SPIRV-Headers revision")

# Does `glslc` answer ggml's cooperative-matrix probe the way ggml assumes? Runs the real feature-test
# shader from the ggml sources already fetched, so this asks about the exact file that will decide it.
# "Answers correctly" means one of two things: it compiles the shader, or it refuses in the words ggml
# greps for. Anything else -- a syntax error, an unknown identifier -- is a glslc that will lie to the
# probe and then fail the build.
function(loom_glslc_is_usable GLSLC_EXECUTABLE RESULT_VAR)
    set(${RESULT_VAR} FALSE PARENT_SCOPE)
    if(NOT GLSLC_EXECUTABLE OR NOT EXISTS "${GLSLC_EXECUTABLE}")
        return()
    endif()
    set(probe_shader "${ggml_SOURCE_DIR}/src/ggml-vulkan/vulkan-shaders/feature-tests/coopmat.comp")
    if(NOT EXISTS "${probe_shader}")
        # A ggml layout this file has not seen. Trust the machine rather than fetch on a guess: a wrong
        # "too old" here costs a source build nobody needed, and the real build will still say so.
        set(${RESULT_VAR} TRUE PARENT_SCOPE)
        return()
    endif()
    execute_process(
        COMMAND "${GLSLC_EXECUTABLE}" -o - -fshader-stage=compute --target-env=vulkan1.3 "${probe_shader}"
        RESULT_VARIABLE probe_result
        OUTPUT_VARIABLE probe_stdout
        ERROR_VARIABLE probe_stderr
    )
    if(probe_result EQUAL 0 OR probe_stderr MATCHES "extension not supported: GL_KHR_cooperative_matrix")
        set(${RESULT_VAR} TRUE PARENT_SCOPE)
    endif()
endfunction()

# Do the Vulkan headers on the include path carry what ggml-vulkan.cpp names? Compiled rather than
# version-compared, for the reason in this file's header.
function(loom_vulkan_headers_are_usable RESULT_VAR)
    set(CMAKE_REQUIRED_QUIET TRUE)
    check_cxx_source_compiles("
        #include <vulkan/vulkan_core.h>
        int main() {
            VkPhysicalDeviceCooperativeMatrixFeaturesKHR f{};
            f.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
            return (int)f.sType;
        }" LOOM_VK_HEADERS_HAVE_COOPMAT)
    check_cxx_source_compiles("
        #include <vulkan/vulkan.hpp>
        int main() { vk::LayerSettingEXT s{}; (void)s; return 0; }" LOOM_VK_HEADERS_HAVE_LAYER_SETTING)
    if(LOOM_VK_HEADERS_HAVE_COOPMAT AND LOOM_VK_HEADERS_HAVE_LAYER_SETTING)
        set(${RESULT_VAR} TRUE PARENT_SCOPE)
    else()
        set(${RESULT_VAR} FALSE PARENT_SCOPE)
    endif()
endfunction()

# Entry point. Call AFTER ggml's sources are populated (the glslc probe reads a shader out of them) and
# BEFORE ggml is added as a subdirectory (everything it sets has to be in place for ggml's own
# `find_package(Vulkan COMPONENTS glslc)` and for its compile lines).
function(loom_setup_vulkan_toolchain)
    # SPIRV-Headers is needed by ggml's Vulkan CMakeLists whatever the rest of the machine looks like,
    # and it is needed in two ways. `find_package(SPIRV-Headers CONFIG REQUIRED)` is satisfied by
    # FetchContent's own find_package redirect; the INCLUDE PATH is not, because ggml never links the
    # target it just found -- so `spirv/unified1/spirv.hpp` is only on the compile line if something
    # puts it there, and on a machine with no system spirv-headers package nothing does. That is the
    # `'spv' has not been declared` error, and this include_directories is the fix.
    FetchContent_Declare(SPIRV-Headers
        GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
        GIT_TAG ${LOOM_SPIRV_HEADERS_TAG}
        GIT_SHALLOW TRUE
    )
    set(SPIRV_HEADERS_SKIP_EXAMPLES ON CACHE BOOL "" FORCE)
    set(SPIRV_HEADERS_SKIP_INSTALL ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(SPIRV-Headers)
    include_directories(SYSTEM "${spirv-headers_SOURCE_DIR}/include")

    # ...and a package config for it, because FetchContent's own find_package redirect does not appear
    # for this dependency: SPIRV-Headers defines its config through `install(EXPORT)`, which produces
    # nothing in a build tree that never installs, and CMake leaves the redirect alone once a
    # subproject has taken responsibility for its own config. So ggml's `find_package(SPIRV-Headers
    # CONFIG REQUIRED)` finds neither and stops. The targets it would import already exist by now --
    # `add_subdirectory` created them -- so this file has nothing to do but say so.
    set(spirv_headers_config_dir "${CMAKE_BINARY_DIR}/loom-cmake/SPIRV-Headers")
    file(WRITE "${spirv_headers_config_dir}/SPIRV-HeadersConfig.cmake"
         "# Generated by loom.cpp's cmake/VulkanToolchain.cmake.\n"
         "# The SPIRV-Headers targets are already defined by the FetchContent'd subproject in this same\n"
         "# build; this exists only so find_package(SPIRV-Headers CONFIG) can succeed against it.\n"
         "if(NOT TARGET SPIRV-Headers::SPIRV-Headers)\n"
         "    message(FATAL_ERROR \"SPIRV-Headers targets are missing from this build\")\n"
         "endif()\n"
         "set(SPIRV-Headers_FOUND TRUE)\n")
    set(SPIRV-Headers_DIR "${spirv_headers_config_dir}" CACHE PATH "" FORCE)

    loom_vulkan_headers_are_usable(headers_ok)
    find_program(Vulkan_GLSLC_EXECUTABLE NAMES glslc)
    loom_glslc_is_usable("${Vulkan_GLSLC_EXECUTABLE}" glslc_ok)

    if(headers_ok AND glslc_ok)
        message(STATUS "Vulkan toolchain: using the system glslc and headers")
        return()
    endif()

    if(NOT LOOM_VULKAN_FETCH_TOOLCHAIN)
        message(FATAL_ERROR
            "This machine's Vulkan build tools are too old for ggml's Vulkan backend "
            "(headers ok: ${headers_ok}, glslc ok: ${glslc_ok}). Install a current Vulkan SDK or a "
            "newer 'glslc'/'libvulkan-dev', or re-run with -DLOOM_VULKAN_FETCH_TOOLCHAIN=ON to build "
            "them here. See cmake/VulkanToolchain.cmake.")
    endif()

    if(NOT headers_ok)
        message(STATUS "Vulkan toolchain: system Vulkan headers are too old, fetching ${LOOM_VULKAN_HEADERS_TAG}")
        FetchContent_Declare(VulkanHeaders
            GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
            GIT_TAG ${LOOM_VULKAN_HEADERS_TAG}
            GIT_SHALLOW TRUE
        )
        FetchContent_Populate(VulkanHeaders)
        # Ahead of the system ones on the compile line. `Vulkan::Vulkan` carries /usr/include as an
        # INTERFACE directory, and a target's own -I entries are emitted before any linked target's, so
        # this wins without having to take the system headers away from anything else.
        include_directories(SYSTEM "${vulkanheaders_SOURCE_DIR}/include")
    endif()

    if(NOT glslc_ok)
        loom_build_glslc(built_glslc)
        # ggml bakes ${Vulkan_GLSLC_EXECUTABLE} into custom commands at CONFIGURE time, and -- more
        # importantly -- it RUNS it at configure time for five `test_shader_extension_support` probes
        # whose answers decide which shader variants get generated and which compile definitions get
        # set. That is why glslc is built here and now rather than as an ExternalProject: a binary that
        # does not exist yet makes every one of those probes fail to launch, which ggml reads as
        # "supported" and acts on. The cost is a slow first configure on a machine that needed this;
        # the alternative is five probes answering from a process that never ran.
        set(Vulkan_GLSLC_EXECUTABLE "${built_glslc}" CACHE FILEPATH "" FORCE)
    endif()
endfunction()

# Builds glslc from a pinned shaderc, at configure time, into the build tree. Idempotent: an existing
# binary that answers the probe is reused, so only the first configure of a build directory pays.
function(loom_build_glslc RESULT_VAR)
    set(prefix "${CMAKE_BINARY_DIR}/vulkan-toolchain")
    set(glslc "${prefix}/bin/glslc")
    loom_glslc_is_usable("${glslc}" already_built)
    if(already_built)
        message(STATUS "Vulkan toolchain: reusing the glslc already built in ${prefix}")
        set(${RESULT_VAR} "${glslc}" PARENT_SCOPE)
        return()
    endif()

    message(STATUS "Vulkan toolchain: building shaderc ${LOOM_SHADERC_TAG} -- several minutes, once "
                   "per build directory. Pass -DLOOM_VULKAN_FETCH_TOOLCHAIN=OFF to require a system "
                   "glslc instead.")
    FetchContent_Declare(shaderc
        GIT_REPOSITORY https://github.com/google/shaderc.git
        GIT_TAG ${LOOM_SHADERC_TAG}
        GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(shaderc)
    if(NOT shaderc_POPULATED)
        FetchContent_Populate(shaderc)
    endif()

    # shaderc pins glslang, SPIRV-Tools, SPIRV-Headers and abseil in its own DEPS file and ships the
    # script that resolves them. Using it means the single tag above fixes the whole toolchain, and
    # keeping those four consistent with each other stays upstream's problem rather than ours.
    execute_process(
        COMMAND ${Python3_EXECUTABLE} "${shaderc_SOURCE_DIR}/utils/git-sync-deps"
        WORKING_DIRECTORY "${shaderc_SOURCE_DIR}"
        RESULT_VARIABLE sync_result
        OUTPUT_VARIABLE sync_output
        ERROR_VARIABLE sync_output
    )
    if(NOT sync_result EQUAL 0)
        message(FATAL_ERROR "Vulkan toolchain: shaderc's git-sync-deps failed:\n${sync_output}")
    endif()

    # A nested configure+build, deliberately isolated from this one: shaderc brings its own glslang,
    # SPIRV-Tools and abseil, and add_subdirectory'ing that into a build that also carries ggml's
    # SPIRV-Headers is a target-name collision waiting to happen. Nothing of shaderc's needs to reach
    # this project except one executable.
    execute_process(
        COMMAND ${CMAKE_COMMAND} -S "${shaderc_SOURCE_DIR}" -B "${prefix}/build"
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_INSTALL_PREFIX=${prefix}
                -DSHADERC_SKIP_TESTS=ON
                -DSHADERC_SKIP_EXAMPLES=ON
                -DSHADERC_SKIP_COPYRIGHT_CHECK=ON
        RESULT_VARIABLE configure_result
        OUTPUT_VARIABLE configure_output
        ERROR_VARIABLE configure_output
    )
    if(NOT configure_result EQUAL 0)
        message(FATAL_ERROR "Vulkan toolchain: configuring shaderc failed:\n${configure_output}")
    endif()

    include(ProcessorCount)
    ProcessorCount(nproc)
    if(nproc EQUAL 0)
        set(nproc 1)
    endif()
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build "${prefix}/build" --target install --parallel ${nproc}
        RESULT_VARIABLE build_result
        OUTPUT_VARIABLE build_output
        ERROR_VARIABLE build_output
    )
    if(NOT build_result EQUAL 0)
        # The tail, not the head: a compiler error is at the end of a build log, and the whole of one
        # of these is tens of thousands of lines.
        string(LENGTH "${build_output}" output_length)
        if(output_length GREATER 4000)
            math(EXPR tail_start "${output_length} - 4000")
            string(SUBSTRING "${build_output}" ${tail_start} -1 build_output)
        endif()
        message(FATAL_ERROR "Vulkan toolchain: building shaderc failed:\n${build_output}")
    endif()

    loom_glslc_is_usable("${glslc}" built_ok)
    if(NOT built_ok)
        message(FATAL_ERROR "Vulkan toolchain: built ${glslc}, and it still does not answer ggml's "
                            "cooperative-matrix probe. Bump LOOM_SHADERC_TAG.")
    endif()
    set(${RESULT_VAR} "${glslc}" PARENT_SCOPE)
endfunction()
