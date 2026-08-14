# The ggml revision this project builds against, alone in a file so that it can be read by builds
# outside this repo.
#
# It is here rather than inline in Dependencies.cmake for one reason: an accelerator package
# (`loom-py-rt-vulkan`, `loom-py-rt-cuda` -- BACKLOG.md P4.8) ships a `libggml-vulkan.so` built from
# these sources, to be loaded at run time next to a `libggml-base.so` built from them by somebody
# else. ggml offers no ABI guarantee across versions, so those two builds agreeing is not a nicety --
# a mismatch is a loaded backend whose symbols do not line up with the base library it links.
#
# The Python side of that agreement is an `==` pin between the base wheel's version and each backend
# package's. This is the build side: a backend package includes THIS file out of the engine checkout
# it was built from, so there is no second copy of the tag to drift.
set(LOOM_GGML_REPOSITORY "https://github.com/ggml-org/ggml.git")
# v0.16.0 == commit 524f974bb21a1013408f76d71c15732482c0c3fe.
set(LOOM_GGML_TAG "v0.16.0")
