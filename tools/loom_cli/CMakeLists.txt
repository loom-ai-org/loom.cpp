add_executable(loom_cli main.cpp wav_file.cpp)
target_link_libraries(loom_cli PRIVATE loom::engine)
