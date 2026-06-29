add_executable(noorray_test
    main.cpp
)
target_compile_definitions(noorray_test PRIVATE
    TEST_SCENE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(NAME noorray_e2e
    COMMAND noorray_test
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
