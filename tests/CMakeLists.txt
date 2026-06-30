add_executable(noorray_test
    main.cpp
)
target_compile_definitions(noorray_test PRIVATE
    TEST_SCENE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

add_test(NAME noorray_e2e
    COMMAND noorray_test
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")

add_executable(sampler_test
    sampler.cpp
)
target_include_directories(sampler_test PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
)
add_test(NAME sampler_regression COMMAND sampler_test)
