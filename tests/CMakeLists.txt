find_package(Catch2 3 REQUIRED)

file(GLOB_RECURSE test_files CONFIGURE_DEPENDS *.cpp)

add_executable(pong_ai_tests ${test_files})
target_include_directories(pong_ai_tests
                           PRIVATE ${PROJECT_SOURCE_DIR}/src/include)
target_link_libraries(pong_ai_tests PRIVATE Catch2::Catch2WithMain)

add_test(NAME all_tests COMMAND pong_ai_tests)
