cmake_minimum_required(VERSION 3.29)
project(pong-ai)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type to Debug
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")

enable_testing()

find_package(Catch2 3 REQUIRED)

file(GLOB_RECURSE TEST_FILES CONFIGURE_DEPENDS "tests/*.cpp")

add_executable(pong_ai_tests ${TEST_FILES})
target_include_directories(pong_ai_tests PRIVATE "src/include")
target_link_libraries(pong_ai_tests PRIVATE Catch2::Catch2WithMain)

add_test(NAME all_tests COMMAND pong_ai_tests)

# Compiler warnings
if(MSVC)
  target_compile_options(pong_ai_tests PRIVATE /W4)
else()
  target_compile_options(pong_ai_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
