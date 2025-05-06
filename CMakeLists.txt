cmake_minimum_required(VERSION "3.29")
project("pong-ai")

set(CMAKE_CXX_STANDARD "20")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Og")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

file(GLOB_RECURSE TEST_FILES CONFIGURE_DEPENDS "tests/*.cpp")

find_package(Catch2 3 REQUIRED)
# These tests can use the Catch2-provided main
add_executable(tests ${TEST_FILES})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

target_include_directories(tests PRIVATE "src/include")

if(MSVC)
  target_compile_options(tests PRIVATE /W4)
else()
  target_compile_options(tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
