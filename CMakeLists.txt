cmake_minimum_required(VERSION 3.29)
project(pong-ai)

set(font_ttf ${PROJECT_SOURCE_DIR}/res/ComicShannsMono-Regular.ttf)
set(font_source ${CMAKE_CURRENT_BINARY_DIR}/font_data.c)

add_custom_command(
  OUTPUT ${font_source}
  COMMAND ${CMAKE_COMMAND} -E env xxd -i -C -n DISPLAY_FONT_DATA ${font_ttf} >
          ${font_source}
  DEPENDS ${font_ttf}
  COMMENT "Generating font source")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type to Debug
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "Build type" FORCE)
endif()

# Compiler warnings
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(SDL3 REQUIRED CONFIG REQUIRED COMPONENTS SDL3-shared)
find_package(SDL3_ttf REQUIRED CONFIG REQUIRED)

file(GLOB_RECURSE source_files CONFIGURE_DEPENDS src/*.cpp)

add_executable(pong_ai ${source_files} ${font_source})
target_include_directories(pong_ai PUBLIC src/include)
target_link_libraries(pong_ai PRIVATE SDL3::SDL3)
target_link_libraries(pong_ai PRIVATE SDL3_ttf::SDL3_ttf)

include(CTest)
add_subdirectory(tests)
