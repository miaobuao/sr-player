cmake_minimum_required(VERSION 3.20)
project(sr-native CXX)

# The one AI runtime. Static library plus three gates; see README.md.
#
# ncnn is consumed through its own CMake package rather than by guessing at its
# transitive dependencies. `third_party/ncnn` is populated by
# `setup-third-party.ps1` from a release pinned in `pins.json`, and is not
# committed: 45 MB of prebuilt libraries do not belong in a source tree, and the
# pin is what makes the download reproducible.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ncnn's headers and MSVC both need these; without NOMINMAX, windows.h's min/max
# macros collide with std::min/std::max throughout the ncnn headers.
add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)

# The sources are UTF-8 and contain punctuation outside ASCII. MSVC otherwise
# reads them in the system code page and emits C4819 on every file.
if(MSVC)
    add_compile_options(/utf-8)
endif()

set(SR_NCNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn")
set(SR_VULKAN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vulkan")

if(NOT EXISTS "${SR_NCNN_ROOT}/include/ncnn/net.h")
    message(FATAL_ERROR
        "third_party/ncnn is missing. Run:\n"
        "  pwsh -File ${CMAKE_CURRENT_SOURCE_DIR}/setup-third-party.ps1")
endif()

find_package(ncnn REQUIRED PATHS "${SR_NCNN_ROOT}/lib/cmake/ncnn" NO_DEFAULT_PATH)

# The Vulkan loader import library. Absent when the Vulkan SDK is not installed,
# which is the normal case on a machine that can *run* Vulkan but not *build*
# against it; setup-third-party.ps1 regenerates it from the system DLL.
find_library(SR_VULKAN_LIBRARY
    NAMES vulkan-1
    PATHS "${SR_VULKAN_ROOT}"
    NO_DEFAULT_PATH)
if(NOT SR_VULKAN_LIBRARY)
    message(FATAL_ERROR
        "third_party/vulkan/vulkan-1.lib is missing. Run:\n"
        "  pwsh -File ${CMAKE_CURRENT_SOURCE_DIR}/setup-third-party.ps1")
endif()

add_library(sr-native STATIC
    src/runtime.cpp
    src/realesrgan.cpp
    src/rife.cpp
    src/rife/warp.cpp)

target_include_directories(sr-native
    PUBLIC include
    PRIVATE
        src
        # The vendored RIFE sources live in their own directory and include each
        # other by bare name.
        src/rife
        # The vendored RIFE sources include ncnn headers the way ncnn's own
        # sources do (`#include "layer.h"`), so the inner include directory has
        # to be on the path for them.
        "${SR_NCNN_ROOT}/include/ncnn")

target_link_libraries(sr-native PUBLIC ncnn PRIVATE "${SR_VULKAN_LIBRARY}")

# The warp unit test exercises the vendored layer directly, so it needs the same
# private include paths the library builds with.
enable_testing()

# The three Phase 1 gates. Each takes the model directory it needs from an
# environment variable, so the same binary runs against any pinned checkpoint.
foreach(gate device_test restore_image_test rife_pair_test warp_unit_test)
    add_executable(${gate} tests/${gate}.cpp)
    target_link_libraries(${gate} PRIVATE sr-native)
    add_test(NAME ${gate} COMMAND ${gate})
endforeach()

# warp_unit_test includes the vendored layer header directly.
target_include_directories(warp_unit_test PRIVATE
    src/rife
    "${SR_NCNN_ROOT}/include/ncnn")