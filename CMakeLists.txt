# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.20)
project(ort_rocket LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ONNX Runtime session headers. Defaults to the vendored copy (the MIT-licensed
# public C/C++ API headers for the ORT version this EP targets); override with
# -DORT_HEADER_DIR=<path> to build against another ORT release's headers.
set(ORT_HEADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include"
    CACHE PATH "Directory containing onnxruntime_c_api.h / onnxruntime_ep_c_api.h")

# The NPU datapath. Links librocketnpu (the validated encoder/backbone primitives).
#
# An EXPLICIT -DROCKETNPU_DIR wins over an installed package, and names a rocket-userspace
# SOURCE checkout which is built as a subdirectory. This is the same contract ggml-rocket
# and tflite-rocket carry, and it exists for the same reason: a machine with a stale
# /usr/local install otherwise builds this EP against THAT driver while the configure log
# names the tree the user asked for, so the .so reports on code that is not the code under
# test. That silently invalidated one arm of the tflite differential test before it was
# caught, and an EP built against the wrong driver is exactly the failure a two-driver A/B
# cannot see from its own outputs.
#
# cmake's own -Drocketnpu_DIR=<dir holding rocketnpu-config.cmake> still works and still
# means what it means to find_package. Prefer ROCKETNPU_DIR, which names a source tree.
if(DEFINED ROCKETNPU_DIR)
    message(STATUS "ort-rocket: ROCKETNPU_DIR set explicitly; ignoring any installed rocketnpu package")
else()
    find_package(rocketnpu QUIET CONFIG)
endif()
if(rocketnpu_FOUND)
    message(STATUS "ort-rocket: using installed rocketnpu package")
else()
    # The fallback probes SIBLING first, then one level further out. A public clone puts
    # ort-rocket and rocket-userspace side by side, which is the layout the other two
    # frontends default to; this checkout is one level deeper than they are. Probing both
    # is what lets ONE CMakeLists serve both trees -- a default tuned to whichever layout
    # the author happens to sit in is how this file diverged from its published copy and
    # left the published one unable to configure at all.
    if(NOT DEFINED ROCKETNPU_DIR)
        foreach(_cand ../rocket-userspace ../../rocket-userspace)
            if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_cand}/CMakeLists.txt)
                set(ROCKETNPU_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${_cand} CACHE PATH "Path to a rocket-userspace checkout")
                break()
            endif()
        endforeach()
    endif()
    if(NOT DEFINED ROCKETNPU_DIR)
        set(ROCKETNPU_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../rocket-userspace CACHE PATH "Path to a rocket-userspace checkout")
    endif()
    if(NOT EXISTS ${ROCKETNPU_DIR}/CMakeLists.txt)
        message(FATAL_ERROR
            "rocketnpu package not found and no checkout at ROCKETNPU_DIR=${ROCKETNPU_DIR}. "
            "Install rocket-userspace (cmake --install) or set -DROCKETNPU_DIR=/path/to/rocket-userspace.")
    endif()
    message(STATUS "ort-rocket: building rocketnpu from ${ROCKETNPU_DIR}")
    set(ROCKETNPU_BUILD_TESTS OFF CACHE BOOL "" FORCE)   # don't build driver tests here
    add_subdirectory(${ROCKETNPU_DIR} ${CMAKE_BINARY_DIR}/rocketnpu)
endif()

# The EP is a plugin: ONNX Runtime dlopen()s it and calls in through the OrtApi
# table passed to CreateEpFactories. It therefore does NOT link libonnxruntime.
add_library(ort_rocket SHARED src/rocket_ep.cc src/rocket_match.cc src/rocket_convt.cc src/rocket_backbone.c
            src/rocket_projector.c src/rocket_ort_host.c)
target_include_directories(ort_rocket PRIVATE ${ORT_HEADER_DIR})
target_link_libraries(ort_rocket PRIVATE rocketnpu::rocketnpu)
set_target_properties(ort_rocket PROPERTIES OUTPUT_NAME onnxruntime_rocket)

# Export only the two plugin entry points. -fvisibility=hidden hides our own objects'
# symbols; --exclude-libs,ALL keeps the statically-linked librocketnpu symbols out of the
# .so's dynamic export table too (they are already-compiled with default visibility).
target_compile_options(ort_rocket PRIVATE -fvisibility=hidden -Wall -Wextra)
target_link_options(ort_rocket PRIVATE -Wl,--exclude-libs,ALL)

message(STATUS "ort-rocket: building libonnxruntime_rocket.so against ORT headers at ${ORT_HEADER_DIR}")
