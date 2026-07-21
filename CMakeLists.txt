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
# Uses the installed package by default (find_package); reinstall rocket-userspace after
# changing it, or point rocketnpu_DIR at a build tree.
find_package(rocketnpu REQUIRED)

# The EP is a plugin: ONNX Runtime dlopen()s it and calls in through the OrtApi
# table passed to CreateEpFactories. It therefore does NOT link libonnxruntime.
add_library(ort_rocket SHARED src/rocket_ep.cc src/rocket_match.cc src/rocket_backbone.c
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
