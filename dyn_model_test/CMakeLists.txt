#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
cmake_minimum_required(VERSION 3.5)
project (dyn_model_test)

set (CMAKE_CXX_STANDARD 14)
set (PROGRAM dyn_model_test)

find_library(MIGX_LIB
    NAMES libmigraphx
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../docker_build/lib/"
)
if(NOT MIGX_LIB)
    message(FATAL_ERROR "migraphx library not found")
endif()
message("migraphx library: " ${MIGX_LIB})

find_library(MIGX_REF_LIB
    NAMES libmigraphx_ref
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../docker_build/lib/"
)
if(NOT MIGX_REF_LIB)
    message(FATAL_ERROR "migraphx_ref library not found")
endif()
message("migraphx_ref library: " ${MIGX_REF_LIB})

find_library(MIGX_ONNX_LIB
    NAMES libmigraphx_onnx
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../docker_build/lib/"
)
if(NOT MIGX_ONNX_LIB)
    message(FATAL_ERROR "migraphx_onnx library not found")
endif()
message("migraphx_onnx library: " ${MIGX_ONNX_LIB})

message("source file: " ${PROGRAM}.cpp " ---> bin: " ${PROGRAM})
add_executable(${PROGRAM} ${PROGRAM}.cpp)

target_link_libraries(${PROGRAM} ${MIGX_LIB} ${MIGX_REF_LIB} ${MIGX_ONNX_LIB})
target_include_directories(${PROGRAM} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../src/include/")
target_compile_features(${PROGRAM} PRIVATE cxx_std_17)
