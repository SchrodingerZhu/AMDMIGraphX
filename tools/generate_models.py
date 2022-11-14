import os, sys, subprocess, string
__source_dir__ = os.path.normpath(
    os.path.join(os.path.realpath(__file__), '..', '..'))
driver_exe = os.path.abspath(sys.argv[1])
cpp_function = string.Template('''
/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/json.hpp>
#include "models.hpp"
namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {
migraphx::program ${name}(unsigned batch) // NOLINT(readability-function-size)
{
    ${body}
    return p;
}
} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
''')


def sed(cmd, extended=False):
    flags = ''
    if extended:
        flags = '-E'
    return f"| sed {flags} '{cmd}'"


def get_generated_body(model):
    run_driver = f"{driver_exe} read --cpp {model} | sed '/Reading/d'"
    cmd = run_driver + sed('/add_parameter/s/{1,/{batch,/') + sed(
        's/\\\\\\"(\\w+)\\\\\\":/\\1:/g', extended=True)
    # cmd = run_driver + " | sed '/add_parameter/s/{1,/{batch,/'"
    cp = subprocess.run(cmd,
                        shell=True,
                        capture_output=True,
                        check=True,
                        text=True)
    return cp.stdout


def generate_cpp(name, body):
    return cpp_function.substitute(name=name, body=body)


models = {
    'resnet50': '/codes/onnx_models/resnet50_v1.onnx',
    'alexnet': '/codes/onnx_models/bvlcalexnet-12.onnx',
    'inceptionv3': '/codes/onnx_models/inceptionv3_fp32.onnx'
}

for name, model in models.items():
    src = generate_cpp(name, get_generated_body(model))
    cpp_path = os.path.join(__source_dir__, "src", "driver", name + '.cpp')
    # print(src)
    with open(cpp_path, 'w') as f:
        f.write(src)
