/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(isinf_double_pos_test)
{
    migraphx::program p = migraphx::parse_onnx("isinf_double_pos_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::double_type, {2, 3}};
    migraphx::parameter_map pp;
    double nan               = std::numeric_limits<double>::quiet_NaN();
    double infinity          = std::numeric_limits<double>::infinity();
    double max               = std::numeric_limits<double>::max();
    double min               = std::numeric_limits<double>::min();
    std::vector<double> data = {-infinity, nan, min, 3.6, max, infinity};
    pp["t1"]                 = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0, 0, 0, 0, 0, 1};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
