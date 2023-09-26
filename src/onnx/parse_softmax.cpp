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
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_softmax : op_parser<parse_softmax>
{
    std::vector<op_desc> operators() const
    {
        return {{"Softmax", "softmax"}, {"LogSoftmax", "logsoftmax"}};
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        // default axis value is -1 for opset 13
        int64_t axis = -1;

        // axis value is 1 for previous opset versions
        if(parser.opset_version < 13)
        {
            axis = 1;
        }

        if(contains(info.attributes, "axis"))
        {
            axis = parser.parse_value(info.attributes.at("axis")).at<int>();
        }

        // previous version
        return info.add_instruction(make_op(opd.op_name, {{"axis", axis}}), args);

        // converted version
        // auto convert0 = info.add_instruction(make_op("convert", {{"target_type",
        // shape::float_type}}), args); auto softmax_ins = info.add_instruction(make_op(opd.op_name,
        // {{"axis", axis}}), convert0); return info.add_instruction(make_op("convert",
        // {{"target_type", shape::half_type}}), softmax_ins);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
