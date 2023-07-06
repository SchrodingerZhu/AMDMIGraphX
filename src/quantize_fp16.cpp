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
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/quantize_fp16.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/target.hpp>
#include <migraphx/make_op.hpp>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static void quantize_module(module& m, const std::vector<std::string>& ins_names)
{
    for(auto ins : iterator_for(m))
    {
        // instructions are not in the set to be quantized
        if(not(contains(ins_names, ins->name()) or contains(ins_names, "all")))
            continue;

        // skip return and convert instructions
        if(contains({"@return", "convert"}, ins->name()))
            continue;

        if(ins->inputs().empty())
            continue;

        auto mod_inputs = ins->module_inputs();
        auto s          = ins->get_shape();
        // Convert each of the inputs that are floating point to fp16
        auto inputs = ins->inputs();
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto input) {
            auto input_type = input->get_shape().type();
            if(input_type != shape::float_type and input_type != shape::double_type)
                return input;
            return m.insert_instruction(
                ins, make_op("convert", {{"target_type", shape::half_type}}), input);
        });

        // Insert quantized ins
        auto converted_ins = m.insert_instruction(ins, ins->get_operator(), inputs, mod_inputs);

        // Convert back to original type after quantizing
        if(mod_inputs.empty())
        {
            converted_ins = m.insert_instruction(
                ins, make_op("convert", {{"target_type", s.type()}}), converted_ins);
        }
        // Replace original instruction
        m.replace_instruction(ins, converted_ins);
    }
    // m.debug_print();
    // std::cout << "HERE" << std::endl;
    
}

static void quantize_params(module& m)
{
    std::vector<std::string> param_names = m.get_parameter_names();
    std::unordered_set<std::string> processed_params;
    for( auto param_name : param_names)
    {
        auto param = m.get_parameter(param_name);
        // m.debug_print(param);
        if( not contains(processed_params, param_name) and param->get_shape().type() == shape::float_type)
        {
            auto new_param = m.add_parameter(param_name, migraphx::shape{shape::half_type, param->get_shape().lens()});
            // m.debug_print(new_param);
            // m.debug_print();

            // m.debug_print();
            m.replace_instruction(param, new_param);
            // std::cout << "HERE" << std::endl;

        }
        processed_params.insert(param_name);
    }
}

void quantize_fp16_pass::apply(module_pass_manager& mpm) const { 
    module m = mpm.get_module();
    quantize_module(m, ins_names); 
    // mpm.run_pass(dead_code_elimination{});
    // m.debug_print();
    // quantize_params(m);
    }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
