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
#ifndef MIGRAPHX_GUARD_OPERATORS_RESHAPE_HPP
#define MIGRAPHX_GUARD_OPERATORS_RESHAPE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/dyn_output.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct reshape
{
    std::vector<int64_t> dims;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dims, "dims"));
    }

    value attributes() const { return {{"require_std_shape", true}}; }

    std::string name() const { return "reshape"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        auto n_neg_dims = std::count(dims.begin(), dims.end(), -1);
        if(n_neg_dims > 1)
            MIGRAPHX_THROW("Reshape: Dimensions for reshape can only have one -1 dim");
        auto s0 = inputs[0];
        if(s0.dynamic())
        {
            auto dyn_dims       = s0.dyn_dims();
            int not_fixed_index = -1;
            // track number of fixed elements in input and output
            std::size_t num_dims_ele = 1;
            std::size_t num_dd_ele   = 1;
            for(std::size_t i = 0; i < dyn_dims.size(); ++i)
            {
                if(dyn_dims[i].is_fixed())
                {
                    num_dims_ele *= dims[i];
                    num_dd_ele *= dyn_dims[i].min;
                }
                else
                {
                    if(not_fixed_index == -1)
                    {
                        not_fixed_index = i;
                    }
                    else
                    {
                        MIGRAPHX_THROW("Reshape: Only support one non-fixed dynamic_dimension");
                    }
                }
            }
            if(num_dims_ele != num_dd_ele)
            {
                MIGRAPHX_THROW("Reshape: Number of fixed elements must match. Input: " +
                               std::to_string(num_dd_ele) +
                               " Output: " + std::to_string(num_dims_ele));
            }
            if(dims[not_fixed_index] != 0 and dims[not_fixed_index] != -1)
            {
                MIGRAPHX_THROW("Reshape: Non-fixed dynamic_dimension doesn't match with 0 or -1 "
                               "output dimension");
            }
            // construct output dynamic shape from dims attribute
            std::vector<shape::dynamic_dimension> output_dyn_dims = {};
            for(std::size_t i = 0; i < dims.size(); ++i)
            {
                if(i == not_fixed_index)
                {
                    output_dyn_dims.push_back(dyn_dims[not_fixed_index]);
                }
                else
                {
                    auto d = static_cast<std::size_t>(dims[i]);
                    output_dyn_dims.push_back({d, d, 0});
                }
            }
            return {s0.type(), output_dyn_dims};
        }
        else
        {
            check_shapes{inputs, *this}.standard();
            auto&& idims = inputs.front().lens();
            std::vector<std::size_t> rdims(dims.begin(), dims.end());

            for(std::size_t i = 0; i < dims.size(); i++)
            {
                if(dims[i] == 0)
                    rdims[i] = idims[i];

                // since rdims using size_t type, -1 is the max value
                // is size_t that cause later compuation incorrect
                if(dims[i] == -1)
                    rdims[i] = 1;
            }

            if(n_neg_dims > 0)
            {
                size_t missing_dim =
                    inputs.front().elements() /
                    std::accumulate(rdims.begin(), rdims.end(), 1, std::multiplies<int64_t>());
                for(std::size_t i = 0; i < rdims.size(); i++)
                {
                    if(dims[i] == -1)
                        rdims[i] = missing_dim;
                }
            }

            shape s{inputs.front().type(), rdims};
            if(s.elements() != inputs.front().elements())
                MIGRAPHX_THROW("Reshape: Wrong number of elements for reshape: reshape has " +
                               std::to_string(s.elements()) + " elements whereas the input has " +
                               std::to_string(inputs.front().elements()));
            return s;
        }
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        return args[0].reshape(dyn_out.computed_shape);
    }

    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
