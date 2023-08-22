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

/**
 * Random Uniform distribution operator.  Given a shape, populate it with random
 * values.  Calls to random_uniform using the same randomization seed will
 * always generate the same pseudo-random sequence.  Seed can
 * be given as a runtime argument containing a single value, or a compile-time
 * attribute.
 *
 *      Inputs:   (1) randomization seed (uint64)
 *                (2) the shape of the set to be populated.
 *
 *
 *      Attributes:  none
 *
 *      Output:   Same shape.
 *
 */
#ifndef MIGRAPHX_GUARD_OPERATORS_RANDOM_UNIFORM_HPP
#define MIGRAPHX_GUARD_OPERATORS_RANDOM_UNIFORM_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/reflect.hpp>
#include <random>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct random_uniform
{
    // The random_uniform operation does not contain a random number generator seed
    // as a member, and expects it to be passed as a runtime input.

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack();
    }

    /**
     *   Input 1:  seed
     *   Input 2:  output shape
     */
    std::string name() const { return "random_uniform"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(2);

        if(inputs.front().type() != shape::type_t::uint64_type)
            MIGRAPHX_THROW("RANDOM_UNIFORM:  Input 1 (seed) must have type long unsigned int");
        auto s = inputs.at(1);
        if(s.dynamic())
        {
            return s;
        }
        else
        {
            return s.with_lens(s.lens());
        }
    }

    argument compute(const shape&, std::vector<argument> args) const
    {
        // Output goes into the passed buffer, not the shape output.
        auto result = args[1];

        uint64_t local_seed = args[0].at<uint64_t>(0);
        std::mt19937 gen(local_seed);

        result.visit([&](auto output) {
            using type = typename decltype(output)::value_type;
            if constexpr(std::is_integral<type>{})
            {
                // default range for all integer types is (0,
                // std::uniform_int_distribution<type>::max()).
                // To clamp to a different range, apply min or max ops. to the output of this.
                std::uniform_int_distribution<type> dis;
                std::generate(output.begin(), output.end(), [&] { return dis(gen); });
            }
            else
            {
                // default real distribution type is double with range (0, 1);
                std::uniform_real_distribution<> dis;
                std::generate(output.begin(), output.end(), [&] { return dis(gen); });
            }
        });
        return result;
    }

    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 1; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
