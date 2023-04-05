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

#include <iostream>
#include <vector>
#include <migraphx/gpu/gemm.hpp>
#include <hip/hip_runtime_api.h>
#include <migraphx/gpu/target.hpp>
#include <migraphx/verify.hpp>
#include <test.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>

// includes needed for run_lowering
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>

// Abbreviated lowering; we don't need the usual cleanup passes for this test
void run_lowering(migraphx::program& p, bool offload_copy = false)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(
        *p.get_main_module(),
        {migraphx::auto_contiguous{}, migraphx::gpu::lowering{&ctx, offload_copy}});
}

/**
 * Tests the automatic GEMM tuning feature.  In the finalize() method of the gemm op,
 * rocBLAS API functions are called to quickly benchmark all the GEMM solutions
 * available in the currently installed rocBLAS library and choose the index of the fastest.
 */
TEST_CASE(gemm_tune_with_rocblas)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape sa{migraphx::shape::float_type, {4, 2}};
    migraphx::shape sb{migraphx::shape::float_type, {2, 3}};
    auto a = mm->add_parameter("a", sa);
    auto b = mm->add_parameter("b", sb);

    migraphx::operation dot_op = migraphx::make_op("dot");
    mm->add_instruction(dot_op, a, b);

    // lowering adds gemm implementation for dot operator
    run_lowering(p);

    migraphx::target gpu_t = migraphx::gpu::target{};
    migraphx::compile_options options;
    options.exhaustive_tune = true;
    p.compile(gpu_t, options);

    migraphx::value solution_idx(0);
    for(auto ins : iterator_for(*p.get_main_module()))
    {
        if(ins->name() == "gpu::gemm")
        {
            auto gemm_op = migraphx::get_operation(ins);

            // tuned solution index is not deterministic, but anything other than 0
            // (default, invalid, or not available) is good.
            // gemm_op.to_value().debug_print();
            solution_idx = gemm_op.to_value()["solution_idx"];
            break;
        }
    }
#ifdef ROCBLAS_BETA_FEATURES_API
    EXPECT(0 != solution_idx.to<std::size_t>());
#else
    EXPECT(0 == solution_idx.to<std::size_t>());
#endif
}

// TODO:  make tests that run both strided and not-strided GEMMs.  Also, try tuning with an invalid
// start value.

int main(int argc, const char* argv[]) { test::run(argc, argv); }
