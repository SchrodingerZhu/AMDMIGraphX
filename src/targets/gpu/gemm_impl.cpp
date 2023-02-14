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
#define ROCBLAS_BETA_FEATURES_API 1
#include <rocblas/rocblas.h>
#include <migraphx/gpu/gemm_impl.hpp>
// #include "rocblas_gemm_ex_get_solutions.hpp"
#include <migraphx/generate.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/time.hpp>


namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

rocblas_datatype get_type(shape::type_t type)
{
    switch(type)
    {
    case shape::double_type: return rocblas_datatype_f64_r;
    case shape::float_type: return rocblas_datatype_f32_r;
    case shape::half_type: return rocblas_datatype_f16_r;
    case shape::int8_type: return rocblas_datatype_i8_r;
    case shape::uint8_type: return rocblas_datatype_u8_r;
    case shape::int32_type: return rocblas_datatype_i32_r;
    case shape::uint32_type: return rocblas_datatype_u32_r;
    case shape::tuple_type:
    case shape::bool_type:
    case shape::uint16_type:
    case shape::int16_type:
    case shape::int64_type:
    case shape::uint64_type: MIGRAPHX_THROW("ROCBLAS_GEMM: data type not supported!");
    }

    MIGRAPHX_THROW("ROCBLAS_GEMM: data type not supported!");
}

void blas_shape(const shape& s)
{
    if(s.lens().size() < 2)
        return;
    if(std::none_of(s.strides().end() - 2, s.strides().end(), [&](auto i) { return i == 1; }))
        MIGRAPHX_THROW("GPU_GEMM: needs to have one matrix stride as 1");
    if(s.lens().size() < 3)
        return;
    shape batch_shape{s.type(),
                      {s.lens().begin(), s.lens().end() - 2},
                      {s.strides().begin(), s.strides().end() - 2}};
    auto batch_shapes = reduce_dims({batch_shape});
    if(batch_shapes.front().lens().size() != 1)
        MIGRAPHX_THROW("GPU_GEMM: Batch dimension is not collapsible");
}

shape transpose_batch(const shape& s, unsigned trans_batch)
{
    if(trans_batch == 0)
        return s;
    if(s.lens().size() < 3)
        return s;
    auto batch = s.lens().size() - 3;
    std::vector<int64_t> perm(s.lens().size());
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[batch], perm[batch + trans_batch]);
    return shape::from_permutation(s.type(), s.lens(), perm);
}

template <class R, class... Ts, class... Us>
R rocblas_invoke(R (*f)(Ts...), Us... xs)
{
    if constexpr(sizeof...(Ts) == sizeof...(Us))
        return f(xs...);
    else
        return f(xs..., nullptr, nullptr);
}

static bool is_transposed(const shape& s)
{
    if(not s.transposed())
        return false;
    return s.strides().back() != 1;
}

static rocblas_int get_batch_stride(const argument& a)
{
    return a.get_shape().strides()[a.get_shape().strides().size() - 3];
}


std::vector<argument> generate_arguments(const std::vector<shape>& shapes, unsigned long seed = 0)
{
    std::vector<argument> args;
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(args), [&](auto& s) {
        return to_gpu(generate_argument(s, seed++));
    });
    return args;
}

// from perf.cpp
using milliseconds = std::chrono::duration<double, std::milli>;
std::pair<double, double>
time_op(context& ictx, operation op, const std::vector<shape>& inputs, int n)
{

    // TODO: Use std::ref
    migraphx::context ctx = ictx;
    auto& gctx            = any_cast<migraphx::gpu::context>(ctx);
    auto output           = op.compute_shape(inputs);
    // op.finalize(ctx, output, inputs);
    auto args = generate_arguments(inputs);
    auto run  = [&] {
        op.compute(ctx, output, args);
        ctx.finish();
    };
    gctx.enable_perf_measurement();
    run();
    double host_time   = 0.0;
    double device_time = 0.0;
    for(auto i : range(n))
    {
        (void)i;
        host_time += time<milliseconds>(run);
        device_time += gctx.get_elapsed_ms();
    }
    return std::make_pair(host_time / n, device_time / n);
}

template <class T>
void gemm_impl(context& ctx,
               const shape& output_shape,
               const std::vector<argument>& args,
               T alpha,
               T beta,
               bool int8_x4_format,
               bool compute_fp32)
{
    const bool is_3inputs = (args.size() == 4);
    if(not is_3inputs)
    {
        beta = 0;
    }

    bool transa     = is_transposed(args[0].get_shape());
    bool transb     = is_transposed(args[1].get_shape());
    auto n_dim      = output_shape.lens().size();
    auto dim_1      = n_dim - 1;
    auto dim_0      = n_dim - 2;
    rocblas_int lda = args[0].get_shape().strides()[transa ? dim_1 : dim_0];
    rocblas_int ldb = args[1].get_shape().strides()[transb ? dim_1 : dim_0];
    rocblas_int ldc = args[2].get_shape().strides()[dim_0];
    rocblas_int ldd = is_3inputs ? args[3].get_shape().strides()[dim_0] : ldc;

    rocblas_datatype arg_type = get_type(args[0].get_shape().type());
    auto output_type          = arg_type;
    if(output_type == rocblas_datatype_i8_r)
    {
        output_type = rocblas_datatype_i32_r;
    }
    auto compute_type = output_type;
    if(compute_fp32)
    {
        if(arg_type == rocblas_datatype_f16_r)
            compute_type = rocblas_datatype_f32_r;
    }

#if ROCBLAS_VERSION_MAJOR >= 2 && ROCBLAS_VERSION_MINOR >= 38
    rocblas_gemm_flags flag =
        int8_x4_format ? rocblas_gemm_flags_pack_int8x4 : rocblas_gemm_flags_none;
#else
    (void)int8_x4_format;
    int flag = 0;
#endif

    auto a_lens = args[0].get_shape().lens();
    auto b_lens = args[1].get_shape().lens();
    output_shape.visit_type([&](auto as) {
        auto alpha_r = as(alpha);
        auto beta_r  = as(beta);

        // use void pointer to select different data type if using fp32 mode
        void* alpha_v = &alpha_r;
        void* beta_v  = &beta_r;

        if(compute_fp32)
        {
            alpha_v = &alpha;
            beta_v  = &beta;
        }

        auto out_lens   = output_shape.lens();
        rocblas_int m   = out_lens[dim_0];
        rocblas_int n   = out_lens[dim_1];
        rocblas_int k   = args[0].get_shape().lens()[dim_1];
        auto to_pointer = [&](auto&& arg) { return as.from(arg.data()); };
        if(args[0].get_shape().type() == shape::int8_type and (k % 4) != 0 and int8_x4_format)
        {
            MIGRAPHX_THROW("ROCBLAS_GEMM: k size of int8 type input must be mutlple of 4!");
        }

        auto num_matrices = std::accumulate(
            out_lens.rbegin() + 2, out_lens.rend(), std::size_t{1}, std::multiplies<std::size_t>());
        if(num_matrices == 1 or (num_matrices > 1 and get_batch_stride(args[1]) == 0))
        {
            // If the batch dimension of B is broadcasted, then we can
            // multiply m by the batch_size and use rocblas_gemm_ex
            // instead of rocblas_gemm_strided_batched_ex.
            m *= num_matrices;

        }
        
        auto da = to_pointer(args.at(0));
        auto db = to_pointer(args.at(1));
        auto dc = to_pointer(args.at(2));

        auto type = arg_type;

            #define GEMM_EX_ARGS                                                                               \
                handle, transa, transb, m, n, k, alpha_v, da, type, lda, db, type, ldb, beta_v, dc, type, ldc, \
                    dc, type, ldc, type, rocblas_gemm_algo_solution_index

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));
    // Get number of solutions
    rocblas_int size;
    CHECK_ROCBLAS_ERROR(
        rocblas_gemm_ex_get_solutions(GEMM_EX_ARGS, rocblas_gemm_flags_none, NULL, &size));
    rocblas_cout << size << " solution(s) found" << std::endl;

    // Fill array with list of solutions
    std::vector<rocblas_int> ary(size);
    CHECK_ROCBLAS_ERROR(
        rocblas_gemm_ex_get_solutions(GEMM_EX_ARGS, rocblas_gemm_flags_none, ary.data(), &size));

    // GEMM_EX_ARGS;        
// rocblas_gemm_ex_get_solutions_template(rocblas_handle    handle,
//                                                       rocblas_operation trans_a,
//                                                       rocblas_operation trans_b,
//                                                       rocblas_int       m,
//                                                       rocblas_int       n,
//                                                       rocblas_int       k,
//                                                       const void*       alpha,
//                                                       const void*       a,
//                                                       rocblas_datatype  a_type,
//                                                       rocblas_stride    offsetAin,
//                                                       rocblas_int       lda,
//                                                       rocblas_stride    stride_a,
//                                                       const void*       b,
//                                                       rocblas_datatype  b_type,
//                                                       rocblas_stride    offsetBin,
//                                                       rocblas_int       ldb,
//                                                       rocblas_stride    stride_b,
//                                                       const void*       beta,
//                                                       const void*       c,
//                                                       rocblas_datatype  c_type,
//                                                       rocblas_stride    offsetCin,
//                                                       rocblas_int       ldc,
//                                                       rocblas_stride    stride_c,
//                                                       void*             d,
//                                                       rocblas_datatype  d_type,
//                                                       rocblas_stride    offsetDin,
//                                                       rocblas_int       ldd,
//                                                       rocblas_stride    stride_d,
//                                                       rocblas_int       batch_count,
//                                                       rocblas_datatype  compute_type,
//                                                       uint32_t          flags,
//                                                       rocblas_int*      list_array,
//                                                       rocblas_int*      list_size)


            //         return pack(ctx.get_stream().get_rocblas());
    // Get number of solutions
    // rocblas_int size;
    // CHECK_ROCBLAS_ERROR(
    //     rocblas_gemm_ex_get_solutions(GEMM_EX_ARGS, rocblas_gemm_flags_none, NULL, &size));

    });
printf("here I am =                          ================================================");
    exit(8);
}

void gemm(context& ctx,
          const shape& output_shape,
          const std::vector<argument>& args,
          float alpha,
          float beta,
          bool int8_x4_format,
          bool compute_fp32)
{
    gemm_impl(ctx, output_shape, args, alpha, beta, int8_x4_format, compute_fp32);
}

void gemm(context& ctx,
          const shape& output_shape,
          const std::vector<argument>& args,
          int32_t alpha,
          int32_t beta,
          bool int8_x4_format,
          bool compute_fp32)
{
    gemm_impl(ctx, output_shape, args, alpha, beta, int8_x4_format, compute_fp32);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
