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
#include <rocblas/rocblas.h>
#include <migraphx/gpu/gemm_impl.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/permutation.hpp>

// #include <migraphx/config.hpp>
// #include <migraphx/gpu/context.hpp>
#include <migraphx/generate.hpp>
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


template <class T>
void gemm_impl(context& ctx,
               const shape& output_shape,
               const std::vector<argument>& args,
               T alpha,
               T beta,
               bool int8_x4_format,
               bool compute_fp32)
{
    output_shape.visit_type([&](auto as) {   // TODO:  not needed?
    (void)as;
        auto out_lens   = output_shape.lens();
        auto num_matrices = std::accumulate(
            out_lens.rbegin() + 2, out_lens.rend(), std::size_t{1}, std::multiplies<std::size_t>());
        if(num_matrices == 1 or (num_matrices > 1 and get_batch_stride(args[1]) == 0))
        {
            // If the batch dimension of B is broadcasted, then we can
            // multiply m by the batch_size and use rocblas_gemm_ex
            // instead of rocblas_gemm_strided_batched_ex.

            // the rocblas_gemm API handles inputs and output matrices as
            // column-major format. When doing a C = A * B, we actually do
            // C^T = (B^T) * (A^T). That is the reason we input args[1] as
            // A and args[0] as B in calling the rocblas_gemm.
            auto to_invoke = 
            create_gemm_args(ctx, ROCBLAS_CALL::ROCBLAS_GEMM_EX, output_shape, args, 
                                              alpha, beta, int8_x4_format, compute_fp32);
            // rocblas_invoke(&rocblas_gemm_ex,
            //                to_invoke);
        }
        else
        {
            auto to_invoke = 
            create_gemm_args(ctx, ROCBLAS_CALL::ROCBLAS_GEMM_STRIDED_BATCHED_EX, 
                                              output_shape, args, alpha, beta, int8_x4_format, compute_fp32);
            // rocblas_invoke(&rocblas_gemm_strided_batched_ex,
            //               to_invoke);
        }
    });
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


/**
 * Create a list of the arguments needed for rocBLAS GEMM calls, from
 * a set of MigraphX arguments.
 */
template <class T>
static auto create_gemm_args(context& ctx,
                      ROCBLAS_CALL rocblas_call,
                      const shape& output_shape,
                      const std::vector<argument>& inputs,
                      T alpha,
                      T beta,
                      bool int8_x4_format,
                      bool compute_fp32)
{
    const bool is_3inputs = (inputs.size() == 4);
    if(not is_3inputs)
    {
        beta = 0;
    }

    bool transa     = is_transposed(inputs[0].get_shape());
    bool transb     = is_transposed(inputs[1].get_shape());
    auto n_dim      = output_shape.lens().size();
    auto dim_1      = n_dim - 1;
    auto dim_0      = n_dim - 2;
    rocblas_int lda = inputs[0].get_shape().strides()[transa ? dim_1 : dim_0];
    rocblas_int ldb = inputs[1].get_shape().strides()[transb ? dim_1 : dim_0];
    rocblas_int ldc = inputs[2].get_shape().strides()[dim_0];
    rocblas_int ldd = is_3inputs ? inputs[3].get_shape().strides()[dim_0] : ldc;

    rocblas_datatype arg_type = get_type(inputs[0].get_shape().type());
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

    auto a_lens = inputs[0].get_shape().lens();
    auto b_lens = inputs[1].get_shape().lens();
    void * alpha_v = nullptr;
    void* beta_v = nullptr;
     output_shape.visit_type([&](auto as) {
        auto alpha_r = as(alpha);
        auto beta_r  = as(beta);

        // use void pointer to select different data type if using fp32 mode
        alpha_v = &alpha_r;
        beta_v  = &beta_r;
        if(compute_fp32)
        {
            alpha_v = &alpha;
            beta_v  = &beta;
        }
     });

        auto out_lens   = output_shape.lens();
        rocblas_int m   = out_lens[dim_0];
        rocblas_int n   = out_lens[dim_1];
        rocblas_int k   = inputs[0].get_shape().lens()[dim_1];
        auto to_pointer = [&](auto&& arg) { return reinterpret_cast<T*>(arg.data()); };
        if(inputs[0].get_shape().type() == shape::int8_type and (k % 4) != 0 and int8_x4_format)
        {
            MIGRAPHX_THROW("create_gemm_args: k size of int8 type input must be multiple of 4!");
        }

        auto num_matrices = std::accumulate(
            out_lens.rbegin() + 2, out_lens.rend(), std::size_t{1}, std::multiplies<std::size_t>());

        switch(rocblas_call){
            case     ROCBLAS_GEMM_EX:
            {
                m *= num_matrices;

                return pack(

                    // the rocblas_gemm API handles inputs and output matrices as
                    // column-major format. When doing a C = A * B, we actually do
                    // C^T = (B^T) * (A^T). That is the reason we input inputs[1] as
                    // A and inputs[0] as B in calling the rocblas_gemm.
                    // rocblas_invoke(&rocblas_gemm_ex,
                    ctx.get_stream().get_rocblas(),
                    transb ? rocblas_operation_transpose : rocblas_operation_none,
                    transa ? rocblas_operation_transpose : rocblas_operation_none,
                    n,
                    m,
                    k,
                    alpha_v,
                    to_pointer(inputs.at(1)),
                    arg_type,
                    ldb,
                    to_pointer(inputs.at(0)),
                    arg_type,
                    lda,
                    beta_v,
                    to_pointer(inputs[2]),
                    output_type,
                    ldc,
                    is_3inputs ? to_pointer(inputs[3]) : to_pointer(inputs[2]),
                    output_type,
                    ldd,
                    compute_type,
                    rocblas_gemm_algo_standard,
                    0,
                    flag);
            }

            case     ROCBLAS_GEMM_STRIDED_BATCHED_EX:
            default:
            {
                auto a_stride = get_batch_stride(inputs[0]);
                auto b_stride = get_batch_stride(inputs[1]);
                auto c_stride = get_batch_stride(inputs[2]);
                auto d_stride = is_3inputs ? get_batch_stride(inputs[3]) : c_stride;
                return pack(
                    // rocblas_invoke(  &rocblas_gemm_strided_batched_ex,
                    ctx.get_stream().get_rocblas(),
                    transb ? rocblas_operation_transpose : rocblas_operation_none,
                    transa ? rocblas_operation_transpose : rocblas_operation_none,
                    n,
                    m,
                    k,
                    alpha_v,
                    to_pointer(inputs.at(1)),
                    arg_type,
                    ldb,
                    b_stride,
                    to_pointer(inputs.at(0)),
                    arg_type,
                    lda,
                    a_stride,
                    beta_v,
                    to_pointer(inputs[2]),
                    output_type,
                    ldc,
                    c_stride,
                    is_3inputs ? to_pointer(inputs[3]) : to_pointer(inputs[2]),
                    output_type,
                    ldd,
                    d_stride,
                    num_matrices,
                    compute_type,
                    rocblas_gemm_algo_standard,
                    0,
                    flag);
            }

            // case    ROCBLAS_GEMM_EX_GET_SOLUTIONS:
            // default:
            //     // the original macro in rocBLAS-internal/rocBLAS/clients/samples/example_user_driven_tuning.cpp is
            //     //  Note different order of m, n, k
            //     // #define GEMM_EX_ARGS \
            //     //     handle, transa, transb, m, n, k, &alpha, da, type, lda, db, type, ldb, &beta, dc, type, ldc,
            //     //     \
            //     //         dc, type, ldc, type, rocblas_gemm_algo_solution_index
            // #define GEMM_EX_ARGS                                                                               \
            //     handle, transa, transb, m, n, k, alpha_v, da, type, lda, db, type, ldb, beta_v, dc, type, ldc, \
            //         dc, type, ldc, type, rocblas_gemm_algo_solution_index
            //         return pack(ctx.get_stream().get_rocblas());
    // Get number of solutions
    // rocblas_int size;
    // CHECK_ROCBLAS_ERROR(
    //     rocblas_gemm_ex_get_solutions(GEMM_EX_ARGS, rocblas_gemm_flags_none, NULL, &size));
        } // end switch

            // default:
            // MIGRAPHX_THROW ("create_gemm_args(): rocBLAS command not supported");
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
