
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/serialize.hpp>

#include <migraphx/op/common.hpp>

struct test_gru_forward_3args_und : verify_program<test_gru_forward_3args_und>
{
    migraphx::program create_program() const
    {
        int batch_size  = 2;
        int seq_len     = 3;
        int hidden_size = 5;
        int input_size  = 8;
        int num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        auto seq = mm->add_parameter("seq", in_shape);
        auto w   = mm->add_parameter("w", w_shape);
        auto r   = mm->add_parameter("r", r_shape);
        auto und = mm->add_instruction(migraphx::make_op("undefined"));
        mm->add_instruction(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                                      migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            seq,
            w,
            r,
            und,
            und,
            und);

        return p;
    }
    std::string section() const { return "rnn"; }
};
