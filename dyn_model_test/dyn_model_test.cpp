#include <migraphx/onnx.hpp>

int main()
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 10, 0};

    migraphx::parse_onnx("resnet50_v1.onnx", options);
}
