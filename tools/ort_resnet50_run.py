import argparse
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort


def read_pb_file(filename):
    with open(filename, 'rb') as pfile:
        data_str = pfile.read()
        tensor = onnx.TensorProto()
        tensor.ParseFromString(data_str)
        np_array = numpy_helper.to_array(tensor)

    return np_array


def write_pb_file(data, filename):
    tensor = numpy_helper.from_array(data)
    with open(filename + ".pb", "wb") as f:
        f.write(tensor.SerializeToString())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run resnet50 model with input protobuff and write output")
    parser.add_argument("model", type=str, help="resnet50 onnx file")
    parser.add_argument("input_pb", type=str, help="input data protobuff")
    parser.add_argument("--out_name",
                        type=str,
                        default="output",
                        help="output filename")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ort_sess = ort.InferenceSession(args.model,
                                    providers=['CPUExecutionProvider'])
    input_tensor = read_pb_file(args.input_pb)
    output_tensor = ort_sess.run(None, {"data": input_tensor})
    output = output_tensor[0]
    print("Output tensor shape")
    print(output.shape)
    write_pb_file(output, args.out_name)


if __name__ == "__main__":
    main()
