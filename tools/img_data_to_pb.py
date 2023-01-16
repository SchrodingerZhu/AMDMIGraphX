import os
import argparse
import numpy as np
from onnx import numpy_helper
from PIL import Image


def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def process_img(filename, dim0, dim1):
    # output shape will be [3, 244, 244]
    test_img = Image.open(filename)
    test_img = np.array(test_img.resize([dim0, dim1])).T
    test_img = normalize(test_img)
    test_img = test_img.astype(np.float32)
    return test_img


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Process and batch jpg images from a dir to [num_images, 3, dim0, dim1]"
    )
    parser.add_argument("test_dir",
                        type=str,
                        default=".",
                        help="folder where the test images are stored")
    parser.add_argument("--out_name",
                        type=str,
                        default="tensor",
                        help="output filename")
    parser.add_argument("--dim0",
                        type=int,
                        default=224,
                        help="resize image dim 0")
    parser.add_argument("--dim1",
                        type=int,
                        default=224,
                        help="resize image dim 1")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    img_dir = args.test_dir
    images = []
    for x in os.listdir(img_dir):
        if x.endswith(".jpg") or x.endswith(".jpeg"):
            images.append(
                process_img(os.path.join(img_dir, x), args.dim0, args.dim1))
    batch_images = np.array(images)
    print("Output tensor shape:")
    print(batch_images.shape)
    tensor = numpy_helper.from_array(batch_images)
    with open(args.out_name + ".pb", "wb") as f:
        f.write(tensor.SerializeToString())


if __name__ == "__main__":
    main()
