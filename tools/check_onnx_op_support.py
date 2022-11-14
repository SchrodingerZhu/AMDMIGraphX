#!/usr/bin/env python3

import subprocess

onnx_dir = "../src/onnx"
ops_supported = []
with open("all_onnx_ops.txt", "r") as f:
    for op in f.readlines():
        op = op.rstrip("\n")
        result = subprocess.run(
            ["grep", "-q", "-n", "--directories=recurse", op, onnx_dir])
        if result.returncode == 0:
            ops_supported.append(op)

for op in ops_supported:
    print(op)
