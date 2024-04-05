#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Optional

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchProgramManager,
    to_edge,
)

from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from model import GPT, GPTConfig
from test_utils import check_executorch_output_consistency, ErrorLimits
from torch import nn
from torch._export import capture_pre_autograd_graph
from torch.export import export

from torch.nn.attention import SDPBackend


def main(args):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Prep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model = GPT.from_pretrained("gpt2")
    example_inputs = (torch.randint(0, 100, (1, 8), dtype=torch.long),)

    #  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Export  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Using a custom SDPA kernel for LLMs
    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        m = capture_pre_autograd_graph(model, example_inputs)

        if args.backend == "XnnPack":
            edge_config = get_xnnpack_edge_compile_config()
        else:
            edge_config = EdgeCompileConfig(_check_ir_validity=False)

        print("Exporting program...")
        core_aten_ep = export(m, example_inputs)
        print("Lowering to edge...")
        edge_manager: EdgeProgramManager = to_edge(
            core_aten_ep,
            compile_config=edge_config,
        )
        if args.backend == "XnnPack":
            print("Lowering to XnnPack...")
            edge_manager = edge_manager.to_backend(XnnpackPartitioner())

        print("Creating ExecuTorch program...")
        et_program: ExecutorchProgramManager = edge_manager.to_executorch()

        if args.verifiy_runtime:
            print("Checking the outputs of the ExecuTorch program...")
            error_limits = ErrorLimits(atol=args.atol, rtol=args.rtol)
            res = check_executorch_output_consistency(
                flatbuffer_buff=et_program.buffer,
                model=model,
                method_name="forward",
                example_inputs=example_inputs,
                load_fn=_load_for_executorch_from_buffer,
                error_limits=error_limits,
            )
            if res.is_same:
                print("Outputs are the same!")
            else:
                print("Outputs are different!")
                print("Reasons: {}".format("\n ".join(res.reasons)))

    # Write the serialized ExecuTorch program to a file.
    with open(args.output_file, "wb") as file:
        file.write(et_program.buffer)
        print(f"ExecuTorch program saved to {args.output_file}.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="NanoGPT example.")
    parser.add_argument("--backend", type=str, choices=["XnnPack", None], default=None)
    parser.add_argument("--output_file", type=str, default="nanogpt.pte")
    parser.add_argument("--verifiy_runtime", action="store_true", default=False)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)

    args = parser.parse_args()
    main(args)
