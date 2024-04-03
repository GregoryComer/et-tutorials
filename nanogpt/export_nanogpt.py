#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from executorch.exir import to_edge
import torch
from torch.export import export
from torch import nn

from torch.nn.attention import SDPBackend
from model import GPT, GPTConfig
from torch._export import capture_pre_autograd_graph
from executorch.exir import EdgeCompileConfig, EdgeProgramManager, ExecutorchProgramManager
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from test_utils import check_executorch_output_consistency, ErrorLimits
from typing import Optional

<<<<<<< HEAD
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)


class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = GPT.from_pretrained('gpt2') # use gpt2 weight as pretrained weight

    def forward(self, input_ids: torch.Tensor):
        return self.model.generate(input_ids, 20)
=======
# from executorch.extension.pybindings.portable_lib import (
#     _load_for_executorch_from_buffer,
# )


class NanoGPT(GPT):
    def __init__(self, config: GPTConfig):
        super().__init__(config)

    def forward(self, input_ids: torch.Tensor):
        return self.generate(input_ids, 20)
>>>>>>> 48bdf5fe81082634a5235413f3e19c89c9f3bd48


def main(args):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Prep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
<<<<<<< HEAD
    model = NanoGPT()
    example_inputs = (torch.randint(0, 100, (1, 3), dtype=torch.long), )

    example_outputs = model(*example_inputs)
    print("~~~~~~~~~~~~~~~~~~~~~~")
    print(type(example_outputs))
    print(len(example_outputs))
    print("~~~~~~~~~~~~~~~~~~~~~~")

=======
    model = NanoGPT.from_pretrained('gpt2') # use gpt2 weight as pretrained weight
    example_inputs = (torch.randint(0, 100, (1, 1024), dtype=torch.long), )

>>>>>>> 48bdf5fe81082634a5235413f3e19c89c9f3bd48
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

<<<<<<< HEAD
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
=======
        # print("Checking the outputs of the ExecuTorch program...")
        # error_limits = ErrorLimits(atol=args.atol, rtol=args.rtol)
        # res = check_executorch_output_consistency(
        #     flatbuffer_buff=et_program.buffer,
        #     model=model,
        #     method_name=method_name,
        #     example_inputs=example_inputs,
        #     load_fn=_load_for_executorch_from_buffer,
        #     error_limits=error_limits,
        # )
        # if res.is_same:
        #     print("Outputs are the same!")
        # else:
        #     print("Outputs are different!")
        #     print("Reasons: {}".format("\n ".join(res.reasons)))
>>>>>>> 48bdf5fe81082634a5235413f3e19c89c9f3bd48

    # Write the serialized ExecuTorch program to a file.
    with open(args.output_file, "wb") as file:
        file.write(et_program.buffer)
        print(f"ExecuTorch program saved to {args.output_file}.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='NanoGPT example.')
    parser.add_argument('--backend', type=str, choices=["XnnPack", None], default=None)
    parser.add_argument('--output_file', type=str, default="nanogpt.pte")
    parser.add_argument('--verifiy_runtime', action='store_true', default=False)
    parser.add_argument('--atol', type=float, default=None)
    parser.add_argument('--rtol', type=float, default=None)

    args = parser.parse_args()
    main(args)
