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

# This variable sets the number of tokens to generate.
# It aims to balance between generating a sentence with sufficient information and maintaining reasonable computation time.
# In this case, we've set it to generate 20 tokens.
GENERATE_SEQ_LENGTH = 20


# This is a wrapper class for the NanoGPT model, designed for demonstration purposes.
# It includes a custom forward function that generates a sentence of a specified length
# based on a given tokenized prompt with a single forward pass.
# Please note that this wrapper is quite resource-intensive due to the inclusion of a for loop for sentence generation.
# For a more efficient sequence generation, please refer to the implementation in the llama runner.
# class NanoGPT(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = GPT.from_pretrained("gpt2")  # use gpt2 weight as pretrained weight

#     def forward(self, idx):
#         for _ in range(GENERATE_SEQ_LENGTH):
#             # if the sequence context is growing too long we must crop it at block_size
#             idx_cond = (
#                 idx
#                 if idx.size(1) <= self.model.config.block_size
#                 else idx[:, -self.model.config.block_size :]
#             )
#             # forward the model to get the logits for the index in the sequence
#             logits, _ = self.model(idx_cond)
#             # choose the highest probability token as the next index to continue the sequence with
#             idx_next = torch.argmax(logits).view(1, 1)
#             # append sampled index to the running sequence and continue
#             idx = torch.cat((idx, idx_next), dim=1)

#         return idx


def main(args):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Prep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # model = NanoGPT()
    model = GPT.from_pretrained("gpt2")  # use gpt2 weight as pretrained weight
    example_inputs = (
        torch.randint(0, 100, (1, model.config.block_size - 1), dtype=torch.long),
    )
    dynamic_shape = (
        {1: torch.export.Dim("token_dim", max=model.config.block_size - 1)},
    )

    #  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Export  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Using a custom SDPA kernel for LLMs
    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        m = capture_pre_autograd_graph(
            model, example_inputs, dynamic_shapes=dynamic_shape
        )

        if args.backend == "XnnPack":
            edge_config = get_xnnpack_edge_compile_config()
        else:
            edge_config = EdgeCompileConfig(_check_ir_validity=False)

        print("Exporting program...")
        core_aten_ep = export(m, example_inputs, dynamic_shapes=dynamic_shape)
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

        example_inputs = (torch.randint(0, 100, (1, 512), dtype=torch.long),)
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
