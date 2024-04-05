#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Optional

import torch
from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
    XnnpackDynamicallyQuantizedPartitioner,
)
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchProgramManager,
    to_edge,
)
from executorch.exir.backend.utils import (
    get_delegation_info,
    print_delegated_graph,
)
from tabulate import tabulate

from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from model import GPT, GPTConfig
from test_utils import check_executorch_output_consistency, ErrorLimits
from torch import nn
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
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
class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = GPT.from_pretrained("gpt2")  # use gpt2 weight as pretrained weight

    def forward(self, idx):
        for _ in range(GENERATE_SEQ_LENGTH):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.model.config.block_size
                else idx[:, -self.model.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self.model(idx_cond)
            # choose the highest probability token as the next index to continue the sequence with
            idx_next = torch.argmax(logits).view(1, 1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def main(args):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Prep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model = GPT.from_pretrained("gpt2")
    example_inputs = (torch.randint(0, 100, (1, 8), dtype=torch.long),)

    #  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Export  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Using a custom SDPA kernel for LLMs
    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        m = capture_pre_autograd_graph(model, example_inputs)

        if args.quantize:
            # Use dynamic, per-channel quantization.
            xnnpack_quant_config = get_symmetric_quantization_config(
                is_per_channel=True, is_dynamic=True
            )
            xnnpack_quantizer = XNNPACKQuantizer()
            xnnpack_quantizer.set_global(xnnpack_quant_config)

            m = prepare_pt2e(m, xnnpack_quantizer)
            m(*example_inputs)
            m = convert_pt2e(m, fold_quantize=False)
            DuplicateDynamicQuantChainPass()(m)

        if args.backend == "xnnpack":
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
        if args.backend == "xnnpack":
            print("Lowering to XnnPack...")
            edge_manager = edge_manager.to_backend(XnnpackPartitioner())
        elif args.quantize: # Note the using XnnpackPartitioner for everything should also work for quant.
            print("Lowering to XNNPACK (quantized)...")
            edge_manager = edge_manager.to_backend(XnnpackDynamicallyQuantizedPartitioner())


        graph_module = edge_manager.exported_program().graph_module
        delegation_info = get_delegation_info(graph_module)
        print(delegation_info.get_summary())
        df = delegation_info.get_operator_delegation_dataframe()
        print(tabulate(df, headers="keys", tablefmt="fancy_grid"))

        graph_module = edge_manager.exported_program().graph_module
        print(print_delegated_graph(graph_module))

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
    parser.add_argument("--backend", type=str.lower, choices=["xnnpack", None], default=None)
    parser.add_argument("--output_file", type=str, default="nanogpt.pte")
    parser.add_argument("--verifiy_runtime", action="store_true", default=False)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--quantize", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
